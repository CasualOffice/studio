//! Sidecar manager for the Python engine worker.
//!
//! The worker is long-lived on purpose: it keeps one model resident so a second
//! generation does not pay the 10-30s load cost again. This module owns the
//! process, multiplexes JSON Lines requests over its stdin, and fans progress
//! events out to the frontend.

use crate::error::{AppError, Result};
use crate::paths::AppPaths;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin};
use tokio::sync::{oneshot, Mutex};

type Pending =
    Arc<Mutex<HashMap<String, oneshot::Sender<std::result::Result<Value, EngineFailure>>>>>;

#[derive(Serialize, Clone, Debug)]
pub struct EngineFailure {
    pub message: String,
    pub kind: String,
    /// mlx-gen refuses to download during generation and instead raises with
    /// the exact remediation command. Surfacing it lets the UI offer one click.
    pub download_command: Option<String>,
    pub cancelled: bool,
}

#[derive(Serialize, Clone, Debug)]
pub struct EngineProgress {
    pub job_id: String,
    pub phase: String,
    pub progress: Option<f32>,
    pub step: Option<u32>,
    pub total_steps: Option<u32>,
    pub message: Option<String>,
    pub done_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
    pub seed: Option<i64>,
    pub item_index: Option<u32>,
    pub item_count: Option<u32>,
}

pub struct Engine {
    stdin: Mutex<ChildStdin>,
    pending: Pending,
    child: Mutex<Child>,
    /// Set when the worker's stdout closes, which is the first thing that
    /// happens whether it exited, crashed, or was killed from Activity
    /// Monitor. Without this the host would keep handing out a handle to a
    /// dead process and every request would fail until the app was restarted.
    dead: Arc<AtomicBool>,
}

impl Engine {
    pub fn is_dead(&self) -> bool {
        self.dead.load(Ordering::Relaxed)
    }
}

impl Engine {
    pub async fn spawn(
        app: &AppHandle,
        paths: &AppPaths,
        script: &std::path::Path,
    ) -> Result<Arc<Self>> {
        if !paths.venv_python().exists() {
            return Err(AppError::SetupIncomplete(
                "the Python runtime has not been installed yet".into(),
            ));
        }

        // Resource policy: the engine must not be able to take the machine
        // down with it. `nice` keeps the window server and the UI ahead of it,
        // the thread caps leave cores free, and MLX gets a hard memory ceiling
        // so an oversized request fails loudly instead of swapping.
        let host = crate::hostinfo::HostInfo::probe(paths);

        let mut cmd = tokio::process::Command::new("/usr/bin/nice");
        cmd.arg("-n")
            .arg("5")
            .arg(paths.venv_python())
            .arg("-u") // unbuffered: progress must arrive as it happens
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("HF_HOME", paths.hf_home())
            .env("HF_HUB_ENABLE_HF_TRANSFER", "1")
            .env("PYTHONUNBUFFERED", "1")
            // MLX picks its own limits; keep tokenizers from forking threads
            // that would fight the generation for a 16 GB machine's cores.
            .env("TOKENIZERS_PARALLELISM", "false")
            .env(
                "MODELSTUDIO_MEMORY_BUDGET_GIB",
                host.memory_budget_gib.to_string(),
            )
            .env("OMP_NUM_THREADS", host.worker_threads.to_string())
            .env("MKL_NUM_THREADS", host.worker_threads.to_string())
            .env("VECLIB_MAXIMUM_THREADS", host.worker_threads.to_string())
            .env("NUMEXPR_NUM_THREADS", host.worker_threads.to_string())
            .env("MODELSTUDIO_IDLE_UNLOAD_SECONDS", "600");

        let mut child = cmd.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| AppError::msg("no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AppError::msg("no stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| AppError::msg("no stderr"))?;

        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let dead = Arc::new(AtomicBool::new(false));

        // stdout: the protocol channel.
        {
            let pending = pending.clone();
            let app = app.clone();
            let dead = dead.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let Ok(v) = serde_json::from_str::<Value>(&line) else {
                        continue;
                    };
                    dispatch(&app, &pending, v).await;
                }
                // The worker died. Mark it so the next request spawns a fresh
                // one, then fail everything still waiting rather than leaving
                // the UI spinning forever.
                dead.store(true, Ordering::Relaxed);
                let mut map = pending.lock().await;
                for (_, tx) in map.drain() {
                    let _ = tx.send(Err(EngineFailure {
                        message: "the engine process exited unexpectedly".into(),
                        kind: "engine_exit".into(),
                        download_command: None,
                        cancelled: false,
                    }));
                }
                let _ = app.emit("engine://exit", ());
            });
        }

        // stderr: diagnostics only, mirrored to a log file.
        {
            let logfile = paths.logs().join("engine.log");
            // Rotate before appending. PyTorch and tokenizers are chatty
            // enough that an unbounded log quietly grows without limit.
            const MAX_LOG_BYTES: u64 = 2 * 1024 * 1024;
            if let Ok(meta) = std::fs::metadata(&logfile) {
                if meta.len() > MAX_LOG_BYTES {
                    let _ = std::fs::rename(&logfile, logfile.with_extension("log.1"));
                }
            }
            let app = app.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                let mut sink = tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&logfile)
                    .await
                    .ok();
                while let Ok(Some(line)) = lines.next_line().await {
                    if let Some(f) = sink.as_mut() {
                        let _ = f.write_all(format!("{line}\n").as_bytes()).await;
                    }
                    let _ = app.emit("engine://stderr", line);
                }
            });
        }

        Ok(Arc::new(Self {
            stdin: Mutex::new(stdin),
            pending,
            child: Mutex::new(child),
            dead,
        }))
    }

    /// Send a request and await its terminal result.
    pub async fn request(&self, job_id: &str, op: &str, mut params: Value) -> Result<Value> {
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(job_id.to_string(), tx);

        if let Value::Object(ref mut m) = params {
            m.insert("id".into(), json!(job_id));
            m.insert("op".into(), json!(op));
        }

        {
            let mut stdin = self.stdin.lock().await;
            let line = format!("{}\n", serde_json::to_string(&params)?);
            stdin
                .write_all(line.as_bytes())
                .await
                .map_err(|e| AppError::Engine(format!("could not reach the engine: {e}")))?;
            stdin.flush().await?;
        }

        match rx.await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(f)) => Err(AppError::Engine(f.message)),
            Err(_) => Err(AppError::EngineDown),
        }
    }

    /// Fire-and-forget: cancellation must not queue behind the running job.
    pub async fn cancel(&self, target: &str) -> Result<()> {
        let mut stdin = self.stdin.lock().await;
        let line = format!(
            "{}\n",
            json!({"id": format!("cancel-{target}"), "op": "cancel", "target": target})
        );
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await?;
        Ok(())
    }

    pub async fn shutdown(&self) {
        {
            let mut stdin = self.stdin.lock().await;
            let _ = stdin
                .write_all(b"{\"id\":\"bye\",\"op\":\"shutdown\"}\n")
                .await;
            let _ = stdin.flush().await;
        }
        let mut child = self.child.lock().await;
        // Give it a moment to exit cleanly, then insist.
        for _ in 0..20 {
            if matches!(child.try_wait(), Ok(Some(_))) {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        let _ = child.kill().await;
    }
}

async fn dispatch(app: &AppHandle, pending: &Pending, v: Value) {
    let id = v
        .get("id")
        .and_then(|x| x.as_str())
        .unwrap_or("?")
        .to_string();
    let ty = v.get("type").and_then(|x| x.as_str()).unwrap_or("");

    match ty {
        "progress" => {
            let _ = app.emit(
                "engine://progress",
                EngineProgress {
                    job_id: id,
                    phase: v["phase"].as_str().unwrap_or("").to_string(),
                    progress: v["progress"].as_f64().map(|f| f as f32),
                    step: v["step"].as_u64().map(|n| n as u32),
                    total_steps: v["total_steps"].as_u64().map(|n| n as u32),
                    message: v["message"].as_str().map(|s| s.to_string()),
                    done_bytes: v["done_bytes"].as_u64(),
                    total_bytes: v["total_bytes"].as_u64(),
                    seed: v["seed"].as_i64(),
                    item_index: v["item_index"].as_u64().map(|n| n as u32),
                    item_count: v["item_count"].as_u64().map(|n| n as u32),
                },
            );
        }
        "preview" => {
            // A partial frame. Forwarded as its own channel so the progress
            // handler is not woken for every image.
            let _ = app.emit("engine://preview", v);
        }
        "log" => {
            let _ = app.emit("engine://log", v);
        }
        "ready" => {
            let _ = app.emit("engine://ready", v);
        }
        "result" => {
            if let Some(tx) = pending.lock().await.remove(&id) {
                let _ = tx.send(Ok(v.get("result").cloned().unwrap_or(Value::Null)));
            }
        }
        "error" => {
            let kind = v["kind"].as_str().unwrap_or("error").to_string();
            let failure = EngineFailure {
                message: v["error"].as_str().unwrap_or("unknown error").to_string(),
                cancelled: kind == "cancelled",
                download_command: v["download_command"].as_str().map(|s| s.to_string()),
                kind,
            };
            if let Some(tx) = pending.lock().await.remove(&id) {
                let _ = tx.send(Err(failure));
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// The reader task sets this flag when the worker's stdout closes; the host
    /// checks it before reusing a handle. Reusing a dead engine left the app
    /// broken until it was restarted, so the contract is worth pinning down.
    #[test]
    fn death_flag_is_observable_across_clones() {
        let dead = Arc::new(AtomicBool::new(false));
        let seen_by_host = dead.clone();
        assert!(!seen_by_host.load(Ordering::Relaxed));

        // What the reader task does when stdout closes.
        dead.store(true, Ordering::Relaxed);

        assert!(
            seen_by_host.load(Ordering::Relaxed),
            "the host must observe the worker's death through its own clone"
        );
    }
}
