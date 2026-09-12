//! First-run runtime bootstrap.
//!
//! The models are Python/MLX only, so the app provisions a private, standalone
//! CPython and installs `mlx-gen` into a venv underneath it. Nothing touches
//! the user's system Python, and uninstalling means deleting one directory.

use crate::error::{AppError, Result};
use crate::paths::AppPaths;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Stdio;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

/// A reviewed, immutable runtime artifact. Do not replace this with a
/// "latest" lookup: setup downloads executable code and must be reproducible.
const PYTHON_URL: &str = "https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.12.14%2B20260901-aarch64-apple-darwin-install_only.tar.gz";
const PYTHON_SHA256: &str = "3ee3ee547cedfeb7c2b16b2b7156039f7b470bb8f857e226fd3d2eb11db83c76";
const PY_SERIES: &str = "3.12";
const MLX_GEN_VERSION: &str = "0.36.0";
const MLX_LM_VERSION: &str = "0.31.3";
const MLX_VLM_VERSION: &str = "0.7.0";
const ENGINE_REQUIREMENTS: &str = include_str!("../../engine/requirements.lock");

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct InstallStamp {
    pub python_version: String,
    pub mlxgen_version: String,
    pub installed_at: String,
}

#[derive(Serialize, Clone, Debug)]
pub struct SetupState {
    pub ready: bool,
    pub python_present: bool,
    pub venv_present: bool,
    pub engine_present: bool,
    pub stamp: Option<InstallStamp>,
    pub root: String,
}

#[derive(Serialize, Clone, Debug)]
pub struct SetupProgress {
    pub step: String,
    pub detail: String,
    /// 0.0..1.0 over the whole setup, or null when indeterminate.
    pub progress: Option<f32>,
    pub done: bool,
    pub error: Option<String>,
}

/// Where progress goes. The app forwards it to the window; tests print it.
pub type Reporter<'a> = &'a (dyn Fn(&str, &str, Option<f32>) + Send + Sync);

fn tauri_reporter(app: &AppHandle) -> impl Fn(&str, &str, Option<f32>) + Send + Sync + '_ {
    move |step: &str, detail: &str, progress: Option<f32>| {
        let _ = app.emit(
            "setup://progress",
            SetupProgress {
                step: step.into(),
                detail: detail.into(),
                progress,
                done: false,
                error: None,
            },
        );
    }
}

pub fn state(paths: &AppPaths) -> SetupState {
    let python_present = paths.python_bin().exists();
    let venv_present = paths.venv_python().exists();
    let stamp: Option<InstallStamp> = std::fs::read_to_string(paths.stamp())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    let engine_present = stamp.as_ref().is_some_and(|s| {
        s.python_version.starts_with(PY_SERIES) && s.mlxgen_version == MLX_GEN_VERSION
    }) && required_distributions_present(paths);
    SetupState {
        ready: python_present && venv_present && engine_present,
        python_present,
        venv_present,
        engine_present,
        stamp,
        root: paths.root.to_string_lossy().to_string(),
    }
}

fn required_distributions_present(paths: &AppPaths) -> bool {
    let site = paths
        .venv()
        .join("lib")
        .join(format!("python{PY_SERIES}"))
        .join("site-packages");
    let Ok(entries) = std::fs::read_dir(site) else {
        return false;
    };
    let names: Vec<String> = entries
        .flatten()
        .map(|e| {
            e.file_name()
                .to_string_lossy()
                .to_lowercase()
                .replace('-', "_")
        })
        .collect();
    [
        format!("mlx_gen-{MLX_GEN_VERSION}.dist_info"),
        format!("mlx_lm-{MLX_LM_VERSION}.dist_info"),
        format!("mlx_vlm-{MLX_VLM_VERSION}.dist_info"),
        "cryptography-".into(),
        "pillow_heif-".into(),
        "huggingface_hub-".into(),
    ]
    .iter()
    .all(|want| names.iter().any(|name| name.starts_with(want)))
}

async fn download_to(
    report: Reporter<'_>,
    url: &str,
    expected_sha256: &str,
    dest: &Path,
    span: (f32, f32),
) -> Result<()> {
    let client = reqwest::Client::builder()
        .user_agent("melp-model-studio")
        .build()?;
    let resp = client.get(url).send().await?.error_for_status()?;
    let total = resp.content_length().unwrap_or(0);

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = tokio::fs::File::create(dest).await?;
    let mut stream = resp.bytes_stream();
    let mut done: u64 = 0;
    let mut last_emit = std::time::Instant::now();
    let mut digest = Sha256::new();

    use tokio::io::AsyncWriteExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        done += chunk.len() as u64;
        digest.update(&chunk);
        file.write_all(&chunk).await?;
        if last_emit.elapsed().as_millis() > 120 {
            last_emit = std::time::Instant::now();
            let frac = if total > 0 {
                done as f32 / total as f32
            } else {
                0.0
            };
            report(
                "python",
                &format!(
                    "Downloading Python runtime — {:.0} MB",
                    done as f32 / 1_048_576.0
                ),
                Some(span.0 + (span.1 - span.0) * frac),
            );
        }
    }
    file.flush().await?;
    let actual = format!("{:x}", digest.finalize());
    if actual != expected_sha256 {
        let _ = tokio::fs::remove_file(dest).await;
        return Err(AppError::msg(format!(
            "Python runtime checksum mismatch: expected {expected_sha256}, got {actual}"
        )));
    }
    Ok(())
}

fn extract_targz(archive: &Path, into: &Path) -> Result<()> {
    let f = std::fs::File::open(archive)?;
    let dec = flate2::read::GzDecoder::new(f);
    let mut tar = tar::Archive::new(dec);
    std::fs::create_dir_all(into)?;
    tar.unpack(into)?;
    Ok(())
}

/// Run a command, streaming its output into the setup progress channel so the
/// user sees pip working rather than a frozen bar.
async fn run_streaming(
    report: Reporter<'_>,
    step: &str,
    mut cmd: Command,
    span: (f32, f32),
) -> Result<()> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn()?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    // Both pipes must be drained concurrently. pip writes progress to stderr;
    // waiting for stdout to close first can fill the stderr pipe and deadlock.
    let stdout_future = async {
        if let Some(out) = stdout {
            let mut lines = BufReader::new(out).lines();
            let mut n = 0u32;
            while let Ok(Some(line)) = lines.next_line().await {
                n += 1;
                // pip is verbose; a slow crawl beats a fake percentage.
                let frac = span.0 + (span.1 - span.0) * (1.0 - (-(n as f32) / 220.0).exp());
                report(step, &line, Some(frac));
            }
        }
    };

    let stderr_future = async {
        let mut err_text = String::new();
        if let Some(err) = stderr {
            let mut lines = BufReader::new(err).lines();
            let mut n = 0u32;
            while let Ok(Some(line)) = lines.next_line().await {
                n += 1;
                err_text.push_str(&line);
                err_text.push('\n');
                if err_text.len() > 8000 {
                    err_text.drain(..4000);
                }
                let frac = span.0 + (span.1 - span.0) * (1.0 - (-(n as f32) / 220.0).exp());
                report(step, &line, Some(frac));
            }
        }
        err_text
    };

    let (_, err_text) = tokio::join!(stdout_future, stderr_future);

    let status = child.wait().await?;
    if !status.success() {
        return Err(AppError::msg(format!(
            "{step} failed (exit {}):\n{}",
            status.code().unwrap_or(-1),
            err_text.trim()
        )));
    }
    Ok(())
}

pub async fn run(app: AppHandle, paths: AppPaths, force: bool) -> Result<()> {
    paths.ensure_dirs()?;

    let reporter = tauri_reporter(&app);
    let result = bootstrap(&reporter, &paths, force).await;
    match &result {
        Ok(()) => {
            let _ = app.emit(
                "setup://progress",
                SetupProgress {
                    step: "done".into(),
                    detail: "Runtime ready".into(),
                    progress: Some(1.0),
                    done: true,
                    error: None,
                },
            );
        }
        Err(e) => {
            let _ = app.emit(
                "setup://progress",
                SetupProgress {
                    step: "error".into(),
                    detail: String::new(),
                    progress: None,
                    done: true,
                    error: Some(e.to_string()),
                },
            );
        }
    }
    result
}

/// Provision the private Python runtime and the engine.
///
/// Free of any Tauri dependency so it can be exercised headlessly against a
/// scratch directory -- otherwise the only way to test the first-run path
/// would be to destroy a working installation.
/// Just the Python runtime and virtual environment -- everything up to, but
/// not including, the multi-gigabyte engine install.
pub async fn bootstrap_runtime_only(
    report: Reporter<'_>,
    paths: &AppPaths,
    force: bool,
) -> Result<()> {
    paths.ensure_dirs()?;
    provision_python(report, paths, force).await?;
    provision_venv(report, paths, force).await
}

async fn provision_python(report: Reporter<'_>, paths: &AppPaths, force: bool) -> Result<()> {
    if std::env::consts::ARCH != "aarch64" {
        return Err(AppError::msg(
            "Model Studio requires Apple Silicon: MLX has no Intel Mac backend.",
        ));
    }
    if !force && paths.python_bin().exists() {
        return Ok(());
    }

    report("python", "Preparing verified Python runtime", Some(0.01));
    let archive = paths.runtime().join("python.tar.gz");
    download_to(report, PYTHON_URL, PYTHON_SHA256, &archive, (0.02, 0.20)).await?;

    report("python", "Extracting Python runtime", Some(0.22));
    let staging = paths.runtime().join("python-staging");
    let _ = std::fs::remove_dir_all(&staging);
    extract_targz(&archive, &staging)?;

    // install_only archives unpack to a single `python/` directory.
    let unpacked = staging.join("python");
    let src = if unpacked.exists() {
        unpacked
    } else {
        staging.clone()
    };
    let _ = std::fs::remove_dir_all(paths.python_dir());
    std::fs::rename(&src, paths.python_dir())?;
    let _ = std::fs::remove_dir_all(&staging);
    let _ = std::fs::remove_file(&archive);

    if !paths.python_bin().exists() {
        return Err(AppError::msg("Python runtime missing after extraction"));
    }
    Ok(())
}

async fn provision_venv(report: Reporter<'_>, paths: &AppPaths, force: bool) -> Result<()> {
    if !force && paths.venv_python().exists() {
        return Ok(());
    }
    report("venv", "Creating virtual environment", Some(0.25));
    let _ = std::fs::remove_dir_all(paths.venv());
    let mut cmd = Command::new(paths.python_bin());
    cmd.arg("-m").arg("venv").arg(paths.venv());
    run_streaming(report, "venv", cmd, (0.25, 0.28)).await
}

pub async fn bootstrap(report: Reporter<'_>, paths: &AppPaths, force: bool) -> Result<()> {
    if std::env::consts::ARCH != "aarch64" {
        return Err(AppError::msg(
            "Model Studio requires Apple Silicon: MLX has no Intel Mac backend.",
        ));
    }

    // ---- 1. Standalone CPython ------------------------------------------
    provision_python(report, paths, force).await?;

    // ---- 2. Virtual environment -----------------------------------------
    provision_venv(report, paths, force).await?;

    // ---- 3. Engine ------------------------------------------------------
    report("engine", "Installing pinned packaging tools", Some(0.30));
    let mut up = Command::new(paths.venv_python());
    up.args([
        "-m",
        "pip",
        "install",
        "pip==26.2.1",
        "wheel==0.48.0",
        "--no-input",
    ]);
    run_streaming(report, "engine", up, (0.30, 0.34)).await?;

    report(
        "engine",
        "Installing mlx-gen and PyTorch — this is the long part (~3 GB)",
        Some(0.35),
    );
    let requirements = paths.runtime().join("requirements.lock");
    std::fs::write(&requirements, ENGINE_REQUIREMENTS)?;
    let mut pip = Command::new(paths.venv_python());
    pip.args([
        "-m",
        "pip",
        "install",
        "--no-input",
        "--only-binary=:all:",
        "--requirement",
    ]);
    pip.arg(&requirements);
    // Keep pip's own cache inside our root so the disk meter stays honest.
    pip.env("PIP_CACHE_DIR", paths.root.join("pipcache"));
    run_streaming(report, "engine", pip, (0.35, 0.95)).await?;

    // ---- 4. Verify and stamp --------------------------------------------
    report("verify", "Verifying installation", Some(0.96));
    let probe = Command::new(paths.venv_python())
        .args([
            "-c",
            "import json,sys,mlx.core as mx,mlxgen,cryptography,mlx_lm;from importlib.metadata import version;print(json.dumps({'py':sys.version.split()[0],'mlxgen':version('mlx-gen'),'mlx':getattr(mx,'__version__','unknown'),'mlx_lm':version('mlx-lm')}))",
        ])
        .output()
        .await?;
    if !probe.status.success() {
        return Err(AppError::msg(format!(
            "verification failed:\n{}",
            String::from_utf8_lossy(&probe.stderr)
        )));
    }
    let v: serde_json::Value = serde_json::from_slice(&probe.stdout)
        .map_err(|e| AppError::msg(format!("could not parse verification output: {e}")))?;

    let stamp = InstallStamp {
        python_version: v["py"].as_str().unwrap_or("?").into(),
        mlxgen_version: v["mlxgen"].as_str().unwrap_or("?").into(),
        installed_at: chrono::Utc::now().to_rfc3339(),
    };
    std::fs::write(paths.stamp(), serde_json::to_vec_pretty(&stamp)?)?;
    Ok(())
}
