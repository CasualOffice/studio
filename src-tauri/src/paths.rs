use crate::error::{AppError, Result};
use std::path::PathBuf;

/// Everything the app installs at runtime lives under one directory so the
/// user can reclaim the whole footprint by deleting a single folder.
#[derive(Clone, Debug)]
pub struct AppPaths {
    pub root: PathBuf,
}

impl AppPaths {
    pub fn resolve() -> Result<Self> {
        // An override makes the runtime bootstrap testable against a scratch
        // directory instead of the user's real installation.
        if let Ok(custom) = std::env::var("MODELSTUDIO_HOME") {
            if !custom.trim().is_empty() {
                return Ok(Self { root: PathBuf::from(custom) });
            }
        }
        let home = std::env::var("HOME")
            .map_err(|_| AppError::msg("HOME is not set"))?;
        let root = PathBuf::from(home)
            .join("Library/Application Support/com.melp.modelstudio");
        Ok(Self { root })
    }

    pub fn at(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn runtime(&self) -> PathBuf { self.root.join("runtime") }
    pub fn python_dir(&self) -> PathBuf { self.runtime().join("python") }
    pub fn python_bin(&self) -> PathBuf { self.python_dir().join("bin/python3") }
    pub fn venv(&self) -> PathBuf { self.runtime().join("venv") }
    pub fn venv_python(&self) -> PathBuf { self.venv().join("bin/python3") }
    pub fn stamp(&self) -> PathBuf { self.runtime().join("install.json") }

    /// Hugging Face cache. Kept inside our root so the disk meter tells the
    /// truth and "delete model" actually reclaims space.
    pub fn hf_home(&self) -> PathBuf { self.root.join("models") }
    pub fn hf_hub(&self) -> PathBuf { self.hf_home().join("hub") }

    /// The encrypted vault. There is deliberately no plaintext output
    /// directory: generated and imported content only ever lands in here.
    pub fn vault(&self) -> PathBuf { self.root.join("vault") }
    /// Models the user added by Hugging Face repo id.
    pub fn custom_models(&self) -> PathBuf { self.root.join("custom_models.json") }
    /// Measured generation speed on this machine.
    pub fn timings(&self) -> PathBuf { self.root.join("timings.json") }
    pub fn logs(&self) -> PathBuf { self.root.join("logs") }

    pub fn ensure_dirs(&self) -> Result<()> {
        for d in [self.runtime(), self.hf_hub(), self.vault(), self.logs()] {
            std::fs::create_dir_all(&d)?;
        }
        Ok(())
    }

    /// Locate `engine/worker.py`.
    ///
    /// Packaged builds carry it as a Tauri resource. Development builds are
    /// messier: the working directory depends on how the binary was launched,
    /// so a single `../engine` guess is unreliable — it silently resolved
    /// outside the repository when run from the project root. Walk up from
    /// both the executable and the working directory instead, and report every
    /// place that was tried when nothing is found.
    pub fn worker_script(&self, app: &tauri::AppHandle) -> Result<PathBuf> {
        use tauri::Manager;

        let mut tried: Vec<PathBuf> = Vec::new();

        if let Ok(p) = app.path().resolve("engine/worker.py", tauri::path::BaseDirectory::Resource) {
            if p.exists() {
                return Ok(p);
            }
            tried.push(p);
        }

        let mut roots: Vec<PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            roots.extend(exe.ancestors().take(6).map(PathBuf::from));
        }
        if let Ok(cwd) = std::env::current_dir() {
            roots.extend(cwd.ancestors().take(4).map(PathBuf::from));
        }

        for root in roots {
            let candidate = root.join("engine/worker.py");
            if candidate.exists() {
                return Ok(candidate.canonicalize().unwrap_or(candidate));
            }
            tried.push(candidate);
        }

        Err(AppError::msg(format!(
            "could not locate engine/worker.py. Looked in:\n  {}",
            tried
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join("\n  ")
        )))
    }
}
