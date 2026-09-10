use crate::error::{AppError, Result};
use std::path::{Path, PathBuf};

/// Everything the app installs at runtime lives under one directory so the
/// user can reclaim the whole footprint by deleting a single folder.
#[derive(Clone, Debug)]
pub struct AppPaths {
    pub root: PathBuf,
    /// Where downloaded weights live. Separable from `root` so multi-gigabyte
    /// models can sit on an external drive while keys, the vault and the
    /// runtime stay on the internal one.
    models_root: PathBuf,
}

/// Persisted alongside the vault, not inside it: the app has to know where the
/// models are before anything is unlocked.
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct StoredConfig {
    models_root: Option<String>,
}

impl AppPaths {
    pub fn resolve() -> Result<Self> {
        // An override makes the runtime bootstrap testable against a scratch
        // directory instead of the user's real installation.
        let root = if let Ok(custom) = std::env::var("MODELSTUDIO_HOME") {
            if custom.trim().is_empty() {
                default_root()?
            } else {
                PathBuf::from(custom)
            }
        } else {
            default_root()?
        };
        Ok(Self::at(root))
    }

    pub fn at(root: PathBuf) -> Self {
        let models_root = read_config(&root)
            .models_root
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| root.join("models"));
        Self { root, models_root }
    }

    /// Move model storage somewhere else, e.g. an external drive.
    pub fn set_models_root(&mut self, new_root: PathBuf) -> Result<()> {
        std::fs::create_dir_all(new_root.join("hub"))?;
        let cfg = StoredConfig {
            models_root: Some(new_root.to_string_lossy().to_string()),
        };
        let tmp = self.config_path().with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&cfg)?)?;
        std::fs::rename(&tmp, self.config_path())?;
        self.models_root = new_root;
        Ok(())
    }

    pub fn models_root(&self) -> &PathBuf {
        &self.models_root
    }
    fn config_path(&self) -> PathBuf {
        self.root.join("config.json")
    }

    pub fn runtime(&self) -> PathBuf {
        self.root.join("runtime")
    }
    pub fn python_dir(&self) -> PathBuf {
        self.runtime().join("python")
    }
    pub fn python_bin(&self) -> PathBuf {
        self.python_dir().join("bin/python3")
    }
    pub fn venv(&self) -> PathBuf {
        self.runtime().join("venv")
    }
    pub fn venv_python(&self) -> PathBuf {
        self.venv().join("bin/python3")
    }
    pub fn stamp(&self) -> PathBuf {
        self.runtime().join("install.json")
    }

    /// Hugging Face cache. Kept inside our root so the disk meter tells the
    /// truth and "delete model" actually reclaims space.
    pub fn hf_home(&self) -> PathBuf {
        self.models_root.clone()
    }
    pub fn hf_hub(&self) -> PathBuf {
        self.hf_home().join("hub")
    }

    /// The encrypted vault. There is deliberately no plaintext output
    /// directory: generated and imported content only ever lands in here.
    pub fn vault(&self) -> PathBuf {
        self.root.join("vault")
    }
    /// Models the user added by Hugging Face repo id.
    pub fn custom_models(&self) -> PathBuf {
        self.root.join("custom_models.json")
    }
    /// Measured generation speed on this machine.
    pub fn timings(&self) -> PathBuf {
        self.root.join("timings.json")
    }
    pub fn logs(&self) -> PathBuf {
        self.root.join("logs")
    }

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

        if let Ok(p) = app
            .path()
            .resolve("engine/worker.py", tauri::path::BaseDirectory::Resource)
        {
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

fn default_root() -> Result<PathBuf> {
    let home = std::env::var("HOME").map_err(|_| AppError::msg("HOME is not set"))?;
    Ok(PathBuf::from(home).join("Library/Application Support/com.melp.modelstudio"))
}

fn read_config(root: &Path) -> StoredConfig {
    std::fs::read_to_string(root.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}
