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
    /// The configured location, when it could not be used for this session and
    /// `models_root` above is the internal fallback. Kept so the UI can say
    /// which drive is missing instead of silently pointing at somewhere else.
    unavailable_models_root: Option<PathBuf>,
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

    /// Read the stored configuration, falling back to the internal model
    /// directory when the configured one cannot be used.
    ///
    /// A models_root on an external drive is unreachable whenever that drive is
    /// not attached, and the path is then under /Volumes, which belongs to
    /// root: creating it fails with EACCES. Boot used to carry that failure out
    /// of `ensure_dirs` and panic before `tauri::Builder` ran, so the app died
    /// with no window and no message, and the Storage panel that could have
    /// pointed the path somewhere else never rendered -- the only way back was
    /// to reattach that exact drive or hand-edit config.json. Use the app's own
    /// `models` directory for the session instead. config.json is deliberately
    /// left untouched: the drive may come back, and rewriting the user's choice
    /// behind their back would lose the models already on it.
    pub fn at(root: PathBuf) -> Self {
        let internal = root.join("models");
        let configured = read_config(&root)
            .models_root
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty());
        let (models_root, unavailable_models_root) = match configured {
            Some(p) if !is_usable_models_root(&p) => (internal, Some(p)),
            Some(p) => (p, None),
            None => (internal, None),
        };
        Self {
            root,
            models_root,
            unavailable_models_root,
        }
    }

    /// The configured model location, when this session could not use it.
    pub fn unavailable_models_root(&self) -> Option<&Path> {
        self.unavailable_models_root.as_deref()
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
        // The location just proved itself writable, so nothing is missing any
        // more even if this session started with the drive detached.
        self.unavailable_models_root = None;
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
    /// Where the Hugging Face access token is kept. Gated repositories --
    /// every official FLUX.1 model among them -- cannot be downloaded without
    /// one, and it belongs to the user's account rather than to any model, so
    /// it lives beside the app's own state rather than in the model cache.
    pub fn hf_token(&self) -> PathBuf {
        self.root.join("hf_token")
    }

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

    /// Create every directory the app needs, reporting the first failure.
    ///
    /// It used to stop at the first error, which meant an unusable model
    /// location -- the first entry that can live on another volume -- also
    /// prevented the vault and the log directory from being created, on the
    /// same internal disk where they would have succeeded. Keep going, so a
    /// caller that chooses to continue anyway has as much as the machine can
    /// give it, and still return the error for callers that treat it as fatal.
    pub fn ensure_dirs(&self) -> Result<()> {
        let mut first: Option<std::io::Error> = None;
        for d in [self.runtime(), self.hf_hub(), self.vault(), self.logs()] {
            if let Err(error) = std::fs::create_dir_all(&d) {
                first = first.or(Some(error));
            }
        }
        match first {
            Some(error) => Err(error.into()),
            None => Ok(()),
        }
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

/// Whether weights can actually be stored at `p` right now.
///
/// The question is not whether the path exists -- a fresh external drive has no
/// `hub` directory yet -- but whether the directory the cache needs can be
/// created. That is the same call `ensure_dirs` makes, so a true answer here
/// means boot will not fail later on this path, and a false one means the drive
/// is detached, read-only, or no longer ours.
fn is_usable_models_root(p: &Path) -> bool {
    std::fs::create_dir_all(p.join("hub")).is_ok()
}

fn read_config(root: &Path) -> StoredConfig {
    std::fs::read_to_string(root.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway directory that cleans itself up, so these tests never touch
    /// a real installation.
    struct Scratch {
        path: PathBuf,
    }

    impl Scratch {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("modelstudio-paths-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn write_config(root: &Path, models_root: &Path) {
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(&StoredConfig {
                models_root: Some(models_root.to_string_lossy().to_string()),
            })
            .unwrap(),
        )
        .unwrap();
    }

    /// A models_root that cannot be created is what a detached external drive
    /// looks like: the path is under /Volumes, which belongs to root, so
    /// creating it fails with EACCES. Here the same failure is provoked with an
    /// ordinary file in the way, which fails for every user including root and
    /// stays inside the scratch directory.
    ///
    /// Boot used to panic on it before any window existed, leaving no way back
    /// except reattaching that drive or hand-editing config.json.
    #[test]
    fn an_unreachable_models_root_does_not_stop_boot() {
        let scratch = Scratch::new();
        let root = scratch.path.clone();
        let blocker = root.join("not-a-directory");
        std::fs::write(&blocker, b"x").unwrap();
        let configured = blocker.join("models");
        write_config(&root, &configured);

        let paths = AppPaths::at(root.clone());

        assert_eq!(
            paths.models_root(),
            &root.join("models"),
            "the session should fall back to the internal model directory"
        );
        assert_eq!(
            paths.unavailable_models_root(),
            Some(configured.as_path()),
            "the configured location has to stay visible so the UI can say what is missing"
        );
        assert!(
            paths.ensure_dirs().is_ok(),
            "boot must be able to create its directories after falling back"
        );
        assert!(paths.vault().is_dir());
        assert!(paths.logs().is_dir());
        assert!(paths.hf_hub().is_dir());

        // The user's choice must survive: the drive may come back, and the
        // models already on it are only reachable through this path.
        let stored = read_config(&root).models_root.unwrap();
        assert_eq!(PathBuf::from(stored), configured);
    }

    #[test]
    fn a_reachable_models_root_is_used_as_configured() {
        let scratch = Scratch::new();
        let root = scratch.path.clone();
        let external = root.join("elsewhere");
        write_config(&root, &external);

        let paths = AppPaths::at(root);

        assert_eq!(paths.models_root(), &external);
        assert_eq!(paths.unavailable_models_root(), None);
    }

    /// Creating directories used to stop at the first failure, and the model
    /// cache -- the one entry that can live on another volume -- comes before
    /// the vault and the logs.
    #[test]
    fn a_failed_model_cache_still_leaves_the_vault_and_logs_in_place() {
        let scratch = Scratch::new();
        let root = scratch.path.clone();
        let blocker = root.join("not-a-directory");
        std::fs::write(&blocker, b"x").unwrap();

        let paths = AppPaths {
            root: root.clone(),
            models_root: blocker.join("models"),
            unavailable_models_root: None,
        };

        assert!(
            paths.ensure_dirs().is_err(),
            "the failure still has to be reported to callers that treat it as fatal"
        );
        assert!(paths.vault().is_dir());
        assert!(paths.logs().is_dir());
        assert!(paths.runtime().is_dir());
    }
}
