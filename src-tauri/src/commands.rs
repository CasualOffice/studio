use crate::engine::Engine;
use crate::error::{AppError, Result};
use crate::hostinfo::HostInfo;
use crate::models::{self, CustomModel, ModelStatus};
use crate::paths::AppPaths;
use crate::setup::{self, SetupState};
use crate::vault::{RepairReport, Vault, VaultItem, VaultStatus};
use serde::Deserialize;
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, State};
use tokio::sync::{Mutex, RwLock};

pub struct AppState {
    /// Behind a lock because model storage can be relocated at runtime, and
    /// every later path lookup must see the new location without a restart.
    paths_inner: std::sync::RwLock<AppPaths>,
    pub vault: Arc<Vault>,
    pub engine: Mutex<Option<Arc<Engine>>>,
    job_gate: Arc<RwLock<()>>,
    setup_running: Arc<AtomicBool>,
}

impl AppState {
    pub fn new(paths: AppPaths, vault: Arc<Vault>) -> Self {
        Self {
            paths_inner: std::sync::RwLock::new(paths),
            vault,
            engine: Mutex::new(None),
            job_gate: Arc::new(RwLock::new(())),
            setup_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// A snapshot of the current paths. Cheap: `AppPaths` is two `PathBuf`s.
    pub fn paths(&self) -> AppPaths {
        self.paths_inner
            .read()
            .expect("paths lock poisoned")
            .clone()
    }

    fn replace_paths(&self, paths: AppPaths) {
        *self.paths_inner.write().expect("paths lock poisoned") = paths;
    }
}

impl AppState {
    /// Lazily start the worker, replacing it if the previous one died.
    ///
    /// Reusing the cached handle unconditionally meant that once the Python
    /// process exited -- an out-of-memory kill, a crash, or the user killing it
    /// -- every later request wrote to a closed pipe and failed until the whole
    /// app was restarted.
    async fn engine(&self, app: &AppHandle) -> Result<Arc<Engine>> {
        let mut guard = self.engine.lock().await;
        if let Some(e) = guard.as_ref() {
            if !e.is_dead() {
                return Ok(e.clone());
            }
            *guard = None;
        }
        let script = self.paths().worker_script(app)?;
        let engine = Engine::spawn(app, &self.paths(), &script).await?;
        *guard = Some(engine.clone());
        Ok(engine)
    }

    /// Drop the running worker, so the next request starts a fresh one.
    ///
    /// Used when something the worker only reads at spawn has changed -- the
    /// Hugging Face token, for instance. Nothing is killed mid-generation that
    /// the user did not ask to change.
    async fn stop_engine(&self) {
        let mut guard = self.engine.lock().await;
        if let Some(engine) = guard.take() {
            engine.shutdown().await;
        }
    }

    fn require_unlocked(&self) -> Result<()> {
        if !self.vault.is_unlocked() {
            return Err(AppError::VaultLocked);
        }
        Ok(())
    }
}

fn default_true() -> bool {
    true
}

fn new_job_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Host / setup
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn host_info(state: State<'_, AppState>) -> HostInfo {
    HostInfo::probe(&state.paths())
}

#[tauri::command]
pub fn setup_state(state: State<'_, AppState>) -> SetupState {
    setup::state(&state.paths())
}

#[tauri::command]
pub async fn run_setup(app: AppHandle, state: State<'_, AppState>, force: bool) -> Result<()> {
    if state.setup_running.swap(true, Ordering::AcqRel) {
        return Err(AppError::msg("Runtime setup is already running."));
    }
    let exclusive = state.job_gate.clone().try_write_owned().map_err(|_| {
        state.setup_running.store(false, Ordering::Release);
        AppError::msg("Wait for the current local job before repairing the runtime.")
    })?;
    let paths = state.paths();
    // Cleared on drop rather than by a statement after the await. A panic in
    // the setup task skipped that statement, leaving the flag set for the rest
    // of the process -- and every later attempt then answered "Runtime setup
    // is already running", permanently, with the only fix being a restart.
    struct ClearOnDrop(Arc<AtomicBool>);
    impl Drop for ClearOnDrop {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }
    let running = ClearOnDrop(state.setup_running.clone());
    tauri::async_runtime::spawn(async move {
        let _exclusive = exclusive;
        let _running = running;
        let _ = setup::run(app, paths, force).await;
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Vault
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn vault_status(state: State<'_, AppState>) -> VaultStatus {
    state.vault.status()
}

#[tauri::command]
pub fn vault_create(
    state: State<'_, AppState>,
    passphrase: String,
    enable_biometry: bool,
) -> Result<VaultStatus> {
    state.vault.create(&passphrase, enable_biometry)?;
    Ok(state.vault.status())
}

#[tauri::command]
pub fn vault_unlock_passphrase(
    state: State<'_, AppState>,
    passphrase: String,
) -> Result<VaultStatus> {
    state.vault.unlock_with_passphrase(&passphrase)?;
    Ok(state.vault.status())
}

/// Triggers the system Touch ID prompt. Runs on a blocking thread because the
/// Keychain call is synchronous and shows UI.
#[tauri::command]
pub async fn vault_unlock_biometry(state: State<'_, AppState>) -> Result<VaultStatus> {
    let vault = state.vault.clone();
    let res = tauri::async_runtime::spawn_blocking(move || {
        vault.unlock_with_biometry().map(|()| vault.status())
    })
    .await
    .map_err(|e| AppError::msg(format!("authentication task failed: {e}")))?;
    Ok(res?)
}

#[tauri::command]
pub async fn vault_lock(_app: AppHandle, state: State<'_, AppState>) -> Result<VaultStatus> {
    let _exclusive = state.job_gate.try_write().map_err(|_| {
        AppError::msg("A local job is still running. Cancel it or wait before locking.")
    })?;
    // A job that finishes after the key is destroyed cannot commit its sealed
    // output to the index. Refuse the lock while a request is active; the idle
    // timer will try again after later activity, and an explicit cancel can
    // still stop the work first.
    let engine = state.engine.lock().await.as_ref().cloned();
    if let Some(engine) = engine.as_ref() {
        if engine.has_pending().await {
            return Err(AppError::msg(
                "A local job is still running. Cancel it or wait for it to finish before locking.",
            ));
        }
    }

    state.vault.lock();
    // Locking should also drop model weights. Use only an existing worker: a
    // lock action must never start a fresh engine just to unload it.
    if let Some(engine) = engine {
        let _ = engine.request(&new_job_id(), "unload", json!({})).await;
    }
    Ok(state.vault.status())
}

#[tauri::command]
pub async fn vault_enable_biometry(state: State<'_, AppState>) -> Result<VaultStatus> {
    let vault = state.vault.clone();
    let res = tauri::async_runtime::spawn_blocking(move || {
        vault.enable_biometry().map(|()| vault.status())
    })
    .await
    .map_err(|e| AppError::msg(format!("keychain task failed: {e}")))?;
    Ok(res?)
}

#[tauri::command]
pub fn vault_disable_biometry(state: State<'_, AppState>) -> Result<VaultStatus> {
    state.vault.disable_biometry()?;
    Ok(state.vault.status())
}

#[tauri::command]
pub fn vault_change_passphrase(
    state: State<'_, AppState>,
    current: String,
    next: String,
) -> Result<()> {
    state.vault.change_passphrase(&current, &next)?;
    Ok(())
}

/// Reconcile the index against the blobs on disk.
#[tauri::command]
pub fn vault_repair(state: State<'_, AppState>) -> Result<RepairReport> {
    Ok(state.vault.repair()?)
}

#[tauri::command]
pub fn vault_list(state: State<'_, AppState>) -> Result<Vec<VaultItem>> {
    Ok(state.vault.list()?)
}

/// The in-progress Board is private project content, not an interface
/// preference. Keep it in the encrypted index so locking the vault locks the
/// manuscript, cast, notes and panel plan as well.
#[tauri::command]
pub fn board_state_get(state: State<'_, AppState>, key: Option<String>) -> Result<Option<String>> {
    Ok(state.vault.get_state(&board_key(key)?)?)
}

/// Which board this state belongs to.
///
/// `board-draft` is the one in front of you. Every comic that has been divided
/// also keeps its own entry, `board-<project id>`, so a finished board can be
/// opened again months later -- the pictures were always in the vault, but the
/// division, the notes and the character binding lived only in the single
/// draft slot and were overwritten by the next story.
fn board_key(key: Option<String>) -> Result<String> {
    let key = key.unwrap_or_else(|| "board-draft".into());
    if key != "board-draft" && !key.starts_with("board-") {
        return Err(AppError::msg("not a Board key"));
    }
    if key.len() > 64 || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
        return Err(AppError::msg("not a Board key"));
    }
    Ok(key)
}

#[tauri::command]
pub fn board_state_set(
    state: State<'_, AppState>,
    value: String,
    key: Option<String>,
) -> Result<()> {
    if value.len() > 2 * 1024 * 1024 {
        return Err(AppError::msg("the Board draft is unexpectedly large"));
    }
    serde_json::from_str::<serde_json::Value>(&value)
        .map_err(|e| AppError::msg(format!("the Board draft is not valid JSON: {e}")))?;
    Ok(state.vault.set_state(&board_key(key)?, value)?)
}

/// The project ids of every board that can be reopened.
#[tauri::command]
pub fn board_state_list(state: State<'_, AppState>) -> Result<Vec<String>> {
    Ok(state
        .vault
        .state_keys()?
        .into_iter()
        .filter(|k| k.starts_with("board-") && k != "board-draft")
        .map(|k| k["board-".len()..].to_string())
        .collect())
}

/// Forget a saved board. The pictures it made stay in the vault.
#[tauri::command]
pub fn board_state_forget(state: State<'_, AppState>, key: String) -> Result<()> {
    let key = board_key(Some(key))?;
    if key == "board-draft" {
        return Err(AppError::msg("that is the open board, not a saved one"));
    }
    Ok(state.vault.remove_state(&key)?)
}

#[tauri::command]
pub fn vault_delete(state: State<'_, AppState>, id: String) -> Result<()> {
    Ok(state.vault.delete(&id)?)
}

/// The single sanctioned path for plaintext to leave the vault.
#[tauri::command]
pub fn vault_export(
    state: State<'_, AppState>,
    id: String,
    dest: String,
    overwrite: bool,
) -> Result<u64> {
    Ok(state
        .vault
        .export(&id, std::path::Path::new(&dest), overwrite)?)
}

/// Store bytes the interface produced, such as a painted mask.
///
/// Masks are transient working data but still go through the vault: they are
/// derived from a private image and would otherwise be the one thing written
/// to disk in the clear.
#[tauri::command]
pub fn vault_import_bytes(
    state: State<'_, AppState>,
    data: Vec<u8>,
    name: String,
    mime: String,
    kind: String,
) -> Result<String> {
    state.require_unlocked()?;
    if data.is_empty() {
        return Err(AppError::msg("nothing to store"));
    }
    if !matches!(kind.as_str(), "mask" | "adjust") {
        return Err(AppError::msg("unsupported in-memory import kind"));
    }
    let safe_name = std::path::Path::new(&name)
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty())
        .unwrap_or("imported")
        .to_string();
    let item = VaultItem {
        id: uuid::Uuid::new_v4().to_string(),
        content_hash: None,
        kind,
        name: safe_name,
        mime,
        bytes: data.len() as u64,
        model: String::new(),
        prompt: String::new(),
        seed: 0,
        width: None,
        height: None,
        steps: None,
        guidance: None,
        inputs: vec![],
        created_at: chrono::Local::now().to_rfc3339(),
        duration_ms: 0,
        project: None,
        project_name: None,
        project_index: None,
        ..Default::default()
    };
    Ok(state.vault.put(&data, item)?)
}

/// Bring an outside file in. The original is left untouched; only a sealed
/// copy enters the vault.
#[tauri::command]
pub fn vault_import(state: State<'_, AppState>, source: String, kind: String) -> Result<String> {
    state.require_unlocked()?;
    let src = std::path::Path::new(&source);
    if !src.exists() {
        return Err(AppError::msg("that file no longer exists"));
    }
    let bytes = std::fs::read(src)?;
    let name = src
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("imported")
        .to_string();
    let ext = src
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let mime = mime_for(&ext);
    if kind == "image" && !mime.starts_with("image/") {
        return Err(AppError::msg(format!("{name} is not an image")));
    }

    let item = VaultItem {
        id: uuid::Uuid::new_v4().to_string(),
        content_hash: None,
        kind: if mime.starts_with("image/") {
            "import".into()
        } else {
            "doc".into()
        },
        name,
        mime: mime.into(),
        bytes: bytes.len() as u64,
        model: String::new(),
        prompt: String::new(),
        seed: 0,
        width: None,
        height: None,
        steps: None,
        guidance: None,
        inputs: vec![],
        created_at: chrono::Local::now().to_rfc3339(),
        duration_ms: 0,
        project: None,
        project_name: None,
        project_index: None,
        ..Default::default()
    };
    let before = state.vault.list()?.len();
    let id = state.vault.put(&bytes, item)?;
    let deduplicated = state.vault.list()?.len() == before;
    if deduplicated {
        return Err(AppError::AlreadyInVault(id));
    }
    Ok(id)
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
pub struct StorageInfo {
    pub models_root: String,
    pub models_bytes: u64,
    pub volume_free_gib: f32,
    pub is_external: bool,
}

/// What the cache is holding that nothing points at.
#[tauri::command]
pub fn find_orphans(state: State<'_, AppState>) -> models::Orphans {
    models::find_orphans(&state.paths())
}

/// Delete it, returning the bytes recovered.
#[tauri::command]
pub fn sweep_orphans(state: State<'_, AppState>) -> Result<u64> {
    sweep_orphans_gated(&state)
}

/// The body of `sweep_orphans`, split out only because `State` cannot be built
/// outside a running Tauri app and the gate below is worth a test.
///
/// This is a model-store mutation and takes the exclusive job gate like every
/// other one. It used to take nothing at all, and huggingface_hub materialises a
/// blob in `blobs/` before it creates the snapshot symlink that points at it --
/// so a sweep that ran during a download saw a weight file that had just landed
/// as unreferenced and deleted it out from under the downloader.
fn sweep_orphans_gated(state: &AppState) -> Result<u64> {
    let _exclusive = state.job_gate.try_write().map_err(|_| {
        AppError::msg("Wait for the current local job before clearing unused files.")
    })?;
    models::sweep_orphans(&state.paths())
}

#[tauri::command]
pub fn storage_info(state: State<'_, AppState>) -> StorageInfo {
    let root = state.paths().models_root().clone();
    let host = HostInfo::probe_for(&root);
    StorageInfo {
        models_bytes: models::dir_size_of(&root),
        volume_free_gib: host.free_disk_gib,
        is_external: root.starts_with("/Volumes/"),
        models_root: root.to_string_lossy().to_string(),
    }
}

/// Top-level entries in a model root that this app puts there itself.
///
/// `HF_HOME` is pointed at the model root, so everything the downloader writes
/// lands directly inside it: the hub cache, the Xet chunk cache, the lock
/// directory it serialises concurrent fetches with, and the few smaller caches
/// and token files huggingface_hub keeps beside them. Nothing else in there is
/// ours, and a move may only touch what is.
const APP_OWNED_MODEL_ENTRIES: &[&str] = &[
    "hub",
    "xet",
    ".locks",
    "assets",
    "datasets",
    "modules",
    "token",
    "stored_tokens",
];

/// Prefix of the staging directory a move copies into, recognised by name so an
/// abandoned one is neither mistaken for the user's own files nor left to block
/// every later move.
const MOVE_STAGING_PREFIX: &str = ".modelstudio-move-";

/// Whether an entry name is one a move may copy and then delete.
fn is_app_owned_model_entry(name: &str) -> bool {
    APP_OWNED_MODEL_ENTRIES.contains(&name)
}

/// Whether an entry is neither ours nor worth refusing over. Finder writes a
/// `.DS_Store` into any folder that has been looked at, and an abandoned
/// staging directory holds nothing but a copy of a model root.
fn is_ignorable_model_entry(name: &str) -> bool {
    name == ".DS_Store" || name.starts_with(MOVE_STAGING_PREFIX)
}

/// Names in a model root that this app did not create.
fn foreign_model_entries(root: &std::path::Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|name| !is_app_owned_model_entry(name) && !is_ignorable_model_entry(name))
        .collect();
    names.sort();
    names
}

/// The entries a move is allowed to copy, and afterwards to remove.
fn app_owned_model_entries(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|e| is_app_owned_model_entry(&e.file_name().to_string_lossy()))
        .map(|e| e.path())
        .collect()
}

/// Staging directories left behind by an interrupted move. Their contents are
/// always a copy of a model root that still exists, so they are disposable.
fn stale_staging_dirs(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(MOVE_STAGING_PREFIX)
        })
        .map(|e| e.path())
        .collect()
}

/// Whether the destination holds anything a move could overwrite.
fn dest_holds_other_files(dest: &std::path::Path) -> Result<bool> {
    for entry in std::fs::read_dir(dest)? {
        if !is_ignorable_model_entry(&entry?.file_name().to_string_lossy()) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Copy one top-level entry of a model root. Usually a directory, but
/// huggingface_hub also caches plain files such as `token` beside them.
fn copy_model_entry(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    if std::fs::symlink_metadata(src)?.is_dir() {
        models::copy_tree(src, dst)
    } else {
        std::fs::copy(src, dst)?;
        Ok(())
    }
}

/// Delete one, whichever kind it is. Best effort: the bytes are already safe at
/// the destination by the time this runs, so a failure here is wasted space
/// rather than lost data.
fn remove_model_entry(path: &std::path::Path) {
    let is_dir = std::fs::symlink_metadata(path)
        .map(|md| md.is_dir())
        .unwrap_or(false);
    if is_dir {
        let _ = std::fs::remove_dir_all(path);
    } else {
        let _ = std::fs::remove_file(path);
    }
}

/// The first few names, for an error message that has to fit in a dialog.
fn name_list(names: &[String]) -> String {
    let shown = names.len().min(3);
    let mut out = names[..shown].join(", ");
    if names.len() > shown {
        out.push_str(&format!(" and {} more", names.len() - shown));
    }
    out
}

/// Move model storage to another location, e.g. an external drive.
///
/// Copying rather than renaming, because the destination is usually a
/// different filesystem where rename cannot work. The originals are only
/// removed once every byte has landed, and only the entries this app created
/// are copied or removed at all.
#[tauri::command]
pub async fn set_models_location(
    app: AppHandle,
    state: State<'_, AppState>,
    path: String,
    move_existing: bool,
) -> Result<StorageInfo> {
    let _exclusive = state.job_gate.try_write().map_err(|_| {
        AppError::msg("Wait for the current local job before moving model storage.")
    })?;
    let dest = std::path::PathBuf::from(path.trim());
    if !dest.is_absolute() {
        return Err(AppError::msg("Choose a folder, not a relative path."));
    }
    std::fs::create_dir_all(&dest)?;

    // Refuse to move onto a volume that cannot hold what is already stored.
    let current = state.paths().models_root().clone();
    let current_cmp = current.canonicalize().unwrap_or_else(|_| current.clone());
    let dest_cmp = dest.canonicalize().unwrap_or_else(|_| dest.clone());
    if current_cmp != dest_cmp
        && (dest_cmp.starts_with(&current_cmp) || current_cmp.starts_with(&dest_cmp))
    {
        return Err(AppError::msg(
            "Choose a folder outside the current model folder. Parent and child locations cannot be moved into each other.",
        ));
    }
    let needed = models::dir_size_of(&current);
    if move_existing && needed > 0 {
        let host = HostInfo::probe_for(&dest);
        let free = (host.free_disk_gib * 1024.0 * 1024.0 * 1024.0) as u64;
        if needed + 2 * 1024 * 1024 * 1024 > free {
            return Err(AppError::msg(format!(
                "That volume has {:.1} GB free but the models need {:.1} GB.",
                host.free_disk_gib,
                needed as f32 / 1024.0 / 1024.0 / 1024.0
            )));
        }
    }

    // Check everything that can be checked before stopping anything. The
    // engine used to be shut down first, so choosing a folder that turned out
    // to be non-empty left the app with no engine and nothing moved -- a
    // rejected request that still cost the user their running session.
    if move_existing && current.exists() && current_cmp != dest_cmp {
        // Refuse outright when the folder this is about to empty holds anything
        // the app did not put there. The emptiness check below only ever ran
        // with `move_existing` set, so changing the location *without* moving
        // accepted a folder full of the user's own files and made it the model
        // root -- and a later move with the box ticked then copied that whole
        // folder and removed the original.
        let foreign = foreign_model_entries(&current);
        if !foreign.is_empty() {
            return Err(AppError::msg(format!(
                "The current model folder holds files this app did not create ({}). \
                 Move them elsewhere first, or change the location without moving.",
                name_list(&foreign)
            )));
        }
        if dest_holds_other_files(&dest)? {
            return Err(AppError::msg(
                "Choose an empty destination folder so existing files cannot be overwritten.",
            ));
        }
    }

    // The engine holds the old location in its environment.
    {
        let mut guard = state.engine.lock().await;
        if let Some(engine) = guard.take() {
            engine.shutdown().await;
        }
    }

    if move_existing && current.exists() && current_cmp != dest_cmp {
        let entries = app_owned_model_entries(&current);
        // Stage inside the destination, not beside it. The staging directory
        // used to be created at `dest.parent()`, which for `/Volumes/T7` is
        // `/Volumes` -- on the boot volume, whose free space the check above
        // never probed. The rename into place then failed with EXDEV, and unlike
        // the copy-failure branch this path never removed the staging copy, so
        // tens of gigabytes were left in a dot-directory nobody would look for.
        for stale in stale_staging_dirs(&dest) {
            let _ = std::fs::remove_dir_all(stale);
        }
        let staging = dest.join(format!("{MOVE_STAGING_PREFIX}{}", uuid::Uuid::new_v4()));
        let sources = entries.clone();
        let to = staging.clone();
        let copied = tauri::async_runtime::spawn_blocking(move || -> Result<()> {
            std::fs::create_dir_all(&to)?;
            for src in &sources {
                let Some(name) = src.file_name() else {
                    continue;
                };
                copy_model_entry(src, &to.join(name))?;
            }
            Ok(())
        })
        .await
        .map_err(|e| AppError::msg(format!("move task failed: {e}")))?;
        if let Err(error) = copied {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(error);
        }
        // Staging and destination are the same filesystem, so each of these is
        // a directory-entry move rather than a second copy. Anything that fails
        // here is cleaned up on both sides: the originals are still untouched,
        // so the copies are what gets thrown away.
        for src in &entries {
            let Some(name) = src.file_name() else {
                continue;
            };
            if let Err(error) = std::fs::rename(staging.join(name), dest.join(name)) {
                for entry in &entries {
                    if let Some(name) = entry.file_name() {
                        remove_model_entry(&dest.join(name));
                    }
                }
                let _ = std::fs::remove_dir_all(&staging);
                return Err(error.into());
            }
        }
        let _ = std::fs::remove_dir(&staging);
    }

    let mut paths = state.paths();
    // The old directory is intentionally still intact until this succeeds. If
    // configuration persistence fails, the copied destination can be selected
    // again and no model data has been removed.
    paths.set_models_root(dest.clone())?;
    state.replace_paths(paths.clone());
    if move_existing && current.exists() && current_cmp != dest_cmp {
        // Only what was copied. This used to be `remove_dir_all(&current)`,
        // which for a model root the user had pointed at a folder of their own
        // deleted everything in it along with the cache.
        for entry in app_owned_model_entries(&current) {
            remove_model_entry(&entry);
        }
        // Succeeds only if nothing else is in there, which is the intent: a
        // stray `.DS_Store` keeps the folder rather than losing it.
        let _ = std::fs::remove_dir(&current);
    }
    let _ = app;
    Ok(storage_info_for(&paths))
}

fn storage_info_for(paths: &AppPaths) -> StorageInfo {
    let root = paths.models_root().clone();
    let host = HostInfo::probe_for(&root);
    StorageInfo {
        models_bytes: models::dir_size_of(&root),
        volume_free_gib: host.free_disk_gib,
        is_external: root.starts_with("/Volumes/"),
        models_root: root.to_string_lossy().to_string(),
    }
}

#[tauri::command]
pub fn list_models(state: State<'_, AppState>) -> Result<Vec<ModelStatus>> {
    let host = HostInfo::probe(&state.paths());
    models::list(&state.paths(), &host)
}

#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model_id: String,
    job_id: String,
) -> Result<serde_json::Value> {
    let _job = state.job_gate.read().await;
    let host = HostInfo::probe(&state.paths());
    let entry = models::find(&state.paths(), &host, &model_id)
        .ok_or_else(|| AppError::msg(format!("unknown model: {model_id}")))?;

    if !entry.installed {
        let need = entry.package_gib;
        let have = host.free_disk_gib - host.disk_headroom_gib;
        if need > have {
            return Err(AppError::msg(format!(
                "{} needs {:.1} GiB but only {:.1} GiB is free after headroom. \
                 Delete a model first.",
                entry.name, need, have
            )));
        }
    }

    let engine = state.engine(&app).await?;
    engine
        .request(
            &job_id,
            "download",
            json!({
                "model": entry.repo.clone(),
                // The catalog's published package size is a better progress
                // denominator than the repository total: MLX-Gen fetches a
                // weight/tokenizer subset, not every file in the repo.
                "expected_bytes": (entry.package_gib as f64 * 1024.0 * 1024.0 * 1024.0) as u64,
                // The prompt assistant is a VLM that MLX-Gen does not know
                // about. `mlxgen download` exits 0 without fetching anything
                // for an unrecognised repo, so it has to come through
                // huggingface_hub directly.
                // `mlxgen download` exits 0 without fetching anything for a
                // repo it does not recognise, which looks exactly like a
                // successful zero-byte download. Two kinds of model are
                // outside its registry: the prompt assistant, which is a VLM,
                // and FLUX.1, which mflux runs. Both come through
                // huggingface_hub directly.
                "via": if entry.backend.is_some() {
                    "mflux"
                } else if entry.tasks.contains(&crate::catalog::Task::Assist) {
                    "hf"
                } else {
                    "mlxgen"
                },
            }),
        )
        .await
}

#[tauri::command]
pub fn delete_model(state: State<'_, AppState>, model_id: String) -> Result<u64> {
    let _exclusive = state
        .job_gate
        .try_write()
        .map_err(|_| AppError::msg("Wait for the current local job before deleting a model."))?;
    let host = HostInfo::probe(&state.paths());
    let entry = models::find(&state.paths(), &host, &model_id)
        .ok_or_else(|| AppError::msg(format!("unknown model: {model_id}")))?;
    models::delete(&state.paths(), &entry.repo)
}

/// Media type for an extension.
///
/// WKWebView renders HEIC, AVIF and WebP natively, so originals can be stored
/// and displayed as they are rather than transcoded on the way in — the engine
/// converts to PNG only when a model is about to read the file.
/// Inverse of `mime_for`, for naming a staged temporary file.
pub fn ext_for_mime(mime: &str) -> Option<&'static str> {
    Some(match mime {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        "image/avif" => "avif",
        "image/heic" => "heic",
        "image/tiff" => "tif",
        "image/bmp" => "bmp",
        "image/gif" => "gif",
        _ => return None,
    })
}

pub fn mime_for(ext: &str) -> &'static str {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" | "jpe" => "image/jpeg",
        "webp" => "image/webp",
        "avif" | "avifs" => "image/avif",
        "heic" | "heif" | "heics" => "image/heic",
        "tif" | "tiff" => "image/tiff",
        "bmp" | "dib" => "image/bmp",
        "gif" => "image/gif",
        "ico" => "image/x-icon",
        "jp2" | "j2k" | "jpf" | "jpx" => "image/jp2",
        "tga" => "image/x-tga",
        "ppm" | "pgm" | "pbm" => "image/x-portable-anymap",
        "pdf" => "application/pdf",
        "txt" | "md" => "text/plain",
        "json" => "application/json",
        _ => "application/octet-stream",
    }
}

/// Accept anything that identifies a Hugging Face repo: a bare `owner/name`,
/// a full page URL, or a link to a file or tab inside the repo.
fn normalize_repo(input: &str) -> Result<String> {
    let mut s = input.trim();

    for prefix in [
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://www.huggingface.co/",
        "huggingface.co/",
        "hf.co/",
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest;
            break;
        }
    }
    // Drop a query string or fragment from a copied address-bar URL.
    s = s.split(['?', '#']).next().unwrap_or(s).trim_matches('/');

    // A link to some other site is not a repo id, and must not be mangled into
    // one: "https://example.com" would otherwise parse as "https:/example.com".
    if s.contains("://") || s.starts_with("www.") {
        return Err(AppError::msg(
            "That link is not on huggingface.co. Paste a Hugging Face model link, \
             or type owner/name.",
        ));
    }

    // Model pages carry extra path segments: /tree/main, /blob/..., /discussions.
    let parts: Vec<&str> = s.split('/').filter(|p| !p.is_empty()).collect();
    let repo = match parts.as_slice() {
        // Some org pages are prefixed with a namespace kind.
        ["models", owner, name, ..] => format!("{owner}/{name}"),
        [owner, name, ..] => format!("{owner}/{name}"),
        _ => {
            return Err(AppError::msg(
                "That does not look like a Hugging Face model. Paste a link such as \
                 https://huggingface.co/owner/name, or just type owner/name.",
            ))
        }
    };

    let valid = |part: &str| {
        !part.is_empty()
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    if repo.matches('/').count() != 1 || !repo.split('/').all(valid) {
        return Err(AppError::msg(
            "Enter a Hugging Face repo as owner/name, or paste its page link.",
        ));
    }
    Ok(repo)
}

/// Inspect a Hugging Face repo before offering to download it.
#[tauri::command]
pub async fn resolve_model(
    app: AppHandle,
    state: State<'_, AppState>,
    repo: String,
) -> Result<serde_json::Value> {
    let _job = state.job_gate.read().await;
    let repo = normalize_repo(&repo)?;
    let engine = state.engine(&app).await?;
    let mut resolved = engine
        .request(&new_job_id(), "resolve", json!({ "model": repo }))
        .await?;

    // Judge it against this machine before a single byte is spent.
    let host = HostInfo::probe(&state.paths());
    let bytes = resolved["bytes"].as_u64().unwrap_or(0);
    let package_gib = bytes as f32 / 1024.0 / 1024.0 / 1024.0;
    let (fit, reason) = models::judge_uninstalled(package_gib, &host);
    let peak = models::estimate_peak_for(package_gib);

    resolved["package_gib"] = json!(package_gib);
    resolved["peak_gib"] = json!(peak);
    resolved["fit"] = serde_json::to_value(fit)?;
    resolved["fit_reason"] = json!(reason);
    resolved["required_ram_gib"] = json!(models::required_ram_gib(peak));

    // What this machine can take, so a refusal explains itself in the same
    // terms the user was thinking in.
    resolved["max_params_4bit"] = json!(host.max_params_4bit);
    resolved["max_params_8bit"] = json!(host.max_params_8bit);
    Ok(resolved)
}

/// What the resolver worked out about a repository, ready to be saved.
///
/// One struct rather than eight positional arguments: the list had grown to
/// the point where two adjacent `Option<String>`s could be swapped silently.
#[derive(serde::Deserialize)]
pub struct NewModel {
    pub repo: String,
    pub name: String,
    pub tasks: Vec<String>,
    pub bytes: u64,
    pub quantize: Option<u8>,
    /// Router family, when the resolver had to identify it from the repo.
    pub family: Option<String>,
    /// Names an mflux backend for FLUX.1, which the unified router cannot place.
    pub backend: Option<String>,
}

#[tauri::command]
pub fn add_custom_model(state: State<'_, AppState>, spec: NewModel) -> Result<()> {
    let NewModel {
        repo,
        name,
        tasks,
        bytes,
        quantize,
        family,
        backend,
    } = spec;
    let repo = normalize_repo(&repo)?;
    if tasks.is_empty() {
        return Err(AppError::msg(
            "The engine cannot route this repository, so it cannot be generated from.",
        ));
    }
    // A stable id derived from the repo keeps re-adding idempotent.
    let id = format!("custom:{}", repo.replace('/', "--"));
    models::add_custom(
        &state.paths(),
        CustomModel {
            id,
            name: if name.trim().is_empty() {
                repo.clone()
            } else {
                name
            },
            repo,
            tasks,
            quantize,
            package_gib: bytes as f32 / 1024.0 / 1024.0 / 1024.0,
            family,
            backend,
            steps_default: 8,
            added_at: chrono::Local::now().to_rfc3339(),
        },
    )
}

#[tauri::command]
pub fn remove_custom_model(state: State<'_, AppState>, model_id: String) -> Result<()> {
    let _exclusive = state.job_gate.try_write().map_err(|_| {
        AppError::msg("Wait for the current local job before changing the model catalog.")
    })?;
    models::remove_custom(&state.paths(), &model_id)
}

#[tauri::command]
pub async fn unload_model(app: AppHandle, state: State<'_, AppState>) -> Result<()> {
    let _exclusive = state
        .job_gate
        .try_write()
        .map_err(|_| AppError::msg("Wait for the current local job before unloading the model."))?;
    let engine = state.engine(&app).await?;
    engine.request(&new_job_id(), "unload", json!({})).await?;
    Ok(())
}

#[tauri::command]
pub async fn engine_ping(app: AppHandle, state: State<'_, AppState>) -> Result<serde_json::Value> {
    let _job = state.job_gate.read().await;
    let engine = state.engine(&app).await?;
    engine.request(&new_job_id(), "ping", json!({})).await
}

#[tauri::command]
pub async fn cancel_job(_app: AppHandle, state: State<'_, AppState>, job_id: String) -> Result<()> {
    let engine = state
        .engine
        .lock()
        .await
        .as_ref()
        .cloned()
        .ok_or_else(|| AppError::msg("that job is no longer running"))?;
    engine.cancel(&job_id).await
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
pub struct GenerateArgs {
    pub job_id: String,
    pub model_id: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: Option<f32>,
    pub seed: i64,
    pub count: u32,
    /// Vault item ids used as sources, never filesystem paths.
    #[serde(default)]
    pub images: Vec<String>,
    pub image_strength: Option<f32>,
    pub i2i_mode: Option<String>,
    #[serde(default)]
    pub low_ram: bool,
    pub cache_limit_gb: Option<f32>,
    #[serde(default)]
    pub allow_over_budget: bool,
    /// Show the picture forming, step by step.
    #[serde(default = "default_true")]
    pub preview: bool,
    /// The project this output belongs to, if any.
    ///
    /// A picture board sets this on every panel it draws, so the library shows
    /// one comic rather than seventeen loose pictures with the same timestamp.
    /// Absent for an ordinary one-off generation.
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub project_name: Option<String>,
    #[serde(default)]
    pub project_index: Option<u32>,
    /// Adapter handles with their strengths.
    #[serde(default)]
    pub loras: Vec<(String, f32)>,
    /// Vault id of a painted mask: white where the model may change the image.
    pub mask: Option<String>,
    /// CSS-like padding for outpainting, e.g. "10%,25%,10%,25%".
    pub outpaint_padding: Option<String>,
    /// How the added area is seeded before denoising.
    pub outpaint_fill: Option<String>,
}

/// What produced this, in one line.
///
/// Upstream drift has silently changed behaviour twice -- a model that stopped
/// loading on an mlx-gen bump, and a download path that started reporting
/// every repository as gated. Neither was visible from the item it produced,
/// because nothing recorded what had made it.
fn engine_versions() -> String {
    format!(
        "app {} · mlx-gen {}",
        env!("CARGO_PKG_VERSION"),
        crate::setup::MLX_GEN_VERSION,
    )
}

async fn run_job(
    app: &AppHandle,
    state: &AppState,
    args: GenerateArgs,
    op: &str,
) -> Result<Vec<String>> {
    let _job = state.job_gate.read().await;
    state.require_unlocked()?;

    let host = HostInfo::probe(&state.paths());
    let entry = models::find(&state.paths(), &host, &args.model_id)
        .ok_or_else(|| AppError::msg(format!("unknown model: {}", args.model_id)))?;

    if let Some(reason) = entry.broken.as_deref() {
        return Err(AppError::msg(format!(
            "{} cannot run. {}",
            entry.name, reason
        )));
    }

    if !entry.installed {
        return Err(AppError::msg(format!(
            "{} is not downloaded yet. Install it from the Models tab.",
            entry.name
        )));
    }

    if entry.peak_gib > host.usable_ram_gib {
        if entry.hopeless {
            return Err(AppError::msg(format!(
                "{} cannot run here: its weights alone are ~{:.1} GiB against ~{:.1} GiB \
                 usable. Low-RAM mode trims transient memory, not resident weights.",
                entry.name,
                models::weight_floor_gib(entry.package_gib, entry.peak_gib),
                host.usable_ram_gib
            )));
        }
        if !args.allow_over_budget {
            return Err(AppError::msg(format!(
                "{} peaks around {:.1} GiB but this Mac can offer about {:.1} GiB. \
                 Its ~{:.1} GiB weight floor does fit, so low-RAM mode may close the gap \
                 — enable it and try anyway.",
                entry.name,
                entry.peak_gib,
                host.usable_ram_gib,
                models::weight_floor_gib(entry.package_gib, entry.peak_gib)
            )));
        }
    }

    let count = args.count.max(1);
    let seeds: Vec<i64> = (0..count as i64).map(|i| args.seed + i).collect();

    // Reserve one sealed destination per image. The engine receives a key that
    // opens only these slots; the vault's data key stays in this process.
    let mut slots = Vec::new();
    let mut slot_ids = Vec::new();
    for _ in 0..count {
        let (id, file_id, key, path) = state.vault.reserve_slot()?;
        slot_ids.push(id.clone());
        slots.push(json!({
            "id": id,
            "file_id": hex(&file_id),
            "key": hex(&key),
            "path": path.to_string_lossy(),
        }));
    }

    // Source images are handed over the same way: one key per input blob.
    let mut vault_inputs = Vec::new();
    for src_id in &args.images {
        let (file_id, key, path) = state.vault.input_key(src_id)?;
        let ext = state
            .vault
            .get_item(src_id)
            .ok()
            .and_then(|i| ext_for_mime(&i.mime))
            .unwrap_or("png");
        vault_inputs.push(json!({
            "id": src_id,
            "file_id": hex(&file_id),
            "key": hex(&key),
            "path": path.to_string_lossy(),
            "ext": ext,
        }));
    }

    let mut params = json!({
        "model": entry.repo.clone(),
        "quantize": entry.quantize,
        // Routing hint. MLX-Gen otherwise infers the family from the
        // repository name and refuses anything it cannot place.
        "family": entry.family.clone(),
        // Names an mflux backend for FLUX.1, which the unified router does not
        // cover. Absent for everything else.
        "backend": entry.backend.clone(),
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seeds": seeds,
        "vault_slots": slots,
        "vault_inputs": vault_inputs,
    });
    if let Some(g) = args.guidance {
        params["guidance"] = json!(g);
    }
    if let Some(n) = args
        .negative_prompt
        .as_ref()
        .filter(|s| !s.trim().is_empty())
    {
        params["negative_prompt"] = json!(n);
    }
    if let Some(s) = args.image_strength {
        params["image_strength"] = json!(s);
    }
    if let Some(m) = args.i2i_mode.as_ref() {
        params["i2i_mode"] = json!(m);
    }
    if args.low_ram {
        params["low_ram"] = json!(true);
    }
    params["preview"] = json!(args.preview);
    // What this run is expected to need. The engine compares it against the
    // memory actually free, which the catalog cannot know: a model that fits
    // an empty machine still dies when something else is holding the memory,
    // and a Metal out-of-memory aborts the engine rather than raising.
    params["peak_gib"] = json!(entry.peak_gib);
    if !args.loras.is_empty() {
        params["loras"] = json!(args
            .loras
            .iter()
            .map(|(path, scale)| json!({ "path": path, "scale": scale }))
            .collect::<Vec<_>>());
    }
    if let Some(c) = args.cache_limit_gb {
        params["cache_limit_gb"] = json!(c);
    }
    if let Some(mask_id) = args.mask.as_ref() {
        let (file_id, key, path) = state.vault.input_key(mask_id)?;
        params["mask"] = json!({
            "id": mask_id,
            "file_id": hex(&file_id),
            "key": hex(&key),
            "path": path.to_string_lossy(),
            "ext": "png",
        });
    }
    if let Some(p) = args
        .outpaint_padding
        .as_ref()
        .filter(|s| !s.trim().is_empty())
    {
        params["outpaint_padding"] = json!(p);
        if let Some(f) = args.outpaint_fill.as_ref() {
            params["outpaint_fill"] = json!(f);
        }
    }

    let engine = state.engine(app).await?;
    let started = std::time::Instant::now();
    let result = engine.request(&args.job_id, op, params).await;
    let elapsed = started.elapsed().as_millis() as u64;

    // A failed or cancelled run must not leave orphaned blobs behind.
    let result = match result {
        Ok(r) => r,
        Err(e) => {
            state.vault.discard_slots(&slot_ids);
            return Err(e);
        }
    };

    let produced: Vec<String> = result["outputs"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let sizes: Vec<u64> = result["sizes"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();

    for (i, id) in produced.iter().enumerate() {
        state.vault.commit_slot(VaultItem {
            id: id.clone(),
            content_hash: None,
            kind: op.to_string(),
            name: format!(
                "{}-{}.png",
                op,
                chrono::Local::now().format("%Y%m%d-%H%M%S")
            ),
            mime: "image/png".into(),
            bytes: sizes.get(i).copied().unwrap_or(0),
            model: entry.name.to_string(),
            prompt: args.prompt.clone(),
            seed: seeds.get(i).copied().unwrap_or(args.seed),
            width: Some(args.width),
            height: Some(args.height),
            steps: Some(args.steps),
            guidance: args.guidance,
            inputs: args.images.clone(),
            created_at: chrono::Local::now().to_rfc3339(),
            duration_ms: elapsed,
            project: args.project.clone(),
            project_name: args.project_name.clone(),
            // One request can produce several pictures, so the index walks
            // from whatever the caller said this batch starts at.
            project_index: args.project_index.map(|n| n + i as u32),
            // Everything needed to run this again and get this picture. All of
            // it was already resolved above and thrown away, which is why
            // "use these settings" restored the prompt and the seed and
            // silently dropped the negative prompt, the adapters and the edit
            // mode -- then produced something else.
            model_repo: Some(entry.repo.clone()),
            quantize: entry.quantize,
            negative_prompt: args
                .negative_prompt
                .clone()
                .filter(|n| !n.trim().is_empty()),
            loras: args.loras.clone(),
            i2i_mode: args.i2i_mode.clone(),
            image_strength: args.image_strength,
            outpaint_padding: args.outpaint_padding.clone(),
            outpaint_fill: args.outpaint_fill.clone(),
            low_ram: args.low_ram,
            engine: Some(engine_versions()),
            transient: false,
        })?;
    }

    if !produced.is_empty() {
        // Learn from the run so the next estimate comes from this machine.
        // The engine separates loading from denoising; only the latter scales
        // with steps and canvas size.
        let generate_ms = result["generate_ms"].as_u64().unwrap_or(elapsed);
        let load_ms = result["load_ms"].as_u64().unwrap_or(0);
        // Keyed by whether the run was conditioned on reference images, which
        // is what `args.images` is. Folding a board panel -- two to three times
        // the cost per step -- into the same average as a plain text-to-image
        // run produced an estimate that described neither.
        let _ = crate::timings::record_for(
            &state.paths(),
            &entry.id,
            !args.images.is_empty(),
            args.steps,
            args.width,
            args.height,
            produced.len(),
            generate_ms,
            load_ms,
        );
    }

    // Anything reserved but not produced is dead weight.
    let unused: Vec<String> = slot_ids
        .into_iter()
        .filter(|i| !produced.contains(i))
        .collect();
    state.vault.discard_slots(&unused);

    Ok(produced)
}

#[tauri::command]
pub async fn generate(
    app: AppHandle,
    state: State<'_, AppState>,
    args: GenerateArgs,
) -> Result<Vec<String>> {
    run_job(&app, &state, args, "generate").await
}

#[tauri::command]
pub async fn edit_image(
    app: AppHandle,
    state: State<'_, AppState>,
    args: GenerateArgs,
) -> Result<Vec<String>> {
    if args.images.is_empty() {
        return Err(AppError::msg("add at least one source image to edit"));
    }
    run_job(&app, &state, args, "edit").await
}

#[tauri::command]
pub async fn upscale(
    app: AppHandle,
    state: State<'_, AppState>,
    job_id: String,
    model_id: String,
    image: String,
    resolution: String,
    low_ram: bool,
) -> Result<Vec<String>> {
    let _job = state.job_gate.read().await;
    state.require_unlocked()?;
    let host = HostInfo::probe(&state.paths());
    let entry = models::find(&state.paths(), &host, &model_id)
        .ok_or_else(|| AppError::msg(format!("unknown model: {model_id}")))?;
    if !entry.installed {
        return Err(AppError::msg(format!(
            "{} is not downloaded yet.",
            entry.name
        )));
    }

    let (slot_id, file_id, key, path) = state.vault.reserve_slot()?;
    let (in_file_id, in_key, in_path) = state.vault.input_key(&image)?;

    let engine = state.engine(&app).await?;
    let started = std::time::Instant::now();
    let result = engine
        .request(
            &job_id,
            "upscale",
            json!({
                "model": entry.repo.clone(),
                "quantize": entry.quantize,
                "family": entry.family.clone(),
                "resolution": resolution,
                "low_ram": low_ram,
                "seed": 0,
                "peak_gib": entry.peak_gib,
                "vault_slots": [{
                    "id": slot_id, "file_id": hex(&file_id),
                    "key": hex(&key), "path": path.to_string_lossy(),
                }],
                "vault_inputs": [{
                    "id": image, "file_id": hex(&in_file_id),
                    "key": hex(&in_key), "path": in_path.to_string_lossy(),
                }],
            }),
        )
        .await;

    let result = match result {
        Ok(r) => r,
        Err(e) => {
            state.vault.discard_slots(&[slot_id]);
            return Err(e);
        }
    };

    let produced: Vec<String> = result["outputs"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let sizes: Vec<u64> = result["sizes"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();

    for (i, id) in produced.iter().enumerate() {
        state.vault.commit_slot(VaultItem {
            id: id.clone(),
            content_hash: None,
            kind: "upscale".into(),
            name: format!(
                "upscale-{}.png",
                chrono::Local::now().format("%Y%m%d-%H%M%S")
            ),
            mime: "image/png".into(),
            bytes: sizes.get(i).copied().unwrap_or(0),
            model: entry.name.to_string(),
            prompt: format!("Upscale {resolution}"),
            seed: 0,
            width: None,
            height: None,
            steps: None,
            guidance: None,
            inputs: vec![image.clone()],
            created_at: chrono::Local::now().to_rfc3339(),
            duration_ms: started.elapsed().as_millis() as u64,
            project: None,
            project_name: None,
            project_index: None,
            ..Default::default()
        })?;
    }
    Ok(produced)
}

/// Rewrite a prompt with the local assistant. For an edit, it is shown the
/// source image so it can name what to change and what to leave alone.
#[tauri::command]
pub async fn assist_prompt(
    app: AppHandle,
    state: State<'_, AppState>,
    job_id: String,
    prompt: String,
    mode: String,
    images: Vec<String>,
) -> Result<serde_json::Value> {
    let _job = state.job_gate.read().await;
    state.require_unlocked()?;

    let host = HostInfo::probe(&state.paths());

    // Two different models, and which one is needed depends on the request.
    // Writing a prompt is a text job; only reading the picture needs the
    // vision model. Requiring both would make a plain text request wait on a
    // download it never uses.
    let writer = models::find(&state.paths(), &host, "qwen3-4b-instruct-4bit")
        .ok_or_else(|| AppError::msg("the prompt writer is missing from the catalog"))?;
    if !writer.installed {
        return Err(AppError::msg(
            "The prompt writer is not installed yet. Add it from the Models tab \
             \u{2014} it is a 2.3 GiB download and runs entirely on this Mac.",
        ));
    }
    // The reader is what looks at the picture; the writer cannot, because
    // Qwen3 has no vision encoder. But a missing reader is no reason to refuse
    // the whole request: the rewrite still works from the words alone, and the
    // engine says so in its reply. Refusing outright was the worse trade --
    // it left people with a writer they could not use and no way to proceed.
    let can_read = !images.is_empty()
        && models::find(&state.paths(), &host, "qwen2-vl-2b-4bit").is_some_and(|r| r.installed);

    // Hand over per-image keys only, exactly as generation does.
    let mut vault_inputs = Vec::new();
    for id in &images {
        let (file_id, key, path) = state.vault.input_key(id)?;
        vault_inputs.push(json!({
            "id": id,
            "file_id": hex(&file_id),
            "key": hex(&key),
            "path": path.to_string_lossy(),
        }));
    }

    let engine = state.engine(&app).await?;
    engine
        .request(
            &job_id,
            "assist",
            json!({
                // The reader, used only when there is a picture and it is
                // installed. Null tells the engine to rewrite from the words.
                "assistant": can_read
                    .then_some("mlx-community/Qwen2-VL-2B-Instruct-4bit"),
                "writer": writer.repo,
                "prompt": prompt,
                "mode": mode,
                "vault_inputs": vault_inputs,
            }),
        )
        .await
}

/// Generate a clip.
///
/// Frame count is the cost lever, not fps: fps is playback metadata written
/// into the file, so the same frames tagged 16 or 24 cost the same to make.
/// Attention is quadratic in sequence length, so frames are also where memory
/// goes non-linear.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn generate_video(
    app: AppHandle,
    state: State<'_, AppState>,
    job_id: String,
    model_id: String,
    prompt: String,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    frames: u32,
    fps: u32,
    steps: u32,
    guidance: Option<f32>,
    seed: i64,
    first_frame: Option<String>,
) -> Result<Vec<String>> {
    let _job = state.job_gate.read().await;
    state.require_unlocked()?;

    let host = HostInfo::probe(&state.paths());
    let entry = models::find(&state.paths(), &host, &model_id)
        .ok_or_else(|| AppError::msg(format!("unknown model: {model_id}")))?;
    if let Some(reason) = entry.broken.as_deref() {
        return Err(AppError::msg(format!(
            "{} cannot run. {}",
            entry.name, reason
        )));
    }
    if !entry.installed {
        return Err(AppError::msg(format!(
            "{} is not downloaded yet. Install it from the Models tab.",
            entry.name
        )));
    }
    if entry.hopeless {
        return Err(AppError::msg(format!(
            "{} needs more memory than this Mac has.",
            entry.name
        )));
    }

    let (slot_id, file_id, key, path) = state.vault.reserve_slot()?;

    let mut vault_inputs = Vec::new();
    if let Some(src) = first_frame.as_ref() {
        let (fid, k, p) = state.vault.input_key(src)?;
        let ext = state
            .vault
            .get_item(src)
            .ok()
            .and_then(|i| ext_for_mime(&i.mime))
            .unwrap_or("png");
        vault_inputs.push(json!({
            "id": src, "file_id": hex(&fid), "key": hex(&k),
            "path": p.to_string_lossy(), "ext": ext,
        }));
    }

    let mut params = json!({
        "model": entry.repo.clone(),
        "quantize": entry.quantize,
        "family": entry.family.clone(),
        "peak_gib": entry.peak_gib,
        "prompt": prompt,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "steps": steps,
        "seed": seed,
        "vault_inputs": vault_inputs,
        "vault_slots": [{
            "id": slot_id, "file_id": hex(&file_id),
            "key": hex(&key), "path": path.to_string_lossy(),
        }],
    });
    if let Some(g) = guidance {
        params["guidance"] = json!(g);
    }
    if let Some(n) = negative_prompt.as_ref().filter(|s| !s.trim().is_empty()) {
        params["negative_prompt"] = json!(n);
    }

    let engine = state.engine(&app).await?;
    let started = std::time::Instant::now();
    let result = match engine.request(&job_id, "video", params).await {
        Ok(r) => r,
        Err(e) => {
            state.vault.discard_slots(&[slot_id]);
            return Err(e);
        }
    };

    let produced: Vec<String> = result["outputs"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let sizes: Vec<u64> = result["sizes"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();

    for (i, id) in produced.iter().enumerate() {
        state.vault.commit_slot(VaultItem {
            id: id.clone(),
            content_hash: None,
            kind: "video".into(),
            name: format!("clip-{}.mp4", chrono::Local::now().format("%Y%m%d-%H%M%S")),
            mime: "video/mp4".into(),
            bytes: sizes.get(i).copied().unwrap_or(0),
            model: entry.name.to_string(),
            prompt: prompt.clone(),
            seed,
            width: Some(width),
            height: Some(height),
            steps: Some(steps),
            guidance,
            inputs: first_frame.clone().into_iter().collect(),
            created_at: chrono::Local::now().to_rfc3339(),
            duration_ms: started.elapsed().as_millis() as u64,
            project: None,
            project_name: None,
            project_index: None,
            ..Default::default()
        })?;
    }
    Ok(produced)
}

/// An adapter the user has installed, remembered between launches.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct Lora {
    /// `owner/repo:file.safetensors` -- the handle MLX-Gen expects.
    pub handle: String,
    pub repo: String,
    pub name: String,
    pub bytes: u64,
    pub scale: f32,
    /// True while its weights are present locally.
    #[serde(default)]
    pub installed: bool,
}

fn loras_path(paths: &AppPaths) -> std::path::PathBuf {
    paths.root.join("loras.json")
}

fn load_loras(paths: &AppPaths) -> Vec<Lora> {
    std::fs::read_to_string(loras_path(paths))
        .ok()
        .and_then(|s| serde_json::from_str::<Vec<Lora>>(&s).ok())
        .unwrap_or_default()
        .into_iter()
        .map(|mut l| {
            l.installed = models::is_installed(paths, &l.repo, 0.0).0;
            l
        })
        .collect()
}

fn save_loras(paths: &AppPaths, list: &[Lora]) -> Result<()> {
    let tmp = loras_path(paths).with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(list)?)?;
    std::fs::rename(&tmp, loras_path(paths))?;
    Ok(())
}

#[tauri::command]
pub fn list_loras(state: State<'_, AppState>) -> Vec<Lora> {
    load_loras(&state.paths())
}

/// Inspect an adapter repository before installing it.
#[tauri::command]
pub async fn resolve_lora(
    app: AppHandle,
    state: State<'_, AppState>,
    repo: String,
) -> Result<serde_json::Value> {
    let _job = state.job_gate.read().await;
    let repo = normalize_repo(&repo)?;
    let engine = state.engine(&app).await?;
    engine
        .request(&new_job_id(), "lora_info", json!({ "model": repo }))
        .await
}

/// Download an adapter and remember it.
///
/// `handle` is `owner/repo:file.safetensors`; the file part matters because a
/// repository may publish several adapters and MLX-Gen needs to know which.
#[tauri::command]
pub async fn add_lora(
    app: AppHandle,
    state: State<'_, AppState>,
    job_id: String,
    repo: String,
    file: String,
    name: String,
    bytes: u64,
) -> Result<Vec<Lora>> {
    let _job = state.job_gate.read().await;
    let repo = normalize_repo(&repo)?;
    let engine = state.engine(&app).await?;
    engine
        .request(
            &job_id,
            "download",
            json!({ "model": repo.clone(), "expected_bytes": bytes, "via": "lora" }),
        )
        .await?;

    let paths = state.paths();
    let mut list = load_loras(&paths);
    let handle = if file.is_empty() {
        repo.clone()
    } else {
        format!("{repo}:{file}")
    };

    // Confirm it holds adapter weights before it is written down.
    //
    // The repository listing cannot tell an adapter from a VAE -- both are one
    // .safetensors file with no pipeline declared -- so a VAE downloaded,
    // appeared as installed, could be selected and given a strength, and then
    // changed nothing. The failure surfaced at draw time as mflux's "did not
    // match any known adapter keys", naming a file but not what was wrong with
    // it. The weights are on disk the moment the download above returns, which
    // is the first point the question can be answered; answering it here means
    // a file that is not an adapter never enters the list.
    engine
        .request(&job_id, "verify_adapter", json!({ "handle": handle }))
        .await?;

    list.retain(|l| l.handle != handle);
    list.push(Lora {
        name: if name.trim().is_empty() {
            handle.clone()
        } else {
            name
        },
        handle,
        repo,
        bytes,
        scale: 1.0,
        installed: true,
    });
    save_loras(&paths, &list)?;
    Ok(load_loras(&paths))
}

#[tauri::command]
pub fn remove_lora(state: State<'_, AppState>, handle: String) -> Result<Vec<Lora>> {
    let _exclusive = state
        .job_gate
        .try_write()
        .map_err(|_| AppError::msg("Wait for the current local job before removing an adapter."))?;
    let paths = state.paths();
    let mut list = load_loras(&paths);
    list.retain(|l| l.handle != handle);
    save_loras(&paths, &list)?;
    Ok(load_loras(&paths))
}

#[cfg(test)]
mod tests {
    use super::{
        app_owned_model_entries, dest_holds_other_files, foreign_model_entries, normalize_repo,
        sweep_orphans_gated, AppPaths, AppState, Vault, MOVE_STAGING_PREFIX,
    };
    use std::sync::Arc;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("modelstudio-{name}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn names_of(paths: &[std::path::PathBuf]) -> Vec<String> {
        let mut names: Vec<String> = paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        names.sort();
        names
    }

    /// A move copies and then deletes; it may only ever touch the entries this
    /// app created. It used to take the whole folder, so a model root the user
    /// had pointed at a folder of their own lost everything in it.
    #[test]
    fn a_move_only_touches_the_apps_own_entries() {
        let root = scratch("models-root");
        std::fs::create_dir_all(root.join("hub/models--owner--model")).unwrap();
        std::fs::create_dir_all(root.join("xet")).unwrap();
        std::fs::write(root.join(".DS_Store"), b"finder").unwrap();
        std::fs::create_dir_all(root.join(format!("{MOVE_STAGING_PREFIX}abandoned"))).unwrap();
        assert!(
            foreign_model_entries(&root).is_empty(),
            "a cache folder should be movable"
        );
        assert_eq!(names_of(&app_owned_model_entries(&root)), ["hub", "xet"]);

        std::fs::write(root.join("tax-return.pdf"), b"mine").unwrap();
        std::fs::create_dir_all(root.join("Holiday photos")).unwrap();
        assert_eq!(
            foreign_model_entries(&root),
            ["Holiday photos", "tax-return.pdf"]
        );
        assert_eq!(names_of(&app_owned_model_entries(&root)), ["hub", "xet"]);
        std::fs::remove_dir_all(root).unwrap();
    }

    /// The destination has to be empty so nothing can be overwritten, but the
    /// two files that mean nothing must not stand in the way.
    #[test]
    fn destination_emptiness_ignores_what_does_not_matter() {
        let dest = scratch("models-dest");
        assert!(!dest_holds_other_files(&dest).unwrap());
        std::fs::write(dest.join(".DS_Store"), b"finder").unwrap();
        std::fs::create_dir_all(dest.join(format!("{MOVE_STAGING_PREFIX}abandoned"))).unwrap();
        assert!(!dest_holds_other_files(&dest).unwrap());
        std::fs::create_dir_all(dest.join("hub")).unwrap();
        assert!(dest_holds_other_files(&dest).unwrap());
        std::fs::remove_dir_all(dest).unwrap();
    }

    /// huggingface_hub writes a blob before it links the snapshot at it, so a
    /// sweep that ran during a download deleted a weight that had just landed.
    /// It took no job gate at all, unlike every other model-store mutation.
    #[test]
    fn sweeping_orphans_waits_for_a_running_job() {
        let root = scratch("sweep-gate");
        let paths = AppPaths::at(root.clone());
        let vault = Arc::new(Vault::new(paths.vault()));
        let state = AppState::new(paths, vault);

        let job = state.job_gate.try_read().unwrap();
        assert!(
            sweep_orphans_gated(&state).is_err(),
            "swept the cache while a job held the gate"
        );
        drop(job);
        assert!(sweep_orphans_gated(&state).is_ok());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn accepts_the_shapes_people_actually_paste() {
        for input in [
            "Qwen/Qwen-Image",
            "  Qwen/Qwen-Image  ",
            "https://huggingface.co/Qwen/Qwen-Image",
            "http://huggingface.co/Qwen/Qwen-Image",
            "https://huggingface.co/Qwen/Qwen-Image/tree/main",
            "https://huggingface.co/Qwen/Qwen-Image/blob/main/config.json",
            "https://huggingface.co/Qwen/Qwen-Image?library=diffusers",
            "https://huggingface.co/Qwen/Qwen-Image#usage",
            "huggingface.co/Qwen/Qwen-Image",
            "hf.co/Qwen/Qwen-Image/",
            "https://huggingface.co/models/Qwen/Qwen-Image",
        ] {
            assert_eq!(
                normalize_repo(input).unwrap(),
                "Qwen/Qwen-Image",
                "input: {input}"
            );
        }
    }

    #[test]
    fn rejects_what_is_not_a_repo() {
        for input in ["", "   ", "Qwen", "https://example.com"] {
            assert!(normalize_repo(input).is_err(), "should reject: {input}");
        }
    }
}

/// Work each panel up into a scene that can be drawn.
///
/// Separate from dividing the story on purpose. The division is terse because
/// terse is what you need to judge whether the story was cut correctly; this
/// is the stage that fills in surface, background and light, and it is worth
/// reading before anything is drawn.
#[tauri::command]
pub async fn enrich_panels(
    app: AppHandle,
    state: State<'_, AppState>,
    job_id: String,
    panels: serde_json::Value,
    style: String,
    // What the story says about each place, from the cast read. The place
    // writer invents a room from the panel actions without it, so a kitchen
    // the story calls green comes back yellow. Optional so a caller that has
    // not read the cast still works.
    places: Option<serde_json::Value>,
) -> Result<serde_json::Value> {
    let _job = state.job_gate.read().await;
    state.require_unlocked()?;

    let host = HostInfo::probe(&state.paths());
    let writer = models::find(&state.paths(), &host, "qwen3-4b-instruct-4bit")
        .ok_or_else(|| AppError::msg("the prompt writer is missing from the catalog"))?;
    if !writer.installed {
        return Err(AppError::msg(
            "Working up the panels needs the prompt writer. Add it from the \
             Models tab \u{2014} it is a 2.1 GiB download and runs entirely on \
             this Mac.",
        ));
    }

    let engine = state.engine(&app).await?;
    engine
        .request(
            &job_id,
            "enrich_panels",
            json!({
                "panels": panels,
                "style": style,
                "places": places.unwrap_or_else(|| json!([])),
                "writer": writer.repo,
            }),
        )
        .await
}

/// Stack the drawn panels into one page, captions underneath.
///
/// Composed in the engine rather than the browser because the panels are
/// sealed: that is the process holding the keys, and the finished page is
/// sealed again before it reaches disk.
/// One board, ready to be set as pages.
///
/// A struct rather than eight positional arguments: five of them are parallel
/// lists indexed by panel, and swapping two would be silent.
#[derive(serde::Deserialize)]
pub struct BoardPages {
    /// Vault ids of the drawn panels, in reading order.
    pub panels: Vec<String>,
    pub captions: Vec<String>,
    pub shots: Vec<String>,
    pub dialogue: serde_json::Value,
    /// Which scene each panel belongs to. Pages break where this changes.
    pub scenes: Vec<u32>,
    /// "page" for tiers, "strip" for a vertical scroll.
    pub layout: String,
    pub project: Option<String>,
    pub project_name: Option<String>,
}

/// Bind a board's finished pages into one PDF.
///
/// Sealed into the vault like everything else, then exported through
/// `vault_export` -- the one sanctioned path for plaintext to leave. A comic
/// that leaves as a folder of PNGs keeps its reading order in the file names,
/// and the first person to sort them differently loses it.
#[tauri::command]
pub async fn bind_comic(
    app: AppHandle,
    state: State<'_, AppState>,
    job_id: String,
    pages: Vec<String>,
    title: String,
    project: Option<String>,
    project_name: Option<String>,
) -> Result<String> {
    let _job = state.job_gate.read().await;
    state.require_unlocked()?;
    if pages.is_empty() {
        return Err(AppError::msg("compose the pages before binding them"));
    }

    let mut vault_inputs = Vec::new();
    for id in &pages {
        let (fid, k, p) = state.vault.input_key(id)?;
        vault_inputs.push(json!({
            "id": id, "file_id": hex(&fid), "key": hex(&k),
            "path": p.to_string_lossy(), "ext": "png",
        }));
    }

    let (slot_id, file_id, key, path) = state.vault.reserve_slot()?;
    let started = std::time::Instant::now();
    let engine = state.engine(&app).await?;
    let result = match engine
        .request(
            &job_id,
            "bind_comic",
            json!({
                "vault_inputs": vault_inputs,
                "title": title,
                "vault_slots": [json!({
                    "id": slot_id, "file_id": hex(&file_id),
                    "key": hex(&key), "path": path.to_string_lossy(),
                })],
            }),
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            state.vault.discard_slots(std::slice::from_ref(&slot_id));
            return Err(e);
        }
    };

    let bytes = result["sizes"][0].as_u64().unwrap_or(0);
    if bytes == 0 {
        state.vault.discard_slots(std::slice::from_ref(&slot_id));
        return Err(AppError::msg("the engine bound an empty comic"));
    }
    // A name that is about to be a file name. The separators and the Windows
    // reserved set become hyphens rather than being dropped, so two comics
    // whose titles differ only there do not collide.
    let safe: String = title
        .chars()
        .map(|c| match c {
            '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '-',
            other => other,
        })
        .collect();
    state.vault.commit_slot(VaultItem {
        id: slot_id.clone(),
        content_hash: None,
        kind: "document".into(),
        name: format!("{}.pdf", safe.trim()),
        mime: "application/pdf".into(),
        bytes,
        model: "composed".into(),
        prompt: format!("{} pages", pages.len()),
        seed: 0,
        width: None,
        height: None,
        steps: None,
        guidance: None,
        inputs: pages.clone(),
        created_at: chrono::Local::now().to_rfc3339(),
        duration_ms: started.elapsed().as_millis() as u64,
        project,
        project_name,
        // Last, so it sorts after every page it was made from.
        project_index: Some(u32::MAX),
        ..Default::default()
    })?;
    Ok(slot_id)
}

#[tauri::command]
pub async fn compose_board(
    app: AppHandle,
    state: State<'_, AppState>,
    job_id: String,
    board: BoardPages,
) -> Result<Vec<String>> {
    let _job = state.job_gate.read().await;
    let BoardPages {
        panels,
        captions,
        shots,
        dialogue,
        scenes,
        layout,
        project,
        project_name,
    } = board;
    state.require_unlocked()?;
    if panels.is_empty() {
        return Err(AppError::msg(
            "draw the panels before making a page of them",
        ));
    }

    let mut vault_inputs = Vec::new();
    for id in &panels {
        let (fid, k, p) = state.vault.input_key(id)?;
        let ext = state
            .vault
            .get_item(id)
            .ok()
            .and_then(|i| ext_for_mime(&i.mime))
            .unwrap_or("png");
        vault_inputs.push(json!({
            "id": id, "file_id": hex(&fid), "key": hex(&k),
            "path": p.to_string_lossy(), "ext": ext,
        }));
    }

    // A board can run to several pages and the engine decides how many, so
    // reserve the most it could need and hand back whatever goes unused.
    //
    // One slot per panel. The bound used to be `panels.len() / 3`, on the
    // reasoning that three panels is the smallest readable page -- but the
    // engine paginates on scene boundaries, not on a fixed page size, and a
    // short scene makes a short page. Scenes [1, 2, 2] make two pages from
    // three panels, and the board was refused with "this board makes 2 pages
    // but only 1 was reserved": a number the user cannot influence, on the
    // last step of a job that had already cost them half an hour. Pages
    // partition the panels and every page holds at least one, so the panel
    // count is the only bound that cannot be wrong.
    let max_pages = if layout == "page" {
        panels.len().max(1)
    } else {
        1
    };
    let mut reserved = Vec::new();
    let mut slot_json = Vec::new();
    for _ in 0..max_pages {
        let (id, file_id, key, path) = state.vault.reserve_slot()?;
        slot_json.push(json!({
            "id": id, "file_id": hex(&file_id),
            "key": hex(&key), "path": path.to_string_lossy(),
        }));
        reserved.push(id);
    }

    let started = std::time::Instant::now();
    let engine = state.engine(&app).await?;
    let result = match engine
        .request(
            &job_id,
            "compose_board",
            json!({
                "vault_inputs": vault_inputs,
                "captions": captions,
                "shots": shots,
                "dialogue": dialogue,
                "scenes": scenes,
                "layout": layout,
                "vault_slots": slot_json,
            }),
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            state.vault.discard_slots(&reserved);
            return Err(e);
        }
    };

    let produced: Vec<String> = result["outputs"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    if produced.is_empty() {
        state.vault.discard_slots(&reserved);
        return Err(AppError::msg("the engine composed no pages"));
    }

    let sizes: Vec<u64> = result["sizes"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();
    let total = produced.len();
    for (i, id) in produced.iter().enumerate() {
        let dims = &result["dimensions"][i];
        state.vault.commit_slot(VaultItem {
            id: id.clone(),
            content_hash: None,
            kind: "image".into(),
            name: format!(
                "page-{}-of-{}-{}.png",
                i + 1,
                total,
                chrono::Local::now().format("%Y%m%d-%H%M%S")
            ),
            mime: "image/png".into(),
            bytes: sizes.get(i).copied().unwrap_or(0),
            model: "composed".into(),
            prompt: format!("page {} of {}", i + 1, total),
            seed: 0,
            width: dims[0].as_u64().map(|v| v as u32),
            height: dims[1].as_u64().map(|v| v as u32),
            steps: None,
            guidance: None,
            inputs: panels.clone(),
            created_at: chrono::Local::now().to_rfc3339(),
            duration_ms: started.elapsed().as_millis() as u64,
            project: project.clone(),
            project_name: project_name.clone(),
            project_index: Some(i as u32),
            ..Default::default()
        })?;
    }

    // Hand back the pages this board turned out not to need.
    let unused: Vec<String> = reserved
        .into_iter()
        .filter(|id| !produced.contains(id))
        .collect();
    if !unused.is_empty() {
        state.vault.discard_slots(&unused);
    }
    Ok(produced)
}

/// Break a story into an ordered list of panels.
///
/// The first step of a picture board, and the one everything after it depends
/// on: a bad division of the story cannot be rescued by good panels. It runs
/// on the writer, because the smaller vision model could not hold the panel
/// count or cover the whole story.
#[tauri::command]
pub async fn shot_list(
    app: AppHandle,
    state: State<'_, AppState>,
    job_id: String,
    story: String,
    // `panels: None` lets the engine divide by beat and report what the story
    // needed. A number picked before anyone read the story is a guess, and a
    // slider capped at twelve is why a whole chapter came back as twelve
    // panels regardless of what was in it.
    panels: Option<u32>,
) -> Result<serde_json::Value> {
    let _job = state.job_gate.read().await;
    state.require_unlocked()?;

    let host = HostInfo::probe(&state.paths());
    let writer = models::find(&state.paths(), &host, "qwen3-4b-instruct-4bit")
        .ok_or_else(|| AppError::msg("the prompt writer is missing from the catalog"))?;
    if !writer.installed {
        return Err(AppError::msg(
            "Breaking a story into panels needs the prompt writer. Add it from \
             the Models tab \u{2014} it is a 2.1 GiB download and runs entirely \
             on this Mac.",
        ));
    }

    let engine = state.engine(&app).await?;
    engine
        .request(
            &job_id,
            "shotlist",
            json!({ "story": story, "panels": panels, "writer": writer.repo }),
        )
        .await
}

/// Read a story and report who is in it and where it happens.
///
/// Runs on its own as soon as a story is pasted: it costs a little of the
/// text model and no picture time at all, and everything downstream — which
/// character sheet a panel is drawn against, which room it is set in — is only
/// as good as this. Every name it returns has been checked against the story.
#[tauri::command]
pub async fn story_cast(
    app: AppHandle,
    state: State<'_, AppState>,
    job_id: String,
    story: String,
) -> Result<serde_json::Value> {
    let _job = state.job_gate.read().await;
    state.require_unlocked()?;

    let host = HostInfo::probe(&state.paths());
    let writer = models::find(&state.paths(), &host, "qwen3-4b-instruct-4bit")
        .ok_or_else(|| AppError::msg("the prompt writer is missing from the catalog"))?;
    if !writer.installed {
        return Err(AppError::msg(
            "Reading a story needs the prompt writer. Add it from the Models \
             tab \u{2014} it is a 2.1 GiB download and runs entirely on this Mac.",
        ));
    }

    let engine = state.engine(&app).await?;
    engine
        .request(
            &job_id,
            "cast",
            json!({ "story": story, "writer": writer.repo }),
        )
        .await
}

/// Whether a Hugging Face token is stored, without ever handing it back out.
#[tauri::command]
pub fn hf_token_status(state: State<'_, AppState>) -> Result<serde_json::Value> {
    let token = models::hf_token(&state.paths());
    Ok(json!({
        "present": token.is_some(),
        // Enough to recognise which token is stored, not enough to use it.
        "hint": token.map(|t| {
            let first: String = t.chars().take(6).collect();
            let mut last: String = t.chars().rev().take(4).collect();
            last = last.chars().rev().collect();
            format!("{first}...{last}")
        }),
    }))
}

/// Store or clear the Hugging Face access token.
///
/// The engine reads the token from its environment at spawn, so it is stopped
/// afterwards; the next request starts a fresh one that can see the new value.
#[tauri::command]
pub async fn set_hf_token(state: State<'_, AppState>, token: String) -> Result<()> {
    let _exclusive = state.job_gate.try_write().map_err(|_| {
        AppError::msg("Wait for the current local job before changing the access token.")
    })?;
    models::set_hf_token(&state.paths(), &token)?;
    state.stop_engine().await;
    Ok(())
}

#[cfg(test)]
mod board_key_tests {
    use super::board_key;

    #[test]
    fn the_live_draft_is_the_default() {
        assert_eq!(board_key(None).unwrap(), "board-draft");
    }

    #[test]
    fn a_board_is_addressed_by_its_project_id() {
        let id = "board-0f2a9c11-4b6e-4a0d-9f3e-2c7b8d5e1a44";
        assert_eq!(board_key(Some(id.into())).unwrap(), id);
    }

    #[test]
    fn nothing_outside_the_board_namespace_is_reachable() {
        // The key names a slot in the encrypted index, so a caller must not be
        // able to read or overwrite anything else that lives there.
        for bad in [
            "vault-dek",
            "../board-draft",
            "board-draft/../secrets",
            "board-with spaces",
            "board-with_underscore",
        ] {
            assert!(board_key(Some(bad.into())).is_err(), "{bad} was accepted");
        }
    }

    #[test]
    fn an_over_long_key_is_refused() {
        assert!(board_key(Some(format!("board-{}", "a".repeat(60)))).is_err());
    }
}
