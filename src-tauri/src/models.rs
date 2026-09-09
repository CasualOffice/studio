use crate::catalog::{self, ModelEntry, Task};
use crate::error::Result;
use crate::hostinfo::HostInfo;
use crate::paths::AppPaths;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Fit {
    /// Comfortably within memory and disk.
    Good,
    /// Runs, but leaves little headroom -- expect swap pressure.
    Tight,
    /// Peak memory exceeds what this machine can give it.
    TooMuchMemory,
    /// Would not fit in free disk space.
    TooMuchDisk,
    /// Cannot run for reasons unrelated to this machine, e.g. an upstream bug.
    Broken,
}

/// A model the user added by Hugging Face repo id.
///
/// Sizes come from the repo listing and memory is an estimate, so these are
/// always marked as estimated in the UI -- unlike the curated catalog, where
/// the figures are published benchmarks.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CustomModel {
    pub id: String,
    pub repo: String,
    pub name: String,
    pub tasks: Vec<String>,
    pub quantize: Option<u8>,
    pub package_gib: f32,
    pub steps_default: u32,
    pub added_at: String,
}

/// Flattened view of either a catalog entry or a user-added model.
#[derive(Serialize, Clone, Debug)]
pub struct ModelStatus {
    pub id: String,
    pub repo: String,
    pub name: String,
    pub family: Option<String>,
    pub tasks: Vec<Task>,
    pub tasks_str: Vec<String>,
    pub quantize: Option<u8>,
    pub package_gib: f32,
    pub peak_gib: f32,
    pub peak_estimated: bool,
    pub steps_default: u32,
    pub max_edit_images: u32,
    pub guidance_default: f32,
    pub guidance_max: f32,
    pub notes: String,
    pub broken: Option<String>,
    pub custom: bool,
    pub installed: bool,
    pub installed_bytes: u64,
    pub fit: Fit,
    pub fit_reason: String,
    pub low_ram_may_help: bool,
    pub hopeless: bool,
    /// Measured ms per denoise step per megapixel on this machine, once there
    /// is at least one completed run. `None` means estimates are still guesses.
    pub measured_ms_per_step_mpx: Option<f32>,
    pub measured_runs: u32,
    /// Measured milliseconds to bring the weights into memory.
    pub measured_load_ms: f32,
}

fn snapshot_dir(paths: &AppPaths, repo: &str) -> PathBuf {
    paths.hf_hub().join(format!("models--{}", repo.replace('/', "--")))
}

/// Bytes actually occupied, counting each blob once.
///
/// The Hugging Face cache keeps real content in `blobs/` and symlinks it into
/// `snapshots/`. `metadata()` follows symlinks, so counting those too reports
/// exactly double; `symlink_metadata` sees the link itself and we skip it.
fn dir_size(p: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(rd) = std::fs::read_dir(p) else { return 0 };
    for e in rd.flatten() {
        let Ok(md) = std::fs::symlink_metadata(e.path()) else { continue };
        if md.is_symlink() {
            continue;
        }
        if md.is_dir() {
            total += dir_size(&e.path());
        } else if md.is_file() {
            total += md.len();
        }
    }
    total
}

/// A snapshot counts as installed only if it holds real weight files, so a
/// half-finished or cancelled download does not masquerade as complete.
/// Public wrapper so callers outside this module can size a directory without
/// duplicating the symlink handling.
pub fn dir_size_of(p: &Path) -> u64 {
    dir_size(p)
}

/// Copy a directory tree, preserving symlinks as symlinks.
///
/// The Hugging Face cache links snapshots at blobs; following those links
/// would duplicate every weight file and double the space used.
pub fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let src = entry.path();
        let dst = to.join(entry.file_name());
        let md = std::fs::symlink_metadata(&src)?;
        if md.is_symlink() {
            let target = std::fs::read_link(&src)?;
            let _ = std::fs::remove_file(&dst);
            std::os::unix::fs::symlink(target, &dst)?;
        } else if md.is_dir() {
            copy_tree(&src, &dst)?;
        } else {
            std::fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}

pub fn is_installed(paths: &AppPaths, repo: &str) -> (bool, u64) {
    let dir = snapshot_dir(paths, repo);
    if !dir.exists() {
        return (false, 0);
    }
    let bytes = dir_size(&dir.join("blobs"));
    let complete = dir.join("snapshots").exists() && bytes > 64 * 1024 * 1024;
    (complete, bytes)
}

/// Resident weight floor. Published package size is a close proxy: quantized
/// weights land in memory at roughly their on-disk size.
pub fn weight_floor_gib(package_gib: f32) -> f32 {
    package_gib
}

/// Low-RAM mode caps the MLX buffer cache and releases the text encoder. It
/// bounds transients, never the resident weights -- so it can only close a gap
/// that sits *above* the weight floor.
fn low_ram_helps(package_gib: f32, peak_gib: f32, host: &HostInfo) -> bool {
    peak_gib > host.usable_ram_gib && weight_floor_gib(package_gib) < host.usable_ram_gib * 0.92
}

fn is_hopeless(package_gib: f32, host: &HostInfo) -> bool {
    weight_floor_gib(package_gib) >= host.usable_ram_gib
}

/// The smallest common Apple Silicon memory configuration that would run a
/// model needing `peak_gib`, leaving the same headroom this app reserves here.
pub fn required_ram_gib(peak_gib: f32) -> Option<u32> {
    // Configurations Apple actually ships.
    const TIERS: [u32; 8] = [8, 16, 24, 32, 48, 64, 96, 128];
    TIERS
        .iter()
        .copied()
        .find(|t| (*t as f32 - crate::hostinfo::RESERVED_GIB) >= peak_gib)
}

fn describe_elsewhere(peak_gib: f32) -> String {
    match required_ram_gib(peak_gib) {
        Some(t) => format!(" A Mac with {t} GB of unified memory would run it."),
        None => " No current Mac configuration has enough memory for it.".to_string(),
    }
}

fn classify(
    package_gib: f32,
    peak_gib: f32,
    broken: Option<&str>,
    host: &HostInfo,
    installed: bool,
) -> (Fit, String) {
    // An upstream defect outranks any hardware verdict: saying "fits" about a
    // model that cannot run would be worse than useless.
    if let Some(reason) = broken {
        return (Fit::Broken, reason.to_string());
    }

    let have_disk = host.free_disk_gib - host.disk_headroom_gib;

    if peak_gib > host.usable_ram_gib {
        let reason = if is_hopeless(package_gib, host) {
            format!(
                "Weights alone are ~{:.1} GiB against ~{:.1} GiB usable. Reduced-memory mode \
                 cannot help: it trims transients, not resident weights.{}",
                weight_floor_gib(package_gib),
                host.usable_ram_gib,
                describe_elsewhere(peak_gib)
            )
        } else {
            format!(
                "Peak ~{peak_gib:.1} GiB exceeds ~{:.1} GiB usable, but the ~{:.1} GiB weight \
                 floor fits. Reduced-memory mode may close the gap.{}",
                host.usable_ram_gib,
                weight_floor_gib(package_gib),
                describe_elsewhere(peak_gib)
            )
        };
        return (Fit::TooMuchMemory, reason);
    }
    if !installed && package_gib > have_disk {
        return (
            Fit::TooMuchDisk,
            format!(
                "Package is {package_gib:.1} GiB; only {have_disk:.1} GiB is free after headroom."
            ),
        );
    }
    if peak_gib > host.usable_ram_gib - 2.0 {
        return (
            Fit::Tight,
            format!(
                "Peak ~{peak_gib:.1} GiB against ~{:.1} GiB usable. Close other apps first.",
                host.usable_ram_gib
            ),
        );
    }
    (
        Fit::Good,
        format!("Peak ~{peak_gib:.1} GiB, package {package_gib:.1} GiB."),
    )
}

fn task_name(t: Task) -> &'static str {
    match t {
        Task::TextToImage => "Generate",
        Task::Edit => "Edit",
        Task::Upscale => "Upscale",
        Task::Video => "Video",
        Task::Assist => "Prompt help",
    }
}

fn parse_task(s: &str) -> Option<Task> {
    match s {
        "text_to_image" => Some(Task::TextToImage),
        "edit" => Some(Task::Edit),
        "upscale" => Some(Task::Upscale),
        "video" => Some(Task::Video),
        "assist" => Some(Task::Assist),
        _ => None,
    }
}

/// Peak memory for a model with no published benchmark.
///
/// Exposed so the UI can judge a repo the user pasted *before* spending disk
/// on it.
pub fn estimate_peak_for(package_gib: f32) -> f32 {
    estimate_peak(package_gib)
}

/// Verdict for a model that is not installed and not in the catalog.
pub fn judge_uninstalled(package_gib: f32, host: &HostInfo) -> (Fit, String) {
    classify(package_gib, estimate_peak(package_gib), None, host, false)
}


///
/// Quantized weights sit in memory at roughly package size; activations, the
/// text encoder and the MLX cache add on top. The multiplier is deliberately
/// pessimistic -- an over-estimate costs the user a warning, an under-estimate
/// costs them a hung machine.
fn estimate_peak(package_gib: f32) -> f32 {
    package_gib * 1.25 + 1.0
}

fn measured(paths: &AppPaths, id: &str) -> (Option<f32>, u32, f32) {
    match crate::timings::get(paths, id) {
        Some(s) if s.runs > 0 => (Some(s.ms_per_step_mpx), s.runs, s.load_ms),
        _ => (None, 0, 0.0),
    }
}

fn status_from_entry(entry: &ModelEntry, paths: &AppPaths, host: &HostInfo) -> ModelStatus {
    let (installed, installed_bytes) = is_installed(paths, entry.repo);
    let (fit, fit_reason) = classify(
        entry.package_gib,
        entry.peak_gib,
        entry.broken,
        host,
        installed,
    );
    ModelStatus {
        id: entry.id.into(),
        repo: entry.repo.into(),
        name: entry.name.into(),
        family: entry.family.map(|f| f.to_string()),
        tasks: entry.tasks.to_vec(),
        tasks_str: entry.tasks.iter().map(|t| task_name(*t).to_string()).collect(),
        quantize: entry.quantize,
        package_gib: entry.package_gib,
        peak_gib: entry.peak_gib,
        peak_estimated: entry.peak_estimated,
        steps_default: entry.steps_default,
        max_edit_images: entry.max_edit_images,
        guidance_default: entry.guidance_default,
        guidance_max: entry.guidance_max,
        notes: entry.notes.into(),
        broken: entry.broken.map(|b| b.to_string()),
        custom: false,
        installed,
        installed_bytes,
        low_ram_may_help: entry.broken.is_none()
            && low_ram_helps(entry.package_gib, entry.peak_gib, host),
        hopeless: is_hopeless(entry.package_gib, host),
        measured_ms_per_step_mpx: measured(paths, entry.id).0,
        measured_runs: measured(paths, entry.id).1,
        measured_load_ms: measured(paths, entry.id).2,
        fit,
        fit_reason,
    }
}

fn status_from_custom(m: &CustomModel, paths: &AppPaths, host: &HostInfo) -> ModelStatus {
    let (installed, installed_bytes) = is_installed(paths, &m.repo);
    // Once installed, on-disk size beats the pre-download estimate.
    let package_gib = if installed && installed_bytes > 0 {
        installed_bytes as f32 / 1024.0 / 1024.0 / 1024.0
    } else {
        m.package_gib
    };
    let peak_gib = estimate_peak(package_gib);
    let (fit, fit_reason) = classify(package_gib, peak_gib, None, host, installed);
    let tasks: Vec<Task> = m.tasks.iter().filter_map(|t| parse_task(t)).collect();

    ModelStatus {
        id: m.id.clone(),
        repo: m.repo.clone(),
        name: m.name.clone(),
        family: None,
        tasks_str: tasks.iter().map(|t| task_name(*t).to_string()).collect(),
        tasks,
        quantize: m.quantize,
        package_gib,
        peak_gib,
        peak_estimated: true,
        steps_default: m.steps_default.max(2),
        // Unknown route: assume single-reference, the common case.
        max_edit_images: 1,
        // Unknown route: assume no classifier-free guidance, which every route
        // accepts. Offering a value a distilled checkpoint rejects would fail
        // the run only after the weights were already resident.
        guidance_default: 1.0,
        guidance_max: 1.0,
        notes: format!("Added by you. Memory is estimated, not a published benchmark."),
        broken: None,
        custom: true,
        installed,
        installed_bytes,
        low_ram_may_help: low_ram_helps(package_gib, peak_gib, host),
        hopeless: is_hopeless(package_gib, host),
        measured_ms_per_step_mpx: measured(paths, &m.id).0,
        measured_runs: measured(paths, &m.id).1,
        measured_load_ms: measured(paths, &m.id).2,
        fit,
        fit_reason,
    }
}

// ---------------------------------------------------------------------------
// Custom model store
// ---------------------------------------------------------------------------

pub fn load_custom(paths: &AppPaths) -> Vec<CustomModel> {
    std::fs::read_to_string(paths.custom_models())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_custom(paths: &AppPaths, list: &[CustomModel]) -> Result<()> {
    let tmp = paths.custom_models().with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(list)?)?;
    std::fs::rename(&tmp, paths.custom_models())?;
    Ok(())
}

pub fn add_custom(paths: &AppPaths, m: CustomModel) -> Result<()> {
    let mut list = load_custom(paths);
    list.retain(|x| x.repo != m.repo);
    list.push(m);
    save_custom(paths, &list)
}

pub fn remove_custom(paths: &AppPaths, id: &str) -> Result<()> {
    let mut list = load_custom(paths);
    list.retain(|x| x.id != id);
    save_custom(paths, &list)
}

/// Look up a model by id across both the catalog and user-added entries.
pub fn find(paths: &AppPaths, host: &HostInfo, id: &str) -> Option<ModelStatus> {
    if let Some(e) = catalog::find(id) {
        return Some(status_from_entry(e, paths, host));
    }
    load_custom(paths)
        .iter()
        .find(|m| m.id == id)
        .map(|m| status_from_custom(m, paths, host))
}

pub fn list(paths: &AppPaths, host: &HostInfo) -> Result<Vec<ModelStatus>> {
    let mut out: Vec<ModelStatus> = catalog::CATALOG
        .iter()
        .map(|e| status_from_entry(e, paths, host))
        .collect();
    out.extend(load_custom(paths).iter().map(|m| status_from_custom(m, paths, host)));

    // Installed first, then by ascending memory cost: the order a user on a
    // constrained machine actually wants to read.
    out.sort_by(|a, b| {
        b.installed
            .cmp(&a.installed)
            .then(a.peak_gib.partial_cmp(&b.peak_gib).unwrap_or(std::cmp::Ordering::Equal))
    });
    Ok(out)
}

pub fn delete(paths: &AppPaths, repo: &str) -> Result<u64> {
    let dir = snapshot_dir(paths, repo);
    let bytes = dir_size(&dir.join("blobs"));
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    Ok(bytes)
}
