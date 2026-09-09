use crate::paths::AppPaths;
use serde::Serialize;
use sysinfo::{Disks, System};

/// Unified memory macOS itself and the app shell need before any model loads.
/// Empirically the window server, our WebView and the Python interpreter sit
/// around this on a quiet machine; being conservative here is the point.
pub const RESERVED_GIB: f32 = 3.5;

/// Never let a download take the volume below this.
const DISK_HEADROOM_GIB: f32 = 6.0;

#[derive(Serialize, Clone, Debug)]
pub struct HostInfo {
    pub chip: String,
    pub arch: String,
    pub total_ram_gib: f32,
    /// What is realistically available to a model.
    pub usable_ram_gib: f32,
    pub free_disk_gib: f32,
    pub total_disk_gib: f32,
    pub disk_headroom_gib: f32,
    pub apple_silicon: bool,
}

impl HostInfo {
    pub fn probe(paths: &AppPaths) -> Self {
        let mut sys = System::new();
        sys.refresh_memory();
        let total_ram_gib = sys.total_memory() as f32 / 1024.0 / 1024.0 / 1024.0;

        let disks = Disks::new_with_refreshed_list();
        // Pick the disk whose mount point is the longest prefix of our root:
        // that is the volume a download will actually land on.
        let root = paths.root.to_string_lossy().to_string();
        let mut best: Option<(usize, u64, u64)> = None;
        for d in disks.list() {
            let mp = d.mount_point().to_string_lossy().to_string();
            if root.starts_with(&mp) {
                let len = mp.len();
                if best.map_or(true, |(b, _, _)| len > b) {
                    best = Some((len, d.available_space(), d.total_space()));
                }
            }
        }
        let (_, avail, total) = best.unwrap_or((0, 0, 0));

        let arch = std::env::consts::ARCH.to_string();
        Self {
            chip: cpu_brand(),
            apple_silicon: arch == "aarch64",
            arch,
            total_ram_gib,
            usable_ram_gib: (total_ram_gib - RESERVED_GIB).max(0.0),
            free_disk_gib: avail as f32 / 1024.0 / 1024.0 / 1024.0,
            total_disk_gib: total as f32 / 1024.0 / 1024.0 / 1024.0,
            disk_headroom_gib: DISK_HEADROOM_GIB,
        }
    }
}

fn cpu_brand() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Unknown CPU".into())
}
