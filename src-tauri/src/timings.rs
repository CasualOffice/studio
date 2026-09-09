//! Measured generation speed, so estimates come from this machine rather than
//! a constant baked in on someone else's.
//!
//! Cost is modelled as milliseconds per denoise step per megapixel, which is
//! close enough to linear in both to be useful and needs only one number per
//! model. The first estimate for an unmeasured model falls back to a figure
//! scaled from resident size; every real run pulls it toward the truth.

use crate::error::Result;
use crate::paths::AppPaths;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Sample {
    /// Exponentially weighted mean of ms per step per megapixel, measured over
    /// denoising only.
    pub ms_per_step_mpx: f32,
    pub runs: u32,
    /// Slowest observed value, so a warning can be honest about the tail.
    pub worst_ms_per_step_mpx: f32,
    /// Time to bring the weights into memory. A fixed cost paid once per
    /// session, not something that scales with steps or canvas size — keeping
    /// it out of the rate above is the whole point of tracking it separately.
    #[serde(default)]
    pub load_ms: f32,
    #[serde(default)]
    pub load_samples: u32,
}

type Table = HashMap<String, Sample>;

fn load(paths: &AppPaths) -> Table {
    std::fs::read_to_string(paths.timings())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save(paths: &AppPaths, table: &Table) -> Result<()> {
    let tmp = paths.timings().with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(table)?)?;
    std::fs::rename(&tmp, paths.timings())?;
    Ok(())
}

pub fn get(paths: &AppPaths, model_id: &str) -> Option<Sample> {
    load(paths).get(model_id).cloned()
}

/// Fold one completed run into the model's running average.
///
/// `elapsed_ms` should exclude nothing: the wait a user actually experiences
/// includes loading the model, so a first run legitimately reads slower than a
/// warm one. Weighting recent runs more heavily lets the number settle on the
/// warm case, which is what most runs will be.
#[allow(clippy::too_many_arguments)] // one call site; a struct would only add indirection
pub fn record(
    paths: &AppPaths,
    model_id: &str,
    steps: u32,
    width: u32,
    height: u32,
    images: usize,
    generate_ms: u64,
    load_ms: u64,
) -> Result<()> {
    let mut table = load(paths);
    let entry = table.entry(model_id.to_string()).or_default();

    // A load only happened if the model was not already resident.
    if load_ms > 0 {
        entry.load_samples += 1;
        entry.load_ms = if entry.load_samples == 1 {
            load_ms as f32
        } else {
            entry.load_ms * 0.67 + load_ms as f32 * 0.33
        };
    }

    let mpx = (width as f32 * height as f32) / 1_000_000.0;
    let denom = (steps.max(1) as f32) * mpx.max(0.01) * (images.max(1) as f32);
    if denom > 0.0 && generate_ms > 0 {
        let observed = generate_ms as f32 / denom;
        entry.runs += 1;
        entry.worst_ms_per_step_mpx = entry.worst_ms_per_step_mpx.max(observed);
        entry.ms_per_step_mpx = if entry.runs == 1 {
            observed
        } else {
            // Weight the newest run at 1/3: settles quickly without chasing noise.
            entry.ms_per_step_mpx * 0.67 + observed * 0.33
        };
    }
    save(paths, &table)
}
