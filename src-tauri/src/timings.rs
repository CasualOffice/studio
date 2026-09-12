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

/// Key for one model in one shape of run.
///
/// Reference-conditioned work -- a board panel drawn from earlier panels and
/// character sheets -- costs two to three times as much per step as plain
/// text-to-image work on the same model. Both used to fold into one key per
/// model, so the average belonged to neither: Generate quoted "about 19
/// minutes" for a run that takes a few, and presented it as measured, because
/// most of the samples behind it came from panels. Measure the shapes apart.
///
/// The marker also retires the mixed entries written before the split. They
/// are left in the file rather than rewritten -- a few dead keys cost nothing,
/// and editing a user's measurements to drop them is the worse trade -- but
/// nothing reads them again, because falling back to the size-scaled guess is
/// honest where quoting an average of two different things is not. The
/// separator cannot occur in a model id, which is either a catalog slug or a
/// Hugging Face repo name.
fn rate_key(model_id: &str, has_reference: bool) -> String {
    let shape = if has_reference { "ref" } else { "plain" };
    format!("{model_id}\u{1}{shape}")
}

/// Measured cost of plain text-to-image work, which is what an estimate shown
/// beside a model means. Reference-conditioned runs are asked for by name.
pub fn get(paths: &AppPaths, model_id: &str) -> Option<Sample> {
    get_for(paths, model_id, false)
}

pub fn get_for(paths: &AppPaths, model_id: &str, has_reference: bool) -> Option<Sample> {
    load(paths).get(&rate_key(model_id, has_reference)).cloned()
}

/// Fold one completed run into the running average for this model in this
/// shape of run -- see `rate_key` for why the shape is part of the key.
///
/// `generate_ms` should exclude nothing but the load: the wait a user actually
/// experiences includes loading the model, so a first run legitimately reads
/// slower than a warm one. Weighting recent runs more heavily lets the number
/// settle on the warm case, which is what most runs will be.
#[allow(clippy::too_many_arguments)] // one call site; a struct would only add indirection
pub fn record_for(
    paths: &AppPaths,
    model_id: &str,
    has_reference: bool,
    steps: u32,
    width: u32,
    height: u32,
    images: usize,
    generate_ms: u64,
    load_ms: u64,
) -> Result<()> {
    let mut table = load(paths);
    let entry = table.entry(rate_key(model_id, has_reference)).or_default();

    // A load only happened if the model was not already resident. Its cost
    // belongs to the model rather than to the shape of the run, so keying by
    // shape makes it settle twice. That is a slower estimate, not a wrong one
    // -- unlike the rate, where mixing the shapes produced a figure that
    // described no run anybody had made.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway installation root, so these tests never read or write the
    /// user's own measurements.
    struct Scratch {
        path: std::path::PathBuf,
    }

    impl Scratch {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("modelstudio-timings-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn paths(&self) -> AppPaths {
            AppPaths::at(self.path.clone())
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// One 1 Mpx, 10-step run that took `ms`.
    fn run(paths: &AppPaths, has_reference: bool, ms: u64) {
        record_for(
            paths,
            "flux1-dev-4bit",
            has_reference,
            10,
            1000,
            1000,
            1,
            ms,
            0,
        )
        .unwrap();
    }

    /// The defect: board panels are conditioned on reference images and cost
    /// two to three times as much per step, but they were folded into the same
    /// key as plain text-to-image runs. Generate then quoted an average of the
    /// two -- "about 19 minutes" for a run that takes a few -- as measured.
    #[test]
    fn reference_runs_do_not_move_the_plain_rate() {
        let scratch = Scratch::new();
        let paths = scratch.paths();

        run(&paths, false, 10_000); // 1000 ms per step per Mpx
        for _ in 0..5 {
            run(&paths, true, 30_000); // three times as slow
        }

        let plain = get(&paths, "flux1-dev-4bit").expect("the plain run was measured");
        assert_eq!(plain.runs, 1);
        assert!(
            (plain.ms_per_step_mpx - 1000.0).abs() < 1.0,
            "five panel runs dragged the plain rate to {}",
            plain.ms_per_step_mpx
        );
        assert!(
            (plain.worst_ms_per_step_mpx - 1000.0).abs() < 1.0,
            "the tail warning inherited a panel's cost: {}",
            plain.worst_ms_per_step_mpx
        );

        let panels = get_for(&paths, "flux1-dev-4bit", true).expect("the panels were measured");
        assert_eq!(panels.runs, 5);
        assert!(
            (panels.ms_per_step_mpx - 3000.0).abs() < 1.0,
            "the panel rate was diluted by the plain run: {}",
            panels.ms_per_step_mpx
        );
    }

    /// Entries written before the split mixed both shapes, so there is no way
    /// to tell what they measured. They are ignored rather than quoted.
    #[test]
    fn an_entry_from_before_the_split_is_not_quoted() {
        let scratch = Scratch::new();
        let paths = scratch.paths();
        let mut mixed = Table::new();
        mixed.insert(
            "flux1-dev-4bit".to_string(),
            Sample {
                ms_per_step_mpx: 19_000.0,
                runs: 12,
                worst_ms_per_step_mpx: 40_000.0,
                load_ms: 0.0,
                load_samples: 0,
            },
        );
        save(&paths, &mixed).unwrap();

        assert!(get(&paths, "flux1-dev-4bit").is_none());
        assert!(get_for(&paths, "flux1-dev-4bit", true).is_none());

        // And a fresh measurement lands where it will be read again.
        run(&paths, false, 10_000);
        assert_eq!(get(&paths, "flux1-dev-4bit").map(|s| s.runs), Some(1));
    }
}
