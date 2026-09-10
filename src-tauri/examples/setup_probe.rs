//! Exercise the first-run runtime bootstrap against a scratch directory.
//!
//! The app's setup wizard is otherwise only reachable by destroying a working
//! installation. This runs the same `setup::bootstrap` the wizard calls,
//! pointed somewhere disposable.
//!
//!   cargo run --example setup_probe -- /tmp/scratch [--full]
//!
//! Without `--full` it stops after the Python runtime and virtual environment,
//! skipping the multi-gigabyte engine install that has already been proven.

use melp_model_studio_lib::setup_probe as probe;
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/tmp/modelstudio-probe".to_string());
    let full = args.iter().any(|a| a == "--full");

    let paths = probe::paths_at(PathBuf::from(&root));
    println!("root: {root}");
    println!(
        "mode: {}\n",
        if full {
            "full (installs mlx-gen)"
        } else {
            "runtime only"
        }
    );

    let report = |step: &str, detail: &str, progress: Option<f32>| {
        let pct = progress
            .map(|p| format!("{:>3.0}%", p * 100.0))
            .unwrap_or("    ".into());
        // pip and curl are chatty; one line per step keeps the log readable.
        println!("  [{pct}] {step:<8} {}", &detail[..detail.len().min(96)]);
    };

    let started = std::time::Instant::now();
    let result = if full {
        probe::bootstrap(&report, &paths, true).await
    } else {
        probe::bootstrap_runtime_only(&report, &paths, true).await
    };

    match result {
        Ok(()) => {
            println!("\nOK in {:.0}s", started.elapsed().as_secs_f32());
            println!("  python  : {}", paths.python_bin().display());
            println!("  venv    : {}", paths.venv_python().display());
            for (label, p) in [
                ("python", paths.python_bin()),
                ("venv", paths.venv_python()),
            ] {
                println!("  {label} exists: {}", p.exists());
            }
        }
        Err(e) => {
            eprintln!("\nFAILED: {e}");
            std::process::exit(1);
        }
    }
}
