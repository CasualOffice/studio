//! Curated model catalog.
//!
//! Sizes and peak-memory figures are taken from MLX-Gen's published
//! quantization matrix and memory-tier benchmarks, not guessed from parameter
//! counts. `peak_gib` is what actually decides whether a model runs on a given
//! Mac -- package size on disk is often a poor proxy for it.

use serde::Serialize;

#[derive(Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    TextToImage,
    Edit,
    Upscale,
    Video,
    /// Reads and rewrites prompts; not an image generator.
    Assist,
}

#[derive(Serialize, Clone, Debug)]
pub struct ModelEntry {
    pub id: &'static str,
    pub repo: &'static str,
    pub name: &'static str,
    /// Router family hint for local packages; `None` lets mlx-gen infer it.
    pub family: Option<&'static str>,
    pub tasks: &'static [Task],
    pub quantize: Option<u8>,
    /// On-disk size of the published package, GiB.
    pub package_gib: f32,
    /// Measured (or, where marked estimated, interpolated) peak RSS, GiB.
    pub peak_gib: f32,
    /// True when `peak_gib` is interpolated rather than published.
    pub peak_estimated: bool,
    pub steps_default: u32,
    /// How many source images an edit may take. Most routes are
    /// single-reference: FLUX.2 Klein accepts one and rejects two outright
    /// ("does not support edit-reference image-to-image generation"), so
    /// offering more would fail only after the weights were resident.
    pub max_edit_images: u32,
    /// Guidance the UI starts at.
    pub guidance_default: f32,
    /// Highest guidance this route accepts. Distilled and Turbo checkpoints are
    /// trained without classifier-free guidance and reject anything above 1.0.
    pub guidance_max: f32,
    pub notes: &'static str,
    /// Set when the model cannot currently run regardless of hardware, e.g. an
    /// upstream defect. Shown to the user instead of a memory verdict.
    pub broken: Option<&'static str>,
}

pub const CATALOG: &[ModelEntry] = &[
    // ---- Prompt assistant ------------------------------------------------
    ModelEntry {
        id: "qwen2-vl-2b-4bit",
        repo: "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        name: "Prompt assistant (Qwen2-VL 2B)",
        family: None,
        tasks: &[Task::Assist],
        quantize: Some(4),
        package_gib: 1.5,
        peak_gib: 2.6,
        peak_estimated: true,
        steps_default: 0,
        max_edit_images: 1,
        guidance_default: 1.0,
        guidance_max: 1.0,
        notes: "Rewrites your prompt, and can look at the image you are editing. \
                Small enough to stay loaded beside an image model. Runs entirely \
                on this Mac — nothing is sent anywhere.",
        broken: None,
    },

    // ---- Comfortable on 16 GB -------------------------------------------
    ModelEntry {
        id: "bonsai-2bit",
        repo: "prism-ml/bonsai-image-ternary-4B-mlx-2bit",
        name: "Bonsai Image 4B (ternary 2-bit)",
        family: Some("bonsai"),
        tasks: &[Task::TextToImage],
        quantize: None,
        package_gib: 3.6,
        peak_gib: 3.57,
        peak_estimated: false,
        steps_default: 4,
        max_edit_images: 1,
        guidance_default: 1.0,
        guidance_max: 1.0,
        notes: "Pre-packed ternary weights. Would be the fastest option at ~2.9s per 512px image.",
        // Verified on mlx-gen 0.36.0: BonsaiImage inherits FLUX.2 Klein's
        // generate_image, which reads self.compiled_predict_cache -- an
        // attribute the Bonsai class never assigns. Every run raises
        // AttributeError after the weights are already resident.
        broken: Some(
            "Broken in mlx-gen 0.36.0: the Bonsai route raises AttributeError on \
             compiled_predict_cache. Nothing to do but wait for an upstream fix.",
        ),
    },
    ModelEntry {
        id: "flux2-klein-4b-4bit",
        repo: "AbstractFramework/flux.2-klein-4b-4bit",
        name: "FLUX.2 Klein 4B (q4)",
        family: Some("flux2"),
        tasks: &[Task::TextToImage, Task::Edit],
        quantize: Some(4),
        package_gib: 4.3,
        peak_gib: 5.6,
        peak_estimated: true,
        steps_default: 4,
        max_edit_images: 1,
        guidance_default: 1.0,
        guidance_max: 1.0,
        notes: "Best all-rounder under 16 GB: does both generation and instruction editing.",
        broken: None,
    },
    ModelEntry {
        id: "flux2-klein-base-4b-4bit",
        repo: "AbstractFramework/flux.2-klein-base-4b-4bit",
        name: "FLUX.2 Klein Base 4B (q4)",
        family: Some("flux2"),
        tasks: &[Task::TextToImage, Task::Edit],
        quantize: Some(4),
        package_gib: 4.3,
        peak_gib: 5.6,
        peak_estimated: true,
        steps_default: 20,
        max_edit_images: 1,
        guidance_default: 3.5,
        guidance_max: 8.0,
        notes: "Same size as the distilled Klein but accepts guidance and negative prompts, \
                so prompts steer harder. Needs more steps in exchange.",
        broken: None,
    },
    ModelEntry {
        id: "seedvr2-3b-4bit",
        repo: "AbstractFramework/seedvr2-3b-4bit",
        name: "SeedVR2 3B (q4) — upscaler",
        family: None,
        tasks: &[Task::Upscale],
        quantize: Some(4),
        package_gib: 2.54,
        peak_gib: 2.89,
        peak_estimated: false,
        steps_default: 0,
        max_edit_images: 1,
        guidance_default: 1.0,
        guidance_max: 1.0,
        notes: "Image super-resolution and restoration. Pairs well with a small generator.",
        // Verified on mlx 0.31.2: SeedVR2 passes computed values as the
        // `repeats` argument of mx.repeat, which older MLX accepted as a 0-d
        // array and 0.31.2 rejects. It runs correctly on mlx 0.31.0 -- but
        // mlx-vlm needs 0.31.2 for mx.new_thread_local_stream, so the prompt
        // assistant and the upscaler cannot both work in one environment.
        broken: Some(
            "Unavailable in this build: SeedVR2 needs mlx 0.31.0, while the prompt \
             assistant needs 0.31.2. Only one can be installed at a time, and the \
             assistant is the default.",
        ),
    },
    ModelEntry {
        id: "seedvr2-3b-8bit",
        repo: "AbstractFramework/seedvr2-3b-8bit",
        name: "SeedVR2 3B (q8) — upscaler",
        family: None,
        tasks: &[Task::Upscale],
        quantize: Some(8),
        package_gib: 4.39,
        peak_gib: 4.73,
        peak_estimated: false,
        steps_default: 0,
        max_edit_images: 1,
        guidance_default: 1.0,
        guidance_max: 1.0,
        notes: "Higher-fidelity upscaling than the q4 package.",
        // Verified on mlx 0.31.2: SeedVR2 passes computed values as the
        // `repeats` argument of mx.repeat, which older MLX accepted as a 0-d
        // array and 0.31.2 rejects. It runs correctly on mlx 0.31.0 -- but
        // mlx-vlm needs 0.31.2 for mx.new_thread_local_stream, so the prompt
        // assistant and the upscaler cannot both work in one environment.
        broken: Some(
            "Unavailable in this build: SeedVR2 needs mlx 0.31.0, while the prompt \
             assistant needs 0.31.2. Only one can be installed at a time, and the \
             assistant is the default.",
        ),
    },
    ModelEntry {
        id: "seedvr2-7b-4bit",
        repo: "AbstractFramework/seedvr2-7b-4bit",
        name: "SeedVR2 7B (q4) — upscaler",
        family: None,
        tasks: &[Task::Upscale],
        quantize: Some(4),
        package_gib: 4.79,
        peak_gib: 5.10,
        peak_estimated: false,
        steps_default: 0,
        max_edit_images: 1,
        guidance_default: 1.0,
        guidance_max: 1.0,
        notes: "Strongest upscaler that still fits a 16 GB machine.",
        // Verified on mlx 0.31.2: SeedVR2 passes computed values as the
        // `repeats` argument of mx.repeat, which older MLX accepted as a 0-d
        // array and 0.31.2 rejects. It runs correctly on mlx 0.31.0 -- but
        // mlx-vlm needs 0.31.2 for mx.new_thread_local_stream, so the prompt
        // assistant and the upscaler cannot both work in one environment.
        broken: Some(
            "Unavailable in this build: SeedVR2 needs mlx 0.31.0, while the prompt \
             assistant needs 0.31.2. Only one can be installed at a time, and the \
             assistant is the default.",
        ),
    },
    ModelEntry {
        id: "z-image-4bit",
        repo: "AbstractFramework/z-image-4bit",
        name: "Z-Image (q4)",
        family: Some("z-image"),
        tasks: &[Task::TextToImage, Task::Edit],
        quantize: Some(4),
        package_gib: 5.5,
        peak_gib: 6.8,
        peak_estimated: true,
        steps_default: 20,
        max_edit_images: 1,
        guidance_default: 4.0,
        guidance_max: 10.0,
        notes: "Base checkpoint, so it accepts real guidance and negative prompts.",
        broken: None,
    },
    ModelEntry {
        id: "z-image-turbo-4bit",
        repo: "AbstractFramework/z-image-turbo-4bit",
        name: "Z-Image Turbo (q4)",
        family: Some("z-image"),
        tasks: &[Task::TextToImage, Task::Edit],
        quantize: Some(4),
        package_gib: 5.5,
        peak_gib: 6.8,
        peak_estimated: true,
        steps_default: 9,
        max_edit_images: 1,
        guidance_default: 1.0,
        guidance_max: 1.0,
        notes: "Few-step distilled variant. Also handles edits. No guidance or negative prompt.",
        broken: None,
    },
    ModelEntry {
        id: "flux2-klein-4b-8bit",
        repo: "AbstractFramework/flux.2-klein-4b-8bit",
        name: "FLUX.2 Klein 4B (q8)",
        family: Some("flux2"),
        tasks: &[Task::TextToImage, Task::Edit],
        quantize: Some(8),
        package_gib: 8.0,
        peak_gib: 9.23,
        peak_estimated: false,
        steps_default: 4,
        max_edit_images: 1,
        guidance_default: 1.0,
        guidance_max: 1.0,
        notes: "Higher quality than the q4 package; passed strict outpaint validation.",
        broken: None,
    },
    ModelEntry {
        id: "flux2-klein-9b-4bit",
        repo: "AbstractFramework/flux.2-klein-9b-4bit",
        name: "FLUX.2 Klein 9B (q4)",
        family: Some("flux2"),
        tasks: &[Task::TextToImage, Task::Edit],
        quantize: Some(4),
        package_gib: 8.9,
        peak_gib: 10.4,
        peak_estimated: true,
        steps_default: 4,
        max_edit_images: 1,
        guidance_default: 1.0,
        guidance_max: 1.0,
        notes: "Gated, non-commercial source terms. Accept the licence on Hugging Face first.",
        broken: None,
    },
    ModelEntry {
        id: "seedvr2-7b-8bit",
        repo: "AbstractFramework/seedvr2-7b-8bit",
        name: "SeedVR2 7B (q8) — upscaler",
        family: None,
        tasks: &[Task::Upscale],
        quantize: Some(8),
        package_gib: 8.62,
        peak_gib: 8.90,
        peak_estimated: false,
        steps_default: 0,
        max_edit_images: 1,
        guidance_default: 1.0,
        guidance_max: 1.0,
        notes: "Best published image upscaler.",
        // Verified on mlx 0.31.2: SeedVR2 passes computed values as the
        // `repeats` argument of mx.repeat, which older MLX accepted as a 0-d
        // array and 0.31.2 rejects. It runs correctly on mlx 0.31.0 -- but
        // mlx-vlm needs 0.31.2 for mx.new_thread_local_stream, so the prompt
        // assistant and the upscaler cannot both work in one environment.
        broken: Some(
            "Unavailable in this build: SeedVR2 needs mlx 0.31.0, while the prompt \
             assistant needs 0.31.2. Only one can be installed at a time, and the \
             assistant is the default.",
        ),
    },
    // ---- Tight on 16 GB --------------------------------------------------
    ModelEntry {
        id: "z-image-turbo-8bit",
        repo: "AbstractFramework/z-image-turbo-8bit",
        name: "Z-Image Turbo (q8)",
        family: Some("z-image"),
        tasks: &[Task::TextToImage, Task::Edit],
        quantize: Some(8),
        package_gib: 10.2,
        peak_gib: 11.49,
        peak_estimated: false,
        steps_default: 9,
        max_edit_images: 1,
        guidance_default: 1.0,
        guidance_max: 1.0,
        notes: "Adds native inpainting (10.57 GiB peak) and latent img2img (11.49 GiB peak).",
        broken: None,
    },
    ModelEntry {
        id: "fibo-4bit",
        repo: "AbstractFramework/fibo-4bit",
        name: "FIBO (q4)",
        family: Some("fibo"),
        tasks: &[Task::TextToImage],
        quantize: Some(4),
        package_gib: 10.2,
        peak_gib: 11.39,
        peak_estimated: false,
        steps_default: 8,
        max_edit_images: 1,
        guidance_default: 3.5,
        guidance_max: 12.0,
        notes: "Mixed q4/BF16 policy. Text-to-image only; FIBO Edit is a separate model.",
        broken: None,
    },
    ModelEntry {
        id: "ernie-image-turbo-8bit",
        repo: "AbstractFramework/ernie-image-turbo-8bit",
        name: "ERNIE Image Turbo (q8)",
        family: Some("ernie-image"),
        tasks: &[Task::TextToImage, Task::Edit],
        quantize: Some(8),
        package_gib: 11.5,
        peak_gib: 12.9,
        peak_estimated: false,
        steps_default: 8,
        max_edit_images: 1,
        guidance_default: 1.0,
        guidance_max: 1.0,
        notes: "Good latent restyle. The published -4bit repo is mislabelled q8; avoid it.",
        broken: None,
    },
    // ---- Documented but out of reach on 16 GB ---------------------------
    ModelEntry {
        id: "fibo-8bit",
        repo: "AbstractFramework/fibo-8bit",
        name: "FIBO (q8)",
        family: Some("fibo"),
        tasks: &[Task::TextToImage],
        quantize: Some(8),
        package_gib: 14.5,
        peak_gib: 15.89,
        peak_estimated: false,
        steps_default: 8,
        max_edit_images: 1,
        guidance_default: 3.5,
        guidance_max: 12.0,
        notes: "Needs roughly 24 GB of unified memory.",
        broken: None,
    },
    ModelEntry {
        id: "qwen-image-edit-2511-4bit",
        repo: "AbstractFramework/qwen-image-edit-2511-4bit",
        name: "Qwen Image Edit 2511 (q4)",
        family: Some("qwen"),
        tasks: &[Task::Edit],
        quantize: Some(4),
        package_gib: 17.0,
        peak_gib: 20.0,
        peak_estimated: true,
        steps_default: 4,
        max_edit_images: 3,
        guidance_default: 4.0,
        guidance_max: 12.0,
        notes: "Strongest open editor, but MLX-Gen puts Qwen edit routes at the 64 GB tier.",
        broken: None,
    },
    ModelEntry {
        id: "qwen-image-edit-2511-8bit",
        repo: "AbstractFramework/qwen-image-edit-2511-8bit",
        name: "Qwen Image Edit 2511 (q8)",
        family: Some("qwen"),
        tasks: &[Task::Edit],
        quantize: Some(8),
        package_gib: 28.3,
        peak_gib: 30.91,
        peak_estimated: false,
        steps_default: 4,
        max_edit_images: 3,
        guidance_default: 4.0,
        guidance_max: 12.0,
        notes: "30.91 GiB measured peak at only 768x432. Needs a 64 GB machine.",
        broken: None,
    },
    ModelEntry {
        id: "qwen-image-2512-8bit",
        repo: "AbstractFramework/qwen-image-2512-8bit",
        name: "Qwen Image 2512 (q8)",
        family: Some("qwen"),
        tasks: &[Task::TextToImage],
        quantize: Some(8),
        package_gib: 27.5,
        peak_gib: 10.73,
        peak_estimated: false,
        steps_default: 4,
        max_edit_images: 1,
        guidance_default: 3.5,
        guidance_max: 12.0,
        notes: "Memory fits 16 GB, but the 27.5 GiB package will not fit your free disk.",
        broken: None,
    },
    ModelEntry {
        id: "wan22-ti2v-5b-8bit",
        repo: "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit",
        name: "Wan 2.2 TI2V-5B (q8) — video",
        family: Some("wan"),
        tasks: &[Task::Video],
        quantize: Some(8),
        package_gib: 16.9,
        peak_gib: 103.7,
        peak_estimated: false,
        steps_default: 20,
        max_edit_images: 1,
        guidance_default: 4.0,
        guidance_max: 12.0,
        notes: "103.7 GiB peak at 1280x704. Listed for completeness; a 128 GB-class workload.",
        broken: None,
    },
    ModelEntry {
        id: "wan22-t2v-a14b-8bit",
        repo: "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
        name: "Wan 2.2 T2V-A14B (q8) — video",
        family: Some("wan"),
        tasks: &[Task::Video],
        quantize: Some(8),
        package_gib: 39.5,
        peak_gib: 33.0,
        peak_estimated: false,
        steps_default: 12,
        max_edit_images: 1,
        guidance_default: 4.0,
        guidance_max: 12.0,
        notes: "33 GiB peak for a 384x224, 33-frame clip. 64 GB is the first sane tier.",
        broken: None,
    },
];

pub fn find(id: &str) -> Option<&'static ModelEntry> {
    CATALOG.iter().find(|m| m.id == id)
}
