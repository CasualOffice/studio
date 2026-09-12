# What this machine actually does

Measured on a 16 GB M4, not estimated. Recorded so the same questions are not
argued from intuition a second time.

## Generation

| Task | Model | Settings | Time |
|---|---|---|---|
| Text to image | FLUX.2 Klein 4B q4 | 512², 4 steps | 51 s |
| Reference panel | FLUX.2 Klein 4B q4 | 512², 4 steps, 1 reference | 44–46 s |
| Reference panel, photoreal | FLUX.2 Klein 4B q4 | 512², 4 steps, 1 reference | 139–144 s |
| Upscale 2× | SeedVR2 3B q4 | 256×160 → 512×320 | 6.2 s |
| Model load | FLUX.2 Klein 4B q4 | — | 2.4 s |

## Image to video

Wan 2.2 TI2V-5B q8, `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit`, on this
16 GB machine, starting from a still:

| Clip | Steps | Peak | Load | Generate | Wall |
|---|---|---|---|---|---|
| 320x192, 9 frames | 8 | 9.72 GiB | 3.1 s | 24.2 s | 34 s |
| 480x320, 17 frames | 12 | did not complete | 3.2 s | ~90 s/step, paging | stopped at step 10 |

9.72 GiB against a 12 GiB budget, so the small size fits with room. The larger
one reached step 10 of 12 while paging heavily and was stopped; it is not known
whether it completes.

The catalog previously claimed **103.7 GiB** for this model with
`peak_estimated: false`, which marked it hopeless and hid the only working
image-to-video model on this machine. That figure was for the model's
recommended 1280x704x81 and had never been measured here. Attention is
quadratic in sequence length and the latent sequence at 320x192x9 is roughly
sixty times shorter, so almost all of it disappears.

The text encoder is the largest single component at 10.58 GiB, against a 5.03
GiB transformer and a 1.31 GiB VAE. It is released before denoising --
`keep_text_encoder_resident` defaults to `False`, which is what makes the model
fit. `release_text_encoder`, which the engine passes, is not a parameter this
route accepts and is dropped with a warning; the default already does the right
thing.

## Upscale memory, and why it crashed the engine

SeedVR2 3B q4, MLX cache limit 1 GiB, one pass, on this 16 GB machine:

| Output | Peak memory | Time | Fits in the 12 GiB budget? |
|---|---|---|---|
| 512×512 | 6.43 GiB | 27 s | yes |
| 768×768 | 10.44 GiB | 64 s | yes |
| 1024×1024 | 16.57 GiB | 173 s | **no** |

Peak tracks output area linearly across that range:

    peak GiB = 3.06 + 12.88 x output-megapixels

which holds to within 0.25 GiB at all three points. At a 12 GiB budget that
caps a single pass near **833 pixels on the longest edge** — less than the
source for any picture worth enlarging, which is why one pass is not enough.

Going over does not raise a Python error. MLX reports the Metal failure from a
command-buffer completion handler, which cannot raise into Python: the C++
exception reaches `std::terminate` and aborts the whole engine, so the app only
sees "the engine stopped unexpectedly". It has to be refused before dispatch.

Tiled, the same work fits and finishes:

| Output | Tiles | Peak memory | Time |
|---|---|---|---|
| 1024×1024 | 2×2 at 372 px source | 10.11 GiB | 85–118 s |

Seams were checked on a smooth source, where any gradient spike can only be a
join: worst 2.73 against a median of 0.63, about 1% of the 0–255 range. Not
visible. The earlier measurement in the generation table (256×160 → 512×320,
6.2 s) was the only one ever taken, and being that small is exactly why the
ceiling went unnoticed.

Photoreal panels came out roughly three times slower than stylised ones in the
same session, but the machine had been downloading and building for hours by
then, so treat that ratio as unconfirmed rather than a property of the style.

## Prompt work

| Task | Model | Time |
|---|---|---|
| Clarify a prompt | Qwen3-4B-Instruct 4bit | ~2 s |
| Clarify a prompt | Qwen2-VL-2B 4bit | 6–12 s, and wrong |
| Story to 4-panel shot list | Qwen3-4B-Instruct 4bit | 43–51 s |
| Story to 4-panel shot list | Qwen2-VL-2B 4bit | fails: returns 1–2 panels, skips the story |

## LoRA training — works, but not on this machine

`mflux-train` trains LoRA adapters for `flux2` and `z_image`. It runs here and
produces a valid checkpoint. It is not practical:

    6 steps, 512px, rank 16, Klein 4B q4, --low-ram
    13 min 03 s total, mean 130.5 s/step, 41% CPU
    per-step degraded 91.9 s -> 148.2 s as the machine heated

Extrapolated, a real character adapter costs:

| Steps | At the mean | At the degraded rate |
|---|---|---|
| 100 | 3.6 h | 4.1 h |
| 200 | 7.2 h | 8.2 h |
| 500 | 18.1 h | 20.6 h |

The literature puts a trained LoRA ahead of every reference-image trick for
character identity, so this is the better answer in principle. At seven hours
per character it is not the answer here. Multi-reference conditioning costs
45 s per panel and needs no training, which is why the picture-board pipeline
uses it.

Two traps found on the way, in case anyone retries this:

- LoRA target paths are per-architecture. FLUX.2 has two block types with
  different shapes: `transformer_blocks.{n}.attn.to_q` on the double-stream
  blocks, and `single_transformer_blocks.{n}.attn.to_qkv_mlp_proj` on the
  single-stream ones, which fuse qkv and the MLP into one projection. The
  `layers.{n}.attention.to_q` paths in the shipped example are z-image's.
- Klein 4B is distilled and rejects guidance above 1.0, but the training
  preview generator asks for its own. Omit `monitoring` from the config
  entirely, or previews fail before the first step runs.

## What a video model costs

No image-to-video model both fits this machine and works:

- Wan 2.1 VACE 1.3B — fits, but transforms existing footage and takes no still
  image at all. Verified: it raises on `image_path`.
- Wan 2.2 TI2V-5B — takes a still, 103.7 GiB peak.
- MiniMax-H3 — has proper first-frame animation, 464 GiB repo.
- Bernini-R 1.3B — fits and takes a still, 16.4 GiB download, untested here.

Video models also want more canvas than fits. VACE is trained at 832×480 and
Bernini outputs 480p, while 9.5 GiB peak was measured at 320×192 with 17
frames. Running well below native resolution is its own quality problem.
