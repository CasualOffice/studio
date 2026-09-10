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
