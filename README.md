<div align="center">
  <img src="docs/logo.svg" alt="Model Studio" width="96" height="96">

  <h1>Model Studio</h1>

  <p>
    <b>Image models that run on your Mac and nothing else.</b><br>
    Generate, edit, upscale, and turn a story into an illustrated page —
    entirely offline, with everything encrypted at rest.
  </p>

  <p>
    <img alt="platform" src="https://img.shields.io/badge/platform-macOS%20%7C%20Apple%20Silicon-black">
    <img alt="licence" src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue">
    <img alt="tests" src="https://img.shields.io/badge/tests-180-green">
  </p>
</div>

---

<div align="center">
  <img src="docs/img/page-one.webp" alt="A finished comic page: three panels of the same woman in the same room, with typeset captions." width="560">
  <p><sub>A page the app made, from Kate Chopin&rsquo;s <i>The Story of an Hour</i> pasted in whole.<br>
  Six panels, sixteen minutes on an M4. Captions are typeset into the page, not drawn into the picture.</sub></p>
</div>

---

Nothing leaves the machine. No account, no API key, no telemetry, no cloud
inference. The models run on Apple's MLX framework, and everything the app
produces is sealed with ChaCha20-Poly1305 before it touches disk.

That matters for work you are contractually unable to paste into a hosted
tool — an unreleased script, a client brief under NDA, anything you would
rather not upload.

Tauri 2 (Rust) shell, React frontend, and a long-lived Python sidecar that
drives [MLX-Gen](https://github.com/lpalbou/mlx-gen) and
[mflux](https://github.com/filipstrand/mflux).

## What it does

- **Generate** — text to image, with live previews as the picture forms.
- **Edit** — instruction editing, inpainting with a painted mask, outpainting,
  latent image-to-image, and reference-guided edits.
- **Board** — paste prose and get an illustrated comic. It reads the story
  first: who is in it, where it happens, and how many panels the story needs —
  there is no panel count to choose, because a number picked before the story
  is read is a guess. Every name it returns is checked against your text, so a
  character it invented is dropped rather than drawn. One character sheet and
  one room per scene are cast once, and every panel references both, so the
  same person appears in the same place throughout. Panels are drawn a few at a
  time, and a frame that comes out wrong is repaired by writing what is wrong
  with it rather than rolling a new seed. The result is one comic in the
  library, not a pile of loose pictures.

  <table>
  <tr>
  <td width="50%"><img src="docs/img/character-sheet.webp" alt="A character reference sheet: the same woman from three angles."></td>
  <td width="50%"><img src="docs/img/room-panel.webp" alt="A panel of the same character standing at a window in the room drawn for that scene."></td>
  </tr>
  <tr>
  <td><sub>The sheet, cast once from the description or from a photograph.</sub></td>
  <td><sub>A panel referencing both the sheet and that scene&rsquo;s room, which is what keeps the same person in the same place.</sub></td>
  </tr>
  </table>
- **Upscale** — SeedVR2 restoration. About 27 seconds to 512 px and 64 to 768;
  anything too large for one pass is done in overlapping pieces, which is the
  difference between a finished picture and a dead engine.

  <img src="docs/img/upscale.webp" alt="The same panel enlarged naively on the left and by the model on the right." width="620">
- **Video** — text to video and animating a still, at 640x368 or larger.
  Smaller canvases return noise; see [the limits](#video).
- **Prompt help** — makes a request precise without inventing a scene, and can
  read the picture you are editing to say *which* jacket you meant.
- **Vault** — everything produced is encrypted at rest, unlocked by passphrase
  (or Touch ID, given a signed build), and exported only when you ask.

Measured timings for all of this are in
[docs/measurements.md](docs/measurements.md) — taken on a 16 GB M4, not
estimated.

## Installing

Download the latest `.dmg` from
[Releases](https://github.com/CasualOffice/studio/releases), open it, and drag
Model Studio to Applications.

It will refuse to open the first time — macOS says any app not signed by a
paid Apple Developer account "is damaged". It is not damaged, it is unsigned.
Clear the quarantine flag once:

```sh
xattr -dr com.apple.quarantine "/Applications/Model Studio.app"
```

On first launch the app provisions a private Python runtime (about 3 GB) under
`~/Library/Application Support/com.melp.modelstudio`, then offers to install
the models that make it work (about 9 GB for the starter set). Nothing is
installed system-wide. Budget roughly 12 GB and twenty minutes before the
first picture, most of it downloading.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).

Every dependency is permissively licensed and therefore compatible in this
direction: MLX, mlx-gen and mlx-vlm are MIT, PyTorch and transformers are
Apache-2.0, pillow-heif is BSD-3-Clause, and the Tauri and RustCrypto stacks
are MIT OR Apache-2.0. Combining them under the GPL is allowed; the reverse
would not be.

If you distribute a modified build, the GPL requires you to offer its source
under the same terms.

## Requirements

- Apple Silicon. MLX has no Intel Mac backend, and the app refuses to set up on one.
- macOS 13+.
- Disk for the runtime (~3 GB) plus whatever models you install.

## Getting started

```sh
npm install
npm run tauri dev          # needs port 5173 free
npm run tauri build        # produces a .app and .dmg
```

First launch installs a private Python 3.12 (from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone))
and `mlx-gen` into a venv under
`~/Library/Application Support/com.melp.modelstudio`. Your system Python is
never touched, and uninstalling is `rm -rf` on that one directory.

## Architecture

Board development: [production tracker](docs/board-production-tracker.md),
[test plan](docs/board-production-test-plan.md), and
[research/decision register](docs/board-production-research.md). These distinguish
the current illustrated-story workflow from the phased manga-production target.

```
src/                 React UI
src-tauri/src/
  catalog.rs         curated model list with published size/memory figures
  models.rs          fit classification against this machine
  hostinfo.rs        chip, unified memory, free disk on the right volume
  setup.rs           first-run Python + mlx-gen bootstrap
  engine.rs          sidecar process manager, JSON Lines multiplexer
  commands.rs        Tauri command surface
  vault/
    crypto.rs        chunked AEAD file format
    keychain.rs      Keychain KEK behind a biometric ACL
    store.rs         vault state, manifest, encrypted index
engine/
  worker.py          long-lived MLX worker
  vaultcrypto.py     the file format again, engine side
```

### Why a persistent Python worker

Loading a 4-8 GiB quantized package costs 10-30 seconds. A one-shot CLI call
per generation would pay that every time. The worker keeps one model resident
and is addressed over JSON Lines on stdin/stdout, with progress events streamed
back as they happen.

Only one model is held at a time. On a 16 GB machine there is no second slot.

## Memory: what actually decides whether a model runs

Download size is a poor proxy for peak memory, and the gap runs both ways.
Qwen Image 2512's package is 27.5 GiB but it peaks at 10.7 GiB; Qwen Image Edit
2511 q8 is 28.3 GiB on disk and peaks at **30.9 GiB** at only 768x432.

The catalog therefore carries two numbers per model:

- **weight floor** — roughly the package size; must be resident for the whole run.
- **peak memory** — measured where MLX-Gen publishes a benchmark, interpolated
  otherwise (marked with `*` in the UI).

### Low-RAM mode

`low_ram` clears the MLX cache at transformer-block boundaries and between
denoise steps, releases the text conditioner after encoding, and releases
inactive denoisers before decode. It also runs one image at a time, because
MLX-Gen refuses low-RAM together with several seeds.

It bounds **transient** memory. It does not touch resident weights. So:

- peak above the ceiling, weight floor below it → offered, low-RAM auto-enabled
- weight floor above the ceiling → refused outright, no override

That second case is why Qwen Image Edit is not reachable on 16 GB: its smallest
package is 17 GiB of weights, more than the machine has in total.

## The vault

Everything the app generates or imports is encrypted before it reaches disk.
There is no plaintext output directory.

```
vault/
  vault.json    wrapped data keys + KDF parameters (no secrets)
  index.enc     encrypted metadata: names, prompts, models, seeds
  blobs/<uuid>  encrypted content, random names, no extensions
```

### Keys

A random 32-byte data key (DEK) encrypts all content. The DEK is never stored
raw; it is wrapped by either of:

1. **Passphrase** — Argon2id (64 MiB, 3 passes, 4 lanes) → KEK. Always present.
2. **Touch ID** — a random KEK in the macOS Keychain behind
   `SecAccessControl(.userPresence)` and `WhenUnlockedThisDeviceOnly`, so it
   never syncs to iCloud and never leaves the Mac. Optional, **and unavailable
   in an unsigned build** — see below.

### Touch ID needs a signed build

Biometric access control lives only in the data-protection Keychain, which
macOS gates on a signature carrying an Apple Developer *team identifier*.
Without one, `SecItemAdd` returns `errSecMissingEntitlement` (-34018); adding a
`keychain-access-groups` entitlement to an ad-hoc signature is worse — taskgated
SIGKILLs the process.

Since this app is deliberately unsigned, the app probes the capability once at
startup with a throwaway item (which never prompts) and hides the Touch ID
option when it cannot work, explaining why. `cargo run --example keychain_probe`
reports the same thing.

The passphrase is the root of trust regardless; Touch ID was only ever a
shortcut to it. Nothing about at-rest encryption changes.

Either unwraps the same DEK. Losing the Keychain item (fingerprint change, new
Mac, reinstall) costs you the fast unlock, not the data. Changing the passphrase
re-wraps the DEK with a fresh salt, so nothing has to be re-encrypted.

### File format

`crypto.rs` and `vaultcrypto.py` implement the same format and are verified
byte-identical by `cargo run --example vault_vectors` piped into the Python
side. Chunked ChaCha20-Poly1305, 256 KiB chunks, per-file subkey via
HKDF-SHA256 so a counter nonce is safe. Each chunk's AAD binds the header, the
chunk index, and whether it is final, which makes truncation, reordering and
cross-file splicing detectable rather than silent.

### Plaintext handling

- **Output**: the worker asks MLX-Gen for in-memory artifacts, encodes PNG in
  process, and writes only sealed bytes. Plaintext never reaches disk.
- **Display**: a `vault://` scheme handler in Rust decrypts into memory and
  hands bytes to the WebView, with `Cache-Control: no-store`. It refuses while
  locked.
- **Input**: MLX-Gen loads source images by path, so sealed sources are
  decrypted into a `0700` temp directory for the duration of one run and
  removed in a `finally`. This is the one bounded window where plaintext
  exists on disk.
- **Export**: the only sanctioned way out. Explicit, user-chosen destination.

The engine subprocess is handed **per-file keys**, never the DEK, so it can seal
the artifact it just produced and open the inputs it was given — nothing else.

### What this does and does not protect against

Defeats: a stolen Mac, a lifted disk, a Time Machine backup, another user
account, anyone reading the app's directory.

Does not defeat: a process running as you while the vault is unlocked. No
desktop application can promise otherwise, and this one does not.

The vault locks on window close and after 15 minutes idle; locking zeroizes the
data key and releases model weights.

## Video

<a name="video"></a>**Video works, above 640x368.** Below that it returns
coloured noise, and for a long time everything here was run at 320x192, which
is why it was believed broken.

| Size | Peak | Time (9 frames, 20 steps) | Result |
|---|---|---|---|
| 320x192 | 9.72 GiB | 126 s | coloured noise |
| 480x272 | 9.91 GiB | 85 min | coloured noise |
| **640x368** | **10.16 GiB** | **70 min** | **a picture** |
| 704x384 | 10.20 GiB | 26 min | a picture |

The model is trained at 1280x704 and does not degrade gracefully far below it:
it stops converging and returns colour. The step count was not the problem --
fifty steps, the model's own figure, is still noise at 320x192 -- and neither
were the low-RAM flags, which change nothing except saving 3.7 GiB.

The thing worth carrying away: **resolution was never the memory constraint it
was assumed to be.** 704x384 is 4.4 times the pixels of 320x192 and costs half
a gigabyte more. What a larger canvas costs is time. A catalog figure nobody
measured -- 103.7 GiB, taken at the model's native size -- made video look
impossible here, so everything was pushed onto the smallest canvas to make it
fit, and that canvas was the entire fault.

Every technical check passed on the broken output: valid MP4, correct
dimensions, frame count and fps, sealed without complaint. That is why it
shipped as working twice, and it is the argument for checking what a run
produced rather than that it finished.

Memory was never the problem. Wan 2.2 TI2V-5B was measured at **9.72 GiB** for
a complete run, well inside the budget, and the 103.7 GiB this table used to
quote was taken at the model's recommended 1280x704x81 and had never been
measured here.

The problem is the picture. Raising the step count to the model's own 50 did
not help, and neither did removing the image: the failure is in generation
itself, not in how a still is fed to it. The cause is narrowed to two
candidates -- three low-RAM flags this engine passes on every video call, or
the q8 checkpoint's silent conversion of attention weights to BF16 at load --
and the test between them has not been run yet. Until it is, video is not a
feature and the catalog says so.

Wan 2.5 and 2.6 have no public weights — Alibaba released them as a cloud API
only. Of the open ones:

| Model | Fits 16 GB | Takes a still |
|---|---|---|
| Wan 2.1 VACE 1.3B | yes | **no** — transforms existing footage, raises on `image_path` |
| Wan 2.2 TI2V-5B | yes (9.72 GiB measured) | yes, but the clip is noise |
| Wan 2.2 T2V-A14B | no (33 GiB peak) | no |
| MiniMax-H3 | no (464 GiB repo) | yes |
| Bernini-R 1.3B | yes | yes, and the result was poor |

Video models also want more canvas than fits: VACE is trained at 832x480 and
Bernini outputs 480p, while 9.5 GiB peak was measured at 320x192 with 17
frames. Running well below native resolution is its own quality problem.

The app refuses rather than pretending. Choose a model that cannot take a
still and it says so, instead of silently dropping the picture and returning
an unrelated clip — which is what it used to do.

## Distribution

The app is deliberately **not code-signed or notarized**. It is built for this
machine, not for redistribution, so there is no Apple Developer account in the
loop and `hardenedRuntime` is off.

```sh
npm run tauri build          # produces .app and .dmg, unsigned
```

Gatekeeper quarantines anything unsigned that arrives from outside. A build made
locally runs as-is; one that has been copied from another machine or downloaded
needs the quarantine flag cleared once:

```sh
xattr -dr com.apple.quarantine "/Applications/Model Studio.app"
```

For day-to-day work, running the binary directly skips bundling entirely:

```sh
cargo tauri build --debug --no-bundle
./src-tauri/target/debug/melp-model-studio
```

Signing would only become necessary to hand the app to someone else.

## Verified on hardware

Run on an M4 / 16 GB, macOS 26.3, with `AbstractFramework/flux.2-klein-4b-4bit`:

| Step | Result |
| --- | --- |
| Runtime install | 1.4 GB (Python 3.12.14, mlx 0.31.2, mlx-gen 0.36.0, torch 2.14.0) |
| Model download | 4.30 GiB, matching the catalog's published 4.3 GiB exactly |
| Model load | 10-14 s cold, then resident |
| Generate 512x512, 4 steps | ~40 s denoise, ~56 s wall clock including load |
| Seal | 307,311 B PNG -> 307,371 B blob (28 B header + 2 chunk tags) |
| Decrypt and verify | round-trips to a valid 512x512 PNG; wrong key rejected |

## Prompt help

A small vision-language model (`mlx-community/Qwen2-VL-2B-Instruct-4bit`,
1.18 GiB) rewrites prompts locally. Nothing leaves the Mac. It is ~2.6 GiB at
peak, small enough to stay resident beside an image model rather than evicting
it, and it is an ordinary catalog entry so it installs like any other model.

For an edit it is shown the source image, so it can name the subject:

| Typed | Rewritten |
| --- | --- |
| `make it blue` | change the teapot to blue, keeping its shape and background exactly as they are |
| `put it on a dark table` | Put the ceramic teapot on a dark table, keeping its shape and position |

A 2B model handed an image will sometimes describe the picture instead of
following the instruction — an early run turned `make it blue` into
`remove the background`. Silently performing a *different* edit is worse than
performing none, so a rewrite that carries over none of the user's meaningful
words is discarded and the original kept. The UI also offers an explicit
**Undo rewrite**.

Downloads for this model go through `huggingface_hub`, not `mlxgen download`:
MLX-Gen exits 0 without fetching anything for a repo it does not recognise,
which looked exactly like a successful zero-byte download.

## What it does

| Tab | |
| --- | --- |
| **Generate** | Text to image. |
| **Edit** | Five modes: instruction edit, masked inpaint, outpaint, latent restyle, and crop/rotate. |
| **Video** | Text to video, with frame count as the control that matters. Image to video is not usable yet. |
| **Enlarge** | SeedVR2 super-resolution, currently blocked by a version conflict (below). |
| **Models** | Curated catalog plus any Hugging Face repo, judged against this machine before download. |
| **Vault** | Everything made or imported, encrypted, searchable, with the chain that produced each item. |
| **Activity** | Engine versions, resident model, live memory, and every decision the engine made quietly. |
| **Security** | Passphrase, Touch ID, model storage location, vault repair. |

Work carries between tabs by id -- generate, edit that, animate the result --
without ever writing plaintext to disk.

### Editing

Instruction edits, masked inpainting (paint the region, only that is
regenerated), outpainting (grow the canvas past the original frame), latent
restyle, and crop/rotate/straighten. The last of those runs in the browser: it
never needed a diffusion model, and asking one to reframe a picture is both
slow and lossy.

Measured on an M4 at 512x512, 4 steps: inpaint 36s, outpaint 56s (512 to 768),
multi-reference 51s.

### Live previews

Each denoise step is decoded through a published tiny autoencoder and sent to
the interface as a small JPEG. A hundred-second generation was otherwise a
progress bar and nothing else. Six steps cost 91 KB in total.

Implemented against MLX-Gen's own `StepwiseHandler`, not its documentation,
whose example calls an `unpack_latents` that no longer exists -- following it
produced a handler that ran every step and silently failed on every one.

### Prompt help

A small vision-language model (~1.2 GiB) runs locally. It does not rewrite your
request: it reports what the picture contains, and the instruction is assembled
from your own words with those observations appended as context to preserve.

    make it blue
    -> make a beige ceramic teapot blue, preserving its smooth glazed
       stoneware and soft daylight from the left, and leaving the crumpled
       linen cloth and pale wall unchanged

An earlier version let the model restate the instruction after seeing the
picture. It folded the picture's details into the request and sometimes
replaced it outright: "make it blue" came back as "remove the background".

Observations that conflict with the request are dropped, so "put it on a dark
table" does not also say "leaving the linen cloth unchanged".

### Style adapters

LoRA adapters install by repository id. The declared base model is shown before
downloading, because an adapter trained for Klein 9B will not load against the
4B and the file gives no hint. Adapters are part of the resident model's cache
key: without that, switching between runs would reuse the loaded model and
silently apply the wrong style, or none.

## Resource policy

The engine must not be able to take the machine down with it.

- MLX gets a hard memory ceiling and refuses allocations past it rather than
  letting macOS swap. A failed generation beats an unresponsive Mac.
- Threads are capped, leaving cores free, and the worker runs `nice`d.
- One model is resident at a time. This is an invariant, not an optimisation:
  the prompt assistant briefly had its own slot on the theory that 2.6 GiB
  would sit happily beside 5.6 GiB, and on 16 GB that combination paged badly
  enough that a two-second rewrite took minutes.
- Weights are released after ten minutes idle.

## Simple and Advanced

Simple mode is the default and shows a prompt box, Shape, Effort, and one
button. The model is chosen automatically — the heaviest installed one that
still fits comfortably — and steps come from the model's own default scaled by
the Effort setting, because a distilled checkpoint tuned for 4 steps gains
nothing from 40 while a base model at 4 steps produces mush.

Advanced exposes model choice, steps, guidance, seed, batch count, edit mode
and the memory controls.

### Known upstream issues

- **Bonsai is broken in mlx-gen 0.36.0.** `BonsaiImage` inherits FLUX.2 Klein's
  `generate_image`, which reads `self.compiled_predict_cache` -- an attribute
  the Bonsai class never assigns, so every run raises `AttributeError` once the
  weights are already resident. The catalog marks it unavailable.
- **The cache-only download gate misfires.** With `huggingface_hub` 1.x,
  MLX-Gen's `snapshot_download(..., local_files_only=True)` probe raises
  `LocalEntryNotFoundError` even when every file is cached, so a fully
  downloaded model is refused at generation time. `mlxgen download` reports
  success and `mlxgen generate` then fails on the same model. The worker lifts
  the gate with `allow_downloads()`; loading inside it transfers zero new bytes.
  This is safe here because the Rust host enforces the same policy one level up
  -- it refuses to start a job unless the model is installed, and checks the
  disk budget before any download.
- **Upscaling is not a generation capability.** `get_model_capabilities()`
  reports SeedVR2 with zero modes, because restoration lives in a separate
  `ModelCapabilities.restoration` field. Reading only `capabilities` makes four
  working upscalers look broken — which is exactly what happened here, and the
  catalog carried a wrong "unavailable" note for some time as a result. Both
  fields are read now.
- **The scheduler needs at least two steps.** `FlowMatchEulerDiscreteScheduler`
  rejects `num_inference_steps=1`, so the UI's step slider starts at 2.

## Tests

```sh
cargo test --lib                                   # crypto, URL parsing, engine liveness
python engine/test_worker.py                       # 48 engine tests
cargo run --example vault_vectors > vectors.json   # cross-language format vectors
cargo run --example vault_lifecycle                # create, dedup, repair, export
cargo run --example storage_move                   # relocation preserves symlinks
cargo run --example keychain_probe                 # Touch ID capability
cargo run --example setup_probe -- /tmp/scratch    # first-run bootstrap, headless
```

CI runs all of these on macOS. Clippy warnings are failures: the codebase is at
zero and the only way that stays true is if drift breaks the build.

The probes cover the two paths that can destroy data and are otherwise only
reachable through the interface. `vault_lifecycle` plants an orphaned blob and
checks repair identifies it; `storage_move` checks a relocation preserves
symlinks, since following them would duplicate every weight file and silently
double the space used.

`setup_probe` runs the real `setup::bootstrap` against a throwaway directory, so
the first-run path can be verified without destroying a working installation.
Add `--full` to include the multi-gigabyte engine install.
