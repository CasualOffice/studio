# Model Studio

A macOS desktop app for running open image-generation and image-editing models
locally on Apple Silicon, with everything it produces encrypted at rest.

Tauri 2 (Rust) shell, React frontend, and a long-lived Python sidecar that
drives [MLX-Gen](https://github.com/lpalbou/mlx-gen) on Apple's MLX framework.

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

## Wan and video

Wan 2.5 and 2.6 have no public weights — Alibaba released them as a cloud API
only. Wan 2.1 and 2.2 are the open ones. Wan 2.2 is in the catalog and marked
unreachable here: T2V-A14B peaks at 33 GiB for a 384x224, 33-frame clip, and
TI2V-5B at 103.7 GiB at 1280x704.

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
- **SeedVR2 upscaling and the prompt assistant cannot coexist.** SeedVR2 passes
  computed values as `mx.repeat`'s `repeats` argument, which mlx 0.31.0 accepts
  as a 0-d array and 0.31.2 rejects with a `TypeError`. But mlx-vlm needs
  `mx.new_thread_local_stream`, absent before 0.31.2. Since MLX-Gen also caps
  mlx below 0.32, no single version satisfies both. The install pins
  **mlx 0.31.2** — generation, editing and the prompt assistant all work, and
  SeedVR2 is marked unavailable in the catalog. Downgrading to 0.31.0 reverses
  the trade.
- **The scheduler needs at least two steps.** `FlowMatchEulerDiscreteScheduler`
  rejects `num_inference_steps=1`, so the UI's step slider starts at 2.

## Tests

```sh
cargo test --lib                                   # crypto, URL parsing
cargo run --example vault_vectors > vectors.json   # cross-language vectors
cargo run --example keychain_probe                 # Touch ID capability
cargo run --example setup_probe -- /tmp/scratch    # first-run bootstrap, headless
```

`setup_probe` runs the real `setup::bootstrap` against a throwaway directory, so
the first-run path can be verified without destroying a working installation.
Add `--full` to include the multi-gigabyte engine install.
