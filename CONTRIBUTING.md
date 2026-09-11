# Contributing

## What you need

An Apple Silicon Mac. This is not a portability oversight — MLX has no other
backend and the vault's key handling is macOS Keychain code, so a Linux or
Intel build could not work even in principle.

Rust (stable), Node 22, and Python 3.12.

```sh
npm install
npm run tauri dev
```

The first launch provisions a private Python runtime under
`~/Library/Application Support/com.melp.modelstudio`. Nothing is installed
system-wide.

## Running the tests

```sh
npm test                       # frontend
cd src-tauri && cargo test     # rust
python engine/test_worker.py   # engine
```

CI runs all three plus `cargo fmt --check`, clippy with warnings denied, and
a check that the Rust and Python vault implementations produce byte-identical
output. Run `cargo fmt` before pushing; a formatting difference fails the
build in eight seconds and is a dull way to find out.

The engine tests deliberately avoid loading weights. If a test needs a model,
it is testing the wrong thing — extract the decision and test that.

## What is worth knowing before changing things

**One model is resident at a time.** This is an invariant, not an
optimisation: every model here is measured in gigabytes and two at once on a
16 GB machine means paging, which does not fail loudly, it just makes the
whole Mac crawl. Anything that loads must evict first.

**Capability is asked, not assumed.** Models disagree about what a picture
means — a first frame, a subject reference, or nothing at all — and guessing
produces output that is plausible and wrong. Ask the engine what a model
supports, send exactly that, and refuse when it supports none of it. Refusing
is better than silently ignoring what the user gave you.

**Plaintext is a temporary state.** Anything decrypted lives in the staging
directory and is removed when the job ends, including when it fails. If you
add a path that writes a result, seal it before it touches disk.

**Measure before claiming.** `docs/measurements.md` holds numbers taken on
real hardware. Several long-standing beliefs in this codebase turned out to
be wrong when finally measured, including one that disabled a working feature
for weeks. Add to that file rather than arguing from intuition.

## Commits

Explain why the change exists, not what the diff shows. If it fixes
something, say what the broken behaviour was — the next person reading it is
usually trying to work out whether their problem is the same one.

## Licence

Contributions are accepted under GPL-3.0-or-later, matching the project.
