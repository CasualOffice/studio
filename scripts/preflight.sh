#!/usr/bin/env bash
# Everything CI runs, run the way CI runs it, before anything is pushed.
#
# Every red build in this repository so far came from checking a proxy
# instead of the thing itself: `cargo clippy --lib` instead of --all-targets,
# an interpreter that happened to have numpy, a deploy workflow that reported
# success while publishing nothing. This runs the real checks so that gap
# cannot open again.
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
step() { printf '\n== %s\n' "$1"; }
ok()   { printf '   ok\n'; }
bad()  { printf '   FAILED: %s\n' "$1"; fail=1; }

VENV="$HOME/Library/Application Support/com.melp.modelstudio/runtime/venv/bin/python"
[ -x "$VENV" ] || VENV=python3

step "engine tests, with CI's exact dependency set"
if "$VENV" - <<'PY'
import sys, builtins, unittest

# CI installs pillow, pillow-heif and cryptography and nothing else, so a test
# that reaches for anything beyond them passes here and fails there. This
# blocked numpy alone, which is how four tests importing huggingface_hub went
# green locally and red on push -- the gate said "exactly as CI has it" while
# checking one module out of a dozen.
#
# The list is what the venv has that CI does not, so it has to be kept beside
# the workflow's install line. A module missing from both is already absent and
# needs no help.
ABSENT = {
    "numpy", "huggingface_hub", "mlx", "mlx_lm", "mlx_vlm", "mlxgen", "mflux",
    "torch", "transformers", "safetensors", "arabic_reshaper", "bidi",
    "tokenizers", "sentencepiece", "scipy", "cv2",
}
real = builtins.__import__
def guard(n, *a, **k):
    root = n.split(".")[0]
    # A test that puts a stub in sys.modules has supplied the module itself and
    # works in CI too. Refusing those manufactures failures CI would not have,
    # which is its own way of making the gate untrustworthy.
    if root in ABSENT and root not in sys.modules:
        raise ModuleNotFoundError(f"No module named {root!r}")
    return real(n, *a, **k)
builtins.__import__ = guard
sys.path.insert(0, "engine")
res = unittest.TextTestRunner(verbosity=0).run(unittest.TestLoader().discover("engine"))
print(f"   {res.testsRun} tests, {len(res.failures)} failures, {len(res.errors)} errors")
sys.exit(1 if (res.failures or res.errors) else 0)
PY
then ok; else bad "engine tests"; fi

step "cargo fmt --check"
(cd src-tauri && cargo fmt --check) && ok || bad "formatting"

step "cargo clippy --all-targets -- -D warnings   (examples included)"
(cd src-tauri && cargo clippy --all-targets -- -D warnings >/dev/null 2>&1) && ok || bad "clippy"

step "cargo test"
(cd src-tauri && cargo test >/dev/null 2>&1) && ok || bad "rust tests"

# The probes compile under clippy --all-targets and used to run nowhere, which
# is how vault_lifecycle came to panic on its third assertion without anything
# noticing. They are the only executing coverage of the operations that can
# lose a user's media.
step "example probes (vault lifecycle, storage move)"
if (cd src-tauri \
    && cargo run --quiet --example vault_lifecycle >/dev/null 2>&1 \
    && cargo run --quiet --example storage_move >/dev/null 2>&1); then
  ok
else
  bad "example probes"
fi

# The Rust host and the Python engine both implement the vault format, and a
# disagreement between them is the only defect in this codebase that cannot be
# fixed forward -- the bytes are already sealed on disk. CI has checked this
# all along; the check lived as a heredoc inside the workflow, so the script
# that claims on its first line to run everything CI runs did not run it.
step "vault format interop (Python must seal and open what Rust wrote)"
if out=$("$VENV" scripts/verify_vectors.py); then
  echo "$out" | sed 's/^/   /'
  ok
else
  echo "$out" | sed 's/^/   /'
  bad "vault format interop"
fi

step "tsc --noEmit"
npx tsc --noEmit -p tsconfig.json && ok || bad "typescript"

step "vitest"
npx vitest run >/dev/null 2>&1 && ok || bad "vitest"

step "vite build"
npx vite build >/dev/null 2>&1 && ok || bad "frontend build"

step "no volumes left mounted (a stale mount breaks bundle_dmg.sh)"
# Two kinds, and this only looked for one. A finished DMG mounts as
# /Volumes/Model Studio; an interrupted build leaves its read-write staging
# image mounted as /Volumes/dmg.XXXXXX instead, which is what actually
# accumulates -- three of them were sitting there from earlier builds while
# this check reported clean.
stale=""
ls -d /Volumes/Model* >/dev/null 2>&1 && stale="a Model Studio volume"
if hdiutil info 2>/dev/null | grep -q "/Volumes/dmg\."; then
  stale="${stale:+$stale and }a DMG staging volume"
fi
if [ -n "$stale" ]; then
  bad "$stale is still mounted -- detach it with: hdiutil info | grep /Volumes/dmg"
else
  ok
fi

printf '\n'
if [ $fail -eq 0 ]; then echo "preflight clean"; else echo "preflight FAILED - do not push"; fi
exit $fail
