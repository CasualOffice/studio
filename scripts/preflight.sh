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

step "engine tests, with numpy blocked exactly as CI has it"
if "$VENV" - <<'PY'
import sys, builtins, unittest
real = builtins.__import__
def guard(n, *a, **k):
    if n.split(".")[0] == "numpy":
        raise ModuleNotFoundError("No module named 'numpy'")
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

step "tsc --noEmit"
npx tsc --noEmit -p tsconfig.json && ok || bad "typescript"

step "vitest"
npx vitest run >/dev/null 2>&1 && ok || bad "vitest"

step "vite build"
npx vite build >/dev/null 2>&1 && ok || bad "frontend build"

step "no volumes left mounted (a stale mount breaks bundle_dmg.sh)"
if ls -d /Volumes/Model* >/dev/null 2>&1; then bad "a Model Studio volume is still mounted"; else ok; fi

printf '\n'
if [ $fail -eq 0 ]; then echo "preflight clean"; else echo "preflight FAILED - do not push"; fi
exit $fail
