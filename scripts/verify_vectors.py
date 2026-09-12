#!/usr/bin/env python3
"""Check the Python vault implementation against vectors emitted by Rust.

The Rust host and the Python engine both implement the MSV1 vault format, and
nothing else in the suite notices when they drift. Drift here is the one
failure class that cannot be fixed forward: the bytes are already on disk, so
a host that seals differently than the engine opens leaves a user's media
unopenable. `src-tauri/examples/vault_vectors.rs` is the reference side and
this script is the comparison.

Both directions are checked, because the engine needs both. Sealing is what
the worker does with every image it produces. Opening is what it does with
every img2img, edit, inpaint, upscale and video first-frame input it is
handed, and only Rust wrote those bytes. The Python round trip in
test_worker.py cannot stand in for that: it seals with the hardcoded
CHUNK_SIZE and opens with whatever the header says, so a header misparse
round-trips cleanly there while failing on every blob the host wrote.

Run with no arguments to generate the vectors and check them:

    python3 scripts/verify_vectors.py

Or pass a file already produced by `cargo run --example vault_vectors`:

    python3 scripts/verify_vectors.py vectors.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent


def generate_vectors() -> str:
    """Ask the Rust reference implementation for the vectors."""
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "--example", "vault_vectors"],
        cwd=ROOT / "src-tauri",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"vault_vectors failed with exit code {proc.returncode}")
    return proc.stdout


def plaintext(length: int) -> bytes:
    """The same filler the Rust example seals, so it need not ship the input."""
    return bytes(i % 251 for i in range(length))


def check(document: dict[str, Any]) -> list[str]:
    sys.path.insert(0, str(ROOT / "engine"))
    import vaultcrypto as vc

    file_key = bytes.fromhex(document["file_key_hex"])
    file_id = bytes.fromhex(document["file_id_hex"])
    problems: list[str] = []

    for vector in document["vectors"]:
        length = vector["len"]
        expected = bytes.fromhex(vector["sealed_hex"])
        original = plaintext(length)

        if len(expected) != vector["sealed_len"]:
            problems.append(
                f"{length} bytes: vector claims {vector['sealed_len']} sealed bytes "
                f"but carries {len(expected)}"
            )

        sealed = vc.seal_with_file_key(file_key, file_id, original)
        if sealed != expected:
            problems.append(
                f"{length} bytes: Python sealed {len(sealed)} bytes that differ "
                f"from the {len(expected)} Rust sealed"
            )

        # Opened from the Rust bytes, never from `sealed`, so that a Python
        # header misparse cannot cancel itself out. Every exception is caught
        # rather than let out: a refused header and a failed tag are both drift,
        # and reporting the rest of the vectors is more useful than a traceback
        # from the first one.
        try:
            opened = vc.open_with_file_key(file_key, expected)
        except Exception as err:
            problems.append(
                f"{length} bytes: Python could not open what Rust sealed: "
                f"{type(err).__name__}: {err}"
            )
            continue
        if opened != original:
            problems.append(
                f"{length} bytes: Python opened Rust's bytes as {len(opened)} bytes "
                f"of different plaintext"
            )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "vectors",
        nargs="?",
        help="JSON from `cargo run --example vault_vectors`; generated if omitted",
    )
    args = parser.parse_args()

    raw = (
        pathlib.Path(args.vectors).read_text() if args.vectors else generate_vectors()
    )
    document = json.loads(raw)

    problems = check(document)
    for problem in problems:
        print(f"MISMATCH {problem}")
    if problems:
        print(
            f"vault format drift: {len(problems)} problem(s) across "
            f"{len(document['vectors'])} vectors"
        )
        return 1

    print(
        f"{len(document['vectors'])} vectors byte-identical, "
        "and every one opened back from Rust's bytes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
