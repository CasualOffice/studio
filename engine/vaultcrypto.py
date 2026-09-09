"""Vault file format, engine side.

Byte-compatible with `src-tauri/src/vault/crypto.rs`. The two implementations
must stay in lockstep; the interop test in that module is the contract.

The worker is handed a per-file key and file id, never the vault's data key,
so a compromised engine process can seal the one artifact it just produced
and read nothing that already exists.
"""

from __future__ import annotations

import struct
from typing import Any

MAGIC = b"MSV1"
VERSION = 1
ALG_CHACHA20POLY1305 = 1
HEADER_LEN = 28
CHUNK_SIZE = 256 * 1024
TAG_LEN = 16


def build_header(file_id: bytes, chunk_size: int = CHUNK_SIZE) -> bytes:
    if len(file_id) != 16:
        raise ValueError("file_id must be 16 bytes")
    return (
        MAGIC
        + bytes([VERSION, ALG_CHACHA20POLY1305])
        + b"\x00\x00"           # flags
        + file_id
        + struct.pack("<I", chunk_size)
    )


def _nonce(counter: int) -> bytes:
    # 12-byte IETF nonce: four zero bytes then the little-endian counter.
    return b"\x00\x00\x00\x00" + struct.pack("<Q", counter)


def _aad(header: bytes, counter: int, final_chunk: bool) -> bytes:
    return header + struct.pack("<Q", counter) + bytes([1 if final_chunk else 0])


def seal_with_file_key(file_key: bytes, file_id: bytes, plaintext: bytes) -> bytes:
    """Seal `plaintext` exactly as the Rust host would."""
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    if len(file_key) != 32:
        raise ValueError("file key must be 32 bytes")

    header = build_header(file_id)
    aead = ChaCha20Poly1305(file_key)

    n_chunks = max(1, -(-len(plaintext) // CHUNK_SIZE))
    out = bytearray(header)
    for i in range(n_chunks):
        chunk = plaintext[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        out += aead.encrypt(_nonce(i), chunk, _aad(header, i, i == n_chunks - 1))
    return bytes(out)


def open_with_file_key(file_key: bytes, data: bytes) -> bytes:
    """Inverse of `seal_with_file_key`, for round-trip checks."""
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    if len(data) < HEADER_LEN or data[:4] != MAGIC:
        raise ValueError("not a vault file")
    if data[4] != VERSION:
        raise ValueError(f"unsupported version {data[4]}")
    if data[5] != ALG_CHACHA20POLY1305:
        raise ValueError(f"unsupported cipher {data[5]}")

    header = data[:HEADER_LEN]
    chunk_size = struct.unpack("<I", data[24:28])[0]
    sealed_chunk = chunk_size + TAG_LEN
    body = data[HEADER_LEN:]
    aead = ChaCha20Poly1305(file_key)

    n_chunks = max(1, -(-len(body) // sealed_chunk))
    out = bytearray()
    for i in range(n_chunks):
        blob = body[i * sealed_chunk : (i + 1) * sealed_chunk]
        out += aead.decrypt(_nonce(i), blob, _aad(header, i, i == n_chunks - 1))
    return bytes(out)


def write_sealed(path: str, file_key: bytes, file_id: bytes, plaintext: bytes) -> int:
    """Write a sealed blob, leaving no partial file behind on failure."""
    import os
    import tempfile

    sealed = seal_with_file_key(file_key, file_id, plaintext)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(sealed)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return len(sealed)


def artifact_to_png_bytes(artifact: Any) -> bytes:
    """Get PNG bytes out of an mlx-gen artifact without touching the disk.

    `generate_outputs` returns `GeneratedOutput`, whose `.artifact` is a
    `GeneratedImage`, whose `.image` is the PIL image. Both layers are unwrapped
    here, and the attribute names are probed rather than assumed because they
    have moved between releases.

    There is deliberately no save-to-disk fallback: writing plaintext would
    defeat the point of the vault, so an unrecognised artifact is an error.
    """
    import io

    from PIL import Image

    seen: list[str] = []
    node = artifact
    for _ in range(4):  # bounded: guards against a self-referential wrapper
        if isinstance(node, Image.Image):
            buf = io.BytesIO()
            node.save(buf, format="PNG")
            return buf.getvalue()

        seen.append(type(node).__name__)
        for attr in ("image", "artifact", "_image", "pil_image", "img"):
            child = getattr(node, attr, None)
            if child is not None and child is not node:
                node = child
                break
        else:
            break

    raise TypeError(
        "could not find a PIL image on this artifact "
        f"(unwrapped: {' -> '.join(seen)}); refusing to write plaintext to disk"
    )
