"""
Model Studio engine worker.

A long-lived process that speaks JSON Lines over stdin/stdout so the Tauri/Rust
host can keep an MLX model resident between generations. Reloading a 4-8 GiB
quantized package costs 10-30s, so a persistent worker is the difference between
a usable app and an unusable one.

Protocol
--------
Host -> worker (one JSON object per line on stdin):
    {"id": "<req-id>", "op": "<op>", ...params}

Worker -> host (one JSON object per line on stdout):
    {"id": "<req-id>", "type": "progress"|"result"|"error"|"log"|"ready", ...}

Every request terminates in exactly one "result" or "error" event.
Anything the underlying libraries print goes to stderr, never stdout, so the
stdout stream stays a clean JSON Lines channel.
"""

from __future__ import annotations

import contextlib
import gc
import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Keep third-party chatter off stdout: stdout is the protocol channel.
_STDOUT = sys.stdout
sys.stdout = sys.stderr

_WRITE_LOCK = threading.Lock()

# Requests the host has asked us to cancel. Checked from progress callbacks.
_CANCELLED: set[str] = set()
_ACTIVE: set[str] = set()
_CANCEL_LOCK = threading.Lock()


class Cancelled(Exception):
    """Raised inside a progress callback to unwind an in-flight generation."""


def emit(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, separators=(",", ":"), default=str)
    with _WRITE_LOCK:
        _STDOUT.write(line + "\n")
        _STDOUT.flush()


def log(req_id: str, message: str, level: str = "info") -> None:
    emit({"id": req_id, "type": "log", "level": level, "message": message})


def is_cancelled(req_id: str) -> bool:
    with _CANCEL_LOCK:
        return req_id in _CANCELLED


def clear_cancel(req_id: str) -> None:
    with _CANCEL_LOCK:
        _CANCELLED.discard(req_id)


# --------------------------------------------------------------------------
# Memory controls
# --------------------------------------------------------------------------

def apply_resource_policy() -> dict[str, Any]:
    """Bound what this process may take from the machine.

    Two separate limits:

    * The memory limit is a hard ceiling. Past it MLX raises instead of letting
      macOS swap, which is the difference between one failed generation and a
      Mac that stops responding for minutes.
    * The cache limit bounds only the free-buffer pool MLX keeps for reuse.

    Both are deliberately below physical memory: the window server, the app and
    Python's own working set all have to fit alongside.
    """
    import mlx.core as mx

    applied: dict[str, Any] = {}
    budget = os.environ.get("MODELSTUDIO_MEMORY_BUDGET_GIB")
    if budget:
        try:
            gib = float(budget)
            mx.set_memory_limit(int(gib * (1024 ** 3)))
            applied["memory_limit_gib"] = round(gib, 1)
            # A quarter of the budget for reusable buffers, capped at 2 GiB.
            cache = min(2.0, max(0.5, gib / 4))
            mx.set_cache_limit(int(cache * (1024 ** 3)))
            applied["cache_limit_gib"] = round(cache, 2)
        except Exception as exc:
            applied["error"] = str(exc)
    applied["threads"] = os.environ.get("OMP_NUM_THREADS", "default")
    return applied


def set_cache_limit(gb: float | None) -> dict[str, Any]:
    """Cap MLX's free-buffer cache.

    MLX-Gen treats a host-set limit at or below half of physical RAM as
    deliberate and leaves it alone, so this is the sanctioned way for an
    embedding app to choose the cap instead of the machine ladder
    (total RAM / 8, clamped to [1, 8] GiB).

    This bounds the *free* cache only. Resident weights and in-flight
    activations are unaffected -- a model whose weights exceed RAM does not
    become runnable by shrinking the cache.
    """
    import mlx.core as mx

    if gb is None:
        return {"cache_limit_gb": None}
    try:
        limit = int(gb * (1024 ** 3))
        prev = mx.set_cache_limit(limit)
        return {"cache_limit_gb": gb, "previous_bytes": prev}
    except Exception as exc:
        return {"cache_limit_gb": None, "error": str(exc)}


def _drop_named(opt: dict[str, Any], message: str) -> str | None:
    """Remove whichever optional the error message says is unsupported.

    The offending parameter has to be identified precisely. MLX-Gen's errors
    end with the full list of parameters the route *does* accept -- "'image_path'
    is not a parameter of ... Accepted: guidance, height, image_paths, ... width."
    -- so a naive substring search matches the accepted names and drops the
    wrong argument. Doing that here silently removed the image from an edit and
    surfaced later as an unrelated crash deep inside MLX.

    So: look only at the text before the accepted list, and prefer names the
    message actually quotes.
    """
    head = re.split(r"\bAccepted\b|\baccepts\b|\bAcceptable\b", message)[0]
    quoted = set(re.findall(r"['\"`]([A-Za-z_][A-Za-z0-9_-]*)['\"`]", message))

    def norm(x: str) -> str:
        return x.replace("-", "_")

    normalized_quoted = {norm(q) for q in quoted}
    head_norm = norm(head)

    def matches(candidate: str) -> bool:
        c = norm(candidate)
        # A quoted name is an explicit accusation; the head is the fallback.
        return c in normalized_quoted or c in head_norm

    for key, value in list(opt.items()):
        if matches(key):
            opt.pop(key)
            return key
        if isinstance(value, dict):
            for inner in list(value):
                if matches(inner):
                    value.pop(inner)
                    if not value:
                        opt.pop(key)
                    return f"{key}.{inner}"
    return None


def call_tolerant(req_id: str, fn: Any, base: dict[str, Any],
                  optional: dict[str, Any]) -> Any:
    """Call `fn`, dropping optional kwargs the route refuses.

    MLX-Gen validates generation keywords *before* loading weights and names
    the offending parameter in the error, so probing costs milliseconds rather
    than a wasted model load. Memory options like `low_ram` are universal
    today but per-route tomorrow; this keeps us working either way.
    """
    opt = dict(optional)
    while True:
        try:
            return fn(**base, **opt)
        except (TypeError, ValueError) as exc:
            message = str(exc)
            # MLX-Gen validates generation keywords itself and raises ValueError
            # naming the parameter ("'x' is not a parameter of ...", or a value
            # constraint such as "guidance > 1.0 is only supported for ...").
            # Only a TypeError is guaranteed to be a signature mismatch, so for
            # ValueError require that the message actually names an optional we
            # can drop -- otherwise it is a real error and must surface.
            dropped = _drop_named(opt, message)
            if dropped is None:
                raise
            log(req_id, f"route rejected {dropped!r} ({message.strip()[:120]}); "
                        "continuing without it", "warn")


# --------------------------------------------------------------------------
# Resident model cache
# --------------------------------------------------------------------------

class ModelCache:
    """Holds at most one loaded model. 16 GB Macs cannot afford two."""

    def __init__(self) -> None:
        self.key: str | None = None
        self.loaded: Any = None
        self.label: str | None = None

    def unload(self) -> None:
        if self.loaded is None:
            return
        self.loaded = None
        self.key = None
        self.label = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def get(self, key: str) -> Any | None:
        return self.loaded if self.key == key else None

    def put(self, key: str, loaded: Any, label: str) -> None:
        if self.key != key:
            self.unload()
        self.key = key
        self.loaded = loaded
        self.label = label


CACHE = ModelCache()

# The policy is applied once per process, at the first load.
_POLICY_APPLIED = False

# Drop resident weights after this long with no work. A model held for hours
# while the user does something else is several gigabytes the rest of the
# machine could be using. Reloading costs about ten seconds.
IDLE_UNLOAD_SECONDS = float(os.environ.get("MODELSTUDIO_IDLE_UNLOAD_SECONDS", "600"))

_LAST_USED = time.time()


def _touch() -> None:
    global _LAST_USED
    _LAST_USED = time.time()


def _idle_reaper() -> None:
    """Release resident models once the engine has been quiet long enough."""
    while True:
        time.sleep(15)
        if IDLE_UNLOAD_SECONDS <= 0:
            continue
        idle = time.time() - _LAST_USED
        if idle < IDLE_UNLOAD_SECONDS:
            continue
        with _CANCEL_LOCK:
            busy = bool(_ACTIVE)
        if busy:
            continue
        if CACHE.loaded is not None or _ASSIST.get("model") is not None:
            freed = CACHE.label or _ASSIST.get("key")
            CACHE.unload()
            _unload_assistant()
            emit({"id": "idle", "type": "log", "level": "info",
                  "message": f"released {freed} after {idle / 60:.0f} min idle"})
            _touch()



@contextlib.contextmanager
def _downloads_allowed(req_id: str):
    """Lift MLX-Gen's cache-only gate for the duration of a load.

    MLX-Gen refuses to fetch anything during generation and instead raises
    DownloadRequiredError. That policy exists so a host is never surprised by a
    multi-gigabyte transfer mid-run -- but as of mlx-gen 0.36.0 with
    huggingface_hub 1.x the probe (`snapshot_download(local_files_only=True)`)
    raises LocalEntryNotFoundError even when every file is present, so a fully
    downloaded model is rejected. Verified: loading the same model inside
    `allow_downloads()` transfers zero new bytes.

    Lifting it here is safe because this app enforces the same policy one level
    up: the Rust host refuses to start a job unless the model is already
    installed, and it checks the disk budget before any download. If upstream
    fixes the probe, this becomes a no-op.
    """
    try:
        from mflux.models.common.download_policy import allow_downloads
    except Exception:
        # Policy module moved or gone: nothing to lift.
        yield
        return
    with allow_downloads():
        yield


def _load_model(req_id: str, model: str, quantize: int | None,
                model_path: str | None, image_count: int,
                release_text_encoder: bool = False,
                **plan_kw: Any) -> tuple[Any, float]:
    """Resolve the route, reuse the resident model when the cache key matches."""
    from mlxgen import load_generation_model, resolve_generation_runtime

    runtime = resolve_generation_runtime(model=model, image_count=image_count, **plan_kw)
    key = str(runtime.cache_key(quantize=quantize, model_path=model_path))

    cached = CACHE.get(key)
    if cached is not None:
        log(req_id, f"reusing resident model ({CACHE.label})")
        return cached, 0.0

    # Bound the process before any weights are allocated.
    global _POLICY_APPLIED
    if not _POLICY_APPLIED:
        _POLICY_APPLIED = True
        log(req_id, f"resource policy: {apply_resource_policy()}")

    # Free the old model *before* pulling the new one into memory.
    CACHE.unload()
    log(req_id, f"loading {model} (quantize={quantize})")
    emit({"id": req_id, "type": "progress", "phase": "load", "progress": 0.0,
          "message": f"Loading {model}"})

    t0 = time.time()
    with _downloads_allowed(req_id):
        loaded = call_tolerant(
            req_id,
            load_generation_model,
            {
                "model": model,
                "quantize": quantize,
                "model_path": model_path,
                "image_count": image_count,
                **plan_kw,
            },
            # Release the text conditioner once the prompt is encoded. On routes
            # with a large conditioner this removes its resident size from peak.
            {"model_kwargs": {"release_text_encoder": True}} if release_text_encoder else {},
        )
    CACHE.put(key, loaded, model)
    load_ms = (time.time() - t0) * 1000.0
    log(req_id, f"loaded in {load_ms / 1000:.1f}s")
    return loaded, load_ms


def _progress_owner(obj: Any) -> Any:
    """Find the object that actually owns progress callbacks.

    `load_generation_model` returns a wrapper whose `.model` is the real
    runtime; only the inner object has `.callbacks`. Reaching for
    `wrapper.callbacks` raises AttributeError, and swallowing that silently
    disabled both progress reporting and cancellation -- the UI showed a
    progress bar that never moved and a Cancel button that did nothing.
    """
    if hasattr(obj, "callbacks"):
        return obj
    inner = getattr(obj, "model", None)
    if inner is not None and hasattr(inner, "callbacks"):
        return inner
    return None


def _make_progress_handler(req_id: str):
    """Bridge mlx-gen ProgressEvents onto our JSON Lines channel.

    Raising from the handler is mlx-gen's documented cancellation path:
    progress callback exceptions propagate to the caller.
    """
    state = {"last": 0.0}

    def on_progress(event: Any) -> None:
        if is_cancelled(req_id):
            raise Cancelled()
        now = time.time()
        phase = getattr(event, "phase", None) or "denoise"
        progress = getattr(event, "progress", None)
        # Throttle denoise spam; always let phase transitions through.
        if phase == "denoise" and now - state["last"] < 0.08:
            return
        state["last"] = now
        emit({
            "id": req_id,
            "type": "progress",
            "phase": phase,
            "progress": float(progress) if progress is not None else None,
            "step": getattr(event, "step", None),
            "total_steps": getattr(event, "total_steps", None),
            "seed": getattr(event, "seed", None),
            "item_index": getattr(event, "item_index", None),
            "item_count": getattr(event, "item_count", None),
            "output_path": str(getattr(event, "output_path", "") or "") or None,
        })

    return on_progress


def _subscribe(req_id: str, obj: Any, task: str):
    """Attach a progress handler, returning an unsubscribe callable."""
    handler = _make_progress_handler(req_id)
    owner = _progress_owner(obj)
    if owner is None:
        log(req_id, "no progress callbacks on this runtime; "
                    "progress and cancellation are unavailable", "warn")
        return lambda: None
    try:
        return owner.callbacks.subscribe_progress(handler, task=task)
    except Exception as exc:
        log(req_id, f"could not subscribe to progress: {exc}", "warn")
        return lambda: None


# --------------------------------------------------------------------------
# Vault integration
# --------------------------------------------------------------------------

_STAGE_ROOT: str | None = None


def _stage_dir() -> str:
    """A private directory for briefly-decrypted source images."""
    global _STAGE_ROOT
    import tempfile

    if _STAGE_ROOT is None or not os.path.isdir(_STAGE_ROOT):
        _STAGE_ROOT = tempfile.mkdtemp(prefix="msvault-")
        os.chmod(_STAGE_ROOT, 0o700)
    return _STAGE_ROOT


def _stage_vault_inputs(inputs: list[dict[str, Any]]) -> list[str]:
    """Decrypt sealed source images so mlx-gen, which loads by path, can read them.

    Each entry carries only that blob's own file key, so the engine can open
    exactly the inputs this job was given and nothing else in the vault.
    """
    if not inputs:
        return []
    import vaultcrypto as vc

    staged: list[str] = []
    try:
        for entry in inputs:
            sealed = open(entry["path"], "rb").read()
            plain = vc.open_with_file_key(bytes.fromhex(entry["key"]), sealed)
            dest = os.path.join(_stage_dir(), f"{entry['id']}.png")
            fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(plain)
            staged.append(dest)
    except BaseException:
        _discard_staged(staged)
        raise
    return staged


def _discard_staged(paths: list[str]) -> None:
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def _seal_results(req_id: str, results: Any, slots: list[dict[str, Any]],
                  model: str, seeds: list[int]) -> dict[str, Any]:
    """Seal in-memory artifacts straight into the vault."""
    import vaultcrypto as vc

    if not isinstance(results, (list, tuple)):
        results = [results]
    if len(results) > len(slots):
        raise ValueError(
            f"engine produced {len(results)} images but only {len(slots)} vault "
            "slots were reserved"
        )

    ids: list[str] = []
    sizes: list[int] = []
    for artifact, slot in zip(results, slots):
        png = vc.artifact_to_png_bytes(artifact)
        vc.write_sealed(
            slot["path"], bytes.fromhex(slot["key"]), bytes.fromhex(slot["file_id"]), png
        )
        ids.append(slot["id"])
        sizes.append(len(png))
        log(req_id, f"sealed {len(png)} bytes into vault slot {slot['id']}")

    return {"outputs": ids, "sealed": True, "sizes": sizes,
            "model": model, "seeds": seeds}



# --------------------------------------------------------------------------
# Prompt assistance
# --------------------------------------------------------------------------

# The assistant is small enough (~1.5 GiB at 4-bit) to stay resident beside a
# 4-5 GiB image model, so it gets its own slot rather than evicting the
# generator every time someone asks for help with a prompt.
_ASSIST: dict[str, Any] = {"key": None, "model": None, "processor": None, "config": None}


def _load_assistant(req_id: str, repo: str) -> tuple[Any, Any, Any]:
    if _ASSIST["key"] == repo and _ASSIST["model"] is not None:
        return _ASSIST["model"], _ASSIST["processor"], _ASSIST["config"]

    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    emit({"id": req_id, "type": "progress", "phase": "load", "progress": 0.0,
          "message": "Loading the prompt assistant"})
    model, processor = load(repo)
    config = load_config(repo)
    _ASSIST.update(key=repo, model=model, processor=processor, config=config)
    log(req_id, f"assistant {repo} ready")
    return model, processor, config


def _unload_assistant() -> None:
    _ASSIST.update(key=None, model=None, processor=None, config=None)
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


GENERATE_SYSTEM = (
    "Rewrite the user's idea as a single vivid prompt for an image generator.\n"
    "Rules:\n"
    "- Keep every subject, object and detail they mentioned.\n"
    "- Add concrete visuals: lighting, materials, colour, framing, mood.\n"
    "- Never add text, logos or watermarks.\n"
    "- Output only the prompt. No preamble, no quotes, under 60 words.\n\n"
    "Example\n"
    "User idea: a cat\n"
    "You write: a tabby cat curled on a windowsill in low afternoon sun, warm "
    "rim light through dusty glass, shallow depth of field, soft muted colours\n"
)

DESCRIBE_SYSTEM = (
    "Describe this picture for someone who cannot see it.\n"
    "Name the main subject and its colour and material, then the surface it "
    "sits on, the background, and the lighting.\n"
    "Be factual. Do not guess at anything you cannot see. Do not give opinions.\n"
    "One or two sentences, under 50 words."
)

EDIT_SYSTEM = (
    "You rewrite an edit request into a precise instruction for an image "
    "editor that changes one thing and leaves the rest alone.\n"
    "You are given a description of a picture and the change someone wants.\n"
    "Rules:\n"
    "- Their requested change is the ONLY change. Never replace it with a "
    "different edit, and never add edits they did not ask for.\n"
    "- Use the description to name the specific thing their request refers to.\n"
    "- Say what must stay unchanged.\n"
    "- Output only the instruction. No preamble, no quotes, under 40 words.\n\n"
    "Example\n"
    "Picture shows: a woman in a green jacket standing on a street\n"
    "User asks: make it red\n"
    "You write: change the woman's green jacket to red, keeping her pose, face, "
    "and the street background exactly as they are\n"
)

# Words that carry no intent, so their presence proves nothing about whether a
# rewrite kept the user's meaning.
_STOPWORDS = {
    "a", "an", "the", "it", "its", "this", "that", "make", "change", "to", "into",
    "of", "and", "or", "is", "are", "be", "please", "with", "for", "in", "on",
    "my", "me", "i", "want", "add", "more", "less", "very", "some", "all",
}


def _keeps_intent(original: str, rewritten: str) -> bool:
    """Reject a rewrite that dropped everything the user actually asked for.

    A small model handed an image will sometimes describe the picture instead
    of following the instruction -- turning "make it blue" into "remove the
    background". Silently doing a different edit is worse than doing none, so
    a rewrite must carry over at least one meaningful word.
    """
    def words(text: str) -> set[str]:
        cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
        return {w for w in cleaned.split() if len(w) > 2 and w not in _STOPWORDS}

    wanted = words(original)
    if not wanted:
        return True
    got = words(rewritten)
    # Prefixes catch simple inflections: "blue" vs "bluish", "cat" vs "cats".
    return any(any(w.startswith(g[:4]) or g.startswith(w[:4]) for g in got) for w in wanted)


def _clean_assist(text: str, fallback: str) -> str:
    """Trim the chatter models add around a rewritten prompt."""
    out = (text or "").strip()
    for marker in ("Prompt:", "prompt:", "Instruction:", "instruction:",
                   "You write:", "Answer:", "Output:"):
        if out.startswith(marker):
            out = out[len(marker):].strip()
    # Models often wrap the whole answer in quotes.
    if len(out) > 1 and out[0] in "\"'" and out[-1] == out[0]:
        out = out[1:-1].strip()
    for sep in ("\n\n", "\nNote:", "\nThis "):
        if sep in out:
            out = out.split(sep)[0].strip()
    return out or fallback


# Longest edge handed to the vision model. Qwen2-VL uses dynamic resolution,
# so cost scales with pixel count and does so brutally: measured on an M4,
# 384px took 1.5s, 1024px 7.6s and 2048px 58.4s. A phone photo would stall for
# minutes. 512px is ample for naming a subject and its surroundings.
ASSIST_MAX_EDGE = 512


def _downscale_for_assist(paths: list[str]) -> list[str]:
    """Shrink staged images before the vision model sees them."""
    if not paths:
        return []
    import io

    from PIL import Image

    out = []
    for p in paths:
        try:
            img = Image.open(p)
            if max(img.size) <= ASSIST_MAX_EDGE:
                out.append(p)
                continue
            scale = ASSIST_MAX_EDGE / max(img.size)
            small = img.convert("RGB").resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.LANCZOS,
            )
            dest = f"{p}.small.png"
            small.save(dest, format="PNG")
            out.append(dest)
        except Exception:
            # A resize failure should not block the rewrite entirely.
            out.append(p)
    return out


def op_assist(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a prompt, optionally looking at the image being edited."""
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    repo = req.get("assistant") or "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    user_prompt = (req.get("prompt") or "").strip()
    if not user_prompt:
        raise ValueError("write something first, then ask for help improving it")

    mode = req.get("mode", "generate")
    raw_staged = _stage_vault_inputs(req.get("vault_inputs") or [])
    staged = _downscale_for_assist(raw_staged)
    try:
        if req.get("exclusive"):
            # On a 16 GB machine the image model and the assistant together sit
            # right at the ceiling, and the resulting paging makes everything
            # slow. Give up the image model rather than thrash.
            if CACHE.loaded is not None:
                log(req_id, "releasing the image model to make room for the assistant")
                CACHE.unload()

        model, processor, config = _load_assistant(req_id, repo)

        editing = mode == "edit" and bool(staged)

        def ask(text: str, images: list[str], max_tokens: int, temperature: float) -> str:
            formatted = apply_chat_template(
                processor, config, text, num_images=len(images)
            )
            out = vlm_generate(
                model, processor, formatted, images or None,
                max_tokens=max_tokens, temperature=temperature, verbose=False,
            )
            return out if isinstance(out, str) else getattr(out, "text", str(out))

        description = ""
        if editing:
            # Stage one: look. Asking a 2B model to see and rewrite in one shot
            # made it describe the picture instead of following the request;
            # separating the two gives each step a single job.
            emit({"id": req_id, "type": "progress", "phase": "denoise",
                  "progress": None, "message": "Looking at your picture"})
            description = _clean_assist(ask(DESCRIBE_SYSTEM, staged, 90, 0.2), "")
            log(req_id, f"saw: {description[:110]}")

        emit({"id": req_id, "type": "progress", "phase": "denoise",
              "progress": None, "message": "Writing the prompt"})

        if editing:
            # Stage two is text-only: the description already carries the image.
            instruction = (
                f"{EDIT_SYSTEM}\nNow do the same here.\n"
                f"Picture shows: {description}\n"
                f"User asks: {user_prompt}\nYou write:"
            )
            text = ask(instruction, [], 110, 0.2)
        else:
            instruction = (
                f"{GENERATE_SYSTEM}\nNow do the same here.\n"
                f"User idea: {user_prompt}\nYou write:"
            )
            text = ask(instruction, [], 130, 0.4)

        improved = _clean_assist(text, user_prompt)

        if not _keeps_intent(user_prompt, improved):
            log(req_id, f"discarding rewrite {improved!r}: it lost the request", "warn")
            return {"prompt": user_prompt, "original": user_prompt,
                    "saw_image": bool(staged), "rejected": improved,
                    "description": description}
    finally:
        _discard_staged(staged)
        _discard_staged(raw_staged)

    return {"prompt": improved, "original": user_prompt,
            "saw_image": bool(staged), "description": description}


def op_unload_assistant(req_id: str, _req: dict[str, Any]) -> dict[str, Any]:
    _unload_assistant()
    return {"unloaded": True}


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------

def op_ping(req_id: str, _req: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    import mlxgen

    active = peak = None
    try:
        active = mx.get_active_memory()
        peak = mx.get_peak_memory()
    except Exception:
        pass
    try:
        from importlib.metadata import version as _dist_version

        mlxgen_version = _dist_version("mlx-gen")
    except Exception:
        # The package exposes no __version__ attribute; fall back rather than
        # reporting a misleading "unknown".
        mlxgen_version = getattr(mlxgen, "__version__", "unknown")

    return {
        "python": sys.version.split()[0],
        "mlxgen": mlxgen_version,
        "mlx": getattr(mx, "__version__", "unknown"),
        "resident_model": CACHE.label,
        "threads": os.environ.get("OMP_NUM_THREADS", "default"),
        "memory_budget_gib": os.environ.get("MODELSTUDIO_MEMORY_BUDGET_GIB"),
        "active_bytes": active,
        "peak_bytes": peak,
    }


def op_capabilities(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    from mlxgen import get_model_capabilities

    caps = get_model_capabilities(model=req["model"])
    modes = [c.mode for c in caps.capabilities]
    return {"model": req["model"], "modes": modes}


def op_resolve(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Inspect a Hugging Face repo the built-in catalog does not know about.

    Answers the two questions the UI needs before offering a download: can
    MLX-Gen route this model at all, and how much disk will it cost. Neither
    needs the weights, so this is cheap and safe to run on a typed-in id.
    """
    from mlxgen import get_model_capabilities

    repo = req["model"].strip()

    modes: list[str] = []
    tasks: list[str] = []
    route_error: str | None = None
    try:
        caps = get_model_capabilities(model=repo)
        modes = [c.mode for c in caps.capabilities]
        # Map MLX-Gen's internal modes onto the app's four task buckets.
        if any(m in ("text-only", "text-to-image") for m in modes):
            tasks.append("text_to_image")
        if any(m in ("edit-reference", "multi-reference", "latent-img2img") for m in modes):
            tasks.append("edit")
        if any("video" in m for m in modes):
            tasks.append("video")
    except Exception as exc:
        route_error = str(exc)

    size = 0
    private = False
    gated = False
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        # Count only what MLX-Gen would actually fetch: weights and tokenizers,
        # not the repo's images, ONNX exports or duplicate formats.
        for sib in info.siblings or []:
            name = sib.rfilename
            if name.endswith((".safetensors", ".json", ".model", ".txt", ".bin")):
                size += getattr(sib, "size", None) or 0
        private = bool(getattr(info, "private", False))
        gated = bool(getattr(info, "gated", False))
    except Exception as exc:
        if route_error is None:
            route_error = str(exc)

    return {
        "model": repo,
        "modes": modes,
        "tasks": tasks,
        "bytes": size,
        "private": private,
        "gated": gated,
        "routable": bool(tasks),
        "error": route_error,
    }


def _dir_size(path: Path) -> int:
    """Bytes actually occupied, counting each blob once.

    The Hugging Face cache keeps real content in `blobs/` and symlinks it into
    `snapshots/`. `os.path.getsize` follows symlinks, so a naive walk reports
    exactly double. `lstat` on the link reports the link itself, which is what
    we want to ignore.
    """
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            full = os.path.join(root, name)
            with contextlib.suppress(OSError):
                st = os.lstat(full)
                if not os.path.islink(full):
                    total += st.st_size
    return total


def op_download(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Fetch a model through MLX-Gen's own downloader.

    Shelling out to `mlxgen download` is the documented integration path and,
    more importantly, it applies MLX-Gen's weight/tokenizer patterns. A plain
    `snapshot_download` pulls the whole repository instead -- for Bonsai that
    was 29 files against MLX-Gen's 25 -- which would quietly spend more disk
    than the catalog promises.

    Progress comes from watching the cache directory grow, which works whoever
    is doing the writing.
    """
    repo_id = req["model"]
    expected = int(req.get("expected_bytes") or 0)
    root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    repo_dir = root / ("models--" + repo_id.replace("/", "--"))

    if not expected:
        # Fall back to the repository total; it over-counts when MLX-Gen fetches
        # a subset, so it is only a denominator of last resort.
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(repo_id, files_metadata=True)
            expected = sum(getattr(sib, "size", None) or 0 for sib in (info.siblings or []))
        except Exception as exc:
            log(req_id, f"could not read repo size: {exc}", "warn")

    already = _dir_size(repo_dir)
    emit({"id": req_id, "type": "progress", "phase": "download", "progress": 0.0,
          "total_bytes": expected, "done_bytes": already})

    stop = threading.Event()

    def poll() -> None:
        while not stop.wait(0.7):
            done = _dir_size(repo_dir)
            frac = (done / expected) if expected else None
            emit({
                "id": req_id, "type": "progress", "phase": "download",
                "progress": min(frac, 0.999) if frac is not None else None,
                "total_bytes": expected, "done_bytes": done,
            })

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()

    env = dict(os.environ)
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # Not every model belongs to MLX-Gen. The prompt assistant is a VLM loaded
    # by mlx-vlm, and `mlxgen download` exits 0 without fetching anything for a
    # repo it does not recognise -- which looked like a successful zero-byte
    # download. Route those straight through huggingface_hub instead.
    via = req.get("via") or "mlxgen"

    try:
        if via == "hf":
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo_id)
        else:
            cli = Path(sys.executable).parent / "mlxgen"
            cmd = [str(cli), "download", "--model", repo_id] if cli.exists() else [
                sys.executable, "-m", "mlxgen", "download", "--model", repo_id
            ]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                raise RuntimeError("download failed:\n" + "\n".join(tail[-8:]))
    finally:
        stop.set()
        poller.join(timeout=2)

    size = _dir_size(repo_dir)
    # A downloader that reports success while fetching nothing must not be
    # recorded as an installed model.
    if size < 1024 * 1024:
        raise RuntimeError(
            f"{repo_id} reported success but nothing was written to the cache. "
            "The repository may not exist, be gated, or need a different downloader."
        )
    emit({"id": req_id, "type": "progress", "phase": "download", "progress": 1.0,
          "total_bytes": expected or size, "done_bytes": size})
    return {"model": repo_id, "path": str(repo_dir), "bytes": size}


# Every generate route takes these.
_REQUIRED_KW = (("prompt", "prompt"), ("steps", "num_inference_steps"))

# These vary by route. Qwen edit is guidance-distilled and takes no negative
# prompt; reference-pinned edit routes derive their own canvas and reject
# width/height; only latent img2img takes image_strength. Passing one to a
# route that does not want it is a TypeError, so they go through
# `call_tolerant` and are dropped individually rather than failing the run.
_OPTIONAL_KW = (
    ("negative_prompt", "negative_prompt"),
    ("guidance", "guidance"),
    ("image_strength", "image_strength"),
    ("width", "width"),
    ("height", "height"),
)


def _split_gen_kwargs(req: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    def collect(pairs):
        out: dict[str, Any] = {}
        for src, dst in pairs:
            val = req.get(src)
            if val is not None:
                out[dst] = val
        return out

    return collect(_REQUIRED_KW), collect(_OPTIONAL_KW)


def _run_generation(req_id: str, req: dict[str, Any], task: str,
                    image_count: int) -> dict[str, Any]:
    model = req["model"]
    quantize = req.get("quantize")
    model_path = req.get("model_path")
    output = req.get("output")
    seeds = req.get("seeds") or [req.get("seed", 0)]

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    low_ram = bool(req.get("low_ram"))
    # MLX-Gen refuses low-RAM mode together with several seeds; a serial
    # multi-seed run would also defeat the point of releasing between items.
    if low_ram and len(seeds) > 1:
        raise ValueError(
            "Low-RAM mode runs one image at a time. Set the image count to 1, "
            "or turn low-RAM mode off."
        )

    # `--low-ram` is a CLI flag; the Python API has no such keyword. What a
    # Python host can actually control is the MLX buffer-cache cap (the CLI's
    # low-RAM default is 1 GiB), releasing the text encoder after encoding, and
    # dropping the model once the run ends. The per-transformer-block cache
    # clearing the CLI does is not exposed here.
    cache_limit = req.get("cache_limit_gb")
    if cache_limit is None and low_ram:
        cache_limit = 1.0
    if cache_limit is not None:
        info = set_cache_limit(float(cache_limit))
        log(req_id, f"MLX cache limit: {info}")

    plan_kw: dict[str, Any] = {}
    if req.get("i2i_mode"):
        plan_kw["i2i_mode"] = req["i2i_mode"]

    loaded, load_ms = _load_model(
        req_id, model, quantize, model_path, image_count,
        release_text_encoder=bool(req.get("release_text_encoder")) or low_ram,
        **plan_kw,
    )

    gen_kw, optional_kw = _split_gen_kwargs(req)
    slots = req.get("vault_slots") or []

    # Source images live sealed in the vault. mlx-gen loads images by path, so
    # they are decrypted into a private 0700 directory for the duration of the
    # run and removed in the `finally` below. This is the one moment plaintext
    # exists on disk, and it is bounded by a single generation.
    staged = _stage_vault_inputs(req.get("vault_inputs") or [])
    try:
        return _generate_inner(
            req_id, req, task, loaded, gen_kw, optional_kw, slots, staged,
            seeds, model, output, low_ram, load_ms,
        )
    finally:
        _discard_staged(staged)


def _generate_inner(req_id, req, task, loaded, gen_kw, optional_kw, slots,
                    staged, seeds, model, output, low_ram,
                    load_ms: float = 0.0) -> dict[str, Any]:
    images = list(req.get("images") or []) + staged
    if images:
        # Routes disagree on the spelling: the edit route takes `image_paths`
        # (a list), others take `image_path`. Offer both and let the tolerance
        # layer drop whichever this route rejects -- it names the parameter, so
        # exactly one survives.
        optional_kw["image_paths"] = list(images)
        optional_kw["image_path"] = images[0] if len(images) == 1 else list(images)

    unsubscribe = _subscribe(req_id, loaded, task)
    base: dict[str, Any] = {
        "seeds": list(seeds),
        "progress_callback": _make_progress_handler(req_id),
        **gen_kw,
    }
    if slots:
        # Omit `output` so mlx-gen hands back in-memory artifacts. Nothing is
        # written until we have sealed it, so no plaintext ever reaches disk.
        pass
    else:
        base["output"] = output
        base["save_kwargs"] = {"export_json_metadata": True}

    gen_started = time.time()
    try:
        results = call_tolerant(
            req_id,
            loaded.generate_outputs,
            base,
            # Route-specific generation options ride along here so an
            # unsupported one is dropped rather than failing the whole run.
            optional_kw,
        )
    finally:
        unsubscribe()
        if low_ram:
            # Do not leave several GiB of weights resident after a run that
            # was requested specifically because memory is scarce.
            CACHE.unload()

    generate_ms = (time.time() - gen_started) * 1000.0

    if slots:
        out = _seal_results(req_id, results, slots, model, list(seeds))
        # Reported separately: load time is a fixed cost per session, not
        # something that scales with steps or canvas size. Folding it into a
        # per-step rate makes small or cold runs look catastrophically slow.
        out["load_ms"] = round(load_ms)
        out["generate_ms"] = round(generate_ms)
        return out

    paths = []
    for r in results:
        p = getattr(r, "saved_path", None)
        if p:
            paths.append(str(p))
    return {"outputs": paths, "model": model, "seeds": list(seeds),
            "load_ms": round(load_ms), "generate_ms": round(generate_ms)}


def op_generate(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    return _run_generation(req_id, req, "text-to-image", image_count=0)


def op_edit(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    # Sources normally arrive sealed as `vault_inputs`; `images` is only used
    # by direct callers with plaintext paths. Counting just `images` made every
    # vault-backed edit fail, and would have routed it to text-to-image even if
    # it had not, because image_count drives route selection.
    count = len(req.get("images") or []) + len(req.get("vault_inputs") or [])
    if count == 0:
        raise ValueError("edit requires at least one input image")
    req.setdefault("i2i_mode", "edit")
    return _run_generation(req_id, req, "image-to-image", image_count=count)


def op_upscale(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Restore or enlarge one image with SeedVR2.

    SeedVR2 sits outside MLX-Gen's unified planner, so it is driven directly.
    Like generation, the source arrives sealed and the result is sealed before
    it touches disk -- SeedVR2's own `save()` writes plaintext, so it is
    deliberately not used.
    """
    from mflux.utils.scale_factor import ScaleFactor
    from mlxgen.models.seedvr2 import SeedVR2

    slots = req.get("vault_slots") or []
    if not slots:
        raise ValueError("upscale requires a vault slot to write into")

    staged = _stage_vault_inputs(req.get("vault_inputs") or [])
    direct = list(req.get("images") or [])
    sources = direct + staged
    if not sources:
        raise ValueError("upscale requires an input image")

    try:
        cache_limit = req.get("cache_limit_gb")
        if cache_limit is None and req.get("low_ram"):
            cache_limit = 1.0
        if cache_limit is not None:
            log(req_id, f"MLX cache limit: {set_cache_limit(float(cache_limit))}")

        repo = req["model"]
        quantize = req.get("quantize")
        key = f"seedvr2::{repo}::{quantize}"
        model = CACHE.get(key)
        if model is None:
            CACHE.unload()
            emit({"id": req_id, "type": "progress", "phase": "load",
                  "progress": 0.0, "message": "Loading the upscaler"})
            with _downloads_allowed(req_id):
                model = SeedVR2(
                    quantize=quantize,
                    model_path=req.get("model_path") or repo,
                )
            CACHE.put(key, model, repo)

        raw = req.get("resolution", "2x")
        # `resolution` takes a ScaleFactor or an absolute shortest-edge pixel
        # count, never the "2x" string the UI works in.
        resolution: Any
        if isinstance(raw, str) and raw.strip().endswith("x"):
            resolution = ScaleFactor.parse(raw)
        else:
            resolution = int(raw)

        unsubscribe = _subscribe(req_id, model, "image-to-image")
        try:
            artifact = call_tolerant(
                req_id,
                model.generate_image,
                {
                    "seed": int(req.get("seed", 0)),
                    "image_path": sources[0],
                    "resolution": resolution,
                },
                {"low_ram": True} if req.get("low_ram") else {},
            )
        finally:
            unsubscribe()

        import vaultcrypto as vc

        png = vc.artifact_to_png_bytes(artifact)
        slot = slots[0]
        vc.write_sealed(
            slot["path"], bytes.fromhex(slot["key"]),
            bytes.fromhex(slot["file_id"]), png,
        )
        log(req_id, f"sealed {len(png)} bytes into vault slot {slot['id']}")
    finally:
        _discard_staged(staged)

    return {"outputs": [slot["id"]], "sealed": True,
            "sizes": [len(png)], "model": repo}


def op_unload(req_id: str, _req: dict[str, Any]) -> dict[str, Any]:
    CACHE.unload()
    _unload_assistant()
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except Exception:
        pass
    return {"unloaded": True}


def op_set_memory(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    return set_cache_limit(req.get("cache_limit_gb"))


OPS = {
    "ping": op_ping,
    "capabilities": op_capabilities,
    "resolve": op_resolve,
    "download": op_download,
    "generate": op_generate,
    "edit": op_edit,
    "upscale": op_upscale,
    "unload": op_unload,
    "assist": op_assist,
    "unload_assistant": op_unload_assistant,
    "set_memory": op_set_memory,
}


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def handle(req: dict[str, Any]) -> None:
    req_id = req.get("id", "?")
    op = req.get("op", "")
    with _CANCEL_LOCK:
        _ACTIVE.add(req_id)
    _touch()
    fn = OPS.get(op)
    if fn is None:
        emit({"id": req_id, "type": "error", "error": f"unknown op: {op}",
              "kind": "protocol"})
        return
    try:
        result = fn(req_id, req)
        emit({"id": req_id, "type": "result", "result": result})
    except Cancelled:
        emit({"id": req_id, "type": "error", "error": "cancelled", "kind": "cancelled"})
    except Exception as exc:
        kind = type(exc).__name__
        payload: dict[str, Any] = {
            "id": req_id, "type": "error", "error": str(exc), "kind": kind,
            "traceback": traceback.format_exc(),
        }
        # mlx-gen's cache-only policy raises DownloadRequiredError carrying the
        # exact remediation command; surface it so the UI can offer one click.
        for attr in ("download_command", "prepare_command"):
            val = getattr(exc, attr, None)
            if val:
                payload[attr] = str(val)
        emit(payload)
    finally:
        clear_cancel(req_id)
        _touch()
        with _CANCEL_LOCK:
            _ACTIVE.discard(req_id)


def main() -> None:
    emit({"id": "boot", "type": "ready", "pid": os.getpid()})

    # One worker thread: MLX generation is not safe to run concurrently, and a
    # 16 GB machine cannot hold two models anyway. Cancels jump the queue.
    work: "queue.Queue[dict[str, Any]]" = queue.Queue()

    def worker() -> None:
        while True:
            req = work.get()
            if req is None:
                return
            handle(req)

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    threading.Thread(target=_idle_reaper, daemon=True).start()

    def drain(timeout: float) -> None:
        # Let queued work finish rather than dropping it silently; the host is
        # waiting on a terminal event for every request it sent.
        work.put(None)
        worker_thread.join(timeout=timeout)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"id": "?", "type": "error", "error": f"bad json: {exc}",
                  "kind": "protocol"})
            continue

        op = req.get("op")
        if op in ("unload", "unload_assistant"):
            # Handled on the reader thread, not the work queue: "free memory"
            # is useless if it waits for the job whose memory you want back.
            # Cancel anything in flight first so nothing is unloaded from under
            # a running generation.
            with _CANCEL_LOCK:
                _CANCELLED.update(_ACTIVE)
            try:
                if op == "unload":
                    CACHE.unload()
                _unload_assistant()
                emit({"id": req.get("id", "?"), "type": "result",
                      "result": {"unloaded": True, "cancelled": sorted(_ACTIVE)}})
            except Exception as exc:
                emit({"id": req.get("id", "?"), "type": "error",
                      "error": str(exc), "kind": type(exc).__name__})
            continue
        if op == "cancel":
            target = req.get("target")
            if target:
                with _CANCEL_LOCK:
                    _CANCELLED.add(target)
            emit({"id": req.get("id", "?"), "type": "result",
                  "result": {"cancelling": target}})
            continue
        if op == "shutdown":
            drain(5.0)
            emit({"id": req.get("id", "?"), "type": "result", "result": {"bye": True}})
            return
        work.put(req)

    # stdin closed: the host is gone. Give in-flight work a moment to unwind.
    drain(5.0)


if __name__ == "__main__":
    main()
