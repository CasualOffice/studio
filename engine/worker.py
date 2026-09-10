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


def _register_image_formats() -> None:
    """Teach PIL the formats a Mac user actually has.

    Pillow reads PNG, JPEG, WebP, AVIF, TIFF and BMP on its own, but not HEIC —
    which is what every iPhone photo is. Registering the opener here means the
    rest of the engine can treat any supported file as just an image.
    """
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except Exception:
        # HEIC support is a nicety; everything else still works without it.
        pass


_register_image_formats()

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
    """Holds at most one loaded model.

    This is a hard invariant, not an optimisation. Every model this app can run
    is measured in gigabytes, and two resident at once on a 16 GB machine means
    paging — which does not fail loudly, it just makes the whole Mac crawl.
    Anything that needs a different model evicts the current one first.
    """

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


# The router families MLX-Gen knows.
ROUTER_FAMILIES = ("flux2", "qwen", "z-image", "ernie-image", "fibo", "bonsai", "wan")

# Architecture names a repository declares, mapped to the router that handles
# them. Matched against the pipeline or model class in the repo's own config,
# which is evidence rather than a guess.
_ARCHITECTURE_FAMILY = (
    ("flux2", "flux2"),
    ("fluxkontext", None),      # FLUX.1 lineage: not routable here at all
    ("fluxpipeline", None),
    ("fluxtransformer2d", None),
    ("qwenimage", "qwen"),
    ("zimage", "z-image"),
    ("z_image", "z-image"),
    ("ernie", "ernie-image"),
    ("fibo", "fibo"),
    ("bonsai", "bonsai"),
    ("wan", "wan"),
)


def _detect_family(repo: str) -> str | None:
    """Identify the router family from what the repository declares.

    Reads the pipeline class out of `model_index.json`, falling back to
    `config.json`. Returns None when there is no clear evidence, because a
    wrong family is worse than an honest refusal: the weights would load
    through an architecture they were not trained for.
    """
    import json as _json

    from huggingface_hub import hf_hub_download

    for filename in ("model_index.json", "config.json", "transformer/config.json"):
        try:
            path = hf_hub_download(repo_id=repo, filename=filename)
            with open(path) as fh:
                blob = _json.load(fh)
        except Exception:
            continue

        declared = " ".join(
            str(blob.get(k, "")) for k in ("_class_name", "architectures", "model_type")
        ).lower().replace("-", "").replace("_", "")

        for needle, family in _ARCHITECTURE_FAMILY:
            if needle.replace("_", "") in declared:
                return family
    return None


def _load_model(req_id: str, model: str, quantize: int | None,
                model_path: str | None, image_count: int,
                release_text_encoder: bool = False,
                loras: list[dict[str, Any]] | None = None,
                **plan_kw: Any) -> tuple[Any, float]:
    """Resolve the route, reuse the resident model when the cache key matches."""
    from mlxgen import load_generation_model, resolve_generation_runtime

    if loras:
        plan_kw["has_lora"] = True

    def resolve(**extra: Any) -> Any:
        return resolve_generation_runtime(
            model=model, image_count=image_count, **{**plan_kw, **extra}
        )

    try:
        runtime = resolve()
    except Exception as exc:
        # "could not infer a supported backend for model ... pass family=".
        # The message names the option but not the value, so try each family
        # rather than making the user work it out. A caller that already knows
        # the family never reaches this.
        if "family" not in plan_kw and "family" in str(exc).lower():
            detected = _detect_family(model)
            if detected is None:
                raise
            log(req_id, f"router family identified as {detected!r} from the model's config")
            runtime = resolve(family=detected)
            plan_kw["family"] = detected
        else:
            raise

    key = str(runtime.cache_key(quantize=quantize, model_path=model_path))

    # Adapters change the weights, so they have to change the identity. Reusing
    # a resident model across a LoRA change would silently apply the wrong
    # style, or none at all, with nothing in the output to indicate it.
    if loras:
        key += "::lora:" + ",".join(
            f"{l.get('path')}@{l.get('scale', 1.0)}" for l in loras
        )

    cached = CACHE.get(key)
    if cached is not None:
        log(req_id, f"reusing resident model ({CACHE.label})")
        return cached, 0.0

    # Bound the process before any weights are allocated.
    global _POLICY_APPLIED
    if not _POLICY_APPLIED:
        _POLICY_APPLIED = True
        log(req_id, f"resource policy: {apply_resource_policy()}")

    # Free whatever is resident *before* pulling the new one into memory --
    # including the assistant, which competes for the same budget.
    CACHE.unload()
    if _ASSIST.get("model") is not None:
        log(req_id, "releasing the prompt assistant to make room for the model")
        _unload_assistant()
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


# Formats a model's own image loader can be trusted with. Anything else is
# converted on the way in, so an iPhone HEIC or a WebP is usable everywhere.
_MODEL_SAFE_FORMATS = {"PNG", "JPEG"}


def _to_model_readable(path: str) -> str:
    """Convert a staged image to PNG unless it is already a safe format.

    The alternative -- teaching every model loader about HEIC and AVIF -- is not
    available: they open files themselves. Normalising here is lossless and
    happens once per run on an already-decrypted temporary file.
    """
    from PIL import Image

    try:
        with Image.open(path) as img:
            fmt = (img.format or "").upper()
            if fmt in _MODEL_SAFE_FORMATS:
                return path
            converted = f"{path}.png"
            # Drop alpha and exotic modes: model loaders expect RGB.
            img.convert("RGB").save(converted, format="PNG")
        return converted
    except Exception:
        # If it cannot be opened here it will fail later with a better message.
        return path


# Intermediates created while staging, keyed by the path actually handed to a
# model. Kept apart from the returned list because the two are different things:
# one is "files to feed the model", the other "files to delete afterwards".
# Conflating them meant a converted source was passed twice -- once as the
# original and once as the PNG -- so a single picture reached the model as two.
_STAGE_INTERMEDIATES: dict[str, list[str]] = {}


def _stage_vault_inputs(inputs: list[dict[str, Any]]) -> list[str]:
    """Decrypt sealed source images so mlx-gen, which loads by path, can read them.

    Each entry carries only that blob's own file key, so the engine can open
    exactly the inputs this job was given and nothing else in the vault.

    Returns exactly one path per input, in order.
    """
    if not inputs:
        return []
    import vaultcrypto as vc

    usable: list[str] = []
    try:
        for entry in inputs:
            with open(entry["path"], "rb") as fh:
                sealed = fh.read()
            plain = vc.open_with_file_key(bytes.fromhex(entry["key"]), sealed)
            # Keep the original extension so PIL can sniff the format.
            suffix = entry.get("ext") or "png"
            dest = os.path.join(_stage_dir(), f"{entry['id']}.{suffix}")
            fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(plain)

            readable = _to_model_readable(dest)
            if readable != dest:
                _STAGE_INTERMEDIATES.setdefault(readable, []).append(dest)
            usable.append(readable)
    except BaseException:
        _discard_staged(usable)
        raise
    return usable


def _discard_staged(paths: list[str]) -> None:
    """Remove staged files and anything created alongside them."""
    for p in paths:
        for victim in [p, *_STAGE_INTERMEDIATES.pop(p, [])]:
            try:
                os.unlink(victim)
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

# The assistant obeys the same one-model rule as everything else. It was
# briefly given its own slot on the theory that 2.6 GiB would sit happily
# beside a 5.6 GiB image model; on a 16 GB machine that combination paged, and
# a prompt rewrite that should take two seconds took minutes.
_ASSIST: dict[str, Any] = {"key": None, "model": None, "processor": None, "config": None}


def _load_assistant(req_id: str, repo: str) -> tuple[Any, Any, Any]:
    if _ASSIST["key"] == repo and _ASSIST["model"] is not None:
        return _ASSIST["model"], _ASSIST["processor"], _ASSIST["config"]

    # One model resident, always: give up the generator before taking memory.
    if CACHE.loaded is not None:
        log(req_id, f"releasing {CACHE.label} to make room for the assistant")
        CACHE.unload()

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


# For edits the model reports facts about the picture. It never writes the
# instruction: that is assembled from the user's words plus these facts.
SCENE_SYSTEM = (
    "Describe this picture as four short labelled lines and nothing else.\n"
    "Each value is a phrase of two to six words. No sentences. No opinions.\n"
    "If something is unclear, write: unknown\n\n"
    "SUBJECT: what the main thing is, with colour or material\n"
    "SURFACE: its texture or finish\n"
    "LIGHT: the lighting and its direction\n"
    "SETTING: what it sits on and what is behind it\n\n"
    "Example\n"
    "SUBJECT: a beige ceramic teapot\n"
    "SURFACE: smooth glazed stoneware\n"
    "LIGHT: soft daylight from the left\n"
    "SETTING: crumpled linen cloth, pale wall\n"
)

# For text-only ideas the model supplies extra detail, not a replacement.
DETAIL_SYSTEM = (
    "The user has an idea for a picture. Add visual detail to it.\n"
    "Rules:\n"
    "- Do NOT restate their idea. Write only what should be ADDED.\n"
    "- Lighting, materials, colour palette, camera framing, mood.\n"
    "- A few comma-separated phrases. Under 25 words. No full sentence.\n"
    "- Never add text, logos or watermarks.\n\n"
    "Example\n"
    "User idea: a cat\n"
    "You add: curled on a windowsill, low afternoon sun, warm rim light, "
    "shallow depth of field, muted colours\n"
)

# Words that carry no intent, so their presence proves nothing about whether a
# rewrite kept the user's meaning.
_STOPWORDS = {
    "a", "an", "the", "it", "its", "this", "that", "make", "change", "to", "into",
    "of", "and", "or", "is", "are", "be", "please", "with", "for", "in", "on",
    "my", "me", "i", "want", "add", "more", "less", "very", "some", "all",
}

# References that only make sense if you can see the picture.
_VAGUE = ("it", "this", "that", "them", "these", "those", "the image",
          "the picture", "the photo")


def _keeps_intent(original: str, rewritten: str) -> bool:
    """A rewrite that loses the request is worse than no rewrite."""
    def words(text: str) -> set[str]:
        cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
        return {w for w in cleaned.split() if len(w) > 2 and w not in _STOPWORDS}

    wanted = words(original)
    if not wanted:
        return True
    got = words(rewritten)
    return any(any(w.startswith(g[:4]) or g.startswith(w[:4]) for g in got) for w in wanted)


def _trim_to_sentence(text: str, max_words: int = 55) -> str:
    """Cut back to the last complete sentence.

    A token limit lands wherever it lands, so the model regularly stops
    mid-clause. Handing the generator "...warm rim light throug" is worse than
    handing it one shorter finished sentence, and the same limit also means the
    stated word count is routinely ignored.
    """
    text = text.strip()
    if not text:
        return text

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    if text.endswith((".", "!", "?")):
        return text
    cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if cut > 20:
        return text[: cut + 1]
    # No sentence break to fall back on: end at the last clause instead of
    # leaving a dangling half-word.
    comma = text.rfind(",")
    if comma > 20:
        return text[:comma] + "."
    return text.rstrip(" ,;:-") + "."


def _clean_assist(text: str, fallback: str) -> str:
    """Trim the chatter models add around a rewritten prompt."""
    out = (text or "").strip()
    for marker in ("Prompt:", "prompt:", "Instruction:", "instruction:",
                   "You write:", "Answer:", "Output:", "Subject:"):
        if out.startswith(marker):
            out = out[len(marker):].strip()
    if len(out) > 1 and out[0] in "\"'" and out[-1] == out[0]:
        out = out[1:-1].strip()
    for sep in ("\n\n", "\nNote:", "\nThis "):
        if sep in out:
            out = out.split(sep)[0].strip()
    return out or fallback


# Longest edge handed to the vision model. Qwen2-VL uses dynamic resolution,
# so cost scales with pixel count and does so brutally: measured on an M4,
# 384px took 1.5s, 1024px 7.6s and 2048px 58.4s. A phone photo would stall for
# minutes. 512px is ample for naming a subject.
ASSIST_MAX_EDGE = 512


def _downscale_for_assist(paths: list[str]) -> list[str]:
    """Shrink staged images before the vision model sees them."""
    if not paths:
        return []
    from PIL import Image

    out = []
    for p in paths:
        try:
            with Image.open(p) as img:
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


def _collapse_repetition(text: str) -> str:
    """Cut a prompt short where a small model starts looping.

    Asked for one vivid sentence, a 2B model will sometimes latch onto a phrase
    and repeat it: "a sleek black cat, sleek and smooth, ... sleek black
    curtains, sleek black blinds". The repetition adds nothing and crowds out
    the actual subject, so the prompt is cut at the point it starts.
    """
    parts = [p.strip() for p in text.split(",") if p.strip()]
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        # Compare on content words, so "sleek black cat" and "sleek black
        # curtains" are different but a verbatim repeat is caught.
        key = " ".join(sorted(w.lower().strip(".") for w in part.split()))
        if key in seen:
            break
        seen.add(key)
        kept.append(part)

        # Two consecutive clauses sharing every significant word is a loop.
        if len(kept) >= 3:
            a, b = set(kept[-1].lower().split()), set(kept[-2].lower().split())
            if a and b and len(a & b) >= max(len(a), len(b)) - 1:
                kept.pop()
                break

    out = ", ".join(kept)
    return out if out.endswith((".", "!", "?")) else out.rstrip(" ,;:") + "."


def _significant(text: str) -> list[str]:
    """Content words, ignoring filler that carries no meaning."""
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return [w for w in cleaned.split() if len(w) > 3 and w not in _STOPWORDS]


def _parse_scene(text: str) -> dict[str, str]:
    """Pull the labelled facts out of the model's answer.

    Tolerant on purpose: a small model will drop a line, change the case, or
    wrap the whole thing in prose. Anything missing simply is not used.
    """
    facts: dict[str, str] = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().strip("-*# ").upper()
        value = " ".join(value.split()).strip(" .")
        if key in ("SUBJECT", "SURFACE", "LIGHT", "SETTING") and value:
            if value.lower() in ("unknown", "n/a", "none"):
                continue
            # Guard against an answer that turns into a paragraph.
            facts[key] = " ".join(value.split()[:8])
    return facts


def _compose_edit_instruction(user_prompt: str, subject: str) -> str:
    """Build the edit instruction from the user's own words.

    Deliberately not a rewrite. Asking a small model to restate an instruction
    after showing it a picture meant it mixed the picture's details into the
    request and sometimes replaced the request altogether -- "make it blue"
    came back as "remove the background". The model is now asked only what the
    subject is; the sentence is assembled here, so the user's wording survives
    literally and their intent cannot be substituted.
    """
    text = user_prompt.strip().rstrip(".")
    if not text:
        return text
    if not subject:
        return f"{text}, keeping everything else in the picture unchanged."

    # Replace the first standalone pronoun with what the picture actually
    # shows. Word boundaries matter: a naive search matches inside "it" in
    # "white", and index arithmetic across a padded copy silently shifts by one.
    pronoun = re.compile(r"\b(it|this|that|them|these|those)\b", re.IGNORECASE)
    if pronoun.search(text):
        text = pronoun.sub(subject, text, count=1)
    else:
        # No pronoun to resolve. Name the subject only when the request does
        # not already say what it applies to.
        significant = [w.lower() for w in subject.split() if len(w) > 3]
        if not any(w in text.lower() for w in significant):
            text = f"{text} on {subject}"

    # Collapse any double spaces the substitution introduced.
    text = " ".join(text.split())
    return f"{text}, keeping everything else in the picture unchanged."


def _enrich_edit_instruction(user_prompt: str, facts: dict[str, str]) -> str:
    """Deepen the request with what the picture shows, without replacing it.

    The user's words lead the sentence and are never paraphrased. What the
    model observed is appended as context to preserve, which is the part an
    editor actually needs: told only "make it blue", it is free to discard the
    glaze, the lighting and the backdrop along the way.
    """
    head = _compose_edit_instruction(user_prompt, facts.get("SUBJECT", ""))
    if not facts:
        return head

    # Drop the generic tail; the specific one below replaces it.
    head = head.replace(", keeping everything else in the picture unchanged.", "")

    def conflicts(fact: str) -> bool:
        """Is the user already asking to change this?

        Told "put it on a dark wooden table", appending "leaving the linen
        cloth unchanged" instructs the editor to preserve the exact thing being
        replaced. A preservation hint that contradicts the request is worse
        than no hint, so anything the request touches is left out.
        """
        request = set(_significant(user_prompt))
        return bool(request & set(_significant(fact)))

    # Words naming the parts of a scene, so "move it to a table" is understood
    # to be about the setting even when it shares no word with the observation.
    SETTING_WORDS = {"background", "behind", "table", "floor", "wall", "surface",
                     "scene", "setting", "backdrop", "place", "put", "move"}
    LIGHT_WORDS = {"light", "lighting", "shadow", "bright", "dark", "dim",
                   "sunlit", "exposure", "glow"}
    SURFACE_WORDS = {"texture", "material", "finish", "glossy", "matte",
                     "smooth", "rough", "shiny"}

    request_words = set(_significant(user_prompt)) | set(user_prompt.lower().split())

    preserve = []
    for key, guard in (("SURFACE", SURFACE_WORDS), ("LIGHT", LIGHT_WORDS)):
        fact = facts.get(key)
        if fact and not conflicts(fact) and not (request_words & guard):
            preserve.append(fact)

    keep = facts.get("SETTING")
    if keep and (conflicts(keep) or (request_words & SETTING_WORDS)):
        keep = None

    clauses = [head]
    if preserve:
        clauses.append("preserving its " + " and ".join(preserve))
    if keep:
        clauses.append(f"and leaving {keep} unchanged")
    elif preserve:
        clauses.append("and leaving the rest of the picture unchanged")
    else:
        clauses.append("keeping everything else in the picture unchanged")
    return ", ".join(clauses).replace(", and ", " and ") + "."


def _enrich_idea(user_prompt: str, detail: str) -> str:
    """Put the user's idea first, then the added detail.

    The earlier version asked for a whole new sentence, which is how "a cat"
    became a paragraph about a room with no cat in it. Leading with their exact
    words means the subject cannot be lost.
    """
    idea = user_prompt.strip().rstrip(".,")
    extra = _collapse_repetition(_clean_assist(detail, "")).strip().rstrip(".")
    # Strip a restatement if the model ignored the instruction not to.
    low = extra.lower()
    if low.startswith(idea.lower()):
        extra = extra[len(idea):].lstrip(" ,")
    if not extra:
        return f"{idea}."
    return _trim_to_sentence(f"{idea}, {extra}.")


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
            # One narrow question: what is this? The instruction itself is
            # composed from the user's words, not written by the model.
            emit({"id": req_id, "type": "progress", "phase": "denoise",
                  "progress": None, "message": "Looking at your picture"})
            raw = ask(SCENE_SYSTEM, staged, 90, 0.1)
            facts = _parse_scene(raw)
            description = " · ".join(f"{k.lower()}: {v}" for k, v in facts.items())
            log(req_id, f"scene: {description or 'nothing readable'}")
            improved = _enrich_edit_instruction(user_prompt, facts)
        else:
            emit({"id": req_id, "type": "progress", "phase": "denoise",
                  "progress": None, "message": "Writing the prompt"})
            instruction = (
                f"{DETAIL_SYSTEM}\nNow do the same here.\n"
                f"User idea: {user_prompt}\nYou add:"
            )
            improved = _enrich_idea(user_prompt, ask(instruction, [], 120, 0.5))

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


def op_image_formats(req_id: str, _req: dict[str, Any]) -> dict[str, Any]:
    """Which image extensions this install can actually decode."""
    from PIL import Image

    Image.init()
    exts = sorted({e.lstrip(".").lower() for e in Image.EXTENSION})
    # Only offer what is both decodable and sensible to edit.
    useful = [e for e in exts if e in {
        "png", "jpg", "jpeg", "jpe", "webp", "avif", "heic", "heif",
        "tif", "tiff", "bmp", "gif", "ppm", "tga", "ico", "jp2", "j2k",
    }]
    return {"extensions": useful, "heic": "heic" in useful}


def op_lora_info(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Inspect an adapter repository before offering to install it."""
    from huggingface_hub import HfApi

    repo = req["model"].strip()
    info = HfApi().model_info(repo, files_metadata=True)
    files = [
        {"name": s.rfilename, "bytes": getattr(s, "size", None) or 0}
        for s in (info.siblings or [])
        if s.rfilename.endswith(".safetensors")
    ]
    # A repo with several adapters needs the exact file naming its handle.
    return {
        "repo": repo,
        "files": sorted(files, key=lambda f: -f["bytes"]),
        "bytes": sum(f["bytes"] for f in files),
        "gated": bool(getattr(info, "gated", False)),
        "base_model": (getattr(info, "cardData", None) or {}).get("base_model"),
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
    family: str | None = None

    try:
        try:
            caps = get_model_capabilities(model=repo)
        except Exception:
            # The resolver could not place this repository from its name.
            # Deliberately not probing families until one accepts: acceptance
            # is not identification. FLUX.1 is accepted by the flux2 router and
            # would load its weights through the wrong architecture. Identify
            # it from what the repository declares, or not at all.
            family = _detect_family(repo)
            if family is None:
                raise
            caps = get_model_capabilities(model=repo, family=family)
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
    params = 0
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        # Hugging Face reports this for ordinary repositories but not for
        # quantized packages, whose tensors are stored differently.
        st = getattr(info, "safetensors", None)
        params = int(getattr(st, "total", 0) or 0) if st else 0
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

    if route_error and "infer a supported backend" in route_error.lower():
        route_error = (
            "This model is not one the engine can run. It supports the FLUX.2, "
            "Qwen-Image, Z-Image, ERNIE, FIBO, Bonsai and Wan families. Stable "
            "Diffusion, SDXL and FLUX.1 are different architectures and are not "
            "included."
        )

    return {
        "model": repo,
        "family": family,
        "modes": modes,
        "tasks": tasks,
        "bytes": size,
        "params": params,
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

    def run_download(cmd: list[str], environ: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, env=environ, capture_output=True, text=True)

    def failed_in_transfer(proc: subprocess.CompletedProcess) -> bool:
        """Did this fail in the chunked transfer layer rather than legitimately?

        Hugging Face's xet backend reconstructs files from content-addressed
        chunks, and a truncated response surfaces as a CAS or reconstruction
        error partway through a large download. Retrying the same way tends to
        fail the same way; falling back to plain HTTP range requests does not.
        """
        text = ((proc.stderr or "") + (proc.stdout or "")).lower()
        return any(m in text for m in (
            "cas client error", "file reconstruction error", "xet_get",
            "error decoding response body",
        ))

    # Not every model belongs to MLX-Gen. The prompt assistant is a VLM loaded
    # by mlx-vlm, and `mlxgen download` exits 0 without fetching anything for a
    # repo it does not recognise -- which looked like a successful zero-byte
    # download. Route those straight through huggingface_hub instead.
    via = req.get("via") or "mlxgen"

    try:
        if via == "lora":
            # Adapter repositories are small and MLX-Gen wants the whole thing,
            # not the weight/tokenizer subset it fetches for a model.
            cli = Path(sys.executable).parent / "mlxgen"
            cmd = [str(cli), "download", "--model", repo_id, "--all-files"]
            proc = run_download(cmd, env)
            if proc.returncode != 0 and failed_in_transfer(proc):
                log(req_id, "chunked transfer failed; retrying over plain HTTP", "warn")
                retry_env = dict(env)
                retry_env["HF_HUB_DISABLE_XET"] = "1"
                proc = run_download(cmd, retry_env)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                raise RuntimeError("adapter download failed:\n" + "\n".join(tail[-8:]))
        elif via == "hf":
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo_id)
        else:
            cli = Path(sys.executable).parent / "mlxgen"
            cmd = [str(cli), "download", "--model", repo_id] if cli.exists() else [
                sys.executable, "-m", "mlxgen", "download", "--model", repo_id
            ]
            proc = run_download(cmd, env)
            if proc.returncode != 0 and failed_in_transfer(proc):
                # A large model can lose an hour to this, so it is worth one
                # automatic retry down the slower but sturdier path.
                log(req_id, "chunked transfer failed; retrying over plain HTTP", "warn")
                retry_env = dict(env)
                retry_env["HF_HUB_DISABLE_XET"] = "1"
                emit({"id": req_id, "type": "progress", "phase": "download",
                      "progress": None, "message": "Transfer failed; retrying"})
                proc = run_download(cmd, retry_env)
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



# --------------------------------------------------------------------------
# Live previews
# --------------------------------------------------------------------------

# Longest edge of a preview frame. Small on purpose: these are sent once per
# denoise step over the same JSON Lines channel as everything else, and the
# point is to show the picture taking shape, not to be the picture.
PREVIEW_MAX_EDGE = 320
PREVIEW_QUALITY = 62


def _resolve_latent_creator(model: Any) -> Any:
    """Find the latent unpacker for whichever family this model belongs to.

    Unpacking is per-family and the model does not carry a reference to it, so
    it is resolved from the model's own module path: a class living under
    `mflux.models.<family>.variants...` pairs with a creator under
    `mflux.models.<family>.latent_creator`.
    """
    import importlib
    import inspect
    import pkgutil

    parts = type(model).__module__.split(".")
    if len(parts) < 3 or parts[0] != "mflux" or parts[1] != "models":
        return None
    family = parts[2]

    try:
        pkg = importlib.import_module(f"mflux.models.{family}.latent_creator")
    except Exception:
        return None

    modules = [pkg]
    for info in getattr(pkg, "__path__", []) and pkgutil.iter_modules(pkg.__path__) or []:
        try:
            modules.append(importlib.import_module(
                f"mflux.models.{family}.latent_creator.{info.name}"))
        except Exception:
            continue

    for mod in modules:
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if hasattr(obj, "unpack_latents"):
                return obj
    return None


def _attach_live_preview(req_id: str, loaded: Any) -> Any:
    """Emit a JPEG of the partial image after each denoise step.

    MLX-Gen can decode a step's latents through a published tiny autoencoder
    for the same latent space, far cheaper than the full VAE. Without this a
    hundred-second generation is a progress bar and nothing else; with it you
    can see whether the composition is going anywhere and stop early if not.

    Mirrors MLX-Gen's own StepwiseHandler rather than the documentation's
    example, which references a `LatentCreator.unpack_latents` that no longer
    exists -- following the docs produced a handler that ran every step and
    silently failed on every one.
    """
    import base64
    import io

    try:
        from mflux.models.common.preview.preview_decoder import PreviewDecoder
        from mflux.utils.image_util import ImageUtil
    except Exception as exc:
        log(req_id, f"live preview unavailable: {exc}", "warn")
        return None

    target = getattr(loaded, "model", loaded)
    creator = _resolve_latent_creator(target)
    if creator is None:
        log(req_id, "no latent unpacker for this family; preview disabled", "warn")
        return None

    try:
        # Resolving once, before the loop, keeps each step cheap.
        decoder = PreviewDecoder.resolve(target, mode="auto")
    except Exception as exc:
        log(req_id, f"no preview decoder: {exc}", "warn")
        decoder = None

    state = {"failed": False}

    class LivePreview:
        def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
            if state["failed"]:
                return
            try:
                unpacked = creator.unpack_latents(
                    latents=latents, height=config.height, width=config.width
                )
                vae = getattr(target, "vae", None)
                if decoder is not None:
                    decoded = decoder.decode(unpacked, vae=vae)
                elif hasattr(vae, "decode_packed_latents"):
                    decoded = vae.decode_packed_latents(unpacked)
                else:
                    decoded = vae.decode(unpacked)

                img = ImageUtil.to_pil_image(decoded)
                if max(img.size) > PREVIEW_MAX_EDGE:
                    scale = PREVIEW_MAX_EDGE / max(img.size)
                    img = img.resize(
                        (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                    )
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=PREVIEW_QUALITY)
                emit({
                    "id": req_id,
                    "type": "preview",
                    "step": (t + 1) if isinstance(t, int) else None,
                    "total_steps": getattr(config, "num_inference_steps", None),
                    "jpeg": base64.b64encode(buf.getvalue()).decode("ascii"),
                })
            except Exception as exc:
                # Report once, then stay quiet: a preview is a convenience and
                # must never interrupt or spam a generation. Reporting at all
                # matters, though -- silently swallowing this is what hid a
                # broken handler behind an empty canvas.
                state["failed"] = True
                log(req_id, f"preview disabled after an error: {exc}", "warn")

    handler = LivePreview()
    try:
        target.callbacks.register(handler)
    except Exception as exc:
        log(req_id, f"could not register preview: {exc}", "warn")
        return None
    return handler


def _detach_live_preview(loaded: Any, handler: Any) -> None:
    if handler is None:
        return
    target = getattr(loaded, "model", loaded)
    cb = getattr(target, "callbacks", None)
    for attr in ("unregister", "remove"):
        fn = getattr(cb, attr, None)
        if callable(fn):
            try:
                fn(handler)
                return
            except Exception:
                pass
    # No removal API: drop it from whichever list holds it, so a later run in
    # the same process does not keep emitting into a finished job.
    for name in ("in_loop_callbacks", "callbacks"):
        lst = getattr(cb, name, None)
        if isinstance(lst, list) and handler in lst:
            lst.remove(handler)


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

    # MLX-Gen exposes edit-reference as several capability rows -- plain,
    # masked, reframe and outpaint -- and picks between them from these hints.
    # They must be supplied at load time, not with the generation call.
    plan_kw: dict[str, Any] = {}
    if req.get("family"):
        # Without this, MLX-Gen infers the router family from the repository
        # name and fails outright on anything it does not recognise:
        # "could not infer a supported backend for model ... pass family=".
        # The catalog knows the family for every model it lists.
        plan_kw["family"] = req["family"]
    if req.get("i2i_mode"):
        plan_kw["i2i_mode"] = req["i2i_mode"]
    if req.get("mask"):
        plan_kw["has_mask"] = True
    if req.get("outpaint_padding"):
        plan_kw["has_outpaint"] = True
    if req.get("image_strength") is not None:
        plan_kw["has_image_strength"] = True

    loras = req.get("loras") or []
    if loras:
        log(req_id, "adapters: " + ", ".join(
            f"{l['path']} @ {l.get('scale', 1.0)}" for l in loras))

    loaded, load_ms = _load_model(
        req_id, model, quantize, model_path, image_count,
        release_text_encoder=bool(req.get("release_text_encoder")) or low_ram,
        loras=loras,
        **plan_kw,
    )

    gen_kw, optional_kw = _split_gen_kwargs(req)
    slots = req.get("vault_slots") or []

    # Source images live sealed in the vault. mlx-gen loads images by path, so
    # they are decrypted into a private 0700 directory for the duration of the
    # run and removed in the `finally` below. This is the one moment plaintext
    # exists on disk, and it is bounded by a single generation.
    staged = _stage_vault_inputs(req.get("vault_inputs") or [])
    # The mask is a separate sealed blob with its own key.
    mask_staged = _stage_vault_inputs([req["mask"]] if req.get("mask") else [])
    if mask_staged:
        gen_kw["mask_path"] = mask_staged[0]
    try:
        return _generate_inner(
            req_id, req, task, loaded, gen_kw, optional_kw, slots, staged,
            seeds, model, output, low_ram, load_ms,
        )
    finally:
        _discard_staged(staged)
        _discard_staged(mask_staged)


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
    preview = _attach_live_preview(req_id, loaded) if req.get("preview", True) else None
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

    outpaint_padding = req.get("outpaint_padding")
    if outpaint_padding:
        return _run_expand(
            req_id, req, loaded, gen_kw, optional_kw, slots, images, seeds,
            model, outpaint_padding, load_ms,
        )

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
        _detach_live_preview(loaded, preview)
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



def _run_expand(req_id, req, loaded, gen_kw, optional_kw, slots, images, seeds,
                model, outpaint_padding, load_ms) -> dict[str, Any]:
    """Grow the canvas beyond the original picture.

    Outpaint is a separate pipeline in MLX-Gen: the source is pasted onto a
    larger canvas and the model completes the added area. It takes no
    width/height, since the canvas comes from the source plus the padding.

    Reframe is deliberately not wired up. It is a sibling of outpaint with its
    own session type and no public runner -- `run_outpaint` rejects it with
    "Capability 'flux2.reframe' does not support outpaint" -- and it covers the
    same user-facing need, so it buys nothing for the complexity.
    """
    from mflux.outpaint import run_outpaint

    if not images:
        raise ValueError("expanding needs a source image")

    padding = outpaint_padding
    kwargs = {k: v for k, v in {**gen_kw, **optional_kw}.items()
              if k not in ("width", "height", "image_path", "image_paths", "mask_path")}

    gen_started = time.time()
    results = call_tolerant(
        req_id,
        run_outpaint,
        {
            "loaded": loaded,
            "source_image": images[0],
            "padding": padding,
            "seeds": list(seeds),
            "progress_callback": _make_progress_handler(req_id),
            **kwargs,
        },
        # `fill` only applies to outpaint; reframe derives its own canvas.
        {"fill": req["outpaint_fill"]} if req.get("outpaint_fill") else {},
    )
    generate_ms = (time.time() - gen_started) * 1000.0

    out = _seal_results(req_id, results, slots, model, list(seeds))
    out["load_ms"] = round(load_ms)
    out["generate_ms"] = round(generate_ms)
    return out



# --------------------------------------------------------------------------
# FLUX.1 backend
# --------------------------------------------------------------------------

# MLX-Gen's unified router covers seven families; FLUX.1 is not among them, so
# it is driven through mflux's own classes. That family is worth the second
# path: Kontext is the strongest open instruction editor, and Fill is real
# mask-based inpainting rather than the outpaint-adjacent route Klein offers.
#
# Each entry maps a catalog id to the class that runs it and the ModelConfig
# that describes its weights.
MFLUX_BACKENDS: dict[str, tuple[str, str, str]] = {
    # config factory      module                                              class
    "dev": ("mflux.models.flux.variants.txt2img.flux", "Flux1", "dev"),
    "schnell": ("mflux.models.flux.variants.txt2img.flux", "Flux1", "schnell"),
    "dev_kontext": ("mflux.models.flux.variants.kontext.flux_kontext",
                    "Flux1Kontext", "dev_kontext"),
    "dev_fill": ("mflux.models.flux.variants.fill.flux_fill", "Flux1Fill", "dev_fill"),
}


def _load_mflux_model(req_id: str, backend: str, model_path: str | None,
                      quantize: int | None,
                      loras: list[dict[str, Any]] | None = None) -> tuple[Any, float]:
    """Construct a FLUX.1 model directly, honouring the one-resident rule."""
    import importlib

    from mflux.models.common.config.model_config import ModelConfig

    spec = MFLUX_BACKENDS.get(backend)
    if spec is None:
        raise ValueError(f"unknown FLUX.1 backend {backend!r}")
    module_name, class_name, factory = spec

    key = f"mflux::{backend}::{model_path}::{quantize}"
    if loras:
        key += "::lora:" + ",".join(f"{l['path']}@{l.get('scale', 1.0)}" for l in loras)

    cached = CACHE.get(key)
    if cached is not None:
        log(req_id, f"reusing resident model ({CACHE.label})")
        return cached, 0.0

    global _POLICY_APPLIED
    if not _POLICY_APPLIED:
        _POLICY_APPLIED = True
        log(req_id, f"resource policy: {apply_resource_policy()}")

    CACHE.unload()
    if _ASSIST.get("model") is not None:
        _unload_assistant()

    log(req_id, f"loading {backend} via mflux (quantize={quantize})")
    emit({"id": req_id, "type": "progress", "phase": "load", "progress": 0.0,
          "message": f"Loading {backend}"})

    cls = getattr(importlib.import_module(module_name), class_name)
    t0 = time.time()
    kwargs: dict[str, Any] = {
        "model_config": getattr(ModelConfig, factory)(),
        "quantize": quantize,
        "model_path": model_path,
    }
    if loras:
        kwargs["lora_paths"] = [l["path"] for l in loras]
        kwargs["lora_scales"] = [float(l.get("scale", 1.0)) for l in loras]

    with _downloads_allowed(req_id):
        model = call_tolerant(req_id, cls, {}, kwargs)

    CACHE.put(key, model, backend)
    load_ms = (time.time() - t0) * 1000.0
    log(req_id, f"loaded in {load_ms / 1000:.1f}s")
    return model, load_ms


def _run_mflux(req_id: str, req: dict[str, Any], backend: str) -> dict[str, Any]:
    """Generate or edit through a FLUX.1 model.

    These classes produce one image per call rather than taking a seed list,
    so the multi-image loop lives here instead of in the runtime wrapper.
    """
    slots = req.get("vault_slots") or []
    if not slots:
        raise ValueError("this route needs a vault slot to write into")

    seeds = req.get("seeds") or [req.get("seed", 0)]
    staged = _stage_vault_inputs(req.get("vault_inputs") or [])
    mask_staged = _stage_vault_inputs([req["mask"]] if req.get("mask") else [])

    try:
        model, load_ms = _load_mflux_model(
            req_id, backend,
            req.get("model_path") or req.get("model"),
            req.get("quantize"),
            req.get("loras") or [],
        )

        base: dict[str, Any] = {"prompt": req.get("prompt", "")}
        optional: dict[str, Any] = {}
        for src, dst in (("steps", "num_inference_steps"), ("width", "width"),
                         ("height", "height"), ("guidance", "guidance"),
                         ("image_strength", "image_strength")):
            if req.get(src) is not None:
                optional[dst] = req[src]
        if staged:
            optional["image_path"] = staged[0]
        if mask_staged:
            # Fill takes the mask separately; the name differs from the
            # unified router's `mask_path`.
            optional["masked_image_path"] = mask_staged[0]

        preview = _attach_live_preview(req_id, model) if req.get("preview", True) else None
        artifacts = []
        gen_started = time.time()
        try:
            for seed in seeds:
                artifacts.append(call_tolerant(
                    req_id, model.generate_image,
                    {**base, "seed": int(seed),
                     "progress_callback": _make_progress_handler(req_id)},
                    optional,
                ))
        finally:
            _detach_live_preview(model, preview)
        generate_ms = (time.time() - gen_started) * 1000.0

        out = _seal_results(req_id, artifacts, slots, req.get("model", backend), list(seeds))
        out["load_ms"] = round(load_ms)
        out["generate_ms"] = round(generate_ms)
        return out
    finally:
        _discard_staged(staged)
        _discard_staged(mask_staged)


def op_generate(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    if req.get("backend"):
        return _run_mflux(req_id, req, req["backend"])
    return _run_generation(req_id, req, "text-to-image", image_count=0)


def op_edit(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    # Sources normally arrive sealed as `vault_inputs`; `images` is only used
    # by direct callers with plaintext paths. Counting just `images` made every
    # vault-backed edit fail, and would have routed it to text-to-image even if
    # it had not, because image_count drives route selection.
    count = len(req.get("images") or []) + len(req.get("vault_inputs") or [])
    if count == 0:
        raise ValueError("edit requires at least one input image")
    if req.get("backend"):
        return _run_mflux(req_id, req, req["backend"])
    if count == 1:
        req.setdefault("i2i_mode", "edit")
    else:
        # Let the resolver choose multi-reference; pinning "edit" here selected
        # the single-image row and made a second source an error.
        req.pop("i2i_mode", None)
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



# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------

def op_video(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Generate a clip, optionally starting from a picture.

    Frame count is the only real cost lever. `fps` is written into the file as
    playback metadata and changes the clip's duration, not the work done: the
    same 49 frames tagged 16fps or 24fps cost exactly the same to produce.
    Attention is quadratic in sequence length, so halving frames saves more
    than half the memory.

    Every low-RAM knob the route exposes is on by default. On unified memory
    these bound the working set rather than dodging a bus, which is the only
    thing that makes a video model fit at all on a 16 GB machine.
    """
    slots = req.get("vault_slots") or []
    if not slots:
        raise ValueError("video generation needs a vault slot to write into")

    frames = int(req.get("frames", 33))
    if frames < 5:
        raise ValueError("a clip needs at least 5 frames")

    staged = _stage_vault_inputs(req.get("vault_inputs") or [])
    try:
        # Routes disagree about what a picture means here. Wan treats it as a
        # first frame (image_count); Bernini treats it as a reference to
        # animate (reference_image_count) and rejects image input outright.
        # Offering both lets the resolver pick whichever this model supports.
        plan_kw: dict[str, Any] = {}
        if req.get("family"):
            plan_kw["family"] = req["family"]
        if staged:
            plan_kw["reference_image_count"] = len(staged)

        loaded, load_ms = _load_model(
            req_id,
            req["model"],
            req.get("quantize"),
            req.get("model_path"),
            image_count=0,
            release_text_encoder=True,
            **plan_kw,
        )

        base: dict[str, Any] = {
            "seed": int(req.get("seed", 0)),
            "prompt": req.get("prompt", ""),
            "num_frames": frames,
            "fps": int(req.get("fps", 16)),
            "width": int(req.get("width", 480)),
            "height": int(req.get("height", 320)),
            "num_inference_steps": int(req.get("steps", 20)),
            "progress_callback": _make_progress_handler(req_id),
            # The layer-boundary levers, which for video are not optional.
            "clear_cache_each_step": True,
            "clear_cache_each_transformer_block": True,
            "release_denoisers_before_decode": True,
        }
        optional: dict[str, Any] = {}
        if staged:
            # Same picture, two spellings: a first frame for Wan, an ordered
            # reference set for Bernini. The tolerance layer drops whichever
            # this route names as unsupported.
            optional["reference_image_paths"] = list(staged)
            optional["image_path"] = staged[0]
        if req.get("negative_prompt"):
            optional["negative_prompt"] = req["negative_prompt"]
        if req.get("guidance") is not None:
            # Wan calls this `guidance`; Bernini steers references with
            # `reference_guidance` and its documented default is 4.5. Offering
            # both lets the tolerance layer keep whichever the route names.
            optional["guidance"] = req["guidance"]
            optional["reference_guidance"] = req["guidance"]

        target = getattr(loaded, "model", loaded)
        gen_started = time.time()
        video = call_tolerant(req_id, target.generate_video, base, optional)
        generate_ms = (time.time() - gen_started) * 1000.0

        # MP4 needs a container writer, so unlike images this cannot be encoded
        # purely in memory. It is written to the private staging directory and
        # removed as soon as it has been sealed.
        import vaultcrypto as vc

        tmp = os.path.join(_stage_dir(), f"{slots[0]['id']}.mp4")
        emit({"id": req_id, "type": "progress", "phase": "save",
              "progress": None, "message": "Encoding the clip"})
        video.save(path=tmp, export_json_metadata=False)
        try:
            with open(tmp, "rb") as f:
                data = f.read()
            slot = slots[0]
            vc.write_sealed(
                slot["path"], bytes.fromhex(slot["key"]),
                bytes.fromhex(slot["file_id"]), data,
            )
            log(req_id, f"sealed {len(data)} bytes of video into {slot['id']}")
        finally:
            _discard_staged([tmp])
    finally:
        _discard_staged(staged)
        # Video weights are the largest thing this app loads; do not keep them.
        CACHE.unload()

    return {"outputs": [slots[0]["id"]], "sealed": True, "sizes": [len(data)],
            "model": req["model"], "frames": frames,
            "load_ms": round(load_ms), "generate_ms": round(generate_ms)}


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
    "lora_info": op_lora_info,
    "image_formats": op_image_formats,
    "resolve": op_resolve,
    "download": op_download,
    "generate": op_generate,
    "edit": op_edit,
    "upscale": op_upscale,
    "video": op_video,
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
