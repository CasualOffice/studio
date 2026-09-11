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

    # Some routes name the remedy as well as the fault: "Wan VACE does not take
    # image_path; pass reference_image_paths instead." Dropping the parameter
    # the model just asked for is the opposite of what it said, and it is not
    # a harmless mistake -- losing the image conditioning left video generation
    # running as if no picture had been supplied at all.
    # "do not pass X" is the opposite recommendation to "pass X", and reading
    # one as the other protects exactly the parameter that should go.
    suggested = {
        norm(m) for m in re.findall(
            r"(?<!not )(?<!n't )(?:pass|use|supply|provide|try)\s+"
            r"['\"`]?([A-Za-z_][A-Za-z0-9_-]*)['\"`]?(?:\s+instead)?",
            message,
        )
    }

    normalized_quoted = {norm(q) for q in quoted} - suggested
    head_norm = norm(head)

    def matches(candidate: str) -> bool:
        c = norm(candidate)
        if c in suggested:
            return False
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
        if CACHE.loaded is not None or _helpers_resident() is not None:
            freed = CACHE.label or _helpers_resident()
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


# FLUX.1 pipeline classes, and the mflux backend that runs each one. These are
# a separate lineage from FLUX.2 and the unified router cannot place them, but
# mflux can -- so they are supported, just through a different door.
_MFLUX_PIPELINES = (
    ("fluxkontextpipeline", "dev_kontext"),
    ("fluxfillpipeline", "dev_fill"),
    ("fluxpipeline", None),  # base FLUX.1: dev or schnell, decided below
)


# FLUX.1 weight filenames, which name the variant outright. Checked in order,
# so the more specific ones win over the plain dev/schnell files.
_MFLUX_WEIGHT_NAMES = (
    ("flux1kontext", "dev_kontext"),
    ("flux1fill", "dev_fill"),
    ("flux1schnell", "schnell"),
    ("flux1dev", "dev"),
)


def _hf_token() -> str | None:
    """The Hugging Face access token, if the user has supplied one.

    Gated repositories -- which includes every official FLUX.1 model -- need
    one. The host passes it through the environment so it never has to be
    written into this process's arguments or logs.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        tok = os.environ.get(var, "").strip()
        if tok:
            return tok
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token() or None
    except Exception:
        return None


def _detect_mflux_backend(repo: str, tags: list[str] | None = None,
                          files: list[str] | None = None) -> str | None:
    """Identify a FLUX.1 repository and pick the mflux backend that runs it.

    Returns None for anything that is not FLUX.1. Distinguishing dev from
    schnell matters because schnell is guidance-distilled: run it with dev's
    settings and every image comes out scorched.

    Two sources of evidence, because the first is not always reachable: the
    declared pipeline class, and failing that the weight filenames. The
    official FLUX.1 repos are gated, so `model_index.json` needs a token that
    the file listing does not -- and `flux1-schnell.safetensors` identifies the
    variant just as definitively as the pipeline class does.
    """
    import json as _json

    haystack = (repo + " " + " ".join(tags or [])).lower()
    # FLUX.2 is a different architecture with its own router, which has already
    # had its turn by the time this runs. Never claim one for mflux.
    if "flux2" in haystack.replace(".", "").replace("-", "").replace("_", ""):
        return None

    declared = ""
    for filename in ("model_index.json", "config.json"):
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=repo, filename=filename,
                                   token=_hf_token())
            with open(path) as fh:
                declared = str(_json.load(fh).get("_class_name", ""))
            if declared:
                break
        except Exception:
            continue

    declared = declared.lower().replace("-", "").replace("_", "")
    flat = " ".join(files or []).lower().replace("-", "").replace("_", "")

    def variant() -> str:
        """Which FLUX.1 this is. Order matters: Kontext repos also say "dev"."""
        for needle, backend in _MFLUX_WEIGHT_NAMES:
            if needle in flat:
                return backend
        for needle, backend in (("kontext", "dev_kontext"), ("fill", "dev_fill"),
                                ("schnell", "schnell")):
            if needle in haystack:
                return backend
        return "dev"

    for needle, backend in _MFLUX_PIPELINES:
        if needle in declared:
            return backend or variant()

    for needle, backend in _MFLUX_WEIGHT_NAMES:
        if needle in flat:
            return backend

    # No declaration and no telltale filename: this is how the pre-quantized
    # mflux packages arrive. Their component layout is still evidence -- two
    # text encoders beside a transformer and a VAE is FLUX's shape, and not
    # Stable Diffusion's (a single encoder and a unet) or Qwen-Image's. Only
    # once the shape matches does the name get to pick the variant.
    parts = {f.split("/")[0] for f in (files or []) if "/" in f}
    flux_shaped = {"transformer", "vae", "text_encoder", "text_encoder_2"} <= parts
    if flux_shaped and "flux" in haystack:
        return variant()
    return None


def _looks_like_adapter(info: Any) -> bool:
    """Is this a LoRA adapter repository rather than a whole model?

    Worth answering separately: an adapter downloaded as a model can never
    generate anything, and "unsupported" is the wrong thing to tell someone
    who pasted a perfectly good LoRA.
    """
    files = [s.rfilename for s in (getattr(info, "siblings", None) or [])]
    weights = [f for f in files if f.endswith(".safetensors")]
    if not weights:
        return False
    # A model repository declares its pipeline; an adapter does not.
    if any(f in ("model_index.json", "config.json") for f in files):
        return False
    tags = [t.lower() for t in (getattr(info, "tags", None) or [])]
    return (
        any("lora" in f.lower() for f in weights)
        or any("lora" in t or "adapter" in t for t in tags)
        or len(weights) == 1
    )


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
    if _helpers_resident() is not None:
        log(req_id, "releasing the prompt helper to make room for the model")
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


def _helpers_resident() -> str | None:
    """Which prompt helper is holding memory, if any.

    There are two of them now -- the reader that looks at pictures and the
    writer that composes prose -- and every guard that made room for "the
    assistant" was written when there was one. Missing the writer meant a
    3.2 GiB model could still be resident when a 9.5 GiB video model loaded,
    which is 12.7 GiB against a 12.5 GiB budget and precisely the swapping the
    one-resident rule exists to prevent.
    """
    return _ASSIST.get("key") if _ASSIST.get("model") is not None else (
        _WRITER.get("key") if _WRITER.get("model") is not None else None
    )


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


# The writer is a text model, separate from the one that looks at pictures.
# A 2B vision-language model was doing both, and it could not hold a rule
# while writing: asked to make a request precise without inventing anything,
# it answered "make it look better" with a painting in a dark room. Reading a
# picture and writing careful prose are different jobs, so they are now
# different models -- and the writer answers in about two seconds rather than
# twenty.
_WRITER: dict[str, Any] = {"key": None, "model": None, "tokenizer": None}

WRITER_REPO = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def _load_writer(req_id: str, repo: str | None = None) -> tuple[Any, Any]:
    repo = repo or WRITER_REPO
    if _WRITER["key"] == repo and _WRITER["model"] is not None:
        return _WRITER["model"], _WRITER["tokenizer"]

    if CACHE.loaded is not None:
        log(req_id, f"releasing {CACHE.label} to make room for the writer")
        CACHE.unload()

    from mlx_lm import load as lm_load

    emit({"id": req_id, "type": "progress", "phase": "load", "progress": 0.0,
          "message": "Loading the prompt writer"})
    model, tokenizer = lm_load(repo)
    _WRITER.update(key=repo, model=model, tokenizer=tokenizer)
    log(req_id, f"writer {repo} ready")
    return model, tokenizer


def _write(req_id: str, system: str, user: str, max_tokens: int = 160,
           temperature: float = 0.3, repo: str | None = None) -> str:
    """One turn with the writer."""
    from mlx_lm import generate as lm_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _load_writer(req_id, repo)
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        add_generation_prompt=True,
    )
    out = lm_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                      sampler=make_sampler(temp=temperature), verbose=False)
    return out if isinstance(out, str) else str(out)


def _unload_writer() -> None:
    _WRITER.update(key=None, model=None, tokenizer=None)
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def _unload_assistant() -> None:
    _unload_writer()
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

# Structured scene direction for a text-only idea.
#
# These models read prompts through a T5 encoder, which parses grammar: flowing
# description outperforms a list of tags, and long prompts outperform short
# ones. The earlier version asked for comma-separated keywords under
# twenty-five words, which is how you prompt Stable Diffusion and close to the
# opposite of what works here.
#
# Slots rather than free writing, because a 2B model asked for "a vivid prompt"
# produces adjective soup. Asked what the light is doing, it answers.
SCENE_DIRECTION_SYSTEM = (
    "You make an image request precise. You do not invent a scene.\n\n"
    "Rules, in order of importance:\n"
    "1. Never introduce a thing the request does not already contain. No new "
    "objects, no new people, no new places. If the request says 'a cat on a "
    "chair', there is no book, no window and no rug.\n"
    "2. Keep every thing the request does name, and keep what it says about "
    "them.\n"
    "3. Your only job is to make what is already there specific: give it "
    "material, colour, texture, age, scale. 'a chair' may become 'a worn oak "
    "chair'. It may not become 'a chair beside a fireplace'.\n"
    "4. Never write 'beautiful', 'stunning', 'high quality', '8k', "
    "'cinematic' or 'masterpiece'. They describe nothing.\n"
    "5. If the request names no thing at all -- 'make it better', 'something "
    "nice' -- you cannot make it precise. Reply with exactly: UNCLEAR\n\n"
    "Reply with the rewritten request on one line and nothing else.\n\n"
    "Examples\n"
    "Request: a cat on a chair\n"
    "a tabby cat with dense grey-brown fur, curled on a worn oak chair\n\n"
    "Request: a woman walking in the rain at night\n"
    "a woman in a rain-soaked overcoat walking at night, hair flat with wet, "
    "the road black and reflective under the rain\n\n"
    "Request: make it look better\n"
    "UNCLEAR\n\n"
    "Request: a red car\n"
    "a red car with sun-faded paint and a dented wing, standing still\n"
)


# The same job for a clip. A video model is steered by what *moves*: a still
# scene description tells it nothing about motion, and the result drifts or
# sits frozen. Camera and subject motion are the two levers that matter, and
# both have to be gradual -- these models cover a second or two, so anything
# that reads like a cut or a fast pan comes out as a smear.
MOTION_DIRECTION_SYSTEM = (
    "You make a video request precise. You do not invent a scene.\n\n"
    "Rules, in order of importance:\n"
    "1. Never introduce a thing the request does not already contain. No new "
    "objects, no new people, no new places.\n"
    "2. Keep every thing the request does name.\n"
    "3. Make what is already there specific, and say how it moves. These "
    "clips last a second or two, so motion is gradual: drifting, settling, "
    "rising, swaying. Never 'suddenly', 'quickly' or 'explodes'. No cuts, no "
    "scene changes.\n"
    "4. Never write 'cinematic', 'stunning', 'high quality' or '4k'.\n"
    "5. If you are told what the picture shows, the clip starts from that "
    "picture: say how what is already there moves, and introduce nothing it "
    "does not contain.\n"
    "6. If the request names no thing at all, reply with exactly: UNCLEAR\n\n"
    "Reply with the rewritten request on one line and nothing else.\n\n"
    "Examples\n"
    "Request: steam from a teapot\n"
    "steam rising from a glazed stoneware teapot, thinning and drifting "
    "slowly to the right\n\n"
    "Request: make it move\n"
    "UNCLEAR\n"
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


# Words that can only be attributes -- how a thing looks, not another thing.
# A clarification is allowed to add these freely; anything else it adds is a
# new object, which is the failure this guards against.
_ATTRIBUTE_HINTS = (
    "ed", "ing", "ish", "less", "en", "y",   # worn, faded, greying, reddish
)

# How much longer a clarification may reasonably be than the request. Making
# "a cat" specific costs a few words; producing a paragraph means the model
# stopped clarifying and started writing its own scene.
_MAX_EXPANSION = 5

# How many more things a clarification may name than the request did. Giving a
# subject a hat and a cap is clarification; giving it a book, a window, a rug
# and a fireplace is a different picture.
_MAX_NEW_THINGS = 4


# Endings English adds to a word that is still the same word. Ordered longest
# first so "running" loses "ing" rather than "g".
_INFLECTIONS = ("ing", "ies", "ed", "es", "s")


def _stem(word: str) -> str:
    """Reduce a word to something two inflections of it share.

    Comparing by a fixed prefix does not survive English: "rises" and "rising"
    agree on only three letters, so a four-letter prefix decided the word had
    been dropped and threw away a correct rewrite. Every prompt with a verb in
    it was at risk, which for a video prompt is all of them.
    """
    w = word.lower()
    for suffix in _INFLECTIONS:
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            w = w[: -len(suffix)]
            break
    # "carries" -> "carri" -> "carry" territory: normalise the trailing i.
    return w[:-1] + "y" if w.endswith("i") else w


def _clarified(original: str, raw: str) -> str | None:
    """Take the model's rewrite, or reject it.

    Returns None when the request names nothing that could be drawn. The old
    version answered such requests by inventing a subject -- "make it look
    better" produced a painting in a dark room with a large window, none of
    which the user had asked for -- and with a T5 encoder every invented noun
    is something the image actually contains.
    """
    line = ""
    for candidate in raw.strip().splitlines():
        candidate = candidate.strip().strip("`").strip()
        # Models like to preface. Take the first line that is the answer.
        if not candidate or candidate.lower().startswith(("request:", "here", "sure")):
            continue
        line = candidate
        break
    if not line:
        return None
    if line.strip().upper().startswith("UNCLEAR"):
        return None

    # Deliberately not trimmed to a sentence. The model was asked for one
    # line, and that trim falls back to the last clause when there is no full
    # stop -- which cut "a tabby cat..., curled on a worn oak chair" at the
    # comma, deleting the chair and failing the retention check below.
    line = line.rstrip(" .,;") + "."
    if not _keeps_intent(original, line):
        return None

    # Every thing the request named has to still be there. `_keeps_intent` only
    # asks whether *any* of it survived, which let "an old bicycle against a
    # brick wall" come back as a bicycle with no wall -- the clarification
    # quietly deleting half the request.
    kept = {_stem(g) for g in _significant(line)}
    missing = [w for w in _significant(original)
               if not any(_stem(w) == k or k.startswith(_stem(w))
                          or _stem(w).startswith(k)
                          for k in kept)]
    if missing:
        return None

    # A rewrite far longer than the request is inventing, not clarifying.
    # The floor matters as much as the ratio: "a teapot" is one significant
    # word, and naming its glaze, its spout and its wear is a legitimate
    # clarification that a ratio alone would reject.
    if len(_significant(line)) > max(24, _MAX_EXPANSION * len(_significant(original))):
        return None

    # Length alone cannot tell a described teapot from a list of furniture.
    # Each "a ..." is another thing in the picture, so counting them catches
    # what the ratio cannot: adding attributes keeps the count flat, while
    # inventing objects drives it up one article at a time.
    def things(text: str) -> int:
        return len(re.findall(r"\b(?:a|an|the)\s+[a-z]", text.lower()))

    if things(line) - things(original) > _MAX_NEW_THINGS:
        return None
    return line


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



def _significant(text: str) -> list[str]:
    """Content words, ignoring filler that carries no meaning.

    Three characters counts. The threshold was four, which silently discarded
    "cat", "red", "sun", "car" and "sky" -- short words that are usually the
    entire point of a request.
    """
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return [w for w in cleaned.split() if len(w) > 2 and w not in _STOPWORDS]


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
        if key in ("SUBJECT", "SURFACE", "LIGHT", "SETTING",
                   "ACTION", "MOTION", "CAMERA", "MOOD") and value:
            if value.lower() in ("unknown", "n/a", "none", "skip", "-"):
                continue
            # Guard against an answer that turns into a paragraph.
            facts[key] = " ".join(value.split()[:12])
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


EDIT_CLARIFY_SYSTEM = (
    "You make an image-editing request unambiguous. You are told what the "
    "picture contains. You use that only to say exactly which thing the "
    "request means.\n\n"
    "Rules, in order of importance:\n"
    "1. Change nothing about what is being asked for. If the request says "
    "make it red, the result is still red.\n"
    "2. Use the observations only to replace a vague reference with a "
    "specific one. 'the jacket' becomes 'the green field jacket' when the "
    "picture shows a green field jacket.\n"
    "3. Ignore every observation the request does not refer to. If the "
    "request is about a jacket, the wall, the floor and the lighting are "
    "irrelevant and must not appear.\n"
    "4. Add nothing else. No new objects, no style, no mood, no camera.\n"
    "5. End with 'leave everything else unchanged.'\n\n"
    "Reply with the rewritten instruction on one line and nothing else.\n\n"
    "Examples\n"
    "Observed: subject: a green field jacket; setting: a grey wall\n"
    "Request: make the jacket red\n"
    "change the green field jacket to red, leave everything else unchanged.\n\n"
    "Observed: subject: a beige ceramic teapot; surface: a linen cloth\n"
    "Request: remove the lid\n"
    "remove the lid from the beige ceramic teapot, leave everything else "
    "unchanged.\n"
)


def _clarify_edit(req_id: str, user_prompt: str, facts: dict[str, str],
                  writer: str | None = None) -> str:
    """Rewrite an edit request using what the picture shows.

    The picture is there to settle *which* thing is meant, and nothing more.
    The previous version pasted every observation into a template, so "make the
    jacket red" came back carrying the wall, the floor and the fabric weave --
    detail the editor never needed and, worse, was now being told to preserve.
    """
    if not facts:
        return _enrich_edit_instruction(user_prompt, facts)

    observed = "; ".join(f"{k.lower()}: {v}" for k, v in facts.items())
    try:
        raw = _write(
            req_id, EDIT_CLARIFY_SYSTEM,
            f"Observed: {observed}\nRequest: {user_prompt}",
            max_tokens=120, temperature=0.2, repo=writer,
        )
    except Exception as exc:
        log(req_id, f"writer unavailable, composing directly: {exc}", "warn")
        return _enrich_edit_instruction(user_prompt, facts)

    line = _clarified(user_prompt, raw)
    # A rewrite that lost the request, or that the writer refused, is worse
    # than the plain request: the user's own words already say what they want.
    return line or user_prompt


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


# Words that promise quality without describing anything. They survive from
# Stable Diffusion prompting, where they acted as style tokens; on a T5 encoder
# they consume attention and contribute nothing.
_EMPTY_MODIFIERS = (
    "beautiful", "stunning", "gorgeous", "amazing", "masterpiece", "best quality",
    "high quality", "highly detailed", "ultra detailed", "8k", "4k", "hdr",
    "award winning", "trending on artstation", "professional", "perfect",
)



# --------------------------------------------------------------------------
# Storyboard
# --------------------------------------------------------------------------

SHOTLIST_SYSTEM = (
    "You are a storyboard artist. You break prose into panels a camera could "
    "photograph.\n\n"
    "Rules:\n"
    "1. Never invent an event the story does not contain. You are dividing "
    "what is there, not writing more of it.\n"
    "2. Cover the whole story, in order, from the first beat to the last. Do "
    "not spend every panel on the ending.\n"
    "3. Each panel is one moment. No cuts inside a panel, no passage of time.\n"
    "4. Vary the shot sizes. A page of close-ups reads as flat as a page of "
    "wides.\n"
    "5. Group the panels into scenes. A scene is a continuous stretch in one "
    "place: when the story moves somewhere else, or time passes, a new scene "
    "begins. Number them from 1, in order, and give each a short title.\n"
    "6. A caption carries what the picture cannot: what is felt, known, or "
    "about to happen. Never write a caption that describes the panel -- the "
    "reader is looking at it.\n"
    "7. Every action must be something a camera records. A story beat that is "
    "heard, thought or felt has to become the visible thing that goes with "
    "it, or the panel is drawn as nonsense: \"hears the tap\" was rendered "
    "once as a flooded kitchen.\n\n"
    "Converting the invisible:\n"
    "  hears the tap running   -> turns toward the kitchen doorway\n"
    "  realises she is not alone -> stops still, head lifted\n"
    "  remembers the argument  -> stares past the camera, jaw set\n"
    "  the room feels wrong    -> stands in the doorway, not entering\n"
)


def _shotlist_instruction(story: str, count: int) -> str:
    return (
        f"Break this story into exactly {count} panels.\n\n"
        "Reply with only a JSON array of exactly "
        f"{count} objects, each with:\n"
        '  "shot": one of "wide", "medium", "close-up"\n'
        '  "subject": who or what is in frame\n'
        '  "action": what is happening, as a short phrase\n'
        '  "setting": where it takes place\n'
        '  "character_in_frame": true if the person we follow is visible in '
        'this panel, false if it shows something else\n'
        '  "caption": one short line to sit under the panel, in the voice of '
        'the story. Narration, not description -- the reader can already see '
        'the picture. Six to fourteen words. Empty string if the panel needs '
        'no words. It must belong to THIS panel: never a line about something '
        'that happens in a later one.\n'
        '  "scene": the number of the scene this panel belongs to, from 1\n'
        '  "scene_title": a short name for that scene, the same on every panel '
        'of it\n'
        '  "dialogue": an array of what is spoken aloud in this panel, each '
        '{"speaker": who says it, "text": the words}. Use the story\'s own '
        'words where it has them. Empty array when nobody speaks -- most '
        'panels. At most two lines, each under twelve words.\n\n'
        "No prose before or after the JSON.\n\n"
        f"Story:\n{story}"
    )


def _parse_shotlist(raw: str, count: int) -> list[dict[str, str]]:
    """Pull the panel array out of the model's reply.

    Models fence JSON in backticks and preface it with a sentence however
    firmly they are told not to, so the array is located rather than assumed.
    """
    body = raw.strip()
    if "```" in body:
        parts = body.split("```")
        if len(parts) > 1:
            body = parts[1]
            if body.lstrip().lower().startswith("json"):
                body = body.lstrip()[4:]
    start, end = body.find("["), body.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("the writer did not return a panel list")

    panels = json.loads(body[start:end + 1])
    if not isinstance(panels, list) or not panels:
        raise ValueError("the writer returned no panels")

    cleaned: list[dict[str, str]] = []
    for p in panels[:count]:
        if not isinstance(p, dict):
            continue
        shot = str(p.get("shot", "medium")).strip().lower()
        if shot not in ("wide", "medium", "close-up"):
            shot = "medium"
        # Whether the person we follow is in frame decides whether their
        # description belongs in the prompt at all. A panel whose subject is
        # a running tap, given the protagonist's description, draws her
        # running instead.
        in_frame = p.get("character_in_frame")
        cleaned.append({
            "shot": shot,
            "subject": str(p.get("subject", "")).strip(),
            "action": str(p.get("action", "")).strip(),
            "setting": str(p.get("setting", "")).strip(),
            # Default to showing them: a board is mostly about its character,
            # and a missing flag should not quietly write them out.
            "character_in_frame": True if in_frame is None else bool(in_frame),
            # Narration, kept short. A caption that restates the picture is
            # worse than none, and a long one stops being a caption.
            "caption": " ".join(str(p.get("caption", "")).split())[:120],
            # What is actually said aloud. Bubbles are drawn over the panel, so
            # a long line covers the picture it belongs to -- hence the cap.
            "dialogue": _clean_dialogue(p.get("dialogue")),
            # Which continuous stretch of story this belongs to. Pages are
            # broken on scene boundaries, so this decides the shape of the
            # finished thing more than any other field.
            "scene": max(1, int(p.get("scene", 1) or 1)),
            "scene_title": " ".join(str(p.get("scene_title", "")).split())[:60],
        })
    if not cleaned:
        raise ValueError("the writer returned no usable panels")
    return cleaned


# Fonts that ship with macOS, in order of preference. A board that falls back
# to PIL's bitmap default is unreadable at panel width, so this is worth being
# picky about.
_CAPTION_FONTS = (
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
)


def _caption_font(size: int) -> Any:
    from PIL import ImageFont

    for path in _CAPTION_FONTS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap(draw: Any, text: str, font: Any, width: int) -> list[str]:
    """Break a caption to fit the panel, by measurement rather than by count.

    Character counts guess wrong on a proportional face, and a caption that
    overruns the panel is worse than one that wraps early.
    """
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


# What a page holds. Four to six panels is the working range in print comics;
# fewer reads as padding, more forces every panel to be small and simple.
PANELS_PER_PAGE = 5
PAGE_MIN, PAGE_MAX = 3, 6


def _paginate(scenes: list[int]) -> list[list[int]]:
    """Break the panels into pages, on scene boundaries where possible.

    A scene is a continuous stretch in one place, and a page is the unit a
    reader takes in at once, so the two should agree: a scene that starts
    halfway down a page reads as a jump rather than a change. Scenes longer
    than a page are split across several; short ones sharing a location are
    not merged, because the break is doing work.

    `scenes` is the scene number of each panel, in order. Returns the panel
    indices for each page.
    """
    pages: list[list[int]] = []
    run: list[int] = []
    current: int | None = None

    for i, scene in enumerate(scenes):
        if current is not None and scene != current and run:
            pages.extend(_split_run(run))
            run = []
        current = scene
        run.append(i)
    if run:
        pages.extend(_split_run(run))

    # A single panel alone on a page is an accident, not a splash: pull it
    # back onto the previous page unless that would overfill it.
    merged: list[list[int]] = []
    for page in pages:
        if (len(page) == 1 and merged
                and len(merged[-1]) + 1 <= PAGE_MAX):
            merged[-1].extend(page)
        else:
            merged.append(page)
    return merged


def _split_run(run: list[int]) -> list[list[int]]:
    """Divide one scene into pages of a readable size, evenly.

    Evenly matters: chunking greedily leaves the last page of a long scene
    holding one or two panels, which reads as the scene trailing off.
    """
    n = len(run)
    if n <= PAGE_MAX:
        return [run]
    pages_needed = max(1, round(n / PANELS_PER_PAGE))
    pages_needed = max(pages_needed, (n + PAGE_MAX - 1) // PAGE_MAX)
    size = n / pages_needed
    out: list[list[int]] = []
    start = 0.0
    for k in range(pages_needed):
        end = size * (k + 1)
        out.append(run[round(start):round(end)])
        start = end
    return [p for p in out if p]


def _tiers(shots: list[str]) -> list[list[int]]:
    """Arrange one page's panels into tiers, as a comic page is built.

    Print pages are read as horizontal bands: three of them, holding one to
    three panels each. A wide shot takes its whole tier -- that is what makes
    it establishing, and a large panel is what slows a reader down. Tighter
    shots share, which speeds the page up. The alternation is the pacing.
    """
    tiers: list[list[int]] = []
    i = 0
    while i < len(shots):
        if shots[i] == "wide":
            tiers.append([i])
            i += 1
            continue
        # Take up to three consecutive tight shots, but never leave a single
        # panel stranded as the last tier of a page.
        group = [i]
        i += 1
        while i < len(shots) and shots[i] != "wide" and len(group) < 3:
            remaining = sum(1 for j in range(i, len(shots)) if shots[j] != "wide")
            if len(group) == 2 and remaining == 2:
                break   # leave two for the next tier rather than 3 + 1
            group.append(i)
            i += 1
        tiers.append(group)
    return tiers


def _fit(im: Any, w: int, h: int) -> Any:
    """Scale and centre-crop to fill a cell without distorting the drawing."""
    from PIL import Image

    scale = max(w / im.width, h / im.height)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                   Image.LANCZOS)
    left, top = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((left, top, left + w, top + h))


def _caption_box(draw: Any, text: str, font: Any, box: tuple[int, int, int, int],
                 line_h: int) -> None:
    """A narration box in the corner of a panel, as a comic sets one."""
    x, y, w, _ = box
    lines = _wrap(draw, text, font, w - 18)
    bh = len(lines) * line_h + 12
    draw.rectangle([x, y, x + w, y + bh], fill=(250, 249, 245),
                   outline=(20, 20, 22), width=2)
    for n, line in enumerate(lines):
        draw.text((x + 9, y + 6 + n * line_h), line, font=font, fill=(20, 20, 22))


def _bubble(draw: Any, text: str, speaker: str, font: Any, small: Any,
            cx: int, top: int, max_w: int, line_h: int, tail_down: bool) -> int:
    """A speech balloon with a tail, returning the height it used."""
    lines = _wrap(draw, text, font, max_w - 26)
    tw = max((draw.textlength(l, font=font) for l in lines), default=0)
    w = int(tw) + 26
    h = len(lines) * line_h + 16
    x0, y0 = cx - w // 2, top
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=min(16, h // 2),
                           fill=(252, 252, 250), outline=(20, 20, 22), width=2)
    # The tail points at whoever is speaking, which is why it has a direction.
    ty = y0 + h if tail_down else y0
    dy = 14 if tail_down else -14
    draw.polygon([(cx - 9, ty), (cx + 9, ty), (cx + 1, ty + dy)],
                 fill=(252, 252, 250), outline=(20, 20, 22))
    for n, line in enumerate(lines):
        lw = draw.textlength(line, font=font)
        draw.text((cx - lw / 2, y0 + 8 + n * line_h), line, font=font,
                  fill=(20, 20, 22))
    used = h + 18
    if speaker:
        # Below the balloon, not above it: narration boxes live at the top of
        # a panel, and a label set there collides with them and is unreadable.
        sw = draw.textlength(speaker, font=small)
        sy = y0 + h + (14 if tail_down else 0) + 3
        draw.rectangle([x0, sy, x0 + sw + 12, sy + 19], fill=(20, 20, 22))
        draw.text((x0 + 6, sy + 3), speaker, font=small, fill=(248, 248, 245))
        used += 24
    return used


def _render_page(panels: list[Any], idx: list[int], meta: dict[str, list[Any]],
                 style: dict[str, Any]) -> Any:
    """Draw one page of panels as tiers."""
    from PIL import Image, ImageDraw

    page_w, margin, gutter = style["width"], style["margin"], style["gutter"]
    font, small, line_h = style["font"], style["small"], style["line_h"]
    shots = [meta["shots"][i] for i in idx]
    tiers = _tiers(shots)
    inner = page_w - margin * 2

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    plan, total_h = [], margin
    for tier in tiers:
        cell_w = (inner - gutter * (len(tier) - 1)) // len(tier)
        # One panel across the page is an establishing beat, and is set
        # shallower so the page does not become a ladder of squares.
        cell_h = round(cell_w * (0.58 if len(tier) == 1 else 1.0))
        plan.append((tier, cell_w, cell_h))
        total_h += cell_h + gutter
    total_h += margin - gutter

    page = Image.new("RGB", (page_w, total_h), (243, 241, 236))
    draw = ImageDraw.Draw(page)
    y = margin
    for tier, cell_w, cell_h in plan:
        x = margin
        for local in tier:
            src = idx[local]
            page.paste(_fit(panels[src], cell_w, cell_h), (x, y))
            draw.rectangle([x, y, x + cell_w, y + cell_h],
                           outline=(20, 20, 22), width=3)
            cap = meta["captions"][src].strip()
            if cap:
                _caption_box(draw, cap, font,
                             (x + 10, y + 10, min(cell_w - 20, 430), 0), line_h)
            by = y + (78 if cap else 16)
            for line in meta["dialogue"][src][:2]:
                by += _bubble(draw, line.get("text", ""), line.get("speaker", ""),
                              font, small, x + cell_w // 2, by,
                              min(cell_w - 40, 380), line_h, tail_down=True)
            x += cell_w + gutter
        y += cell_h + gutter
    return page


def _render_strip(panels: list[Any], meta: dict[str, list[Any]],
                  style: dict[str, Any]) -> Any:
    """Draw the panels as one vertical scroll.

    Not a page stretched tall, which is the mistake this form is known for.
    A scroll reveals one moment at a time and the gap between panels is the
    tempo: a short gap reads as a beat, a long one as a held breath. So the
    gutter widens wherever the scene changes, and stays tight within one.
    """
    from PIL import Image, ImageDraw

    width, margin = style["width"], style["margin"]
    font, small, line_h = style["font"], style["small"], style["line_h"]
    beat, breath = 46, 190

    sized, gaps = [], []
    for i, im in enumerate(panels):
        sized.append(_fit(im, width - margin * 2, round((width - margin * 2) * 0.78)))
        if i:
            same = meta["scenes"][i] == meta["scenes"][i - 1]
            gaps.append(beat if same else breath)

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    caps = [_wrap(probe, meta["captions"][i].strip(), font, width - margin * 2 - 40)
            if meta["captions"][i].strip() else [] for i in range(len(panels))]

    total = margin + sum(im.height for im in sized) + sum(gaps)
    total += sum(len(c) * line_h + 18 for c in caps if c) + margin

    page = Image.new("RGB", (width, total), (243, 241, 236))
    draw = ImageDraw.Draw(page)
    y = margin
    for i, im in enumerate(sized):
        page.paste(im, (margin, y))
        draw.rectangle([margin, y, margin + im.width, y + im.height],
                       outline=(20, 20, 22), width=3)
        by = y + 16
        for line in meta["dialogue"][i][:2]:
            by += _bubble(draw, line.get("text", ""), line.get("speaker", ""),
                          font, small, margin + im.width // 2, by,
                          min(im.width - 40, 420), line_h, tail_down=True)
        y += im.height
        if caps[i]:
            for n, line in enumerate(caps[i]):
                draw.text((margin + 2, y + 10 + n * line_h), line, font=font,
                          fill=(30, 30, 33))
            y += len(caps[i]) * line_h + 18
        if i < len(gaps):
            y += gaps[i]
    return page


def op_compose_board(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Set the drawn panels as pages, or as one vertical scroll.

    Composed here rather than in the browser because the panels are sealed:
    this is the process that already holds the keys, and each finished page is
    sealed again before it touches disk.

    Pages break where scenes do and are laid out in tiers, which is how a
    print comic is read. The scroll is a different form with different rules,
    not the same page made tall.
    """
    from PIL import Image

    slots = req.get("vault_slots") or []
    if not slots:
        raise ValueError("composing a page needs a vault slot to write into")

    staged = _stage_vault_inputs(req.get("vault_inputs") or [])
    if not staged:
        raise ValueError("there are no drawn panels to compose")

    n = len(staged)

    def column(key: str, default: Any) -> list[Any]:
        vals = list(req.get(key) or [])
        vals += [default] * (n - len(vals))
        return vals[:n]

    meta = {
        "captions": [str(c or "") for c in column("captions", "")],
        "shots": [str(x or "medium") for x in column("shots", "medium")],
        "dialogue": [list(d or []) for d in column("dialogue", [])],
        "scenes": [int(x or 1) for x in column("scenes", 1)],
    }
    layout = str(req.get("layout") or "page")
    size = int(req.get("font_size", 19))
    style = {
        "width": int(req.get("page_width", 1240 if layout == "page" else 860)),
        "margin": int(req.get("margin", 34)),
        "gutter": int(req.get("gutter", 18)),
        "font": _caption_font(size),
        "small": _caption_font(13),
        "line_h": int(size * 1.4),
    }

    outputs: list[str] = []
    sizes: list[int] = []
    dims: list[list[int]] = []
    try:
        panels = []
        for path in staged:
            with Image.open(path) as im:
                panels.append(im.convert("RGB"))

        pages = (_paginate(meta["scenes"]) if layout == "page"
                 else [list(range(n))])
        if len(pages) > len(slots):
            raise ValueError(
                f"this board makes {len(pages)} pages but only {len(slots)} "
                "were reserved; ask for that many again"
            )

        import io

        import vaultcrypto as vc

        for k, idx in enumerate(pages):
            img = (_render_page(panels, idx, meta, style) if layout == "page"
                   else _render_strip(panels, meta, style))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()
            slot = slots[k]
            vc.write_sealed(slot["path"], bytes.fromhex(slot["key"]),
                            bytes.fromhex(slot["file_id"]), data)
            outputs.append(slot["id"])
            sizes.append(len(data))
            dims.append([img.width, img.height])
            log(req_id, f"sealed page {k + 1}/{len(pages)} "
                        f"({img.width}x{img.height}) into {slot['id']}")
    finally:
        _discard_staged(staged)

    return {"outputs": outputs, "sealed": True, "sizes": sizes,
            "dimensions": dims, "pages": len(outputs),
            "panels": n, "layout": layout}


PLACE_SYSTEM = (
    "You are the background artist on a comic. You are told everything that "
    "happens in one scene, and you describe the place it happens in -- once, "
    "so every panel of that scene can be drawn as the same room.\n\n"
    "Rules:\n"
    "1. Describe the place, not the action. No people, no events.\n"
    "2. Be specific about what does not change: the walls and what is on "
    "them, the floor, the furniture, the windows and what is beyond them, and "
    "where the light comes from.\n"
    "3. Name materials and colours. 'A room' is drawn differently every time; "
    "'dark green wallpaper with a repeating leaf pattern, bare oak boards' is "
    "not.\n"
    "4. Decide anything the scene leaves open -- that is the job -- but never "
    "contradict what it does say.\n"
    "5. Never write 'beautiful', 'cinematic' or 'atmospheric'.\n\n"
    "One paragraph, at most fifty words, no preamble.\n"
)


def _establish_places(req_id: str, panels: list[dict[str, Any]],
                      style: str, writer: str | None) -> dict[int, str]:
    """Describe each scene's location once, for every panel in it to share.

    Panels were described one at a time, so the same room came back as floral
    wallpaper in one and cracked plaster in the next -- nothing was holding
    the place still. Settling it per scene is what makes consecutive frames
    look like the same house.
    """
    by_scene: dict[int, list[dict[str, Any]]] = {}
    for p in panels:
        by_scene.setdefault(int(p.get("scene", 1) or 1), []).append(p)

    places: dict[int, str] = {}
    for n, (scene, group) in enumerate(sorted(by_scene.items())):
        if is_cancelled(req_id):
            raise Cancelled()
        emit({"id": req_id, "type": "progress", "phase": "denoise",
              "progress": n / max(len(by_scene), 1),
              "message": f"Settling the look of scene {scene}"})

        beats = "; ".join(
            f"{p.get('action', '')} ({p.get('setting', '')})".strip()
            for p in group if p.get("action") or p.get("setting")
        )
        title = next((p.get("scene_title") for p in group if p.get("scene_title")), "")
        try:
            raw = _write(req_id, PLACE_SYSTEM,
                         f"Style: {style}\nScene: {title}\nWhat happens here: {beats}",
                         max_tokens=150, temperature=0.4, repo=writer)
            places[scene] = " ".join(raw.strip().splitlines()[0].split())[:400]
        except Cancelled:
            raise
        except Exception as exc:
            log(req_id, f"scene {scene} place not settled: {exc}", "warn")
            places[scene] = ""
    return places


ENRICH_SYSTEM = (
    "You are a storyboard artist working up one panel into something that can "
    "be drawn.\n\n"
    "You are given the moment, the style the board is drawn in, and often the "
    "place it happens in. Write what the camera sees.\n\n"
    "Rules:\n"
    "1. Everything the panel already states must survive. You are adding to "
    "it, not replacing it.\n"
    "2. If you are told the place, it is already settled. Use it. Do not "
    "change the walls, the floor, the furniture or where the light comes "
    "from -- every panel of this scene is the same room, and a reader notices "
    "when it is not.\n"
    "3. Add only what the moment implies -- the surface a thing rests on, "
    "what is behind it, how the light falls on it. Never a new character, "
    "never an event.\n"
    "4. Say where the light is and what it is doing. A panel with no stated "
    "light is drawn with no light.\n"
    "5. Describe only what is visible. No sound, no thought, no dialogue.\n"
    "6. Never write 'beautiful', 'cinematic', 'dramatic', 'high quality' or "
    "'masterpiece'. They describe nothing.\n\n"
    "One paragraph, at most forty words, no preamble.\n\n"
    "Example\n"
    "Place: a narrow kitchen, dark green walls, bare oak boards, one window "
    "over the sink.\n"
    "Panel: medium shot. Anna stands at the kitchen window. kitchen.\n"
    "Anna at the window over the sink, rain running down the glass. Grey "
    "afternoon light through it, no lamp lit. Dark green walls behind her, "
    "bare oak boards underfoot, a cup steaming on the draining board.\n"
)


def op_enrich_panels(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Work each panel up into a scene the generator can draw.

    The division gives the moments, which are deliberately terse -- "stands in
    the hallway" is the right level for judging whether the story was cut
    correctly, and far too thin to draw. This is the stage between: it fills
    in surface, background and light, and nothing else. Kept separate so the
    result can be read and corrected before any of it is drawn, which is the
    whole point of a pipeline over a single button.
    """
    panels: list[dict[str, Any]] = list(req.get("panels") or [])
    if not panels:
        raise ValueError("divide the story into panels first")

    style = (req.get("style") or "").strip()
    writer = req.get("writer")
    # The place first, then the panels within it. Order matters: a panel
    # described before its room has nothing to be consistent with.
    places = _establish_places(req_id, panels, style, writer)
    out: list[dict[str, Any]] = []

    for i, p in enumerate(panels):
        if is_cancelled(req_id):
            raise Cancelled()
        emit({"id": req_id, "type": "progress", "phase": "denoise",
              "progress": (i / len(panels)) if panels else None,
              "message": f"Working up panel {i + 1} of {len(panels)}"})

        moment = ". ".join(x for x in (
            f"{p.get('shot', 'medium')} shot",
            str(p.get("subject", "")).strip(),
            str(p.get("action", "")).strip(),
            str(p.get("setting", "")).strip(),
        ) if x)
        place = places.get(int(p.get("scene", 1) or 1), "")
        user = (f"Style: {style}\n"
                + (f"Place, already settled: {place}\n" if place else "")
                + f"Panel: {moment}.")
        try:
            raw = _write(req_id, ENRICH_SYSTEM, user, max_tokens=140,
                         temperature=0.4, repo=writer)
        except Cancelled:
            raise
        except Exception as exc:
            log(req_id, f"panel {i + 1} not enriched: {exc}", "warn")
            out.append({**p, "description": "",
                        "place": places.get(int(p.get("scene", 1) or 1), "")})
            continue

        text = " ".join(raw.strip().splitlines()[0].split())
        # An enrichment that dropped the moment is worse than none: the panel
        # would be drawn as something the story does not contain.
        if text and not _keeps_intent(moment, text):
            log(req_id, f"panel {i + 1} enrichment lost the moment; keeping it plain", "warn")
            text = ""
        out.append({**p, "description": text[:400],
                    "place": places.get(int(p.get("scene", 1) or 1), "")})

    return {"panels": out}


def _clean_dialogue(raw: Any) -> list[dict[str, str]]:
    """Normalise spoken lines, and keep them short enough to letter.

    A bubble sits on top of the panel it belongs to. Two of them, or one long
    one, and there is no panel left to see -- so this caps both.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw[:2]:
        if isinstance(item, str):
            speaker, text = "", item
        elif isinstance(item, dict):
            speaker = str(item.get("speaker", ""))
            text = str(item.get("text", ""))
        else:
            continue
        text = " ".join(text.split()).strip('"\u201c\u201d')
        if not text:
            continue
        out.append({"speaker": " ".join(speaker.split())[:40], "text": text[:90]})
    return out


def op_shotlist(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Turn a story into an ordered list of panels.

    The riskiest step in the picture-board pipeline, because everything after
    it is only as good as the division of the story. A 2B vision-language
    model could not do it -- asked for four panels it returned one, and
    skipped most of the story -- so this runs on the writer.
    """
    story = (req.get("story") or "").strip()
    if not story:
        raise ValueError("write the story first, then break it into panels")

    count = max(2, min(int(req.get("panels", 6)), 24))
    emit({"id": req_id, "type": "progress", "phase": "denoise",
          "progress": None, "message": f"Breaking the story into {count} panels"})

    raw = _write(req_id, SHOTLIST_SYSTEM, _shotlist_instruction(story, count),
                 max_tokens=180 * count, temperature=0.3, repo=req.get("writer"))
    panels = _parse_shotlist(raw, count)
    if len(panels) < count:
        log(req_id, f"asked for {count} panels, got {len(panels)}", "warn")
    return {"panels": panels, "asked": count}


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
        # Both of these read the picture first. An edit needs to know which
        # jacket is meant; a clip starting from a photograph needs to know
        # what is in it before it can say how it moves.
        editing = mode == "edit" and bool(staged)
        animating = mode == "video" and bool(staged)

        def ask(text: str, images: list[str], max_tokens: int, temperature: float) -> str:
            # Only the picture-reading path needs the vision model, and it is
            # loaded here rather than up front so a plain text request never
            # pays for it.
            model, processor, config = _load_assistant(req_id, repo)
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
            improved = _clarify_edit(req_id, user_prompt, facts,
                                     req.get("writer"))
        else:
            emit({"id": req_id, "type": "progress", "phase": "denoise",
                  "progress": None, "message": "Writing the prompt"})
            system = (MOTION_DIRECTION_SYSTEM if mode == "video"
                      else SCENE_DIRECTION_SYSTEM)
            seen = ""
            if animating:
                # What the picture holds, so the motion belongs to it rather
                # than to the words alone.
                emit({"id": req_id, "type": "progress", "phase": "denoise",
                      "progress": None, "message": "Looking at your picture"})
                facts = _parse_scene(ask(SCENE_SYSTEM, staged, 90, 0.1))
                seen = "; ".join(f"{k.lower()}: {v}" for k, v in facts.items())
                description = seen
            # The writer, not the picture-reader: this is a writing task.
            raw = _write(req_id, system,
                         (f"The picture shows: {seen}\n" if seen else "")
                         + f"Request: {user_prompt}",
                         repo=req.get("writer"))
            improved = _clarified(user_prompt, raw)
            description = ""
            if improved is None:
                # The request names nothing that could be drawn, so there is
                # nothing to make precise. Inventing a subject to fill the gap
                # is what made this useless before.
                return {
                    "prompt": user_prompt, "original": user_prompt,
                    "saw_image": False, "unclear": True, "description": "",
                    "note": ("This does not say what to draw yet. Name the "
                             "thing you want and I can make it specific."),
                }

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
    modes += [c.mode for c in (getattr(caps, "restoration", None) or ())]
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
        # Upscaling is not a generation capability. It lives in a separate
        # `restoration` field, and reading only `capabilities` is what made
        # SeedVR2 look like it had no modes at all -- which is how four
        # working upscalers ended up marked broken.
        modes += [c.mode for c in (getattr(caps, "restoration", None) or ())]
        # Map MLX-Gen's internal modes onto the app's four task buckets.
        if any(m in ("restore-image", "restore-video") for m in modes):
            tasks.append("upscale")
        if any(m in ("text-only", "text-to-image") for m in modes):
            tasks.append("text_to_image")
        if any(m in ("edit-reference", "multi-reference", "latent-img2img") for m in modes):
            tasks.append("edit")
        # Generating a clip, not restoring one. "restore-video" contains the
        # word and means the opposite: SeedVR2 upscales footage it is given
        # and cannot generate a frame of its own.
        if any("video" in m and not m.startswith("restore-") for m in modes):
            tasks.append("video")
    except Exception as exc:
        route_error = str(exc)

    size = 0
    private = False
    gated = False
    params = 0
    hf_info: Any = None
    try:
        from huggingface_hub import HfApi

        info = hf_info = HfApi().model_info(repo, files_metadata=True)
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

    kind = "model"
    backend: str | None = None

    if not tasks and hf_info is not None and _looks_like_adapter(hf_info):
        # A LoRA, not a model. Downloading it as one produces a folder that can
        # never generate anything, so say what it is instead of "unsupported".
        kind = "lora"
        route_error = (
            "This is a LoRA adapter, not a complete model. Add it under "
            "\"Style adapters\" on the Generate panel instead, where it can be "
            "applied on top of a model you already have."
        )
    elif not tasks:
        # The unified router could not place it. FLUX.1 is a separate lineage
        # that mflux runs, so ask that backend before giving up.
        backend = _detect_mflux_backend(
            repo,
            [str(t) for t in (getattr(hf_info, "tags", None) or [])],
            [sib.rfilename for sib in (getattr(hf_info, "siblings", None) or [])],
        )
        if backend:
            tasks = ["text_to_image"]
            if backend in ("dev_kontext", "dev_fill"):
                tasks = ["edit"]
            route_error = None

    if gated and not _hf_token():
        # Being gated is not the same as being unsupported, and the fix is
        # something only the user can do. Say so rather than sending them off
        # to look for a different model.
        route_error = (
            f"This model is gated. Open huggingface.co/{repo} , accept its "
            "licence, then add a Hugging Face access token under Settings. "
            "The download will work after that."
        )
        return {
            "model": repo, "family": family, "backend": backend, "kind": kind,
            "modes": modes, "tasks": tasks, "bytes": size, "params": params,
            "private": private, "gated": True, "routable": False,
            "needs_token": True, "error": route_error,
        }

    if route_error and "infer a supported backend" in route_error.lower():
        route_error = (
            "This model is not one the engine can run. It supports the FLUX.1 "
            "and FLUX.2, Qwen-Image, Z-Image, ERNIE, FIBO, Bonsai and Wan "
            "families. Stable Diffusion and SDXL are a different architecture "
            "and are not included."
        )

    return {
        "model": repo,
        "family": family,
        "backend": backend,
        "kind": kind,
        "modes": modes,
        "tasks": tasks,
        "bytes": size,
        "params": params,
        "private": private,
        "gated": gated,
        "routable": bool(tasks),
        "needs_token": False,
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
    # Not every model id is a repository id. The catalog lists Bernini as
    # "bernini-r-1.3b", which MLX-Gen resolves to ByteDance/Bernini-R-1.3B-
    # Diffusers -- so watching a directory named after the id showed 0.00 GB
    # for an entire 16 GiB download. Measuring the cache as a whole works
    # whatever the id turns out to resolve to, and downloads hold their own
    # lane one at a time, so any growth is this one.
    baseline = _dir_size(root)

    emit({"id": req_id, "type": "progress", "phase": "download", "progress": 0.0,
          "total_bytes": expected, "done_bytes": already})

    stop = threading.Event()

    def measured() -> int:
        direct = _dir_size(repo_dir)
        if direct > already:
            return direct
        # The id was an alias: fall back to how much the cache has grown.
        return already + max(0, _dir_size(root) - baseline)

    def poll() -> None:
        # Rate and stall are measured here rather than inferred in the UI,
        # because only this side knows when the bytes actually moved. A
        # download that quietly stopped used to look exactly like a slow one:
        # one of them ran for 79 minutes before anyone noticed.
        last_bytes, last_moved, last_t = already, time.time(), time.time()
        while not stop.wait(0.7):
            done = measured()
            now = time.time()
            if done > last_bytes:
                rate = (done - last_bytes) / max(now - last_t, 1e-6)
                last_bytes, last_moved, last_t = done, now, now
            else:
                # Chunked transfer writes in bursts, so a gap is normal until
                # it is long. Hold the last known rate rather than showing 0.
                rate = 0.0
            stalled = now - last_moved
            remaining = max(expected - done, 0) if expected else 0
            frac = (done / expected) if expected else None
            emit({
                "id": req_id, "type": "progress", "phase": "download",
                "progress": min(frac, 0.999) if frac is not None else None,
                "total_bytes": expected, "done_bytes": done,
                "bytes_per_second": round(rate),
                "stalled_seconds": round(stalled),
                "eta_seconds": round(remaining / rate) if rate > 0 else None,
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
        elif via == "mflux":
            # FLUX.1. Ask mflux which files it actually reads rather than
            # pulling the repository whole: the official repos also ship a
            # single-file copy of the transformer, so a blind snapshot fetches
            # roughly 24 GB it will never open.
            from huggingface_hub import snapshot_download

            from mflux.models.flux.weights.flux_weight_definition import (
                FluxWeightDefinition,
            )

            patterns = list(FluxWeightDefinition.get_download_patterns())
            # The tokenizers load through a separate path with its own
            # patterns; they are a few MB, so fetch them in the same pass.
            patterns += ["tokenizer/*", "tokenizer_2/*", "scheduler/*",
                         "model_index.json"]
            snapshot_download(repo_id=repo_id, allow_patterns=patterns,
                              token=_hf_token())
        elif via == "hf":
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo_id, token=_hf_token())
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

    size = measured()
    # A downloader that reports success while fetching nothing must not be
    # recorded as an installed model.
    if size < 1024 * 1024:
        raise RuntimeError(
            f"{repo_id} reported success but nothing was written to the cache. "
            "The repository may not exist, be gated, or need a different downloader."
        )
    # Record that this one finished. Judging completeness by comparing bytes
    # against a published figure only ever approximates it -- the engine
    # fetches a subset, and the published figure is sometimes just wrong -- so
    # the downloader says so outright instead of leaving it to be inferred.
    landed = repo_dir
    if not landed.exists():
        # An aliased id: the weights are under the repository name MLX-Gen
        # resolved it to, so mark whichever cache entry was just written.
        try:
            fresh = [d for d in root.glob("models--*") if d.is_dir()]
            landed = max(fresh, key=lambda d: d.stat().st_mtime)
        except (OSError, ValueError):
            landed = repo_dir
    try:
        (landed / ".melp-complete").write_text(str(size))
    except OSError as exc:
        log(req_id, f"could not mark {repo_id} complete: {exc}", "warn")

    emit({"id": req_id, "type": "progress", "phase": "download", "progress": 1.0,
          "total_bytes": expected or size, "done_bytes": size})
    return {"model": repo_id, "path": str(landed), "bytes": size}


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
    if _helpers_resident() is not None:
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

# How each video mode wants its picture. Video models disagree not just on the
# parameter name but on what a picture *means*: a first frame to animate, a
# subject to reference, or a source clip to transform. Sending the wrong one
# is not a near miss -- a reference is a loose hint, so the result is an
# unrelated clip rather than an obviously wrong one.
_VIDEO_IMAGE_KWARG = {
    "first-frame-i2v": "image_path",
    "reference-video": "reference_image_paths",
    "reference-video-edit": "reference_image_paths",
}


def _video_image_kwarg(req_id: str, model: str,
                       family: str | None) -> tuple[str | None, list[str]]:
    """Which parameter carries a still picture into this video model.

    Returns the parameter name and the model's modes. `None` means this model
    cannot start from a picture at all -- Wan VACE, for instance, transforms an
    existing clip and takes no still. Asking beforehand is the difference
    between refusing and quietly generating something unrelated.
    """
    from mlxgen import get_model_capabilities

    try:
        caps = (get_model_capabilities(model=model, family=family) if family
                else get_model_capabilities(model=model))
        modes = [c.mode for c in caps.capabilities]
    except Exception as exc:
        log(req_id, f"could not read video capabilities: {exc}", "warn")
        return None, []

    for mode in modes:
        if mode in _VIDEO_IMAGE_KWARG:
            return _VIDEO_IMAGE_KWARG[mode], modes
    return None, modes


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

        image_kwarg: str | None = None
        if staged:
            image_kwarg, modes = _video_image_kwarg(
                req_id, req["model"], req.get("family"))
            if image_kwarg is None:
                raise ValueError(
                    f"This model cannot start from a picture. It supports "
                    f"{', '.join(modes) or 'text prompts'} only. Generating "
                    "anyway would ignore your image and return an unrelated "
                    "clip, so nothing was run. Choose a model listed as image "
                    "to video."
                )
            if image_kwarg == "reference_image_paths":
                plan_kw["reference_image_count"] = len(staged)
            else:
                plan_kw["image_count"] = len(staged)

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
        if staged and image_kwarg:
            # Exactly the parameter this model declared, never both. Sending
            # both and letting the tolerance layer sort it out is what broke
            # this: Wan VACE answers "does not take image_path; pass
            # reference_image_paths instead", both were dropped in turn, and
            # the run continued as plain text-to-video with the picture
            # silently discarded.
            optional[image_kwarg] = (
                list(staged) if image_kwarg.endswith("_paths") else staged[0]
            )
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

        # `call_tolerant` drops what a route refuses, which is right for a
        # quality knob and wrong for the picture the whole request was about.
        # If it went, the clip that came back has nothing to do with what was
        # asked for, so this fails rather than returning it.
        if staged and image_kwarg and image_kwarg not in optional:
            raise RuntimeError(
                "This model refused the picture, so the clip it produced is "
                "unrelated to it. Nothing was saved. Try a model listed as "
                "image to video."
            )

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
    "shotlist": op_shotlist,
    "enrich_panels": op_enrich_panels,
    "compose_board": op_compose_board,
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

    # A second lane for network work. Fetching weights is pure network I/O: it
    # never touches Metal and never holds a model, so there is no reason for a
    # 20-minute download to block a probe or a generation queued behind it.
    # Downloads still serialise against each other -- one extra lane, not one
    # thread per request -- so two cannot thrash the same disk and network.
    fetch: "queue.Queue[dict[str, Any]]" = queue.Queue()

    def pump(q: "queue.Queue[dict[str, Any]]") -> None:
        while True:
            req = q.get()
            if req is None:
                return
            handle(req)

    worker_thread = threading.Thread(target=pump, args=(work,), daemon=True)
    worker_thread.start()
    fetch_thread = threading.Thread(target=pump, args=(fetch,), daemon=True)
    fetch_thread.start()
    threading.Thread(target=_idle_reaper, daemon=True).start()

    def drain(timeout: float) -> None:
        # Let queued work finish rather than dropping it silently; the host is
        # waiting on a terminal event for every request it sent.
        work.put(None)
        fetch.put(None)
        worker_thread.join(timeout=timeout)
        fetch_thread.join(timeout=timeout)

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
        # Network-only work goes to its own lane so it cannot stall the app.
        (fetch if op in ("download", "probe", "resolve") else work).put(req)

    # stdin closed: the host is gone. Give in-flight work a moment to unwind.
    drain(5.0)


if __name__ == "__main__":
    main()
