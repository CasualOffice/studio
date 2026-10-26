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
import atexit
import gc
import io
import json
import math
import struct
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import tempfile
import uuid
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


def _apply_policy_once(req_id: str) -> None:
    """Bound the process before any weights are allocated.

    A function rather than the same four lines in each loader, because the
    upscale route had neither: both `_load_model` and `_load_mflux_model` set
    the flag themselves and SeedVR2 is constructed directly, so the one
    operation heavy enough to abort the engine -- a 1024x1024 restore peaks at
    16.5 GiB on a 16 GB machine -- was the one running with no MLX cache
    ceiling at all.
    """
    global _POLICY_APPLIED
    if _POLICY_APPLIED:
        return
    _POLICY_APPLIED = True
    log(req_id, f"resource policy: {apply_resource_policy()}")


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
    # What the repository is plainly not. "One .safetensors file" was treated
    # as evidence of an adapter, and a VAE repository is exactly that -- which
    # is how kpsss34/Flux-VAE-HDR was offered, installed and selected as an
    # adapter, and only refused by mflux at draw time with "did not match any
    # known adapter keys". Metadata cannot prove a file IS an adapter, but it
    # can recognise the things that certainly are not; the header check after
    # download is what actually decides.
    haystack = " ".join([getattr(info, "id", "") or ""] + weights + tags).lower()
    if any(word in haystack for word in
           ("vae", "text_encoder", "text-encoder", "clip-", "/clip", "tokenizer",
            "controlnet", "upscaler", "refiner")):
        return False
    return (
        any("lora" in f.lower() for f in weights)
        or any("lora" in t or "adapter" in t for t in tags)
        or len(weights) == 1
    )


# Below this, starting a model run is close to certain to abort the engine.
# Above it, macOS reclaims enough that the reported figure means very little:
# a 9.72 GiB run completed here with 5.2 GiB reported free. Set from that
# evidence, not from the size of the model.
_MIN_FREE_GIB = 1.5


def _free_ram_gib() -> float:
    """Memory actually available right now, in GiB.

    Free pages plus everything the OS can reclaim without swapping: inactive
    pages and the compressor's own purgeable pages. Wired and active memory is
    genuinely spoken for and is not counted.

    Returns -1.0 when it cannot be determined, which callers treat as "do not
    block" -- a guard that guesses wrong and refuses real work is worse than
    no guard.
    """
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=3).stdout
    except Exception:
        return -1.0
    page = 4096
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        page = int(m.group(1))
    vals = {}
    for line in out.splitlines():
        mm = re.match(r'"?([A-Za-z][A-Za-z \-]+?)"?:\s+(\d+)\.', line)
        if mm:
            vals[mm.group(1).strip().lower()] = int(mm.group(2))
    if "pages free" not in vals:
        return -1.0
    # Free plus inactive, and nothing else. "File-backed" and "anonymous"
    # pages are subsets of active and inactive, so adding them double-counts:
    # an earlier version reported 10.1 GiB on a machine with 5.9 GiB really
    # available, which is exactly the direction that makes the guard useless.
    reclaimable = vals.get("pages free", 0) + vals.get("pages inactive", 0)
    return reclaimable * page / (1024 ** 3)


def _check_headroom(req_id: str, need_gib: float) -> None:
    """Refuse before dispatch when the memory plainly is not there.

    A Metal out-of-memory is not an exception this process can survive: MLX
    reports it from a command-buffer completion handler, the C++ exception
    reaches std::terminate, and the engine aborts. The app then says "the
    engine stopped unexpectedly", which tells nobody anything.

    The catalog's peak says whether a model fits an *empty* machine. It cannot
    know that something else is using the memory right now -- another app, a
    browser, or a second copy of this one -- and contention is what actually
    kills these runs. So this asks the machine.
    """
    free = _free_ram_gib()
    if free < 0:
        return
    if need_gib > 0:
        log(req_id, f"{free:.1f} GiB reported free, this run needs about "
                    f"{need_gib:.1f} GiB")

    # Deliberately not "free < need". Free-plus-inactive badly understates what
    # macOS can reclaim: a 9.72 GiB image-to-video run completed on this
    # machine with 5.2 GiB reported free, because the rest was compressed or
    # evicted as it went. Refusing on that comparison would block almost every
    # real run.
    #
    # What actually killed the engine was a second process holding the memory,
    # and in every one of those cases the figure here was near zero. So this
    # only refuses when the machine is genuinely out, where an abort is close
    # to certain, and says nothing otherwise.
    if free >= _MIN_FREE_GIB:
        return
    raise ValueError(
        f"Only {free:.1f} GiB of memory is free, which is not enough to start "
        f"anything. Something else on this Mac is holding it \u2014 closing "
        f"that, or waiting for it to finish, will let this run. Nothing was "
        f"started, because running out part-way stops the engine rather than "
        f"failing cleanly."
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

    _apply_policy_once(req_id)

    # Free whatever is resident *before* pulling the new one into memory --
    # including the assistant, which competes for the same budget.
    CACHE.unload()
    if _helpers_resident() is not None:
        log(req_id, "releasing the prompt helper to make room for the model")
        _unload_assistant()
    log(req_id, f"loading {model} (quantize={quantize})")
    emit({"id": req_id, "type": "progress", "phase": "load", "progress": 0.0,
          "message": f"Loading {model}"})

    # The adapters themselves, not just the fact that there are some.
    #
    # `has_lora` in `plan_kw` only tells the router to choose a route that can
    # take an adapter; it is the loader that attaches one. Passing the first
    # without the second is why picking a style adapter changed nothing on
    # every unified-route model -- Qwen-Image, Z-Image, FLUX.2, Wan -- while
    # the log said "adapters: ... @ 1.0" and the cache key change forced a
    # multi-gigabyte reload to produce the identical image. Only the mflux
    # route ever built these.
    #
    # Required rather than optional on purpose: `resolve_generation_runtime`
    # has already refused the families that cannot take an adapter, so a route
    # reaching this point and then rejecting the keyword is a real error. Let
    # it fail loudly instead of dropping the user's adapter and drawing
    # something that looks like it worked.
    adapters: dict[str, Any] = {}
    if loras:
        for spec in loras:
            _check_is_adapter(req_id, str(spec.get("path") or ""))
        adapters["lora_paths"] = [l["path"] for l in loras]
        adapters["lora_scales"] = [float(l.get("scale", 1.0)) for l in loras]

    t0 = time.monotonic()
    with _downloads_allowed(req_id):
        loaded = call_tolerant(
            req_id,
            load_generation_model,
            {
                "model": model,
                "quantize": quantize,
                "model_path": model_path,
                "image_count": image_count,
                **adapters,
                **plan_kw,
            },
            # Release the text conditioner once the prompt is encoded. On routes
            # with a large conditioner this removes its resident size from peak.
            {"model_kwargs": {"release_text_encoder": True}} if release_text_encoder else {},
        )
    CACHE.put(key, loaded, model)
    load_ms = (time.monotonic() - t0) * 1000.0
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


def _remove_stage_root() -> None:
    """Remove this process's plaintext staging directory on clean exit."""
    global _STAGE_ROOT
    if _STAGE_ROOT:
        shutil.rmtree(_STAGE_ROOT, ignore_errors=True)
        _STAGE_ROOT = None


def _stage_owner_pid(name: str) -> int | None:
    """The worker a staging directory belonged to, from its name."""
    parts = name.split("-")
    if len(parts) >= 3 and parts[0] == "msvault" and parts[1].isdigit():
        return int(parts[1])
    return None


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, and owned by somebody else. Not ours to remove.
        return True
    except OSError:
        return True
    return True


def _scavenge_stale_stage_dirs(max_age_seconds: int = 24 * 60 * 60) -> None:
    """Remove private staging left by a worker that is gone.

    This holds briefly-decrypted copies of vault images, so how long it may
    outlive the process that made it is a security question, not housekeeping.

    It used to keep everything for a day: only directories older than
    `max_age_seconds` were removed, on the reasoning that a second running app
    instance must never be disturbed. But a day is precisely the window that
    matters -- a worker killed mid-edit left decrypted pictures in /tmp and the
    next start walked past them -- and SECURITY.md says the opposite in as many
    words: "stale remnants from a process crash are scavenged at the next
    engine start". That was not true, and the gap was the whole exposure.

    The directory carries the pid that made it, so ownership is a fact rather
    than a guess: a directory whose worker is no longer running is removed at
    once, whatever its age, and one whose worker is alive is left alone however
    old it is. The age rule stays only for names from before this change, which
    carry no pid.
    """
    now = time.time()
    root = Path(tempfile.gettempdir())
    mine = os.getpid()
    for candidate in root.glob("msvault-*"):
        try:
            if candidate.stat().st_uid != os.getuid():
                continue
            owner = _stage_owner_pid(candidate.name)
            if owner == mine:
                continue
            if owner is not None:
                if not _process_alive(owner):
                    shutil.rmtree(candidate, ignore_errors=True)
                continue
            # No pid in the name: an older build made it. Fall back to age.
            if now - candidate.stat().st_mtime >= max_age_seconds:
                shutil.rmtree(candidate, ignore_errors=True)
        except OSError:
            continue


def _install_exit_handlers() -> None:
    """Remove the staging directory on the signals that actually arrive.

    `atexit` alone was not enough. It runs when the interpreter exits of its
    own accord and is skipped entirely by SIGTERM, whose default disposition
    terminates the process without unwinding -- and SIGTERM is exactly what the
    host sends when the app quits. So quitting during an edit left decrypted
    vault images on disk, and the scavenger above then declined to remove them
    for a day.

    The handlers re-raise through the default disposition after cleaning up, so
    the exit status still says what happened and the host is not left waiting.
    """
    def handler(signum: int, _frame: Any) -> None:
        _remove_stage_root()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # Not the main thread, or the platform will not take it. `atexit`
            # still covers the ordinary exit.
            continue


_scavenge_stale_stage_dirs()
atexit.register(_remove_stage_root)
_install_exit_handlers()


def _stage_dir() -> str:
    """A private directory for briefly-decrypted source images."""
    global _STAGE_ROOT
    if _STAGE_ROOT is None or not os.path.isdir(_STAGE_ROOT):
        # The pid is in the name so a later start can tell a directory
        # whose worker is gone from one that is still in use.
        _STAGE_ROOT = tempfile.mkdtemp(prefix=f"msvault-{os.getpid()}-")
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
            try:
                safe_id = str(uuid.UUID(str(entry["id"])))
            except (KeyError, ValueError, AttributeError) as exc:
                raise ValueError("vault input has an invalid id") from exc
            with open(entry["path"], "rb") as fh:
                sealed = fh.read()
            plain = vc.open_with_file_key(bytes.fromhex(entry["key"]), sealed)
            # Keep the original extension so PIL can sniff the format.
            suffix = str(entry.get("ext") or "png").lower()
            if not re.fullmatch(r"[a-z0-9]{1,8}", suffix):
                suffix = "png"
            dest = os.path.join(_stage_dir(), f"{safe_id}.{suffix}")
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
        _WRITER.get("key") if _WRITER.get("model") is not None else (
            "audiobook voices" if _VOICE.get("model") is not None else None)
    )


def _load_assistant(req_id: str, repo: str) -> tuple[Any, Any, Any]:
    if _ASSIST["key"] == repo and _ASSIST["model"] is not None:
        return _ASSIST["model"], _ASSIST["processor"], _ASSIST["config"]

    # Bound the process before any weights are allocated. Every other
    # allocating route does this and the two helper loaders did not, so the
    # 3.2 GiB writer was the one load in the engine running with no MLX cache
    # ceiling -- on the route that is reached first, from a text box, on a
    # machine already holding whatever the user was working on.
    _apply_policy_once(req_id)

    # One model resident, always: give up the generator before taking memory.
    if CACHE.loaded is not None:
        log(req_id, f"releasing {CACHE.label} to make room for the assistant")
        CACHE.unload()
    # And the other helper. `_helpers_resident` exists because there are two of
    # them, and every guard that made room for "the assistant" was written when
    # there was one -- but the guard between the two helpers themselves was
    # never written at all. Reading a picture and then writing about it left
    # both resident, 2B and 4B together, which is the one-resident rule broken
    # by the only pair that always runs back to back.
    if _WRITER["model"] is not None:
        log(req_id, "releasing the prompt writer to make room for the reader")
        _unload_writer()

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
# Writers that reason before answering unless told not to. Left on, a scene's
# JSON arrives after thousands of tokens of thinking, or not at all.
THINKING_WRITERS = {
    "mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-14B-4bit",
    "mlx-community/Qwen3.5-9B-MLX-4bit",
}


def _load_writer(req_id: str, repo: str | None = None) -> tuple[Any, Any]:
    repo = repo or WRITER_REPO
    if _WRITER["key"] == repo and _WRITER["model"] is not None:
        return _WRITER["model"], _WRITER["tokenizer"]

    if _WRITER["model"] is not None:
        # Drop the previous analyst before loading a different one.
        _unload_writer()

    # See `_load_assistant`: the ceiling, then the generator, then the other
    # helper. This is the larger of the two models and the one that was
    # allocating unbounded.
    _apply_policy_once(req_id)

    if CACHE.loaded is not None:
        log(req_id, f"releasing {CACHE.label} to make room for the writer")
        CACHE.unload()
    if _ASSIST["model"] is not None:
        log(req_id, "releasing the picture reader to make room for the writer")
        _release_assistant_only()

    from mlx_lm import load as lm_load

    emit({"id": req_id, "type": "progress", "phase": "load", "progress": 0.0,
          "message": "Loading the prompt writer"})
    model, tokenizer = lm_load(repo)
    _WRITER.update(key=repo, model=model, tokenizer=tokenizer)
    log(req_id, f"writer {repo} ready")
    return model, tokenizer


def _write(req_id: str, system: str, user: str, max_tokens: int = 160,
           temperature: float = 0.3, repo: str | None = None,
           label: str = "Writing", fraction: float | None = None) -> str:
    """One turn with the writer.

    Streamed rather than taken in one call, for two reasons.

    Cancel did nothing. `generate` returns only when the whole answer is
    finished -- up to six thousand tokens for a story division, which is
    minutes -- and there was no point in that window where the request could
    be stopped. Every text step in the app had a Cancel button that was inert:
    dividing a story, working panels up, reading the cast, improving a prompt.

    And the wait had no face. A step that emits nothing for two minutes is
    indistinguishable from one that has hung, which is the complaint that
    started all of this.
    """
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _load_writer(req_id, repo)
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        add_generation_prompt=True,
        **({"enable_thinking": False} if (repo or WRITER_REPO) in THINKING_WRITERS else {}),
    )
    pieces: list[str] = []
    last = 0.0
    written = 0
    for response in stream_generate(model, tokenizer, prompt=prompt,
                                    max_tokens=max_tokens,
                                    sampler=make_sampler(temp=temperature)):
        if is_cancelled(req_id):
            raise Cancelled()
        pieces.append(response.text)
        written += 1
        now = time.time()
        # Throttled: a token arrives every few milliseconds and the channel is
        # shared with everything else.
        if now - last >= 0.2:
            last = now
            # No step total: `max_tokens` is a ceiling, not the work, and was
            # shown as "step 512/3200" on a reply that ends near 600.
            emit({"id": req_id, "type": "progress", "phase": "denoise",
                  "progress": fraction, "message": label,
                  # The words so far. A spinner and a token count say the
                  # machine is busy; they do not say it is writing something
                  # worth waiting for, and eight seconds of a spinner feels
                  # longer than eight seconds of watching a sentence arrive.
                  # Only the tail, because the interface shows a line, not a
                  # transcript, and the channel is shared with everything else.
                  "text": "".join(pieces)[-400:]})
    return "".join(pieces)


def _unload_writer() -> None:
    _WRITER.update(key=None, model=None, tokenizer=None)
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def _release_assistant_only() -> None:
    """Drop the picture reader and leave the writer alone.

    `_unload_assistant` releases both, which is what the idle path and the
    generator loader want. It is not what the writer's own loader wants: there
    the reader is being replaced *by* the writer, and unloading the writer on
    the way to loading it is at best wasted and at worst a loop.
    """
    _ASSIST.update(key=None, model=None, processor=None, config=None)
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def _unload_assistant() -> None:
    _unload_writer()
    _VOICE.update(model=None)
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
    "You are an art director writing the prompt for an image model.\n\n"
    "First work out what the person actually wants to see. They are describing "
    "a picture in their head, often in a hurry, often badly: misspelled, "
    "ungrammatical, half a sentence. Read through that to the picture. Then "
    "write the prompt you would give an illustrator who cannot ask "
    "questions.\n\n"
    "Say it in your own words. You are not editing their sentence, you are "
    "describing what they meant -- so rephrase freely, fix the grammar, and "
    "use the right word where they reached for the nearest one. What must "
    "survive is the thing they asked for, not the words they asked in.\n\n"
    "Work through the fields below, then write them out as one flowing "
    "sentence or two. Everything the fields cover is yours to decide -- and "
    "you must decide it, because a prompt that leaves light, lens and framing "
    "unsaid gets whatever the model felt like.\n\n"
    "  SUBJECT      who or what, with material, age, wear\n"
    "  APPEARANCE   what they are wearing or made of, in detail\n"
    "  ACTION       what is happening, in this instant\n"
    "  SETTING      where it is, and what else is in the frame\n"
    "  COMPOSITION  framing, camera height, what is near and what is far\n"
    "  LIGHT        source, direction, quality, colour\n"
    "  STYLE        medium and finish\n\n"
    "Rules:\n"
    "1. Keep the subject and every attribute given to it. If they say a red "
    "car, it is still a car and still red -- a sedan is a car, a motorbike is "
    "not.\n"
    "2. Correct the spelling and the grammar, always. 'a hous at nite' is a "
    "house at night. 'me and him was walking' is two people walking.\n"
    "3. Never write 'beautiful', 'stunning', 'high quality', '8k', "
    "'cinematic' or 'masterpiece'. They describe nothing.\n"
    "4. No field labels in the answer. One paragraph, 40 to 80 words.\n"
    "5. If the request names no thing at all -- 'make it better', 'something "
    "nice' -- you cannot direct it. Reply with exactly: UNCLEAR\n\n"
    "Reply with the prompt and nothing else.\n\n"
    "Examples\n"
    "Request: a cat on a chair\n"
    "A tabby cat with dense grey-brown fur lies curled on a worn oak chair, "
    "one paw tucked under its chin, tail hanging over the seat edge. Late "
    "afternoon sun comes low through a window to the left, warming the "
    "chair's arm and throwing a long soft shadow across the floorboards. "
    "Close, slightly above eye level, the room falling away out of focus "
    "behind. Oil on canvas, visible brushwork.\n\n"
    "Request: a pictuer of a hous at nite\n"
    "A narrow brick house stands at the end of a wet street at night, paint "
    "flaking from its window frames, one upstairs light burning behind a thin "
    "curtain. The road is black and mirror-slick, holding the orange smear of "
    "a streetlamp just out of frame. Shot from across the road at standing "
    "height, the house small in a wide, quiet frame. Photographic, long "
    "exposure, grain in the shadows.\n\n"
    "Request: make it look better\n"
    "UNCLEAR\n"
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

# How much longer a clarification may reasonably be than the request.
#
# These caps were set to stop the model writing its own scene from nothing,
# and measured against eight ordinary requests they rejected five good
# rewrites -- "a red car with sun-faded paint and a dented wing, water running
# down the windows" was thrown away for being too long. A three word request
# becoming twenty-five words is not the model overreaching, it is the entire
# reason someone pressed the button.
#
# The case they were guarding against is now caught earlier and better: a
# request that names nothing returns UNCLEAR before any of this runs, and the
# retention check below still guarantees nothing the user asked for is lost.
# So these are loose, and retention does the real work.
# Direction, not paraphrase. A four-word request becoming seventy words of
# subject, light, framing and medium is the whole job now, not evidence of
# overreach -- so the ceiling is a sanity bound on runaway output rather than a
# limit on richness. Retention below is what keeps the rewrite honest.
_MAX_EXPANSION = 60

# How many more things a clarification may name than the request did. Enough
# that a bowl of ramen may have its broth, noodles, egg and nori -- which was
# rejected outright before -- while a request that comes back naming a dozen
# unrelated objects is still refused.
# Likewise. An art direction names the room, the window, the light and the
# floor; counting those as "new things" and refusing is how the enhancer ended
# up writing thirteen words when it was asked for a picture.
_MAX_NEW_THINGS = 20


# Endings English adds to a word that is still the same word. Ordered longest
# first so "running" loses "ing" rather than "g".
# No "ies": the trailing-i rule below is what handles it, and stripping all
# three letters defeated that rule. "carries" became "carr" while "carrying"
# became "carry", so the two inflections of one verb did not agree -- which is
# the single thing this function exists to make happen, and the docstring's own
# worked example ("carries" -> "carri" -> "carry") describes stripping "es".
_INFLECTIONS = ("ing", "ed", "es", "s")


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


# Why a rewrite was not offered, in words a person can act on. The reason
# matters: "your prompt was already fine" and "I wrote something worse than
# yours" are different facts, and reporting the first when the second is true
# is what makes this feature look like it does nothing.
_WHY_REJECTED = {
    "dropped": "The rewrite lost something you asked for, so it was not used.",
    "lost_intent": "The rewrite drifted away from what you asked for.",
    "too_long": "The rewrite invented far more than you described.",
    "rejected": "The rewrite added things you did not ask for.",
    "empty": "The writer returned nothing usable.",
}


def _added_words(original: str, improved: str) -> list[str]:
    """Significant words the rewrite introduced.

    Shown so a proposal can be read at a glance instead of diffed by eye. Only
    words that carry content -- the point is to answer "what did it add?",
    which is the first thing anyone asks of a suggestion.
    """
    had = {_stem(w) for w in _significant(original)}
    seen: set[str] = set()
    out: list[str] = []
    for w in _significant(improved):
        st = _stem(w)
        if st in had or st in seen:
            continue
        seen.add(st)
        out.append(w)
    return out


def _first_line(raw: str) -> str:
    """The line of a reply that is the answer, not the preamble."""
    for candidate in raw.strip().splitlines():
        candidate = candidate.strip().strip("`").strip()
        if not candidate or candidate.lower().startswith(("request:", "here", "sure")):
            continue
        return candidate.rstrip(" .,;") + "."
    return ""


def _clean_prompt_request(text: str) -> tuple[str, list[str]]:
    """Remove prompt folklore that does not describe visible content.

    This runs before the writer, not after it, so the intent guard evaluates
    only meaningful words and does not reject a better prompt merely because
    it correctly dropped "8k" or "masterpiece".
    """
    cleaned = text
    removed: list[str] = []
    for phrase in sorted(_EMPTY_MODIFIERS, key=len, reverse=True):
        pattern = re.compile(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])",
                             re.IGNORECASE)
        if pattern.search(cleaned):
            removed.append(phrase)
            cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s*[,;]\s*(?=[,;]|$)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;.-")
    return cleaned, removed


def _edits(a: str, b: str) -> int:
    """Levenshtein distance, for deciding whether one word is the other
    misspelled."""
    if a == b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    row = list(range(len(a) + 1))
    for j, cb in enumerate(b, start=1):
        prev, row[0] = row[0], j
        for i, ca in enumerate(a, start=1):
            prev, row[i] = row[i], min(row[i] + 1, row[i - 1] + 1,
                                       prev + (ca != cb))
    return row[-1]


# Deliberate misspellings, which are not typos and are not close enough in
# edit distance to be treated as any. People type these on purpose.
_SLANG = {
    "nite": "night", "lite": "light", "thru": "through", "tho": "though",
    "foto": "photo", "fotos": "photos", "wanna": "want", "gonna": "going",
    "cuz": "because", "ur": "your", "u": "you", "pls": "please",
    "kool": "cool", "kolor": "colour", "color": "colour", "gray": "grey",
    "nyte": "night", "brite": "bright", "rite": "right", "luv": "love",
}


# The system word list, used to tell a typo from a different word.
#
# Edit distance alone cannot: "tabel" -> "table" and "horse" -> "house" are
# both one or two edits with the same first letter, and the first is the
# correction the user wants while the second silently redraws the picture. What
# separates them is that "tabel" is not a word and "horse" is. Two attempts at
# this without a dictionary were both worse than the bug -- the first let "a
# cat on a chair" become "a hat on a chair" and called it a spelling fix.
_WORDS_FILE = "/usr/share/dict/words"
_DICTIONARY: set[str] | None = None


def _dictionary() -> set[str]:
    """The word list, read once. Empty if the file is not there."""
    global _DICTIONARY
    if _DICTIONARY is None:
        try:
            with open(_WORDS_FILE, encoding="utf-8", errors="ignore") as fh:
                _DICTIONARY = {
                    w for w in (line.strip().lower() for line in fh)
                    if 2 < len(w) <= 20
                }
        except OSError:
            # No dictionary, no guessing: fall back to exact and prefix
            # matching, which is where this started and is never wrong, only
            # sometimes strict.
            _DICTIONARY = set()
    return _DICTIONARY


def _is_word(word: str) -> bool:
    """Whether this is English, allowing for the way people actually write it.

    `/usr/share/dict/words` holds base forms and nothing else. `cars`, `trees`,
    `houses`, `hands`, `holds`, `stands` and `wears` are all absent from it, so
    a check for membership alone answers "not a word" for most of the plural
    nouns and third-person verbs in any real prompt.

    That answer went straight into the typo machinery, whose first question is
    whether the typed word is a word: it is not, so it is a candidate to be
    corrected. "trees in the mist" rewritten as "tree trunks in the mist"
    reported trees -> tree as a spelling the enhancer had fixed, and "hands
    holding a cup" against "hand holds a cup" reported two of them. The user is
    shown their own correct English, labelled as a mistake.

    So an inflection of a word is a word. Stemming, and also the base form with
    a doubled final consonant undone ("holds" -> "hold", "running" -> "run"),
    because that is the other shape English inflections take.
    """
    words = _dictionary()
    if not words:
        return False
    w = word.lower()
    if w in words:
        return True
    stem = _stem(w)
    if stem != w and stem in words:
        return True
    # "runing"/"running" -> "run": the stem keeps the doubled consonant that
    # the base form does not have.
    if len(stem) > 2 and stem[-1] == stem[-2] and stem[:-1] in words:
        return True
    # A plain -s plural on a word that already ends in e. `_stem` tries the
    # suffixes in order and "es" matches first, so "trees" stems to "tre" and
    # the word is not found -- which is the exact case that reported
    # trees -> tree to the user as a spelling correction.
    if w.endswith("s") and len(w) > 3 and w[:-1] in words:
        return True
    # "carries" -> "carri" -> "carry" is handled inside `_stem`; the -es plural
    # of a word ending in a consonant is not.
    return w.endswith("es") and len(w) > 4 and w[:-2] in words


def _corrections(original: str, improved: str) -> list[list[str]]:
    """The spellings the rewrite fixed, as typed-then-written pairs.

    Shown rather than slipped in. Someone who typed "hous" should see that it
    became "house", both so they can tell the enhancer did something and so
    they can object if it guessed wrong.
    """
    written = [g.lower() for g in _significant(improved)]
    pairs: list[list[str]] = []
    for w in _significant(original):
        word = w.lower()
        # An inflection of a word is a word. A bare membership test is not:
        # see `_is_word`, and the plurals it stops being offered as typos.
        if not _dictionary() or _is_word(word):
            continue
        for candidate in written:
            if candidate != word and _misspelling_of(word, candidate):
                pairs.append([word, candidate])
                break
    return pairs


def _misspelling_of(typed: str, candidate: str) -> bool:
    """Whether `typed` is `candidate` with a typing mistake in it.

    A real word is never treated as a misspelling of another real word, which
    is the whole guard: the rewrite may fix what you typed wrong, and may not
    quietly change what you typed right.
    """
    if typed == candidate:
        return True
    if _SLANG.get(typed) == candidate:
        return True
    words = _dictionary()
    if not words:
        return False
    # The same guard from both sides. `typed` being an inflection makes it a
    # word, so it is not a misspelling of anything; `candidate` being one makes
    # it a real target rather than a word the list happens not to carry.
    if _is_word(typed) or not _is_word(candidate):
        return False
    if len(typed) < 4 or typed[0] != candidate[0]:
        return False
    # Two edits once there is something to work with, so a transposition like
    # "tabel" counts; one for the shortest words, where two edits is most of
    # the word.
    limit = 2 if max(len(typed), len(candidate)) >= 5 else 1
    return abs(len(typed) - len(candidate)) <= limit \
        and _edits(typed, candidate) <= limit


# Words that carry no subject of their own, for the retention check only.
#
# Two kinds. The medium -- "picture", "photo", "drawing" -- is not a thing in
# the scene: "a pictuer of a hous" and "a photograph of a house" are the same
# request, and insisting the rewrite keep the word "picture" rejects the
# correction. And pronouns and auxiliaries, because "me and him was walking"
# becoming "two people walking" is the grammar fix being asked for, not a loss.
_NOT_A_SUBJECT = {
    "picture", "pictur", "pic", "photo", "photograph", "image", "img",
    "drawing", "draw", "painting", "paint", "render", "shot", "art",
    "artwork", "illustration", "wallpaper",
    "him", "her", "his", "hers", "she", "he", "they", "them", "their",
    "was", "were", "been", "has", "had", "have", "did", "does", "doing",
    "walking",
    # Words people instruct with rather than describe with. "my dog but make
    # him look like a king" is four content words and five of these, and
    # counting them as things the rewrite lost refused every rendering of it.
    "but", "look", "looks", "looking", "like", "make", "makes", "making",
    "turn", "turns", "into", "show", "shows", "want", "wants", "kind",
    "sort", "type", "style", "version", "also", "then", "just", "really",
    # And the placeholders for a thing not yet named. A request built on one
    # is still a request -- "something scary in an old house" is a house --
    # and UNCLEAR above already catches the case where it is the *only* thing.
    "something", "somthing", "anything", "stuff", "things", "thing",
}


def _dropped_words(original: str, line: str) -> list[str]:
    """Significant words in the request that the rewrite does not carry.

    The comparison is by stem and by prefix in both directions, so "rise" and
    "rises" agree, and "rainy" is satisfied by "rain".
    """
    # Whole words as well as stems: the dictionary check has to see "sitting",
    # not the stem "sitt", to know it is a word.
    line_words = [g.lower() for g in _significant(line)]
    kept = {_stem(g) for g in line_words}
    out: list[str] = []
    for w in _significant(original):
        word = w.lower()
        # The medium is not a subject, misspelled or not: "a pictuer of a
        # hous" and "a photograph of a house" are the same request.
        if any(word == n or _misspelling_of(word, n) for n in _NOT_A_SUBJECT):
            continue
        stem = _stem(word)
        # A prefix match with no bound on the difference is not a stem match,
        # it is "starts with the same letters" -- and a longer word that starts
        # with the same letters is a different thing, not an inflection of this
        # one. Unbounded, the check accepted every one of these as intact:
        #
        #   a cat on a chair    -> a catalogue on a chair
        #   a red car           -> a red cardigan
        #   a man on a boat     -> a mango on a boat
        #   2 red cars          -> three red carts
        #
        # which is exactly what `_misspelling_of` below exists to stop, arrived
        # at from the other side: cat -> hat is caught and cat -> catalogue was
        # not.
        #
        # One letter, not zero. Zero fails two of this file's own tests: "a man
        # runs" against "running", and the inflection cases in
        # `RepairDroppedWords`. That leaves one substitution still accepted --
        # "cars" against "carts", which agree after stemming to within a letter
        # -- and one letter of slack is the price of not calling every English
        # verb a dropped word. The bound is the tested contract; the residue is
        # known and is not claimed to be closed.
        if any(stem == k
               or ((k.startswith(stem) or stem.startswith(k))
                   and abs(len(k) - len(stem)) <= 1)
               for k in kept):
            continue
        # A corrected spelling is the same word, not a missing one.
        if any(_misspelling_of(word, g) for g in line_words):
            continue
        out.append(w)
    return out


def _repair_dropped(original: str, line: str) -> str | None:
    """Put back what the rewrite left out, rather than throwing it away.

    A rewrite that drops a word is usually a paraphrase, not a deletion: asked
    for "a rainy street" the writer returns wet asphalt and slick pavement and
    never says rain. Rejecting that loses a good description over a word that
    is arguably still there, and the user sees their prompt come back
    unchanged and concludes the feature does nothing.

    So the missing words are appended and the result re-checked. The wording
    is plainer than the model's own, which is the right trade: the picture has
    to contain what was asked for, and no amount of better prose is worth
    losing it.
    """
    missing = _dropped_words(original, line)
    # A word whose near-spelling is already in the rewrite is not missing, it
    # is disagreed with: asked for a cat "setting" on a chair the writer wrote
    # "sitting", which is almost certainly what was meant. Appending the
    # original produced "...sitting on a worn oak chair, setting." -- the word
    # dangling off the end, meaning nothing, in every rewrite that corrected a
    # homophone. Better to leave it rejected and show the attempt, so the
    # choice between "setting" and "sitting" is made by the person who typed
    # it rather than by a string that satisfies a check.
    written = [g.lower() for g in _significant(line)]
    if any(_edits(w.lower(), g) <= 2 and w.lower()[:1] == g[:1]
           and len(g) >= 5 for w in missing for g in written):
        return None
    if not missing or len(missing) > 2:
        # One word missing is a paraphrase; two is a paraphrase with a
        # compound in it. Three or more is the rewrite ignoring a clause --
        # "an old bicycle against a brick wall" coming back as a bicycle --
        # and pasting the clause on the end would paper over exactly the
        # failure this check exists to catch. Those are still rejected, and
        # the caller draws again instead.
        return None
    fixed = line.rstrip(" .,;") + ", " + " ".join(missing) + "."
    return fixed if not _dropped_words(original, fixed) else None


def _clarified(original: str, raw: str) -> str | None:
    """The rewrite, or None. Callers that need to know why use the pair."""
    return _clarified_with_reason(original, raw)[0]


def _clarified_with_reason(original: str, raw: str) -> tuple[str | None, str]:
    """Take the model's rewrite, or reject it.

    Returns the rewrite and why it was or was not accepted. The reason
    matters to the caller: "unclear" is a fact about the request and the user
    can act on it, while every other rejection is a fact about the *rewrite* --
    the request was fine and the model's answer was not. Reporting those as
    "this does not say what to draw yet" tells people their prompt is broken
    when it is not, which is worse than saying nothing at all.

    The old version answered vague requests by inventing a subject -- "make it
    look better" produced a painting in a dark room with a large window, none
    of which the user had asked for -- and with a T5 encoder every invented
    noun is something the image actually contains.
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
        return None, "empty"
    if line.strip().upper().startswith("UNCLEAR"):
        return None, "unclear"

    # Deliberately not trimmed to a sentence. The model was asked for one
    # line, and that trim falls back to the last clause when there is no full
    # stop -- which cut "a tabby cat..., curled on a worn oak chair" at the
    # comma, deleting the chair and failing the retention check below.
    line = line.rstrip(" .,;") + "."
    if not _keeps_intent(original, line):
        return None, "lost_intent"

    # A word the rewrite no longer says is reported, not refused.
    #
    # This used to reject. It made rephrasing impossible: "me and him was
    # walking in forrest" cannot become good English without losing "him" and
    # "was", and "a pictuer of a hous" cannot be fixed without losing
    # "pictuer". The guard made sense when the rewrite was applied the instant
    # it existed -- silently swapping the user's words was worth preventing.
    # It is now shown for approval with the changes named, so vetoing a good
    # rewrite on behalf of someone who can see it is the wrong call. The
    # subject test above still refuses a rewrite about something else.
    dropped = _dropped_words(original, line)
    # The same filtered view retention uses. Taken raw, the head word of
    # "somthing scary in a old hosue" is "somthing" -- a placeholder for a
    # thing not yet named -- and every rewrite that named one was refused for
    # not repeating it.
    kept = [w for w in _significant(original)
            if not any(w.lower() == n or _misspelling_of(w.lower(), n)
                       for n in _NOT_A_SUBJECT)]
    # Losing a modifier is rephrasing. Losing the subject is a different
    # picture. "a cat on a chair" coming back as "a hat rests on a chair" keeps
    # the chair and passes any test that asks whether *some* of the request
    # survived, so the head word is checked by name and a rewrite that loses
    # half of what was asked for is refused however well it reads.
    head_lost = bool(kept) and kept[0].lower() in {w.lower() for w in dropped}
    if dropped and (head_lost or len(dropped) * 2 >= len(kept)):
        return None, "dropped"
    log_reason = "rephrased" if dropped else "ok"

    # A rewrite far longer than the request is inventing, not clarifying.
    # The floor matters as much as the ratio: "a teapot" is one significant
    # word, and naming its glaze, its spout and its wear is a legitimate
    # clarification that a ratio alone would reject.
    if len(_significant(line)) > max(90, _MAX_EXPANSION * len(_significant(original))):
        return None, "too_long"

    # Length alone cannot tell a described teapot from a list of furniture.
    # Each "a ..." is another thing in the picture, so counting them catches
    # what the ratio cannot: adding attributes keeps the count flat, while
    # inventing objects drives it up one article at a time.
    def things(text: str) -> int:
        return len(re.findall(r"\b(?:a|an|the)\s+[a-z]", text.lower()))

    if things(line) - things(original) > _MAX_NEW_THINGS:
        return None, "rejected"

    # A count alone cannot separate a described room from a list of its
    # contents: an eighty-word direction legitimately names the coat, the
    # window, the lamp and the floor. What separates them is shape. "a cat, a
    # book, a window, a rug, a fireplace, a lamp..." is almost entirely
    # article-noun pairs, while a direction spends most of its words on what
    # those things look like and how they are lit.
    density = things(line) / max(1, len(_significant(line)))
    if things(line) > 6 and density > 0.45:
        return None, "rejected"
    # "rephrased" when the wording moved off the user's words, "ok" when it did
    # not. Both are accepted; the caller names the difference so the person
    # approving it can see what changed.
    return line, log_reason


def _carries(part: str, text: str) -> bool:
    """Whether `text` still says what `part` said.

    Every content word of `part` has to be represented in `text`, allowing for
    inflection -- "pours the tea" is carried by "pouring tea into a white cup"
    but not by "a narrow kitchen, dark green walls". Used on the one field at a
    time, because that is the grain at which a brief loses the story: it keeps
    the room and drops the thing happening in it.

    Deliberately not `_keeps_intent`, which asks whether *any* word survived
    and so can never fail on a real paragraph.
    """
    wanted = [_stem(w) for w in _significant(part)]
    if not wanted:
        return True
    got = {_stem(w) for w in _significant(text)}
    return all(stem in got for stem in wanted)


SAME_REQUEST_SYSTEM = (
    "You are checking whether a rewritten image prompt still asks for the "
    "same picture as the original request.\n\n"
    "The wording will be different -- that is the point of the rewrite. Added "
    "detail about light, framing, materials or style is fine. Judge only "
    "this: would someone who asked for the original be satisfied by a picture "
    "of the rewrite?\n\n"
    "A dog shown as a golden retriever is the same request. A dog shown as a "
    "cat is not. A house shown as a derelict mansion is the same request. A "
    "house shown as a ship is not.\n\n"
    "Answer with one word: SAME or DIFFERENT."
)


def _same_request(req_id: str, original: str, rewritten: str,
                  writer: str | None) -> bool:
    """Ask the writer whether a rewrite it cannot be string-matched to still holds.

    The literal checks cannot see that "a golden retriever in a velvet robe
    and a gold crown" is "my dog but make him look like a king" -- it repeats
    neither "dog" nor "king". Refusing on that basis threw away exactly the
    rewrites worth having, because the better the rephrasing the fewer of the
    original words it repeats.

    Costs one short answer. Wrong answers fail closed: anything that is not a
    clear SAME leaves the rewrite refused and shown for approval, which is
    where a literal failure sent it anyway.
    """
    try:
        verdict = _write(
            req_id, SAME_REQUEST_SYSTEM,
            f"Original: {original}\nRewrite: {rewritten}",
            max_tokens=6, temperature=0.0, repo=writer,
            label="Checking the rewrite",
        )
    except Cancelled:
        # Pressing Cancel is not "the check failed". Swallowed here, it became
        # a DIFFERENT verdict, the rewrite was refused for drifting, and the
        # job carried on to write a reply about a rewrite the user had already
        # stopped waiting for. Cancellation is the caller's to handle.
        raise
    except Exception as exc:
        log(req_id, f"could not check the rewrite: {exc}", "warn")
        return False
    return verdict.strip().upper().startswith("SAME")


def _keeps_intent(original: str, rewritten: str) -> bool:
    """A rewrite that loses the request is worse than no rewrite.

    Matched on four-character prefixes, which a typo defeats: "hosue" and
    "house" share three. So a misspelled request whose rewrite spelled it
    correctly read as a rewrite about something else, and the correction was
    thrown away for being one -- the same bug the retention check had, in the
    one place it had not been fixed.

    This is a floor, not a guarantee: one surviving word passes it. The check
    that every content word survived is `_carries`, and this is deliberately
    not that. What this stops is a rewrite about something else entirely.

    Its prefix match is deliberately left unbounded, unlike the one in
    `_dropped_words`. Bounding it by length was tried and does not work here:
    "red" -> "reddish" is a legitimate four-letter growth and "man" -> "mango"
    is a two-letter substitution, so no length threshold separates them. The
    place to catch a swapped subject is `_dropped_words`, which compares
    stems -- and it does.
    """
    def words(text: str) -> set[str]:
        cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
        return {w for w in cleaned.split()
                if len(w) > 2 and w not in _STOPWORDS and w not in _NOT_A_SUBJECT}

    wanted = words(original)
    if not wanted:
        return True
    got = words(rewritten)
    return any(
        any(w.startswith(g[:4]) or g.startswith(w[:4]) or _misspelling_of(w, g)
            for g in got)
        for w in wanted
    )




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
                  writer: str | None = None) -> tuple[str, str, str]:
    """Rewrite an edit request using what the picture shows.

    The picture is there to settle *which* thing is meant, and nothing more.
    The previous version pasted every observation into a template, so "make the
    jacket red" came back carrying the wall, the floor and the fabric weave --
    detail the editor never needed and, worse, was now being told to preserve.

    Returns the instruction to use, why the rewrite was or was not taken, and
    the line the writer actually produced. The reason used to be dropped here:
    a rejected rewrite came back indistinguishable from a request that needed
    no work, so `op_assist` answered "Nothing to add." after having tried one
    and thrown it away.
    """
    if not facts:
        return _enrich_edit_instruction(user_prompt, facts), "composed", ""

    observed = "; ".join(f"{k.lower()}: {v}" for k, v in facts.items())
    try:
        raw = _write(
            req_id, EDIT_CLARIFY_SYSTEM,
            f"Observed: {observed}\nRequest: {user_prompt}",
            max_tokens=120, temperature=0.2, repo=writer,
        )
    except Cancelled:
        # See `_same_request`: a cancelled write is not an unavailable writer,
        # and treating it as one composed an instruction and went on working.
        raise
    except Exception as exc:
        log(req_id, f"writer unavailable, composing directly: {exc}", "warn")
        return _enrich_edit_instruction(user_prompt, facts), "composed", ""

    line, why = _clarified_with_reason(user_prompt, raw)
    # A rewrite that lost the request, or that the writer refused, is worse
    # than the plain request: the user's own words already say what they want.
    # The reason travels out with it so the caller can report a rejection as
    # one rather than as a request that was already good enough.
    return (line or user_prompt), why, _first_line(raw)


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
# Words that describe a wish rather than a picture. Removing them before the
# rewrite keeps the intent guard honest: it should judge whether the subject
# survived, not whether "8k" did.
#
# Deliberately narrow. "beautiful", "professional" and "perfect" were in this
# list and are not folklore -- they modify the subject. "a professional
# kitchen" is a kind of kitchen, "a beautiful ruin" is a judgement the picture
# has to carry, and deleting them changed what was asked for while reporting
# that nothing had been removed. Only terms that name a rendering target, an
# award, or a resolution belong here.
_EMPTY_MODIFIERS = (
    "masterpiece", "best quality", "high quality", "highly detailed",
    "ultra detailed", "ultra realistic", "8k", "4k", "uhd", "hdr",
    "award winning", "award-winning", "trending on artstation",
    "artstation", "unreal engine", "octane render",
)



# --------------------------------------------------------------------------
# Storyboard
# --------------------------------------------------------------------------

def _as_scene(value: Any) -> int:
    """Scene number from whatever the writer put in the field.

    It is asked for a number and mostly gives one, but "two", "Scene 3" and
    "" all turn up. int() on those raised, and the raise was not caught per
    panel, so one odd value threw away a whole division the user had waited
    minutes for.
    """
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        return max(1, min(999, int(value)))
    digits = re.search(r"\d+", str(value or ""))
    if digits:
        return max(1, min(999, int(digits.group())))
    words = ("one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten")
    lowered = str(value or "").lower()
    for n, word in enumerate(words, start=1):
        if re.search(rf"\b{word}\b", lowered):
            return n
    return 1


# Fonts that ship with macOS, in order of preference. A board that falls back
# to PIL's bitmap default is unreadable at panel width, so this is worth being
# picky about.
_CAPTION_FONTS = (
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
)

# Every face above is Latin-only, and a face with no glyph for a character
# letters it as .notdef -- an empty box. So a Japanese, Chinese, Korean,
# Arabic, Hebrew or Devanagari caption was drawn as rows of identical boxes,
# with nothing in the pipeline to notice: the wrap measures those boxes and
# they are the same width as real glyphs.
#
# The ranges are only what decides which file to open, not every block a
# script can reach. Latin is absent on purpose -- it is the default -- and so
# is anything the faces above already cover.
_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("kana", ((0x3040, 0x30FF), (0x31F0, 0x31FF))),
    ("han", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))),
    ("hangul", ((0x1100, 0x11FF), (0x3130, 0x318F),
                (0xA960, 0xA97F), (0xAC00, 0xD7AF))),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
                (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("hebrew", ((0x0590, 0x05FF), (0xFB1D, 0xFB4F))),
    ("devanagari", ((0x0900, 0x097F), (0xA8E0, 0xA8FF))),
)

# Faces that ship with macOS and cover those scripts, most preferred first.
# "Arial Unicode" is last in every list deliberately: it is the only one that
# covers all of them, so a caption mixing two scripts finds it as the only
# file the two lists share.
#
# Kana leads with the Japanese face and Han with the Chinese one, so a
# Japanese sentence -- which is kana and han together -- keeps Japanese kanji
# forms rather than being lettered in the Simplified shapes a reader would
# notice. Hangul's list is kept clear of the CJK faces for the same reason:
# Hiragino has no hangul at all.
_SCRIPT_FONTS: dict[str, tuple[str, ...]] = {
    "kana": ("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
             "/System/Library/Fonts/Hiragino Sans GB.ttc",
             "/System/Library/Fonts/STHeiti Medium.ttc",
             "/System/Library/Fonts/AppleSDGothicNeo.ttc",
             "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    "han": ("/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    "hangul": ("/System/Library/Fonts/AppleSDGothicNeo.ttc",
               "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
               "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    "arabic": ("/System/Library/Fonts/SFArabic.ttf",
               "/System/Library/Fonts/GeezaPro.ttc",
               "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    "hebrew": ("/System/Library/Fonts/SFHebrew.ttf",
               "/System/Library/Fonts/Supplemental/NewPeninimMT.ttc",
               "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    "devanagari": ("/System/Library/Fonts/Kohinoor.ttc",
                   "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
                   "/System/Library/Fonts/Supplemental/ITFDevanagari.ttc",
                   "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
}

# Faces opened so far, keyed by file and size. A page asks for the same face
# once per caption and once per balloon, and the wide-coverage fallbacks are
# tens of megabytes each. Only the one compute thread composes a board, so
# this is never shared between renders in flight.
_FONT_CACHE: dict[tuple[str, int], Any] = {}


def _open_font(path: str, size: int) -> Any | None:
    """One face at one size, opened once per process. None if unusable."""
    key = (path, size)
    if key not in _FONT_CACHE:
        from PIL import ImageFont

        try:
            _FONT_CACHE[key] = ImageFont.truetype(path, size)
        except Exception:
            _FONT_CACHE[key] = None
    return _FONT_CACHE[key]


def _caption_font(size: int) -> Any:
    from PIL import ImageFont

    for path in _CAPTION_FONTS:
        font = _open_font(path, size)
        if font is not None:
            return font
    return ImageFont.load_default()


def _scripts(text: str) -> list[str]:
    """Which scripts in this text the Latin caption faces cannot letter."""
    found: list[str] = []
    for name, ranges in _SCRIPT_RANGES:
        if any(any(lo <= ord(ch) <= hi for lo, hi in ranges) for ch in text):
            found.append(name)
    return found


def _lettering(text: str, size: int, default: Any) -> Any:
    """The face to letter this text with: `default` unless it cannot.

    Chosen per piece of lettering rather than per board, because one page can
    carry a Japanese caption and a Latin balloon, and a face picked for the
    page would letter one of them in boxes.
    """
    needed = _scripts(text)
    if not needed:
        return default
    lists = [_SCRIPT_FONTS[name] for name in needed]
    for path in lists[0]:
        if not all(path in other for other in lists[1:]):
            # A caption in two scripts needs one face that has both, and the
            # shared entry is what finds it.
            continue
        font = _open_font(path, size)
        if font is not None:
            return font
    # Boxes are bad; no caption at all would be worse.
    return default


def _has_rtl(text: str) -> bool:
    """Whether any character belongs to a right-to-left script."""
    return any(
        0x0590 <= ord(ch) <= 0x08FF or 0xFB1D <= ord(ch) <= 0xFDFF
        or 0xFE70 <= ord(ch) <= 0xFEFF
        for ch in text
    )


def _shape_for_display(text: str) -> str:
    """Arabic and Hebrew as they are read, not as they are stored.

    Finding the right font was only half of it. A face with the glyphs still
    draws them in logical order and in isolated forms, so Arabic came out as
    disconnected letters running the wrong way -- legible to nobody, and
    arguably worse than the empty boxes it replaced, because it looks like
    text that has been rendered rather than text that has been dropped.

    Pillow does this itself when it is built against libraqm, and the wheel
    this app installs is not: `PIL.features.check("raqm")` is False, and so are
    harfbuzz and fribidi. So the two steps are done here instead --
    `arabic_reshaper` picks the initial, medial and final forms, and
    `python-bidi` reorders the run for display.

    Left-to-right text is returned untouched, so nothing about an English
    caption changes. If either package is missing the text is returned as it
    came, which is the old behaviour rather than a crash.
    """
    if not text or not _has_rtl(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except Exception:
        return text
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def _wrap(draw: Any, text: str, font: Any, width: int) -> list[str]:
    """Break a caption to fit the panel, by measurement rather than by count.

    Character counts guess wrong on a proportional face, and a caption that
    overruns the panel is worse than one that wraps early.

    Right-to-left text is wrapped in logical order and shaped per line. Doing
    it the other way round -- shaping the whole caption and then breaking it --
    puts the words on the wrong lines, because the break has to happen where
    the sentence breaks, not where the display happens to run out.
    """
    def measure(s: str) -> float:
        return draw.textlength(_shape_for_display(s), font=font)

    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if measure(trial) <= width:
            line = trial
            continue
        if line:
            lines.append(line)
            line = ""
        # A word with nowhere to break.
        #
        # `or not line` used to accept any word that did not fit as long as the
        # line was empty, on the reasoning that something has to go somewhere.
        # For an ordinary long word that is right; for a word with no spaces in
        # it at all -- CJK text, a URL, a hashtag, a pasted identifier -- it
        # painted one unbroken run straight through the neighbouring panel and
        # off the page, cut mid-character at the margin. Nothing in the caption
        # path bounds what a person can paste, so the break is made here by
        # measurement, the same way every other break is.
        while measure(word) > width and len(word) > 1:
            cut = len(word)
            while cut > 1 and measure(word[:cut]) > width:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        line = word
    if line:
        lines.append(line)
    # Shaped last, per line, so each is drawn as it is read.
    return [_shape_for_display(l) for l in lines]


# What a page holds. Four to six panels is the working range in print comics;
# fewer reads as padding, more forces every panel to be small and simple.
#
# The two bounds are not enforced in the same place, and reading them as one
# rule is how the page-slot arithmetic came to be wrong. PAGE_MAX is a hard
# ceiling: `_split_run` divides any scene longer than it, and the singleton
# merge refuses to overfill past it. PAGE_MIN is a floor only *within* one
# scene -- it is what `_split_run` divides towards -- and a scene boundary
# deliberately overrides it, because a scene starting halfway down a page reads
# as a jump rather than a change of place. So a two-panel scene is a two-panel
# page on purpose, and the number of pages is bounded by the panel count, not
# by `panels / PAGE_MIN`.
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

    Print pages are read as horizontal bands. A wide shot takes its whole
    tier -- that is what makes it establishing, and a large panel is what
    slows a reader down. Tighter shots share, which speeds the page up. The
    alternation is the pacing.

    Two to a tier, three only on a crowded page, and four tiers at most: the
    page is portrait and its tiers fill it, so three panels across one band
    made cells four times taller than wide. layout.ts arranges a page the
    same way, and a shared fixture holds the two together.
    """
    per = 2 if len(shots) <= 8 else 3
    tiers: list[list[int]] = []
    i = 0
    while i < len(shots):
        if shots[i] == "wide":
            tiers.append([i])
            i += 1
            continue
        group = [i]
        i += 1
        while i < len(shots) and shots[i] != "wide" and len(group) < per:
            remaining = sum(1 for j in range(i, len(shots)) if shots[j] != "wide")
            if per == 3 and len(group) == 2 and remaining == 2:
                break   # leave two for the next tier rather than 3 + 1
            group.append(i)
            i += 1
        tiers.append(group)
    # Past four bands a page is a ladder of slivers: join the two adjacent
    # tiers holding the fewest panels until it is not.
    while len(tiers) > 4:
        k = min(range(len(tiers) - 1), key=lambda t: len(tiers[t]) + len(tiers[t + 1]))
        tiers[k:k + 2] = [tiers[k] + tiers[k + 1]]
    return tiers


PAGE_GROUND = (243, 241, 236)


def _fit(im: Any, w: int, h: int, ground: tuple[int, int, int] = PAGE_GROUND) -> Any:
    """Scale and centre-crop to fill a cell, but never enlarge the drawing.

    `scale = max(...)` on its own always filled the cell, which meant the one
    panel the layout deliberately sets largest was the only one being stretched:
    a full-width establishing cell is 1144px on a 1240px page, against a panel
    drawn at 768, so every establishing shot was resampled 1.49x and came back
    the softest thing on the page. That is the inverse of what the tier is for.

    Capped at 1.0 the drawing keeps its own pixels. Where a cell is larger than
    the panel the remainder is the page's own paper rather than invented detail
    -- and with the page width now derived from the panels, the common path
    lands at exactly 1.0 and no padding is visible at all.
    """
    from PIL import Image

    scale = min(1.0, max(w / im.width, h / im.height))
    if scale < 1.0:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                       Image.LANCZOS)
    if im.width == w and im.height == h:
        return im
    # Trim whatever overflows, then centre what is left on the paper.
    if im.width > w or im.height > h:
        cx, cy = max(0, (im.width - w) // 2), max(0, (im.height - h) // 2)
        im = im.crop((cx, cy, cx + min(w, im.width), cy + min(h, im.height)))
    cell = Image.new("RGB", (w, h), ground)
    cell.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return cell


def _caption_width(cell_w: int) -> int:
    """How wide a narration box may be in a cell this wide.

    A function rather than the expression repeated at each use, because the
    pass that measures a text block and the pass that draws it have to agree
    about it to the pixel.
    """
    return min(cell_w - 20, 430)


def _bubble_width(cell_w: int) -> int:
    """How wide a speech balloon may be in a cell this wide.

    Narrower than the cell by design: a balloon sits on its speaker's side of
    the panel, which it cannot do while spanning the whole width."""
    return max(1, min(cell_w - 24, max(160, int(cell_w * 0.62)), 380))


def _caption_lines(draw: Any, text: str, font: Any, w: int, line_h: int,
                   max_h: int = 0) -> list[str]:
    """The caption as it will be drawn, clipped to `max_h` when one is given.

    Shared by the measuring pass and the drawing pass so the two cannot
    disagree: the cell has to know how tall the caption will be before it can
    decide how many balloons fit beneath it. One line is always kept -- a
    narration box with nothing in it is a grey rectangle over the picture.
    """
    lines = _wrap(draw, text, font, w - 18)
    if max_h:
        lines = lines[:max(1, (max_h - 12) // line_h)]
    return lines


def _caption_box(draw: Any, text: str, font: Any, box: tuple[int, int, int, int],
                 line_h: int, max_h: int = 0) -> int:
    """A narration box in the corner of a panel, as a comic sets one.

    Returns the height it used, which the caller needs: the caption wraps by
    measurement, so only this function knows how many lines it took. It used to
    return nothing and `_render_page` assumed a flat 78px, so any caption that
    wrapped to three lines -- from about thirteen words, which is an ordinary
    length for narration -- had the first speech balloon painted across it and
    both ended up unreadable.

    `max_h` is the room the cell has. Unbounded, a long caption in a narrow
    three-panel cell filled the cell and ran on into the tier below.
    """
    x, y, w, _ = box
    lines = _caption_lines(draw, text, font, w, line_h, max_h)
    bh = len(lines) * line_h + 12
    draw.rectangle([x, y, x + w, y + bh], fill=(250, 249, 245),
                   outline=(20, 20, 22), width=2)
    for n, line in enumerate(lines):
        draw.text((x + 9, y + 6 + n * line_h), line, font=font, fill=(20, 20, 22))
    return bh


# Where in a panel a speaker stands, from the board's blocking, as a fraction
# of the cell's width: the balloon sits on that side and its tail points there.
_SIDE_AT = {"left": 0.28, "right": 0.72, "centre": 0.5}


def _side(position: str) -> str:
    """"left third", "far right", "centre" or "off" (heard, not seen)."""
    position = str(position or "").lower()
    return "left" if "left" in position else "right" if "right" in position \
        else "centre" if "centre" in position or "center" in position else "off"


def _bubble(draw: Any, text: str, font: Any, cell: tuple[int, int, int, int],
            top: int, max_w: int, line_h: int, side: str = "centre") -> int:
    """A speech balloon on its speaker's side of the cell, its tail pointing
    at them, returning the height it used.

    Balloons used to be stacked down the middle of every panel with their
    tails straight down, whoever was talking, each with a black name label
    under it -- a caption's job done badly, over the picture. The reader
    knows who speaks from where the tail points. A line heard from off panel
    has no tail at all."""
    x, _y, w_cell, _h = cell
    lines = _wrap(draw, text, font, max_w - 26)
    tw = max((draw.textlength(l, font=font) for l in lines), default=0)
    # Never wider than the space it was given. The balloon sized itself from
    # its longest line, which before the hard break above could exceed the
    # panel -- and a balloon wider than its cell is drawn over the panel beside
    # it. Clamped here as well, because the balloon is what covers the picture.
    w = min(int(tw) + 26, max_w)
    h = len(lines) * line_h + 16
    # Centred over the speaker where the cell allows, so the tail is short;
    # a line from off panel sits in the middle.
    target = x + round(w_cell * _SIDE_AT.get(side, 0.5))
    x0 = max(x + 12, min(x + w_cell - 12 - w, target - w // 2))
    y0 = top
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=min(16, h // 2),
                           fill=(252, 252, 250), outline=(20, 20, 22), width=2)
    if side in _SIDE_AT:
        base = max(x0 + 16, min(x0 + w - 16, target))
        draw.polygon([(base - 8, y0 + h - 1), (base + 8, y0 + h - 1), (target, y0 + h + 14)],
                     fill=(252, 252, 250), outline=(20, 20, 22))
    for n, line in enumerate(lines):
        lw = draw.textlength(line, font=font)
        draw.text((x0 + (w - lw) / 2, y0 + 8 + n * line_h), line, font=font,
                  fill=(20, 20, 22))
    return h + 18


def _bubble_height(draw: Any, text: str, font: Any,
                   max_w: int, line_h: int) -> int:
    """What `_bubble` will take, without drawing it.

    The same arithmetic as above, deliberately: a balloon has to be measured
    before the cell can decide whether there is room left for it, and the two
    numbers being the same is the whole point of the check.
    """
    lines = _wrap(draw, text, font, max_w - 26)
    return len(lines) * line_h + 16 + 18


# How much of a cell's height the caption and the balloons may take between
# them. A comic letters over the picture, but past about half the cell there is
# no picture left to letter over. In a three-panel tier the cell is only ~369px
# wide, where the caption box (up to 430 wide, wrapping into as many lines as it
# needs) plus two balloons came to more than the cell was tall: both spilled
# into the tier below, and the second line of dialogue was drawn off the bottom
# of the page and lost.
_TEXT_COVER = 0.35
# The most a crowded cell may give its words before a line is left off and
# reported. Past this the panel is a block of text with a picture behind it.
_TEXT_LIMIT = 0.6
# How much of the block the narration box may take when there is dialogue to
# go under it. Without a share, a long caption filled the block on its own and
# both balloons were dropped -- which is the same loss the other way round.
_CAPTION_SHARE = 0.55
# Sizes to try, as a fraction of the board's own, before giving up on a
# balloon. Below `_MIN_TEXT_SIZE` nothing is readable at print size, so
# shrinking further would trade one unreadable page for another.
_TEXT_STEPS = (1.0, 0.85, 0.72, 0.6)
_MIN_TEXT_SIZE = 11


def _text_size(style: dict[str, Any]) -> int:
    """The board's own lettering size.

    Derived from `line_h` when a caller built the style without it, so the
    smaller faces below are always relative to the size the board asked for.
    """
    size = int(style.get("size") or 0)
    return size if size > 0 else max(8, int(style["line_h"] / 1.4))


def _cell_text_plan(probe: Any, cap: str, dialogue: list[Any],
                    style: dict[str, Any], cell_w: int,
                    cell_h: int) -> tuple[int, int, int, bool]:
    """The face size, caption height and balloon count that fit in one cell.

    Nothing may be drawn outside its own cell, and the only bound the lettering
    had was on its width: `_caption_box` wrapped to as many lines as the words
    needed and each balloon was placed under the last one, with nothing
    comparing the total against the cell. In a three-panel tier that covered the
    picture completely, ran into the tier below, and put the second line of
    dialogue off the page.

    So the block is measured before anything is drawn, and it is measured
    twice: first against the share of the cell a comic letters over, and then,
    only if everything will not go there, against the cell itself. The second
    pass is what keeps the lettering whole -- it is better to set a crowded
    panel a couple of sizes down than to lose the words, and either is better
    than drawing them over the next tier.

    Text is dropped only when neither pass has room for it at the smallest
    readable face, and the caption is clipped before a balloon is: a reader can
    follow a story whose narration was cut short far more easily than one whose
    characters stop answering each other.

    When it comes to that, it is reported. Returns the face size, the caption
    height, how many balloons fit, and whether the caption was set whole -- and
    the caller compares that balloon count against the dialogue it passed in.
    Losing a line is sometimes the only thing a cell this size can do; doing it
    without saying so is what turned a composed page into a finished-looking
    comic that had quietly stopped quoting one of its speakers.
    """
    base = _text_size(style)
    cap_w, bub_w = _caption_width(cell_w), _bubble_width(cell_w)
    wanted = list(dialogue)
    best: tuple[tuple[int, bool], int, int, int] | None = None
    for cover in (max(0, int(cell_h * _TEXT_COVER)), max(0, int(cell_h * _TEXT_LIMIT))):
        cap_room = cover if not wanted else int(cover * _CAPTION_SHARE)
        for step in _TEXT_STEPS:
            size = max(_MIN_TEXT_SIZE, int(round(base * step)))
            line_h = max(1, int(size * 1.4))
            default = style["font"] if size == base else _caption_font(size)
            cap_h, whole = 0, True
            if cap:
                font = _lettering(cap, size, default)
                asked = len(_caption_lines(probe, cap, font, cap_w, line_h))
                drawn = len(_caption_lines(probe, cap, font, cap_w, line_h,
                                           cap_room))
                cap_h, whole = drawn * line_h + 12, drawn >= asked
            room = cover - (cap_h + 8 if cap else 0)
            used = count = 0
            for line in wanted:
                text = str(line.get("text", ""))
                height = _bubble_height(probe, text,
                                        _lettering(text, size, default),
                                        bub_w, line_h)
                if used + height > room:
                    break
                used += height
                count += 1
            if count == len(wanted) and whole:
                return size, cap_h, count, True
            # More of the story drawn wins, then a caption that is not cut
            # short, then the larger face -- which is why the largest is tried
            # first and a tie never replaces what is already held.
            score = (count, whole)
            if best is None or score > best[0]:
                best = (score, size, cap_h, count)
    score, size, cap_h, count = best or ((0, True), base, 0, 0)
    return size, cap_h, count, bool(score[1])


# A page's inner area, width over height: a comic page is about 2:3. Every
# page of a board is this shape, so a book of them prints and reads as one.
INNER_ASPECT = 2 / 3
# The gutter between cells, as a fraction of the inner width.
GUTTER = 0.02
# A tier holding one large panel -- an establishing shot or a Peak -- is
# taller than the others, which is what slows the reader down on it.
WIDE_TIER = 1.3


def _page_cells(shots: list[str], rtl: bool = False) -> list[tuple[float, float, float, float]]:
    """Each panel's cell on one page, as (x, y, w, h) fractions of the inner
    area, filling the page.

    The board computes the same layout before anything is drawn (layout.ts)
    and draws each panel at its cell's shape; this is the fallback for a
    caller that sends no cells, and the two are checked against one fixture.
    Cells used to be squares, with a wide tier set at 0.58 of its width: a
    square panel in that cell lost 42% of its height to the crop.

    Manga is read right to left, so a tier's cells run from the right."""
    tiers = _tiers(shots)
    gy = GUTTER * INNER_ASPECT
    weights = [WIDE_TIER if len(t) == 1 and shots[t[0]] == "wide" else 1.0 for t in tiers]
    room = 1 - gy * (len(tiers) - 1)
    cells: list[Any] = [None] * len(shots)
    y = 0.0
    for tier, weight in zip(tiers, weights):
        h = room * weight / sum(weights)
        w = (1 - GUTTER * (len(tier) - 1)) / len(tier)
        for k, i in enumerate(tier):
            x = k * (w + GUTTER)
            cells[i] = (round(1 - x - w if rtl else x, 6), round(y, 6), round(w, 6), round(h, 6))
        y += h + gy
    return cells


def _cell_px(cell: Any, style: dict[str, Any]) -> tuple[int, int, int, int]:
    """A cell in pixels on a page of this style."""
    margin = style["margin"]
    inner_w = style["width"] - margin * 2
    inner_h = round(inner_w / INNER_ASPECT)
    x, y, w, h = (float(v) for v in cell)
    left, top = margin + round(x * inner_w), margin + round(y * inner_h)
    return left, top, max(1, margin + round((x + w) * inner_w) - left), max(1, margin + round((y + h) * inner_h) - top)


def _letter_cell(draw: Any, probe: Any, cap: str, dialogue: list[Any], cell: tuple[int, int, int, int],
                 style: dict[str, Any], rtl: bool) -> dict[str, Any] | None:
    """Letter one panel: the caption in the corner a reader starts from, the
    balloons below it in reading order, each on its speaker's side. Returns a
    record of what would not fit, or None when everything was lettered."""
    x, y, cell_w, cell_h = cell
    font, base = style["font"], _text_size(style)
    size, cap_h, bubbles, whole = _cell_text_plan(probe, cap, dialogue, style, cell_w, cell_h)
    line_h = max(1, int(size * 1.4))
    body = font if size == base else _caption_font(size)
    if cap:
        width = _caption_width(cell_w)
        _caption_box(draw, cap, _lettering(cap, size, body),
                     (x + cell_w - 10 - width if rtl else x + 10, y + 10, width, 0),
                     line_h, max_h=cap_h)
    # Below the caption that was actually drawn, rather than below a guess at
    # how tall it might be.
    by = y + (10 + cap_h + 8 if cap else 16)
    for line in dialogue[:bubbles]:
        text = str(line.get("text", ""))
        by += _bubble(draw, text, _lettering(text, size, body), cell, by,
                      _bubble_width(cell_w), line_h, _side(line.get("side", "centre" if not line.get("offscreen") else "off")))
    if bubbles < len(dialogue) or not whole:
        return {"spoken": len(dialogue), "lettered": bubbles, "caption_clipped": bool(cap) and not whole}
    return None


def _render_page(panels: list[Any], idx: list[int], meta: dict[str, list[Any]],
                 style: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """Draw one page of panels in their cells, and report what would not fit.

    The cells are the board's own layout, the one each panel was drawn for,
    so a panel fills its cell without a crop. The second return is one record
    per panel that lost text -- which panel, how many spoken lines it was
    given, how many were lettered, and whether the caption was cut short.
    Empty is the ordinary case and the only one that should reach an
    exported chapter.
    """
    from PIL import Image, ImageDraw

    page_w, margin = style["width"], style["margin"]
    rtl = style.get("direction") == "rtl"
    total_h = round((page_w - margin * 2) / INNER_ASPECT) + margin * 2
    fallback = _page_cells([meta["shots"][i] for i in idx], rtl)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    page = Image.new("RGB", (page_w, total_h), PAGE_GROUND)
    draw = ImageDraw.Draw(page)
    short: list[dict[str, Any]] = []
    for local, src in enumerate(idx):
        given = (meta.get("cells") or [None] * (src + 1))[src]
        cell = _cell_px(given if given else fallback[local], style)
        x, y, cell_w, cell_h = cell
        page.paste(_fit(panels[src], cell_w, cell_h), (x, y))
        draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(20, 20, 22), width=3)
        lost = _letter_cell(draw, probe, meta["captions"][src].strip(), list(meta["dialogue"][src]),
                            cell, style, rtl)
        if lost:
            short.append({"panel": src, **lost})
    return page, short


def _render_strip(panels: list[Any], meta: dict[str, list[Any]],
                  style: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """Draw the panels as one vertical scroll.

    Not a page stretched tall, which is the mistake this form is known for.
    A scroll reveals one moment at a time and the gap between panels is the
    tempo: a short gap reads as a beat, a long one as a held breath. So the
    gutter widens wherever the scene changes, and stays tight within one.

    Each panel keeps its own shape -- it was drawn for the scroll -- rather
    than being cut to one fixed band. Reports what would not fit, in the
    same shape a page does.
    """
    from PIL import Image, ImageDraw

    width, margin = style["width"], style["margin"]
    font, line_h = style["font"], style["line_h"]
    base = _text_size(style)
    beat, breath = 46, 190
    rtl = style.get("direction") == "rtl"

    sized, gaps = [], []
    inner = width - margin * 2
    for i, im in enumerate(panels):
        sized.append(_fit(im, inner, max(1, round(inner * im.height / im.width))))
        if i:
            same = meta["scenes"][i] == meta["scenes"][i - 1]
            gaps.append(beat if same else breath)

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    # The face per caption, kept so the pass that measures a caption and the
    # pass that draws it use the same one.
    cap_fonts = [_lettering(meta["captions"][i].strip(), base, font)
                 for i in range(len(panels))]
    caps = [_wrap(probe, meta["captions"][i].strip(), cap_fonts[i], inner - 40)
            if meta["captions"][i].strip() else [] for i in range(len(panels))]

    total = margin + sum(im.height for im in sized) + sum(gaps)
    total += sum(len(c) * line_h + 18 for c in caps if c) + margin

    page = Image.new("RGB", (width, total), PAGE_GROUND)
    draw = ImageDraw.Draw(page)
    short: list[dict[str, Any]] = []
    y = margin
    for i, im in enumerate(sized):
        page.paste(im, (margin, y))
        draw.rectangle([margin, y, margin + im.width, y + im.height],
                       outline=(20, 20, 22), width=3)
        lost = _letter_cell(draw, probe, "", list(meta["dialogue"][i]),
                            (margin, y, im.width, im.height), style, rtl)
        if lost:
            short.append({"panel": i, **lost})
        y += im.height
        if caps[i]:
            for n, line in enumerate(caps[i]):
                draw.text((margin + 2, y + 10 + n * line_h), line,
                          font=cap_fonts[i], fill=(30, 30, 33))
            y += len(caps[i]) * line_h + 18
        if i < len(gaps):
            y += gaps[i]
    return page, short


def _as_cell(value: Any) -> tuple[float, float, float, float] | None:
    """A cell as sent by the board, or None when it is missing or unusable."""
    try:
        x, y, w, h = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
        return None
    return (x, y, min(w, 1 - x), min(h, 1 - y))


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
        "scenes": [_as_scene(x) for x in column("scenes", 1)],
        # The board's own layout: each panel's page and cell, the shape it
        # was drawn at. Without them the pages are laid out here.
        "pages": [int(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None
                  for x in column("pages", None)],
        "cells": [_as_cell(c) for c in column("cells", None)],
    }
    layout = str(req.get("layout") or "page")
    # Clamped rather than trusted. These have no control in the app today, so
    # every value seen so far is the default -- which is exactly when a bad one
    # slips through unnoticed, and a zero or negative here divides by zero or
    # draws a page with no room on it.
    def _dim(key: str, default: int, low: int, high: int) -> int:
        try:
            return max(low, min(high, int(req.get(key, default))))
        except (TypeError, ValueError):
            return default

    size = _dim("font_size", 19, 8, 72)
    margin = _dim("margin", 34, 0, 400)
    gutter = _dim("gutter", 18, 0, 400)

    outputs: list[str] = []
    sizes: list[int] = []
    dims: list[list[int]] = []
    try:
        panels = []
        for path in staged:
            with Image.open(path) as im:
                panels.append(im.convert("RGB"))

        # Page width derived from the panels, not fixed at 1240.
        #
        # The widest cell on a page is the full-width tier, so sizing the page
        # so that cell matches the panel is what lets `_fit` place an
        # establishing shot at 1:1. Hardcoded at 1240 against panels drawn at
        # 768, every establishing shot was upscaled 1.49x -- a page that was
        # nominally 1240 wide but carried only 768 of real detail. Derived, the
        # number is smaller and every pixel on it is one the model drew.
        # With cells, the page is as wide as it needs to be for each panel to
        # land in its cell at its own size.
        widest = max((im.width / (cell[2] if cell and layout == "page" else 1)
                      for im, cell in zip(panels, meta["cells"])), default=768)
        style = {
            "width": _dim("page_width", min(4096, round(widest)) + margin * 2, 320, 8192),
            "margin": margin,
            "gutter": gutter,
            "font": _caption_font(size),
            "small": _caption_font(13),
            "line_h": int(size * 1.4),
            # The size itself, because a cell that cannot hold its lettering
            # needs to letter it smaller, and `line_h` only implies it.
            "size": size,
            "direction": "rtl" if req.get("direction") == "rtl" else "ltr",
        }

        if layout != "page":
            pages = [list(range(n))]
        elif all(x is not None for x in meta["pages"]):
            # The board's own page breaks, the ones its panels were drawn for.
            order = list(dict.fromkeys(meta["pages"]))
            pages = [[i for i in range(n) if meta["pages"][i] == k] for k in order]
        else:
            pages = _paginate(meta["scenes"])
        if len(pages) > len(slots):
            raise ValueError(
                f"this board makes {len(pages)} pages but only {len(slots)} "
                "were reserved; ask for that many again"
            )

        # Text the page could not hold, gathered across every page so the
        # board can say which panels lost speech rather than handing back a
        # set of pages that look finished.
        short: list[dict[str, Any]] = []

        import io

        import vaultcrypto as vc

        for k, idx in enumerate(pages):
            img, lost = (_render_page(panels, idx, meta, style)
                         if layout == "page"
                         else _render_strip(panels, meta, style))
            for record in lost:
                short.append({**record, "page": k + 1})
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
            "panels": n, "layout": layout, "overflow": short}


def op_bind_comic(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Bind the finished pages into one PDF, sealed like everything else.

    A comic that leaves as a folder of PNGs is not a comic, it is a folder of
    PNGs: the reading order lives in the file names and the first person to
    sort them differently loses it. A PDF carries the order inside the file.

    Written into a vault slot rather than straight to disk, because the vault
    export command is the one sanctioned path for plaintext to leave. So this
    seals a PDF the same way a page is sealed, and the library exports it the
    same way it exports anything else.
    """
    from PIL import Image

    slots = req.get("vault_slots") or []
    if not slots:
        raise ValueError("binding a comic needs a vault slot to write into")

    staged = _stage_vault_inputs(req.get("vault_inputs") or [])
    if not staged:
        raise ValueError("there are no pages to bind")

    try:
        import io

        import vaultcrypto as vc

        pages: list[Any] = []
        for path in staged:
            # PDF has no alpha channel. Flattening onto white rather than
            # letting PIL drop the channel keeps a page with a transparent
            # gutter looking like the page that was composed.
            img = Image.open(path)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                flat = Image.new("RGB", img.size, (255, 255, 255))
                flat.paste(img, mask=img.split()[-1])
                img = flat
            else:
                img = img.convert("RGB")
            pages.append(img)

        buf = io.BytesIO()
        pages[0].save(buf, format="PDF", save_all=True,
                      append_images=pages[1:],
                      title=str(req.get("title") or "Picture board")[:200],
                      # 72 dpi is PDF's own unit, so a page is its pixel size in
                      # points and nothing is resampled on the way in.
                      resolution=72.0)
        data = buf.getvalue()
        slot = slots[0]
        vc.write_sealed(slot["path"], bytes.fromhex(slot["key"]),
                        bytes.fromhex(slot["file_id"]), data)
        log(req_id, f"bound {len(pages)} pages into {slot['id']} "
                    f"({len(data)} bytes)")
        return {"outputs": [slot["id"]], "sealed": True, "sizes": [len(data)],
                "pages": len(pages)}
    finally:
        _discard_staged(staged)


# The version of the shape `_assist_reply` returns.
#
# Not a compatibility negotiation -- the engine and the interface ship
# together. It is a stamp on the reply, so a log, a bug report or a saved
# response says which contract it was written against rather than leaving the
# reader to infer it from which fields happen to be present.
ASSIST_REPLY_VERSION = 1

# Every field of an assist reply, and what it means when nothing set it.
#
# A reply used to be whichever dict literal the branch that returned it
# happened to write, and there were five of them with five different key sets.
# The interface then reconstructed what it had not been told: `outcome` was
# absent on two paths, so it fell through a chain of guesses that ended at the
# card for a successful rewrite -- and the user was shown their own untouched
# prompt, folklore and all, headed "Suggested prompt", above a button that
# replaced it with itself.
#
# Guessing is the wrong repair. The reply says everything, every time, and a
# branch that forgets a field gets the default below rather than an absence for
# something downstream to interpret.
_ASSIST_DEFAULTS: dict[str, Any] = {
    "outcome": "proposal",
    "changed": False,
    "saw_image": False,
    "unclear": False,
    "description": "",
    "note": "",
    "attempt": "",
    "rejected_because": "",
    "added": [],
    "dropped": [],
    "corrected": [],
    "removed": [],
}


def _assist_reply(prompt: str, original: str, **fields: Any) -> dict[str, Any]:
    """One shape for every answer this operation can give.

    Keyword arguments are checked against the known fields rather than merged
    blindly: a typo in a field name is otherwise a field the interface silently
    never sees, which is the same failure as omitting it and harder to find.
    """
    unknown = set(fields) - set(_ASSIST_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown assist reply field(s): {sorted(unknown)}")
    return {
        "version": ASSIST_REPLY_VERSION,
        "prompt": prompt,
        "original": original,
        **_ASSIST_DEFAULTS,
        **fields,
    }


def op_assist(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a prompt, optionally looking at the image being edited."""
    # An explicit null means the reader is not installed. Absent means the
    # caller did not care and the default applies. The distinction matters:
    # without it, a missing reader would fall back to the default repo and
    # fail at load time instead of degrading to a text-only rewrite.
    repo = req.get("assistant", "mlx-community/Qwen2-VL-2B-Instruct-4bit")
    original_prompt = (req.get("prompt") or "").strip()
    if not original_prompt:
        raise ValueError("write something first, then ask for help improving it")
    user_prompt, removed = _clean_prompt_request(original_prompt)
    if not user_prompt:
        return _assist_reply(
            original_prompt, original_prompt,
            unclear=True, outcome="unclear", removed=removed,
            note="Name something visible instead of quality labels.",
        )

    mode = req.get("mode", "generate")
    # Sources normally arrive sealed as `vault_inputs`, which are decrypted to
    # temporary files this function owns and deletes. `images` carries a
    # plaintext path the caller owns, so it is kept apart: everything in
    # `raw_staged` is deleted on the way out, and a caller's own file must
    # never end up in that list.
    direct = list(req.get("images") or [])
    raw_staged = _stage_vault_inputs(req.get("vault_inputs") or [])
    staged = _downscale_for_assist(direct + raw_staged)
    # The downscaler returns the path it was given when the image is already
    # small enough, so `staged` can contain a caller's own file. Only the
    # files this function actually created may be deleted on the way out.
    derived = [p for p in staged if p not in direct and p not in raw_staged]
    try:
        # Both of these read the picture first. An edit needs to know which
        # jacket is meant; a clip starting from a photograph needs to know
        # what is in it before it can say how it moves.
        # The writer is text-only -- Qwen3 has no vision encoder at all -- so
        # reading a picture is the reader's job and nothing else can stand in
        # for it. When the reader is absent the rewrite still happens, just
        # from the words alone, which is far more use than refusing.
        can_read = bool(repo)
        # An edit is an edit whether or not anything can look at the picture.
        #
        # This used to require a staged image and an installed reader, so an
        # edit request with no reader fell through to the `else` below and was
        # handed the scene-direction system prompt -- the one that writes a
        # fresh picture from nothing. That is the opposite of what was asked
        # for, on the one request where the user has already said exactly what
        # they want changed about a picture that exists.
        #
        # The edit path does not need the reader. `_clarify_edit` composes the
        # instruction from the user's own words and, handed no facts, returns
        # them with "keeping everything else in the picture unchanged" and no
        # inference at all -- which is the conservative answer, and the right
        # one when nothing has seen the picture.
        editing = mode == "edit"
        looking = bool(staged) and can_read
        animating = mode == "video" and looking
        blind = bool(staged) and not can_read

        def ask(text: str, images: list[str], max_tokens: int, temperature: float) -> str:
            # Only the picture-reading path needs the vision model, and it is
            # loaded here rather than up front so a plain text request never
            # pays for it.
            #
            # The import belongs here for the same reason the load does. It sat
            # at the top of this function, so every text-only rewrite -- which
            # is most of them, and now every edit made without a reader --
            # imported mlx_vlm and its dependencies before writing a word.
            from mlx_vlm import generate as vlm_generate
            from mlx_vlm.prompt_utils import apply_chat_template

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
            #
            # Asked only when there is a picture and something to read it with.
            # With no facts the composer does not guess: it keeps the words as
            # given and says the rest of the picture is unchanged.
            facts: dict[str, str] = {}
            if looking:
                emit({"id": req_id, "type": "progress", "phase": "denoise",
                      "progress": None, "message": "Looking at your picture"})
                raw = ask(SCENE_SYSTEM, staged, 90, 0.1)
                facts = _parse_scene(raw)
                description = " · ".join(
                    f"{k.lower()}: {v}" for k, v in facts.items())
                log(req_id, f"scene: {description or 'nothing readable'}")
            improved, why, attempt = _clarify_edit(req_id, user_prompt, facts,
                                                   req.get("writer"))
            if why not in ("ok", "composed"):
                # The rewrite was tried and thrown away, which is a fact about
                # the rewrite and not about the request. Said here because the
                # reason used to stop at `_clarify_edit`: this path returned
                # the untouched request with no `outcome` at all, the
                # interface's fallback chain read that as "your words were
                # already fine", and the answer to a failed rewrite was
                # "Nothing to add."
                log(req_id,
                    f"edit rewrite rejected ({why}); keeping the request",
                    "warn")
                return _assist_reply(
                    original_prompt, original_prompt,
                    saw_image=bool(staged) and not blind,
                    # What the cleaner stripped before anything was judged.
                    # Absent from every rejection reply, so a rejected rewrite
                    # handed back the original with its folklore intact and no
                    # sign that a cleaned version had ever existed -- which is
                    # the one fact explaining why a rewrite was attempted.
                    removed=removed,
                    outcome="rewrite_rejected",
                    description=description, rejected_because=why,
                    # What it wrote, so the choice is the user's. The edit
                    # checks are the same literal ones as below, so the text
                    # they refuse is often the text that was wanted.
                    attempt=attempt,
                    note=_WHY_REJECTED.get(
                        why, "Your words were kept: the rewrite drifted from "
                             "them. Try again, or add a detail yourself."),
                )
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
                         # Room to actually write. The default of 160 tokens is
                         # about 120 words, so a fielded prompt was being cut
                         # off mid-sentence and then judged on the fragment.
                         max_tokens=420, temperature=0.6,
                         repo=req.get("writer"), label="Improving the prompt")
            improved, why = _clarified_with_reason(user_prompt, raw)
            if improved is None and why in ("dropped", "lost_intent"):
                # Ask before repairing. The string comparison could not place
                # it, and the better the rephrasing the fewer of the original
                # words it repeats -- so this is precisely where the literal
                # checks reject the rewrites worth having. "my dog but make
                # him look like a king" rendered as a golden retriever on a
                # marble throne repeats neither "dog" nor "king".
                attempt = _first_line(raw)
                if attempt and _same_request(req_id, user_prompt, attempt,
                                             req.get("writer")):
                    log(req_id, f"{why} by the letter, same request by the "
                                f"writer's own reading; keeping it")
                    improved, why = attempt, "rephrased"
            if improved is None and why == "dropped":
                # Only now, and it shows: repairing appends the missing word to
                # the end of the sentence, so the prompt finishes "...atop a
                # marble throne, his fur catching the light, king." It is worth
                # having when the alternative is losing the request, and worth
                # avoiding whenever the reading above can settle it instead.
                repaired = _repair_dropped(user_prompt, _first_line(raw))
                if repaired:
                    log(req_id, f"rewrite dropped a word; repaired to {repaired!r}")
                    improved, why = repaired, "repaired"
            if improved is None and why not in ("unclear", "empty"):
                # The rewrite was rejected, not the request. These checks are
                # deliberately literal -- they are what stops "an old bicycle
                # against a brick wall" coming back as a bicycle -- so a good
                # paraphrase gets caught too: "a rainy street" rendered as
                # "wet" and "slick" loses the word "rainy" while keeping the
                # picture. Sampling is the difference, so it is worth one more
                # draw at a lower temperature before giving up.
                log(req_id, f"rewrite rejected ({why}); trying once more",
                    "warn")
                raw = _write(req_id, system,
                             (f"The picture shows: {seen}\n" if seen else "")
                             + f"Request: {user_prompt}",
                             max_tokens=420, temperature=0.25,
                             repo=req.get("writer"),
                             label="Improving the prompt")
                improved, why = _clarified_with_reason(user_prompt, raw)
            if improved is None and why in ("dropped", "lost_intent"):
                # Again on the second draw. The first pass adjudicated and the
                # retry did not, so a rewrite that was right both times was
                # refused on the second -- the one that actually reaches the
                # user.
                attempt = _first_line(raw)
                if attempt and _same_request(req_id, user_prompt, attempt,
                                             req.get("writer")):
                    log(req_id, "second draw reads as the same request; keeping it")
                    improved, why = attempt, "rephrased"
            if improved is None:
                if why == "unclear":
                    # A fact about the request: it names nothing that could be
                    # drawn. Inventing a subject to fill the gap is what made
                    # this useless before.
                    return _assist_reply(
                        original_prompt, original_prompt,
                        saw_image=bool(staged) and not blind,
                        unclear=True,
                        outcome="unclear", description=description,
                        removed=removed,
                        note=("This does not say what to draw yet. Name "
                              "the thing you want and I can make it "
                              "specific."),
                    )
                # Everything else is a fact about the *rewrite*, not the
                # request: it drifted, dropped something, or ran long. The
                # request was fine, so it is kept as written rather than the
                # user being told their prompt is the problem.
                log(req_id, f"rewrite rejected ({why}); keeping the request",
                    "warn")
                return _assist_reply(
                    original_prompt, original_prompt,
                    saw_image=bool(staged) and not blind,
                    # What the cleaner stripped before anything was judged. See
                    # the edit rejection above: without it the user is handed
                    # back their folklore with no sign a cleaned version existed.
                    removed=removed,
                    outcome="rewrite_rejected",
                    description=description, rejected_because=why,
                    # What it wrote, so a rejection is a choice rather than a
                    # loss. This path had no `attempt` at all, so the whole
                    # rewrite was discarded in silence -- and the checks here
                    # are literal by design, which means the text they refuse
                    # is often the one the user wanted. Asked for a cat
                    # "setting" on a chair the writer says "sitting", which is
                    # certainly what was meant and cannot be proven so by any
                    # string comparison. Show it and let them decide.
                    attempt=_first_line(raw),
                    note=_WHY_REJECTED.get(
                        why, "Your words were kept: the rewrite drifted from "
                             "them. Try again, or add a detail yourself."),
                )

        if not _keeps_intent(user_prompt, improved):
            log(req_id, f"discarding rewrite {improved!r}: it lost the request", "warn")
            return _assist_reply(
                original_prompt, original_prompt,
                saw_image=bool(staged) and not blind,
                # See the rejection above: without this the user is handed back
                # their folklore with no sign a cleaned version existed.
                removed=removed,
                outcome="rewrite_rejected", rejected_because="lost_intent",
                attempt=improved,
                note=_WHY_REJECTED["lost_intent"],
                description=description)
    finally:
        # Never `staged`: it may hold a path the caller owns.
        _discard_staged(derived)
        _discard_staged(raw_staged)

    changed = improved.strip() != original_prompt.strip()
    return _assist_reply(
        improved, original_prompt,
        saw_image=bool(staged) and not blind, description=description,
        removed=removed, changed=changed,
        # Every reply says which of these happened, so the interface never has
        # to guess from whether the text came back the same. "Yours was already
        # fine" and "I tried and the result was worse" look identical
        # otherwise, and saying the first when the second is true is why this
        # feature reads as doing nothing.
        outcome="proposal" if changed else "already_specific",
        added=_added_words(original_prompt, improved),
        # Spellings it fixed, named so they are visible rather than slipped in.
        # The retention check used to count a corrected word as a lost one and
        # throw the whole rewrite away, so an enhancer that could not fix a
        # typo was the entire behaviour anyone saw.
        corrected=_corrections(original_prompt, improved),
        # Words the rewrite no longer says. Reported rather than refused: a
        # request cannot be put into good English without losing some of the
        # words it was typed in, and the person approving the rewrite can see
        # for themselves whether the one it dropped mattered.
        #
        # Diffed against the prompt that was judged, not the one that was
        # typed. Against the typed one, every word the cleaner had already
        # stripped came back a second time as a word the rewrite had dropped --
        # "a masterpiece 8k cat" reported `removed: [masterpiece, 8k]` and
        # `dropped: [masterpiece]`, and the interface drew both as the same red
        # chip. The user was told one word had been deliberately removed and
        # accidentally lost.
        dropped=_dropped_words(user_prompt, improved),
        note=("Rewritten from your words only. Install the prompt assistant "
              "(1.2 GiB) if you want it to look at the picture as well."
              if blind else ""))


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


def _safetensors_header(path: str) -> dict[str, Any]:
    """The tensor index at the front of a safetensors file.

    Only the header is read -- eight bytes of length then that much JSON -- so
    this costs nothing on a multi-gigabyte file.
    """
    with open(path, "rb") as fh:
        size = struct.unpack("<Q", fh.read(8))[0]
        if not 0 < size <= 100 * 1024 * 1024:
            raise ValueError("safetensors header is not a sensible size")
        return json.loads(fh.read(size))


def _describe_weights(keys: list[str]) -> str:
    """What a file full of these tensors actually is."""
    if any(k.startswith(("decoder.", "encoder.")) for k in keys) and not any(
            k.startswith(("transformer.", "model.", "unet.")) for k in keys):
        return "a VAE -- an image decoder, not an adapter"
    if any(k.startswith(("text_model.", "text_encoder.")) for k in keys):
        return "a text encoder, not an adapter"
    if len(keys) > 400:
        return "a full model, not an adapter"
    return "no adapter weights"


def _adapter_local_file(handle: str) -> str | None:
    """The downloaded file a handle names, if it is on this machine.

    Two places, because adapters land in two. The Hugging Face cache is where
    this app puts them; mflux resolves a handle itself and keeps its own copy
    under `~/Library/Caches/mflux/loras`. Only the first was searched, so for a
    file mflux had fetched the answer was "cannot resolve it", the check
    returned without an opinion, and the adapter went to the loader unexamined
    -- which is how a VAE still reached mflux and came back as "did not match
    any known adapter keys" with the app's own clearer refusal never spoken.
    """
    repo, _, wanted = handle.partition(":")
    root = Path(os.environ.get("HF_HOME",
                               str(Path.home() / ".cache" / "huggingface"))) / "hub"
    base = root / ("models--" + repo.replace("/", "--")) / "snapshots"
    if base.is_dir():
        for snapshot in sorted(base.iterdir(), reverse=True):
            if wanted:
                candidate = snapshot / wanted
                if candidate.exists():
                    return str(candidate)
            else:
                files = sorted(snapshot.glob("*.safetensors"),
                               key=lambda f: -f.stat().st_size)
                if files:
                    return str(files[0])

    # mflux's own adapter cache, flat and named by file.
    mflux_loras = Path.home() / "Library" / "Caches" / "mflux" / "loras"
    if mflux_loras.is_dir():
        if wanted:
            candidate = mflux_loras / Path(wanted).name
            if candidate.exists():
                return str(candidate)
        else:
            stem = repo.replace("/", "_")
            for f in sorted(mflux_loras.glob("*.safetensors")):
                if stem.lower() in f.stem.lower():
                    return str(f)

    # A plain path, which is what the mflux route is handed once resolved.
    if os.path.isfile(handle):
        return handle
    return None


def _check_is_adapter(req_id: str, handle: str) -> None:
    """Refuse a "LoRA" that holds no adapter weights.

    Nothing checked this. `lora_info` listed every .safetensors file in a
    repository with its size and offered them all as adapters, so a VAE
    repository looked exactly like an adapter repository: it downloaded, it
    appeared in the list as installed, it could be selected and given a
    strength, and then it changed nothing, because there is no adapter in it to
    apply. The log said "adapters: ... @ 1.0" the whole time.

    Silence is the wrong answer here. An adapter that does nothing is
    indistinguishable from an adapter that is not working, which is a long way
    to go before finding out the file was never one.
    """
    path = _adapter_local_file(handle)
    if path is None:
        # Not resolvable locally; let the loader speak for itself rather than
        # refusing something that might be fine.
        return
    try:
        header = _safetensors_header(path)
    except Exception as exc:
        log(req_id, f"could not read {handle} to check it: {exc}", "warn")
        return
    keys = [k for k in header if k != "__metadata__"]
    if any("lora" in k.lower() for k in keys):
        return
    raise ValueError(
        f"{os.path.basename(path)} is not a LoRA. It holds "
        f"{len(keys)} tensors and none of them are adapter weights \u2014 it is "
        f"{_describe_weights(keys)}. Remove it from your adapters; leaving it "
        f"selected cannot change the picture."
    )


def op_verify_adapter(req_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Confirm a downloaded file really is an adapter, before it is recorded.

    The check existed and ran at load time, which is the wrong moment: by then
    the file is installed, listed, selectable and given a strength, and the
    person finds out only when they try to draw with it. The file is on disk
    the instant the download finishes, so this is the first moment the question
    can be answered -- and answering it here means a VAE never reaches the
    adapter list at all.
    """
    _check_is_adapter(req_id, str(req.get("handle") or ""))
    return {"ok": True}


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
    ranked = sorted(files, key=lambda f: -f["bytes"])
    out = {
        "repo": repo,
        "files": ranked,
        "bytes": sum(f["bytes"] for f in files),
        "gated": bool(getattr(info, "gated", False)),
        "base_model": (getattr(info, "cardData", None) or {}).get("base_model"),
    }
    # Whether there is an adapter in here at all, decided before the download
    # is offered rather than after it has finished and changed nothing. Only
    # the header is fetched, which is a few kilobytes of a file that may be
    # gigabytes.
    if ranked:
        verdict = _remote_adapter_verdict(repo, ranked[0]["name"])
        if verdict:
            out["not_an_adapter"] = verdict
    return out


def _remote_adapter_verdict(repo: str, filename: str) -> str | None:
    """Why this file is not an adapter, read from its header over HTTP.

    Returns None when it is an adapter, or when the question could not be
    answered -- a probe that fails must not block a download that would work.
    """
    try:
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session

        url = hf_hub_url(repo_id=repo, filename=filename)
        # The app's own token, the same one every other fetch here uses.
        # build_hf_headers() alone reads the environment and the huggingface
        # CLI's login, neither of which this app populates -- so the probe got
        # a 401 and, failing closed, reported nothing and let the download of a
        # VAE proceed exactly as before.
        headers = dict(build_hf_headers(token=_hf_token()))
        headers["Range"] = "bytes=0-1048575"
        response = get_session().get(url, headers=headers, timeout=20)
        response.raise_for_status()
        blob = response.content
        if len(blob) < 8:
            return None
        size = struct.unpack("<Q", blob[:8])[0]
        if not 0 < size <= len(blob) - 8:
            # The index did not fit in the range asked for, which only happens
            # on very large files. Not worth a second request to refuse
            # something that is probably fine.
            return None
        header = json.loads(blob[8:8 + size])
        keys = [k for k in header if k != "__metadata__"]
        if not keys or any("lora" in k.lower() for k in keys):
            return None
        return _describe_weights(keys)
    except Exception:
        return None


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


def _retry_without_xet(fetch: Any, req_id: str) -> Any:
    """Run a Hugging Face download, and run it again over plain HTTP if the
    chunked transport fails.

    Hugging Face serves large files through Xet, a content-addressed store that
    hands back a reconstruction plan of chunks. A truncated or expired response
    surfaces partway through a multi-gigabyte download as a CAS error or a bare
    403, and retrying the same way tends to fail the same way.

    The subprocess routes already do this by setting HF_HUB_DISABLE_XET in the
    child's environment. The in-process routes could not: the variable is read
    into a module constant at import, long before any of this runs. But that
    constant is consulted on every call rather than cached, so setting it
    directly works -- and without it the prompt writer and the picture reader,
    which both come down this path, had no recovery at all. Measured on this
    machine: a 9 GB fetch died at 6.9 GB with a 403 from the CDN, and the same
    fetch completed with Xet disabled.
    """
    try:
        return fetch()
    except Exception as exc:
        if not _is_transfer_failure(exc):
            raise
        log(req_id, "chunked transfer failed; retrying over plain HTTP", "warn")
        return _fetch_with_xet_disabled(fetch)


def _is_transfer_failure(exc: BaseException) -> bool:
    """Whether this failure is the chunked transport rather than the request.

    Separate and pure so the decision can be tested where huggingface_hub is
    not installed -- which is CI, and which is how four tests for this went
    green locally and red on push.
    """
    text = str(exc).lower()
    return any(m in text for m in (
        "cas client error", "file reconstruction error", "xet_get",
        "error decoding response body",
    ))


def _fetch_with_xet_disabled(fetch: Any) -> Any:
    """Run `fetch` with the chunked transport turned off, then put it back."""
    from huggingface_hub import constants

    was = constants.HF_HUB_DISABLE_XET
    constants.HF_HUB_DISABLE_XET = True
    try:
        return fetch()
    finally:
        constants.HF_HUB_DISABLE_XET = was


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
        """Run the downloader as a child this worker can actually stop.

        This was a plain `subprocess.run`, which blocks until the transfer
        finishes whatever else happens. The worker acknowledged `{"op":
        "cancel"}` into `_CANCELLED` and then went on downloading -- on a large
        model, for another twenty minutes -- and `op_download` returned success
        at the end of it, so the Models list announced the model was ready
        after the user had pressed Cancel and watched nothing happen.

        Retrying `communicate` after a timeout is documented as safe and loses
        no output, so polling costs nothing but the half-second granularity.
        """
        proc = subprocess.Popen(
            cmd, env=environ,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        while True:
            try:
                out, err = proc.communicate(timeout=0.5)
                return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
            except subprocess.TimeoutExpired:
                if not is_cancelled(req_id):
                    continue
                log(req_id, "cancelled; stopping the download", "warn")
                proc.terminate()
                try:
                    proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                raise Cancelled() from None

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
            _retry_without_xet(
                lambda: snapshot_download(repo_id=repo_id,
                                          allow_patterns=patterns,
                                          token=_hf_token()),
                req_id)
        elif via == "hf":
            from huggingface_hub import snapshot_download

            _retry_without_xet(
                lambda: snapshot_download(repo_id=repo_id, token=_hf_token()),
                req_id)
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

    # The in-process routes below (`snapshot_download` for mflux and hf) cannot
    # be interrupted partway, so a cancel during one of those is only honoured
    # here. Either way the answer must not be success: the partial cache stays
    # on disk because Hugging Face resumes from it, but nothing writes
    # `.melp-complete` and nothing tells the interface the model is ready.
    if is_cancelled(req_id):
        raise Cancelled()

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
    import audiobook
    if repo_id == audiobook.KOKORO_REPO:
        _install_kokoro_packages(req_id)
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


def _emit_preview_image(req_id: str, img: Any, step: int | None = None,
                        total: int | None = None) -> None:
    """Send a finished picture as a preview frame.

    The step-wise preview decodes latents mid-denoise, which only exists for
    routes that expose a step callback. Work that produces whole pictures in
    stages -- a tiled upscale, a clip's first frame -- has nothing to decode
    but plenty to show, and the receiving end is the same either way.
    """
    import base64
    import io

    try:
        out = img.convert("RGB")
        if max(out.size) > PREVIEW_MAX_EDGE:
            scale = PREVIEW_MAX_EDGE / max(out.size)
            out = out.resize((max(1, int(out.width * scale)),
                              max(1, int(out.height * scale))))
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=PREVIEW_QUALITY)
        emit({"id": req_id, "type": "preview", "step": step,
              "total_steps": total,
              "jpeg": base64.b64encode(buf.getvalue()).decode("ascii")})
    except Exception as exc:
        # A preview is a convenience and must never take the run down with it.
        log(req_id, f"preview frame skipped: {exc}", "warn")


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
    #
    # The policy goes first, as in op_upscale: applied later by the first
    # `_load_model`, its quarter-of-budget ceiling silently raised the 1 GiB
    # cap low-RAM mode had just set, on exactly the first run that needed it.
    _apply_policy_once(req_id)
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

    # The board draws every panel through here, and a Metal out-of-memory takes
    # the whole engine with it rather than raising. Ask the machine first.
    _check_headroom(req_id, float(req.get("peak_gib") or 0))

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

    gen_started = time.monotonic()
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

    generate_ms = (time.monotonic() - gen_started) * 1000.0

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

    gen_started = time.monotonic()
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
    generate_ms = (time.monotonic() - gen_started) * 1000.0

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

    _apply_policy_once(req_id)

    CACHE.unload()
    if _helpers_resident() is not None:
        _unload_assistant()

    log(req_id, f"loading {backend} via mflux (quantize={quantize})")
    emit({"id": req_id, "type": "progress", "phase": "load", "progress": 0.0,
          "message": f"Loading {backend}"})

    cls = getattr(importlib.import_module(module_name), class_name)
    t0 = time.monotonic()
    kwargs: dict[str, Any] = {
        "model_config": getattr(ModelConfig, factory)(),
        "quantize": quantize,
        "model_path": model_path,
    }
    if loras:
        # The same refusal the unified route makes. This route did not check,
        # so a file that holds no adapter weights reached mflux and came back
        # as "did not match any known adapter keys" -- which names the file but
        # not what is wrong with it, nor what to do. Reported from use.
        for spec in loras:
            _check_is_adapter(req_id, str(spec.get("path") or ""))
        kwargs["lora_paths"] = [l["path"] for l in loras]
        kwargs["lora_scales"] = [float(l.get("scale", 1.0)) for l in loras]

    with _downloads_allowed(req_id):
        model = call_tolerant(req_id, cls, {}, kwargs)

    CACHE.put(key, model, backend)
    load_ms = (time.monotonic() - t0) * 1000.0
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
        gen_started = time.monotonic()
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
        generate_ms = (time.monotonic() - gen_started) * 1000.0

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


# Measured on this 16 GB machine, SeedVR2 3B q4, MLX cache limit 1 GiB:
#
#     output        peak
#     512x512     6.43 GiB
#     768x768    10.44 GiB
#    1024x1024   16.57 GiB
#
# Peak tracks output area almost exactly linearly over that range, which fits
# peak = 3.06 + 12.88 * output-megapixels to within 0.25 GiB. The constants are
# from those three points and nothing else; re-measure before trusting them for
# another model or another quantisation.
_UPSCALE_BASE_GIB = 3.06
_UPSCALE_GIB_PER_MPX = 12.88


def _upscale_peak_gib(width: int, height: int) -> float:
    """Estimated peak for producing an image this size."""
    return _UPSCALE_BASE_GIB + _UPSCALE_GIB_PER_MPX * (width * height) / 1e6


# What the tiled path holds beside the model, per pixel of the *output*: the
# float32 accumulator at twelve bytes, the weight sum at four, and while a
# preview is on the divided copy of the accumulator at another twelve plus the
# 8-bit image handed to the preview at three. None of that is in
# `_upscale_peak_gib`, which was fitted to one pass of the model and knows
# nothing about the canvas the pieces are blended into.
_TILE_CANVAS_BYTES_PER_PX = 32


def _upscale_canvas_gib(width: int, height: int) -> float:
    """What the tiled path holds outside the model, for an output this size."""
    return width * height * _TILE_CANVAS_BYTES_PER_PX / (1024 ** 3)


def _upscale_budget_gib() -> float:
    raw = os.environ.get("MODELSTUDIO_MEMORY_BUDGET_GIB")
    try:
        return float(raw) if raw else 12.0
    except ValueError:
        return 12.0


def _largest_upscale_edge(width: int, height: int, budget_gib: float) -> int:
    """The longest output edge that fits, keeping this image's proportions."""
    room = budget_gib - _UPSCALE_BASE_GIB
    if room <= 0:
        return 0
    max_px = room / _UPSCALE_GIB_PER_MPX * 1e6
    scale = math.sqrt(max_px / max(1, width * height))
    return int(max(width, height) * scale)


def _check_upscale_fits(req_id: str, src: str,
                        resolution: Any) -> tuple[int, int, int, int, float, bool]:
    """Refuse an upscale that cannot fit, before anything is dispatched.

    This has to happen up front. MLX reports a Metal out-of-memory from a
    command-buffer completion handler, which is a context that cannot raise
    into Python: the C++ exception reaches std::terminate and aborts the
    whole engine. There is no except clause that catches it, and the user
    sees "the engine stopped unexpectedly" with no idea which knob was wrong.
    So the size is checked here, where a refusal is still possible.
    """
    from PIL import Image

    with Image.open(src) as im:
        in_w, in_h = im.size

    if isinstance(resolution, (int, float)):
        # An absolute target for the shortest edge.
        factor = float(resolution) / max(1, min(in_w, in_h))
    else:
        factor = float(getattr(resolution, "value", 0) or
                       getattr(resolution, "factor", 0) or 0)
        if not factor:
            # Unrecognised: run it in one pass rather than guessing a factor
            # and tiling a request that would have been fine.
            return in_w, in_h, in_w, in_h, 1.0, True
    out_w, out_h = int(in_w * factor), int(in_h * factor)

    need = _upscale_peak_gib(out_w, out_h)
    budget = _upscale_budget_gib()
    log(req_id,
        f"upscale {in_w}x{in_h} -> {out_w}x{out_h}, "
        f"needs about {need:.1f} GiB of {budget:.1f} GiB")
    return in_w, in_h, out_w, out_h, factor, need <= budget * _UPSCALE_SAFETY


# Leave a margin under the budget: the estimate is a fit through three points,
# and being 10% optimistic here costs the whole engine, not just the request.
_UPSCALE_SAFETY = 0.85
# Overlap between neighbouring tiles, in output pixels. Wide enough that the
# feathered blend has somewhere to happen and narrow enough not to double the
# work; at 2x this is 32 source pixels.
_TILE_OVERLAP = 64


def _tile_spans(total: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Cover 0..total with windows of `tile`, overlapping by `overlap`.

    The last window is pulled back to end exactly at `total` rather than
    running past it, so no tile is padded and the edges stay real pixels.
    """
    if total <= tile:
        return [(0, total)]
    step = max(1, tile - overlap)
    spans = []
    start = 0
    while True:
        end = start + tile
        if end >= total:
            spans.append((max(0, total - tile), total))
            return spans
        spans.append((start, end))
        start += step


def _max_upscale_tiles() -> int:
    """How many pieces one enlargement may be cut into.

    Each piece is a separate pass of the upscaler at roughly fifteen seconds,
    so sixty-four is already a quarter of an hour of work. An env var because
    somebody who knows what they are waiting for should be able to wait.
    """
    raw = os.environ.get("MODELSTUDIO_MAX_UPSCALE_TILES")
    try:
        return max(1, int(raw)) if raw else 64
    except ValueError:
        return 64


def _tile_plan(in_w: int, in_h: int, factor: float, budget_gib: float
               ) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    """The tile grid for one tiled enlargement, and the tile size it implies.

    Empty spans mean there is no room at all: the two full-canvas accumulators
    are counted here, and the tile size used to be worked out from the budget
    as though they did not exist. At 5x on a 3024x4032 import they come to
    about 10 GiB on their own, so every tile size the old arithmetic chose was
    one the machine could not actually hold.
    """
    out_w, out_h = round(in_w * factor), round(in_h * factor)
    room = (budget_gib * _UPSCALE_SAFETY - _UPSCALE_BASE_GIB
            - _upscale_canvas_gib(out_w, out_h))
    if room <= 0:
        return 0, 0, [], []
    tile_out = int(math.sqrt(room / _UPSCALE_GIB_PER_MPX * 1e6))
    tile_src = max(64, int(tile_out / factor))
    overlap = int(_TILE_OVERLAP / factor)
    return (tile_src, tile_out,
            _tile_spans(in_w, min(tile_src, in_w), overlap),
            _tile_spans(in_h, min(tile_src, in_h), overlap))


def _largest_tiled_factor(in_w: int, in_h: int, wanted: float,
                          budget_gib: float) -> float:
    """The largest enlargement of this image that is still worth starting.

    Rounded down to a tenth, so the number in a refusal is one the user can
    type back in and have accepted. Found by bisection rather than algebra:
    the tile size depends on the canvas, the canvas depends on the factor, and
    the tile count depends on both with two clamps in between. Zero means not
    even 1x fits, which is a budget problem rather than a size problem.
    """
    limit = _max_upscale_tiles()

    def fits(candidate: float) -> bool:
        _src, _out, xs, ys = _tile_plan(in_w, in_h, candidate, budget_gib)
        count = len(xs) * len(ys)
        return bool(count) and count <= limit

    lo, hi = 1.0, max(1.0, wanted)
    if not fits(lo):
        return 0.0
    for _ in range(24):
        mid = (lo + hi) / 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return math.floor(lo * 10) / 10


def _refuse_tiled_upscale(in_w: int, in_h: int, factor: float,
                          budget_gib: float, tiles: int) -> None:
    """Refuse a tiled enlargement before the first piece is dispatched.

    Each tile was bounded and nothing bounded the number of them, nor the two
    full-canvas float32 accumulators the blend holds for the length of the run.
    A 3024x4032 import at 5x asked for about 690 pieces -- hours of the
    upscaler -- and 10 GiB of canvas before the model allocated anything, on
    exactly the 16 GB machine the tiled path exists to protect. It is said here
    because once the loop is running the only honest outcome left is a
    cancelled job, and because a refusal with a workable size in it is a
    setting the user can change rather than a wall.
    """
    out_w, out_h = round(in_w * factor), round(in_h * factor)
    workable = _largest_tiled_factor(in_w, in_h, factor, budget_gib)
    if workable > 1.0:
        instead = (f"Enlarging to {round(in_w * workable)}x"
                   f"{round(in_h * workable)} ({workable:g}x) would work.")
    else:
        instead = (f"Nothing larger than the picture itself fits in the "
                   f"{budget_gib:.0f} GiB budget, so crop it first.")
    if budget_gib * _UPSCALE_SAFETY <= _UPSCALE_BASE_GIB:
        why = (f"the upscaler needs about {_UPSCALE_BASE_GIB:.0f} GiB before "
               f"it has looked at the picture, and the budget is "
               f"{budget_gib:.0f} GiB")
    elif tiles:
        why = (f"it would be cut into {tiles} pieces, each a separate pass of "
               f"the upscaler")
    else:
        why = (f"holding the finished picture alone takes "
               f"{_upscale_canvas_gib(out_w, out_h):.1f} GiB of the "
               f"{budget_gib:.0f} GiB budget, which leaves the upscaler "
               f"nothing to work in")
    raise ValueError(
        f"{out_w}x{out_h} is too large to enlarge in pieces: {why}. "
        f"{instead} Nothing was started."
    )


def _upscale_tiled(req_id: str, model: Any, src: str, factor: float,
                   budget_gib: float, seed: int = 0,
                   preview: bool = True) -> Any:
    """Enlarge an image in overlapping pieces, each small enough to fit.

    A single pass is bounded by memory, not by the picture: on a 16 GB machine
    SeedVR2 tops out near 833 pixels on the longest edge, which is not an
    upscale of anything worth upscaling. Tiling trades time for size -- each
    tile is a separate pass, so the cost is linear in area -- and the seams are
    handled by overlapping the tiles and blending them with a feathered
    weight, so no join lands on a hard edge.
    """
    from PIL import Image
    import numpy as np

    with Image.open(src) as im:
        source = im.convert("RGB")
    in_w, in_h = source.size
    out_w, out_h = round(in_w * factor), round(in_h * factor)

    # Largest tile whose *output* fits beside the canvases below, expressed
    # back in source pixels, and how many of those the picture comes to.
    tile_src, tile_out, xs, ys = _tile_plan(in_w, in_h, factor, budget_gib)
    total = len(xs) * len(ys)
    if not total or total > _max_upscale_tiles():
        _refuse_tiled_upscale(in_w, in_h, factor, budget_gib, total)
    log(req_id,
        f"tiling {in_w}x{in_h} -> {out_w}x{out_h} as {len(xs)}x{len(ys)} "
        f"tiles of up to {tile_src}px source ({tile_out}px out), over "
        f"{_upscale_canvas_gib(out_w, out_h):.1f} GiB of canvas")

    acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    wsum = np.zeros((out_h, out_w, 1), dtype=np.float32)

    import vaultcrypto as vc

    done = 0
    for (y0, y1) in ys:
        for (x0, x1) in xs:
            # Between pieces is the only place this loop can be stopped. Each
            # piece is a single opaque call into the model, so without a check
            # here Cancel marks the job cancelled, the interface stops showing
            # it, and the engine grinds on through every remaining tile at
            # fifteen seconds each -- which reads, correctly, as a hang.
            if is_cancelled(req_id):
                raise Cancelled()
            done += 1
            emit({"id": req_id, "type": "progress", "phase": "denoise",
                  "progress": (done - 1) / total,
                  "message": f"Enlarging piece {done} of {total}"})
            crop = source.crop((x0, y0, x1, y1))
            tmp = os.path.join(_stage_dir(), f"tile_{req_id}_{done}.png")
            crop.save(tmp)
            try:
                shortest = int(min(crop.size) * factor)
                # One seed for every tile: a different seed per tile would
                # make neighbouring pieces disagree about texture, and the
                # blend cannot hide that.
                art = model.generate_image(seed=seed, image_path=tmp,
                                           resolution=shortest)
                piece = Image.open(io.BytesIO(vc.artifact_to_png_bytes(art)))
                piece = piece.convert("RGB")
            finally:
                _discard_staged([tmp])

            # Place tiles on an exact grid derived from the same rounding as
            # the canvas, and make each piece fit its cell exactly. The model
            # rounds its own output to whatever its resolution target implied,
            # and trusting that left a one-pixel column at the far edge that
            # no tile covered -- which divides by ~0 in the blend and paints a
            # black line down the side of the finished picture.
            px0, px1 = round(x0 * factor), round(x1 * factor)
            py0, py1 = round(y0 * factor), round(y1 * factor)
            px1, py1 = min(px1, out_w), min(py1, out_h)
            pw, ph = px1 - px0, py1 - py0
            if pw <= 0 or ph <= 0:
                continue
            if piece.size != (pw, ph):
                piece = piece.resize((pw, ph), Image.LANCZOS)
            arr = np.asarray(piece, dtype=np.float32)

            # Feather towards every edge that has a neighbour, so the blend
            # happens inside the overlap and never at the picture's border.
            wx = np.ones(pw, dtype=np.float32)
            wy = np.ones(ph, dtype=np.float32)
            fade = max(1, int(_TILE_OVERLAP))
            if x0 > 0:
                wx[:fade] = np.linspace(0.0, 1.0, min(fade, pw), dtype=np.float32)[:fade]
            if x1 < in_w:
                wx[-fade:] = np.linspace(1.0, 0.0, min(fade, pw), dtype=np.float32)[-fade:]
            if y0 > 0:
                wy[:fade] = np.linspace(0.0, 1.0, min(fade, ph), dtype=np.float32)[:fade]
            if y1 < in_h:
                wy[-fade:] = np.linspace(1.0, 0.0, min(fade, ph), dtype=np.float32)[-fade:]
            w = (wy[:, None] * wx[None, :])[:, :, None]

            acc[py0:py0 + ph, px0:px0 + pw] += arr * w
            wsum[py0:py0 + ph, px0:px0 + pw] += w

            # Show the picture so far. Tiling is the slow path -- minutes for a
            # large image -- and without this it is a counter and a blank
            # frame. Dividing by the running weight makes finished tiles look
            # finished instead of dark.
            if preview:
                so_far = np.divide(acc, np.maximum(wsum, 1e-6))
                _emit_preview_image(
                    req_id,
                    Image.fromarray(np.clip(so_far, 0, 255).astype(np.uint8)),
                    step=done, total=total)
                del so_far
            gc.collect()

    # Every pixel must have been written by at least one tile. With the exact
    # grid above that holds by construction, so a gap here means the geometry
    # is wrong and the picture would carry a black seam -- better to say so
    # than to hand back a spoiled image.
    uncovered = int((wsum <= 0).sum())
    if uncovered:
        raise RuntimeError(
            f"tiling left {uncovered} pixels uncovered; refusing to return a "
            "picture with a seam in it"
        )
    blended = np.divide(acc, wsum)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


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
        # Before the explicit cache limit below, not after: the policy's own
        # ceiling is a quarter of the budget and would otherwise *raise* a
        # limit the request had deliberately lowered.
        _apply_policy_once(req_id)

        cache_limit = req.get("cache_limit_gb")
        if cache_limit is None and req.get("low_ram"):
            cache_limit = 1.0
        if cache_limit is not None:
            log(req_id, f"MLX cache limit: {set_cache_limit(float(cache_limit))}")

        _check_headroom(req_id, float(req.get("peak_gib") or 0))

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

        _in_w, _in_h, out_w, out_h, factor, fits = _check_upscale_fits(
            req_id, sources[0], resolution)

        import vaultcrypto as vc

        if fits:
            unsubscribe = _subscribe(req_id, model, "image-to-image")
            preview = (_attach_live_preview(req_id, model)
                       if req.get("preview", True) else None)
            try:
                artifact = call_tolerant(
                    req_id,
                    model.generate_image,
                    {
                        "seed": int(req.get("seed", 0)),
                        "image_path": sources[0],
                        "resolution": resolution,
                    },
                    {},
                )
            finally:
                _detach_live_preview(model, preview)
                unsubscribe()
            png = vc.artifact_to_png_bytes(artifact)
        else:
            # Too big for one pass. Tiling is slower but it is the difference
            # between an enlarged picture and an aborted engine.
            emit({"id": req_id, "type": "progress", "phase": "denoise",
                  "progress": 0.0,
                  "message": f"Too large for one pass \u2014 enlarging to "
                             f"{out_w}x{out_h} in pieces"})
            image = _upscale_tiled(req_id, model, sources[0], factor,
                                   _upscale_budget_gib(),
                                   seed=int(req.get("seed", 0)),
                                   preview=bool(req.get("preview", True)))
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            png = buf.getvalue()
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


# Below this many output pixels, Wan stops converging and returns colour.
#
# Measured on Wan 2.2 TI2V-5B q8, same seed and prompt, 20 steps: 320x192 and
# 480x272 are noise; 640x368 and 704x384 are correct pictures. The threshold
# sits between 130k and 235k pixels, so the floor is set at the first size
# known to work rather than at the edge of what has been tried.
#
# This is not a memory limit and must not be confused with one. 704x384 is 4.4
# times the pixels of 320x192 and peaks 0.5 GiB higher -- 10.20 against 9.72.
# What it costs is time. A catalog figure nobody measured made video look too
# large for this machine, everything was pushed to the smallest canvas to fit,
# and that canvas was the whole fault.
_VIDEO_MIN_PIXELS = 640 * 368


def _check_video_size(req_id: str, width: int, height: int) -> None:
    """Refuse a canvas the model cannot resolve on."""
    if width * height >= _VIDEO_MIN_PIXELS:
        return
    raise ValueError(
        f"{width}x{height} is too small for this model to draw on: below about "
        f"640x368 it returns coloured noise rather than a picture. Nothing was "
        f"run. Choose a larger size -- it costs time rather than memory, and "
        f"the same clip at 640x368 uses barely more than at {width}x{height}."
    )


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

        # image_count has to be passed exactly once. It was being set inside
        # plan_kw above *and* passed explicitly as 0 here, which is a
        # TypeError before any work starts -- so animating a still has never
        # reached the model on a first-frame route, whatever model was chosen.
        # Bounded before the headroom check, like every other route that
        # allocates, so the check measures against the process's own ceiling.
        _apply_policy_once(req_id)
        _check_headroom(req_id, float(req.get("peak_gib") or 0))
        _check_video_size(req_id, int(req.get("width", 480)),
                          int(req.get("height", 320)))

        loaded, load_ms = _load_model(
            req_id,
            req["model"],
            req.get("quantize"),
            req.get("model_path"),
            image_count=int(plan_kw.pop("image_count", 0)),
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
        # A clip is the longest wait in the app -- minutes, for a result that
        # may be wrong from the first step. If the route exposes a step
        # callback this shows it forming; if not, it logs once and stays out
        # of the way.
        preview = (_attach_live_preview(req_id, loaded)
                   if req.get("preview", True) else None)
        gen_started = time.monotonic()
        try:
            video = call_tolerant(req_id, target.generate_video, base, optional)
        finally:
            _detach_live_preview(loaded, preview)
        generate_ms = (time.monotonic() - gen_started) * 1000.0

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



def _story_writer(req_id, req, voice):
    """Writer, progress and passage size shared by analysis and resume.

    Stop is turned into `story.Stop`, so the finished passages come back as a
    result instead of the whole analysis being lost."""
    import story as story_model
    small = req.get("writer", WRITER_REPO) == WRITER_REPO
    # Token events from the writer carry the scene label and fraction too;
    # otherwise they replace "scene i of N" within a fifth of a second.
    where = {"label": "Understanding the story", "fraction": None}
    started = time.monotonic()
    def progress(i, total):
        if is_cancelled(req_id):
            raise story_model.Stop()
        label = f"{'Casting voices in' if voice else 'Understanding'} scene {i + 1} of {total}"
        if i:
            left = (time.monotonic() - started) / i * (total - i)
            label += f" · about {max(1, round(left / 60))} min left"
        where.update(label=label, fraction=i / total)
        emit({"id": req_id, "type": "progress", "phase": "denoise",
              "progress": where["fraction"], "message": where["label"]})
    def write(system, user):
        # A repair carries the failed reply as well; a reply cut off at the
        # first budget would be cut off again at the same one. A compact reply
        # runs to a few hundred tokens; the ceiling only stops a writer that
        # has started repeating itself, which at 3,200 tokens cost minutes.
        repair = "Your previous reply failed validation" in user
        budget = (1000, 1600) if voice else (1800, 2700)
        try:
            return _write(req_id, system, user, max_tokens=budget[repair],
                          temperature=0.2, repo=req.get("writer"), label=where["label"],
                          fraction=where["fraction"])
        except Cancelled:
            raise story_model.Stop()
    def warn(message):
        log(req_id, message, "warn")
    # A voice reply is a few tokens per quote, so a passage can be far longer
    # than one whose every beat is directed for drawing. Compact replies let
    # the small writer take twice the four sentences it once could.
    words, units = (700, 60) if voice else (450, 8) if small else (600, 12)
    return dict(write=write, progress=progress, warn=warn, passage_words=words, passage_units=units)


def req_format(req):
    fmt = req.get("format") or "manga"
    return fmt if fmt in ("manga", "webtoon", "comic", "picture") else "manga"


def req_pages(req):
    """The pages the board is planned for: four unless asked, at most 200."""
    try:
        return max(1, min(200, int(req.get("pages") or 4)))
    except (TypeError, ValueError):
        return 4


def _brief_writer(req_id, req):
    """Writer and progress for the briefs the director's rules left weak.

    One batch of panels per call, so a chapter costs a handful of short
    replies rather than one per scene; Stop keeps the briefs done so far."""
    import director
    started = time.monotonic()
    where = {"label": "Writing panel briefs", "fraction": None}
    def progress(i, total):
        label = f"Writing panel briefs {i + 1} of {total}"
        if i:
            label += f" · about {max(1, round((time.monotonic() - started) / i * (total - i) / 60))} min left"
        where.update(label=label, fraction=i / total)
        emit({"id": req_id, "type": "progress", "phase": "denoise", "progress": i / total, "message": label})
    def write(system, user):
        if is_cancelled(req_id):
            raise director.Stop()
        try:
            return _write(req_id, system, user, max_tokens=1200, temperature=0.3, repo=req.get("writer"),
                          label=where["label"], fraction=where["fraction"])
        except Cancelled:
            raise director.Stop()
    return write, progress


def _directed(req_id, req, book, chapter_id, refine):
    """The chapter's panels and what of the story they carry.

    The coverage is measured, sentence by sentence: the board used to report
    100% whatever the plan had left out."""
    import director
    fmt, pages = req_format(req), req_pages(req)
    if refine and req.get("writer"):
        write, progress = _brief_writer(req_id, req)
        panels = director.direct(book, chapter_id, fmt, write=write, progress=progress, pages=pages)
    else:
        panels = director.plan(book, chapter_id, fmt, pages)
    return panels, director.coverage(director.script(book, chapter_id), panels)


def _story_result(req_id, book, chapter_id, req=None):
    import story as story_model
    req = req or {}
    book = story_model.ensure_cast(book)
    chapter = next(c for c in book["chapters"] if c["id"] == chapter_id)
    left = sum(1 for s in chapter["scenes"] if s.get("unanalysed"))
    stopped = is_cancelled(req_id)
    # The briefs are part of the one automatic pass, while the writer is
    # still loaded -- unless the analysis itself was stopped.
    panels, covered = _directed(req_id, req, book, chapter_id, refine=not stopped)
    return {"book": book, "chapter_id": chapter_id, "panels": panels, "coverage": covered,
            "stopped": stopped, "unanalysed": left}


def op_story_analyze(req_id, req):
    import story as story_model
    mode = req.get("mode") or "full"
    book = story_model.analyze(req.get("source"), previous=req.get("previous"),
                               title=req.get("title", "Chapter"), mode=mode,
                               **_story_writer(req_id, req, mode == "voice"))
    return _story_result(req_id, book, book["chapters"][-1]["id"], req)


def op_story_resume(req_id, req):
    import story as story_model
    if req.get("again"):
        # Analyse the whole chapter again, in place: the same book, chapter
        # and character ids, so narrations and boards filed under them stay.
        req = {**req, "book": {**req["book"], "chapters": [
            {**c, "scenes": [{**s, "unanalysed": True} for s in c["scenes"]]} if c["id"] == req.get("chapter_id") else c
            for c in req["book"]["chapters"]]}}
    chapter = next((c for c in req["book"]["chapters"] if c["id"] == req.get("chapter_id")), None)
    voice = bool(chapter and chapter.get("analysis") == "voice")
    book = story_model.resume(req["book"], req.get("chapter_id"), **_story_writer(req_id, req, voice))
    return _story_result(req_id, book, req["chapter_id"], req)


def op_direct_story(req_id, req):
    """The shot list for a chapter at the page count asked for: the
    director's rules, with weak briefs rewritten by the story model when one
    is named. A rewrite that fails or is stopped keeps the rules' briefs."""
    import story as story_model
    # A book analysed before a self-named narrator was merged, or before a
    # speaking role was a person, is put right here; the book goes back so
    # the board can save it and cast the people it gained.
    book = story_model.ensure_cast(req["book"])
    extra = {"book": book} if book is not req["book"] else {}
    panels, covered = _directed(req_id, req, book, req["chapter_id"], refine=True)
    return {"panels": panels, "coverage": covered, "stopped": is_cancelled(req_id), **extra}


def op_narration_script(req_id, req):
    """Retell a chapter for one storyteller, optionally in Devanagari."""
    import narration
    started = time.monotonic()
    where = {"label": "Preparing the narration script", "fraction": None}
    def progress(i, total):
        label = f"Preparing passage {i + 1} of {total}"
        if i:
            label += f" · about {max(1, round((time.monotonic() - started) / i * (total - i) / 60))} min left"
        where.update(label=label, fraction=i / total)
        emit({"id": req_id, "type": "progress", "phase": "denoise", "progress": i / total, "message": label})
    def write(system, user):
        if is_cancelled(req_id):
            raise narration.Stop()
        try:
            return _write(req_id, system, user, max_tokens=2400, temperature=0.3, repo=req.get("writer"),
                          label=where["label"], fraction=where["fraction"])
        except Cancelled:
            raise narration.Stop()
    text, stopped = narration.script(req.get("source"), write, bool(req.get("devanagari")), progress)
    return {"script": text, "stopped": stopped}


def op_audio_voices(_req_id, _req):
    import audiobook
    model = _req.get("model_id")
    if model == audiobook.KOKORO_MODEL_ID:
        return {"voices": audiobook.kokoro_voices()}
    return {"voices": audiobook.neural_voices() if model == audiobook.NEURAL_MODEL_ID else audiobook.voices()}


def _install_kokoro_packages(req_id):
    """Kokoro's text packages, pinned, installed beside the runtime's own.

    They come with the model rather than with setup so an existing install
    gains them by downloading Kokoro, without rebuilding the runtime."""
    lock = Path(__file__).with_name("kokoro-requirements.lock")
    emit({"id": req_id, "type": "progress", "phase": "download", "progress": None,
          "message": "Installing Kokoro's text packages"})
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--disable-pip-version-check",
                           "--quiet", "-r", str(lock)], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError("Kokoro's text packages could not be installed: "
                           + (proc.stderr.strip().splitlines() or ["pip failed"])[-1])


# The speech model kept between voice previews, so auditioning several voices
# does not reload it each time. Released by `_unload_assistant` -- before any
# other job and by the idle reaper -- so it never sits beside an image model.
_VOICE: dict[str, Any] = {"model": None, "id": None}

# Requests that may find the preview voice model loaded and leave it so.
_KEEPS_VOICE = {"audio_preview", "audio_voices", "ping", "cancel"}


def op_audio_preview(req_id, req):
    """A few seconds of one voice, returned in memory and never written out."""
    import audiobook
    import base64
    text = str(req.get("text") or "").strip()[:300] or "This is how I will read your story."
    voice, rate = req.get("voice"), req.get("rate", 160)
    def check():
        if is_cancelled(req_id):
            raise Cancelled()
    segment = [{"speaker_id": None, "text": text}]
    model_id = req.get("model_id") or "macos"
    if model_id in (audiobook.NEURAL_MODEL_ID, audiobook.KOKORO_MODEL_ID) and _VOICE.get("id") != model_id:
        _VOICE.update(model=None, id=None)
    if model_id == audiobook.KOKORO_MODEL_ID:
        if _VOICE["model"] is None:
            CACHE.unload()
            _unload_assistant()
            _VOICE.update(model=audiobook.load_kokoro_model(), id=model_id)
        data, _ = audiobook.synthesize_kokoro(segment, voice, {}, rate, check, load=lambda: _VOICE["model"])
    elif model_id == audiobook.NEURAL_MODEL_ID:
        if _VOICE["model"] is None:
            CACHE.unload()
            _unload_assistant()
            emit({"id": req_id, "type": "progress", "phase": "load", "progress": 0.0,
                  "message": "Loading neural audiobook voices"})
            _VOICE.update(model=audiobook.load_neural_model(), id=model_id)
        data, _ = audiobook.synthesize_neural(segment, voice, {}, rate, check,
            language=req.get("language") or "auto", delivery=req.get("delivery") or "",
            directions={None: str(req.get("direction") or "")[:1500]},
            load=lambda: _VOICE["model"])
    else:
        data, _ = audiobook.synthesize(segment, voice, {}, rate, _stage_dir(), check)
    return {"wav_base64": base64.b64encode(data).decode("ascii")}


def op_audiobook(req_id, req):
    import audiobook
    import story as story_model
    import vaultcrypto as vc
    slots = req.get("vault_slots") or []
    if len(slots) != 1:
        raise ValueError("audiobook needs one encrypted output slot")
    # A prepared narration script is read by one storyteller; so is the
    # chapter itself when one voice was chosen for everyone.
    if isinstance(req.get("script"), str) and req["script"].strip():
        import narration
        segments = narration.segments(req["script"])
        req = {**req, "voices": {}, "directions": {}}
    else:
        segments = story_model.narration_segments(req["book"], req["chapter_id"])
        if req.get("single"):
            req = {**req, "voices": {}, "directions": {}}
    import narration as _narration
    segments = [{**s, "text": _narration.speakable(s["text"])} for s in segments]
    CACHE.unload()
    _unload_assistant()
    def check():
        if is_cancelled(req_id):
            raise Cancelled()
    def progress(done, total):
        emit({"id": req_id, "type": "progress", "phase": "denoise",
              "progress": done / total, "message": f"Narrating ({round(100 * done / total)}%)"})
    model_id = req.get("model_id") or "macos"
    markers: list[dict[str, Any]] = []
    if model_id == audiobook.NEURAL_MODEL_ID:
        _apply_policy_once(req_id)
        emit({"id": req_id, "type": "progress", "phase": "load", "progress": 0.0,
              "message": "Loading neural audiobook voices"})
        data, seconds = audiobook.synthesize_neural(segments, req["narrator"],
            req.get("voices") or {}, req.get("rate", 160), check, progress,
            language=req.get("language") or "auto", delivery=req.get("delivery") or "",
            directions=req.get("directions") or {},
            character_directions={c["id"]: c.get("voice", "") for c in req["book"]["characters"]},
            markers=markers)
    elif model_id == audiobook.KOKORO_MODEL_ID:
        emit({"id": req_id, "type": "progress", "phase": "load", "progress": 0.0,
              "message": "Loading Kokoro voices"})
        data, seconds = audiobook.synthesize_kokoro(segments, req["narrator"],
            req.get("voices") or {}, req.get("rate", 160), check, progress, markers=markers)
    elif model_id == "macos":
        data, seconds = audiobook.synthesize(segments, req["narrator"],
            req.get("voices") or {}, req.get("rate", 160), _stage_dir(), check, progress, markers=markers)
    else:
        raise ValueError("unsupported audiobook speech model")
    check()
    slot = slots[0]
    size = vc.write_sealed(slot["path"], bytes.fromhex(slot["key"]),
                           bytes.fromhex(slot["file_id"]), data)
    return {"outputs": [slot["id"]], "sizes": [size], "seconds": seconds, "sealed": True,
            "markers": markers}


def op_character_reference(req_id, req):
    """Two to eight cast sheets side by side, in the order given: the lineup
    a panel with several people is drawn against when the model takes one
    reference. The sheets are full-body figures drawn at 2:3, so each fills
    its tile, and the board sends them in the order the people stand, left
    to right, which is how the panel prompt names them."""
    from PIL import Image, ImageOps
    import vaultcrypto as vc
    inputs = req.get("vault_inputs") or []
    slots = req.get("vault_slots") or []
    if not 2 <= len(inputs) <= 8 or len(slots) != 1:
        raise ValueError("compose between two and eight individual character sheets")
    canvas = Image.new("RGB", (512 * len(inputs), 768), "white")
    for i, item in enumerate(inputs):
        if is_cancelled(req_id):
            raise Cancelled()
        data = vc.open_with_file_key(bytes.fromhex(item["key"]), Path(item["path"]).read_bytes())
        with Image.open(io.BytesIO(data)) as original:
            image = ImageOps.contain(original.convert("RGB"), (512, 768))
            canvas.paste(image, (512 * i + (512 - image.width) // 2, (768 - image.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    slot = slots[0]
    size = vc.write_sealed(slot["path"], bytes.fromhex(slot["key"]),
                           bytes.fromhex(slot["file_id"]), buf.getvalue())
    return {"outputs": [slot["id"]], "sizes": [size], "sealed": True}


OPS = {
    "ping": op_ping,
    "story_analyze": op_story_analyze,
    "story_resume": op_story_resume,
    "direct_story": op_direct_story,
    "narration_script": op_narration_script,
    "audio_voices": op_audio_voices,
    "audiobook": op_audiobook,
    "audio_preview": op_audio_preview,
    "character_reference": op_character_reference,
    "capabilities": op_capabilities,
    "lora_info": op_lora_info,
    "verify_adapter": op_verify_adapter,
    "image_formats": op_image_formats,
    "resolve": op_resolve,
    "download": op_download,
    "generate": op_generate,
    "edit": op_edit,
    "upscale": op_upscale,
    "video": op_video,
    "unload": op_unload,
    "assist": op_assist,
    "compose_board": op_compose_board,
    "bind_comic": op_bind_comic,
    "unload_assistant": op_unload_assistant,
    "set_memory": op_set_memory,
}


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def handle(req: dict[str, Any]) -> None:
    req_id = req.get("id", "?")
    op = req.get("op", "")
    if op not in _KEEPS_VOICE and _VOICE["model"] is not None:
        _VOICE.update(model=None)
        gc.collect()
    fn = OPS.get(op)
    if fn is None:
        emit({"id": req_id, "type": "error", "error": f"unknown op: {op}",
              "kind": "protocol"})
        return
    with _CANCEL_LOCK:
        _ACTIVE.add(req_id)
    _touch()
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
