"""Read-only prompt-assistant diagnostics; no downloads or user-vault access.

Default: execute production pure logic and op_assist with stubbed inference.
Optional: --live-writer /absolute/path/to/installed/model/snapshot runs the real
text writer on synthetic prompts, with network disabled. It does not generate
images. JSON lines are observations, NOT a passing acceptance test suite.
Exit zero means diagnostics executed, not that the enhancer is correct.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
original_stdout = sys.stdout
import worker  # noqa: E402

# The worker normally reserves stdout for its protocol. This standalone report
# owns stdout, while inference progress/logs are suppressed below.
sys.stdout = original_stdout


def report(identifier: str, **data: object) -> None:
    print(json.dumps({"id": identifier, **data}, ensure_ascii=False), flush=True)


def stubbed_assist(prompt: str, reply: str, *, mode: str = "generate",
                   image: bool = False, reader: bool = False,
                   semantic_same: bool = False) -> tuple[dict, list[str]]:
    vlm = ModuleType("mlx_vlm")
    vlm.generate = lambda *a, **k: "SUBJECT: a green field jacket"
    template = ModuleType("mlx_vlm.prompt_utils")
    template.apply_chat_template = lambda *a, **k: "stubbed image question"
    systems: list[str] = []

    def write(_id, system, _user, **_kwargs):
        systems.append(
            "image_direction" if system == worker.SCENE_DIRECTION_SYSTEM else
            "video_direction" if system == worker.MOTION_DIRECTION_SYSTEM else
            "edit_clarification" if system == worker.EDIT_CLARIFY_SYSTEM else
            "other")
        return reply

    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {
            "mlx_vlm": vlm, "mlx_vlm.prompt_utils": template,
        }))
        for name, value in {
            "_write": write,
            "_same_request": lambda *a, **k: semantic_same,
            "_stage_vault_inputs": lambda _inputs: ["synthetic.png"] if image else [],
            "_downscale_for_assist": lambda paths: paths,
            "_discard_staged": lambda paths: None,
            "_load_assistant": lambda *a: (None, None, None),
            "emit": lambda *a, **k: None,
            "log": lambda *a, **k: None,
        }.items():
            stack.enter_context(patch.object(worker, name, value))
        result = worker.op_assist("audit", {
            "prompt": prompt, "mode": mode,
            "assistant": "stub-reader" if reader else None,
        })
    return result, systems


def deterministic() -> None:
    pairs = [
        ("PD-01", "negation", "a red car with no sunroof", "a red car with a sunroof"),
        ("PD-02", "numeric count", "2 red cars", "3 red cars"),
        ("PD-03", "spatial relation", "a red cube left of a blue sphere",
         "a red cube right of a blue sphere"),
        ("PD-04", "actor/recipient", "Mira gives Jonas the key",
         "Jonas gives Mira the key"),
        ("PD-05", "attribute binding", "a red cube and a blue sphere",
         "a blue cube and a red sphere"),
        ("PD-06", "motion", "a cat walking through a forest", "a cat sitting in a forest"),
    ]
    for identifier, dimension, original, candidate in pairs:
        result, reason = worker._clarified_with_reason(original, candidate)
        report(identifier, dimension=dimension, original=original,
               candidate=candidate, accepted=result is not None, reason=reason,
               desired="Reject or prominently flag the changed explicit constraint")

    for identifier, prompt in [
        ("PD-07", 'A sign reading "BEST QUALITY 4K"'),
        ("PD-08", "a masterpiece hanging in a gallery"),
        ("PD-09", "Unreal Engine editor interface"),
    ]:
        cleaned, removed = worker._clean_prompt_request(prompt)
        report(identifier, original=prompt, cleaned=cleaned, removed=removed,
               desired="Preserve literal text and meaningful subject/tool names")

    raw = "A red car.\nNo people. Keep the background plain."
    result, reason = worker._clarified_with_reason("a red car", raw)
    report("PD-10", raw=raw, result=result, reason=reason,
           desired="Preserve all candidate constraints or reject incomplete parsing")

    result, systems = stubbed_assist(
        "make the jacket red", "a red jacket on a mannequin in an ornate room",
        mode="edit", image=True, reader=False)
    report("PD-11", systems=systems, result=result,
           desired="Text-only edit policy, not image scene expansion; visible blind-mode notice")

    result, systems = stubbed_assist(
        "make green field jacket red",
        "change green field coat to red, leave everything else unchanged.",
        mode="edit", image=True, reader=True)
    report("PD-12", systems=systems, result=result,
           desired="Consistent edit-validator outcome and actionable constraint warning")

    result, systems = stubbed_assist(
        "my dog but make him look like a king",
        "A golden retriever in a velvet robe and a gold crown.",
        semantic_same=True)
    report("PD-13", systems=systems, result=result,
           desired="One consistent decision after semantic adjudication, not a later lexical veto")

    with patch.object(worker, "_write", side_effect=worker.Cancelled()), \
            patch.object(worker, "log", lambda *a, **k: None):
        try:
            result = worker._clarify_edit("audit", "make it red", {"SUBJECT": "jacket"})
            report("PD-14", cancellation_propagated=False, fallback=result,
                   desired="Propagate Cancelled; do not return a successful fallback")
        except worker.Cancelled:
            report("PD-14", cancellation_propagated=True)
        try:
            result = worker._same_request("audit", "dog", "retriever", None)
            report("PD-15", cancellation_propagated=False, verdict=result,
                   desired="Propagate Cancelled; do not reinterpret it as DIFFERENT")
        except worker.Cancelled:
            report("PD-15", cancellation_propagated=True)

    # Fake model allocations demonstrate cache ownership only, not actual peak
    # memory or MLX behavior. Never load/download real models in this lane.
    lm = ModuleType("mlx_lm")
    lm.load = lambda repo: (object(), object())
    with patch.dict(sys.modules, {"mlx_lm": lm}), \
            patch.object(worker, "_WRITER", {"key": None, "model": None, "tokenizer": None}), \
            patch.object(worker, "_ASSIST", {"key": "reader", "model": object()}), \
            patch.object(worker, "emit", lambda *a, **k: None), \
            patch.object(worker, "log", lambda *a, **k: None):
        worker._load_writer("audit", "synthetic-writer")
        report("PD-16", reader_resident=worker._ASSIST["model"] is not None,
               writer_resident=worker._WRITER["model"] is not None,
               desired="Honor declared one-helper-resident policy during reader/writer transition")

    result, systems = stubbed_assist("a red car", "a red car", image=True, reader=True)
    report("PD-17", systems=systems, saw_image=result.get("saw_image"),
           description=result.get("description"),
           desired="saw_image reports actual vision execution, not merely supplied inputs")

    result, reason = worker._clarified_with_reason("a cat on a chair", "a hat on a chair")
    report("PD-18", positive_control="unrelated main subject rejected",
           accepted=result is not None, reason=reason)


LIVE_PROMPTS = [
    ("PL-01", "generate", "a cat on a chair"),
    ("PL-02", "generate", "2 red cars with no sunroof, plain white background"),
    ("PL-03", "generate", "a red cube left of a blue sphere"),
    ("PL-04", "generate", 'A sign reading exactly "BEST QUALITY 4K"'),
    ("PL-05", "generate", "Mira gives Jonas a brass key. Mira wears red; Jonas wears blue."),
    ("PL-06", "generate", "a hous at nite"),
    ("PL-07", "generate", "Black ink line drawing of a cat on a white background. No shading, no color."),
    ("PL-08", "generate", "make it better"),
    ("PL-09", "edit", "make the jacket red"),
    ("PL-10", "edit", "remove the person on the left, keep the person on the right unchanged"),
    ("PL-11", "video", "A runner sprints quickly across the track. Locked camera."),
    ("PL-12", "generate", "my dog but make him look like a king"),
]


def live(snapshot: str) -> None:
    model_path = Path(snapshot).resolve(strict=True)
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise ValueError("Pass an existing installed model snapshot containing config.json")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx

    with patch.object(worker, "emit", lambda *a, **k: None), \
            patch.object(worker, "log", lambda *a, **k: None):
        for index, (identifier, mode, prompt) in enumerate(LIVE_PROMPTS):
            mx.random.seed(100 + index)
            started = time.monotonic()
            result = worker.op_assist(identifier, {
                "prompt": prompt, "mode": mode, "assistant": None,
                "writer": str(model_path),
            })
            report(identifier, mode=mode, source=prompt,
                   seconds=round(time.monotonic() - started, 3), result=result)
        worker._unload_writer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-writer", help="Existing local snapshot; no download or image generation")
    args = parser.parse_args()
    if args.live_writer:
        live(args.live_writer)
    else:
        deterministic()
