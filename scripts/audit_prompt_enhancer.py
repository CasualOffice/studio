"""Read-only prompt-assistant diagnostics; no downloads or user-vault access.

Default: execute production pure logic and op_assist with stubbed inference.
Optional: --live-writer /absolute/path/to/installed/model/snapshot runs the real
text writer on synthetic prompts, with network disabled. It does not generate
images. JSON lines are observations, NOT a passing acceptance test suite.
Exit zero means diagnostics executed, not that the enhancer is correct.

The PD numbering is kept from the 2026-09 audit so observations can be
compared across versions; the internals each one probes moved to
engine/prompt_assist.py when the enhancer was rebuilt around constraints.
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
import prompt_assist as pa  # noqa: E402

# The worker normally reserves stdout for its protocol. This standalone report
# owns stdout, while inference progress/logs are suppressed below.
sys.stdout = original_stdout


def report(identifier: str, **data: object) -> None:
    print(json.dumps({"id": identifier, **data}, ensure_ascii=False), flush=True)


def stubbed_assist(prompt: str, reply: str, *, mode: str = "generate",
                   image: bool = False, reader: bool = False,
                   semantic_same: bool = False,
                   regions: str = "") -> tuple[dict, list[dict]]:
    """op_assist with the writer and reader answered by fixed text."""
    calls: list[dict] = []

    def write(_id, system, user, **kwargs):
        if system == worker.SAME_REQUEST_SYSTEM:
            calls.append({"kind": "same"})
            return "SAME" if semantic_same else "DIFFERENT"
        calls.append({"kind": "write", "temperature": kwargs.get("temperature"),
                      "blind": "You cannot see the picture" in system,
                      "edit": "image editor" in system})
        return reply

    req = {"prompt": prompt, "mode": mode, "goal": "enhance",
           "target": {"id": "flux2-klein-4b-4bit", "family": "flux2"}}
    if not reader:
        req["assistant"] = None
    with ExitStack() as stack:
        for name, value in {
            "_write": write,
            "_read_picture": lambda *a: regions,
            "_stage_vault_inputs": lambda _inputs: ["synthetic.png"] if image else [],
            "_downscale_for_assist": lambda paths: paths,
            "_discard_staged": lambda paths: None,
            "emit": lambda *a, **k: None,
            "log": lambda *a, **k: None,
        }.items():
            stack.enter_context(patch.object(worker, name, value))
        result = worker.op_assist("audit", req)
    return result, calls


def flagged(original: str, candidate: str) -> list[str]:
    return [c["text"] for c in pa.extract_constraints(original)
            if not pa.satisfied(c, candidate)]


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
        report(identifier, dimension=dimension, original=original,
               candidate=candidate, flagged=flagged(original, candidate),
               desired="Reject or prominently flag the changed explicit constraint")

    for identifier, prompt in [
        ("PD-07", 'A sign reading "BEST QUALITY 4K"'),
        ("PD-08", "a masterpiece hanging in a gallery"),
        ("PD-09", "Unreal Engine editor interface"),
    ]:
        cleaned, removed = pa.clean_request(prompt)
        report(identifier, original=prompt, cleaned=cleaned, removed=removed,
               desired="Preserve literal text and meaningful subject/tool names")

    raw = "A red car.\nNo people. Keep the background plain."
    parsed, how = pa.parse_reply(raw)
    report("PD-10", raw=raw, result=parsed and parsed["prompt"], how=how,
           desired="Preserve all candidate constraints or reject incomplete parsing")

    result, calls = stubbed_assist(
        "make the jacket red",
        json.dumps({"prompt": "Make the jacket red, keeping everything else unchanged."}),
        mode="edit", image=True, reader=False)
    report("PD-11", calls=calls, outcome=result["outcome"], note=result["note"],
           saw_image=result["saw_image"],
           desired="Text-only edit policy, not image scene expansion; visible blind-mode notice")

    result, calls = stubbed_assist(
        "make it red",
        json.dumps({"prompt": "Change the green field jacket in the centre to red, "
                              "leaving the wall unchanged."}),
        mode="edit", image=True, reader=True,
        regions="green field jacket | centre, foreground | green\n"
                "brick wall | background | brown")
    report("PD-12", calls=calls, outcome=result["outcome"],
           target_region=result["target_region"], regions=result["regions"],
           desired="Consistent edit-validator outcome and actionable constraint warning")

    result, calls = stubbed_assist(
        "my dog but make him look like a king",
        json.dumps({"subject": "dog",
                    "prompt": "A golden retriever in a velvet robe and a gold crown."}),
        semantic_same=True)
    report("PD-13", calls=calls, outcome=result["outcome"], dropped=result["dropped"],
           desired="One consistent decision after semantic adjudication, not a later lexical veto")

    with patch.object(worker, "_write", side_effect=worker.Cancelled()), \
            patch.object(worker, "log", lambda *a, **k: None), \
            patch.object(worker, "emit", lambda *a, **k: None):
        try:
            result = worker.op_assist("audit", {"prompt": "make it red", "mode": "edit",
                                                "assistant": None})
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

    result, calls = stubbed_assist("a red car", json.dumps({"prompt": "A red car."}),
                                   image=True, reader=True)
    report("PD-17", calls=calls, saw_image=result.get("saw_image"),
           description=result.get("description"),
           desired="saw_image reports actual vision execution, not merely supplied inputs")

    checks = pa.checks_for([], {"subject": "a cat"}, "a cat on a chair")
    report("PD-18", positive_control="unrelated main subject flagged",
           flagged=[c["text"] for c in checks if not pa.satisfied(c, "a hat on a chair")])


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

# What each mode's prompt is written for in the live run. Edits go to Kontext
# and clips to Wan, the two the studio offers for those modes on 16 GB.
LIVE_TARGETS = {
    "generate": {"id": "flux2-klein-4b-4bit", "family": "flux2"},
    "edit": {"id": "flux1-kontext-4bit", "family": "flux"},
    "video": {"id": "wan22-ti2v-5b-8bit", "family": "wan"},
}


def live(snapshot: str, goal: str, thinking_off: bool) -> None:
    model_path = Path(snapshot).resolve(strict=True)
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise ValueError("Pass an existing installed model snapshot containing config.json")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx

    # The worker turns reasoning off by repo name; a snapshot path is not one,
    # so a reasoning writer (Qwen3.5 9B, Qwen3 8B/14B) needs it said here.
    if thinking_off:
        worker.THINKING_WRITERS.add(str(model_path))
    with patch.object(worker, "emit", lambda *a, **k: None), \
            patch.object(worker, "log", lambda *a, **k: None):
        for index, (identifier, mode, prompt) in enumerate(LIVE_PROMPTS):
            mx.random.seed(100 + index)
            started = time.monotonic()
            result = worker.op_assist(identifier, {
                "prompt": prompt, "mode": mode, "assistant": None, "goal": goal,
                "target": LIVE_TARGETS[mode], "writer": str(model_path),
            })
            report(identifier, mode=mode, goal=goal, source=prompt,
                   seconds=round(time.monotonic() - started, 3), result=result)
        worker._unload_writer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-writer", help="Existing local snapshot; no download or image generation")
    parser.add_argument("--goal", default="enhance", choices=pa.GOALS)
    parser.add_argument("--thinking-off", action="store_true",
                        help="The snapshot is a reasoning writer; turn reasoning off")
    args = parser.parse_args()
    if args.live_writer:
        live(args.live_writer, args.goal, args.thinking_off)
    else:
        deterministic()
