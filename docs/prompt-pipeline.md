# Prompt pipeline contract

> This is the intended contract, not proof the current enhancer meets it.
> The [2026-09-16 code and live-writer audit](prompt-enhancer-audit.md) found
> conflicting scene-expansion instructions, blind-edit routing, literal-text
> deletion and intent-validation gaps. Work is organized in the
> [enhancer tracker](prompt-enhancer-tracker.md) and
> [acceptance plan](prompt-enhancer-test-plan.md). For Board-specific production
> guarantees, use the [Board tracker](board-production-tracker.md).

The prompt assistant is a local editor, not a prompt-lengthener. Its job is to
preserve the user's subject and intent, remove empty quality folklore, resolve
ambiguity only when the selected model needs it, and return an explanation of
what changed. It must never silently replace a failed request with a generic
prompt.

## Inputs and outputs

- Input: the user's prompt, the mode (`generate`, `edit` or `video`), the goal
  (`enhance`, `tighten` or `style`) and the selected model. The host looks the
  model up and sends its id, family and name as `target`; the engine picks a
  template from it (`engine/prompt_assist.py`, `PROFILES`). The writer is the
  best installed one, chosen by the same `story_writer` the story analysts
  use: Qwen3.5 9B first, the Qwen3 4B as the floor.
- Output (version 2): one shape, stamped with a `version`, carrying every field
  on every reply — `prompt`, `original`, `outcome`, `changed`, `added`,
  `dropped`, `corrected`, `removed`, `attempt`, `rejected_because`,
  `saw_image`, `description`, `unclear`, `note`, and since version 2 `goal`,
  `mode`, `target_label`, `variants`, `locked`, `regions`, `target_region` and
  `released`. Each variant (faithful, enriched, stylistic) carries its prompt,
  the phrases it added (each verbatim, so the interface can leave it out), and
  any constraint it no longer satisfies.
- What must survive is extracted from the request, not taken from the writer:
  quoted text, counts, negations, colour-object bindings, positions and pace
  are hard; the subject, an action, who acts on whom, camera words and the
  writer's own list are soft. A soft miss is put to the writer once as a SAME
  question; a hard miss is retried once, cooler, and otherwise offered with
  the loss named rather than discarded.
- `removed` is on rejections too. A rejected rewrite hands back the original
  prompt, and without it there was no sign that a cleaned version had ever been
  judged — which is the one fact explaining why a rewrite was attempted.
- `dropped` is the primary variant's missing constraints, so it cannot name a
  word `removed` has already named. They are different facts and the
  interface draws them under different labels.
- If the request is too unclear to edit safely, it returns the original prompt
  with `outcome: "unclear"` and a reason.
- Empty modifiers such as “masterpiece”, “best quality”, resolution labels,
  and stock-art-site boilerplate are removed before the local writer runs —
  never inside quotes, and not when the word is the subject (“a masterpiece
  hanging in a gallery”).
- An edit names its target region when the picture reader is installed: the
  reader lists things with positions and colours, and “it” resolves to the
  most plausible one. Without the reader the reply says the picture was not
  looked at.
- The interface shows the variants as tabs against the user's own words —
  added words marked, lost ones named — with each added phrase a switch that
  rebuilds the prompt. The writer and the image model are never resident
  together, so `released` names the model that will load again on the next
  run.

## Board prompt stages

Board prompting is intentionally split because each result needs a different
kind of review:

1. **Read** extracts only character names found in the manuscript.
2. **Plan** divides the prose and must attach an exact source passage and the
   visible character names to every panel.
3. **Prepare** expands each accepted panel into a self-contained visual brief
   while preserving its source, action, shot, cast, place, and dialogue.
4. **Draw** deterministically combines the brief, style, cast/place references,
   and the user's per-panel correction. It does not reinterpret the whole story.
5. **Compose** lays completed panels out in story order and stores pages in the
   same Vault project as their panels and reference sheets.

Coverage, character membership, and source anchoring are validated in code;
they are not entrusted to model arithmetic. Editing the manuscript invalidates
every downstream stage.

Editing one panel invalidates what that change actually reaches, decided in one
place — `staleness` in [boardDoc.ts](../src/lib/boardDoc.ts). A picture is a
drawing of an action, in a place, of these people: changing one of those
retires the picture, and changing the caption or the dialogue does not. Every
edit retires the composed pages. A retired picture goes to the panel's history
rather than being abandoned in the vault, and a worked-up description is
cleared when the fields it was written from change, because the prompt prefers
it to those fields.

Panels carry ids and their work is stored against them, so a panel can be
moved, split, joined, inserted or removed without its picture and its notes
staying behind for whatever lands in that position.
