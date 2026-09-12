# Prompt pipeline contract

The prompt assistant is a local editor, not a prompt-lengthener. Its job is to
preserve the user's subject and intent, remove empty quality folklore, resolve
ambiguity only when the selected model needs it, and return an explanation of
what changed. It must never silently replace a failed request with a generic
prompt.

## Inputs and outputs

- Input: the user's prompt, task (`image`, `edit`, or `video`), and optionally
  the selected model's constraints.
- Output: `prompt`, `changed`, and `removed`. If the request is too unclear to
  edit safely, it returns the original prompt with a rejection reason.
- Empty modifiers such as “masterpiece”, “best quality”, resolution labels,
  and stock-art-site boilerplate are removed before the local writer runs.
- The interface shows whether the prompt changed and exactly which phrases
  were removed; the user remains free to edit the result.

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
every downstream stage. Editing one panel invalidates only its existing image
and composed pages.
