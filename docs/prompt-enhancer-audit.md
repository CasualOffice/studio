# Prompt enhancer: implementation and live-behavior audit

Audit date: 2026-09-16 (Asia/Kolkata). Product baseline: `f5bf928`; enhancer code
unchanged from `dc3ebd3`. Scope: Studio image/edit assistance, Video assistance,
shared proposal UI, Rust request contract, Python routing/validation/lifecycle,
existing tests, official model guidance, and a bounded local writer trial.

Companions: [implementation tracker](prompt-enhancer-tracker.md),
[acceptance plan](prompt-enhancer-test-plan.md),
[reproducible diagnostic script](../scripts/audit_prompt_enhancer.py).

## Verdict

The feature is not reliably achieving its advertised job of making a request
precise without changing its meaning. This is not just weak prose or a small
model problem. Its contract, routing, validation and presentation disagree.
Actual local rewrites reproduced meaningful failures before an image model was
involved. A bigger writer alone would not fix text deleted before inference,
wrong task routing, stale proposals, or validators that ignore relationships.

There is a useful foundation: local inference, proposal-before-acceptance, undo,
some spelling repair, mode-specific system prompts, optional visual observations,
bounded retry, staged-image cleanup, and explicit rejection messages. Preserve
those capabilities while replacing the conflicting decision logic.

Achievable target: a task-aware editor that protects explicit constraints and
offers creative additions separately. It can improve clarity and help express
model-relevant detail. It cannot guarantee a correct generated image just because
the rewritten text looks good, or infer a user's unspoken design preference.

## 1. What actually happens

| Stage | Implemented behavior | Important mismatch |
| --- | --- | --- |
| Request | Studio/Video send prompt, mode, image IDs | No target image/video model, edit kind/mask, reference roles, negative prompt, duration, language, or creativity preference |
| Host | Rust requires installed Qwen3 writer; supplies optional Qwen2-VL reader and image keys | Checks reader availability, not whether edit context is sufficient |
| Cleaning | Global phrase removal before writing or validation | Literal lettering, tool names and meaningful nouns can disappear |
| Routing | Image-aware edit only if edit mode + staged image + reader; otherwise video or generic image direction | Text-only/blind edits fall into scene invention |
| Writing | Image direction asks for free rephrasing, mandatory design choices and 40–80 words; video forces gradual motion | Conflicts with “Adds no objects” UI and conservative-editor document |
| Checking | Word stems/prefixes, first content word, retained-word ratio, article count; optional same-writer SAME/DIFFERENT | No reliable count, polarity, actor-role or attribute-binding contract |
| Recovery | Semantic override, append missing words, lower-temperature retry, final lexical check | Good paraphrases can be vetoed later; bad meaning can pass early |
| Proposal | Full proposed text plus word chips; manual accept/undo | No input-revision binding; successful blind-mode note is not rendered |

Sources: [worker.py](../engine/worker.py) `op_assist`, `_clean_prompt_request`,
`_clarified_with_reason`; [commands.rs](../src-tauri/src/commands.rs)
`assist_prompt`; [Studio](../src/components/Studio.tsx) and
[Video](../src/components/Video.tsx) `improve`;
[PromptProposal](../src/components/PromptProposal.tsx).

## 2. Evidence and execution boundaries

Three evidence classes are kept distinct:

- **Live:** actual installed Qwen3 text inference through `op_assist`, 12
  synthetic cases, no vision reader or image/video generation.
- **Reproduced:** production pure logic or request routing with controlled
  inference responses. These prove paths/acceptance behavior, not how often a
  model produces the injected response.
- **Inspected:** code-path finding needing interactive/real-vision validation.

Existing suites, executed during this audit:

| Check | Result | Qualification |
| --- | --- | --- |
| App Python: `-m unittest discover -s engine` | 282 tests passed in 7.123 s | ResourceWarnings occurred; sandboxed MLX emitted an atexit “No Metal device” warning despite exit 0 |
| `npm test -- --run` | 167 tests passed, 6 files | No dedicated PromptProposal/assist workflow tests found among current frontend tests |
| `python3 scripts/audit_prompt_enhancer.py` | 18 diagnostic observations produced | Reporting program; exit 0 is not a quality pass. Includes one positive control |
| Installed local writer trial | 12 completed calls: 11 proposals, 1 unclear | Illustrative sample, not an estimated success rate or image-quality benchmark |

Metal inference required execution outside the sandbox. It used the existing
model snapshot, synthetic prompts only, offline library flags, and no vault
access. Environment: Apple M4, 16 GiB, `Mac16,12`, macOS 26.3; Python 3.12.14;
MLX 0.31.2, mlx-lm 0.31.3, mlx-vlm 0.7.0. Writer:
`mlx-community/Qwen3-4B-Instruct-2507-4bit`, snapshot
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`; seeds 100–111 in case order.
The production code selects temperature/token limits; the audit did not improve
the system prompts or cherry-pick retries outside that code.

## 3. Live trial: the actual outputs matter

Full case inputs are in the diagnostic script. Excerpts below are from this
run, not hypothetical examples. Timings surround `op_assist`, exclude process
startup and imports performed before that call, and are not end-to-end UI or
image-generation timings. The first call also imports the vision library inside
`op_assist` even though this trial uses only the text writer; its time is not an
isolated model-load benchmark.

| Case | User request | Observed outcome | Seconds |
| --- | --- | --- | --- |
| PL-01 | `a cat on a chair` | Reproduced the prompt's own example: tabby, worn oak chair, afternoon window light, oil painting. Subject kept; substantial unrequested art direction | 23.606, first call/load |
| PL-02 | `2 red cars with no sunroof, plain white background` | Preserved count, exclusion and white background; added identical sedans, matte wear, fogged windows and parking context | 9.953 |
| PL-03 | `a red cube left of a blue sphere` | Kept relation/colors; added concrete surface/wall, worn corner, photography; contradictory “low angle, slightly above eye level” wording | 7.075 |
| PL-04 | Sign reading exactly `BEST QUALITY 4K` | Cleaner removed the lettering; writer produced a weathered sign with “letters between two blank spaces.” Still offered as a proposal | 8.607 |
| PL-05 | Mira gives Jonas a brass key; Mira wears red, Jonas blue | Kept handoff but invented ages, appearances and hallway; Mira's red clothing disappeared. `dropped` listed red/blue; spelling UI data even labelled `wears → warm` | 9.059 |
| PL-06 | `a hous at nite` | Corrected spelling; closely followed the system's brick-house/wet-street example | 6.649 |
| PL-07 | Black ink cat drawing, white background, no shading/color | Preserved core style/exclusions; added pose, paper and pen details | 7.632 |
| PL-08 | `make it better` without an image | Correctly returned unclear and kept original | 4.036 |
| PL-09 | Edit: `make the jacket red`, no reader/image | Invented a man, alley, worn leather jacket, dumpster and cracked paper cup instead of an edit instruction | 8.851 |
| PL-10 | Edit: remove left person, keep right person unchanged; no reader/image | Invented the survivor's clothes, pose, street, dusk and camera. Offered a scene description without evidence of the existing image | 14.191 |
| PL-11 | Runner sprints quickly; locked camera | Added “slow, grounded drift” and “camera locked…following the runner's path”; dropped `quickly` | 3.589 |
| PL-12 | Dog looking like a king | Produced a coherent royal-dog direction, but chose breed, throne room and oil-painting style without approval | 10.465 |

Warm-call median: 8.607 s, range 3.589–14.191 s over the remaining 11 calls.
The old approximate two-second table is not a reliable expectation for this
current call path. Do not generalize these times across machines or include
reader/generator reload cost that was not measured here.

PL-01 is a few-shot example repeated by the model, so it is not held-out evidence
of creative quality. PL-06 is also very close to an example. PL-09/10 deliberately
test the no-image edit branch; PD-11 independently reproduces that same wrong
route when an image is present but the reader is absent. Real visual grounding
quality remains unmeasured.

## 4. Findings ordered by impact

### AF-01 — the promised product and the system instruction conflict

**High; inspected + live.** `SCENE_DIRECTION_SYSTEM` tells the writer to choose
age/materials, appearance, setting, framing, light and style. Its examples add
objects and mediums. Studio's tooltip says it adds no unrequested objects;
[prompt-pipeline.md](prompt-pipeline.md) calls it a local editor, not a lengthener.
The model is often following the expansive instruction, not malfunctioning.

Change: make “clarify my request” conservative by default, with separately
approved art direction. Unspecified is a valid state; no compulsory camera or
lighting for icons, typography, diagrams, flat illustration or exact designs.
Track PE-101/PE-102.

### AF-02 — blind/text-only edit is routed to scene generation

**Critical for task fidelity; PD-11, PL-09/10.** `editing` requires both an image
and the installed reader. The `else` selects image direction for any non-video
mode, including an edit with no visual reader. An edit can therefore become a
new scene before it reaches the editor model.

Change: task routing must remain edit routing regardless of reader availability.
Keep explicit edits usable text-only; ask about ambiguous targets or expose a
blind-mode limitation. Never infer a nonexistent scene to fill the missing image.
Track PE-103, PE-301.

### AF-03 — cleanup destroys literal meaning before validation

**Critical; PD-07–PD-09, PL-04.** Global removal strips quoted sign text, the noun
“masterpiece,” and “Unreal Engine” when it names an interface. Validation compares
against the already-cleaned request, so it cannot protect the lost content.

Change: preserve original spans, quoted strings, names, exact text, numbers and
meaningful style/tool terms. Cleanup suggestions are contextual and reversible;
never unconditional deletion. Track PE-201.

### AF-04 — word retention is not intent preservation

**Critical; PD-01–PD-06, PL-05/11.** `_significant` drops short tokens including
`no` and single-digit counts. `_NOT_A_SUBJECT` includes `walking` and media words.
Word sets cannot bind color to object or actor to recipient. `left` can be lost
as an accepted “rephrasing.” `_keeps_intent` needs only one approximate content
match; its name/comment promises more than it proves.

Reproduced accepted mutations: no sunroof → sunroof; 2 → 3 cars; left → right;
Mira/recipient reversal; red/blue swapped between objects; walking → sitting.
Rejecting completely unrelated “hat on chair” remains a useful positive control.

Change: explicit constraint records and per-constraint verdicts for polarity,
counts, bindings, actions, text, style and preservation scope. Keep deterministic
checks for exact constraints; use semantic comparison for paraphrase with honest
uncertainty. Do not call a same-writer judgement ground truth. Track PE-202/203.

### AF-05 — acceptance decisions contradict each other

**High; PD-12/13.** `_clarified_with_reason` accepts `rephrased`, but the edit
caller accepts only `ok`/`composed`. A semantically reasonable jacket → coat
rewrite is presented as rejected. A semantic SAME override can be overturned by
the final `_keeps_intent` check on the same text. Appending missing words is also
not a grammatical or semantic repair contract.

Change: one typed decision pipeline with explicit warning/violation semantics;
bounded targeted repair, then full validation against the original constraints.
Track PE-203/204.

### AF-06 — the enhancer cannot adapt to the actual rendering operation

**High; inspected.** The API does not pass selected model, edit subtype, mask,
reference roles, target aspect ratio, negative prompt, clip duration or FPS.
Video instructions assume a second or two and prohibit fast motion even when the
user asks for sprinting. An outpaint instruction and a masked replacement need
different context from a complete scene description.

Change: versioned capability/task context from the selected runtime, not a
universal prose recipe. Preserve requested motion; explain incompatible duration
or backend limits instead of silently slowing it. Track PE-101/301/302/303.

### AF-07 — visual observations are too coarse to settle many edit targets

**High; inspected, real vision evaluation pending.** `SCENE_SYSTEM` asks for one
main subject plus surface/light/setting, without the user's target question.
All inputs are reduced to at most a 512-pixel edge. Multi-person left/right,
small logos, text and masked regions need target-specific grounding. Multiple
images have no declared source/reference role in the assistant contract.

Change: target-aware observation and region/reference IDs; distinguish observed,
uncertain and user-specified facts. A vague “make it blue” with multiple plausible
objects should not always resolve to the main subject. Track PE-301.

### AF-08 — parsing can silently discard candidate instructions

**High; PD-10.** `_first_line` and `_clarified_with_reason` take only the first
usable line. A second-line constraint disappears. `_write` returns text without
a completion/truncation status. The response's length limit is a heuristic, not
a check that the full result was received.

Change: structured validated output with explicit finish status; preserve text
or visibly reject incomplete/malformed output. Track PE-204.

### AF-09 — cancellation is incomplete across UI and engine

**High; PD-14/15 plus inspection.** `_clarify_edit` catches `Cancelled` as a
generic exception and returns a composed result; `_same_request` turns it into
False. The writer streams cancellation checkpoints, but vision uses blocking
`generate`. Studio/Video assistance never sets the visible `jobId`, and their
Cancel controls are shown for generation, not assistance.

Change: a dedicated assistance job lifecycle; Cancelled propagates; no fallback,
retry or proposal after cancellation; loading/vision cancellation capabilities
reported accurately. Track PE-401/402.

### AF-10 — the one-helper-resident policy is not enforced between helpers

**High; PD-16, actual memory peak not measured.** Both loaders release the image
generator, but `_load_writer` does not unload the vision reader and vice versa.
An image-aware request can leave both helper slots resident. The existing policy
test enumerates generation/upscale loaders, not the helper handoff.

Change: sequential reader→facts→reader release→writer under a shared resource
policy; verify object release and actual peak memory. Do not deliberately force
an OOM on the user's machine to prove a lifecycle bug. Track PE-402.

### AF-11 — proposals are not bound to current input context

**High; inspected, interactive reproduction pending.** Text remains editable
during assistance. `onChange` clears the current proposal, but a late response
calls `setProposal` with no prompt/revision comparison. Image/model/edit-kind
changes are not bound to a proposal either. The Generate button also remains
available during assistance. Undo is useful but is not stale-result prevention.

Change: snapshot/hash prompt, images, target model and operation; discard or mark
stale results; coordinate generation and helper jobs. Track PE-401.

### AF-12 — the explanation can hide the information most needed to decide

**High; inspected + live metadata.** Successful `PromptProposal` does not render
`result.note`, hiding the missing-reader warning. Added words show only the first
14; a late-added oil-painting style can be absent from the chips. Short tokens
and relationships are not meaningfully diffed. The full proposed text remains
visible, which mitigates but does not solve this. PL-05 reports `wears → warm`
as a spelling correction; this is a vocabulary-matching guess, not an actual
edit-span alignment. `saw_image` can be true without vision execution (PD-17,
an API-level image supplied to generate mode, not the current Studio call).

Change: side-by-side original/candidate, semantic changes and exact-text diff;
always visible context/capability warnings; uncertain corrections not asserted
as facts. Track PE-205/403.

### AF-13 — tests protect some historical implementation choices, not the outcome

**High; inspected + new diagnostics.** Existing tests exercise good/bad string
pairs and sometimes assert source text contains an instruction. One test demands
free art direction and explicitly rejects a conservative no-new-objects rule.
Those tests can remain green while routing, semantics and UI are wrong. The
contradictory product contracts must be resolved before deciding which old
expectations to change.

Change: executable end-to-end stubbed assistant tests, UI deferred-result tests,
counterfactual constraint fixtures, real writer trials and paired image/video
evaluation. Do not merely loosen thresholds until the old tests pass.
Track PE-001/002/501/502.

## 5. Research implications, not prompt folklore

All sources below were checked during the audit. Author documentation supports
model-specific practices; it does not establish success for this quantized local
backend. No hosted service is needed for the proposed fix.

| ID | Primary source | Bounded implication |
| --- | --- | --- |
| PR-01 | [BFL single-reference editing](https://docs.bfl.ai/guides/prompting_editing_single_reference) | Separate the requested change from what should stay; simple targeted edits are legitimate. Does not promise exact pixel preservation in this app |
| PR-02 | [BFL FLUX.2 pro/max prompting guide](https://docs.bfl.ai/guides/prompting_guide_flux2) | Guidance varies by complexity and includes short prompts, structured descriptions and exact typography. This pro/max guide is not a Klein benchmark or license for compulsory 40–80-word expansion |
| PR-03 | [BFL Klein 4B text-encoder configuration](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/text_encoder/config.json) | Declares `Qwen3ForCausalLM`; the worker's blanket “these models use T5” rationale is incorrect for this family. Encoder architecture alone does not prove optimal prompt length |
| PR-04 | [Wan2.2 official repository](https://github.com/Wan-Video/Wan2.2) | Distinguishes text/image-conditioned video and optional prompt extension, including local approaches. Preserve motion/duration context rather than treating every request as slow ambient motion |
| PR-05 | [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | The writer is a non-thinking causal language model. General instruction-following claims do not certify strict visual constraint preservation or local quantized performance |
| PR-06 | [GenEval, official research implementation](https://github.com/djghosh13/geneval) | Evaluates object co-occurrence, position, count, color and attribute binding separately. Useful evaluation dimensions, not a complete edit/video/typography oracle or automatic quality pass |

The formerly indexed Klein-specific guide URL returned 404 in this audit. Do
not cite it as if its current contents were verified; use the checked model
configuration and validate prompting choices locally.

### Recommended experiments

| ID | Comparison | Evidence required | Stop/decision rule |
| --- | --- | --- | --- |
| PX-01 | Original vs conservative rewrite vs opt-in direction | Same fixture/model/settings; independent hard-constraint rubric; all attempts, clarity and user preference | Prefer clearer text only if explicit intent is retained; no mandatory expansion target |
| PX-02 | Current word checks vs structured constraint checks + optional semantic adjudication | Minimal pairs covering negation/count/roles/bindings/text, plus legitimate paraphrases | No known explicit-constraint loss called safe; uncertain semantics goes to review, not automatic sameness |
| PX-03 | Main-subject caption vs targeted region/reference observation | Synthetic/consented multi-object images, small text, mask, offscreen ambiguity; human target labels | If target cannot be resolved, ask/retain instruction; do not invent certainty |
| PX-04 | Helper lifecycle and end-to-end latency | Cold/warm writer, reader→writer, subsequent generator reload, cancel at each stage; RAM and timings | Bound resource usage and disclose full workflow cost, not only warm decoding |
| PX-05 | Rewrite-text quality vs generated-output quality | Paired original/enhanced images and clips, multiple seeds, blinded scoring of requested constraints and unintended edits | Ship only supported model/task profiles demonstrating useful gain without unacceptable intent loss |

## 6. What to fix first

1. Separate conservative clarification from optional art direction, and carry
   target task/context explicitly.
2. Correct blind edit routing and preserve quoted text/numbers/exclusions before
   adding any more model instructions.
3. Introduce constraint-aware decisions with a single validation/repair path.
4. Fix input snapshotting, cancellation, helper handoff and visible blind-mode
   warnings; preserve proposal/undo behavior.
5. Evaluate real outputs against originals. Do not equate a longer prompt,
   accepted proposal, or passing unit suite with a better image.

No product implementation was changed by this audit. The companion tracker
turns these findings into gated work; live writer observations are baseline
evidence, not acceptance of the current enhancer.
