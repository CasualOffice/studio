# Prompt enhancer implementation tracker

Updated: 2026-09-16. Baseline: `f5bf928`. Parent evidence:
[audit and research](prompt-enhancer-audit.md). Verification:
[acceptance plan](prompt-enhancer-test-plan.md).

Implementation status: **not started**. The audit and baseline diagnostics exist;
all product/release gates remain pending. This work changes prompt assistance,
not Board adaptation or the image/video generator itself.

## Product contract to ratify

Default **Clarify**: express the user's request more clearly without changing
explicit subject, action, count, relationships, exclusions, exact text, medium,
or edit-preservation scope. Correct unambiguous typos; ask or label uncertainty
where interpretation changes the picture. Concise already-good input may stay
unchanged, with an honest explanation.

Optional **Art direction**: suggest additional setting, materials, lighting,
composition or style as visible choices. Do not silently choose age, costume,
brand, scene, medium or camera just because the user omitted it. Explicit
constraints remain binding in both modes. Task (`image`, `edit`, `video`) and
transformation preference (`clarify`, `direct`) are distinct dimensions.

An edit stays an edit even without a reader; a video keeps its intended action.
An accepted textual proposal is not a guarantee of a correct rendered result.

## Status and ownership rules

`READY`: entry conditions satisfied. `TODO`: planned dependency remains.
`ACTIVE`: named owner working. `REVIEW`: implementation exists, evidence pending.
`BLOCKED`: named unresolved dependency. `DONE`: acceptance evidence linked.
`DEFERRED`: explicit scoped decision with a reason. Owner `—` is unassigned.
S/M/L are relative scope, not calendar estimates. Split L tasks into reviewed
PRs; assign owners/dates before activating them.

Every completed task needs: commit/PR, test IDs and result, before/after evidence,
privacy/cancel/resource review where applicable, and reviewer. Record these in
the Evidence column. Do not mark a task done just because its prompt was edited.

## Phase dashboard

| Phase | Purpose | Entry | Exit gate | Status | Evidence |
| --- | --- | --- | --- | --- | --- |
| E0 | Freeze baseline, supported scope and acceptance rubric | None | EG0: reproducible baseline + approved contract | READY | Audit, 18 diagnostic observations, 12 live writer cases; approval pending |
| E1 | Task/context contract and transformation routing | EG0 | EG1: conservative/default behavior and every route defined | TODO | — |
| E2 | Protected intent, output parsing and consistent decisions | EG1 | EG2: hard-constraint and legitimate-paraphrase suite passes | TODO | — |
| E3 | Grounded edits, motion, and model-aware compilation | EG1; EG2 for acceptance | EG3: no task/context substitutions; supported profiles validated | TODO | — |
| E4 | Proposal UX, lifecycle, cancellation and resource behavior | EG1; EG2/EG3 for final acceptance | EG4: current-context proposals only; correct cancel/cleanup/resource behavior | TODO | — |
| E5 | Real-output experiments, pilot and honest release | EG2 + EG3 + EG4 | EG5: measured usefulness on declared tasks/models | TODO | — |

E2/E3/E4 may be developed in separate small changes once E1 defines interfaces.
This does not authorize skipping acceptance gates or publishing an untested
model/task profile.

## E0 — baseline and scope

| ID | Deliverable / acceptance | Priority / scope | Dependencies | Tests | Status | Owner | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PE-001 | Convert audit cases into versioned fixtures with independent expected constraints; identify useful old tests versus policy assertions needing review | Critical / M | None | PT-001, PT-002 | READY | — | Baseline diagnostic script exists; acceptance fixtures pending |
| PE-002 | Ratify Clarify/Art direction contract, initial language/model/task envelope, retry/latency targets and predeclared evaluation rubric | Critical / S | PE-001 | PT-003 | TODO | — | Proposed contract above |

EG0: original prompts and human expectations are frozen; the initial 12 examples
are baseline evidence, not the held-out evaluation set. Decide whether optional
direction ships now or later; conservative edit routing and literal preservation
are not optional product-quality work.

## E1 — typed task context and honest routing

| ID | Deliverable / acceptance | Priority / scope | Dependencies | Tests | Status | Owner | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PE-101 | Version request/result schemas across TS/Rust/Python; carry source snapshot, task, transformation, actual selected model/capabilities and operation context | Critical / M | PE-002 | PT-010 | TODO | — | — |
| PE-102 | Implement conservative default and explicit optional direction; remove forced length/detail fields and distinguish unchanged from rejected/uncertain | Critical / M | PE-101 | PT-011, PT-012 | TODO | — | — |
| PE-103 | Route edit/video by task independently of reader presence; explicit handling for unsupported/unknown modes and missing context | Critical / S | PE-101 | PT-013, PT-014 | TODO | — | — |

Minimum request contract: request ID, source text/hash, language, task, operation
(text-to-image/instruction-edit/mask/outpaint/latent/T2V/I2V), transformation,
target model/profile version, reference IDs/roles, mask/region metadata where
relevant, dimensions, video duration/frame/FPS context, negative constraints and
user locks. Pass only needed metadata; never duplicate vault keys into reports.

Minimum result contract: same request/input hash, original, complete candidate,
typed outcome, protected-constraint verdicts, proposed additions, exact edits,
observed/uncertain image facts, capability warnings, actual vision-use status,
finish/truncation state and bounded-attempt metadata. Unavailable context must
not be fabricated.

EG1: every supported request takes the intended path in cross-layer tests; no
text-only edit uses image scene direction. UI label and instructions describe
the same transformation. Unsupported operations fail visibly or keep the input.

Primary surfaces: `AssistResult`/API, Rust `assist_prompt`, worker `op_assist`,
Studio/Video request construction. Findings: AF-01/02/06. Research: PR-01–PR-05.

## E2 — preserve meaning before polishing wording

| ID | Deliverable / acceptance | Priority / scope | Dependencies | Tests | Status | Owner | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PE-201 | Replace unconditional phrase deletion with span-aware optional cleanup; protect literal strings, named entities, counts, negation and meaningful style terms | Critical / M | PE-101 | PT-020, PT-021 | TODO | — | — |
| PE-202 | Extract explicit constraints with source spans and bindings; preserve original truth even if normalization/rephrasing changes words | Critical / L | PE-101, PE-201 | PT-022, PT-023, PT-024 | TODO | — | — |
| PE-203 | One consistent constraint verdict/decision path; semantic paraphrase support cannot waive exact constraints or be undone by an unrelated lexical veto | Critical / L | PE-202 | PT-025, PT-026 | TODO | — | — |
| PE-204 | Structured complete-response parsing, stop-status handling and bounded targeted repair; revalidate against original after each repair | Critical / M | PE-203 | PT-027, PT-028 | TODO | — | — |
| PE-205 | Span-aligned typo/diff explanations; numbers and negation visible; uncertain proper-name/homophone changes offered as choices | High / M | PE-201, PE-203 | PT-029 | TODO | — | — |

Constraint examples: actor → action → recipient, object → color/material,
count → entity set, spatial relation with ordered arguments, exact lettering,
negative clauses, style and camera locks, edit target and protected regions.
Do not infer that a regex/NLP/LLM extractor is infallible; uncertainty is an
explicit result, and source text stays visible.

Outcome semantics: `proposal` must not conceal an identified hard violation.
Use a blocked candidate with specific reasons or a clarification request.
Manual override, if retained, is visibly an override and never rewrites the
record as “intent preserved.” “No change needed” is not a generic fallback for
parse failure, cancellation, missing reader or rejected generation.

EG2: no known mutation of literal text/count/polarity/binding in the frozen suite
is certified as safe; legitimate paraphrases have consistent outcomes; no
second-line loss, unreported truncation or appended-word pseudo-repair. Readiness
is fixture-bounded, not a guarantee that all natural-language intent is decidable.

Primary surfaces: cleaning, significance/spelling, validation, `_same_request`,
`_clarify_edit`, `_write` result handling. Findings: AF-03/04/05/08/12. PR-06.

## E3 — use the right information for the actual task

| ID | Deliverable / acceptance | Priority / scope | Dependencies | Tests | Status | Owner | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PE-301 | Target-aware image observations, region/reference roles and explicit ambiguity; text-only edit fallback keeps change/preservation scope | Critical / L | PE-103, PE-203 | PT-030, PT-031 | TODO | — | — |
| PE-302 | Video-specific motion/camera/temporal constraints using actual duration/profile; no silent slow-motion substitution or image invention | High / M | PE-103, PE-203 | PT-032, PT-033 | TODO | — | — |
| PE-303 | Profile-based prompt compilation for supported image/edit/video backends; style-neutral default; reference/negative-prompt limitations surfaced | High / M | PE-101, PE-203, PE-301, PE-302 | PT-034 | TODO | — | — |

EG3: target changes are distinguished from preserved content; unsupported masks,
reference combinations and runtime controls are never silently claimed as used.
Image observations answer the edit target question, not just “what is the main
subject?” Missing or uncertain target identity prompts a choice. Motion/camera
locks remain intact in the rendered prompt. Validate each declared language and
profile; do not apply English token heuristics to all languages by default.

Primary surfaces: `SCENE_SYSTEM`, edit clarification, motion system, request
context and profile selection. Findings: AF-02/06/07. Experiments PX-03/PX-05.

## E4 — proposal trust and lifecycle

| ID | Deliverable / acceptance | Priority / scope | Dependencies | Tests | Status | Owner | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PE-401 | Dedicated assist job ID and input snapshot; stale response checks on text/model/image/mode changes; listener cleanup; generation scheduling and usable cancel | Critical / M | PE-101 | PT-040, PT-041 | TODO | — | — |
| PE-402 | Propagate cancellation, release helper models sequentially, bound retries/load behavior, retain correct temporary-file ownership and sanitize diagnostics | Critical / M | PE-101, PE-204 | PT-042, PT-043, PT-044 | TODO | — | — |
| PE-403 | Side-by-side original/proposal and constraint changes, always-visible reader warnings, honest outcome wording; accept/keep/undo with revision ownership | High / M | PE-205, PE-301, PE-401 | PT-045, PT-046 | TODO | — | — |
| PE-404 | Measure and explain cold/warm read/write/reload costs, progress and cancellation status; recover from missing models without labelling user input defective | High / M | PE-401, PE-402 | PT-047 | TODO | — | — |

EG4: no late proposal replaces newer input; cancel produces no successful
fallback/proposal; helper handoffs obey the measured memory policy; warning text
is visible wherever the result is actionable. The user can inspect every
substantive change and return to the precise source revision they accepted from.

Privacy: assistance must not persist raw prompts or observed image descriptions
in routine disk diagnostics. Current structured `log()` events and stderr have
different paths; do not falsely claim all current log events are written to
disk. Test actual sinks and exported diagnostics, with synthetic data only.
Temporary files must be private and cleanup must never delete caller-owned
files, including a pre-existing filename resembling a derived thumbnail.

Primary surfaces: Studio/Video `improve`, PromptProposal, worker helper loaders
and cleanup, host events. Findings: AF-09–AF-12. Experiment PX-04.

## E5 — prove benefit beyond nicer prose

| ID | Deliverable / acceptance | Priority / scope | Dependencies | Tests | Status | Owner | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PE-501 | Run held-out writer corpus and original/enhanced paired image/edit/video experiments; save all attempts and separate hard fidelity from aesthetics | Critical / L | PE-204, PE-303, PE-403, PE-404 | PT-050, PT-051 | TODO | — | — |
| PE-502 | Small consented user pilot: correctly scoped task, inspect changes, accept/reject/undo, achieve output; record assistance and manual correction effort | High / M | PE-501 | PT-052 | TODO | — | — |
| PE-503 | Ratify supported profiles and limitations; update tooltip/docs/timings/tests based on outcomes; link gate sign-off and release decision | Critical / S | PE-502 | PT-053 | TODO | — | — |

EG5: frozen hard-constraint suite passes; supported profiles show useful output
benefit without unacceptable intent loss; pilot users understand what changes
and what was not observed. Report failures and manual overrides separately;
proposal acceptance rate alone is not success. A profile that fails is excluded
or returned to its responsible phase, not hidden in an aggregate score.

## First implementation queue

1. PE-001/PE-002: freeze policy and the failure fixtures; preserve live baseline.
2. PE-101/PE-103: minimal typed context and correct blind-edit routing.
3. PE-201: protect exact lettering, names, numbers and exclusions before cleanup.
4. PE-202/PE-203: replace contradictory word-based acceptance with explicit
   constraint verdicts; maintain useful paraphrase acceptance.
5. PE-401/PE-402/PE-403: stale/cancel/resource/visibility fixes alongside targeted
   inference tests. Then expand model-aware/grounded capabilities and evaluate.

Do not solve the problem by increasing token limits, relaxing retention checks,
or changing to a larger model before these contracts and fixtures exist.

## Gate sign-off template

Record gate/date, commit, named owner/reviewer, scope/profile version, fixture
version, commands/results, actual input/output examples, device/runtime/model,
unresolved failures/overrides, evidence locations and next task. Store synthetic
fixtures in the repository; private prompts/images remain local unless exported
with the user's approval. All gate sign-offs are currently pending.

### Change log

- 2026-09-16: Created from implementation audit, 18 deterministic observations,
  12 installed-writer trials and primary-source review. No production code
  changes or passed implementation gates are claimed.
