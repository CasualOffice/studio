# Board → manga production tracker

Updated: 2026-09-16. Code baseline: `dc3ebd3`. Status: planning complete;
implementation and acceptance gates below are **not yet passed**.

Companion documents:

- [Test plan and acceptance cases](board-production-test-plan.md)
- [Research, evidence limits, and experiments](board-production-research.md)
- [Historical Board defect checklist](board-pipeline-requirements.md)

## 1. Outcome and scope

Build a local, human-directed light-novel-to-manga production workspace. A user
must be able to preserve the story's essential meaning, approve adaptation
choices, direct readable pages, correct specific defects, resume work safely,
and export the approved chapter. Image generation assists that workflow; it
does not constitute editorial approval.

The existing Board is a useful illustrated-story prototype: cast extraction,
editable panel briefs, references, batched generation, redraw notes, encrypted
drafts, page composition, and Gallery PDF binding already exist. Do not rebuild
those indiscriminately. Missing or insufficiently demonstrated capabilities are
tracked below. Neither code inspection nor a few attractive sample pages proves
that users can reliably finish manga chapters.

### Delivery tiers

| Tier | Achievable deliverable | Required gates | What it does not claim |
| --- | --- | --- | --- |
| A — dependable Board | Safe edits, recoverable projects, honest completeness and text-loss reporting | G0–G1 plus BP-402/BP-601 safety checks | Manga pacing, identity fidelity, or production readiness |
| B — adaptation/name assistant | Approved chapter adaptation, canonical bible, editable rough pages and lettering plans | G0–G4 | Finished generated art or print-ready files |
| C — assisted digital chapter | Accepted art, corrections, correct lettering, current PNG/PDF export and user pilot | G0–G7 for the declared digital profile | Autonomous adaptation, arbitrary novel length, universal model/language support |
| D — production-specific delivery | Validated publisher/printer/scroll specifications and external editing handoff | Tier C plus the relevant BP-604 profile gate | Acceptance by every publisher or professional quality without editorial review |

Recommended first target: Tier A, then Tier B, then a short Tier C chapter.
Start with one declared language and reading direction; design the schema for
others, but do not advertise them until their tests pass. Whole-volume automated
adaptation, model training, cloud collaboration, animation, and a full drawing
application are outside this first release.

## 2. Evidence-based gap register

These are code observations at the baseline, not fresh model benchmarks.
Symbols are provided because line numbers will move during implementation.

| Gap | Current evidence | Production consequence | Work |
| --- | --- | --- | --- |
| Panels stand in for story analysis | `op_shotlist`, `_panel_bounds` in [worker.py](../engine/worker.py); word-derived bounds and a single panel response | No independent beat ledger, causal accounting, or reliable long-input continuation | BP-201–BP-204 |
| Coverage is lexical anchoring | `_story_coverage`, `_snap_source`; missing display capped at 20 with a separate total | An anchored panel can still depict the wrong action; omission accounting is not editorial fidelity | BP-203 |
| Identity is not a canonical registry | [Panel/Cast types](../src/lib/types.ts), `op_cast`, `castForSheet` in [Storyboard](../src/components/Storyboard.tsx) | Aliases, first-person identity, appearance decisions, and changing costumes are not consistently bound across stages | BP-202, BP-301 |
| Reference preparation lacks a hard approval boundary | `draw` generates missing sheets and proceeds into panels | A bad lineup can condition a whole batch before review | BP-302 |
| Edits can leave stale derived content | `edit` merges fields; `panelPrompt` prefers enriched `description` to `action`/`setting` | Changed action can be hidden by old enrichment; caption edits also invalidate costly art | BP-102 |
| Ownership is largely index-based | [BoardDraft v1](../src/lib/boardDraft.ts): parallel panels, drawn, redraws and notes | Structural editing and out-of-order jobs need stable ownership and revision checks | BP-101, BP-103, BP-104 |
| Geometry follows rendering | `drawOne` renders squares; `_page_plan` uses shots; `_fit` center-crops | Final panel shape can crop a face or essential action; emotional purpose is not a layout contract | BP-401, BP-403, BP-501 |
| Lettering can discard approved content | `_clean_dialogue` caps lines and characters; `_cell_text_plan` may shrink, clip, or drop text | A composed page can lose speech without an actionable failure | BP-402, BP-404 |
| Correction still regenerates a whole image | `redraw` changes the seed and passes a note; `drawOne` uses no mask | Better instructions are useful but do not guarantee preservation of correct regions | BP-503 |
| Partial composition is not final completeness | `compose` filters out missing images | An unfinished sequence can become a deceptively complete-looking comic | BP-601 |
| Export revision membership needs validation | [Gallery](../src/components/Gallery.tsx) groups all `model === "composed"` members; PDF binding is reachable there | Recomposition may mix historical page revisions unless membership is explicit | BP-601, BP-603; reproduce before claiming an observed export failure |
| No measured user-success gate | Existing unit tests and shipped examples | Passing software tests does not establish comprehension, repairability, or useful throughput | BP-003, BP-701–BP-704 |

Preserve the deliberate reference policy until experiments justify changes:
[panelReferences](../src/lib/board.ts) currently withholds room images from
character panels after observed room-object/face contamination. More references
are not automatically better, and runtime model limits differ.

## 3. Tracker rules

Statuses: `READY` = actionable with entry conditions satisfied; `TODO` = planned;
`ACTIVE` = owned and in progress; `REVIEW` = implementation exists, evidence
pending; `BLOCKED` = a named dependency/decision prevents progress; `DONE` = all
acceptance evidence linked. `DEFERRED` requires a scope decision and reason.

Every task row is independently updateable. Owner `—` means unassigned, not an
implicit commitment. Effort is relative engineering scope, not a time promise:
S = one bounded change; M = a cross-layer change; L = split into reviewed PRs.
Assign owners and estimates after G0; do not invent a delivery date now.

A task is done only when its acceptance cases pass, applicable existing checks
remain green, migration/cancellation/privacy implications are addressed, and a
reviewer links the commit, test output, and any required visual evidence.

Gates apply to a specific commit, project revision, supported model/profile, and
fixture-set version. An upstream change invalidates affected approvals. An
override cannot turn silent loss, corrupt storage, or stale final output into a
pass. Experimental proofs remain possible but visibly incomplete.

### Phase and gate dashboard

| Phase | Purpose | Entry | Exit | Status | Evidence |
| --- | --- | --- | --- | --- | --- |
| P0 | Baseline, fixtures, scope, honest claims | None | G0: reproducible baseline and frozen acceptance contract | READY | Audit above; fresh execution pending |
| P1 | Durable document and safe editing | G0 | G1: no silent loss, stale acceptance, or index drift | TODO | — |
| P2 | Source → beats → approved chapter adaptation | G1 | G2: essential story accounted for and approved | TODO | — |
| P3 | Canonical visual bible and reference approval | G2 | G3: identities/state/references approved for the chapter | TODO | — |
| P4 | Editable name: page direction and lettering plan | G2; G3 for final art handoff | G4: readable rough chapter, no unresolved text/layout errors | TODO | — |
| P5 | Controlled rendering, repair, visual acceptance | G3 + G4 | G5: every required panel has a current accepted variant | TODO | — |
| P6 | Composition, revision-safe export, delivery preflight | G5 for final; proofs earlier | G6: complete approved revision exported correctly | TODO | — |
| P7 | Real-device evaluation and user pilot | G6 | G7: declared scope meets frozen quality/effort targets | TODO | — |

Technical work on P3/P4 and on early P6 safety checks can overlap after shared
contracts exist. This does not waive the user-facing approval gates.

## 4. P0 — establish a trustworthy starting point

| ID | Change and acceptance deliverable | Priority / effort | Depends on | Tests | Status | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| BP-001 | Capture commit, runtime, installed models, exact commands and results; reproduce the stale-brief, text-loss, partial-compose and revision-membership paths | Critical / S | None | BT-001, BT-002 | READY | — |
| BP-002 | Create versioned, redistributable fixtures and human-authored expected beat/entity/speaker annotations; label unsupported-language cases | Critical / M | None | BT-003 | READY | — |
| BP-003 | Measure current end-to-end baseline, retries, active user time, latency and memory on a declared Mac/model; retain failures, not only selected pages | High / M | BP-001, BP-002 | BT-004 | TODO | — |
| BP-004 | Freeze the first release language/format/input envelope and pilot thresholds; reconcile old claims and attach a scope/decision record | Critical / S | BP-001, BP-002, BP-003 | BT-005 | TODO | — |

G0 passes when all four tasks have evidence, critical known defects have
reproduction steps, and supported versus experimental capabilities are explicit.
If real model execution is unavailable, deterministic foundations can be
developed on a documented provisional baseline, but G0 stays pending and no
performance or visual-quality claim is accepted.

Existing checks: [preflight.sh](../scripts/preflight.sh) and
[CI workflow](../.github/workflows/ci.yml). The Python suite needs CI's pinned
test dependencies; an unrelated system interpreter missing `cryptography` is
an environment failure, not proof of an engine regression. These checks do not
load production image models. Use the separate real-model lane in the test plan.

## 5. P1 — durable project contract and safe edits

| ID | Change and acceptance deliverable | Priority / effort | Depends on | Tests | Status | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| BP-101 | Introduce a versioned Board document with stable IDs, revisioned artifacts, project-owned model/format settings, and non-destructive v1 migration | Critical / L | BP-004 | BT-010, BT-011 | TODO | — |
| BP-102 | Centralize dependency invalidation; action/setting edits invalidate enrichment; lettering-only edits retain art unless its composition constraint changes | Critical / M | BP-101 | BT-012, BT-013 | TODO | — |
| BP-103 | Add insert/delete/split/merge/reorder with undo; remap by IDs, preserve sources and historical variants, explicitly resolve conflicting approvals | High / L | BP-101, BP-102 | BT-014 | TODO | — |
| BP-104 | Snapshot job inputs; reject late results against changed revisions; persist cancellation/resume state and expose save failures | Critical / M | BP-101, BP-102 | BT-015, BT-016 | TODO | — |
| BP-105 | Add explicit artifact states and approve/reject actions; show what changed and why downstream work needs review | Critical / M | BP-101, BP-102 | BT-017 | TODO | — |

### Minimum document contract

This is a design requirement, not a claim that these types already exist.
Define shared validation at the TypeScript/Rust/Python boundary before replacing
the v1 writer. Keep the old encrypted snapshot until migration is verified.

| Artifact | Required relationships and fields |
| --- | --- |
| Project | Stable ID; schema version; revision; title; language; supported format profile; model/runtime recipe; explicit chapter order |
| Source revision | Immutable original text; content hash; chapter ID; spans using one defined offset convention, including Unicode conversion rules |
| Entity | Canonical ID; aliases; source-backed facts; proposed/approved design additions; costume/prop/location states with scope |
| Beat | ID; source spans; event/intent; actors; cause/effect dependencies; story-time versus presentation order; importance; adaptation disposition and reason |
| Panel | ID; beat IDs; visible entities versus speakers; visual action; emotional purpose; shot/composition; continuity state; editable brief with input hash |
| Page/name | ID; panel IDs; geometry; reading order; page-turn/reveal intent; balloon/text objects; reserved art regions; format revision |
| Asset variant | ID; owning artifact/revision; reference IDs; complete render recipe; parent variant; status; acceptance/rejection reason; vault reference |
| Approval | Artifact ID and content/input hash; reviewer; timestamp; decision; notes; never a free-floating boolean |
| Export manifest | Exact ordered page variant IDs; source document revision; profile; preflight results; proof/final classification |

### Required invalidation behavior

| Change | Invalidate/review | Preserve |
| --- | --- | --- |
| Source edit | Affected anchors, extraction, beats, adaptation and descendants; flag uncertain remaps | Prior source/project revision and its outputs |
| Action/setting/visible actor | Brief, continuity/ref selection, panel art approval, affected pages/export | Unrelated panels; old variants |
| Caption/dialogue wording | Text measurement, speaker/reading checks, page and export approval; art only if reserved geometry must change | Approved art when it still fits |
| Costume, prop, place or design approval | Only panels depending on that state/version and relevant continuity descendants | Other chapters/states not depending on it |
| Panel order or geometry | Reading order, causal/reveal checks, reference-chain dependencies, layout; art if framing no longer fits | IDs, reusable assets, provenance |
| Model/style/reference change | Dependent render approvals and estimates | Old recipe and comparable variants |
| Note on a panel | Mark unresolved review request; redraw creates a new variant | Current image/history; no silent automatic approval |

G1: BT-010–BT-017 pass across save/reopen and mocked asynchronous jobs; no
approval survives a mismatched input hash; v1 migration is repeatable and does
not delete original work. Primary surfaces: `types.ts`, `boardDraft.ts`,
`Storyboard.tsx`, `board.ts`, `api.ts`, Rust commands/vault persistence.

## 6. P2 — adaptation is a separate editorial artifact

| ID | Change and acceptance deliverable | Priority / effort | Depends on | Tests | Status | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| BP-201 | Segment immutable source into addressable units and extract beats separately from panels; chunk with overlap, reconciliation and explicit continuation/error state | Critical / L | BP-101, BP-104 | BT-020, BT-021 | TODO | — |
| BP-202 | Resolve canonical entities, aliases, narrator and speaker identities; pass the approved registry to every planning/enrichment stage; surface uncertainty | Critical / L | BP-101, BP-201 | BT-022, BT-023 | TODO | — |
| BP-203 | Replace a single coverage percentage with source accounting, essential-beat disposition, and an independent fidelity review; require reasons for omissions/merges | Critical / M | BP-201, BP-202 | BT-024, BT-025 | TODO | — |
| BP-204 | Add chapter adaptation plan: intended emotional arc, causal spine, reveals, opening/ending, target page budget and deliberate compression; support beat-to-many-panels | High / L | BP-103, BP-203 | BT-026, BT-027 | TODO | — |
| BP-205 | Add source/beat/panel comparison and approval of invented visual decisions, internal monologue treatments, reordered scenes and omissions | High / M | BP-105, BP-204 | BT-028 | TODO | — |

G2: every human-labelled essential beat is represented or has a specifically
approved adaptation disposition; no unresolved contradiction in causality,
speaker attribution, or intended reveal order; all source units are accounted
for as assigned, context-only, deliberately omitted, or unresolved. Unresolved
essential items block approval. This is not a requirement to illustrate every
sentence, and high lexical overlap is not a fidelity pass.

Full chapters cannot silently stop at a panel or token cap. Continue safely,
ask the user to divide the source, or report an explicitly partial draft.
Chunk overlap must not duplicate beats or change canonical identities.

Primary surfaces: `op_cast`, `op_shotlist`, `_story_coverage`, cleaning and
enrichment in `worker.py`; typed API results and Board review UI. Research:
R1, R2, R5, R7 in the research companion.

## 7. P3 — approve the visual bible before spending on panels

| ID | Change and acceptance deliverable | Priority / effort | Depends on | Tests | Status | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| BP-301 | Build character/place/prop bible with source versus design provenance, aliases, explicit unknowns, and time-scoped costumes/conditions | Critical / M | BP-202, BP-205 | BT-030 | TODO | — |
| BP-302 | Separate reference creation/import from panel drawing; inspect, label in UI, accept/reject and version each identity reference; generate only needed assets | Critical / M | BP-105, BP-301 | BT-031 | TODO | — |
| BP-303 | Select references by entity/state/shot and actual model capability; make selection inspectable; test individual, lineup, room and prior-panel alternatives | High / L | BP-302 | BT-032, BT-033 | TODO | — |
| BP-304 | Replace index-based continuity assumptions with explicit state transitions and compatible-reference checks; allow intentional appearance/location changes | High / M | BP-301, BP-303 | BT-034 | TODO | — |

G3: required chapter entities and visual additions are reviewed; approved assets
are unambiguously associated with IDs; no unapproved reference is silently used
for a final render. Missing references can have an explicit text-only exception,
labelled as lower-confidence and requiring panel inspection. Passing G3 approves
the input bible, not the generated likeness of every future panel.

Do not force a four-person lineup, one sheet per character in every request,
or a universal reference-chain length. Select the supported strategy from
SP-02. Primary surfaces: `castForSheet`, `buildPlaces`, `draw`, `panelReferences`,
`previousPanel`, model catalogue capabilities, and reference UI. Research: R5–R6.

## 8. P4 — name, page direction, and lettering before final art

“Name” here means an editable rough manga chapter: panel layout, intended
actions, dialogue, reading flow, and page/scroll reveals. It need not require
image generation; boxes, thumbnails, text and imported sketches are sufficient.

| ID | Change and acceptance deliverable | Priority / effort | Depends on | Tests | Status | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| BP-401 | Plan persistent panel boxes and explicit reading order before rendering; use emotional purpose and page-turn intent; allow manual geometry and order edits | Critical / L | BP-103, BP-204 | BT-040, BT-041 | TODO | — |
| BP-402 | Preserve full approved caption/dialogue text; measure actual fonts/regions and return structured overflow; never silently slice, drop or shrink below profile minimum | Critical / M | BP-101, BP-102 | BT-042, BT-043 | TODO | — |
| BP-403 | Add safe-area/framing constraints and visible crop preview; choose supported target aspect ratios or require explicit fit approval | Critical / M | BP-401, BP-402 | BT-044 | TODO | — |
| BP-404 | Add editable balloon IDs, speaker targets, order, tail direction and protected-face/action regions; handle offscreen speech, captions, thoughts and SFX explicitly | High / L | BP-202, BP-401, BP-402 | BT-045, BT-046 | TODO | — |
| BP-405 | Provide chapter-level rough preview, single/double-page checks and scroll-specific pacing; require name approval before batch final-art generation | High / M | BP-105, BP-401, BP-403, BP-404 | BT-047 | TODO | — |

G4: the author/editor can read the complete rough chapter in the selected
direction, follow who says what, and approve pacing and reveals. All text fits
legibly without loss; protected regions are respected or manually resolved;
no unresolved critical reading-order ambiguity. The selected profile determines
dimensions and minimum type size—not a universal English word-count rule.

Early safety slice: BP-402 should land before the complete name editor if
possible, exposing current truncation as a visible blocker immediately.
Primary surfaces: composer planning/rendering in `worker.py`, Board/API page
contracts, and new layout/lettering controls. Research: R1–R5.

## 9. P5 — render to the plan and repair without losing good work

| ID | Change and acceptance deliverable | Priority / effort | Depends on | Tests | Status | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| BP-501 | Render using approved geometry, current brief, canonical entity state and supported per-model settings; persist the exact recipe and reference order | Critical / M | BP-303, BP-304, BP-403, BP-405 | BT-050, BT-051 | TODO | — |
| BP-502 | Add bounded resumable render queue, per-stage estimates, cancellation checkpoints, retry budget, and actionable OOM/model errors | Critical / M | BP-003, BP-104, BP-501 | BT-052, BT-053 | TODO | — |
| BP-503 | Offer distinct reroll, instruction-edit, masked repair and imported replacement where supported; preserve parent variants and compare/accept/revert | High / L | BP-105, BP-501 | BT-054, BT-055 | TODO | — |
| BP-504 | Review character count/identity, action, prop/state, composition, style and copy-paste repetition independently; optional automated flags never auto-certify correctness | Critical / M | BP-501, BP-503 | BT-056 | TODO | — |
| BP-505 | Run whole-sequence continuity review after individual panel checks; reopen only affected tasks and request specific corrections | High / M | BP-304, BP-504 | BT-057 | TODO | — |

G5: every required panel has one current accepted variant; reviewer sees zero
unresolved critical wrong-actor/action/state defects; all repairs retain history;
every output can be traced to an input snapshot. Judge thresholds and model
support come from SP-01–SP-04, not from similarity scores alone. It is acceptable
to import a manually repaired panel when the local renderer cannot achieve it;
record that effort instead of calling it automatic success.

Primary surfaces: `drawOne`, `redraw`, reference helpers, generation/edit API,
worker model dispatch, vault metadata and variant UI. Research: R5–R6.

## 10. P6 — proof versus final, lettering, and delivery

| ID | Change and acceptance deliverable | Priority / effort | Depends on | Tests | Status | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| BP-601 | Introduce explicit composition revision and ordered page manifest; proofs preserve/identify gaps; final composition rejects missing/stale/unapproved panels | Critical / M | BP-101, BP-105, BP-402 | BT-060, BT-061 | TODO | — |
| BP-602 | Compose accepted geometry and editable text objects deterministically; preflight missing assets, overflow, crop, order and revision freshness | Critical / M | BP-403, BP-404, BP-501, BP-601 | BT-062 | TODO | — |
| BP-603 | Make Gallery PNG/PDF export use exactly one selected approved manifest; show page order and proof/final state; test reopen and repeat composition/export | Critical / M | BP-602 | BT-063, BT-064 | TODO | — |
| BP-604 | Validate declared digital/print/scroll profiles and an external-editing package; include dimensions, color/resolution expectations, fonts/assets provenance and editable text/layout manifest | High / L | BP-603 | BT-065, BT-066 | TODO | — |

G6: final export contains precisely the current approved pages in the approved
order; text is complete and legible; preflight has zero unresolved hard errors;
the actual exported PNG/PDF is opened and inspected outside Board. PDF already
exists—this work makes revision membership and delivery guarantees reliable.

Digital PNG/PDF is the initial required BP-604 profile. Printer-specific trim,
bleed, color and resolution and webtoon platform size/chunk rules are separate
opt-in certifications; obtain their current specifications before claiming
support. An editable JSON + source-assets package is not a layered PSD/Clip
Studio document. Track native layered export separately if a pilot needs it.

## 11. P7 — prove users can finish

| ID | Change and acceptance deliverable | Priority / effort | Depends on | Tests | Status | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| BP-701 | Run the fixed corpus on declared hardware/models; publish all attempts and failure taxonomy in a local, privacy-safe report | Critical / M | BP-003, BP-505, BP-604 | BT-070 | TODO | — |
| BP-702 | Conduct a small consented pilot with source authors/adapters and independent readers; observe completion, comprehension and repair effort | Critical / M | BP-701 | BT-071, BT-072 | TODO | — |
| BP-703 | Compare against baseline; freeze supported envelope and record pass/fail/limitations for each target, without hiding manual intervention | Critical / S | BP-702 | BT-073 | TODO | — |
| BP-704 | Update onboarding, screenshots, limitation disclosures and release checklist to match measured behavior; sign off or explicitly narrow the release | High / S | BP-703 | BT-074 | TODO | — |

G7: all deterministic hard gates pass, the predeclared pilot thresholds are met,
and reviewers approve the actual exported chapters. A failed model/format is
excluded or returned to its responsible phase; an aggregate average must not
hide failure on essential beats or a supported profile. Small pilots support a
limited release decision, not a broad claim about all manga production.

## 12. Start here: ordered first implementation slices

1. **Baseline PR:** BP-001/BP-002. Add fixtures and failing/regression test
   descriptions without “fixing” expected results to current defective output.
   Complete BP-003/BP-004 evidence before signing G0.
2. **Document contract PRs:** BP-101, then BP-102. Migrate safely; reproduce
   action-edit-after-enrichment and caption-only edit cases first. Do not
   redesign all of `Storyboard.tsx` in the same change.
3. **State/approval PRs:** BP-104/BP-105, then BP-103. Demonstrate reorder during
   a late result cannot attach an image to the wrong panel.
4. **Immediate output-safety PRs:** BP-402 and BP-601. Preserve all text, report
   overflow, and make partial proofs distinguishable from final output.
5. **Adaptation PRs:** BP-201/BP-202/BP-203. Establish source and canonical IDs
   before writing a more elaborate panel prompt or name editor.

Do not start by replacing the image model, multiplying references, or adding
more prompt prose. Those are experiments after the failure is measurable.

## 13. Gate sign-off and evidence records

Update the dashboard evidence cell with a record containing:

| Field | Required value |
| --- | --- |
| Gate / date / decision | Gate ID; UTC timestamp; pass, fail, or deferred scope |
| Reviewer and owner | Named people; human quality reviewer distinct from automatic test output |
| Version | Commit; schema; fixture-set version; document/artifact revision hashes |
| Environment | Device/RAM/OS; runtime; model and quantization; render recipe; profile |
| Evidence | Commands and complete results; screenshots/pages for visual cases; local encrypted artifact references |
| Exceptions | Unsupported cases; unresolved defects; manual edits; cancelled/failed attempts |
| Next action | Responsible task IDs, owner and next review date |

Keep manuscript text, references, and generated art in the encrypted project
unless the user explicitly exports them. Commit only synthetic/publicly
redistributable fixtures and sanitized evidence. A screenshot can expose private
story content just as easily as a log.

### Change log

- 2026-09-16: Created tracker, research companion and acceptance plan against
  `dc3ebd3`. No product changes, fresh render results, or implementation gate
  passes are claimed by this documentation change.
