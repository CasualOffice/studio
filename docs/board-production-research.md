# Board production research and decision register

Checked: 2026-09-16. Scope: source adaptation, actual manga/webtoon production
practice, and controllable story-image generation. Parent:
[production tracker](board-production-tracker.md). Acceptance:
[test plan](board-production-test-plan.md).

This document distinguishes published evidence, repository observations, and
proposed product decisions. A paper's benchmark is not a production acceptance
test for this app. A publisher's workflow is evidence of practice, not a rule
that every manga team follows. Research informs the tracker; local experiments
and human approvals determine whether the implementation works.

## 1. Primary sources and their bounded implications

| ID | Source and provenance | What it supports | Proposed application / limitation |
| --- | --- | --- | --- |
| R1 | [Cosmic Publishing: what editors value in adaptation](https://note.com/comic_cosmic/n/n8618665550da), editorial team, 2026-06-24; Japanese | Their author-review stages include overall plot, character roughs, name and first proof; prose-to-manga involves visual and editorial choices | BP-205/BP-302/BP-405 and final proof approval. One publisher's process, not universal turnaround times or automation requirements |
| R2 | [Cosmic Publishing: why request an overall adaptation plot](https://note.com/comic_cosmic/n/nfc2a9f2d9e84), 2026-06-16; Japanese | Important scenes, episode openings/endings, volume breaks and emotional sequence need deliberate adaptation planning | BP-204: chapter/arc intent precedes panel drafting. No supported fixed ratio of novel words to panels |
| R3 | [Kodansha: Noragami work in progress](https://archive.kodansha.us/2015/12/30/noragami-work-in-progress/index.html), editor Yohei Takami's process, 2015-12-30 | Character conception, name review for clarity/pacing/dialogue, pencils/typesetting, and final checks are distinct activities | BP-401/BP-405: an editable rough chapter must be readable before expensive art. This is manga production, not specifically a light-novel adaptation trial |
| R4 | [comico: behind name production, creator interview](https://note.com/comico_jp/n/neebf63ff26d1), official editorial account, 2023-10-16; Japanese | Source/scenario, name, line art and coloring are separate stages; roughs convey panel/dialogue/character positions and scroll direction | BP-404/BP-405: vertical reading needs its own directing/preview. Do not infer a constant page-to-scroll expansion multiplier |
| R5 | [MangaFlow, arXiv:2605.28173v2](https://arxiv.org/abs/2605.28173v2), revised 2026-09-14; author abstract and version record checked | Explicit editable layout/reference variables, staged rendering/composition/lettering, and reusable character/scene/object memory | BP-101/BP-303/BP-401/BP-501. Architecture evidence, not a measured guarantee for a local 4B planner or the installed image model. Inspect version-pinned methods before replicating experiments |
| R6 | [ViStoryBench, v4](https://arxiv.org/html/2505.24862v4), research benchmark | Evaluates identity, style, prompt/action alignment, character sets and novelty beyond reference copy-paste; reports 80 story segments | BP-504/BP-701: separate these quality dimensions. Detector/judge-dependent metrics do not certify editorial fidelity, readable lettering or user success |
| R7 | [BookNLP, official project](https://github.com/booknlp/booknlp), maintained source/README | Explicit character name clustering, coreference, quotation-speaker identification and text offsets for English books | BP-202: canonical IDs and speaker provenance are first-class data. An architectural/comparison reference, not an automatic dependency choice; language, footprint and local performance need evaluation |

Japanese editorial sources were read for their workflow descriptions; summaries
above are paraphrases. No proprietary manga manuscript is copied into the test
corpus. MangaFlow's unversioned abstract currently points to v2; do not combine
its current metadata with unverified claims from older HTML versions.

## 2. Production workflow versus current Board

The proposed sequence is an engineering synthesis of R1–R7, not a verbatim
publisher workflow. A solo creator can fill several human roles.

| Production decision | Durable artifact | Human decision | Board work |
| --- | --- | --- | --- |
| What story is being adapted and what matters? | Source revision, essential beats, chapter/arc plan | Source author/adaptation editor approves emphasis, cuts and inventions | P2 |
| Who/where/what persists? | Canonical bible, scoped costumes/props, approved references | Creator approves visual identity and unsupported design additions | P3 |
| How does the reader experience it? | Name: panel geometry, text, action, reading order and reveals | Editor/creator reads the rough chapter for comprehension and pacing | P4 |
| Does the art actually deliver the plan? | Versioned panel images and complete render recipes | Artist/creator accepts or corrects identity, action and composition | P5 |
| Can this chapter be delivered? | Lettered pages, selected revision manifest, profile-specific export | Final proofreader checks actual output, not just planning metadata | P6 |
| Does the tool help its intended user finish? | Complete trial records and user feedback | Product owner accepts or narrows supported scope | P7 |

## 3. Corrections to earlier research interpretations

[adaptation-research.md](adaptation-research.md) remains useful exploratory
background, but the following statements must not become hard acceptance rules.

| Earlier interpretation | Correct treatment for implementation |
| --- | --- |
| The field universally converged on a fixed stage count | Several systems expose intermediate artifacts; count and order vary. Choose boundaries for review, caching and failure isolation |
| A short-story study's segmentation accuracy is a realistic ceiling; a few fragments is a novel target | Dataset/model-specific findings do not establish a ceiling or long-chapter panel budget; validate source chunking and essential-beat retention locally |
| Better visual captions are always a larger gain than any model upgrade | Brief construction and model choice are separate experimental variables. Compare on the same supported runtime and corpus |
| A reference sheet is a hard pixel lock | Reference conditioning is fallible and can copy poses or contaminate faces. Require visual checks and retain alternative variants |
| Every panel should always receive both cast and room references | Contradicted by current reference-selection policy and its recorded contamination observations. Test role/state-specific combinations within model caps |
| A note makes regeneration a precise correction | Current redraw changes instructions and seed but regenerates the full image. Preservation is an outcome to test, not a guarantee |
| One fixed maximum cast count, balloons-per-panel count, or words-per-balloon rule | Model and layout limits are constraints to disclose, not permission to silently remove people or text. Measure font fit and approve adaptation splits |
| Fixed panel/page ratios, splash quotas, gutters or scroll multipliers are manga standards | Treat as optional editable presets only; approve the selected format and evaluate pacing with readers |
| Source anchoring guarantees nothing important is dropped | Anchoring verifies provenance relationships, not actual depicted meaning. Essential-beat accounting and independent fidelity review are both required |
| An automated visual judge can approve a whole chapter | Judges are triage aids until calibrated; semantic, typographic and narrative defects need human decisions and deterministic checks |

The code-derived observations in the parent tracker are higher-confidence than
unreproduced runtime outcomes. In particular: Gallery does expose PDF binding;
it is not a missing feature. Its selection of composition revisions needs tests.
Existing encrypted draft storage should be extended/migrated, not replaced as
though persistence were absent.

## 4. Research spikes with decisions and stop conditions

All spikes are `TODO`; none has a newly measured result in this documentation
change. Each ends in a short recorded decision, including a negative result.
Use the fixture/test IDs in the test plan. Keep asset evidence local/encrypted
unless it is explicitly safe and authorized to export.

### SP-01 — can the local planner support the adaptation contract?

- Work: BP-201/BP-202/BP-204. Tests: BT-021–BT-028.
- Compare current direct panel drafting against staged source units → beats →
  panel plan on the same source, model and token/time budget. Use FX-01/02/04/05;
  include a held-out ending and alias ambiguity, not only simple action scenes.
- Measure valid complete responses, essential-beat accounting, false invented
  events, identity/speaker errors, human edits, total tokens and elapsed time.
- Control retry counts, chunk boundaries and decoding settings; log truncation.
  Reusing a larger context silently is not a fair equal-budget comparison.
- Decision: ship staged local planning, narrow input bounds, or ship a
  manual-first beat/name editor. Larger/downloaded or hosted models are separate
  product/resource choices, not an implicit remedy.
- Stop condition: after the fixed corpus/budget is exhausted, report failures;
  do not loop prompts until one attractive shot list appears.

### SP-02 — which reference strategy works for each supported model?

- Work: BP-303/BP-304. Tests: BT-032–BT-034, BT-056.
- Compare no reference, approved combined lineup, selected individual identity
  refs, identity + previous compatible panel, and identity + room where supported.
  Include empty rooms, close-ups, two/three people, changed pose, and night return.
- Hold source/brief/geometry/settings fixed within each pair; repeat with at
  least three declared seeds. Respect available image-reference slots; a model
  supporting one input is not tested with three hidden inputs.
- Record count, likeness, actor/action alignment, contamination, pose repetition,
  memory and latency. Keep failures such as window-frame/face contamination.
- Decision: per-model role-priority/fallback table and clear UI explanation;
  maintain current room-withholding policy unless local evidence overturns it.
- Stop condition: choose the least costly strategy meeting the frozen quality
  target; if none qualifies, declare that cast complexity unsupported for final
  generation and preserve the manually editable/import path.

### SP-03 — does geometry-first rendering protect the intended panel?

- Work: BP-403/BP-501. Tests: BT-044, BT-050, BT-051.
- Compare current square-render/center-crop against supported target-aspect
  rendering and an explicit fit/pad workflow; use wide, tall and normal cells.
- Measure protected-face/action crop loss, prompt/pose fidelity, actual output
  dimensions, memory and elapsed time. Include reserved lettering regions.
- Decision: model capability table for dimensions and padding/alignment rules;
  approved crop preview where native aspect ratios are unavailable. Unsupported
  sizes must fail early, not be silently rounded into a different composition.
- Stop condition: no geometry option qualifies → preserve art via visible fit
  and manual layout change, and block misleading “matches approved framing”.

### SP-04 — are automated visual checks useful enough to ship?

- Work: BP-504/BP-701. Tests: BT-056/BT-057.
- Create human-labelled positive and negative examples: identity swap, extra
  person, wrong handoff, wrong outfit, reference copying, and valid pose changes.
- Evaluate each optional local detector/judge against the labels; record false
  positives/negatives, disagreements, runtime, model footprint and supported styles.
  A model that generated the panel is not an independent ground-truth oracle.
- Decision: ship explainable advisory flags if helpful; otherwise a structured
  manual checklist. No automatic approval solely from a similarity score.
- Stop condition: insufficient label agreement or too many misleading flags →
  do not add a dependency just to display a confidence number.

### SP-05 — can users repair a panel within an acceptable effort budget?

- Work: BP-503/BP-702. Tests: BT-054/BT-055/BT-071.
- Compare note + reroll, supported instruction edit, supported masked repair,
  and imported replacement on the same concrete defects.
- Measure corrected-region success, unintended changes outside the region,
  identity retention, elapsed/active time and attempts before acceptance.
- Decision: name operations honestly and expose preservation limitations;
  choose default operation by defect type/capability, not by one universal button.
- Stop condition: hit the predeclared attempts/time budget → offer manual/import
  fallback with history intact, not unbounded regeneration.

### SP-06 — format, typography, and actual handoff requirements

- Work: BP-004/BP-402/BP-604. Tests: BT-043/BT-065/BT-066.
- Establish the first digital reading profile and fonts; measure real text fit
  in supported language(s). For print/scroll extensions, obtain the named
  recipient's current official specification and record URL/date/version.
- Ask a pilot artist/editor to open the assets + layout/text package in their
  actual tool. Record what they can edit and what must be reconstructed.
- Decision: certify only demonstrated profiles; distinguish raster PDF, editable
  manifest and a native layered manuscript. Source/asset/font permissions must
  be recorded without claiming a blanket legal clearance.
- Stop condition: missing recipient specification or unusable handoff → leave
  that profile experimental; digital preview delivery can remain in scope.

## 5. Decision log

Decisions below are recommendations for BP-004 to ratify, not user approvals
already obtained or implementation already landed.

| ID | Proposed decision | Why | Revisit when |
| --- | --- | --- | --- |
| DR-01 | Keep inference local; support a manual-first path when model reasoning fails | Matches repository product constraints; failure must be recoverable | User explicitly changes deployment/model requirements |
| DR-02 | First target a short chapter in one declared language/format | Allows meaningful end-to-end evidence without pretending a complete novel is proven | G7 passes and longer/chapter-continuity corpus is available |
| DR-03 | Persist reviewable intermediate artifacts and revision-bound approvals | Prevents stale output and makes editorial intervention possible | Schema validation or migration findings require refinement |
| DR-04 | Preserve complete text; overflow is a structured error | Text loss changes the delivered story | Never waive silent loss; improve layout instead |
| DR-05 | Differentiate incomplete proof from approved final output | Partial progress is useful without claiming completion | Proof UI testing identifies a clearer presentation |
| DR-06 | Human acceptance remains authoritative for meaning and visual fidelity | Available research metrics do not establish chapter-level correctness | Calibrated evidence supports limited automated triage, not blanket approval |

## 6. Unanswered questions before broad production claims

1. How often can this installed local planner account for a chapter's essential
   beats without substantial rewriting? SP-01 and the held-out pilot answer it.
2. Which model/settings/reference strategy sustains identity **and** action
   across transitions? SP-02, not a reference-sheet demo, answers it.
3. Can useful framing and lettering be achieved at acceptable latency/RAM on
   the declared minimum Mac? SP-03 plus BP-003/BP-502 answer it.
4. Does the name editor reduce adaptation/repair labor enough to matter to
   creators? BP-702 measures observed work rather than asking only preference.
5. What does the recipient actually require: readable digital PDF, vertical
   delivery files, or editable production manuscripts? SP-06 controls scope.

No numeric readiness rating is assigned. Readiness is the set of passed gates,
supported profiles, and evidence-linked limitations in the parent tracker.
