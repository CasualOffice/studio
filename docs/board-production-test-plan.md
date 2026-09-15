# Board production test plan

Updated: 2026-09-16. Parent: [production tracker](board-production-tracker.md).
All cases below are **specified, not newly implemented or executed**. Existing
tests may cover parts of a case; BP-001 must identify those before adding more.
Test IDs are stable so tasks, PRs, failures and gate evidence can refer to them.

## Test lanes and evidence

| Lane | Purpose | Execution and oracle |
| --- | --- | --- |
| U — unit/property | IDs, source spans, validation, invalidation, layout math, manifest membership | Deterministic functions; human-authored expected data; generated edge-case inputs |
| I — integration | TS/Rust/Python contracts, vault lifecycle, migration, cancellation and export | Stub expensive generation; exercise real serialization/composition/storage where possible |
| UI — workflow | User decisions, stale-state display, controls, keyboard navigation, reopen | Component tests plus a real Tauri workflow; introduce a harness if missing, do not imply one exists |
| M — real model | Local planner/image behavior, latency, RAM, reference interference, repair limits | Installed supported models on declared Mac; retain every attempted output and recipe |
| H — human | Story comprehension, emotional beats, likeness, page flow, final acceptability | Author/adaptation reviewer and independent readers using a rubric fixed before seeing variants |

U/I/UI tests are eligible for regular CI, avoiding model downloads or real user
vaults. M/H are explicit pre-release evidence lanes, not “covered” by mocked
tests. Use isolated temporary vaults and synthetic source material for storage
and cancellation tests. Never fault-inject into the user's working project.

Current entry points: [board.test.ts](../src/lib/board.test.ts),
[test_worker.py](../engine/test_worker.py), Rust tests/probes, and
[preflight.sh](../scripts/preflight.sh). Run the full preflight for implementation
changes once its prerequisites are available; record missing dependencies as
blocked execution, not passing/skipped quality evidence. New test files and
fixture manifests should be added in BP-002, not assumed to exist today.

## Fixture contract

Each fixture needs: ID/version, original source, language, source hash,
permitted use, essential beats, acceptable adaptation alternatives, entity and
speaker truth, continuity transitions, format assumptions, and expected
comprehension questions. Keep answer keys independent of model output. Source
unitization and offset conventions must be fixed across JS/Python/Rust.

| Fixture | Contents | Required challenge |
| --- | --- | --- |
| FX-01 | Original short scene below | Same-person aliases, unnamed narrator, offscreen speech, key transfer, cause/effect |
| FX-02 | Original reflective scene with a memory, hypothetical fear and present action | Do not depict a thought or possible future as a present event without an approved treatment |
| FX-03 | Two short chapters; return to the same room at night, new coat, damaged prop | Canonical identity persists while time-scoped appearance/state changes |
| FX-04 | Four named people, two sharing a surname; two overlapping conversations | Count, alias ambiguity, actor/speaker separation; no forced invented cast reduction |
| FX-05 | Chapter exceeding current planner bounds; essential reveal in final paragraph | Chunk continuation, no lost ending, overlap deduplication, explicit resource envelope |
| FX-06 | Dialogue-heavy page: three speakers, long caption, long unbroken token, punctuation and line breaks | Full text preservation, overflow errors, manual split/reword workflow |
| FX-07 | Human-checked Japanese punctuation/names plus combining characters and emoji offset probes | Unicode-safe provenance; declared unsupported language is an honest outcome |
| FX-08 | Identical approved panels in LTR, RTL page order and vertical scroll names | Independent order/reveal contracts; no simplistic image mirroring or fixed scroll multiplier |
| FX-09 | Generated geometric art with face/action safe boxes and known text lengths | Deterministic cropping/occlusion/layout assertions without a generative-image oracle |
| FX-10 | v1 projects: empty, half-drawn, multiple references, notes, archived versions | Migration, corrupt/missing asset references, reopen, history preservation |
| FX-11 | Synthetic delayed/cancelled/failed jobs and repeated composition | Ownership under edits, retry idempotence, export revision membership |
| FX-12 | Consented short pilot chapter not used for prompt tuning | Unseen-source generalization, manual labor and reader comprehension |

### FX-01 — ready-to-implement seed fixture

Original text authored for this plan; deliberately unspecified narrator appearance:

> My name is Mira Vale, though Jonas calls me Mi. In the station waiting room,
> I unhooked the brass key from my red coat and placed it in Jonas's open hand.
> He closed his fingers around it. A guard called from behind the locked door,
> “Last train!” I stayed beside the bench while Jonas used the key to unlock
> the door. When he opened it, he found the platform empty. The train had gone.

Human oracle:

- Mira Vale / Mi / the first-person narrator are one entity. Jonas is another.
  The guard is an unnamed third entity, heard but not necessarily visible.
- Key initially belongs to Mira's visible red-coat state; possession changes to
  Jonas before he uses it. The door is initially locked, then unlocked/opened.
- Essential beats: key handoff; offscreen warning; Jonas unlocks/opens while
  Mira stays by the bench; empty platform/missed train reveal.
- Speech “Last train!” belongs to the guard, not to Mira or Jonas. Do not
  require the guard to appear just because the guard speaks.
- No train needs to be visible. A moving train shown outside is an added
  adaptation choice, not a fact licensed by the final sentence alone.
- Reasonable panel counts/layouts vary. Acceptance does not compare to one
  literal shot list. Questions: who gave whom the key, who warned them, why
  could Jonas open the door, and what did he discover?

BP-002 must turn these facts into exact spans and expected structured records;
this paragraph is not yet a machine-readable fixture suite.

## P0 cases — baseline and scope

| ID | Lane / fixture | Procedure | Acceptance oracle | Task / gate |
| --- | --- | --- | --- | --- |
| BT-001 | I / existing suites | Record commit/runtime; execute existing checks with correct dependencies; capture errors and skips | Exact commands/results retained; no fabricated pass count or confusion between missing package and behavior defect | BP-001 / G0 |
| BT-002 | U + UI / FX-06, FX-11 | Reproduce enriched action edit, >2 dialogue entries, missing middle image, and two compositions of one project | Record actual/expected behavior and smallest reproduction for each; mark suspected export mixing verified or not reproduced | BP-001 / G0 |
| BT-003 | U + H / fixture set | Validate manifest, rights/provenance, source hashes and human answer keys; review ambiguity | Every supported case has an independent oracle; unsupported cases labelled, not quietly removed | BP-002 / G0 |
| BT-004 | M + H / FX-01, FX-03 | Run from source through Gallery PDF; record cold/warm timings, failures, retries, active human time and memory | Reproducible run record including unsuccessful attempts; no quality threshold asserted from old screenshots | BP-003 / G0 |
| BT-005 | H / scope record | Review format/language/model/input envelope and quality targets before implementation comparison | Dated approved targets and exclusions; values fixed before final evaluation | BP-004 / G0 |

## P1 cases — state integrity

| ID | Lane / fixture | Procedure | Acceptance oracle | Task / gate |
| --- | --- | --- | --- | --- |
| BT-010 | U + I / FX-10 | Migrate all v1 variants; reopen; repeat migration; round-trip TS/Rust/Python serialization | Stable unique IDs, preserved notes/assets/order/style, idempotence; original encrypted snapshot retained | BP-101 / G1 |
| BT-011 | I / FX-10 | Open malformed/unknown-version draft and one referencing a missing asset; fail a write | Visible recoverable error; no overwrite with empty state; no plaintext manuscript in preferences/logs | BP-101 / G1 |
| BT-012 | U + UI / FX-01 | Enrich, render, change action and setting, inspect next recipe; alter cast/state afterward | Old description/place/ref dependencies cannot override edit; affected approvals stale, unrelated panels retained | BP-102 / G1 |
| BT-013 | U + UI / FX-06 | Change caption/dialogue only; first keep within reserved area, then force overflow | Art ID unchanged for fitting text; text/page approval invalidated; overflow requires layout resolution, never automatic expensive redraw | BP-102 / G1 |
| BT-014 | U/property + UI / FX-10 | Insert, split, merge, reorder, delete and undo panels including one with notes/variants | No duplicate IDs or dangling ownership; coverage/order revalidated; conflicts surfaced; historical assets recoverable | BP-103 / G1 |
| BT-015 | I + UI / FX-11 | Delay a render/compose response, edit/reorder/switch project, then deliver result | Result attaches only to original ID/revision/history; cannot become current on another panel/project | BP-104 / G1 |
| BT-016 | I + UI / FX-11 | Cancel at enqueue/mid-job/save boundary; simulate crash and reopen; fail archive save | Completed committed work survives; cancelled/unknown jobs reconciled; no stuck busy state or silent archive loss | BP-104 / G1 |
| BT-017 | U + UI / FX-10 | Approve an artifact, edit its input, reload and attempt downstream final action | Revision-bound approval becomes stale; blocker explains dependency and repair; rejection does not delete variants | BP-105 / G1 |

## P2 cases — source fidelity and adaptation

| ID | Lane / fixture | Procedure | Acceptance oracle | Task / gate |
| --- | --- | --- | --- | --- |
| BT-020 | U + I / FX-01, FX-07 | Slice source around punctuation, repeated sentences, emoji and combining characters; round-trip offsets | Every valid span returns the exact original text; no normalization drift; repeated phrases retain distinct locations | BP-201 / G2 |
| BT-021 | I + M / FX-05 | Force token cutoff, invalid JSON and overlapping chunks; resume with ending in a later chunk | No false complete result or duplicate beats; final reveal accounted for or explicit incomplete/error state | BP-201 / G2 |
| BT-022 | U + M + H / FX-01, FX-04 | Extract/approve aliases; ambiguous same surname; feed registry through planner/enrichment | Mira/Mi/narrator use one ID; same-surname people not collapsed; uncertainty exposed; planner does not invent replacement names | BP-202 / G2 |
| BT-023 | U + M + H / FX-01, FX-02 | Resolve offscreen speech and first-person thought after manual identity correction | Guard retains speech without forced visual presence; manual narrator mapping reaches every affected stage | BP-202 / G2 |
| BT-024 | U + H / FX-01 | Anchor every source sentence but change the key recipient or omit the opening-door action | Source-accounting score may remain high; separate fidelity/essential-beat review must fail | BP-203 / G2 |
| BT-025 | U + UI / FX-05 | Have >20 unresolved units; approve one omission with a reason, then edit source | Correct total and pageable complete list; explicit dispositions; affected approval becomes stale | BP-203 / G2 |
| BT-026 | M + H / FX-02, FX-05 | Plan under two page budgets and deliberately move a reveal; review causal dependencies | Constraints negotiated visibly; essential causality preserved; no fixed words-to-panels rule certified as narrative truth | BP-204 / G2 |
| BT-027 | U + UI + H / FX-01 | Split handoff into anticipation/action/reaction; merge context beats; reorder panels | Many-to-many beat/panel mapping remains correct; no falsely missing or duplicated essential beat | BP-204 / G2 |
| BT-028 | UI + H / FX-02 | Choose thought caption versus visual metaphor; approve invented setting detail; reject a reordered memory | Source facts, inference and adaptation decisions distinguishable and revisioned; unresolved material changes block approval | BP-205 / G2 |

## P3 cases — bible and references

| ID | Lane / fixture | Procedure | Acceptance oracle | Task / gate |
| --- | --- | --- | --- | --- |
| BT-030 | U + UI / FX-03 | Give same entity different coats per chapter; approve unspecified appearance; reuse a place | Canonical identity persists; state changes are scoped; inferred design never relabelled as source fact | BP-301 / G3 |
| BT-031 | UI + I / FX-04 | Generate/import references, reject one, approve others, attempt batch render | No final batch silently consumes rejected/unreviewed refs; human can identify which entity each approved image depicts | BP-302 / G3 |
| BT-032 | U + I / FX-04 | Exercise models with zero, one and several supported refs; inspect selection/order/deduplication | Request respects actual capability including zero; selected entities/state match; unsupported combination produces explanation | BP-303 / G3 |
| BT-033 | M + H / FX-01, FX-04 | Paired lineup/individual/room/prior-panel runs using SP-02 | Measure likeness, count, action and reference contamination together; no strategy wins solely from identity similarity | BP-303 / G3 |
| BT-034 | U + M / FX-03 | Change rainy morning to rainy night, coat, key possession, location and cast membership | Shared weather word cannot override incompatible time/state; reference chain stops/changes appropriately; intended transitions preserved | BP-304 / G3 |

## P4 cases — name and lettering

| ID | Lane / fixture | Procedure | Acceptance oracle | Task / gate |
| --- | --- | --- | --- | --- |
| BT-040 | U/property / FX-09 | Generate/edit varied page rectangles, gutters and panel orders | Boxes valid and in bounds; no unintended overlap; every panel occurs exactly where manifest specifies | BP-401 / G4 |
| BT-041 | UI + H / FX-08 | Place a reveal on a page turn; compare LTR/RTL orders and spreads | Intended sequence and reveal retained; RTL means explicit order, not mirrored character art | BP-401 / G4 |
| BT-042 | U + I / FX-06 | Pass three or more lines and long captions through cleaning, API, storage and composition | Full approved text survives; explicit measured overflow with panel/text IDs instead of slicing or dropped bubbles | BP-402 / G4 |
| BT-043 | U + UI / FX-06, FX-07 | Missing font, unsupported glyph, long token, configured minimum type size; resolve by edit/split | Missing/overflowing text blocks final; no tofu accepted or hidden tiny font; saved original text preserved | BP-402 / G4 |
| BT-044 | U + UI + M / FX-09 | Render wide/tall planned cells; put protected face near a boundary; change page geometry | Planned/actual fit reported; critical crop blocked or deliberately corrected; final box does not unexpectedly center-crop action | BP-403 / G4 |
| BT-045 | U + UI + H / FX-01, FX-04 | Crossed dialogue, offscreen guard, narrator caption; move a speaker region | Balloon order and target remain correct; no tail implies the wrong visible person; tail-less/offscreen forms supported | BP-404 / G4 |
| BT-046 | U + UI / FX-09 | Place balloon on protected face/hand/key region; add thought/SFX objects | Collision surfaces; manual placement resolves it; object type/order persists, including on reopen | BP-404 / G4 |
| BT-047 | UI + H / FX-08 | Read full rough chapter in page and scroll modes; change layout after approval | Reader follows intended order; format-specific preview; changed name invalidates dependent final-art approval | BP-405 / G4 |

## P5 cases — rendering, corrections, and sequence quality

| ID | Lane / fixture | Procedure | Acceptance oracle | Task / gate |
| --- | --- | --- | --- | --- |
| BT-050 | U + I / FX-01 | Dispatch renders for different model capability/default combinations | Dimensions, steps, guidance, refs and current brief are valid per model; complete recipe saved without assuming one universal guidance value | BP-501 / G5 |
| BT-051 | I + M / FX-09 | Compare approved geometry with actual request/output; reopen recipe and variant | Correct owner/input hashes and output dimensions; reproduction information complete; exact pixels not promised across runtime versions | BP-501 / G5 |
| BT-052 | I + UI / FX-11 | Fail one middle job; retry then cancel; resume after restart | Successful panels retained; bounded retries; no duplicates, silent skips or starvation; per-panel actionable error | BP-502 / G5 |
| BT-053 | M / FX-03 | Cold/warm batch at supported settings; memory pressure and one safe resource-limit failure | Record stage latency/RAM/failure recovery; ETA based on observed configuration and includes references/retries | BP-502 / G5 |
| BT-054 | UI + M + H / FX-01 | Correct wrong key/hand region using supported edit mode; compare outside requested area | Preserve accepted variant; report unintended identity/background changes; user can reject/revert rather than forced acceptance | BP-503 / G5 |
| BT-055 | I + UI / FX-11 | Try unsupported mask mode; import replacement; cancel repair; accept previous variant | Honest capability response; imported/parent lineage preserved; no loss of current accepted output | BP-503 / G5 |
| BT-056 | M + H / FX-04 | Review good likeness/wrong action, right count/wrong identities, copy-pasted poses, and clean control panels | Separate rubric catches each critical error; automated judge false positives/negatives recorded against humans | BP-504 / G5 |
| BT-057 | M + H / FX-03 | Review locally good panels with a cross-panel key/costume/daylight contradiction | Sequence review fails affected transition even if every isolated image looks good; only dependent work reopened | BP-505 / G5 |

## P6 cases — composition and export

| ID | Lane / fixture | Procedure | Acceptance oracle | Task / gate |
| --- | --- | --- | --- | --- |
| BT-060 | I + UI / FX-11 | Missing middle panel, failed panel, rejected image and stale approved variant; request proof then final | Proof identifies gaps and incompleteness; final blocked with exact panel reasons; no silent sequence compression | BP-601 / G6 |
| BT-061 | I + UI / FX-11 | Compose twice after changing page order/art; retain both sets in vault | Each composition has immutable ordered membership; current selection cannot mix historical pages | BP-601 / G6 |
| BT-062 | U + I / FX-06, FX-09 | Compose same approved manifest twice; inject text overflow or missing asset | Same geometry/text/order for same inputs; structured error, not partial “success”; font/version context captured | BP-602 / G6 |
| BT-063 | I + UI / FX-11 | Export PNG and bind PDF in Gallery after reopening with two composition revisions | Exactly selected revision's page count/order/content; inspect exported files, not only internal metadata | BP-603 / G6 |
| BT-064 | I + UI / FX-11 | Cancel file dialog, fail export write, choose existing filename, retry | No reported success before completed write; overwrite follows explicit UX; encrypted originals remain intact | BP-603 / G6 |
| BT-065 | I + H / FX-08, FX-09 | Validate selected digital profile and any separately declared printer/scroll profile | Actual dimensions/order/color/resolution/chunking match recorded specification; reject unvalidated “print-ready” claims | BP-604 / G6 |
| BT-066 | I + H / approved sample | Export editing package; inspect manifest, assets, text and font/provenance notes outside app | Another tool/person can recover panel placement and full text; plaintext export intentional; no false layered-native claim | BP-604 / G6 |

## P7 cases — user outcome

| ID | Lane / fixture | Procedure | Acceptance oracle | Task / gate |
| --- | --- | --- | --- | --- |
| BT-070 | M + H / frozen corpus | Run fixed sample with all attempts, including rejected and manually repaired outputs | Per-fixture metrics and failure taxonomy; no cherry-picked best page as entire result | BP-701 / G7 |
| BT-071 | H / FX-12 | Author adapts source, approves name, repairs art, reopens project and exports without developer rescue | Completion, abandonment and assistance recorded; any developer rescue counted as failure of independent completion | BP-702 / G7 |
| BT-072 | H / FX-12 | Independent readers see exported chapter without source; answer fixed questions and identify speakers | Evaluate comprehension against source-authored key; record reader disagreement and ambiguous questions | BP-702 / G7 |
| BT-073 | H / baseline + candidate | Compare quality, active effort, retries and elapsed time under comparable settings | Frozen thresholds met by supported profile, or scope explicitly narrowed; quality gains do not conceal excessive manual labor | BP-703 / G7 |
| BT-074 | UI + H / release candidate | Follow onboarding and every advertised Board claim on clean test project | Claims match tested capabilities; limitations visible; release decision and evidence signed | BP-704 / G7 |

## Provisional pilot thresholds to ratify at G0

These are proposed product targets, **not research-derived constants or current
results**. Change them before the final evaluation, with a reason in the scope
record. Do not tune them after seeing the candidate's scores.

| Measure | Proposed first pilot contract |
| --- | --- |
| Safety/completeness | Zero lost approved text, corrupted projects, wrong-revision final pages, or unresolved essential-beat omissions |
| Corpus | At least 6 short, varied scenes plus one two-chapter continuity case; a held-out pilot chapter; supported-language/profile subsets explicit |
| Real-image sample | At least 24 planned panels across the corpus, including 2–3-person scenes and action/prop interactions; all attempted variants counted |
| Author/editor review | Every final chapter approved for source meaning, dialogue attribution and intended reveal; zero unresolved critical visual contradictions |
| Reader check | At least 3 independent readers; each answers at least 80% of the fixed essential-story questions correctly; report per-reader/per-story counts |
| Independent completion | At least 2 of 3 pilot creators finish the scoped workflow without developer intervention; count onboarding/help use and abandonment |
| Repair effort | Median generation attempts per accepted panel at most 3; report tail/max and fraction imported or manually repaired; confirm acceptable manual effort with pilot creators |
| Performance | No invented universal minutes-per-page goal. Set device/profile-specific latency, memory and active-effort ceilings from BP-003 and pilot user tolerance before BP-701 |

Tiny samples do not establish a population-wide success rate. Report actual
numerators/denominators and scope; expand the pilot before broader claims. Repeat
the paired reference/geometry experiments across more than one seed, but do not
assert bitwise determinism across different model/runtime/hardware versions.

## Failure severity and triage

- **Critical:** data loss, private source leakage, wrong project/revision output,
  silent missing essential beat/dialogue, wrong actor causing narrative reversal,
  unreadable final order. Blocks the responsible gate and final release.
- **Major:** continuity/crop/lettering fault requiring reader inference or manual
  repair, unreliable resume, repeated generation failures. Blocks final artifact
  acceptance until resolved; an unsupported capability can be excluded explicitly.
- **Minor:** cosmetic mismatch not changing meaning or legibility. Record and
  obtain reviewer acceptance; do not silently aggregate into a quality pass.

Each result records: test ID, status (`NOT RUN`, `PASS`, `FAIL`, `BLOCKED`, or
`NOT APPLICABLE` with scope reason), commit, environment, fixture version,
actual versus expected, evidence location, defect/task ID, owner, and retest.
New cases remain `NOT RUN` until such a result exists.
