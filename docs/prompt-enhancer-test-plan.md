# Prompt enhancer acceptance and regression plan

Updated: 2026-09-16. [Tracker](prompt-enhancer-tracker.md).
[Audit, research and actual baseline observations](prompt-enhancer-audit.md).

The PT cases below are proposed acceptance tests, **not newly passing tests**.
PD-01–PD-18 are executed diagnostic observations; PL-01–PL-12 are executed live
writer cases. They are deliberately not confused with acceptance of a fix.

## Execution lanes

| Lane | Purpose | Restrictions |
| --- | --- | --- |
| U | Cleaning, exact constraints, decisions, schemas, diffs | Deterministic; include valid paraphrases and adversarial minimal pairs |
| I | Production `op_assist`/host boundaries with controlled inference | Stub model work, not the decision logic under test; no downloads/user vault |
| UI | Deferred responses, proposal acceptance, warning visibility, cancel/undo | Real component/workflow interaction; source-string assertions are insufficient |
| M | Installed writer/reader and generated image/video outputs | Explicit supported runtime; offline unless product scope changes; retain failures |
| H | Human intent, target grounding, usefulness and output preference | Fixed answer keys/rubric; model-written text is not its own ground truth |

Use a synthetic fixture registry with source prompt, task/profile, protected
constraints/spans, permitted additions, ambiguous choices, candidate mutations,
reference/mask truth, expected outcome, language and provenance. Distinguish
literal “no sunroof” from an optional negative-prompt field. Keep exact strings
and Unicode untouched where they are intended image text.

## Ready-to-use fixture families

| Fixture | Inputs / variation | Expected truth |
| --- | --- | --- |
| PF-01 | `2 red cars with no sunroof, plain white background`; mutate count/exclusion individually | Two cars; red; no sunroof; white background; no invented scenery in Clarify |
| PF-02 | `A red cube left of a blue sphere`; swap colors or relation | Red belongs to cube; blue to sphere; ordered spatial relation preserved |
| PF-03 | `Mira gives Jonas a brass key. Mira wears red; Jonas wears blue.` | Actor/recipient and costume bindings survive independently of word overlap |
| PF-04 | Exact sign `BEST QUALITY 4K`; `a masterpiece hanging in a gallery`; `Unreal Engine editor interface` | Quoted lettering, subject noun and named tool preserved; no global blacklist deletion |
| PF-05 | `a hous at nite`; uncommon proper names; `wears`; `setting/sitting` | Clear spelling fixes allowed; inflection/name/homophone guesses not falsely certified |
| PF-06 | Black ink line art, no shading/color; diagram; logo; product reference with empty background | Medium and exclusions preserved; no compulsory camera, narrative scene or oil-painting style |
| PF-07 | Green jacket recolor, with/without image/reader; two jackets; remove left person | Always an edit; ambiguity explicit; no invented person/setting; target and preserved regions labelled |
| PF-08 | Masked region, outpaint sides, source image + style reference | Operation and role-specific context retained; unsupported capabilities visible |
| PF-09 | Fast runner, locked camera; still subject with moving camera; actual short/long clip profiles | Requested speed/action and camera constraints kept or explicitly negotiated |
| PF-10 | I2V image with two plausible motion subjects; tiny target/text; missing reader | Grounding uncertainty and reader use reported accurately; no fake observations |
| PF-11 | Multi-line result, truncated JSON/text, commentary, empty answer, cancellation/error | Complete structured result or honest failure; no successful first-line fragment |
| PF-12 | Delayed jobs while typing/changing image/model/mode; existing proposal then another request | Proposal tied to exact input snapshot; no stale overwrite; cancel/reopen/undo consistent |
| PF-13 | Human-checked non-English prompts, mixed language, CJK literal text, emoji/combining characters | Exact spans survive; unsupported language labelled rather than processed as English with false confidence |
| PF-14 | Held-out user-authored tasks not copied from system examples | Independent generalization/usefulness evidence; privacy/provenance recorded |

## Acceptance cases

| ID | Lane / fixture | Procedure and expected result | Task / gate |
| --- | --- | --- | --- |
| PT-001 | U + I / PF-01–PF-12 | Reproduce each PD finding and positive control against pinned commit; distinguish injected candidate from real generated output | PE-001 / EG0 |
| PT-002 | H / all fixtures | Independently annotate explicit constraints, legitimate paraphrases and optional additions; review ambiguous expected results before coding | PE-001 / EG0 |
| PT-003 | H / scope | Approve default transformation, supported tasks/models/languages and evaluation limits before tuning; keep held-out prompts separate | PE-002 / EG0 |
| PT-010 | U + I / PF-07–PF-10 | Round-trip request/result across TS/Rust/Python; preserve IDs/hash/task/profile/ref roles/mask/duration; reject malformed/unknown version | PE-101 / EG1 |
| PT-011 | I + M + H / PF-01, PF-06 | Clarify sparse and detailed requests: no invented substantive design choices, exact constraints retained; already-good input may stay concise | PE-102 / EG1 |
| PT-012 | UI + M + H / PF-06 | Opt into direction; additions labelled and rejectable; style/subject/exclusion locks still bind; explain proposal versus unchanged/rejected | PE-102 / EG1 |
| PT-013 | I / PF-07 | Full matrix edit × source present/absent × reader present/absent; all combinations remain edit policy or a named missing-input outcome | PE-103 / EG1 |
| PT-014 | I / PF-09, PF-10 | Unknown mode rejected; video remains video; actual vision use—not input count—sets reader status | PE-103 / EG1 |
| PT-020 | U + I / PF-04 | Preserve exact quoted phrase through cleaning→writer→validator→proposal; meaningful named terms unchanged | PE-201 / EG2 |
| PT-021 | U / PF-01, PF-13 | Preserve digit/word counts, `no`/`not`, punctuation-sensitive quoted text and Unicode spans; contextual cleanup is reversible | PE-201 / EG2 |
| PT-022 | U + I / PF-01 | Inject 2→3 and no-sunroof→sunroof candidates; hard conflict explicitly blocks “safe proposal”; valid two/2 paraphrase allowed | PE-202 / EG2 |
| PT-023 | U + I / PF-02, PF-03 | Reverse relation, exchange colors, or swap actor/recipient without changing vocabulary; all detected as meaning changes or uncertain, never safe by overlap | PE-202 / EG2 |
| PT-024 | U + I + H / PF-06, PF-09 | Walking→sitting, sprint→slow drift, ink→photo and locked→tracking camera mutations fail preservation; valid action paraphrases pass | PE-202 / EG2 |
| PT-025 | I / PF-05, PF-07 | Jacket→coat and valid descriptive paraphrases have one consistent decision across image/edit; no hidden `rephrased` rejection | PE-203 / EG2 |
| PT-026 | I / semantic pairs | Controlled SAME verdict cannot waive exact constraints; approved semantic match is not then rejected by unchanged lexical logic; malformed verdict is uncertain | PE-203 / EG2 |
| PT-027 | U + I / PF-11 | Multi-line/truncated/preamble/empty response: no discarded constraints or first-line success; complete candidate/finish status validated | PE-204 / EG2 |
| PT-028 | I / PF-11 | Repair targets specific violation, uses bounded attempts, revalidates original locks and returns honest failure when exhausted; never append dangling nouns as proof | PE-204 / EG2 |
| PT-029 | U + UI / PF-05 | Correct hous→house; do not label wears→warm or normal plural→singular as a confident typo; actual source/candidate spans support every displayed edit | PE-205 / EG2 |
| PT-030 | I + M + H / PF-07, PF-10 | Ground the requested object among two similar subjects using region/reference truth; uncertainty triggers choice instead of main-subject substitution | PE-301 / EG3 |
| PT-031 | I + UI + M / PF-07, PF-08 | Missing reader retains text-only edit; mask/outpaint and preserve scope reach prompt plan; visible limitation and no invented observations | PE-301 / EG3 |
| PT-032 | I + M + H / PF-09 | Preserve fast action and fixed/moving camera choices; incompatible duration/profile reported with alternatives, not silently changed motion | PE-302 / EG3 |
| PT-033 | I + M + H / PF-10 | I2V brief uses actual observed source and approved motion; absent reader/ambiguous target clearly signalled; scene additions require permission | PE-302 / EG3 |
| PT-034 | U + I + M / declared profiles | Compile same constraints for each supported backend; respect supported prompt controls/reference counts; profile guidance cannot overwrite source intent | PE-303 / EG3 |
| PT-040 | UI + I / PF-12 | Delay assist, then type/change images/model/edit-kind/switch context; old result cannot become current or overwrite latest text | PE-401 / EG4 |
| PT-041 | UI + I / PF-12 | Start assist with an old proposal present, invoke generation/cancel, fail listener registration, unmount/remount; ownership and busy/cleanup recover correctly | PE-401 / EG4 |
| PT-042 | I + M / PF-11 | Cancel during writing, semantic check, image read, retry and load boundary; propagate cancellation; no composed fallback or late usable proposal | PE-402 / EG4 |
| PT-043 | I + M / helper transition | Reader→writer→generator and reverse helper order: released objects and actual memory comply with policy; failed load leaves usable state | PE-402 / EG4 |
| PT-044 | I / synthetic files | Success/failure/cancel cleanup removes only owned files; pre-existing derived-like filename survives; inspect real log/export sinks for raw prompt leakage | PE-402 / EG4 |
| PT-045 | UI / PF-01, PF-07 | Successful blind result visibly shows note; constraint/negation/count changes and late style additions are inspectable beyond 14 word chips | PE-403 / EG4 |
| PT-046 | UI / PF-12 | Accept/keep/override/undo, including after manual edits and source changes; correct source revision restored and override remains labelled | PE-403 / EG4 |
| PT-047 | M + UI / cold/warm routes | Measure imports/load/read/write/retry/generator reload separately; correct progress, missing-model guidance and cancellation latency; no generic two-second claim | PE-404 / EG4 |
| PT-050 | M + H / PF-14 | Held-out original/Clarify/direction writer comparisons; report all attempts, explicit-constraint loss, unsupported assumptions, clarity and latency | PE-501 / EG5 |
| PT-051 | M + H / PF-07–PF-10, PF-14 | Paired original/enhanced image/edit/clip outputs with fixed model/settings and repeated seeds; evaluate task fidelity and unintended changes separately from aesthetics | PE-501 / EG5 |
| PT-052 | UI + H / user pilot | Users inspect decisions and finish scoped tasks without developer rescue; measure incorrect acceptance, repair effort, abandonment and trust in warnings | PE-502 / EG5 |
| PT-053 | H + UI / release | Advertised model/task/language scope, tooltips, docs and time estimates match evidence; unresolved profiles excluded with reasons | PE-503 / EG5 |

## Proposed release rubric — ratify at EG0

These are proposed product thresholds, not externally established standards or
current measured achievements. Do not lower them after viewing final results.

- Frozen counterfactual suite: zero known changes to exact lettering, count,
  negation, role/binding or explicit preservation scope labelled safely preserved.
- At least 30 held-out writer requests covering the fixture families; report per
  task/profile/language, including rejection/uncertainty/repair rates. Add benign
  paraphrases to measure false rejection as well as unsafe acceptance.
- At least 12 paired image-generation cases over three seeds; separately at
  least 6 targeted edit cases over three seeds. A video-support claim also needs
  its own bounded paired-clip set (proposed 4 cases over two seeds). These are
  resource-conscious pilot samples, not broad statistical proof.
- Score each pair on all explicit constraints first. Aesthetic preference cannot
  compensate for wrong lettering, recipient, count, target, or preserved content.
  Proposed benefit target: at least 60% of pairs judged more useful with no hard
  fidelity regression, by two independent reviewers with disagreements reported.
- For Clarify, zero unlabelled substantive design additions in the reviewed
  release set. For direction, additions are reviewable and explicit locks hold.
- At least three pilot users; at least two finish their scoped task without
  developer intervention and can accurately explain the proposed change and
  whether an image was actually read. Count corrections and overrides as labor.
- Set cold/warm/vision-assisted latency, cancel-response and peak-memory budgets
  from the target device and user tolerance at EG0. Current live writer timings
  are not a budget for the vision/generator pipeline.

Use blinded output order where possible. Keep all failures and seed/settings
records; do not present only the best image. A small pilot permits a limited
scope decision, not a universal claim about all prompts or models.

## Running the audit diagnostics

Deterministic, no model download or vault access:

```sh
python3 scripts/audit_prompt_enhancer.py
```

For the installed writer, use the app's Python and pass the **existing local
snapshot directory** to `--live-writer`; the script validates the path and sets
offline flags. Metal access is required. The script runs the 12 declared
synthetic prompts, seeds them in order, prints JSON observations and unloads its
own writer. It does not render images, open the application vault, or prove UI
behavior. Do not substitute a remote model ID that triggers an installation.

Existing regression commands:

```sh
npm test -- --run
```

Run `-m unittest discover -s engine` using the app interpreter or an environment
with CI's pinned dependencies. Product changes should also pass the repository's
[preflight](../scripts/preflight.sh); the diagnostic script alone is not a gate.

## Result record and severity

For every PT result record: status (`NOT RUN`, `PASS`, `FAIL`, `BLOCKED`, or
explicitly scoped `NOT APPLICABLE`), commit, task/gate, fixture version, request
snapshot, actual/expected constraint verdicts, environment, evidence location,
reviewer, defect owner and retest. All PT cases initially remain `NOT RUN`.

Critical: destroyed literal content, wrong task/target, explicit constraint loss
certified as safe, stale overwrite, private data loss/leakage. High: hidden
context warnings, inconsistent valid-paraphrase rejection, ineffective cancel,
resource-policy violation. Cosmetic wording is lower priority. Do not use an
aggregate “quality score” to bury a critical failure.
