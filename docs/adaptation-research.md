# How prose actually becomes a comic

Research gathered before redesigning the board pipeline, because the first
version of it was invented rather than learned. Every number below comes from
comics craft practice, adaptation writing, or published work on character
extraction from literature. Kept here so the design is answerable to
something other than my own guesses.

## Deciding how long it should be

*What prior art exists for turning prose/scripts into comics, storyboards, or illustrated sequences  -  what stages does a good pipeline have, how do existing systems handle character consistency, length, segmentation and layout, and where does everyone fail?*

Confidence: **high**

**The field has converged on a 5-7 stage modular pipeline, and every recent system that works well makes the intermediate representations explicit and editable rather than generating pages end-to-end.**

MangaFlow (arXiv 2605.28173) decomposes into six sequential stages: planning → grounding → layout construction → reference-conditioned rendering → composition → text placement. TaleCrafter (arXiv 2305.18247) uses four: Story-to-Prompt (S2P) → Text-to-Layout (T2L) → Controllable Text-to-Image (C-T2I) → Image-to-Video. StoryAgent (arXiv 2411.04925) uses specialized agents for story design, storyboard generation, generation, coordination, and evaluation. MangaFlow's explicit thesis is that prior generative models 'entangle these factors in a single visual output, limiting precise control'  -  so it exposes panel geometry, references, and text placement as user-editable intermediate variables.

*What it means here:* Build the tool as a staged pipeline with a durable intermediate document (a JSON storyboard: scenes → panels → {shot, characters present, action, dialogue, layout box}). Do not go prose → page image in one model call. The intermediate is both the control surface and the thing you can cache, diff, and let the user fix.

**The single highest-leverage stage is prompt construction, not image generation  -  an LLM that rewrites a story fragment into a self-contained visual caption beats feeding it the raw text by a wide margin.**

'LLMs Behind the Scenes: Enabling Narrative Scene Illustration' (arXiv 2509.22940) compared four ways to turn a fragment into an image prompt. The LLM-generated Caption (instructed to 'elaborately describe the most salient way to visualize the fragment' and 'completely specify all the information the illustrator needs') won 78.1% vs. the raw isolated fragment, 74.7% vs. fragment-plus-full-story-context, and 72.5% vs. a fragment rewritten only to resolve implicit references. Inter-annotator agreement showed scene descriptions influenced human judgment more than the choice of image generator (κᵤ = 0.483 for varying scene descriptions vs. 0.364 for varying image generators).

*What it means here:* Spend engineering effort on a Caption/visual-brief stage that resolves coreference, injects entity appearance from a character bible, and states composition explicitly. This buys more quality than upgrading the image model, and it is cheap. Never pass narrative prose directly to the renderer.

**Scene segmentation is best done as a bracketing/annotation task over the original text, and real stories chunk to roughly 4 illustratable fragments per short story.**

The same paper defines a scene as 'an abstract unit of a story that can be distinctly illustrated by a single image' and the aligned text as a 'fragment.' They prompt the LLM to 'insert brackets into the given story text to annotate the beginning and end of each fragment'  -  reproducing the text with delimiters rather than emitting a summary. On 50 ROCStories this gave 206 fragments (avg 4.12 per story) with ~96% judged correctly sized by humans. Other pipelines segment on narrative structure instead: some segment to three temporal frames (beginning/middle/end) following Freytag's pyramid.

*What it means here:* Segment by asking the model to bracket spans of the source text, not to summarize into panels  -  it makes segmentation auditable, keeps every panel traceable to its sentences, and guards against the model inventing or dropping beats. ~96% correct-sizing is a realistic accuracy ceiling, so expose a merge/split control.

**Character consistency has four distinct technique families with sharply different cost/quality profiles, and the current practical winner is hard pixel-level reference conditioning from a pre-generated character sheet.**

Practitioner comparison: (1) seed locking  -  breaks as soon as the prompt changes pose ('the moment you add sitting down... the seed lock stops helping'); (2) per-character fine-tuned adapters (DreamBooth/Textual Inversion/LoRA)  -  'effective but slow, expensive, and brittle  -  you are training a new adapter for every character'; (3) chained multi-image prompting off previous panels  -  'regress to the mean face after a few hops'; (4) reference-sheet conditioning (e.g. FLUX.1-Kontext) where the model treats the sheet 'as constraint, not suggestion.' Research analogues: StoryDiffusion's consistent self-attention across the batch, DiffSensei's MLLM identity adapter with masked cross-attention, MangaFlow's 'story section memory' linking section descriptions to character/scene/object references for reuse, SEAM's shot entity-attribute memory.

*What it means here:* Generate a character sheet once per character as an explicit pipeline artifact, then condition every panel on it. Treat it as a stage output, not a prompt string. Avoid per-character training in a product pipeline  -  the latency and cost per new character kills the UX.

**There is a fundamental identity/variation trade-off: the harder you lock a character, the more the output degenerates into pasted cutouts  -  and benchmarks now measure this explicitly because methods game it.**

ViStoryBench (arXiv 2505.24862) includes a dedicated Copy-Paste detection metric, and reports that a naive Copy-Paste baseline (directly placing reference images into shots) 'achieves optimal metrics across most dimensions but fails alignment scoring.' DiffSensei's own reported limitation: even trained on multiple appearances of a character, 'generated portraits often tend to rigidly follow the pixel distribution of the input character image, resulting in limited variations in appearance, pose, and motion based on the text input.' Reviewers of AI comics independently complain of 'dull, repetitive backgrounds' and slideshow-like compositions.

*What it means here:* Any consistency mechanism you ship must be evaluated against a paired variation metric, or you will optimize into an unreadable comic where every panel is the same bust shot on the same background. Consider deliberately varying shot type/camera/background per panel as a separate planning field that the renderer must honor.

**Multi-character panels are the reliable breaking point  -  getting the right number of the right people on stage is a measured, unsolved failure.**

ViStoryBench defines Onstage Character Count Matching (OCCM) as a top-level metric precisely because systems get it wrong. GPT-4o leads at 93.5% OCCM and 3.67/4.0 prompt alignment but underperforms on visual quality (IS 9.02, Aesthetic 5.49); dedicated story-image models score far worse on alignment (SEED-Story and TheaterGen < 2.0). Commercial platforms are wildly uneven  -  MorphicStudio hits CSD style consistency 0.653 while MOKI collapses on character consistency at CIDS 0.214. Independent tool reviews say the same thing in plain language: 'complex multi-character action panels less reliable than dialogue scenes' and action sequences are 'consistently unreliable across platforms.'

*What it means here:* Cap characters per panel in the planner (2, occasionally 3), and prefer dialogue/reaction framing over crowded action. If a beat needs four people, split it into panels. Validate generated panels against the planned character count before accepting them.

### Rules worth implementing

- Pipeline stages to implement, in this order (consensus across MangaFlow, TaleCrafter, StoryAgent): 1. Story analysis / entity extraction (build a character + setting bible). 2. Beat segmentation (bracket spans of the source prose). 3. Panel planning (shot type, characters present, action, dialogue, emphasis weight). 4. Character sheet generation (once per character, reused forever). 5. Layout construction (panel geometry from emphasis weights + balloon area). 6. Reference-conditioned panel rendering. 7. Composition/assembly. 8. Text/typography layer + validation. Keep every stage's output as an editable artifact.
- Segment by bracketing the original text, not summarizing: prompt the LLM to reproduce the story with delimiters marking each illustratable fragment. Expect ~96% correctly-sized fragments and ~4 fragments per short story; always expose merge/split.
- Always insert a Caption stage between segmentation and rendering: rewrite each fragment into a self-contained visual brief that resolves pronouns, injects character appearance from the bible, and names the composition. This beat raw fragments 78.1% of the time  -  bigger gain than any image-model upgrade.
- Default pacing: 6 panels per page. Use 5-7 for standard, 2-4 for slow/literary, 8-10 for dense action or comedy, never more than 10 (reading order collapses). Action sequences: 3-4 panels per page. Splash pages: 1-2 per 22 pages, hard cap 4.
- Hard text limits as validators: max ~25 words per balloon, max ~50 words per panel across all balloons and captions. If a beat exceeds it, split into another panel  -  do not shrink the type.
- Layout defaults: 5-6 panel base grid, ~7:10 page proportion, gutters 0.2in standard (0.05in fast action, 0.4in contemplative), gutters constant within a page. Hold the base grid on ~80% of pages, break it on ~20% at reveals.
- Compute reading order explicitly (Western Z-path: top-left → top-right → sweep down-left → right; manga mirrored; webtoon = single vertical column, one beat per screen) and validate balloon ordering against it. Webtoon conversion expands a page ~4-6×.
- Character consistency: generate a character reference sheet once, condition every panel on it as hard pixel-level reference (FLUX.1-Kontext-style), and re-anchor to the sheet at least every 10 panels. Do not chain panel N off panel N-1  -  references degrade to a mean face within a few hops. Avoid per-character LoRA training in a product path.
- Cap planned characters per panel at 2 (occasionally 3) and prefer dialogue/reaction framing over crowded action  -  multi-character action is the measured breaking point of every system tested.
- Guard against the copy-paste degenerate: pair every consistency metric with a variation metric, and make shot type / camera angle / background an explicit planner field the renderer must vary. Otherwise you ship a readable-but-lifeless slideshow.
- Render dialogue as a compositing layer over the art  -  have the image model produce empty bubbles or marked regions, then typeset with real fonts. Never let the diffusion model render the words.
- Budget balloon area during planning, before layout: dialogue length must be an input to panel composition, since the classic failure is text changing after the space was allocated.
- Add two cheap automated validators before accepting a page: face-occlusion check (no balloon over a speaker's face) and reading-order check. These catch the failures reviewers notice first.
- Run a second global refinement pass over the full sequence after all panels exist (Story-Adapter's iterative paradigm, ~21.7% aFID improvement) rather than trying to get each panel right in one shot.
- For evaluation, use a criteria-driven VLM judge rather than raw similarity scores  -  LLM-generated criteria sets predicted human preference at ~70% vs. 59% for baseline raters.

<details><summary>Sources</summary>

- https://arxiv.org/abs/2605.28173  -  MangaFlow: End-to-End Agentic Framework for Controllable Story to Manga Generation (6-stage pipeline, story section memory, editable layout)
- https://arxiv.org/abs/2305.18247  -  TaleCrafter: Interactive Story Visualization with Multiple Characters (S2P → T2L → C-T2I → I2V)
- https://arxiv.org/html/2505.24862v4  -  ViStoryBench: Comprehensive Benchmark Suite for Story Visualization (12 metrics, CIDS/OCCM/copy-paste, commercial + open-source rankings)
- https://arxiv.org/html/2509.22940  -  LLMs Behind the Scenes: Enabling Narrative Scene Illustration (fragment bracketing, caption win rates, SceneIllustrations dataset)
- https://arxiv.org/abs/2410.06244  -  Story-Iter / Story-Adapter: Training-free Iterative Paradigm for Long Story Visualization (100 frames, cumulative error, 21.71% aFID)
- https://arxiv.org/abs/2412.07589  -  DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation (MangaZero, masked cross-attention, rigidity limitation)
- https://arxiv.org/abs/2412.19303  -  MangaDiffusion: Manga Generation via Layout-controllable Diffusion (panel count control, Manga109Story)
- https://arxiv.org/pdf/2101.11111  -  Automatic Comic Generation with Stylistic Multi-page Layouts and Emotion-driven Text Balloon Generation
- https://www.ying-cao.com/projects/stylistic_layout/files/layout_paper.pdf  -  Cao, Chan, Lau: Automatic Stylistic Manga Layout
- https://arxiv.org/pdf/1804.05490  -  A survey of comics research in computer science
- https://arxiv.org/pdf/2409.09502  -  One Missing Piece in Vision and Language: A Survey on Comics Understanding
- https://arxiv.org/html/2411.04925v2  -  StoryAgent: Customized Storytelling Video Generation via Multi-Agent Collaboration

</details>

## Covering the story without skipping it

*How do adapters turn prose into panels without skipping the story? (beat analysis, must-show vs. implied, adaptation failure modes, coverage verification, interiority handling)  -  for a tool that currently compresses any story into a fixed small number of panels.*

Confidence: **high**

**A beat is defined by change, not by sentence or paragraph  -  the operative test is "what just changed?"**

Across theatre and screenwriting, a beat is "the smallest unit of action," an exchange of action/reaction that shifts something. Beat changes are triggered by four concrete, detectable events: (1) a character enters or exits (the classic "French scene"  -  the easiest beat boundary to find), (2) the situation shifts (topic change, a discovery, an external interruption), (3) a character changes tactic while pursuing the same goal (guilt-trip → beg → threaten = three beats), (4) time or place jumps. McKee's stricter version operates at scene level: a scene must turn a value-charged condition; "if nothing changes, nothing happened  -  it is a non-event."

*What it means here:* Beat segmentation must be a distinct extraction pass over the source with its own detectors (entrance/exit, topic shift, discovery/reversal, tactic change, time/place jump), producing a beat list whose length is a property of the story. Splitting on paragraphs or sentence counts is the root bug: prose paragraph boundaries and beat boundaries are uncorrelated.

**Panel count is a derived quantity, not an input  -  the standard pipeline is source → beat list → panel allocation, and a fixed panel budget inverts it**

Practitioner guidance for adaptation is explicit about ordering: "Source material arrives with its events already in order, so the instinct is to beat it out immediately. Do the reverse. List what actually happens in the book first, then find the beats hiding inside it, then build the outline." The comics unit of currency is the moment: McCloud's first of five creator choices is Choice of Moment  -  which moments to depict as panels and which to leave to closure in the gutter. Density conventions: roughly 4-7 beats per scene (~3 beats per script page); American comics 5-6 panels/page (3-9 acceptable), manga 3-4; no more than four major story points per page.

*What it means here:* Refactor so the tool computes N = number of beats extracted, then allocates panels ≥ N. If a panel cap exists, compare it to N and report the deficit explicitly rather than silently compressing. A fixed small panel count is mathematically a fixed story-point budget, so any source richer than that budget loses material by construction.

**Panels are hard-capacity containers, which is the physical reason compression destroys content rather than condensing it**

Hard limits from comics practice: max ~35 words per panel (25 is much better; manga 25 max, ideally ≤12), ~200 words per page total including dialogue, captions and SFX, and no more than four major story points per page. Dialogue in a panel typically reduces to one or two lines.

*What it means here:* Give each panel an explicit word budget (target 25, hard cap 35) covering caption + dialogue + SFX combined. When a beat's text exceeds budget, the correct operation is to split the beat across more panels  -  never to truncate it. Budget overflow is a reliable automated signal that panel count is too low.

**Prose does not compress uniformly into panels  -  physical action expands while summary and exposition contract**

John Rozum (comics writer, on adapting screenplays): a single paragraph of screenplay action  -  gunfire, a window breaking  -  "requires a minimum of three panels, and not small panels either. It could easily be the entire page." Meanwhile novelistic exposition, interior rumination and multi-scene setups collapse hard: the canonical film example is a book detailing bomb planning, placement and detonation across several scenes where the adaptation shows only the explosion.

*What it means here:* Classify each beat by type before allocating: physical action / reversal → 2-4 panels; dialogue exchange → 1 panel per tactic change; exposition or summary → 1 panel or fold into a caption; travel, arrivals, departures → 0 panels. A uniform panels-per-beat constant will simultaneously starve action and bloat exposition.

**What must be shown is determined by two tests  -  value change and causal necessity  -  plus the genre's obligatory scenes**

(a) Value test (McKee): a beat that turns a value charge (safe→unsafe, together→apart, hope→despair) is mandatory; a beat that merely conveys information is a merge candidate. (b) Causal test (Parker/Stone's "but/therefore" rule, from their 2011 NYU class): every retained beat must join its neighbour with "but" or "therefore"; if "and then" fits between two beats, the causal link is missing. (c) Obligatory scenes (scène à faire  -  "the scene that must be made"): the encounter the preceding action inevitably tends toward, the payoff of a planted setup, the genre-promised climax. Standard compressions: "get in late, get out early" (cut hellos and goodbyes), merge similar characters, combine several source scenes into one, cut any scene that does not advance the story.

*What it means here:* Three implementable gates. Tag each beat with (value_turned: bool), (causal_role: cause/consequence/neither), (obligatory: payoff-of-setup / promised-climax). Never drop a beat that is obligatory or that is the sole causal link between two retained beats. A dropped setup whose payoff survives  -  or vice versa  -  is a coverage bug the tool should detect and report, not a stylistic choice.

**The dominant adaptation failure mode is exactly the tool's current behavior: a fixed budget applied to unbounded source, producing correct density at the front and collapse at the back**

Serialized adaptation supplies the clearest documented cases  -  ReLIFE attempted ~220 chapters in 17 episodes; The Promised Neverland season two "rushed through arcs, skipped entire storylines"; Tokyo Ghoul is the standard example of bungling its source. The described symptom is invariant: "a final arc that compresses major story beats into too few episodes, sacrificing character development and emotional weight." The mechanism is a fixed container met by a source that exceeds it, so the overflow lands entirely at the end. Front-loading has an additional cause specific to automated extraction: openings in prose are concrete and scene-rendered (setting, character introductions, physical description) and therefore yield panel-able material easily, while middles are frequently narrated in summary and causal linkage, yielding fewer surface cues.

*What it means here:* Measure and enforce positional coverage. Bucket the source into quartiles (or acts) by word offset, compute panels-per-1000-source-words per bucket, and flag variance above a threshold. Expect and correct for act-1 over-sampling; the naive extractor's bias is structural, not incidental.

### Rules worth implementing

- Two passes, always: pass 1 extracts a beat list from the source; pass 2 allocates panels to beats. Never map source spans (paragraphs, sentences, chunks) directly to panels  -  that produces an illustrated novel, not an adaptation.
- Panel count is an output, not an input. Compute N = beats extracted. Target panels ≥ N. If a hard cap exists and cap < N, emit an explicit coverage deficit report naming which beats were dropped and where in the source they sit.
- Detect beat boundaries with five signals: a character enters or exits; the topic, place, or time changes; a discovery or reversal occurs; a character changes tactic toward the same goal; an external event interrupts. Validate each candidate with "what just changed?"  -  if nothing changed, it is not a beat.
- Density calibration: roughly 4-7 beats per scene; 5-6 panels per page (3-9 acceptable, 9 reads claustrophobic); manga 3-4 per page; no more than 4 major story points per page. Vertical scroll: 40-80 panels per episode.
- Word budget per panel: target 25 words, hard cap 35, counting caption + dialogue + SFX together; ~200 words per page. Overflow means split the beat into more panels  -  never truncate the beat.
- Allocate panels by beat type, not by a constant: physical action or reversal 2-4 panels (one paragraph of prose action is a minimum of 3 panels); dialogue 1 panel per tactic change; exposition 1 panel or a caption; travel, arrivals, departures and greetings 0 panels ("get in late, get out early").
- Two keep/cut gates. Value gate: does this beat turn a value (safe/unsafe, together/apart, hope/despair)? Value-turning beats are mandatory; information-only beats are merge candidates. Causal gate: join each retained beat to the next with "but" or "therefore"; any pair where only "and then" fits means a causal link was dropped  -  recover the beat between them.
- Mark obligatory beats and never drop them: the payoff of any planted setup, the promised climax, the inevitable confrontation (scène à faire). Treat a surviving setup with a dropped payoff (or the reverse) as a hard coverage error, reported not silently accepted.
- When over budget, compress by widening the transition, not by deleting. Promote runs of moment-to-moment gaps to action-to-action, then to scene-to-scene; the beat survives as an implied step across a wider gutter that the reader closes. Deleting a beat outright is the last resort.
- Enforce positional coverage: bucket the source by word offset into quartiles or acts, compute panels-per-1000-source-words per bucket, and flag variance beyond a threshold. Default target allocation 25% / 50% / 25% across acts 1/2/3 (act 2 gets half the cards on a 40-card board), never uniform-per-word.
- Expect and correct front-loading. Openings are scene-rendered and concrete so they over-generate panels; middles are often summarized so they under-generate. A naive extractor over-samples act 1 by construction  -  check for it every run.
- Route interiority by rule, not by default: inferable from face or posture → performance, zero words; tied to a concrete image or memory → subjective rendering or aspect-to-aspect panels; reasoning, irony, retrospection, or distinctive narrative voice → italicized caption box (not thought balloon); restates what the image already shows → cut. Charge caption words against the panel's 25-word budget.
- Reserve reaction beats. After every reversal or reveal, allocate at least one wordless "beat panel"  -  a held, silent panel of the character absorbing it. An event-only extractor drops exactly these, and they carry the emotion.
- Ship a coverage pass as a pipeline stage: reverse-outline the generated panels back into concrete events (what happened, not what it meant), diff against the source beat list, and report (a) unmatched source beats with offsets, (b) per-act panel distribution, (c) every "and then" join between consecutive panels, (d) any panel over word budget.
- Two-artifact discipline mirrors the beat-sheet/outline split: keep a ~15-line structural summary (the diagnostic  -  does this read as a story with a working middle?) separate from the full panel list (the work plan). Validate the summary first; a story that fails at 15 lines will not be saved by more panels.
- For scroll formats, place a strong beat at the 30-40% mark of each output unit  -  that is the documented drop-off point, and a deliberate second hook there moves retention from 60-70% to 85-95%.

<details><summary>Sources</summary>

- https://dramatics.org/beat-analysis/  -  beat definition, the four beat-change triggers, French scenes, tactic changes
- https://www.goodreads.com/notes/40389794-story  -  Robert McKee, Story: beat as action/reaction exchange, value charge, "if nothing changes, nothing happened", scene turning points, sequences of 2-5 scenes
- https://www.studiobinder.com/blog/story-beat-in-screenplay/  -  story beats, beat density per scene and per page
- https://www.masterclass.com/articles/what-is-a-beat-in-screenwriting  -  beats and beat sheets, ~4-7 beats per scene
- https://buzzdixon.com/home/writing-2/a-few-rough-rules-of-thumb-for-writing-comics-graphic-novels  -  5-6 panels/page American, 3-4 manga, 35 words/panel max (25 better), 200 words/page, no more than 4 major story points per page
- http://johnrozum.blogspot.com/2016/03/screenplays-into-comics.html  -  action expansion (one prose paragraph = minimum 3 panels), page budgeting, 20-22 page units, scene start/end at page boundaries
- https://understandingcomics177.wordpress.com/about/1-2/2-2/  -  McCloud's six panel transitions and closure/the gutter
- https://www.ucreative.com/articles/comicstheory1/  -  McCloud's five choices: Moment, Frame, Image, Word, Flow; Choice of Moment as what to depict and what to omit
- https://en.wikipedia.org/wiki/Making_Comics  -  Making Comics, clarity and the five choices
- https://tvtropes.org/pmwiki/pmwiki.php/Main/BeatPanel  -  the silent beat panel as a deliberate device
- https://www.cbr.com/compressed-storytelling-versus-decompressed-storytelling-pros-and-cons/  -  compressed vs decompressed pacing, panels per unit of story time
- https://comicbookglossary.wordpress.com/compressed-and-decompressed-storytelling/  -  decompression defined as multiple panels doing one panel's work

</details>

## Pages, panels, and how much text fits

*How does professional comic and manga adaptation decide LENGTH from source prose  -  pages, panels, words-per-page ratios, format differences, and what breaks when the page count is too short?*

Confidence: **medium**

**The craft does not decide page count from word count directly  -  it decides from BEATS, and word count is only used afterward as a sanity check on whether the chosen page count is survivable.**

The consistent professional method is: break prose into beats → assign one beat per panel → group panels into pages at ~5 panels/page. Comicory's outline method makes this explicit as 'roughly one line per page of finished comic' (a 24-page short = ~24 outline lines; a 160-page graphic novel = ~160 lines), where each outline line converts to 1-3 panels of script. John Rozum (pro writer, Xombi/Scooby-Doo) states the atomic rule as 'in general, one action per panel.' Comicory: 'A panel handles one beat. If your description has two beats  -  Maya enters the room and then sees the body  -  split it into two panels.'

*What it means here:* The tool's primary pipeline should be prose → beat list → panels → pages, NOT prose → word count → pages. Word count enters only as a downstream validator. A beat extractor is the core component; the page count is an output of it, not an input to it.

**There is a hard, widely agreed text budget per page and per panel, and it functions as the real constraint on how short an adaptation can be.**

Buzz Dixon (pro comics writer, G.I. Joe/Thundercats): American comics ≤35 words/panel (25 'much, much better'), ≤200 words/page total including captions, dialogue, titles and SFX; manga ≤25 words/panel (12 or fewer ideal), ~60 words/page. Karen Wasson's published-GN survey recommends ≤25 words/panel and 125 words/page. General cited range for a comic page is 70-200 words. Big Red Hair (Shannon/Dean Hale's studio) adds 'wordy panels almost always slow down the reader.'

*What it means here:* Implement a hard validator: required_on_page_words / target_pages must stay under 200 (Western) or 60 (manga). Exceeding it is the single most reliable machine-detectable signal that the requested page count is too short  -  and it is detectable BEFORE generating any art.

**One rigorously measured professional prose-to-comic adaptation gives a full set of hard conversion ratios, and it lands almost exactly on the craft rules of thumb.**

Jim Ottaviani's adaptation of E.O. Wilson's Naturalist: 111,638 prose words → 226 pages, 1,223 panels, 28,906 words appearing on the page, from a 70,320-word script. Derived: 494 prose words per comic page; 91.3 prose words per panel; 5.41 panels per page; 127.9 on-page words per page; 23.6 on-page words per panel; 25.9% text retention; 311 script words per comic page; 57.5 script words per panel. Ottaviani computes that images carried ~84,000 words of the original, i.e. ~68 words per panel replaced by image.

*What it means here:* These are directly implementable defaults. Note the cross-validation: 23.6 words/panel sits right on Dixon's 'ideal 25', and 128 words/page is 64% of the 200 ceiling. An award-winning adaptation ran comfortably inside budget  -  a tool should target ~60-70% of the ceiling, not 100%, because hitting the ceiling is already the failure mode.

**Prose compresses at wildly different rates depending on what KIND of prose it is, so a single global words-per-panel constant will mis-size any specific passage.**

Naturalist retained only 25.9% of its text  -  but that is dense expository nonfiction. The compression is not uniform: dialogue transfers close to 1:1 (minus speech tags and hedging), physical action converts at roughly one discrete action per panel, and interiority/exposition either dies or survives as a ≤25-word caption. The adaptation literature is explicit that adaptations 'remove much of the interiority of the novel' and that in light-novel-to-manga work 'internal monologues vanish or are reduced to brief captions' while 'entire chapters may be summarized into a single page.'

*What it means here:* Segment the source prose by type before estimating. A dialogue-heavy scene needs FAR more pages per thousand words than a descriptive one. Suggested per-type rates: dialogue ~40-60 prose words/panel, action/description ~90-120, exposition/interiority ~200+ (most of it discarded). Weight the page estimate by the scene's composition.

**Panels per page is 4-6 as the professional standard with ~5 as the mean, but the variation by scene type is governed by two independent variables that are commonly conflated: panel COUNT tracks the number of beats, panel SIZE tracks duration and weight.**

Big Red Hair: 'the average number of panels per page is usually five.' Working range 1-9; most sources converge on 4-6. By scene type the sources appear to conflict  -  Comicory lists action-heavy pages at 6-9 panels and emotional beats at 2-3 or a splash, while Rozum says action scenes get 'fewer larger panels per page, so they read faster' and dialogue scenes get 'more panels' because they need less setting ('often like talking heads'). Both are correct at different scales: a fight has many beats (many panels) but each impact gets a large panel; a single dramatic action gets one big panel. Pipeline Artists: 'a page shouldn't have more than nine panels,' and avoid consecutive nine-panel pages. Dixon notes >6 panels/page 'tends to make the story a bit more cramped and claustrophobic' (Watchmen's 9-grid being deliberate).

*What it means here:* Model panel count and panel area separately. Set panels/page from beat density; set panel size from emotional weight. Practical defaults: dialogue 5-7 panels/page, general narrative 4-6, choreographed action 6-9 (small panels, few words), single climactic action or emotional beat 1-3 panels/page or a splash. Cap at 9; flag consecutive 8+ pages as cramped.

**Manga spends more panels per page AND more panels per story beat than Western comics, so it advances roughly half as much plot per page  -  this is the core structural difference, not a stylistic one.**

McCloud's transition census: mainstream American comics run ~65% action-to-action, ~20% subject-to-subject, ~15% scene-to-scene, with moment-to-moment and aspect-to-aspect essentially absent. Japanese manga uses subject-to-subject and moment-to-moment as often as action-to-action and has a substantial aspect-to-aspect share  -  transitions that consume panels without advancing plot. McCloud: 'Comics is an art of intervals.' Panel density per page in manga is cited at 6-8 (Comicory) vs Dixon's 3-4; the honest reading is that manga panel count per page is equal-or-higher than Western while its word budget is ~3.3x lower (60 vs 200 words/page).

*What it means here:* Do not port a Western pages-per-beat constant to manga. Use ~4-6 beats/page for Western print, ~2-3 beats/page for manga (more panels, ~1.5-2 panels per beat), and expect a manga adaptation of the same prose to run roughly 1.5-2x the page count of a Western one. Dixon's 3-4 figure appears to describe word-bearing panels; treat 5-8 as the physical panel count.

### Rules worth implementing

- PRIMARY PIPELINE: prose → beats → panels → pages. One beat = one panel. One outline line = one finished page. Never estimate pages directly from word count; use word count only to validate the result.
- PANELS PER PAGE: default 5. Working range 4-6. Dialogue 5-7, general narrative 4-6, choreographed action 6-9 (small panels, minimal text), single climactic action or emotional beat 1-3 or a splash. Hard cap 9; flag two or more consecutive 8+ panel pages as cramped.
- BEATS PER PAGE: Western print ~5 (range 4-6). Manga ~2-3 (more panels but ~1.5-2 panels per beat). Webtoon episode ~30-40 beats across ~60 panels.
- TEXT CEILING (the hard constraint): Western ≤35 words/panel (target 25) and ≤200 words/page (target 125-130). Manga ≤25 words/panel (target 12) and ~60 words/page. Target 60-70% of the ceiling, not 100%  -  the measured Naturalist adaptation ran at 128 words/page and 23.6 words/panel.
- PROSE-WORDS-TO-COMIC-PAGE: Western graphic novel ≈ 400-500 prose words per finished page (measured: 494 in Naturalist). Manga ≈ 150-225 prose words per page (derived from the industry rate of ~2 manga volumes per light-novel volume). Expect manga to run 1.5-2x the page count of a Western adaptation of identical prose.
- PROSE-WORDS-TO-PANEL: ≈90 prose words per panel overall (measured: 91.3). But segment by type  -  dialogue ~40-60 words/panel, action/description ~90-120, exposition/interiority ~200+ with most discarded. A dialogue-heavy chapter needs far more pages per thousand words than a descriptive one.
- TEXT RETENTION: expect ~25-30% of source words to survive onto the page (measured: 25.9%). Images carry the remainder  -  roughly 68 words of prose replaced per panel.
- SCRIPT SIZING: generate ~300 script words per comic page and ~55-60 per panel, of which ~24 are on-page text (balloons + captions) and ~35 are art direction. A panel description under ~35 words is under-directed.
- FORMAT PAGE BUDGETS (fixed by production, not story): floppy 22 story pages (32 physical); graphic novel 100-180; manga weekly chapter 17-20; tankōbon 180-220 = 8-10 chapters; Franco-Belgian album 46-62 at 8-12 panels/page; webtoon episode 40-80 panels (target 60).
- STRUCTURE SPLIT: Act 1 ~20-25% of pages, Act 2 ~50-65%, Act 3 ~10-30%. For a 20-22 page issue: 2-3 pages setup, 10-12 rising action, 3-4 showdown, 1 strong final page. Webtoon episode: hook panels 1-10, development 11-60, cliffhanger final 10, with a re-hook at the 30-40% mark (~panel 20 of 60).
- OVER-BUDGET DETECTOR: if (required on-page words ÷ target pages) > 200 Western or > 60 manga, the page count is too short. Report it; do not silently cram. This check runs before any art is generated.
- CUT ORDER when over budget (strict priority queue): 1) subplots that do not touch the climax  -  if removing a thread does not change the ending, it goes first; 2) repeated/restated motivation beats; 3) travel and transitions  -  collapse a journey to one establishing shot; 4) interior monologue, converted to facial performance or cut; 5) tertiary characters who deliver one fact  -  reassign that fact to an existing character. Budget ~33% of source material cut as normal.
- NEVER CUT (protect these even under compression): one establishing panel per new location, and one large or silent panel per emotional climax. These are the cheapest to cut and the most costly to lose  -  cutting them is what makes a compressed adaptation read as plot-complete but disorienting and flat.
- ILLUSTRATED-PROSE GATE: if narration/caption words exceed ~40% of on-page words, the output has stopped being sequential art. The art must do independent narrative work, not document the prose. Fix by cutting beats or adding pages  -  never by shrinking the art.
- WEBTOON CONVERSION: 1 manga page ≈ 5-7 webtoon panels; 1-3 panels per screen height; 60-panel episode ≈ 8,000-10,000px at 800px width (hard practical max 15,000px), 5-8 minutes reading. Emit gutter height as a first-class pacing parameter: 50-100px for fast beats, 300-500px for emotional pauses.
- TRANSITION MIX by tradition (McCloud's census): Western ≈ 65% action-to-action, 20% subject-to-subject, 15% scene-to-scene, near-zero moment-to-moment and aspect-to-aspect. Manga adds substantial moment-to-moment and aspect-to-aspect  -  panels that consume space without advancing plot. Setting this mix is what makes output read as manga rather than a Western comic with manga art.
- PANEL COUNT AND PANEL SIZE ARE INDEPENDENT: count tracks number of beats, size tracks duration and emotional weight. A fight is many beats (many panels) with large panels on impacts. Resolving these two separately dissolves the apparent contradiction between 'action needs fewer, bigger panels' and 'action pages run 6-9 panels'.

<details><summary>Sources</summary>

- https://buzzdixon.com/home/writing-2/a-few-rough-rules-of-thumb-for-writing-comics-graphic-novels  -  Buzz Dixon (professional comics writer, G.I. Joe/Thundercats): words/panel, words/page, panels/page for both American comics and manga. Highest-value single source for the text-budget numbers.
- https://www.goodreads.com/author/show/2402.Jim_Ottaviani/blog?page=3  -  Jim Ottaviani, 'NATURALIST: How much is a picture worth?' The only rigorously measured prose-to-comic adaptation found: 111,638 prose words → 226 pages, 1,223 panels, 28,906 on-page words, 70,320-word script.
- http://johnrozum.blogspot.com/2016/03/screenplays-into-comics.html  -  John Rozum (professional comics writer, Xombi, Scooby-Doo): 'one action per panel', action vs dialogue panel density, 22-page comic ≈ 40-page script, why no fixed page-conversion ratio exists.
- https://www.bigredhair.com/about/teaching-comic-books/writing-guide  -  Big Red Hair (Shannon and Dean Hale's studio) comic writing guide: 5 panels/page average, 32-page floppy with 22-26 story pages, act-structure split, pacing and wordiness rules.
- https://shannonhale.com/extras/ez9tky4pvcyo713rgv6n83vmrgyli5  -  Shannon Hale on scripting a graphic novel: why page limits are set by illustrator time and printing cost, and why words crowd out illustration. Qualitative but from a working adapter of her own prose.
- https://pipelineartists.com/adapting-a-screenplay-into-a-graphic-novel-screenplay-to-script/  -  Working writer's adaptation account: Blowback = 110 GN pages from a 226-page script; ~100-page GN target; 9-panel hard cap; thumbnailing all pages to validate the page count.
- https://karenwasson.substack.com/p/graphic-novel-word-counts  -  Measured word counts of published graphic novels; Inked = ~22,000 words across 304 pages (~72 words/page); 25 words/panel and 125 words/page recommendations; middle-grade GN category ranges.
- https://www.screenweaver.ai/blog/graphic-novel-to-animated-film  -  The explicit cut-order hierarchy (subplots not touching the climax → repeated beats → travel/transitions → interior monologue → tertiary characters) and the ~33%-of-pages-do-not-survive figure. Note: SEO-style content site, so treat the numbers as indicative rather than authoritative, but the cut hierarchy matches other adaptation literature.
- https://www.comicory.com/blog/comic-outline-template  -  The one-outline-line-per-page heuristic (24-line short → 24 pages; 160-line outline → 160-page GN), act percentage splits, outline-line-to-panel conversion. SEO content site; corroborated in direction by practitioner sources.
- https://www.comicory.com/blog/paneling-in-comics  -  Panels-per-page by scene type (4-6 typical, 6-9 action, 2-3 emotional, 6-8 manga, 20-60 webtoon) and the 'one panel = one beat' rule. SEO content site.
- https://www.comicory.com/blog/webtoon-vs-manga  -  The 1 manga page ≈ 5-7 webtoon panels conversion, panel density comparison, ~19 manga pages/week, volume structure. SEO content site.
- https://www.comicory.com/blog/webtoon-episode-length  -  Webtoon episode metrics: 40-80 panels, 6,000-12,000px scroll, 5-8 min read, panel-indexed act structure, gutter spacing 50-100px vs 300-500px, mid-episode drop point.

</details>

## Finding the cast in the prose

*How do comic/film adaptation and computational literary analysis extract and FIX characters from prose  -  building a character bible, handling sparse descriptions, identifying characters computationally (NER + coreference), tiering one-off vs recurring characters, and what breaks when design is decided per-scene?*

Confidence: **high**

**The adaptation-side character bible is built in a strict order: harvest every textual description first, resolve gaps with the author, then design, then lock  -  before any page is drawn.**

A working novel-to-comic adapter describes compiling "every character and location description that I can find in the book" as step one; where the book is incomplete they contact the original author directly; the artist then produces designs that are "approved by the author" through revision rounds, and only then does the script go editorial review → author/licensor approval → pencil roughs → finished art. Gareth Hinds (Beowulf, The Odyssey, The Iliad, Macbeth) works script and designs "simultaneously, going back and forth" but says only once he has a first draft AND firm designs does he rough out the entire book.

*What it means here:* A prose-to-comic tool needs a two-phase architecture with a hard barrier: a whole-text extraction/design pass that produces a frozen character record, then a rendering pass that may only read that record. Design must never be a side effect of rendering a scene. The 'ask the author' step is the tool's gap-filling/inference step, and it should be surfaced as an explicit, reviewable decision rather than hidden.

**A model sheet has a stable, enumerable field set  -  it is effectively a schema, not a mood board.**

Standard contents: (1) turnaround with four canonical views (front, 3/4 front, profile, 3/4 back, sometimes back); (2) expression sheet covering the emotional range from multiple angles, with notes on how eyes/mouth change; (3) pose sheet including a neutral/T-pose, characteristic action poses, interaction poses, and silhouette outlines checked for readability; (4) props, drawn from multiple angles with material/texture notes; (5) color palette  -  base swatches for skin, hair, clothing, each with light/mid/dark shades, labeled with RGB/HEX/Pantone, plus the character under different lighting; (6) written annotations for proportions, distinctive features, personality, mannerisms, and speech quirks. Height/lineup charts sit alongside for relative scale.

*What it means here:* This is a directly copyable JSON schema for the tool's character record. Crucially, the fields are mostly things prose does NOT supply (3/4 view, hex codes, proportions, silhouette)  -  so the record is 90% inference from 10% text. The color field with explicit hex values is the highest-leverage one for downstream image generation consistency.

**The film-side analogue (character breakdown) is deliberately minimal and includes physical description ONLY when load-bearing.**

Standard breakdown fields are age range, profession, personality, emotional context, and conflict. Explicit professional guidance: "only communicate physical descriptors like 'slender and slight of frame' if they're important to the character." Example entry: "Lauren. 30s. Stand-up comedian. Larger than life personality. Naturally exuberant yet slightly awkward." Script breakdown separately tags every character in every scene into the schedule, distinguishing speaking roles from background.

*What it means here:* There are two different documents with two different jobs: a sparse *identity* record (who this is, invariant across the whole work) and a dense *appearance* record (what to draw). Prose reliably supplies the first and rarely the second. The tool should extract identity from text and generate appearance from identity, not attempt to extract appearance directly.

**When prose is silent on appearance, professional practice is to derive the look from non-visual constraints the text DOES supply, not to invent freely  -  the constraint is negative (must not contradict) plus derivational (must follow from role, era, class, occupation).**

Novelists deliberately underdescribe: Maugham observed readers "seldom form any exact image" from physical description, and the standard craft argument is that the less a reader must imagine, the less invested they are. Dickens's illustrator Phiz produced "imaginary images conceived by the illustrator" rather than mirror-image portrayals. Costume designers on period adaptations work as "psychologists, sociologists, researchers, and historians," deriving garments from period, class, and occupation rather than from any description in the book. Hinds researched Classical-period pottery paintings to derive designs for The Iliad.

*What it means here:* The inference step should be a constrained generator, not a free one. Structure it as: (a) collect hard textual constraints (explicit attributes, age, gender/pronouns, kinship, occupation, era, class, how other characters react to them); (b) generate appearance conditioned on those plus a world/period style reference; (c) validate the generated appearance against every textual mention for contradiction. Free-form 'imagine a character' prompting will violate constraints stated 200 pages away from the first appearance.

**Visual distinctness between characters is engineered deliberately, as a separate design constraint from individual fidelity.**

Hinds says his "first job" on The Iliad's large cast was making each captain look distinct  -  a unique shield and helmet design per captain. General comics/animation practice is the silhouette test: characters must be identifiable in filled-black outline alone. Height/lineup charts exist to guarantee relative differentiation. In practice designers differentiate by silhouette, then proportion, then color, then detail.

*What it means here:* The tool must optimize the cast jointly, not each character independently. Sample a character, then check it against all already-fixed characters for collision in silhouette/hair/build/palette; re-sample on collision. Reserve a distinct palette per major character and enforce global uniqueness. A per-character independent generator will produce three interchangeable brown-haired white men in a six-person cast.

**Not every named person gets a design  -  production tiers characters by speaking/agency, and only the top tier gets bespoke work.**

Film distinguishes principals and day players (anyone with a line is a day player, individually cast and contracted) from background actors/extras (non-speaking, the largest and lowest-paid group, cast by type not by identity). Animation mirrors this: hero characters get full model sheets; background characters are populated by repurposing existing rigs/designs with minor variations (shirt color, hat, hairstyle), and studios producing 3+ projects a year report 30-40% lower per-project cost from reusable asset libraries.

*What it means here:* Implement three tiers. Tier 1 (recurring, speaks, drives scenes): full character record, locked, reused everywhere. Tier 2 (named but one or two appearances): minimal record  -  silhouette, age, one distinguishing feature, palette slot. Tier 3 (crowd, 'a waiter', 'the crowd'): no record, drawn from a generic type library parameterized by era/setting. Spending Tier-1 effort on all named entities is the main cost blowout, and Vala's data shows the tail is enormous.

### Rules worth implementing

- Two hard phases, never interleaved: a whole-text CAST pass that produces frozen character records, then a RENDER pass that may read but never write them. Design decided during rendering is the root cause of every drift failure.
- Never trust an NER entity list as a cast list. Naive NER averages ~339 'characters' per novel; real counts are ~70-90 for a big 19th-century novel, with roughly 5 surface names per character (The Moonstone: 419 names, 78 characters).
- Resolve identity with a name graph plus explicit anti-edges. Block a merge when: inferred genders differ; the names share a surname but differ in first name; honorifics differ (Miss vs Mrs). Add an anti-edge when two names are joined by a conjunction, when one speaks and names the other in dialogue, or when both appear inside one quotation. These are cheap and catch the worst errors.
- Bias coreference toward precision. An unattached pronoun costs you one datum; a misattached pronoun assigns one character's attributes to another and silently corrupts the record. BookNLP's own default restricts pronoun coreference for exactly this reason.
- Rank the cast by mention count and expect a Zipf/Benford-shaped curve (verified across 45,939 books). Allocate design budget log-proportionally, not uniformly.
- Three tiers. Tier 1: has attributed dialogue or falls in the top ~80% of cumulative mentions -> full model sheet, locked, globally unique. Tier 2: >2 mentions -> minimal record (age band, build, one distinguishing feature, reserved palette slot). Tier 3: everything else -> generic type from a reusable library parameterized by era/setting. Mirrors principal / day player / extra.
- Detect unnamed role-characters ('the coachman', 'the Archbishop') by animacy and agency  -  does the noun phrase take character-like verbs or speak  -  not by NER. Gate this stage behind good coreference; it makes things worse when coreference is already failing.
- Character record schema, copied from model-sheet practice: identity (canonical name, all aliases, pronouns/referential gender, age band, role, tier); hard textual constraints (every explicit attribute with its source quote and offset); derived appearance (build, height relative to cast, hair, face, distinguishing marks); wardrobe per story phase; palette as explicit hex values for skin/hair/each garment; expression range; props; annotations. Keep the source quote for every hard constraint so contradictions are auditable.
- Separate what prose gives from what you invent, and mark each field with its provenance. Prose reliably gives identity, age, gender, kinship, class, occupation, era, and how others react; it rarely gives appearance. Generate appearance conditioned on the identity fields plus a world/period style reference, never free-form.
- Inference must be negatively constrained: after generating an appearance, re-scan every mention of that character across the whole text for contradiction (a stated eye color 200 pages later, a remark that she is the tallest in the room). Adapters do this by compiling all descriptions before designing  -  do the same, and treat a late contradiction as a regeneration trigger, not an exception.
- Optimize the cast jointly. After fixing each character, test against all already-fixed characters for silhouette, hair, build, and palette collision, and re-sample on collision. Enforce globally unique palette slots. Hinds's first job on The Iliad was giving each captain a unique shield and helmet  -  distinctness is a design requirement, not an emergent property.
- Apply the silhouette test as an automated check: if two Tier-1 characters are not distinguishable as filled black outlines, the design has failed regardless of how good each looks alone.
- Lock clothing, hair, and build, not just the face. Face-only conditioning drifts on costume and build, and readers track those too. Condition rendering on a generated reference image (the model sheet) plus a fixed attribute string, never on re-derived text alone.
- Add an automated check-and-correct pass: compare every rendered panel's character against the canonical reference, and propagate a correction to all panels, not just the current one. This is the animation 'checking' step, and the documented cause of off-model drift is precisely that this step gets skipped under time pressure.
- Compute per-book risk up front and escalate hard books to human review or a more expensive resolution path. Risk signals: first-person narration (narrator may be unnamed for hundreds of pages), multiple narrators, shared surnames across a family, high alias-per-character ratio, heavy nicknames, non-Anglo naming. NER difficulty is a property of the book, not the tool  -  a better model will not rescue it.
- Surface the cast list and the design decisions for confirmation before generating pages. The professional workflow has an explicit author-approval gate between design and art, precisely because a wrong design discovered at page 150 is unrecoverable.
- Persist character records across works/sessions. Hinds reused Odysseus and Agamemnon's designs from The Odyssey when drawing The Iliad; a registry keyed by canonical identity gives the same continuity and amortizes design cost (reusable asset libraries cut per-project cost 30-40%).

<details><summary>Sources</summary>

- https://mythicscribes.com/miscellaneous/adapting-novels-comics/  -  practitioner account of novel-to-comic adaptation: compiling all descriptions, author approval loop, full-script panel notes
- https://www.publishersweekly.com/pw/by-topic/childrens/childrens-authors/article/79322-q-a-with-gareth-hinds.html  -  Hinds on simultaneous script/design iteration, firm designs before roughing the book, cross-book design reuse
- https://www.cbcbooks.org/2019/03/12/from-the-sketchbook-gareth-hinds/ and https://yaledailynews.com/blog/2025/01/30/graphic-novelist-gareth-hinds-on-imagining-and-illustrating-the-classics/  -  research-driven design (Classical pottery), unique shield/helmet per captain for cast distinctness
- https://blog.cg-wire.com/character-sheet-animation/  -  model sheet field-by-field contents (turnaround, expressions, poses/silhouette, props, palette with RGB/HEX/Pantone, annotations)
- https://blog.cg-wire.com/background-characters-animation/  -  reusable background-character libraries vs hero designs
- https://en.wikipedia.org/wiki/Off-model and https://tvtropes.org/pmwiki/pmwiki.php/Main/OffModel  -  off-model drift, its causes (skipped checking, outsourcing, no distributed model sheet) and countermeasures
- https://www.studiobinder.com/blog/how-to-write-a-character-breakdown-for-a-script/  -  character breakdown fields; 'only include physical descriptors if important'
- https://www.backstage.com/magazine/article/acting-roles-75755/ and https://en.wikipedia.org/wiki/Extra_(acting)  -  principal / day player / background tiering by speaking role
- https://spectator.com/article/do-we-need-to-know-what-a-character-looks-like/  -  Maugham on description, Phiz illustrating Dickens as invention not transcription, sparse-description craft rationale
- https://jasna.org/publications-2/persuasions-online/volume-43-no-1/kim-sherman/  -  period costume design for Austen adaptations; design as world-building beyond garment reconstruction
- https://aclanthology.org/D15-1088/ (PDF read in full)  -  Vala, Jurgens, Piper & Ruths, 'Mr. Bennet, his coachman, and the Archbishop...': 8-stage pipeline, anti-edge merge prohibitions, animacy-verb bootstrapping for unnamed characters, F1 tables, 339-characters/novel NER baseline, P&P 73 vs 17, Moonstone 419 names / 78 characters
- https://github.com/booknlp/booknlp  -  BookNLP pipeline stages, output schemas (.tokens/.entities/.quotes/.book), F1 numbers (entity 88.2/90.0, coref 76.4/79.0, speaker attribution 86.4/89.9), precision-biased coref default

</details>
