import { useEffect, useId, useMemo, useRef, useState } from "react";
import { api, errDetail, errText, isCancelled, fmtDuration, newJobId, onEnginePreview, onEngineProgress, vaultUrl } from "../lib/api";
import type { AssistGoal, AssistResult, EngineProgress, ModelStatus, Recipe } from "../lib/types";
import { autoPick, estimateSeconds, humanDuration, QUALITY_LABEL, SHAPES, stepsFor, type Quality } from "../lib/presets";
import { editKindFor, isOutpaintFill, OUTPAINT_FILLS, parseOutpaintPadding, type EditKind } from "../lib/recipe";
import { exportItem, ImageDrop, JobProgress } from "./shared";
import MaskCanvas from "./MaskCanvas";
import SendTo, { type Destination } from "./SendTo";
import PromptProposal, { PromptGoal } from "./PromptProposal";
import { ANALYST_IDS } from "../lib/story";
import Loras from "./Loras";
import Adjust from "./Adjust";
import { loadPref, savePref } from "../lib/prefs";
import Field from "./Field";
import { Choice, LiveStatus, ResultStrip, RunError, useRunShortcut, useStatus, type Made } from "./creation";

export type Mode = "generate" | "edit";

export default function Studio({
  mode, models, notify, onProduced, images: controlledImages, onImagesChange,
  onSendToEdit, send, recipe, onRecipeUsed,
}: {
  mode: Mode;
  models: ModelStatus[];
  notify: (m: string, bad?: boolean) => void;
  onProduced: () => void;
  /**
   * Edit sources, owned by the parent. Generate and Edit are separate
   * component instances, so "Edit this" writing to local state put the image
   * into the wrong one -- it looked like nothing happened.
   */
  images?: string[];
  onImagesChange?: (ids: string[]) => void;
  /** Hand an image to the Edit tab and switch to it. */
  onSendToEdit?: (ids: string[]) => void;
  /** Carry a result into any other tab. */
  send?: (dest: Destination, ids: string[]) => void;
  /** Settings restored from a past run. */
  recipe?: Recipe | null;
  onRecipeUsed?: () => void;
}) {
  const wantTask = mode === "edit" ? "edit" : "text_to_image";

  // A model whose peak exceeds the ceiling is still offered when its resident
  // weight floor fits: reduced-memory mode targets exactly that gap. A model
  // whose weights alone exceed memory is never offered.
  const usable = useMemo(
    () => models.filter(
      (m) => m.installed && !m.hopeless && m.fit !== "broken"
        && m.tasks.includes(wantTask as never)
    ),
    [models, wantTask]
  );

  const assistantReady = useMemo(
    // A writer, specifically: any of them, since the engine uses the best one
    // installed. `.some` over every assist model was satisfied by the
    // picture-reader alone -- so the button looked ready and then failed with
    // "the prompt writer is not installed".
    () => models.some((m) => ANALYST_IDS.includes(m.id) && m.installed && !m.hopeless),
    [models]
  );
  const [goal, setGoal] = useState<AssistGoal>(() => loadPref("promptGoal", "enhance" as AssistGoal));
  useEffect(() => { savePref("promptGoal", goal); }, [goal]);

  const [advanced, setAdvanced] = useState(() => loadPref("advanced", false));
  const [modelId, setModelId] = useState<string>("");
  const [autoModel, setAutoModel] = useState(true);

  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [shape, setShape] = useState(() => loadPref(`shape:${mode}`, 0));
  const [quality, setQuality] = useState<Quality>(() => loadPref("quality", "balanced" as Quality));
  const [steps, setSteps] = useState(8);
  const [guidance, setGuidance] = useState(1);
  const [seed, setSeed] = useState(0);
  const [randomSeed, setRandomSeed] = useState(true);
  const [count, setCount] = useState(1);
  const [localImages, setLocalImages] = useState<string[]>([]);
  const images = controlledImages ?? localImages;
  const setImages = onImagesChange ?? setLocalImages;
  const [strength, setStrength] = useState(0.6);
  // What kind of edit this is. Each maps to a different route in the engine.
  // The type lives in lib/recipe because reading a past run back into it is a
  // lossy mapping worth testing on its own.
  const [editKind, setEditKind] = useState<EditKind>("instruct");
  const [maskBytes, setMaskBytes] = useState<Uint8Array | null>(null);
  const [pad, setPad] = useState({ top: 0, right: 25, bottom: 0, left: 25 });
  const [fill, setFill] = useState("auto");
  const i2iMode: "edit" | "latent" = editKind === "latent" ? "latent" : "edit";

  const [loras, setLoras] = useState<[string, number][]>([]);
  // "Memory saver" on screen. The engine and the saved recipes still call it
  // low_ram, which is not a name anyone choosing it should have to learn.
  const [lowRam, setLowRam] = useState(false);
  const [capCache, setCapCache] = useState(false);
  const [cacheLimit, setCacheLimit] = useState(1);

  const [running, setRunning] = useState(false);
  const [assisting, setAssisting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [prog, setProg] = useState<EngineProgress | null>(null);
  /**
   * Everything made in this screen this session, newest first.
   *
   * Each run used to replace the last, so comparing two seeds meant leaving
   * for the library. The strip grows instead, and each entry keeps the prompt
   * and seed that made it so either can be taken back.
   */
  const [made, setMade] = useState<(Made & { took?: number; batch: number })[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  /** Why the last run failed, shown on the canvas until the next one. */
  const [failure, setFailure] = useState<string | null>(null);
  /** A run refused before it started, shown beside the field at fault. */
  const [invalid, setInvalid] = useState<{ field: "prompt" | "images" | "mask" | "pad"; msg: string } | null>(null);
  const [undoPrompt, setUndoPrompt] = useState<string | null>(null);
  /** What the enhancer suggests, pending your decision. */
  const [proposal, setProposal] = useState<AssistResult | null>(null);
  /** The prompt as it is being written, shown while you wait for it. */
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useStatus(prog);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const advancedId = useId();
  const [elapsed, setElapsed] = useState(0);
  // Which model the engine currently holds, so a warm run is not quoted the
  // cold-start penalty.
  const [residentModel, setResidentModel] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState(() => loadPref("preview", true));
  useEffect(() => { savePref("preview", showPreview); }, [showPreview]);

  // A visible counter during the wait, so a long run never looks like a hang.
  useEffect(() => {
    if (!running) { setElapsed(0); return; }
    const started = Date.now();
    const t = window.setInterval(() => {
      setElapsed(Math.round((Date.now() - started) / 1000));
    }, 500);
    return () => window.clearInterval(t);
  }, [running]);

  const picked = autoModel ? autoPick(usable, wantTask) : usable.find((m) => m.id === modelId);
  const model = picked ?? usable[0];

  useEffect(() => {
    if (!model) return;
    if (!autoModel && model.id !== modelId) setModelId(model.id);
    setSteps(stepsFor(model, quality));
    setGuidance(model.guidance_default);
    if (model.low_ram_may_help) { setLowRam(true); setCapCache(true); }
    if (images.length > model.max_edit_images) {
      setImages(images.slice(0, model.max_edit_images));
    }
  // Follows the selected model, not every render.
  }, [model?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (model && !advanced) setSteps(stepsFor(model, quality));
  }, [quality, model, advanced]);

  // Restore a past run's settings. Applied once, then cleared, so editing a
  // field afterwards is not undone by a re-render.
  //
  // Every field below is guarded on being present rather than merely truthy
  // where the difference matters, because "the item does not record this" and
  // "the item recorded this as off" are different facts and only the first one
  // means "leave the control alone". Items made before any of this was
  // recorded therefore restore exactly what they restored before.
  useEffect(() => {
    if (!recipe) return;
    setPrompt(recipe.prompt);
    if (recipe.steps) { setSteps(recipe.steps); setAdvanced(true); }
    if (recipe.guidance != null) setGuidance(recipe.guidance);
    if (recipe.width && recipe.height) {
      const i = SHAPES.findIndex((sh) => sh.w === recipe.width && sh.h === recipe.height);
      if (i >= 0) setShape(i);
    }
    setRandomSeed(!recipe.reuseSeed);
    if (recipe.reuseSeed) setSeed(recipe.seed);

    if (recipe.negativePrompt != null) setNegative(recipe.negativePrompt);
    // Restored as recorded, including handles that are no longer installed.
    // Dropping an uninstalled adapter would hand back a different picture
    // under the same seed and call it a reproduction; failing loudly at the
    // engine at least says which adapter is missing.
    if (recipe.loras) setLoras(recipe.loras);

    const kind = editKindFor(recipe.i2iMode, recipe.outpaintPadding);
    if (kind) setEditKind(kind);
    // Only latent runs record a strength at all -- the other routes send null
    // -- so this never overwrites the slider for an instruct or expand recipe.
    if (recipe.imageStrength != null) setStrength(recipe.imageStrength);

    const padding = parseOutpaintPadding(recipe.outpaintPadding);
    if (padding) setPad(padding);
    // A fill this build no longer offers would leave the dropdown showing
    // nothing at all, which reads as a broken control rather than an old one.
    if (isOutpaintFill(recipe.outpaintFill)) setFill(recipe.outpaintFill);

    // Reduced-memory mode is forced on for a model whose peak exceeds this
    // Mac's ceiling, and a recipe made on a roomier machine must not switch
    // that back off: the reproduction would simply run out of memory. Where
    // the model does not need it, the recorded value wins.
    if (recipe.lowRam != null && !model?.low_ram_may_help) setLowRam(recipe.lowRam);

    onRecipeUsed?.();
  }, [recipe]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { savePref("advanced", advanced); }, [advanced]);
  useEffect(() => { savePref(`shape:${mode}`, shape); }, [shape, mode]);
  useEffect(() => { savePref("quality", quality); }, [quality]);

  // Distilled and Turbo checkpoints reject guidance above 1.0 outright.
  const fixedGuidance = (model?.guidance_max ?? 1) <= 1;
  const effectiveCount = lowRam ? 1 : count;
  const shapeDef = SHAPES[shape];
  const eta = estimateSeconds(
    model, steps, shapeDef.w * shapeDef.h, effectiveCount, residentModel === model?.name
  );

  const improve = async () => {
    if (!prompt.trim()) {
      setInvalid({ field: "prompt", msg: "Write a rough idea first." });
      promptRef.current?.focus();
      return;
    }
    const id = newJobId();
    setAssisting(true);
    setDraft("");
    const un = await onEngineProgress((p) => {
      if (p.job_id !== id) return;
      setProg(p);
      // The words as they arrive. A spinner says the machine is busy; it does
      // not say it is writing something worth waiting for, and eight seconds
      // of a spinner feels far longer than eight seconds of watching a
      // sentence form. The engine has streamed these all along and the
      // interface was throwing them away.
      if (typeof p.text === "string") setDraft(p.text);
    });
    try {
      // Shown, not applied. Overwriting the prompt the moment a rewrite
      // existed meant you could not see what changed, and the three ways it
      // could decline -- yours was fine, the rewrite was worse, the request
      // named nothing -- were reported with one misleading sentence.
      // Written for the model that will read it: FLUX.2, Qwen-Image and
      // Z-Image each want a different prompt, and an edit model wants an
      // instruction rather than a scene.
      const r = await api.assistPrompt(
        id, prompt, mode, mode === "edit" ? images : [],
        { modelId: model?.id ?? null, goal });
      setProposal(r);
      // Said once, at the end. The draft above is silent on purpose.
      setStatus("Suggestion ready, below the prompt.");
    } catch (e) {
      notify(errText(e), true);
    } finally {
      un(); setAssisting(false); setProg(null); setDraft("");
    }
  };

  /** Refuse a run that cannot work, next to the field that is wrong. */
  const validate = (): boolean => {
    if (!prompt.trim()) {
      setInvalid({ field: "prompt", msg: "Describe what you want first." });
      promptRef.current?.focus();
      return false;
    }
    if (mode === "edit" && images.length === 0) {
      setInvalid({ field: "images", msg: "Add a picture to edit." }); return false;
    }
    if (mode === "edit" && editKind === "mask" && !maskBytes) {
      setInvalid({ field: "mask", msg: "Paint over the part you want changed first." });
      return false;
    }
    if (mode === "edit" && editKind === "expand"
        && pad.top + pad.right + pad.bottom + pad.left === 0) {
      setInvalid({ field: "pad", msg: "Choose at least one side to extend." }); return false;
    }
    setInvalid(null);
    return true;
  };

  // A retry with Memory saver forced on has to use it now, not after the
  // state update lands, so the override is passed in rather than read back.
  const run = async (opts: { lowRam?: boolean } = {}) => {
    if (!model || running) return;
    if (!validate()) return;
    const lr = opts.lowRam ?? lowRam;

    const id = newJobId();
    const effectiveSeed = randomSeed ? Math.floor(Math.random() * 2_000_000_000) : seed;
    const runCount = lr ? 1 : count;
    setRunning(true); setJobId(id); setProg(null); setFailure(null);
    setStatus("Starting");

    setPreview(null);
    // The previous result stays, dimmed, until the new one lands. Clearing it
    // at the start meant a failed run left nothing at all on screen; the live
    // preview still takes precedence over it, so a second run does not look
    // like nothing is happening.
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    // If this throws, the progress listener above would never be released.
    // Failing to get a preview is not worth leaking one.
    let unPrev: (() => void) | undefined;
    try {
      unPrev = await onEnginePreview((f) => {
        if (f.jobId === id) setPreview(f.src);
      });
    } catch { unPrev = undefined; }
    const t0 = performance.now();
    let maskId: string | null = null;
    try {
      // A painted mask is stored like any other content, then referenced by id.
      if (editKind === "mask" && maskBytes) {
        maskId = await api.vaultImportBytes(
          maskBytes, "mask.png", "image/png", "mask"
        );
      }
      const padding = editKind === "expand"
        ? `${pad.top}%,${pad.right}%,${pad.bottom}%,${pad.left}%`
        : null;

      const args = {
        job_id: id,
        model_id: model.id,
        prompt,
        negative_prompt: negative || null,
        width: shapeDef.w,
        height: shapeDef.h,
        steps,
        guidance: Math.min(guidance, model.guidance_max),
        seed: effectiveSeed,
        count: runCount,
        images: mode === "edit" ? images : [],
        image_strength: mode === "edit" && i2iMode === "latent" ? strength : null,
        i2i_mode: mode === "edit" ? i2iMode : null,
        loras,
        low_ram: lr,
        preview: showPreview,
        cache_limit_gb: capCache ? cacheLimit : null,
        allow_over_budget: lr && model.low_ram_may_help,
        mask: maskId,
        outpaint_padding: padding,
        outpaint_fill: editKind === "expand" ? fill : null,
      };
      const res = mode === "edit" ? await api.editImage(args) : await api.generate(args);
      setResidentModel(model.name);
      const took = performance.now() - t0;
      // The engine gives image i of a batch the seed base + i, so each one can
      // be reproduced on its own.
      const fresh = res.map((rid, i) => ({
        id: rid, prompt, seed: effectiveSeed + i, took, batch: res.length,
      }));
      setMade((m) => [...fresh, ...m]);
      if (fresh.length > 0) setSelectedId(fresh[0].id);
      setSeed(effectiveSeed);
      onProduced();
      if (res.length === 0) setFailure("The engine finished but returned no picture.");
      else setStatus(`Done, ${res.length} image${res.length === 1 ? "" : "s"}.`);
    } catch (e) {
      const detail = errDetail(e);
      // Keep the engine's own wording where it differs; the card is short by
      // design and the original is what a diagnosis needs.
      if (detail) console.error("engine:", detail);
      if (isCancelled(e)) { setStatus("Cancelled."); notify("Cancelled."); }
      else { setStatus(""); setFailure(errText(e)); }
    } finally {
      // Masks are private working material, not library items. The engine has
      // finished reading it at this point, regardless of success or failure.
      if (maskId) {
        try { await api.vaultDelete(maskId); } catch { /* lock/cleanup race */ }
      }
      un(); unPrev?.();
      setRunning(false); setJobId(null); setProg(null);
      setPreview(null);
    }
  };

  const cancel = async () => {
    if (jobId) { try { await api.cancelJob(jobId); } catch { /* already gone */ } }
  };

  const runnable = !(mode === "edit" && editKind === "adjust");
  const shortcut = useRunShortcut(() => { void run(); }, runnable && !running && usable.length > 0);

  if (usable.length === 0) {
    return (
      <div className="empty-state">
        <span className="big" aria-hidden="true">◍</span>
        Nothing installed yet that can {mode === "edit" ? "edit images" : "make images"}.
        <div style={{ marginTop: 10, fontSize: 12 }}>
          Open <b>Models</b> and install one. For this Mac, FLUX.2 Klein 4B is
          the one to get — it does both.
        </div>
      </div>
    );
  }

  const current = made.find((m) => m.id === selectedId) ?? made[0] ?? null;
  const currentIndex = current ? made.indexOf(current) : -1;
  const resultAlt = current
    ? `Result ${currentIndex + 1} of ${made.length}: ${current.prompt}` : "";
  const errorFor = (f: NonNullable<typeof invalid>["field"]) =>
    invalid?.field === f ? invalid.msg : undefined;

  return (
    <div className="content split creation" style={{ padding: 0 }}
         ref={shortcut.ref} onKeyDown={shortcut.onKeyDown}>
      <LiveStatus text={status} />
      <div className="creation-params">
        <div className="panel">
          <div style={{ display: "flex", alignItems: "center", marginBottom: 12 }}>
            <h2 style={{ margin: 0 }}>{mode === "edit" ? "Edit a picture" : "Make a picture"}</h2>
            {/* One label whichever way it is set. "Simple"/"Advanced"
                swapped names with state, so the button said where it would
                go rather than what it was; the expanded state says that. */}
            <button
              className="btn small"
              style={{ marginLeft: "auto" }}
              aria-expanded={advanced}
              aria-controls={advancedId}
              onClick={() => setAdvanced(!advanced)}
            >
              <span aria-hidden="true">{advanced ? "▾ " : "▸ "}</span>Advanced settings
            </button>
          </div>

          {mode === "edit" && (
            <Choice
              label="What kind of edit?"
              value={editKind}
              onChange={(k) => { setEditKind(k); setInvalid(null); }}
              options={[
                ["instruct", "Change something"],
                ["mask", "Paint a region"],
                ["expand", "Extend the picture"],
                ["latent", "Reinterpret"],
                ["adjust", "Crop & rotate"],
              ] as const}
              hint={
                editKind === "instruct" ? "Describe one change; the rest is kept."
                : editKind === "mask" ? "Paint over an area and only that area is regenerated."
                : editKind === "expand" ? "Grow the canvas; the model invents what was outside the frame."
                : editKind === "latent" ? "Reinterpret the whole picture, keeping its composition."
                : "Crop, rotate or straighten. Instant, and no model involved."
              }
            />
          )}

          {mode === "edit" && (
            <div className="field" role="group" aria-labelledby={`${advancedId}-pic`}>
              <div className="choice-label" id={`${advancedId}-pic`}>Your picture</div>
              <ImageDrop
                images={images}
                onChange={(ids) => { setImages(ids); setInvalid(null); }}
                max={model?.max_edit_images ?? 1}
                onError={(m) => notify(m, true)}
              />
              {(model?.max_edit_images ?? 1) === 1 && editKind === "instruct" && (
                <div className="t-help" style={{ marginTop: 5 }}>
                  {model?.name} edits one picture at a time.
                </div>
              )}
              {errorFor("images") && (
                <div className="field-error" role="alert">{errorFor("images")}</div>
              )}
            </div>
          )}

          {mode === "edit" && editKind === "adjust" && images.length > 0 && (
            <div className="field">
              <div className="choice-label">Adjust the frame</div>
              <Adjust
                sourceId={images[0]}
                onSaved={(id) => { setImages([id]); onProduced(); }}
                notify={notify}
              />
            </div>
          )}

          {mode === "edit" && editKind === "mask" && images.length > 0 && (
            <div className="field">
              <div className="choice-label">Paint what should change</div>
              <MaskCanvas
                sourceId={images[0]}
                onMaskChange={(b) => { setMaskBytes(b); if (b) setInvalid(null); }}
                disabled={running}
              />
              {errorFor("mask") && (
                <div className="field-error" role="alert">{errorFor("mask")}</div>
              )}
            </div>
          )}

          {mode === "edit" && editKind === "expand" && (
            <div className="field" role="group" aria-labelledby={`${advancedId}-pad`}>
              <div className="choice-label" id={`${advancedId}-pad`}>
                How much to add <em>percent of the original</em>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {(["top", "right", "bottom", "left"] as const).map((side) => (
                  <Field key={side} className="field tight"
                    label={<><span style={{ textTransform: "capitalize" }}>{side}</span>
                      <em aria-hidden="true">{pad[side]}%</em></>}>
                    <input
                      type="range" min={0} max={100} step={5} value={pad[side]}
                      aria-valuetext={`${pad[side]} percent`}
                      aria-invalid={invalid?.field === "pad" || undefined}
                      onChange={(e) => { setPad({ ...pad, [side]: +e.target.value }); setInvalid(null); }}
                    />
                  </Field>
                ))}
              </div>
              {errorFor("pad") && (
                <div className="field-error" role="alert">{errorFor("pad")}</div>
              )}
              {/* Options come from the same list that decides whether a
                  restored fill is still offered; two copies would drift and
                  the drift would show up as an empty dropdown. */}
              <Field label="How the new area starts out" className="field" >
                <select value={fill} onChange={(e) => setFill(e.target.value)} style={{ marginTop: 0 }}>
                  {OUTPAINT_FILLS.map((f) => (
                    <option key={f.id} value={f.id}>{f.label}</option>
                  ))}
                </select>
              </Field>
            </div>
          )}

          {runnable && (
            <div className="field">
              <Field
                className="prompt-field"
                label={mode === "edit" ? "What should change?" : "What do you want to see?"}
                error={errorFor("prompt")}
              >
                <textarea
                  ref={promptRef}
                  value={prompt}
                  onChange={(e) => {
                    setPrompt(e.target.value); setUndoPrompt(null);
                    setProposal(null);
                    if (invalid?.field === "prompt") setInvalid(null);
                  }}
                  placeholder={mode === "edit"
                    ? "make the jacket red"
                    : "a ceramic teapot on a linen cloth by a window"}
                />
              </Field>
              {/* Out of the label, where it was: a button inside a label
                  became part of the text box's name. */}
              {undoPrompt !== null && (
                <button
                  className="btn small"
                  style={{ marginTop: 6 }}
                  onClick={() => { setPrompt(undoPrompt); setUndoPrompt(null); }}
                >
                  Undo the rewrite
                </button>
              )}
              <PromptGoal value={goal} onChange={(g) => { setGoal(g); setProposal(null); }}
                          disabled={assisting} />
              <button
                className="btn small full"
                style={{ marginTop: 7 }}
                disabled={assisting || running || !prompt.trim()}
                onClick={improve}
                // Said as it is. The old tooltip promised it added no objects
                // while its instructions told the writer to choose a room,
                // a light and a medium for every request.
                title={assistantReady
                  ? `Written for ${model?.name ?? "the selected model"}. Shown `
                    + "next to your words before anything changes. The picture "
                    + "model is unloaded while the writer runs and loads again "
                    + "on your next run."
                  : "Needs the prompt writer from the Models tab"}
              >
                {assisting
                  ? "Thinking…"
                  : goal === "style"
                    ? "Suggest styles"
                    : `${goal === "tighten" ? "Tighten" : "Enhance"} my prompt`
                      + (mode === "edit" && images.length > 0 ? " using my picture" : "")}
              </button>
              {assisting && draft && (
                // Silent on purpose: read aloud, every token would interrupt
                // the last. Completion is announced once instead.
                <div className="proposal plain" aria-live="off">
                  <b>Writing…</b>
                  <div className="text">{draft}</div>
                </div>
              )}
              {proposal && (
                <PromptProposal
                  result={proposal}
                  busy={assisting}
                  onAccept={(text) => {
                    setUndoPrompt(prompt);
                    setPrompt(text);
                    setProposal(null);
                    notify("Prompt updated. Undo is below the box.");
                  }}
                  onDismiss={() => setProposal(null)}
                />
              )}
              {!assistantReady && (
                <div className="t-help" style={{ marginTop: 5 }}>
                  Needs the prompt writer — a 2.1 GB download in the Models tab.
                  It runs on this Mac; nothing is sent anywhere.
                </div>
              )}
            </div>
          )}

          {(mode === "generate" || i2iMode === "latent") && (
            <Choice
              label="Shape"
              value={shape}
              onChange={setShape}
              options={SHAPES.map((s, i) => [i, s.label] as const)}
            />
          )}

          {!advanced && runnable && (
            <Choice
              label={<>Effort <em>{humanDuration(eta.seconds)}{eta.measured ? "" : " (rough)"}</em></>}
              value={quality}
              onChange={setQuality}
              options={(["fast", "balanced", "best"] as Quality[]).map((q) => [q, QUALITY_LABEL[q]] as const)}
            />
          )}

          <div className="t-note" style={{ marginTop: 4 }}>
            Using <b>{model?.name}</b>
            {autoModel && usable.length > 1 && (
              <button
                className="btn small"
                style={{ marginLeft: 7 }}
                aria-label={`Change model, now ${model?.name}`}
                onClick={() => { setAutoModel(false); setModelId(model?.id ?? ""); setAdvanced(true); }}
              >
                change
              </button>
            )}
          </div>
        </div>

        {/* The run button sits right under the settings it runs, before the
            optional adapters and Advanced panels, so it neither drifts out of
            view nor floats over panels further down. */}
        {runnable && (
          <div className="run-footer">
            {running ? (
              <>
                <JobProgress p={prog} label="Working" />
                <div className="t-note run-footer-times">
                  <span>{elapsed}s elapsed</span>
                  <span>
                    {elapsed > eta.seconds * 1.4
                      ? "longer than expected"
                      : `${humanDuration(eta.seconds)} in total`}
                  </span>
                </div>
                <button className="btn danger full" style={{ marginTop: 8 }} onClick={cancel}>
                  Cancel
                </button>
              </>
            ) : (
              <>
                <button className="btn primary full" onClick={() => void run()}
                  aria-keyshortcuts="Meta+Enter">
                  {mode === "edit" ? "Apply the change" : "Make it"}
                  <span className="kbd" aria-hidden="true">⌘↵</span>
                </button>
                <div className="t-note" style={{ marginTop: 6, textAlign: "center" }}>
                  Takes {humanDuration(eta.seconds)}
                  {eta.measured
                    ? ` — measured over ${model?.measured_runs} run${model?.measured_runs === 1 ? "" : "s"} on this Mac`
                    : " — a rough guess until the first run"}
                  {eta.loadSeconds > 0 && `, including ~${eta.loadSeconds}s to load the model`}
                </div>
              </>
            )}
          </div>
        )}

        {/* Style is what a picture board is made of, so the adapters that
            carry it are not an advanced setting. This used to sit inside the
            Advanced block, which is off by default -- so they could not be
            found at all without knowing they were there. */}
        <Loras selected={loras} onSelectedChange={setLoras} notify={notify} />

        <div id={advancedId}>
        {advanced && (
          <>
            <div className="panel">
              <h2>Model and parameters</h2>
              <Field label="Model">
                <select
                  value={autoModel ? "__auto" : (model?.id ?? "")}
                  onChange={(e) => {
                    if (e.target.value === "__auto") { setAutoModel(true); }
                    else { setAutoModel(false); setModelId(e.target.value); }
                  }}
                >
                  <option value="__auto">Choose for me{model ? ` (${model.name})` : ""}</option>
                  {usable.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} — ~{m.peak_gib.toFixed(1)} GiB
                    </option>
                  ))}
                </select>
              </Field>

              {mode === "edit" && editKind === "latent" && (
                <Field label={<>How far from the original <em aria-hidden="true">{strength.toFixed(2)}</em></>}
                  hint="Low keeps close to your picture; high reinvents more of it.">
                  <input type="range" min={0.1} max={0.95} step={0.05} value={strength}
                    aria-valuetext={`strength ${strength.toFixed(2)}`}
                    onChange={(e) => setStrength(+e.target.value)} />
                </Field>
              )}

              {!fixedGuidance && (
                <Field label={<>Negative prompt <em>optional</em></>}>
                  <input type="text" value={negative} onChange={(e) => setNegative(e.target.value)}
                    placeholder="blurry, low quality" />
                </Field>
              )}

              <div className="row">
                <Field label={<>Steps <em aria-hidden="true">{steps}</em></>}>
                  <input type="range" min={2} max={50} value={steps}
                    aria-valuetext={`${steps} steps`}
                    onChange={(e) => setSteps(+e.target.value)} />
                </Field>
                <Field
                  label={<>Guidance <em aria-hidden="true">{fixedGuidance ? "not adjustable" : guidance.toFixed(1)}</em></>}
                  hint={fixedGuidance ? "This model is tuned to work without it." : undefined}
                >
                  <input
                    type="range" min={0} max={model?.guidance_max ?? 1} step={0.5}
                    value={guidance} disabled={fixedGuidance}
                    aria-valuetext={fixedGuidance ? "not adjustable" : `guidance ${guidance.toFixed(1)}`}
                    onChange={(e) => setGuidance(+e.target.value)}
                  />
                </Field>
              </div>

              <div className="row">
                <div>
                  <Field label="Seed" className="field tight">
                    <input type="number" value={seed} disabled={randomSeed}
                      onChange={(e) => setSeed(+e.target.value)} />
                  </Field>
                  <label className="check">
                    <input type="checkbox" checked={randomSeed}
                      onChange={(e) => setRandomSeed(e.target.checked)} />
                    Different every time
                  </label>
                </div>
                <Field
                  label={<>How many <em aria-hidden="true">{effectiveCount}</em></>}
                  hint={lowRam && count > 1 ? "Memory saver makes one at a time." : undefined}
                >
                  <input type="range" min={1} max={4} value={count}
                    aria-valuetext={`${effectiveCount} picture${effectiveCount === 1 ? "" : "s"}`}
                    onChange={(e) => setCount(+e.target.value)} />
                </Field>
              </div>
            </div>

            <div className="panel">
              <h2>Memory</h2>
              {model?.low_ram_may_help && (
                <div className="notice warn" style={{ marginBottom: 12 }}>
                  <strong>Above this Mac&rsquo;s memory ceiling</strong>
                  {model.fit_reason} Memory saver has been switched on for you.
                </div>
              )}
              <div className="field">
                <label className="check">
                  <input type="checkbox" checked={lowRam}
                    aria-describedby={`${advancedId}-saver`}
                    onChange={(e) => setLowRam(e.target.checked)} />
                  Memory saver
                </label>
                <div id={`${advancedId}-saver`} className="t-help" style={{ marginTop: 5 }}>
                  Uses less memory so larger models fit. Slower, and makes one
                  picture at a time.
                </div>
              </div>
              <div className="field">
                <label className="check">
                  <input type="checkbox" checked={showPreview}
                    aria-describedby={`${advancedId}-preview`}
                    onChange={(e) => setShowPreview(e.target.checked)} />
                  Show the picture forming
                </label>
                <div id={`${advancedId}-preview`} className="t-help" style={{ marginTop: 5 }}>
                  A rough preview while it works. Turn it off for the fastest run.
                </div>
              </div>
              {/* A tuning knob for diagnosing memory trouble, not a choice
                  anyone makes up front, so it is folded away. */}
              <details className="diagnostics">
                <summary>Diagnostics</summary>
                <div className="field">
                  <label className="check">
                    <input type="checkbox" checked={capCache}
                      onChange={(e) => setCapCache(e.target.checked)} />
                    Cap the working memory cache
                  </label>
                </div>
                {capCache && (
                  <Field label={<>Cache limit <em aria-hidden="true">{cacheLimit} GiB</em></>}>
                    <input type="range" min={0.5} max={8} step={0.5} value={cacheLimit}
                      aria-valuetext={`${cacheLimit} gigabytes`}
                      onChange={(e) => setCacheLimit(+e.target.value)} />
                  </Field>
                )}
                <div className="t-help">
                  Without a cap the engine uses an eighth of this Mac&rsquo;s memory,
                  between 1 and 8 GiB.
                </div>
              </details>
            </div>
          </>
        )}
        </div>

      </div>

      <div className="creation-canvas">
        <div className="canvas">
          {failure && !running ? (
            <div className="canvas-stack">
              <RunError
                message={failure}
                onRetry={() => void run()}
                onMemorySaver={lowRam ? undefined : () => { setLowRam(true); void run({ lowRam: true }); }}
              />
              {current && <img className="stale" src={vaultUrl(current.id)} alt={`Previous ${resultAlt}`} />}
            </div>
          ) : running && preview ? (
            <div style={{ textAlign: "center" }}>
              {/* Deliberately soft: this is a cheap decode of a partial step,
                  not the output, and it should not be mistaken for it. */}
              <img
                src={preview}
                alt="Preview of the picture forming"
                style={{
                  maxWidth: "100%", maxHeight: "58vh", borderRadius: 7,
                  imageRendering: "auto", opacity: 0.92,
                }}
              />
              <div className="t-note" style={{ marginTop: 8 }}>
                Taking shape{prog?.step && prog?.total_steps
                  ? ` — step ${prog.step} of ${prog.total_steps}` : "…"}
              </div>
            </div>
          ) : running && current ? (
            <div className="canvas-stack">
              <img className="stale" src={vaultUrl(current.id)} alt={`Previous ${resultAlt}`} />
              <div className="t-note">Working — the previous result stays until the new one lands.</div>
            </div>
          ) : current ? (
            <img src={vaultUrl(current.id)} alt={resultAlt} />
          ) : (
            <div className="empty">
              <span className="big" aria-hidden="true">{mode === "edit" ? "✎" : "✦"}</span>
              {running
                ? "Working — the first run also loads the model into memory."
                : mode === "edit"
                  ? "Add a picture and say what to change."
                  : "Describe a picture and press Make it."}
            </div>
          )}
        </div>

        {current && (
          <>
            <ResultStrip items={made} selected={current.id} onSelect={setSelectedId} />
            <div className="result-actions">
              <button className="btn small"
                onClick={() => exportItem({ id: current.id, name: "" }, notify)}>
                Save a copy…
              </button>
              {/* Carry the result onward. It is already a vault item, so this
                  is just passing an id -- no export, no re-import. */}
              {mode === "edit" && (
                <button
                  className="btn small primary"
                  title="Use this result as the new source and keep going"
                  onClick={() => {
                    setImages([current.id]);
                    notify("Now editing the result.");
                  }}
                >
                  Keep editing this
                </button>
              )}
              <button className="btn small"
                onClick={() => {
                  setSeed(current.seed); setRandomSeed(false);
                  notify(`Seed ${current.seed} will be used for the next run.`);
                }}>
                Reuse seed
              </button>
              <button className="btn small"
                disabled={current.prompt === prompt}
                onClick={() => { setUndoPrompt(prompt); setPrompt(current.prompt); }}>
                Use prompt
              </button>
              {send && (
                <SendTo
                  ids={[current.id]}
                  send={send}
                  models={models}
                  exclude={mode === "edit" ? ["edit"] : []}
                />
              )}
              <span className="t-note" style={{ marginLeft: "auto" }}>
                {current.took != null && <>Took {fmtDuration(Math.round(current.took))}
                  {current.batch > 1 ? ` for ${current.batch}` : ""} · </>}
                seed {current.seed}
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
