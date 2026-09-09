import { useEffect, useMemo, useState } from "react";
import { api, errText, fmtDuration, newJobId, onEngineProgress, vaultUrl } from "../lib/api";
import type { EngineProgress, ModelStatus } from "../lib/types";
import { autoPick, estimateSeconds, humanDuration, QUALITY_LABEL, SHAPES, stepsFor, type Quality } from "../lib/presets";
import { exportItem, ImageDrop, JobProgress } from "./shared";
import MaskCanvas from "./MaskCanvas";
import { loadPref, savePref } from "../lib/prefs";

export type Mode = "generate" | "edit";

export default function Studio({
  mode, models, notify, onProduced, images: controlledImages, onImagesChange,
  onSendToEdit,
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
    () => models.some((m) => m.tasks.includes("assist" as never) && m.installed),
    [models]
  );

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
  type EditKind = "instruct" | "mask" | "expand" | "latent";
  const [editKind, setEditKind] = useState<EditKind>("instruct");
  const [maskBytes, setMaskBytes] = useState<Uint8Array | null>(null);
  const [pad, setPad] = useState({ top: 0, right: 25, bottom: 0, left: 25 });
  const [fill, setFill] = useState("auto");
  const i2iMode: "edit" | "latent" = editKind === "latent" ? "latent" : "edit";

  const [lowRam, setLowRam] = useState(false);
  const [capCache, setCapCache] = useState(false);
  const [cacheLimit, setCacheLimit] = useState(1);

  const [running, setRunning] = useState(false);
  const [assisting, setAssisting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [prog, setProg] = useState<EngineProgress | null>(null);
  const [outputs, setOutputs] = useState<string[]>([]);
  const [selected, setSelected] = useState(0);
  const [took, setTook] = useState<number | null>(null);
  const [undoPrompt, setUndoPrompt] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  // Which model the engine currently holds, so a warm run is not quoted the
  // cold-start penalty.
  const [residentModel, setResidentModel] = useState<string | null>(null);

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
    if (!prompt.trim()) { notify("Write a rough idea first.", true); return; }
    const id = newJobId();
    setAssisting(true);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    try {
      const r = await api.assistPrompt(id, prompt, mode, mode === "edit" ? images : []);
      setUndoPrompt(prompt);
      setPrompt(r.prompt);
      notify(r.saw_image ? "Rewritten, using your image for context." : "Prompt rewritten.");
    } catch (e) {
      notify(errText(e), true);
    } finally {
      un(); setAssisting(false); setProg(null);
    }
  };

  const run = async () => {
    if (!model) return;
    if (!prompt.trim()) { notify("Describe what you want first.", true); return; }
    if (mode === "edit" && images.length === 0) {
      notify("Add an image to edit.", true); return;
    }
    if (mode === "edit" && editKind === "mask" && !maskBytes) {
      notify("Paint over the part you want changed first.", true); return;
    }
    if (mode === "edit" && editKind === "expand"
        && pad.top + pad.right + pad.bottom + pad.left === 0) {
      notify("Choose at least one side to extend.", true); return;
    }

    const id = newJobId();
    const effectiveSeed = randomSeed ? Math.floor(Math.random() * 2_000_000_000) : seed;
    setRunning(true); setJobId(id); setProg(null); setTook(null);

    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    const t0 = performance.now();
    try {
      // A painted mask is stored like any other content, then referenced by id.
      let maskId: string | null = null;
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
        count: effectiveCount,
        images: mode === "edit" ? images : [],
        image_strength: mode === "edit" && i2iMode === "latent" ? strength : null,
        i2i_mode: mode === "edit" ? i2iMode : null,
        low_ram: lowRam,
        cache_limit_gb: capCache ? cacheLimit : null,
        allow_over_budget: model.low_ram_may_help,
        mask: maskId,
        outpaint_padding: padding,
        outpaint_fill: editKind === "expand" ? fill : null,
      };
      const res = mode === "edit" ? await api.editImage(args) : await api.generate(args);
      setResidentModel(model.name);
      setOutputs(res);
      setSelected(0);
      setTook(performance.now() - t0);
      setSeed(effectiveSeed);
      onProduced();
      if (res.length === 0) notify("The engine returned no output.", true);
    } catch (e) {
      const msg = errText(e);
      notify(msg.includes("cancelled") ? "Cancelled." : msg, !msg.includes("cancelled"));
    } finally {
      un();
      setRunning(false); setJobId(null); setProg(null);
    }
  };

  const cancel = async () => {
    if (jobId) { try { await api.cancelJob(jobId); } catch { /* already gone */ } }
  };

  if (usable.length === 0) {
    return (
      <div className="empty-state">
        <span className="big">◍</span>
        Nothing installed yet that can {mode === "edit" ? "edit images" : "make images"}.
        <div style={{ marginTop: 10, fontSize: 12 }}>
          Open <b>Models</b> and install one. For this Mac, FLUX.2 Klein 4B is
          the one to get — it does both.
        </div>
      </div>
    );
  }

  return (
    <div className="content split" style={{ padding: 0 }}>
      <div>
        <div className="panel">
          <div style={{ display: "flex", alignItems: "center", marginBottom: 12 }}>
            <h2 style={{ margin: 0 }}>{mode === "edit" ? "Edit a picture" : "Make a picture"}</h2>
            <button
              className="btn small"
              style={{ marginLeft: "auto" }}
              onClick={() => setAdvanced(!advanced)}
            >
              {advanced ? "Simple" : "Advanced"}
            </button>
          </div>

          {mode === "edit" && (
            <div className="field">
              <label>What kind of edit?</label>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 4 }}>
                {([
                  ["instruct", "Change something"],
                  ["mask", "Paint a region"],
                  ["expand", "Extend the picture"],
                  ["latent", "Reinterpret"],
                ] as [EditKind, string][]).map(([k, label]) => (
                  <button
                    key={k}
                    className={"btn small" + (editKind === k ? " primary" : "")}
                    onClick={() => setEditKind(k)}
                  >{label}</button>
                ))}
              </div>
              <div style={{ fontSize: 10.5, color: "var(--text-faint)", lineHeight: 1.55 }}>
                {editKind === "instruct" && "Describe one change; the rest is kept."}
                {editKind === "mask" && "Paint over an area and only that area is regenerated."}
                {editKind === "expand" && "Grow the canvas; the model invents what was outside the frame."}
                {editKind === "latent" && "Reinterpret the whole picture, keeping its composition."}
              </div>
            </div>
          )}

          {mode === "edit" && (
            <div className="field">
              <label>Your picture</label>
              <ImageDrop
                images={images}
                onChange={setImages}
                max={model?.max_edit_images ?? 1}
                onError={(m) => notify(m, true)}
              />
              {(model?.max_edit_images ?? 1) === 1 && editKind === "instruct" && (
                <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5 }}>
                  {model?.name} edits one picture at a time.
                </div>
              )}
            </div>
          )}

          {mode === "edit" && editKind === "mask" && images.length > 0 && (
            <div className="field">
              <label>Paint what should change</label>
              <MaskCanvas
                sourceId={images[0]}
                onMaskChange={setMaskBytes}
                disabled={running}
              />
            </div>
          )}

          {mode === "edit" && editKind === "expand" && (
            <div className="field">
              <label>How much to add <em>percent of the original</em></label>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {(["top", "right", "bottom", "left"] as const).map((side) => (
                  <div key={side}>
                    <label style={{ marginBottom: 3 }}>
                      <span style={{ textTransform: "capitalize" }}>{side}</span>
                      <em>{pad[side]}%</em>
                    </label>
                    <input
                      type="range" min={0} max={100} step={5} value={pad[side]}
                      onChange={(e) => setPad({ ...pad, [side]: +e.target.value })}
                    />
                  </div>
                ))}
              </div>
              <div className="field" style={{ marginTop: 8 }}>
                <label>How the new area starts out</label>
                <select value={fill} onChange={(e) => setFill(e.target.value)}>
                  <option value="auto">Choose for me</option>
                  <option value="edge">Continue the edges outward</option>
                  <option value="neutral">Flat colour, invent new subject matter</option>
                  <option value="blur">Blurred copy of the original</option>
                </select>
              </div>
            </div>
          )}

          <div className="field">
            <label>
              {mode === "edit" ? "What should change?" : "What do you want to see?"}
              {undoPrompt !== null && (
                <button
                  className="btn small"
                  style={{ padding: "1px 7px", fontSize: 10.5 }}
                  onClick={() => { setPrompt(undoPrompt); setUndoPrompt(null); }}
                >
                  Undo rewrite
                </button>
              )}
            </label>
            <textarea
              value={prompt}
              onChange={(e) => { setPrompt(e.target.value); setUndoPrompt(null); }}
              placeholder={mode === "edit"
                ? "make the jacket red"
                : "a ceramic teapot on a linen cloth by a window"}
            />
            <button
              className="btn small full"
              style={{ marginTop: 7 }}
              disabled={assisting || running || !prompt.trim()}
              onClick={improve}
              title={assistantReady
                ? "Rewrites your wording into something the model follows better"
                : "Needs the prompt assistant from the Models tab"}
            >
              {assisting
                ? "Rewriting…"
                : mode === "edit" && images.length > 0
                  ? "✨ Improve this, using my picture"
                  : "✨ Improve my wording"}
            </button>
            {!assistantReady && (
              <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5 }}>
                Needs the prompt assistant — a 1.5 GB download in the Models tab.
                It runs on this Mac; nothing is sent anywhere.
              </div>
            )}
          </div>

          {(mode === "generate" || i2iMode === "latent") && (
            <div className="field">
              <label>Shape</label>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {SHAPES.map((s, i) => (
                  <button
                    key={s.key}
                    className={"btn small" + (i === shape ? " primary" : "")}
                    onClick={() => setShape(i)}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {!advanced && (
            <div className="field">
              <label>
                Effort
                <em>{humanDuration(eta.seconds)}{eta.measured ? "" : " (rough)"}</em>
              </label>
              <div style={{ display: "flex", gap: 6 }}>
                {(["fast", "balanced", "best"] as Quality[]).map((q) => (
                  <button
                    key={q}
                    className={"btn small" + (q === quality ? " primary" : "")}
                    onClick={() => setQuality(q)}
                  >
                    {QUALITY_LABEL[q]}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 4 }}>
            Using <b>{model?.name}</b>
            {autoModel && usable.length > 1 && (
              <button
                className="btn small"
                style={{ padding: "1px 7px", fontSize: 10, marginLeft: 7 }}
                onClick={() => { setAutoModel(false); setModelId(model?.id ?? ""); setAdvanced(true); }}
              >
                change
              </button>
            )}
          </div>

          {running ? (
            <>
              <JobProgress p={prog} label="Working" />
              <div style={{
                display: "flex", justifyContent: "space-between",
                fontSize: 11, color: "var(--text-faint)", marginTop: 6,
              }}>
                <span>{elapsed}s elapsed</span>
                <span>
                  {elapsed > eta.seconds * 1.4
                    ? "longer than expected"
                    : `${humanDuration(eta.seconds)} in total`}
                </span>
              </div>
              <button className="btn danger full" style={{ marginTop: 10 }} onClick={cancel}>
                Cancel
              </button>
            </>
          ) : (
            <>
              <button className="btn primary full" style={{ marginTop: 12 }} onClick={run}>
                {mode === "edit" ? "Apply the change" : "Make it"}
              </button>
              <div style={{
                fontSize: 10.5, color: "var(--text-faint)",
                marginTop: 6, textAlign: "center",
              }}>
                Takes {humanDuration(eta.seconds)}
                {eta.measured
                  ? ` — measured over ${model?.measured_runs} run${model?.measured_runs === 1 ? "" : "s"} on this Mac`
                  : " — a rough guess until the first run"}
                {eta.loadSeconds > 0 && `, including ~${eta.loadSeconds}s to load the model`}
              </div>
            </>
          )}
        </div>

        {advanced && (
          <>
            <div className="panel">
              <h2>Model and parameters</h2>
              <div className="field">
                <label>Model</label>
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
              </div>

              {mode === "edit" && (
                <>
                  {editKind === "latent" && (
                    <div className="field">
                      <label>How far from the original <em>{strength.toFixed(2)}</em></label>
                      <input type="range" min={0.1} max={0.95} step={0.05} value={strength}
                        onChange={(e) => setStrength(+e.target.value)} />
                    </div>
                  )}
                </>
              )}

              {!fixedGuidance && (
                <div className="field">
                  <label>Negative prompt <em>optional</em></label>
                  <input type="text" value={negative} onChange={(e) => setNegative(e.target.value)}
                    placeholder="blurry, low quality" />
                </div>
              )}

              <div className="row">
                <div className="field">
                  <label>Steps <em>{steps}</em></label>
                  <input type="range" min={2} max={50} value={steps}
                    onChange={(e) => setSteps(+e.target.value)} />
                </div>
                <div className="field">
                  <label>
                    Guidance
                    <em>{fixedGuidance ? "not adjustable" : guidance.toFixed(1)}</em>
                  </label>
                  <input
                    type="range" min={0} max={model?.guidance_max ?? 1} step={0.5}
                    value={guidance} disabled={fixedGuidance}
                    onChange={(e) => setGuidance(+e.target.value)}
                  />
                  {fixedGuidance && (
                    <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5 }}>
                      Distilled checkpoint — trained without classifier-free guidance.
                    </div>
                  )}
                </div>
              </div>

              <div className="row">
                <div className="field">
                  <label>Seed</label>
                  <input type="number" value={seed} disabled={randomSeed}
                    onChange={(e) => setSeed(+e.target.value)} />
                  <label style={{ marginTop: 6, cursor: "pointer" }}>
                    <span>
                      <input type="checkbox" checked={randomSeed}
                        style={{ width: "auto", marginRight: 6 }}
                        onChange={(e) => setRandomSeed(e.target.checked)} />
                      Different every time
                    </span>
                  </label>
                </div>
                <div className="field">
                  <label>How many <em>{effectiveCount}</em></label>
                  <input type="range" min={1} max={4} value={count}
                    onChange={(e) => setCount(+e.target.value)} />
                  {lowRam && count > 1 && (
                    <div style={{ fontSize: 10.5, color: "var(--warn)", marginTop: 5 }}>
                      Reduced-memory mode makes one at a time.
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="panel">
              <h2>Memory</h2>
              {model?.low_ram_may_help && (
                <div className="notice warn" style={{ marginBottom: 12 }}>
                  <strong>Above this Mac&rsquo;s memory ceiling</strong>
                  {model.fit_reason} Reduced-memory mode has been switched on for you.
                </div>
              )}
              <div className="field">
                <label style={{ cursor: "pointer" }}>
                  <span>
                    <input type="checkbox" checked={lowRam} style={{ width: "auto", marginRight: 6 }}
                      onChange={(e) => setLowRam(e.target.checked)} />
                    Reduced-memory mode
                  </span>
                  <em>{lowRam ? "slower" : "off"}</em>
                </label>
                <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5, lineHeight: 1.55 }}>
                  Caps the MLX buffer cache at 1 GiB, releases the text encoder once the prompt is
                  encoded, and drops the model when the run ends. Bounds transient memory, not the
                  resident weights, and makes one image at a time.
                </div>
              </div>
              <div className="field">
                <label style={{ cursor: "pointer" }}>
                  <span>
                    <input type="checkbox" checked={capCache} style={{ width: "auto", marginRight: 6 }}
                      onChange={(e) => setCapCache(e.target.checked)} />
                    Cap the MLX buffer cache
                  </span>
                  <em>{capCache ? `${cacheLimit} GiB` : "auto"}</em>
                </label>
                {capCache && (
                  <input type="range" min={0.5} max={8} step={0.5} value={cacheLimit}
                    onChange={(e) => setCacheLimit(+e.target.value)} style={{ marginTop: 6 }} />
                )}
                <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5, lineHeight: 1.55 }}>
                  Without a cap the engine uses total RAM &divide; 8, clamped to 1&ndash;8 GiB.
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      <div>
        <div className="canvas">
          {outputs.length > 0 ? (
            <img src={vaultUrl(outputs[selected])} alt="" />
          ) : (
            <div className="empty">
              <span className="big">{mode === "edit" ? "✎" : "✦"}</span>
              {running
                ? "Working — the first run also loads the model into memory."
                : mode === "edit"
                  ? "Add a picture and say what to change."
                  : "Describe a picture and press Make it."}
            </div>
          )}
        </div>

        {outputs.length > 0 && (
          <>
            {outputs.length > 1 && (
              <div className="results-strip">
                {outputs.map((p, i) => (
                  <img key={p} src={vaultUrl(p)} alt=""
                    className={i === selected ? "sel" : ""}
                    onClick={() => setSelected(i)} />
                ))}
              </div>
            )}
            <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center", flexWrap: "wrap" }}>
              <button className="btn small"
                onClick={() => exportItem({ id: outputs[selected], name: "" }, notify)}>
                Save a copy…
              </button>
              {mode === "generate" && onSendToEdit && (
                <button
                  className="btn small"
                  // The result is already a vault item, so hand the id to the
                  // Edit tab directly. Writing to this component's own state
                  // put it in the wrong instance and looked like a no-op.
                  onClick={() => onSendToEdit([outputs[selected]])}
                >
                  Edit this
                </button>
              )}
              {took != null && (
                <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-faint)" }}>
                  Took {fmtDuration(Math.round(took))}
                  {outputs.length > 1 ? ` for ${outputs.length}` : ""} · seed {seed}
                </span>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
