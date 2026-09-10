import { useMemo, useState } from "react";
import { api, errText, newJobId, onEngineProgress, vaultUrl } from "../lib/api";
import type { EngineProgress, ModelStatus, Panel } from "../lib/types";
import { JobProgress } from "./shared";

/**
 * A story, divided into panels, drawn as one consistent character.
 *
 * The pieces here are proven separately: the writer divides prose reliably,
 * and a character sheet fed back as a reference holds the character across
 * shots. This puts them in order and keeps the one thing that makes the
 * result read as a story rather than a set of unrelated pictures -- the same
 * sheet references every panel, and the same style words lead every prompt.
 */

/** Style is a whole-board decision, not a per-panel one, so it lives here. */
const STYLES: [string, string, string][] = [
  ["anime", "Anime",
   "flat cel-shaded anime illustration, clean linework, muted palette"],
  ["photo", "Photographic",
   "photographic, natural skin texture, available light, 35mm, shallow depth of field"],
  ["ink", "Ink and wash",
   "black ink illustration with grey wash, visible brushwork, high contrast"],
  ["paint", "Painted",
   "digital painting, visible brush strokes, soft edges, muted colour"],
];

/** Panels are small on purpose: a board is read at a glance, not printed. */
const PANEL_W = 512;
const PANEL_H = 512;

type Stage = "idle" | "dividing" | "casting" | "drawing";

export default function Storyboard({
  models, notify, onProduced,
}: {
  models: ModelStatus[];
  notify: (m: string, bad?: boolean) => void;
  onProduced: () => void;
}) {
  const usable = useMemo(
    () => models.filter(
      (m) => m.installed && !m.hopeless && m.fit !== "broken"
        && m.tasks.includes("edit" as never)
        && m.tasks.includes("text_to_image" as never)
    ),
    [models]
  );
  const writerReady = useMemo(
    () => models.some((m) => m.id === "qwen3-4b-instruct-4bit" && m.installed),
    [models]
  );

  const [modelId, setModelId] = useState("");
  const model = usable.find((m) => m.id === modelId) ?? usable[0];

  const [story, setStory] = useState("");
  const [count, setCount] = useState(6);
  const [style, setStyle] = useState(STYLES[0][0]);
  const [character, setCharacter] = useState("");

  const [panels, setPanels] = useState<Panel[] | null>(null);
  const [sheet, setSheet] = useState<string | null>(null);
  const [drawn, setDrawn] = useState<(string | null)[]>([]);
  const [stage, setStage] = useState<Stage>("idle");
  const [prog, setProg] = useState<EngineProgress | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [done, setDone] = useState(0);

  const styleWords = STYLES.find((s) => s[0] === style)?.[2] ?? "";
  const busy = stage !== "idle";

  /** Step one: divide the prose. Nothing is drawn yet. */
  const divide = async () => {
    if (!story.trim()) { notify("Write the story first.", true); return; }
    const id = newJobId();
    setStage("dividing"); setJobId(id); setPanels(null); setDrawn([]); setSheet(null);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    try {
      const r = await api.shotList(id, story, count);
      setPanels(r.panels);
      if (r.panels.length < r.asked) {
        notify(`Divided into ${r.panels.length} panels rather than ${r.asked}.`);
      }
    } catch (e) {
      notify(errText(e), true);
    } finally {
      un(); setStage("idle"); setProg(null); setJobId(null);
    }
  };

  /** The prompt for one panel: style, then character, then the moment. */
  const panelPrompt = (p: Panel) => {
    const bits = [
      styleWords,
      `${p.shot} shot`,
      character.trim(),
      p.action,
      p.setting,
    ].map((b) => b.trim()).filter(Boolean);
    return bits.join(". ") + ".";
  };

  /** Step two: cast the character once, then draw every panel against it. */
  const draw = async () => {
    if (!model) { notify("No model that can both generate and edit is installed.", true); return; }
    if (!panels) return;
    if (!character.trim()) {
      notify("Describe the character, so every panel can hold the same one.", true);
      return;
    }

    setDone(0);
    setDrawn(new Array(panels.length).fill(null));

    // The sheet is the anchor. Every panel references it, which is the whole
    // reason the character survives from one shot to the next.
    const castId = newJobId();
    setStage("casting"); setJobId(castId);
    let un = await onEngineProgress((p) => { if (p.job_id === castId) setProg(p); });
    let sheetId: string;
    try {
      const res = await api.generate({
        job_id: castId, model_id: model.id,
        prompt: `${styleWords}. Character reference sheet, full body, neutral `
              + `pose, plain background. ${character.trim()}.`,
        negative_prompt: null,
        width: PANEL_W, height: PANEL_H,
        steps: model.steps_default || 4,
        guidance: Math.min(1.0, model.guidance_max),
        seed: 7, count: 1, images: [], image_strength: null, i2i_mode: null,
        low_ram: true, preview: false, cache_limit_gb: null,
        allow_over_budget: false, loras: [], mask: null,
        outpaint_padding: null, outpaint_fill: null,
      });
      if (!res.length) throw new Error("the character sheet came back empty");
      sheetId = res[0];
      setSheet(sheetId);
      onProduced();
    } catch (e) {
      notify(errText(e), true);
      un(); setStage("idle"); setProg(null); setJobId(null);
      return;
    }
    un();

    // Panels run one at a time. Two models never share this machine's memory,
    // and a board of six is minutes of work either way.
    setStage("drawing");
    for (let i = 0; i < panels.length; i++) {
      const id = newJobId();
      setJobId(id);
      un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
      try {
        const res = await api.editImage({
          job_id: id, model_id: model.id,
          prompt: panelPrompt(panels[i]),
          negative_prompt: null,
          width: PANEL_W, height: PANEL_H,
          steps: model.steps_default || 4,
          guidance: Math.min(1.0, model.guidance_max),
          seed: 7 + i, count: 1,
          images: [sheetId],
          image_strength: null, i2i_mode: "reference",
          low_ram: true, preview: false, cache_limit_gb: null,
          allow_over_budget: false, loras: [], mask: null,
          outpaint_padding: null, outpaint_fill: null,
        });
        setDrawn((d) => { const n = [...d]; n[i] = res[0] ?? null; return n; });
        setDone(i + 1);
        onProduced();
      } catch (e) {
        const msg = errText(e);
        notify(`Panel ${i + 1}: ${msg}`, true);
        if (msg.includes("ancelled")) break;
      } finally {
        un();
      }
    }
    setStage("idle"); setProg(null); setJobId(null);
  };

  const cancel = async () => {
    if (jobId) { try { await api.cancelJob(jobId); } catch { /* already gone */ } }
  };

  if (usable.length === 0) {
    return (
      <div className="panel">
        <h2>Picture board</h2>
        <div className="notice warn">
          <strong>Needs a model that can both generate and edit</strong>
          A board is drawn by making one character sheet and then referencing it
          in every panel, so the model has to do both. FLUX.2 Klein is the
          smallest that does — add it from the Models tab.
        </div>
      </div>
    );
  }

  return (
    <div className="content split">
      <div>
        <div className="panel">
          <h2>Picture board</h2>

          <div className="field">
            <label>The story</label>
            <textarea
              value={story}
              rows={7}
              onChange={(e) => setStory(e.target.value)}
              placeholder={"Mira comes home and finds the door already open. "
                + "Nothing is taken, but every photograph has been turned to "
                + "face the wall."}
            />
            <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5 }}>
              Prose, not prompts. It is divided into panels for you, and nothing
              is invented that the story does not contain.
            </div>
          </div>

          <div className="field">
            <label>Who we follow <em>every panel holds this</em></label>
            <textarea
              value={character}
              rows={3}
              onChange={(e) => setCharacter(e.target.value)}
              placeholder="a young woman with short black hair and a red scarf, worn green field jacket"
            />
          </div>

          <div className="field">
            <label>Style</label>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {STYLES.map(([id, label]) => (
                <button
                  key={id}
                  className={"btn small" + (id === style ? " primary" : "")}
                  disabled={busy}
                  onClick={() => setStyle(id)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="field">
            <label>Panels <em>{count}</em></label>
            <input
              type="range" min={2} max={12} value={count} disabled={busy}
              onChange={(e) => setCount(Number(e.target.value))}
            />
          </div>

          <div className="field">
            <label>Model</label>
            <select value={model?.id ?? ""} disabled={busy}
                    onChange={(e) => setModelId(e.target.value)}>
              {usable.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </div>

          {!writerReady && (
            <div className="notice warn" style={{ marginBottom: 12 }}>
              <strong>Needs the prompt writer</strong>
              Dividing a story into panels runs on it — a 2.1 GB download in the
              Models tab. It runs on this Mac; nothing is sent anywhere.
            </div>
          )}

          <div style={{ display: "flex", gap: 7 }}>
            <button
              className="btn primary small"
              disabled={busy || !writerReady || !story.trim()}
              onClick={divide}
            >
              {stage === "dividing" ? "Dividing…" : "Break into panels"}
            </button>
            {panels && (
              <button className="btn primary small" disabled={busy} onClick={draw}>
                {stage === "drawing" ? `Drawing ${done}/${panels.length}…`
                  : stage === "casting" ? "Casting…"
                  : `Draw ${panels.length} panels`}
              </button>
            )}
            {busy && <button className="btn small" onClick={cancel}>Cancel</button>}
          </div>

          {prog && <JobProgress p={prog} label={
            stage === "dividing" ? "Dividing the story"
              : stage === "casting" ? "Casting the character"
              : `Panel ${done + 1} of ${panels?.length ?? 0}`} />}

          {panels && stage === "idle" && (
            <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 9,
                          lineHeight: 1.55 }}>
              Roughly {Math.round((panels.length + 1) * 50 / 60)} minutes for
              {" "}{panels.length} panels and the character sheet, at about
              50 seconds each.
            </div>
          )}
        </div>
      </div>

      <div>
        {panels && (
          <div className="panel">
            <h2>{panels.length} panels</h2>
            {panels.map((p, i) => (
              <div key={i} style={{ display: "flex", gap: 11, marginBottom: 13,
                                    alignItems: "flex-start" }}>
                <div style={{
                  width: 92, height: 92, flex: "0 0 92px", borderRadius: 6,
                  background: "var(--bg-sunk)", overflow: "hidden",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 10.5, color: "var(--text-faint)",
                }}>
                  {drawn[i]
                    ? <img src={vaultUrl(drawn[i]!)} alt={`Panel ${i + 1}`}
                           style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                    : stage === "drawing" && done === i ? "drawing…" : i + 1}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 10, textTransform: "uppercase",
                                letterSpacing: "0.06em", color: "var(--text-faint)" }}>
                    {p.shot}
                  </div>
                  <div style={{ fontSize: 12.5, marginTop: 2 }}>{p.action}</div>
                  <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 2 }}>
                    {p.setting}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {sheet && (
          <div className="panel">
            <h2>Character sheet</h2>
            <img src={vaultUrl(sheet)} alt="Character sheet"
                 style={{ width: "100%", borderRadius: 6 }} />
            <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 6 }}>
              Every panel is drawn against this, which is what keeps the same
              person on the page from one shot to the next.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
