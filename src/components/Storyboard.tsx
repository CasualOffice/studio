import { useEffect, useMemo, useRef, useState } from "react";
import { api, errText, newJobId, onEngineProgress, vaultUrl } from "../lib/api";
import type { EngineProgress, ModelStatus, Panel } from "../lib/types";
import { ImageDrop, JobProgress } from "./shared";
import { loadPref, savePref } from "../lib/prefs";
import { STYLES, panelPrompt, panelSeed, sheetPrompt } from "../lib/board";

/**
 * A story, divided into panels, drawn as one consistent character.
 *
 * The pieces here are proven separately: the writer divides prose reliably,
 * and a character sheet fed back as a reference holds the character across
 * shots. This puts them in order and keeps the one thing that makes the
 * result read as a story rather than a set of unrelated pictures -- the same
 * sheet references every panel, and the same style words lead every prompt.
 */

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

  const [story, setStory] = useState(() => loadPref("boardStory", ""));
  const [count, setCount] = useState(() => loadPref("boardCount", 6));
  const [style, setStyle] = useState(() => loadPref("boardStyle", STYLES[0].id));
  const [character, setCharacter] = useState(() => loadPref("boardCharacter", ""));

  const [panels, setPanels] = useState<Panel[] | null>(
    () => loadPref<Panel[] | null>("boardPanels", null));
  const [sheet, setSheet] = useState<string | null>(
    () => loadPref<string | null>("boardSheet", null));
  const [drawn, setDrawn] = useState<(string | null)[]>(
    () => loadPref<(string | null)[]>("boardDrawn", []));
  const [stage, setStage] = useState<Stage>("idle");
  const [prog, setProg] = useState<EngineProgress | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [done, setDone] = useState(0);
  /** Set when the user cancels, checked between panels. Cancelling the job in
   *  flight only ever stopped one panel; the loop then started the next. */
  const stop = useRef(false);
  /** How many times each panel has been redrawn, so a retry gets a new seed. */
  const [redraws, setRedraws] = useState<number[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [composing, setComposing] = useState(false);
  const [page, setPage] = useState<string | null>(null);
  /** A character picture the user brought, used instead of casting one.
   *  Their own face, an earlier board's sheet, a photograph -- whatever it
   *  is, it anchors the character better than a description can. */
  const [ownSheet, setOwnSheet] = useState<string[]>([]);
  const timer = useRef<number | null>(null);


  // A board is minutes of work, so it survives a quit. Only the text and the
  // vault ids are stored; the pictures themselves stay sealed in the vault.
  useEffect(() => { savePref("boardStory", story); }, [story]);
  useEffect(() => { savePref("boardCount", count); }, [count]);
  useEffect(() => { savePref("boardStyle", style); }, [style]);
  useEffect(() => { savePref("boardCharacter", character); }, [character]);
  useEffect(() => { savePref("boardPanels", panels); }, [panels]);
  useEffect(() => { savePref("boardSheet", sheet); }, [sheet]);
  useEffect(() => { savePref("boardDrawn", drawn); }, [drawn]);

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

  /** Step two: cast the character once, then draw every panel against it. */
  const draw = async () => {
    if (!model) { notify("No model that can both generate and edit is installed.", true); return; }
    if (!panels) return;
    if (!character.trim() && ownSheet.length === 0) {
      notify("Describe the character, or bring a picture of them.", true);
      return;
    }

    stop.current = false;
    setDone(0);
    setRedraws(new Array(panels.length).fill(0));
    setElapsed(0);
    const started = Date.now();
    if (timer.current) window.clearInterval(timer.current);
    timer.current = window.setInterval(
      () => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    setDrawn(new Array(panels.length).fill(null));

    // The sheet is the anchor. Every panel references it, which is the whole
    // reason the character survives from one shot to the next -- so if the
    // user brought their own, there is nothing to cast.
    if (ownSheet.length > 0) {
      setSheet(ownSheet[0]);
      setStage("drawing");
      for (let i = 0; i < panels.length; i++) {
        if (stop.current) { notify(`Stopped after ${i} panels.`); break; }
        const ok = await drawOne(i, ownSheet[0]);
        if (!ok && stop.current) break;
      }
      if (timer.current) { window.clearInterval(timer.current); timer.current = null; }
      setStage("idle"); setProg(null); setJobId(null);
      return;
    }

    const castId = newJobId();
    setStage("casting"); setJobId(castId);
    let un = await onEngineProgress((p) => { if (p.job_id === castId) setProg(p); });
    let sheetId: string;
    try {
      const res = await api.generate({
        job_id: castId, model_id: model.id,
        prompt: sheetPrompt(style, character),
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
      if (timer.current) { window.clearInterval(timer.current); timer.current = null; }
      un(); setStage("idle"); setProg(null); setJobId(null);
      return;
    }
    un();

    // Panels run one at a time. Two models never share this machine's memory,
    // and a board of six is minutes of work either way.
    setStage("drawing");
    for (let i = 0; i < panels.length; i++) {
      if (stop.current) { notify(`Stopped after ${i} panels.`); break; }
      const ok = await drawOne(i, sheetId);
      if (!ok && stop.current) break;
    }
    if (timer.current) { window.clearInterval(timer.current); timer.current = null; }
    setStage("idle"); setProg(null); setJobId(null);
  };

  /** Draw a single panel against the sheet. Shared by the run and by redraw,
   *  so a board with one bad panel costs one panel to fix rather than six. */
  const drawOne = async (i: number, sheetId: string): Promise<boolean> => {
    if (!model || !panels) return false;
    const id = newJobId();
    setJobId(id);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    try {
      const res = await api.editImage({
        job_id: id, model_id: model.id,
        prompt: panelPrompt(panels[i], style, character),
        negative_prompt: null,
        width: PANEL_W, height: PANEL_H,
        steps: model.steps_default || 4,
        guidance: Math.min(1.0, model.guidance_max),
        // Redrawing the same panel with the same seed reproduces the picture
        // that was already rejected, so a redraw moves the seed on.
        seed: panelSeed(i, redraws[i] ?? 0),
        count: 1,
        images: [sheetId],
        // "edit" is the reference-conditioned route -- the model is handed
        // the sheet and the panel description together. "latent" would
        // instead redraw the sheet itself, which is not what a panel is.
        image_strength: null, i2i_mode: "edit",
        low_ram: true, preview: false, cache_limit_gb: null,
        allow_over_budget: false, loras: [], mask: null,
        outpaint_padding: null, outpaint_fill: null,
      });
      setDrawn((d) => { const n = [...d]; n[i] = res[0] ?? null; return n; });
      setDone((v) => Math.max(v, i + 1));
      onProduced();
      return true;
    } catch (e) {
      const msg = errText(e);
      notify(`Panel ${i + 1}: ${msg}`, true);
      if (msg.includes("ancelled")) stop.current = true;
      return false;
    } finally {
      un();
    }
  };

  /** Change one field of one panel. The division is a draft, not a verdict. */
  const edit = (i: number, patch: Partial<Panel>) =>
    setPanels((ps) => {
      if (!ps) return ps;
      const n = [...ps];
      n[i] = { ...n[i], ...patch };
      return n;
    });

  /** Redraw one panel, keeping the rest of the board. */
  const redraw = async (i: number) => {
    if (!sheet) { notify("Draw the board first.", true); return; }
    stop.current = false;
    setRedraws((r) => { const n = [...r]; n[i] = (n[i] ?? 0) + 1; return n; });
    setStage("drawing");
    await drawOne(i, sheet);
    setStage("idle"); setProg(null); setJobId(null);
  };

  /** Stack the drawn panels into one page, captions underneath.
   *
   *  Only the panels that actually exist go in: a half-drawn board still
   *  makes a readable page, and a gap for a missing one would not.
   */
  const compose = async () => {
    if (!panels) return;
    const pairs = panels
      .map((p, i) => ({ id: drawn[i], caption: p.caption }))
      .filter((x): x is { id: string; caption: string } => Boolean(x.id));
    if (pairs.length === 0) { notify("Draw at least one panel first.", true); return; }

    setComposing(true);
    try {
      const id = await api.composeBoard(
        newJobId(), pairs.map((p) => p.id), pairs.map((p) => p.caption));
      setPage(id);
      onProduced();
      notify(`Composed a page from ${pairs.length} panels. It is in the Vault.`);
    } catch (e) {
      notify(errText(e), true);
    } finally {
      setComposing(false);
    }
  };

  const cancel = async () => {
    stop.current = true;
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
    <div className="content split" style={{ padding: 0 }}>
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
            <label>
              Or bring your own <em>optional</em>
            </label>
            <ImageDrop
              images={ownSheet}
              onChange={setOwnSheet}
              max={1}
              onError={(m) => notify(m, true)}
            />
            <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5,
                          lineHeight: 1.55 }}>
              {ownSheet.length > 0
                ? "Every panel is drawn against this picture. No character sheet "
                  + "is generated, so the board starts a minute sooner."
                : "A picture of the character — a photo, a drawing, a sheet from "
                  + "an earlier board. It anchors them better than words can."}
            </div>
          </div>

          <div className="field">
            <label>Style</label>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {STYLES.map(({ id, label }) => (
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
            {!busy && panels && drawn.some(Boolean) && (
              <button className="btn small" onClick={() => void compose()}>
                {composing ? "Composing…" : "Make a page"}
              </button>
            )}
          </div>

          {prog && <JobProgress p={prog} label={
            stage === "dividing" ? "Dividing the story"
              : stage === "casting" ? "Casting the character"
              : `Panel ${done + 1} of ${panels?.length ?? 0}`} />}

          {panels && stage === "idle" && drawn.every((d) => !d) && (
            <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 9,
                          lineHeight: 1.55 }}>
              Expect roughly {Math.ceil((panels.length + (ownSheet.length ? 0 : 1)) * 0.6)}–
              {Math.ceil((panels.length + (ownSheet.length ? 0 : 1)) * 2)} minutes for
              {" "}{panels.length} panels{ownSheet.length ? "" : " and the character sheet"}. The range is
              wide because this Mac slows as it warms: measured panels ran 34
              seconds cold and 123 seconds after a few minutes of work.
            </div>
          )}
          {stage === "drawing" && elapsed > 0 && (
            <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 9 }}>
              {Math.floor(elapsed / 60)}m {elapsed % 60}s elapsed
              {done > 0 && ` · ${Math.round(elapsed / done)}s a panel so far`}
            </div>
          )}
        </div>
      </div>

      <div>
        {panels && (
          <div className="panel">
            <h2>{panels.length} panels</h2>
            <div style={{ fontSize: 10.5, color: "var(--text-faint)",
                          marginBottom: 11, lineHeight: 1.55 }}>
              Edit anything here before drawing. Fixing a panel now costs
              nothing; fixing it afterwards costs a minute of drawing.
            </div>
            {panels.map((p, i) => (
              <div key={i} style={{ display: "flex", gap: 11, marginBottom: 15,
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
                  <div style={{ display: "flex", gap: 5, alignItems: "center",
                                marginBottom: 4 }}>
                    <select
                      value={p.shot}
                      disabled={busy}
                      style={{ width: "auto", fontSize: 11, padding: "2px 5px" }}
                      onChange={(e) => edit(i, { shot: e.target.value as Panel["shot"] })}
                    >
                      <option value="wide">wide</option>
                      <option value="medium">medium</option>
                      <option value="close-up">close-up</option>
                    </select>
                    <label style={{ fontSize: 10.5, color: "var(--text-dim)",
                                    display: "flex", alignItems: "center", gap: 4,
                                    cursor: "pointer", width: "auto" }}>
                      <input
                        type="checkbox"
                        checked={p.character_in_frame}
                        disabled={busy}
                        style={{ width: "auto", margin: 0 }}
                        onChange={(e) => edit(i, { character_in_frame: e.target.checked })}
                      />
                      in frame
                    </label>
                    <div style={{ flex: 1 }} />
                    {drawn[i] && (
                      <button className="btn small" disabled={busy}
                              style={{ fontSize: 10.5, padding: "2px 8px" }}
                              onClick={() => void redraw(i)}>
                        Redraw
                      </button>
                    )}
                  </div>
                  <input
                    type="text" value={p.subject} disabled={busy}
                    placeholder="who or what is in frame"
                    style={{ fontSize: 12, padding: "4px 6px", marginBottom: 3 }}
                    onChange={(e) => edit(i, { subject: e.target.value })}
                  />
                  <input
                    type="text" value={p.action} disabled={busy}
                    placeholder="what is happening"
                    style={{ fontSize: 12, padding: "4px 6px", marginBottom: 3 }}
                    onChange={(e) => edit(i, { action: e.target.value })}
                  />
                  <input
                    type="text" value={p.setting} disabled={busy}
                    placeholder="where"
                    style={{ fontSize: 12, padding: "4px 6px", marginBottom: 3 }}
                    onChange={(e) => edit(i, { setting: e.target.value })}
                  />
                  <input
                    type="text" value={p.caption} disabled={busy}
                    placeholder="caption — what the picture cannot say"
                    style={{ fontSize: 12, padding: "4px 6px",
                             fontStyle: p.caption ? "italic" : "normal" }}
                    onChange={(e) => edit(i, { caption: e.target.value })}
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {page && (
          <div className="panel">
            <h2>The page</h2>
            <img src={vaultUrl(page)} alt="Composed page"
                 style={{ width: "100%", borderRadius: 6 }} />
            <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 6 }}>
              Sealed in the Vault like anything else. Export it from there.
            </div>
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
