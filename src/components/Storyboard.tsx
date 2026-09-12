import { useEffect, useMemo, useRef, useState } from "react";
import { api, errText, newJobId, onEngineProgress, vaultUrl } from "../lib/api";
import type { Cast, EngineProgress, ModelStatus, Panel } from "../lib/types";
import { ImageDrop, JobProgress } from "./shared";
import { loadPref, savePref } from "../lib/prefs";
import { STYLES, panelPrompt, panelReferences, panelSeed, placePrompt,
         sheetPrompt, styleWords } from "../lib/board";

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

type Stage = "idle" | "dividing" | "enriching" | "casting" | "building" | "drawing";

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
  // The cast, read out of the story rather than typed in. `null` means the
  // story has not been read yet.
  const [cast, setCast] = useState<Cast | null>(
    () => loadPref<Cast | null>("boardCast", null));
  const [reading, setReading] = useState(false);
  /**
   * How many panels one press draws.
   *
   * Drawing a whole board is twenty to sixty minutes, and the style is either
   * right or wrong within the first two panels. Committing to all of it before
   * seeing any of it is the expensive way to find out. A batch is the unit of
   * commitment: draw four, look, fix what is wrong, draw four more.
   */
  const [batch, setBatch] = useState(() => loadPref("boardBatch", 4));
  /**
   * The id every panel of this board is stamped with.
   *
   * It is what makes the board one entry in the library instead of a pile of
   * loose pictures sharing a timestamp. Kept across restarts with the rest of
   * the board, and replaced when a new story is divided.
   */
  const [projectId, setProjectId] = useState<string | null>(
    () => loadPref<string | null>("boardProject", null));


  const [style, setStyle] = useState(() => loadPref("boardStyle", STYLES[0].id));
  const [character, setCharacter] = useState(() => loadPref("boardCharacter", ""));

  const [panels, setPanels] = useState<Panel[] | null>(
    () => loadPref<Panel[] | null>("boardPanels", null));
  const [sheet, setSheet] = useState<string | null>(
    () => loadPref<string | null>("boardSheet", null));
  const [drawn, setDrawn] = useState<(string | null)[]>(
    () => loadPref<(string | null)[]>("boardDrawn", []));
  /** The distinct scenes the panels fall into, in order. */
  const scenes = useMemo(() => {
    const seen: number[] = [];
    for (const p of panels ?? []) {
      const n = Number(p.scene ?? 1) || 1;
      if (!seen.includes(n)) seen.push(n);
    }
    return seen;
  }, [panels]);

  /**
   * Who the panels are drawn against.
   *
   * What you typed wins, because it is the most direct statement of intent.
   * Failing that, the story's own lead -- the most-mentioned person, with the
   * description the story gave them. Failing that, nothing, and the panels are
   * drawn from their own subjects. Shared by the run and by redraw, so a
   * redrawn panel is cast exactly like the one it replaces.
   */
  const who = useMemo(() => {
    const lead = cast?.people.find((p) => p.tier === 1);
    return character.trim() || lead?.description || lead?.name || "";
  }, [character, cast]);

  /** What to call this board in the library: the story's own opening. */
  const projectName = useMemo(() => {
    const first = story.trim().split(/\s+/).slice(0, 7).join(" ");
    return first ? (story.trim().length > first.length ? first + "\u2026" : first)
                 : "Picture board";
  }, [story]);

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
  const [pages, setPages] = useState<string[]>([]);
  /** A page sets panels in rows with gutters; a strip stacks them at one
   *  width for scrolling. Different things, and people want both. */
  const [layout, setLayout] = useState(() => loadPref("boardLayout", "page"));
  /** A character picture the user brought, used instead of casting one.
   *  Their own face, an earlier board's sheet, a photograph -- whatever it
   *  is, it anchors the character better than a description can. */
  const [ownSheet, setOwnSheet] = useState<string[]>([]);
  /** One location sheet per scene, keyed by scene number. The character sheet
   *  holds the person; these hold the rooms. */
  const [placeSheets, setPlaceSheets] = useState<Record<number, string>>(
    () => loadPref<Record<number, string>>("boardPlaces", {}));
  const timer = useRef<number | null>(null);


  // A board is minutes of work, so it survives a quit. Only the text and the
  // vault ids are stored; the pictures themselves stay sealed in the vault.
  useEffect(() => { savePref("boardStory", story); }, [story]);
  useEffect(() => { savePref("boardCast", cast); }, [cast]);
  useEffect(() => { savePref("boardProject", projectId); }, [projectId]);
  useEffect(() => { savePref("boardBatch", batch); }, [batch]);
  useEffect(() => { savePref("boardStyle", style); }, [style]);
  useEffect(() => { savePref("boardCharacter", character); }, [character]);
  useEffect(() => { savePref("boardPanels", panels); }, [panels]);
  useEffect(() => { savePref("boardSheet", sheet); }, [sheet]);
  useEffect(() => { savePref("boardDrawn", drawn); }, [drawn]);
  useEffect(() => { savePref("boardLayout", layout); }, [layout]);
  useEffect(() => { savePref("boardPlaces", placeSheets); }, [placeSheets]);

  const busy = stage !== "idle";

  /** Step one: divide the prose. Nothing is drawn yet. */
  /**
   * Read the story: who is in it, where it happens.
   *
   * This runs on its own, because it costs a little of the text model and no
   * picture time at all. Everything that costs picture time is asked for
   * explicitly; this is not. It is also what makes pasting a story *do*
   * something, instead of sitting there until you type a character in.
   */
  const read = async () => {
    if (!story.trim()) { notify("Paste a story first.", true); return; }
    const id = newJobId();
    setReading(true); setJobId(id);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    try {
      const c = await api.storyCast(id, story);
      setCast(c);
      if (c.people.length === 0) {
        notify("No one named in that story could be verified against the text.");
      }
    } catch (e) {
      notify(errText(e), true);
    } finally {
      un(); setReading(false); setProg(null); setJobId(null);
    }
  };

  const divide = async () => {
    if (!story.trim()) { notify("Write the story first.", true); return; }
    const id = newJobId();
    setStage("dividing"); setJobId(id); setPanels(null); setDrawn([]); setSheet(null);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    try {
      const r = await api.shotList(id, story, null);
      setPanels(r.panels);
      // A fresh division is a different comic, so it gets its own identity
      // rather than adding panels to whatever was in the library before.
      setProjectId(newJobId());
      setDrawn(new Array(r.panels.length).fill(null));
      if (r.out_of_range) {
        // Say it rather than hide it: a division far outside what this much
        // prose should produce usually means the story was cut short or the
        // writer lost the thread, and the panels are worth a look before any
        // of them are drawn.
        notify(
          `Divided into ${r.panels.length} panels; ${r.words} words usually ` +
          `makes ${r.expected_low}\u2013${r.expected_high}. Worth checking.`
        );
      }
    } catch (e) {
      notify(errText(e), true);
    } finally {
      un(); setStage("idle"); setProg(null); setJobId(null);
    }
  };

  /** Which panels have a picture, and which are still waiting. */
  const drawnCount = useMemo(
    () => drawn.filter(Boolean).length, [drawn]);
  const remaining = Math.max(0, (panels?.length ?? 0) - drawnCount);

  /**
   * The indices this press will draw: the next `batch` panels with no picture.
   *
   * Skipping ones that already have a picture is what makes a second press
   * continue rather than start over, and what lets a redrawn panel keep the
   * version you accepted.
   */
  const nextBatch = (): number[] => {
    const want = Math.max(1, Math.min(batch, panels?.length ?? 1));
    const out: number[] = [];
    for (let i = 0; i < (panels?.length ?? 0) && out.length < want; i++) {
      if (!drawn[i]) out.push(i);
    }
    return out;
  };

  /** Step two: cast the character once, then draw every panel against it. */
  const draw = async () => {
    if (!model) { notify("No model that can both generate and edit is installed.", true); return; }
    if (!panels) return;
    // No hard stop for a missing character. The cast is read out of the story,
    // so the common case is that the person is already known; and a story with
    // no one in it -- a place, a mood, a sequence of weather -- is still a
    // board worth drawing. Refusing to start until you had typed a character
    // in was the single thing that made pasting a story feel like it did
    // nothing at all.


    stop.current = false;
    setDone(0);
    setRedraws(new Array(panels.length).fill(0));
    setElapsed(0);
    const started = Date.now();
    if (timer.current) window.clearInterval(timer.current);
    timer.current = window.setInterval(
      () => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    // Only a first run starts from nothing. A later batch adds to what is
    // already there, which is the whole point of drawing in batches.
    setDrawn((d) => {
      const n = new Array(panels.length).fill(null);
      for (let i = 0; i < Math.min(d.length, n.length); i++) n[i] = d[i];
      return n;
    });

    // The sheet is the anchor. Every panel references it, which is the whole
    // reason the character survives from one shot to the next -- so if the
    // user brought their own, there is nothing to cast.
    if (ownSheet.length > 0) {
      setSheet(ownSheet[0]);
      const places = await buildPlaces();
      setStage("drawing");
      for (const i of nextBatch()) {
        if (stop.current) { notify("Stopped."); break; }
        const ok = await drawOne(i, ownSheet[0], places);
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
        prompt: sheetPrompt(style, who),
        negative_prompt: null,
        width: PANEL_W, height: PANEL_H,
        steps: model.steps_default || 4,
        guidance: Math.min(1.0, model.guidance_max),
        seed: 7, count: 1, images: [], image_strength: null, i2i_mode: null,
        low_ram: true, preview: false, cache_limit_gb: null,
        allow_over_budget: false, loras: [], mask: null,
        outpaint_padding: null, outpaint_fill: null,
        // Reference material for this board, not a loose picture:
        // the sheets live inside the comic they were drawn for.
        project: projectId, project_name: projectName,
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
    const places = await buildPlaces();
    setStage("drawing");
    for (const i of nextBatch()) {
      if (stop.current) { notify("Stopped."); break; }
      const ok = await drawOne(i, sheetId, places);
      if (!ok && stop.current) break;
    }
    if (timer.current) { window.clearInterval(timer.current); timer.current = null; }
    setStage("idle"); setProg(null); setJobId(null);
  };

  /** Draw each scene's room once, empty, for its panels to be set inside.
   *
   *  The same mechanism that holds the character, applied to the place. A
   *  described room is drawn differently every time; a referenced one is not.
   */
  const buildPlaces = async (): Promise<Record<number, string>> => {
    if (!model || !panels) return {};
    const wanted = new Map<number, string>();
    for (const p of panels) {
      const key = p.scene ?? 1;
      const place = (p.place ?? "").trim();
      if (place && !wanted.has(key) && !placeSheets[key]) wanted.set(key, place);
    }
    if (wanted.size === 0) return placeSheets;

    setStage("building");
    const made: Record<number, string> = { ...placeSheets };
    for (const [scene, place] of wanted) {
      if (stop.current) break;
      const id = newJobId();
      setJobId(id);
      const un = await onEngineProgress((e) => { if (e.job_id === id) setProg(e); });
      try {
        const res = await api.generate({
          job_id: id, model_id: model.id,
          prompt: placePrompt(style, place),
          negative_prompt: null,
          width: PANEL_W, height: PANEL_H,
          steps: model.steps_default || 4,
          guidance: Math.min(1.0, model.guidance_max),
          seed: 21 + scene, count: 1, images: [], image_strength: null,
          i2i_mode: null, low_ram: true, preview: false, cache_limit_gb: null,
          allow_over_budget: false, loras: [], mask: null,
          outpaint_padding: null, outpaint_fill: null,
          // Reference material for this board, not a loose picture:
          // the sheets live inside the comic they were drawn for.
          project: projectId, project_name: projectName,
        });
        if (res[0]) made[scene] = res[0];
        onProduced();
      } catch (e) {
        // A missing room is survivable: the panel falls back to its written
        // description, which is what it used before this existed.
        notify(`Scene ${scene}: ${errText(e)}`, true);
      } finally {
        un();
      }
    }
    setPlaceSheets(made);
    return made;
  };

  /** Draw a single panel against the sheet. Shared by the run and by redraw,
   *  so a board with one bad panel costs one panel to fix rather than six. */
  const drawOne = async (i: number, sheetId: string,
                         places: Record<number, string> = placeSheets): Promise<boolean> => {
    if (!model || !panels) return false;
    const id = newJobId();
    setJobId(id);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    try {
      const res = await api.editImage({
        job_id: id, model_id: model.id,
        prompt: panelPrompt(panels[i], style, who),
        negative_prompt: null,
        width: PANEL_W, height: PANEL_H,
        steps: model.steps_default || 4,
        guidance: Math.min(1.0, model.guidance_max),
        // Redrawing the same panel with the same seed reproduces the picture
        // that was already rejected, so a redraw moves the seed on.
        seed: panelSeed(i, redraws[i] ?? 0),
        count: 1,
        images: panelReferences(panels[i], sheetId,
                                places[panels[i].scene ?? 1] ?? null),
        // "edit" is the reference-conditioned route -- the model is handed
        // the sheet and the panel description together. "latent" would
        // instead redraw the sheet itself, which is not what a panel is.
        image_strength: null, i2i_mode: "edit",
        low_ram: true, preview: false, cache_limit_gb: null,
        allow_over_budget: false, loras: [], mask: null,
        outpaint_padding: null, outpaint_fill: null,
        // Every panel is stamped with the board it belongs to, so the library
        // shows one comic instead of a pile of pictures that happen to share
        // a timestamp. The index keeps them in reading order.
        project: projectId, project_name: projectName, project_index: i,
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

  /** Work each panel up into a scene: surface, background, light.
   *
   *  Its own step, and readable before anything is drawn. The division is
   *  terse on purpose -- that is the right level for judging whether the
   *  story was cut correctly -- and far too thin to draw from.
   */
  const enrich = async () => {
    if (!panels) return;
    const id = newJobId();
    setStage("enriching"); setJobId(id);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    try {
      const r = await api.enrichPanels(id, panels, styleWords(style));
      setPanels(r.panels);
      const filled = r.panels.filter((p) => (p.description ?? "").trim()).length;
      notify(filled === r.panels.length
        ? "Every panel worked up. Read them before drawing."
        : `${filled} of ${r.panels.length} worked up; the rest stay as written.`);
    } catch (e) {
      notify(errText(e), true);
    } finally {
      un(); setStage("idle"); setProg(null); setJobId(null);
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
    type Ready = { id: string; caption: string; shot: Panel["shot"];
                   dialogue: { speaker: string; text: string }[]; scene: number };
    const pairs = panels
      .map((p, i) => ({ id: drawn[i], caption: p.caption, shot: p.shot,
                        dialogue: p.dialogue ?? [], scene: p.scene ?? 1 }))
      .filter((x): x is Ready => Boolean(x.id));
    if (pairs.length === 0) { notify("Draw at least one panel first.", true); return; }

    setComposing(true);
    try {
      const made = await api.composeBoard(
        newJobId(),
        pairs.map((p) => p.id),
        pairs.map((p) => p.caption),
        pairs.map((p) => p.shot),
        pairs.map((p) => p.dialogue ?? []),
        pairs.map((p) => p.scene),
        layout);
      setPages(made);
      onProduced();
      notify(made.length === 1
        ? `Composed one page from ${pairs.length} panels. It is in the Vault.`
        : `Composed ${made.length} pages from ${pairs.length} panels, broken `
          + "where the scenes change. They are in the Vault.");
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
            {/* Reading costs a little of the text model and no picture time,
                so it is offered the moment there is a story -- rather than the
                tab sitting inert until you type a character in yourself. */}
            {story.trim() && !cast && (
              <button
                className="btn small"
                style={{ marginTop: 8 }}
                disabled={reading || busy}
                onClick={read}
              >
                {reading ? "Reading the story\u2026" : "Read the story"}
              </button>
            )}
          </div>

          {cast && (
            <div className="field">
              <label>
                The cast <em>read from your story</em>
              </label>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {cast.people.map((p) => (
                  <button
                    key={p.name}
                    className="pill"
                    title={
                      (p.description || "The story never says what they look like.")
                      + `  \u00b7 ${p.mentions} mentions`
                    }
                    style={{
                      cursor: "pointer",
                      opacity: p.tier === 1 ? 1 : p.tier === 2 ? 0.8 : 0.6,
                    }}
                    onClick={() => setCharacter(p.description || p.name)}
                  >
                    {p.name}
                  </button>
                ))}
                {cast.people.length === 0 && (
                  <span style={{ fontSize: 11, color: "var(--text-faint)" }}>
                    No one the story names could be verified in the text.
                  </span>
                )}
              </div>
              {cast.places.length > 0 && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 }}>
                  {cast.places.map((pl) => (
                    <span key={pl.name} className="pill" style={{ opacity: 0.7 }}
                          title={pl.description || "The story never describes it."}>
                      {pl.name}
                    </span>
                  ))}
                </div>
              )}
              <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 6 }}>
                Click a name to cast them as the one we follow. Names the story
                never actually uses are dropped rather than drawn.
              </div>
            </div>
          )}

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

          {/* No slider. The number of panels is whatever the story turns out
              to need, worked out by dividing it into beats, and reported after
              the fact. A count chosen before the story is read is a guess --
              and a slider capped at twelve is why a whole chapter came back as
              twelve panels no matter what was in it. */}
          {panels && (
            <div className="field">
              <label>Panels</label>
              <div style={{ fontSize: 12, color: "var(--text-faint)", lineHeight: 1.6 }}>
                <b style={{ color: "var(--text)" }}>{panels.length}</b>
                {" panels, from "}
                <b style={{ color: "var(--text)" }}>{scenes.length}</b>
                {scenes.length === 1 ? " scene" : " scenes"}
              </div>
            </div>
          )}

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
              <button className="btn small" disabled={busy} onClick={() => void enrich()}>
                {stage === "enriching" ? "Working up…" : "Add detail"}
              </button>
            )}
            {panels && remaining > 0 && (
              <>
                {/* How much you are committing to, before you commit. The
                    button carries its own cost, so a wrong style costs one
                    batch to discover instead of the whole board. */}
                {!busy && (
                  <span className="stepper" title="Panels per press">
                    <button type="button"
                            onClick={() => setBatch((n) => Math.max(1, n - 1))}>
                      &minus;
                    </button>
                    <span>{Math.min(batch, remaining)} at a time</span>
                    <button type="button"
                            onClick={() => setBatch((n) => Math.min(24, n + 1))}>
                      +
                    </button>
                  </span>
                )}
                <button className="btn primary small" disabled={busy} onClick={draw}>
                  {stage === "drawing"
                    ? `Drawing ${done}/${panels.length}\u2026`
                    : stage === "building" ? "Building the rooms\u2026"
                    : stage === "casting" ? "Casting\u2026"
                    : drawnCount === 0
                      ? `Draw ${Math.min(batch, remaining)} of ${panels.length}`
                      : `Draw next ${Math.min(batch, remaining)}`}
                </button>
              </>
            )}
            {panels && remaining === 0 && drawnCount > 0 && !busy && (
              <span style={{ fontSize: 11, color: "var(--good)" }}>
                All {panels.length} drawn
              </span>
            )}
            {busy && <button className="btn small" onClick={cancel}>Cancel</button>}
            {!busy && panels && drawn.some(Boolean) && (
              <>
                <select
                  value={layout}
                  style={{ width: "auto", fontSize: 11, padding: "2px 6px" }}
                  onChange={(e) => setLayout(e.target.value)}
                >
                  <option value="page">Page</option>
                  <option value="strip">Scrolling strip</option>
                </select>
                <button className="btn small" onClick={() => void compose()}>
                  {composing ? "Composing…" : "Make a page"}
                </button>
              </>
            )}
          </div>

          {prog && <JobProgress p={prog} label={
            stage === "dividing" ? "Dividing the story"
              : stage === "casting" ? "Casting the character"
              : `Panel ${done + 1} of ${panels?.length ?? 0}`} />}
        {panels && !busy && drawnCount > 0 && remaining > 0 && (
          <div style={{ fontSize: 11, color: "var(--text-faint)", marginTop: 8 }}>
            <b style={{ color: "var(--text)" }}>{drawnCount}</b> of {panels.length} drawn.
            {" "}Look at them before drawing more \u2014 a note on a wrong panel
            costs one redraw, and a wrong style caught now costs one batch.
          </div>
        )}

          {panels && stage === "idle" && drawn.every((d) => !d) && (
            <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 9,
                          lineHeight: 1.55 }}>
              Expect roughly {Math.ceil((Math.min(batch, remaining) + (ownSheet.length ? 0 : 1)) * 0.6)}–
              {Math.ceil((Math.min(batch, remaining) + (ownSheet.length ? 0 : 1)) * 2)} minutes for
              {" "}{Math.min(batch, remaining)} of {panels.length} panels{ownSheet.length ? "" : " and the character sheet"}. The range is
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
              <div key={`g${i}`}>
              {(p.place ?? "").trim()
                && (i === 0 || panels[i - 1].scene !== p.scene) && (
                <div style={{ fontSize: 10.5, color: "var(--text-dim)",
                              background: "var(--bg-sunk)", borderRadius: 5,
                              padding: "7px 9px", margin: "4px 0 10px",
                              lineHeight: 1.5 }}>
                  <b>{p.scene_title || `Scene ${p.scene ?? 1}`}</b> — every
                  panel here is drawn in this place:
                  <div style={{ marginTop: 3 }}>{p.place}</div>
                </div>
              )}
              <div style={{ display: "flex", gap: 11, marginBottom: 15,
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
                  <input
                    type="text"
                    value={(p.dialogue ?? []).map(
                      (d) => (d.speaker ? `${d.speaker}: ` : "") + d.text).join(" / ")}
                    disabled={busy}
                    placeholder="dialogue — Name: what they say"
                    style={{ fontSize: 12, padding: "4px 6px", marginTop: 3 }}
                    onChange={(e) => edit(i, {
                      // "Name: line / Name: line" is quicker to correct than
                      // a pair of fields per speaker, and matches how the
                      // writer returns it.
                      dialogue: e.target.value.split("/").map((raw) => {
                        const [a, ...rest] = raw.split(":");
                        return rest.length
                          ? { speaker: a.trim(), text: rest.join(":").trim() }
                          : { speaker: "", text: a.trim() };
                      }).filter((d) => d.text),
                    })}
                  />
                  {(p.description ?? "").trim() && (
                    <textarea
                      value={p.description} disabled={busy} rows={3}
                      style={{ fontSize: 11.5, padding: "5px 6px", marginTop: 3,
                               color: "var(--text-dim)" }}
                      onChange={(e) => edit(i, { description: e.target.value })}
                    />
                  )}
                </div>
              </div>
              </div>
            ))}
          </div>
        )}

        {pages.length > 0 && (
          <div className="panel">
            <h2>{pages.length === 1 ? "The page" : `${pages.length} pages`}</h2>
            {pages.map((id, i) => (
              <div key={id} style={{ marginBottom: 14 }}>
                {pages.length > 1 && (
                  <div style={{ fontSize: 10.5, color: "var(--text-faint)",
                                marginBottom: 4 }}>
                    Page {i + 1} of {pages.length}
                  </div>
                )}
                <img src={vaultUrl(id)} alt={`Page ${i + 1}`}
                     style={{ width: "100%", borderRadius: 6 }} />
              </div>
            ))}
            <div style={{ fontSize: 10.5, color: "var(--text-faint)" }}>
              Sealed in the Vault like anything else. Export them from there.
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
