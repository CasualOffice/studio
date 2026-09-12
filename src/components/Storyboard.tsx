import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { api, errText, newJobId, onEngineProgress, vaultUrl } from "../lib/api";
import type { Cast, EngineProgress, ModelStatus, Panel } from "../lib/types";
import { ImageDrop, JobProgress } from "./shared";
import { loadPref, savePref } from "../lib/prefs";
import { loadBoardDraft, type BoardDraft, type Coverage } from "../lib/boardDraft";
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

  const [story, setStory] = useState("");
  // The cast, read out of the story rather than typed in. `null` means the
  // story has not been read yet.
  const [cast, setCast] = useState<Cast | null>(null);
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
  const [projectId, setProjectId] = useState<string | null>(null);


  const [style, setStyle] = useState(() => loadPref("boardStyle", STYLES[0].id));
  const [character, setCharacter] = useState("");

  const [panels, setPanels] = useState<Panel[] | null>(null);
  const [sheet, setSheet] = useState<string | null>(null);
  const [drawn, setDrawn] = useState<(string | null)[]>([]);
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
  const lead = useMemo(() => cast?.people.find((p) => p.tier === 1), [cast]);
  const personBrief = (name: string) => {
    const person = cast?.people.find((p) => p.name.toLowerCase() === name.toLowerCase());
    if (!person) return name;
    if (person === lead && character.trim()) return `${person.name}: ${character.trim()}`;
    return `${person.name}: ${person.description || "appearance not specified"}`;
  };
  const castForSheet = useMemo(() => {
    const major = cast?.people.filter((p) => p.tier <= 2).slice(0, 4) ?? [];
    if (major.length === 0) return character.trim();
    return major.map((p) => p === lead && character.trim()
      ? `${p.name}: ${character.trim()}`
      : `${p.name}: ${p.description || "appearance not specified"}`).join("\n");
  }, [cast, character, lead]);
  const panelWho = (panel: Panel) => {
    const named = panel.characters?.map(personBrief).filter(Boolean) ?? [];
    if (named.length > 0) return named.join("; ");
    return panel.character_in_frame
      ? (character.trim() || lead?.description || lead?.name || "") : "";
  };

  /** What to call this board in the library: the story's own opening. */
  const projectName = useMemo(() => {
    const first = story.trim().split(/\s+/).slice(0, 7).join(" ");
    return first ? (story.trim().length > first.length ? first + "\u2026" : first)
                 : "Picture board";
  }, [story]);

  /** Which panel the notes rail is showing. The board owns the selection. */
  const [selected, setSelected] = useState(0);

  const [stage, setStage] = useState<Stage>("idle");
  const [prog, setProg] = useState<EngineProgress | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [done, setDone] = useState(0);
  /** Set when the user cancels, checked between panels. Cancelling the job in
   *  flight only ever stopped one panel; the loop then started the next. */
  const stop = useRef(false);
  /** How many times each panel has been redrawn, so a retry gets a new seed. */
  const [redraws, setRedraws] = useState<number[]>([]);
  /**
   * A written correction for one panel, kept with it.
   *
   * The repair for a wrong frame is a sentence about what is wrong with it --
   * "older, grey at the temples", "pull back so the window is in frame" --
   * not another roll of the dice. It is appended to that panel's brief and
   * survives every later redraw, so the panel stays the one you asked for.
   */
  const [notes, setNotes] = useState<Record<number, string>>({});
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
  const [placeSheets, setPlaceSheets] = useState<Record<number, string>>({});
  const [coverage, setCoverage] = useState<Coverage | null>(null);
  const timer = useRef<number | null>(null);
  const [draftLoaded, setDraftLoaded] = useState(false);


  // A Board contains the manuscript and creative history. Load it from the
  // encrypted vault, migrating old plaintext preferences exactly once.
  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const draft: Partial<BoardDraft> | null = await loadBoardDraft();
        if (!live || !draft) return;
        setStory(typeof draft.story === "string" ? draft.story : "");
        setCast(draft.cast ?? null);
        setProjectId(draft.projectId ?? null);
        setCharacter(typeof draft.character === "string" ? draft.character : "");
        setPanels(Array.isArray(draft.panels) ? draft.panels : null);
        setSheet(typeof draft.sheet === "string" ? draft.sheet : null);
        setDrawn(Array.isArray(draft.drawn) ? draft.drawn : []);
        setRedraws(Array.isArray(draft.redraws) ? draft.redraws : []);
        setNotes(draft.notes ?? {});
        setPages(Array.isArray(draft.pages) ? draft.pages : []);
        setOwnSheet(Array.isArray(draft.ownSheet) ? draft.ownSheet : []);
        setPlaceSheets(draft.placeSheets ?? {});
        setCoverage(draft.coverage ?? null);
      } catch (e) {
        if (live) notify(`Could not restore the encrypted Board draft: ${errText(e)}`, true);
      } finally {
        if (live) setDraftLoaded(true);
      }
    })();
    return () => { live = false; };
  }, [notify]);

  useEffect(() => {
    if (!draftLoaded) return;
    const draft: BoardDraft = {
      version: 1, story, cast, projectId, character, panels, sheet, drawn,
      redraws, notes, pages, ownSheet, placeSheets,
      coverage,
    };
    const timeout = window.setTimeout(() => {
      void api.boardStateSet(JSON.stringify(draft)).catch((e) =>
        notify(`Could not save the encrypted Board draft: ${errText(e)}`, true));
    }, 350);
    return () => window.clearTimeout(timeout);
  }, [draftLoaded, story, cast, projectId, character, panels, sheet, drawn,
      redraws, notes, pages, ownSheet, placeSheets, coverage, notify]);

  // These are interface choices, not project content, so they remain prefs.
  useEffect(() => { savePref("boardBatch", batch); }, [batch]);
  useEffect(() => { savePref("boardStyle", style); }, [style]);
  useEffect(() => { savePref("boardLayout", layout); }, [layout]);

  const busy = stage !== "idle" || reading;

  const changeStory = (next: string) => {
    if (next === story) return;
    setStory(next);
    // Every downstream decision was made from the previous manuscript. Keeping
    // any of it would quietly mix two different stories.
    setCast(null); setPanels(null); setProjectId(null); setSheet(null);
    setDrawn([]); setRedraws([]); setNotes({}); setPages([]);
    setPlaceSheets({}); setCoverage(null); setSelected(0);
  };

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
    // Everything keyed by panel index has to go with the panels. A note is a
    // correction to one moment in the story; carried across a fresh division
    // it silently attaches to a different moment, and the panel it was written
    // for is drawn without it. Same for the redraw counters, which move the
    // seed on, and for the selection, which can now point past the end.
    setNotes({}); setRedraws([]); setSelected(0); setPages([]);
    // Rooms are keyed by scene number, and scene 1 of a new story is not scene
    // 1 of the last one. Kept, they would be reused: buildPlaces skips any
    // scene that already has a sheet, so the new comic would quietly inherit
    // the previous comic's rooms.
    setPlaceSheets({});
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    try {
      const r = await api.shotList(id, story, null);
      setPanels(r.panels);
      setCoverage(r.coverage ?? null);
      // A fresh division is a different comic, so it gets its own identity
      // rather than adding panels to whatever was in the library before.
      setProjectId(newJobId());
      setDrawn(new Array(r.panels.length).fill(null));
      if (r.too_long) {
        // Sixty panels is as much as one board holds, so a chapter this long
        // becomes a part of itself. Say that, rather than quoting a range.
        notify(
          `${r.words} words is more than one board holds, so this is the ` +
          `first ${r.panels.length} panels of it. Split the story and ` +
          `divide each part for the rest.`
        );
      } else if (r.out_of_range) {
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
  const preparedCount = panels?.filter((p) => (p.description ?? "").trim()).length ?? 0;
  // Every panel still waiting for a picture has to be worked up first. Gating
  // on "at least one" let a single prepared panel unlock drawing all of them,
  // and the rest were drawn from the terse fields -- which is the difference
  // between a scene and a caption, and costs a minute of drawing each to find
  // out.
  const unpreparedRemaining = panels?.filter(
    (p, i) => !drawn[i] && !(p.description ?? "").trim()).length ?? 0;
  const workflow = !cast ? "read" : !panels ? "plan" : unpreparedRemaining > 0
    ? "prepare" : remaining > 0 ? "draw" : "compose";

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
    setRedraws((current) => Array.from(
      { length: panels.length }, (_, i) => current[i] ?? 0));
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
    const existingSheet = ownSheet[0] ?? sheet;
    if (existingSheet || !castForSheet.trim()) {
      setSheet(existingSheet ?? null);
      const places = await buildPlaces();
      setStage("drawing");
      for (const i of nextBatch()) {
        if (stop.current) { notify("Stopped."); break; }
        const ok = await drawOne(i, existingSheet ?? null, places);
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
        prompt: sheetPrompt(style, castForSheet),
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
  const drawOne = async (i: number, sheetId: string | null,
                         places: Record<number, string> = placeSheets,
                         redrawCount = redraws[i] ?? 0): Promise<boolean> => {
    if (!model || !panels) return false;
    const id = newJobId();
    setJobId(id);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    const refs = panelReferences(panels[i], sheetId,
                                 places[panels[i].scene ?? 1] ?? null);
    try {
      const params = {
        job_id: id, model_id: model.id,
        prompt: panelPrompt(panels[i], style, panelWho(panels[i]), notes[i]),
        negative_prompt: null,
        width: PANEL_W, height: PANEL_H,
        steps: model.steps_default || 4,
        guidance: Math.min(1.0, model.guidance_max),
        // Redrawing the same panel with the same seed reproduces the picture
        // that was already rejected, so a redraw moves the seed on.
        seed: panelSeed(i, redrawCount),
        count: 1,
        images: refs,
        // "edit" is the reference-conditioned route -- the model is handed
        // the sheet and the panel description together. "latent" would
        // instead redraw the sheet itself, which is not what a panel is.
        image_strength: null, i2i_mode: refs.length ? "edit" : null,
        low_ram: true, preview: false, cache_limit_gb: null,
        allow_over_budget: false, loras: [], mask: null,
        outpaint_padding: null, outpaint_fill: null,
        // Every panel is stamped with the board it belongs to, so the library
        // shows one comic instead of a pile of pictures that happen to share
        // a timestamp. The index keeps them in reading order.
        project: projectId, project_name: projectName, project_index: i,
      };
      // A panel with no reference is a plain drawing. Sending it down the edit
      // route fails outright -- an edit needs an input image -- which is what
      // would happen to every panel in a board whose character is not in them
      // and whose rooms did not draw.
      const res = refs.length
        ? await api.editImage(params)
        : await api.generate(params);
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
  const edit = (i: number, patch: Partial<Panel>) => {
    setPanels((ps) => {
      if (!ps) return ps;
      const n = [...ps];
      n[i] = { ...n[i], ...patch };
      return n;
    });
    // The existing picture represents the old brief. Keep it in the Vault,
    // but mark this slot for drawing again instead of presenting stale art as
    // though it matched the edited panel.
    setDrawn((items) => {
      if (!items[i]) return items;
      const next = [...items]; next[i] = null; return next;
    });
    setPages([]);
  };

  /** Redraw one panel, keeping the rest of the board. */
  const redraw = async (i: number) => {
    if (!drawn[i]) { notify("Draw this panel first.", true); return; }
    stop.current = false;
    const nextRedraw = (redraws[i] ?? 0) + 1;
    setRedraws((r) => { const n = [...r]; n[i] = nextRedraw; return n; });
    setStage("drawing");
    await drawOne(i, sheet, placeSheets, nextRedraw);
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
        layout, projectId, projectName);
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

  // Clamp rather than trust: any path that shortens the board would otherwise
  // leave the inspector pointing at a panel that no longer exists.
  const selIndex = panels ? Math.min(selected, panels.length - 1) : 0;
  const sel = panels?.[selIndex] ?? null;

  if (!draftLoaded) {
    return <div className="empty-state">Opening your encrypted Board draft…</div>;
  }

  return (
    <div style={{ padding: 14 }}>
      <div className="board-flow" aria-label="Board workflow">
        {["read", "plan", "prepare", "draw", "compose"].map((name) => (
          <span key={name} className={name === workflow ? "current" : ""}>
            {name === "read" ? "Read story" : name === "plan" ? "Plan panels"
              : name === "prepare" ? "Prepare briefs" : name === "draw" ? "Draw"
                : "Compose"}
          </span>
        ))}
        <span className="board-save">Encrypted draft saved automatically</span>
      </div>
      <div className="board-work">

        {/* ---- the manuscript: your prose, never rewritten ---- */}
        <div className="board-pane">
          <div className="board-pane-head">
            Manuscript
            <span className="n">
              {story.trim() ? `${story.trim().split(/\s+/).length} words` : "empty"}
            </span>
          </div>
          <div className="inner">
            {!panels ? (
              <>
                <textarea
                  value={story}
                  rows={14}
                  disabled={busy}
                  onChange={(e) => changeStory(e.target.value)}
                  placeholder={"Mira comes home and finds the door already open. "
                    + "Nothing is taken, but every photograph has been turned to "
                    + "face the wall."}
                />
                <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 6,
                              lineHeight: 1.55 }}>
                  Prose, not prompts. It is divided into panels for you, and
                  nothing is invented that the story does not contain.
                </div>
              </>
            ) : (
              <div className="ms-prose">
                {/* Divided: the prose stays on screen, and the sentences behind
                    the selected panel are lit. Clicking one selects its panel,
                    so the story and the board point at each other. */}
                {panels.map((p, i) => (
                  <span
                    key={i}
                    className={"ms-beat" + (i === selected ? " lit" : "")}
                    onClick={() => setSelected(i)}
                  >
                    {p.source || [p.subject, p.action].filter(Boolean).join(", ")
                      || p.caption || `Panel ${i + 1}`}
                  </span>
                ))}
                {coverage && coverage.total > 0 && (
                  <div className={"notice" + (coverage.missing.length ? " warn" : " good")}
                       style={{ marginTop: 10 }}>
                    <strong>{coverage.percent}% of the prose is anchored</strong>
                    {coverage.missing.length > 0 ? (
                      <details>
                        <summary>{coverage.missing.length} passage{coverage.missing.length === 1 ? "" : "s"} need review</summary>
                        {coverage.missing.map((text, i) => <p key={i}>{text}</p>)}
                      </details>
                    ) : "Every passage has a panel source quote."}
                  </div>
                )}
                <button className="btn small" style={{ marginTop: 8 }} disabled={busy}
                        onClick={() => {
                          setPanels(null); setProjectId(null); setSheet(null);
                          setDrawn([]); setRedraws([]); setNotes({}); setPages([]);
                          setPlaceSheets({}); setCoverage(null); setSelected(0);
                        }}>
                  Back to the prose
                </button>
              </div>
            )}
          </div>
        </div>

        {/* ---- the board: the only place panels exist ---- */}
        <div className="board-pane">
          <div className="board-pane-head">
            Board
            <span className="n">
              {panels ? `${drawnCount}/${panels.length} drawn · ${scenes.length} scenes`
                      : "nothing yet"}
            </span>
          </div>
          <div className="inner">
            {!panels ? (
              <div style={{ fontSize: 12, color: "var(--text-faint)", lineHeight: 1.7 }}>
                Paste a story on the left, then divide it. The number of panels
                is worked out from the story's own beats — there is no number to
                choose, because a number chosen before the story is read is a
                guess.
              </div>
            ) : (
              <div className="board-cards">
                {panels.map((p, i) => (
                  <Fragment key={i}>
                    {(i === 0 || panels[i - 1].scene !== p.scene) && (
                      <div className="board-scene">
                        {p.scene_title || `Scene ${p.scene ?? 1}`}
                      </div>
                    )}
                    <button
                      className="board-card"
                      aria-current={i === selected ? "true" : "false"}
                      onClick={() => setSelected(i)}
                    >
                      <span className="shot">
                        {drawn[i]
                          ? <img src={vaultUrl(drawn[i]!)} alt={`Panel ${i + 1}`} />
                          : stage === "drawing" && done === i ? "drawing…" : "—"}
                        <span className="n">{String(i + 1).padStart(2, "0")}</span>
                      </span>
                      <span className="cap">
                        <b>{p.shot}</b>
                        {(p.caption || p.action || p.subject || "").slice(0, 54)}
                      </span>
                    </button>
                  </Fragment>
                ))}
              </div>
            )}

            {sheet && (
              <div style={{ marginTop: 18 }}>
                <div className="sub-head">Character sheet</div>
                <img src={vaultUrl(sheet)} alt="Character sheet"
                     style={{ width: "100%", maxWidth: 260, borderRadius: 6 }} />
                <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5,
                              lineHeight: 1.5 }}>
                  Panels share this cast reference, so recurring people keep the
                  same appearance from one shot to the next.
                </div>
              </div>
            )}

            {pages.length > 0 && (
              <div style={{ marginTop: 18 }}>
                <div className="sub-head">
                  {pages.length === 1 ? "The page" : `${pages.length} pages`}
                </div>
                {pages.map((id, i) => (
                  <img key={id} src={vaultUrl(id)} alt={`Page ${i + 1}`}
                       style={{ width: "100%", borderRadius: 6, marginBottom: 10 }} />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ---- notes: what produced the selected panel, and how to correct it ---- */}
        <div className="board-pane">
          <div className="board-pane-head">
            Notes
            <span className="n">{panels ? `Panel ${selIndex + 1}` : "settings"}</span>
          </div>
          <div className="inner">
            {!panels || !sel ? (
              <>
                <div className="field">
                  <label>Lead appearance override <em>optional</em></label>
                  <textarea
                    value={character}
                    rows={3}
                    disabled={busy}
                    onChange={(e) => setCharacter(e.target.value)}
                    placeholder="short black hair, red scarf, narrow face"
                  />
                </div>
                <div className="field">
                  <label>Or bring a picture of them</label>
                  <ImageDrop images={ownSheet} onChange={setOwnSheet} max={1}
                             onError={(m) => notify(m, true)} />
                  <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5,
                                lineHeight: 1.5 }}>
                    {ownSheet.length > 0
                      ? "Every panel is drawn against this picture, and no character "
                        + "sheet is made \u2014 it anchors them better than a description can."
                      : "A photo, a drawing, or a sheet from an earlier board. Given one, "
                        + "the board skips casting and draws against it directly."}
                  </div>
                </div>
                <div className="field">
                  <label>Style</label>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                    {STYLES.map(({ id, label }) => (
                      <button key={id} className={"pill" + (style === id ? " installed" : "")}
                              style={{ cursor: "pointer" }}
                              onClick={() => setStyle(id)}>
                        {label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="field">
                  <label>Model</label>
                  <select value={model?.id ?? ""} disabled={busy}
                          onChange={(e) => setModelId(e.target.value)}>
                    {usable.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
                  </select>
                </div>
              </>
            ) : (
              <>
                <div className="field">
                  <label>Shot</label>
                  <select value={sel.shot} disabled={busy}
                          onChange={(e) => edit(selIndex, { shot: e.target.value as Panel["shot"] })}>
                    <option value="wide">wide</option>
                    <option value="medium">medium</option>
                    <option value="close-up">close-up</option>
                  </select>
                </div>
                <div className="field">
                  <label>Story source <em>coverage anchor</em></label>
                  <div style={{ fontSize: 11, lineHeight: 1.55, color: "var(--text-dim)",
                                background: "var(--bg-sunk)", borderRadius: 5,
                                padding: "6px 8px" }}>
                    {sel.source || "No exact source quote was returned. Review this panel against the manuscript."}
                  </div>
                </div>
                <div className="field">
                  <label style={{ display: "flex", alignItems: "center", gap: 6,
                                  cursor: "pointer" }}>
                    <input type="checkbox" checked={sel.character_in_frame} disabled={busy}
                           style={{ width: "auto", margin: 0 }}
                           onChange={(e) => edit(selIndex, { character_in_frame: e.target.checked })} />
                    the one we follow is in this frame
                  </label>
                </div>
                <div className="field">
                  <label>People in frame <em>names from the story</em></label>
                  <input type="text" value={(sel.characters ?? []).join(", ")} disabled={busy}
                         onChange={(e) => edit(selIndex, {
                           characters: e.target.value.split(",").map((v) => v.trim()).filter(Boolean),
                           character_in_frame: Boolean(e.target.value.trim()),
                         })} />
                </div>
                <div className="field">
                  <label>Subject</label>
                  <input type="text" value={sel.subject} disabled={busy}
                         onChange={(e) => edit(selIndex, { subject: e.target.value })} />
                </div>
                <div className="field">
                  <label>Action</label>
                  <input type="text" value={sel.action} disabled={busy}
                         onChange={(e) => edit(selIndex, { action: e.target.value })} />
                </div>
                <div className="field">
                  <label>Setting</label>
                  <input type="text" value={sel.setting} disabled={busy}
                         onChange={(e) => edit(selIndex, { setting: e.target.value })} />
                </div>
                <div className="field">
                  <label>Caption <em>what the picture cannot say</em></label>
                  <input type="text" value={sel.caption} disabled={busy}
                         onChange={(e) => edit(selIndex, { caption: e.target.value })} />
                </div>
                <div className="field">
                  <label>Dialogue <em>spoken aloud in this frame</em></label>
                  <input
                    type="text"
                    value={(sel.dialogue ?? []).map(
                      (d) => (d.speaker ? `${d.speaker}: ` : "") + d.text).join(" / ")}
                    disabled={busy}
                    placeholder="Name: what they say / Name: what they say"
                    onChange={(e) => edit(selIndex, {
                      // "Name: line / Name: line" is quicker to correct than a
                      // pair of fields per speaker, and matches how the writer
                      // returns it.
                      dialogue: e.target.value.split("/").map((raw) => {
                        const [a, ...rest] = raw.split(":");
                        return rest.length
                          ? { speaker: a.trim(), text: rest.join(":").trim() }
                          : { speaker: "", text: a.trim() };
                      }).filter((d) => d.text),
                    })}
                  />
                </div>
                {(sel.description ?? "").trim() && (
                  <div className="field">
                    <label>The worked-up scene <em>replaces the terse fields</em></label>
                    <textarea
                      value={sel.description} disabled={busy} rows={3}
                      style={{ fontSize: 11.5, color: "var(--text-dim)" }}
                      onChange={(e) => edit(selIndex, { description: e.target.value })}
                    />
                  </div>
                )}
                {(sel.place ?? "").trim() && (
                  <div className="field">
                    <label>This scene's place <em>every panel here shares it</em></label>
                    <div style={{ fontSize: 11, lineHeight: 1.55, color: "var(--text-dim)",
                                  background: "var(--bg-sunk)", borderRadius: 5,
                                  padding: "6px 8px" }}>
                      {sel.place}
                    </div>
                  </div>
                )}
                <div className="field">
                  <label>The brief <em>the exact words asked for</em></label>
                  <div style={{ fontSize: 11, lineHeight: 1.6, color: "var(--text-dim)",
                                background: "var(--bg-sunk)", borderRadius: 5,
                                padding: "7px 9px", userSelect: "text" }}>
                    {panelPrompt(sel, style, panelWho(sel), notes[selIndex])}
                  </div>
                </div>
                {drawn[selIndex] && (
                  <div className="field">
                    <label>Your note <em>redraws just this frame</em></label>
                    <input
                      type="text"
                      value={notes[selIndex] ?? ""}
                      disabled={busy}
                      placeholder="older, grey at the temples"
                      style={{ borderStyle: notes[selIndex]?.trim() ? "solid" : "dashed" }}
                      onChange={(e) =>
                        setNotes((n) => ({ ...n, [selIndex]: e.target.value }))}
                    />
                    <button className="btn small full" style={{ marginTop: 6 }}
                            disabled={busy} onClick={() => void redraw(selIndex)}>
                      {notes[selIndex]?.trim() ? "Redraw with note" : "Redraw this panel"}
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* ---- the cast, along the bottom: it belongs to the whole board ---- */}
      {cast && (cast.people.length > 0 || cast.places.length > 0) && (
        <div className="board-shelf">
          <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: ".13em",
                         textTransform: "uppercase", color: "var(--text-faint)" }}>
            Cast
          </span>
          {cast.people.map((p) => (
            <span key={p.name} className="pill" style={{
                     opacity: p.tier === 1 ? 1 : p.tier === 2 ? 0.8 : 0.62 }}
                    title={(p.description || "The story never says what they look like.")
                           + ` · ${p.mentions} mentions`}>
              {p.name}
            </span>
          ))}
          {cast.places.length > 0 && <span className="divider" />}
          {cast.places.map((pl) => (
            <span key={pl.name} className="pill" style={{ opacity: 0.62 }}
                  title={pl.description || "The story never describes it."}>
              {pl.name}
            </span>
          ))}
        </div>
      )}

      {/* ---- the run bar: what it costs, said before you commit ---- */}
      <div className="board-run">
        {!panels ? (
          <>
            {story.trim() && writerReady && (
              <button className={"btn small" + (!cast ? " primary" : "")}
                      disabled={busy} onClick={read}>
                {reading ? "Reading the story…" : cast ? "Read again" : "Read story and cast"}
              </button>
            )}
            {cast && (
              <button className="btn primary small"
                      disabled={busy || !story.trim() || !writerReady}
                      onClick={divide}>
                {stage === "dividing" ? "Planning panels…" : "Plan the panels"}
              </button>
            )}
            {!writerReady && (
              // Reading and dividing are both the writer's work. Saying so here
              // beats a failure after the press.
              <span style={{ fontSize: 10.5, color: "var(--warn)" }}>
                Needs the prompt writer — add it from the Models tab.
              </span>
            )}
          </>
        ) : (
          <>
            <span style={{ fontSize: 11.5, color: "var(--text-dim)",
                           fontVariantNumeric: "tabular-nums" }}>
              <b style={{ color: "var(--text)" }}>{drawnCount}</b> of {panels.length} drawn
            </span>
            {remaining > 0 && preparedCount > 0 && !busy && (
              <span className="stepper" title="Panels per press">
                <button type="button" onClick={() => setBatch((n) => Math.max(1, n - 1))}>
                  &minus;
                </button>
                <span>{Math.min(batch, remaining)} at a time</span>
                <button type="button" onClick={() => setBatch((n) => Math.min(24, n + 1))}>
                  +
                </button>
              </span>
            )}
            {preparedCount === 0 && (
              <button className="btn primary small" disabled={busy}
                      onClick={() => void enrich()}>
                {stage === "enriching" ? "Preparing briefs…" : "Prepare drawing briefs"}
              </button>
            )}
            {remaining > 0 && preparedCount > 0 && (
              <button className="btn primary small" disabled={busy} onClick={draw}>
                {stage === "drawing" ? `Drawing ${done}/${panels.length}…`
                  : stage === "building" ? "Building the rooms…"
                  : stage === "casting" ? "Casting…"
                  : drawnCount === 0
                    ? `Draw ${Math.min(batch, remaining)} of ${panels.length} · about ${Math.ceil((Math.min(batch, remaining) + (ownSheet.length ? 0 : 1)) * 0.6)}–${Math.ceil((Math.min(batch, remaining) + (ownSheet.length ? 0 : 1)) * 2)} min`
                    : `Draw next ${Math.min(batch, remaining)}`}
              </button>
            )}
            {remaining === 0 && !busy && (
              <span style={{ fontSize: 11, color: "var(--good)" }}>
                All {panels.length} drawn
              </span>
            )}
            {!busy && preparedCount > 0 && drawnCount === 0 && (
              <button className="btn small" disabled={busy} onClick={() => void enrich()}>
                Prepare again
              </button>
            )}
            {busy && <button className="btn small" onClick={cancel}>Cancel</button>}
            {!busy && remaining === 0 && drawn.some(Boolean) && (
              <>
                <select value={layout} style={{ width: "auto", fontSize: 11, padding: "2px 6px" }}
                        onChange={(e) => setLayout(e.target.value)}>
                  <option value="page">Page</option>
                  <option value="strip">Scrolling strip</option>
                </select>
                <button className="btn small" disabled={composing}
                        onClick={() => void compose()}>
                  {composing ? "Composing…" : "Make a page"}
                </button>
              </>
            )}
          </>
        )}
        <div style={{ flex: 1 }} />
        {!busy && (story || panels) && (
          <button className="btn small" onClick={() => {
            setStory(""); setCast(null); setPanels(null); setProjectId(null);
            setCharacter(""); setSheet(null); setDrawn([]); setRedraws([]);
            setNotes({}); setPages([]); setOwnSheet([]); setPlaceSheets({});
            setCoverage(null);
            setSelected(0);
          }}>
            New board
          </button>
        )}
        {elapsed > 0 && busy && (
          <span style={{ fontSize: 10.5, color: "var(--text-faint)",
                         fontVariantNumeric: "tabular-nums" }}>
            {Math.floor(elapsed / 60)}m {elapsed % 60}s
          </span>
        )}
      </div>

      {prog && (
        <div style={{ marginTop: 8 }}>
          <JobProgress p={prog} label={
            stage === "dividing" ? "Dividing the story"
              : stage === "casting" ? "Casting the character"
              : `Panel ${done + 1} of ${panels?.length ?? 0}`} />
        </div>
      )}
    </div>
  );
}
