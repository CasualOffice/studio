import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { api, errText, isCancelled, newJobId, onEngineProgress, vaultUrl } from "../lib/api";
import type { Cast, EngineProgress, ModelStatus, Panel } from "../lib/types";
import { ImageDrop, JobProgress } from "./shared";
import { loadPref, savePref } from "../lib/prefs";
import { loadBoardDraft, type BoardDraft, type Coverage } from "../lib/boardDraft";
import { STYLES, isKnownStyle, lockStyle, panelPrompt, panelReferences,
         panelSeed, placePrompt, reviveLock, sheetPrompt, styleHasMoved,
         styleLabel } from "../lib/board";
import type { StyleLock } from "../lib/board";

/**
 * A story, divided into panels, drawn as one consistent character.
 *
 * The pieces here are proven separately: the writer divides prose reliably,
 * and a character sheet fed back as a reference holds the character across
 * shots. This puts them in order and keeps the one thing that makes the
 * result read as a story rather than a set of unrelated pictures -- the same
 * sheet references every panel, and the same style words lead every prompt.
 */

/**
 * How large each panel is drawn.
 *
 * 512 was chosen because a board is read at a glance, and it was wrong.
 * Measured on FLUX.2 Klein 4-bit, the same prompt and seed down the same
 * reference-conditioned route:
 *
 *     512x512     36s    ~4 GiB    the face came back as a blank shadow
 *     768x768     77s    9.11 GiB  the face came back as a face
 *    1024x1024   164s   13.22 GiB  past the 12 GiB budget
 *
 * A wide shot of a person at 512 does not have the pixels to put a face in,
 * so the panel that establishes who we are following is the one that fails.
 * Twice the time for a panel you can use is not a cost, and 1024 is not
 * available on this machine at all.
 *
 * Memory does not move with the number of references: the same panel drawn
 * against both a character sheet and a room sheet also peaked at 9.11 GiB,
 * and took 130 seconds instead of 77. References cost time, not memory --
 * the same shape the video sizes turned out to have.
 *
 * 512 stays on offer because 9.11 GiB is most of what this Mac has: with a
 * browser open the run can be refused for want of memory, and a smaller panel
 * you can finish beats a larger one you cannot start.
 */
const PANEL_SIZES: [string, number][] = [
  ["Detailed (768px)", 768],
  ["Quick (512px)", 512],
];

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
  /** Panel size, remembered. See PANEL_SIZES for what the choice costs. */
  const [panelPx, setPanelPx] = useState(() => {
    const saved = loadPref("boardPanelPx", 768);
    return PANEL_SIZES.some(([, px]) => px === saved) ? saved : 768;
  });
  /**
   * The id every panel of this board is stamped with.
   *
   * It is what makes the board one entry in the library instead of a pile of
   * loose pictures sharing a timestamp. Kept across restarts with the rest of
   * the board, and replaced when a new story is divided.
   */
  const [projectId, setProjectId] = useState<string | null>(null);


  /**
   * The words this board is drawn from, not the name of a preset.
   *
   * The remembered id is only the starting point for a board that has not
   * pinned one yet. Once pinned it travels in the encrypted draft, so a board
   * half-drawn under one build is finished under the same words -- and chapter
   * eleven can be made to match chapter one however the presets move.
   */
  const [style, setStyle] = useState<StyleLock>(
    () => lockStyle(loadPref("boardStyle", STYLES[0].id)));
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
  /**
   * Which half of the rail is showing: the selected panel, or the board.
   *
   * The rail used to swap to per-panel controls the instant a division
   * existed, which put the three board-wide decisions -- the look, the cast
   * sheet, the rooms -- out of reach for the whole rest of the job. The only
   * way back to them was `Back to the prose`, which throws away the division,
   * the briefs and every note. So the two failures this tool calls fatal were
   * the two a person could not fix without losing everything else.
   */
  const [railTab, setRailTab] = useState<"panel" | "board">("panel");
  /** A style change that would discard drawn work, waiting on a second press. */
  const [confirmStyle, setConfirmStyle] = useState<string | null>(null);

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
  /**
   * Which reset is armed, if any.
   *
   * Both resets discard the board, and neither used to ask. "Back to the
   * prose" is a small button directly under the coverage panel -- exactly
   * where someone goes after being told a passage needs review -- and it threw
   * away every drawn panel, every note, the composed pages and the project
   * identity in one click. The pictures survive in the Vault, but nothing
   * loads a Vault project back into a board, and the notes exist nowhere else.
   */
  const [confirmReset, setConfirmReset] = useState<"prose" | "new" | null>(null);


  // A Board contains the manuscript and creative history. Load it from the
  // encrypted vault, migrating old plaintext preferences exactly once.
  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const draft: Partial<BoardDraft> | null = await loadBoardDraft();
        if (!live) return;
        if (draft) {
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
          // The style the board was drawn under wins over the remembered
          // preset. A draft written before the lock existed has none, so it
          // takes the preset it named -- which is the last moment that
          // substitution is invisible, and the notice below is why it is not.
          const locked = reviveLock(draft.style);
          if (locked) {
            setStyle(locked);
            if (styleHasMoved(locked) && (draft.drawn ?? []).some(Boolean)) {
              notify(
                `This board is drawn in ${styleLabel(locked.id)} as it was when `
                + "it was started. The preset has changed since, so a new board "
                + "will not match it.",
              );
            }
          } else {
            // The raw remembered id, not `style.id` -- `lockStyle` has already
            // substituted the first preset by the time it is state, so asking
            // the state whether the id was known can only ever answer yes.
            const remembered = loadPref("boardStyle", STYLES[0].id);
            if (!isKnownStyle(remembered)) {
              notify(
                "This board was made in a style this build no longer has "
                + `("${remembered}"), so it is now set to ${STYLES[0].label}. `
                + "Anything already drawn was drawn in the old one.",
                true,
              );
            }
          }
        }
        // Arm the autosave only once the restore has actually succeeded. A
        // draft of `null` counts: that is a good read of a board nobody has
        // started yet, and a first-time user still needs saving to work.
        setDraftLoaded(true);
      } catch (e) {
        // Deliberately leaves the autosave disarmed, and says so.
        //
        // This used to be a `finally`, so a failed restore armed the save
        // effect below with every piece of state at its default -- and 350 ms
        // later wrote `{story: "", panels: null, notes: {}}` over the stored
        // draft. A transient read error was enough to destroy the manuscript
        // and every hand-typed note, and the legacy migration then cleared the
        // plaintext copies too, so there was nowhere left to recover from.
        // Losing the session is recoverable; overwriting the only copy is not.
        if (live) {
          notify(
            `Could not restore the encrypted Board draft: ${errText(e)}. `
            + "Nothing will be saved over it -- reopen the app to try again.",
            true,
          );
        }
      }
    })();
    return () => { live = false; };
  }, [notify]);

  useEffect(() => {
    if (!draftLoaded) return;
    const draft: BoardDraft = {
      version: 1, story, cast, projectId, style, character, panels, sheet, drawn,
      redraws, notes, pages, ownSheet, placeSheets,
      coverage,
    };
    const timeout = window.setTimeout(() => {
      void api.boardStateSet(JSON.stringify(draft)).catch((e) =>
        notify(`Could not save the encrypted Board draft: ${errText(e)}`, true));
    }, 350);
    return () => window.clearTimeout(timeout);
  }, [draftLoaded, story, cast, projectId, style, character, panels, sheet, drawn,
      redraws, notes, pages, ownSheet, placeSheets, coverage, notify]);

  // An armed reset is disarmed by anything else happening: pressing Escape,
  // or the board changing under it. A destructive button left loaded is how a
  // later, unrelated click deletes something.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setConfirmReset(null); setConfirmStyle(null); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => {
    setConfirmReset(null); setConfirmStyle(null);
  }, [stage, drawn, panels]);

  // These are interface choices, not project content, so they remain prefs.
  useEffect(() => { savePref("boardBatch", batch); }, [batch]);
  useEffect(() => { savePref("boardPanelPx", panelPx); }, [panelPx]);
  useEffect(() => { savePref("boardStyle", style.id); }, [style.id]);
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

  /** Whether anything has been drawn in the current style. */
  const drawnInStyle = drawn.some(Boolean) || !!sheet
    || Object.keys(placeSheets).length > 0;

  /**
   * Change the look of a board that is already under way.
   *
   * The style leads every prompt, so everything already drawn was drawn in the
   * old one and a half-restyled board is not a board -- the pictures go, and
   * they are still in the Vault if they were wanted. What stays is everything
   * that is a reading of the manuscript rather than of the style: the
   * division, the briefs, the notes and the coverage report. Having to retype
   * a note to change a look is what made the look uncorrectable in practice.
   */
  const changeStyle = (id: string) => {
    if (id === style.id) return;
    if (drawnInStyle && confirmStyle !== id) { setConfirmStyle(id); return; }
    setStyle(lockStyle(id));
    setSheet(null); setDrawn([]); setRedraws([]);
    setPlaceSheets({}); setPages([]);
    setConfirmStyle(null);
  };

  /**
   * Throw away the cast sheet so the next draw builds a new one.
   *
   * Panels already drawn were drawn against the old sheet and are left alone:
   * which of them to redraw is a judgement about the pictures, and the board
   * already has a per-panel redraw for making it. The composed pages go,
   * because a page is only as current as the panels in it.
   */
  const recastSheet = () => {
    setSheet(null); setPages([]);
    notify("The cast sheet will be drawn again on the next run. Panels already "
      + "drawn were drawn against the old one -- redraw the ones that no "
      + "longer match.");
  };

  /** The same, for one scene's room. */
  const rebuildPlace = (scene: number) => {
    setPlaceSheets((prev) => {
      const next = { ...prev };
      delete next[scene];
      return next;
    });
    setPages([]);
    notify(`Scene ${scene}'s room will be drawn again on the next run.`);
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
  /**
   * How long the next press will take.
   *
   * Measured warm on this machine: 36 seconds a panel at 512, 77 at 768. The
   * first panel of a run also loads the model, which is most of the spread.
   * The old figure was a flat 0.6-2 minutes regardless of size, written when
   * every panel was 512.
   */
  const estimate = useMemo(() => {
    // Every image the run will draw, not just the panels in the batch.
    // `draw()` calls buildPlaces() first, which draws a room sheet for each
    // scene that has a place and has not got one yet, and the cast sheet if it
    // is missing. Counting the panels alone understated a first run by a sheet
    // per scene -- so the one figure quoted before the expensive part was the
    // one number the user could not rely on.
    const rooms = new Set(
      (panels ?? [])
        .filter((p) => (p.place ?? "").trim() && !placeSheets[p.scene ?? 1])
        .map((p) => p.scene ?? 1)
    ).size;
    const pieces = Math.min(batch, Math.max(1, remaining))
      + (ownSheet.length ? 0 : 1) + rooms;
    const per = panelPx >= 768 ? 77 : 36;
    return { low: Math.ceil(pieces * per / 60),
             high: Math.ceil(pieces * per * 2 / 60) };
  }, [batch, remaining, ownSheet.length, panelPx, panels, placeSheets]);

  /** What a reset would throw away, named so the question can be answered. */
  const atRisk = useMemo(() => {
    const bits: string[] = [];
    const noted = Object.values(notes).filter((n) => n.trim()).length;
    const plural = (n: number, word: string) =>
      `${n} ${word}${n === 1 ? "" : "s"}`;
    if (drawnCount) bits.push(plural(drawnCount, "drawn panel"));
    if (noted) bits.push(plural(noted, "note"));
    if (pages.length) bits.push(plural(pages.length, "composed page"));
    return bits;
  }, [drawnCount, notes, pages]);
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
        width: panelPx, height: panelPx,
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
          width: panelPx, height: panelPx,
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
        width: panelPx, height: panelPx,
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
      if (isCancelled(e)) stop.current = true;
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
      const r = await api.enrichPanels(id, panels, style.words);
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
                {coverage && coverage.total > 0 && (() => {
                  // The engine caps the listed passages at twenty. Counting
                  // the list meant the worst-covered boards reported the
                  // smallest shortfall, so count what it actually found and
                  // say when the list below it is only part of that.
                  const lost = coverage.missing_total ?? coverage.missing.length;
                  const shown = coverage.missing.length;
                  return (
                    <div className={"notice" + (lost ? " warn" : " good")}
                         style={{ marginTop: 10 }}>
                      <strong>{coverage.percent}% of the prose is anchored</strong>
                      {lost > 0 ? (
                        <details>
                          <summary>
                            {lost} passage{lost === 1 ? "" : "s"} need review
                            {shown < lost ? ` (first ${shown} shown)` : ""}
                          </summary>
                          {coverage.missing.map((text, i) => <p key={i}>{text}</p>)}
                        </details>
                      ) : "Every passage has a panel source quote."}
                    </div>
                  );
                })()}
                <button
                  className={"btn small" + (confirmReset === "prose" ? " danger" : "")}
                  style={{ marginTop: 8 }} disabled={busy}
                  onClick={() => {
                    if (atRisk.length && confirmReset !== "prose") {
                      setConfirmReset("prose"); return;
                    }
                    setConfirmReset(null);
                    setPanels(null); setProjectId(null); setSheet(null);
                    setDrawn([]); setRedraws([]); setNotes({}); setPages([]);
                    setPlaceSheets({}); setCoverage(null); setSelected(0);
                  }}>
                  {confirmReset === "prose"
                    ? `Discard ${atRisk.join(", ")}? The pictures stay in the Vault`
                    : "Back to the prose"}
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
            {panels ? (
              <span className="n" style={{ display: "flex", gap: 4 }}>
                <button className={"pill" + (railTab === "panel" ? " installed" : "")}
                        style={{ cursor: "pointer" }}
                        onClick={() => setRailTab("panel")}>
                  Panel {selIndex + 1}
                </button>
                <button className={"pill" + (railTab === "board" ? " installed" : "")}
                        style={{ cursor: "pointer" }}
                        onClick={() => setRailTab("board")}>
                  Board
                </button>
              </span>
            ) : <span className="n">settings</span>}
          </div>
          <div className="inner">
            {!panels || !sel || railTab === "board" ? (
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
                      <button key={id}
                              className={"pill" + (style.id === id ? " installed" : "")
                                + (confirmStyle === id ? " danger" : "")}
                              style={{ cursor: "pointer" }} disabled={busy}
                              onClick={() => changeStyle(id)}>
                        {confirmStyle === id ? `Restyle? ${label}` : label}
                      </button>
                    ))}
                  </div>
                  {confirmStyle && (
                    <div style={{ fontSize: 10.5, color: "var(--bad)", marginTop: 5,
                                  lineHeight: 1.5 }}>
                      Press again to restyle. The {drawnCount} drawn panel
                      {drawnCount === 1 ? "" : "s"}, the cast sheet and the rooms
                      go, because they were drawn in {styleLabel(style.id)}. The
                      division, the briefs and your notes stay. The old pictures
                      remain in the Vault.
                    </div>
                  )}
                  {styleHasMoved(style) && (
                    <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5,
                                  lineHeight: 1.5 }}>
                      This board keeps the {styleLabel(style.id)} it was started in.
                      The preset has changed since, so a new board will look
                      different -- picking {styleLabel(style.id)} again here takes
                      the current version.
                    </div>
                  )}
                </div>
                {panels && (
                  <div className="field">
                    <label>References <em>drawn once, reused by every panel</em></label>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 5,
                                  alignItems: "center" }}>
                      <button className="btn small" disabled={busy || !sheet}
                              onClick={recastSheet}>
                        {sheet ? "Draw the cast sheet again" : "No cast sheet yet"}
                      </button>
                      {scenes.filter((s) => placeSheets[s]).map((s) => (
                        <button key={s} className="btn small" disabled={busy}
                                onClick={() => rebuildPlace(s)}>
                          Redraw scene {s}&rsquo;s room
                        </button>
                      ))}
                    </div>
                    <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5,
                                  lineHeight: 1.5 }}>
                      A reference is drawn once and every panel that needs it is
                      drawn against the same one -- which is what keeps a face and
                      a room the same from panel to panel. Throwing one away here
                      rebuilds it on the next run and leaves the panels alone, so
                      you choose which of them no longer match.
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
            {remaining > 0 && !busy && (
              <select
                value={panelPx}
                onChange={(e) => setPanelPx(Number(e.target.value))}
                title="How large each panel is drawn"
                style={{ width: "auto", fontSize: 11, padding: "3px 6px" }}
              >
                {PANEL_SIZES.map(([label, px]) => (
                  <option key={px} value={px}>{label}</option>
                ))}
              </select>
            )}
            {/* Gated on the panels still waiting for a picture, which is what
                `unpreparedRemaining` was written for. `preparedCount > 0` let
                one prepared panel unlock drawing all of them, and the rest went
                down the terse-field path -- a minute of drawing each to find
                out. Preparing comes first, as the stage order promises. */}
            {unpreparedRemaining > 0 && (
              <button className="btn primary small" disabled={busy}
                      onClick={() => void enrich()}>
                {stage === "enriching" ? "Preparing briefs…"
                  : preparedCount === 0 ? "Prepare drawing briefs"
                  : `Prepare ${unpreparedRemaining} remaining brief${unpreparedRemaining === 1 ? "" : "s"}`}
              </button>
            )}
            {remaining > 0 && unpreparedRemaining === 0 && (
              <button className="btn primary small" disabled={busy} onClick={draw}>
                {stage === "drawing" ? `Drawing ${done}/${panels.length}…`
                  : stage === "building" ? "Building the rooms…"
                  : stage === "casting" ? "Casting…"
                  : drawnCount === 0
                    ? `Draw ${Math.min(batch, remaining)} of ${panels.length} · about ${estimate.low}–${estimate.high} min`
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
          <button
            className={"btn small" + (confirmReset === "new" ? " danger" : "")}
            onClick={() => {
              if (atRisk.length && confirmReset !== "new") {
                setConfirmReset("new"); return;
              }
              setConfirmReset(null);
              setStory(""); setCast(null); setPanels(null); setProjectId(null);
              setCharacter(""); setSheet(null); setDrawn([]); setRedraws([]);
              setNotes({}); setPages([]); setOwnSheet([]); setPlaceSheets({});
              setCoverage(null);
              setSelected(0);
            }}>
            {confirmReset === "new"
              ? `Start over, discarding ${atRisk.join(", ")}?`
              : "New board"}
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
