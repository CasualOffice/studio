import { ANALYST_IDS, castFromBook, unanalysedIn, type CharacterReference, type StoryBook,
         type StoryLibrary } from "../lib/story";
import { Fragment, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { api, errText, isCancelled, isEngineDeath, newJobId, onEngineProgress, vaultUrl } from "../lib/api";
import type { BookFormat, CastMember, ModelStatus, Panel } from "../lib/types";
import { JobProgress } from "./shared";
import Field from "./Field";
import { anchorOverflow, artOf, fromArrays, setArt, toArrays,
         edit as editPanel, insert as insertPanel, merge as mergePanelIn,
         move as movePanelIn, remove as removePanel, split as splitPanelIn,
         replan, type BoardDoc, type PanelArt } from "../lib/boardDoc";
import { loadBoardDraft, loadSavedBoard, type BoardDraft } from "../lib/boardDraft";
import { CUSTOM_STYLE, STYLES, castSeed, castSheetPrompt, drawBatch, isKnownStyle, lockCustomStyle,
         lockStyle, panelPrompt, panelReferences, panelSeed, reviveLock, SHEET_ASPECT, shotClass,
         styleHasMoved, styleLabel } from "../lib/board";
import { boardReducer, coverageOf, emptyBoard, estimateMinutes, finishedPages, idleRun, missingSheets,
         nextPage, placesOf, runReducer, sheetPeople, type BoardAction, type BoardState,
         type RunState } from "../lib/boardFlow";
import { DEFAULT_PAGES, panelSize, speakerSide } from "../lib/layout";
import { loadPref, savePref } from "../lib/prefs";
import CastScreen from "./board/CastScreen";
import PanelInspector from "./board/PanelInspector";
import PagesPane from "./board/PagesPane";
import StylePicker from "./board/StylePicker";

/**
 * A story, made into a comic.
 *
 * Paste the story, choose a look and a number of pages, and press Make
 * comic: the story is analysed and planned in one pass, with a visual brief
 * for every panel. The cast screen is the one gate -- a full-body sheet per
 * recurring person, in the outfit they wear in this chapter -- and once it is
 * approved the comic is drawn a page at a time, page one first, each page
 * composed as soon as it is finished. A panel that is wrong is fixed where it
 * is: a few alternatives to pick from, or a painted repair.
 *
 * The workflow's rules live in boardFlow's reducer; this component runs the
 * jobs and lays out the screen.
 */

/**
 * How large each panel is drawn, as the side of a square of the same area.
 *
 * Measured on FLUX.2 Klein 4-bit down the reference-conditioned route:
 * 512x512 in 36s at ~4 GiB came back with a face as a blank shadow; 768x768
 * in 77s at 9.11 GiB came back with a face; 1024x1024 is past the 12 GiB
 * budget. A panel is drawn at its cell's shape with this many pixels.
 */
const PANEL_SIZES: [string, number][] = [
  ["Detailed (768px)", 768],
  ["Quick (512px)", 512],
];

const STEPS = [["story", "Story"], ["cast", "Cast"], ["draw", "Draw"], ["pages", "Pages"]] as const;
type StepName = typeof STEPS[number][0];

/** Where a person stands, left to right, so the reference lineup matches the panel. */
const SLOT_ORDER = ["far left", "left third", "centre", "right third", "far right"];

export default function Storyboard({
  models, notify, onProduced, openProject, onOpened, storyLibrary, onStoryAnalyzed, onCharacterReference,
}: {
  models: ModelStatus[];
  storyLibrary: StoryLibrary | null;
  onStoryAnalyzed: (book: StoryBook) => Promise<void>;
  onCharacterReference: (bookId: string, characterId: string, reference: CharacterReference) => Promise<void>;
  notify: (m: string, bad?: boolean) => void;
  onProduced: () => void;
  /** A saved board the Vault asked to reopen, by project id. */
  openProject?: string | null;
  onOpened?: () => void;
}) {
  const usable = useMemo(
    () => models.filter(
      (m) => m.installed && !m.hopeless && m.fit !== "broken"
        && m.tasks.includes("edit" as never) && m.tasks.includes("text_to_image" as never)),
    [models]);
  const analysts = useMemo(
    () => models.filter((m) => ANALYST_IDS.includes(m.id) && m.installed && !m.hopeless), [models]);
  const [modelId, setModelId] = useState("");
  const model = usable.find((m) => m.id === modelId) ?? usable[0];
  const [analystId, setAnalystId] = useState("");
  const [panelPx, setPanelPx] = useState(() => {
    const saved = loadPref("boardPanelPx", 768);
    return PANEL_SIZES.some(([, px]) => px === saved) ? saved : 768;
  });

  // ---- the board, through one reducer -------------------------------------
  const [board, dispatch] = useReducer(boardReducer, undefined, () => emptyBoard(
    lockStyle(loadPref("boardStyle", STYLES[0].id)), loadPref<BookFormat>("boardFormat", "manga"),
    loadPref("boardLayout", "page")));
  /**
   * The board as it is now, for a job that started minutes ago. Every change
   * goes through `act`, which applies it here at once as well as to React's
   * state, so a panel drawn late in a run reads the board as it is then --
   * a brief corrected while it waited is the brief that is drawn.
   */
  const boardRef = useRef<BoardState>(board);
  const act = (action: BoardAction) => {
    boardRef.current = boardReducer(boardRef.current, action);
    dispatch(action);
  };
  const [run, runDispatch] = useReducer(runReducer, idleRun);
  const runRef = useRef<RunState>(run);
  useEffect(() => { runRef.current = run; }, [run]);

  const [draftLoaded, setDraftLoaded] = useState(false);
  const [unsaved, setUnsaved] = useState<string | null>(null);
  const [previous, setPrevious] = useState<Partial<BoardDraft> | null>(null);
  const [selected, setSelected] = useState(0);
  const [choices, setChoices] = useState<{ panelId: string; ids: string[] } | null>(null);
  const [maskBytes, setMaskBytes] = useState<Uint8Array | null>(null);
  const [pagesStale, setPagesStale] = useState(false);
  const [viewStep, setViewStep] = useState<StepName | null>(null);
  const [confirmReset, setConfirmReset] = useState(false);
  const [editingStory, setEditingStory] = useState(false);
  const [appendChapter, setAppendChapter] = useState(false);
  const [continuationBookId, setContinuationBookId] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const stop = useRef(false);
  const lastFailure = useRef<unknown>(null);
  const timer = useRef<number | null>(null);
  const runButton = useRef<HTMLButtonElement>(null);
  const pagesHead = useRef<HTMLDivElement>(null);
  /** Composite references already made, by the sheets they join. */
  const referenceCache = useRef<Record<string, string>>({});
  /** Panel times measured on this machine, by panel size. */
  const panelTimes = useRef<Record<number, number[]>>({});
  /** Pages this session composed and has since composed again, to discard. */
  const autoPages = useRef<string[]>([]);

  const { doc, cast, castApproved } = board;
  const panels = board.divided ? doc.panels : null;
  const { drawn, notes } = useMemo(() => toArrays(doc), [doc]);
  const places = useMemo(() => placesOf(doc.panels, board.format, board.layout),
    [doc.panels, board.format, board.layout]);
  const coverage = useMemo(() => coverageOf(doc.panels, board.coverage?.sentences) ?? board.coverage,
    [doc.panels, board.coverage]);
  const legacy = board.divided && doc.panels.some((p) => p.cast === undefined);
  const casting = sheetPeople(board);
  const minor = (cast?.people ?? []).filter((p) => p.id && !casting.includes(p));
  const sheetsLeft = missingSheets(board).length;
  const drawnCount = drawn.filter(Boolean).length;
  const remaining = Math.max(0, (panels?.length ?? 0) - drawnCount);
  const busy = run.stage !== "idle";
  const unanalysed = unanalysedIn(board.storyBook, board.chapterId);
  const finished = useMemo(() => finishedPages(doc, places), [doc, places]);
  const measured = panelTimes.current[panelPx] ?? [];
  const estimate = estimateMinutes(castApproved ? 0 : sheetsLeft, remaining, panelPx,
    measured.length ? measured.reduce((a, b) => a + b, 0) / measured.length : undefined);
  const projectName = useMemo(() => {
    const first = board.story.trim().split(/\s+/).slice(0, 7).join(" ");
    return first ? (board.story.trim().length > first.length ? first + "…" : first) : "Picture board";
  }, [board.story]);

  // ---- the encrypted draft --------------------------------------------------
  /** A stored board, as this screen's state. Old drafts open read-only. */
  const applyDraft = (draft: Partial<BoardDraft>) => {
    const style = reviveLock(draft.style) ?? lockStyle(loadPref("boardStyle", STYLES[0].id));
    if (!reviveLock(draft.style) && !isKnownStyle(loadPref("boardStyle", STYLES[0].id))) {
      notify(`This board was made in a style this build no longer has, so it is now set to ${STYLES[0].label}.`, true);
    } else if (reviveLock(draft.style) && styleHasMoved(style) && (draft.drawn ?? []).some(Boolean)) {
      notify(`This board is drawn in ${styleLabel(style.id)} as it was when it was started. The preset has `
        + "changed since, so a new board will not match it.");
    }
    const panelsIn = Array.isArray(draft.panels) ? draft.panels : null;
    const docIn = fromArrays(panelsIn, draft.drawn ?? [], draft.redraws ?? [], draft.notes ?? {}, draft.variants ?? {});
    // A cast saved before cast sheets has no idea who recurs: read it again
    // from the story, which knows.
    const castIn = draft.storyBook && !draft.cast?.people.some((p) => p.recurring !== undefined)
      ? castFromBook(draft.storyBook, draft.chapterId) : draft.cast ?? null;
    const state: BoardState = {
      ...emptyBoard(style, (draft.format as BookFormat) ?? board.format, board.layout),
      story: typeof draft.story === "string" ? draft.story : "",
      chapterTitle: draft.chapterTitle ?? "Chapter 1",
      storyBook: draft.storyBook ?? null, chapterId: draft.chapterId ?? null,
      cast: castIn, leadId: draft.leadId ?? null,
      sheets: draft.castApproved !== undefined ? draft.characterSheets ?? {} : {},
      sheetRegens: draft.sheetRegens ?? {}, recast: draft.recast ?? [],
      castApproved: false, doc: docIn, divided: Array.isArray(draft.panels),
      pages: Array.isArray(draft.pages) ? draft.pages : [], overflow: draft.overflow ?? [],
      coverage: draft.coverage ?? null, pageCount: draft.pageCount ?? DEFAULT_PAGES, projectId: draft.projectId ?? null,
    };
    // Approved stays approved only while every recurring person has a sheet;
    // a board from before cast sheets is cast again, keeping its pictures.
    state.castApproved = !!draft.castApproved && missingSheets(state).length === 0;
    boardRef.current = state;
    dispatch({ type: "load", state });
    referenceCache.current = {};
    setSelected(0); setChoices(null); setPagesStale(false);
  };

  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const draft = await loadBoardDraft();
        if (!live) return;
        if (draft) applyDraft(draft);
        // A board that restored empty while an earlier state of it exists is
        // the shape the destroyed-manuscript bug had: offered, never applied.
        if (!draft?.story?.trim() && !draft?.panels?.length) {
          try {
            const back = await api.boardStatePrevious();
            if (live && back) {
              const parsed = JSON.parse(back) as Partial<BoardDraft>;
              if (parsed.story?.trim() || parsed.panels?.length) setPrevious(parsed);
            }
          } catch { /* no step back is not an error */ }
        }
        // Armed only once the restore has succeeded: a failed read must never
        // be followed by a save of the empty defaults over the only copy.
        setDraftLoaded(true);
      } catch (e) {
        if (live) {
          notify(`Could not restore the encrypted Board draft: ${errText(e)}. `
            + "Nothing will be saved over it -- reopen the app to try again.", true);
        }
      }
    })();
    return () => { live = false; };
  }, [notify]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!draftLoaded) return;
    const arrays = toArrays(board.doc);
    const draft: BoardDraft = {
      version: 1, story: board.story, cast: board.cast, projectId: board.projectId, style: board.style,
      character: "", panels: board.divided ? board.doc.panels : null, sheet: null,
      drawn: arrays.drawn, redraws: arrays.redraws, notes: arrays.notes, variants: arrays.variants,
      pages: board.pages, ownSheet: [], placeSheets: {}, coverage: board.coverage, overflow: board.overflow,
      storyBook: board.storyBook, chapterId: board.chapterId, characterSheets: board.sheets,
      sheetRegens: board.sheetRegens, castApproved: board.castApproved, recast: board.recast,
      leadId: board.leadId, pageCount: board.pageCount, format: board.format, chapterTitle: board.chapterTitle,
    };
    const timeout = window.setTimeout(() => {
      const body = JSON.stringify(draft);
      void api.boardStateSet(body).then(() => setUnsaved(null), (e) => {
        const why = errText(e);
        setUnsaved(why);
        notify(`Could not save the encrypted Board draft: ${why}`, true);
      });
      // A copy under this board's own id, so the comic survives the next story.
      if (board.projectId) {
        void api.boardStateSet(body, `board-${board.projectId}`).catch(() => { /* the live draft saved */ });
      }
    }, 350);
    return () => window.clearTimeout(timeout);
  }, [draftLoaded, board, notify]);

  // A comic the Vault asked for, once the launch restore is done.
  useEffect(() => {
    if (!openProject || !draftLoaded) return;
    if (openProject === board.projectId) { onOpened?.(); return; }
    let live = true;
    void (async () => {
      try {
        const saved = await loadSavedBoard(openProject);
        if (!live) return;
        if (!saved) {
          notify("That comic was made before boards were saved individually, so its panels are in the "
            + "Vault but its story and notes are not.", true);
        } else { applyDraft(saved); notify("Opened in the Board."); }
      } catch (e) {
        if (live) notify(`Could not open that board: ${errText(e)}`, true);
      } finally {
        if (live) onOpened?.();
      }
    })();
    return () => { live = false; };
  }, [openProject, draftLoaded]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { savePref("boardPanelPx", panelPx); }, [panelPx]);
  useEffect(() => { savePref("boardStyle", board.style.id); }, [board.style.id]);
  useEffect(() => { savePref("boardLayout", board.layout); }, [board.layout]);
  useEffect(() => { savePref("boardFormat", board.format); }, [board.format]);
  // A painted mask belongs to one picture; a different panel closes it.
  useEffect(() => { setMaskBytes(null); }, [selected]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setConfirmReset(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  // Starting a job disables most of the screen; focus goes to the run button,
  // which is Stop while the job runs (WCAG 2.4.3).
  useEffect(() => {
    if (!busy) return;
    const at = document.activeElement as HTMLButtonElement | null;
    if (!at || at === document.body || at.disabled) runButton.current?.focus();
  }, [busy]);

  const workflowStep: StepName = !board.divided ? "story" : !castApproved ? "cast"
    : remaining > 0 ? "draw" : "pages";
  useEffect(() => { setViewStep(null); }, [workflowStep]);

  // ---- jobs ---------------------------------------------------------------
  const startTimer = () => {
    setElapsed(0);
    const started = Date.now();
    if (timer.current) window.clearInterval(timer.current);
    timer.current = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
  };
  const stopTimer = () => { if (timer.current) { window.clearInterval(timer.current); timer.current = null; } };

  /** Run one engine job the way every job here is run: watched, stoppable. */
  const runJob = async (stage: RunState["stage"], work: (id: string) => Promise<void>) => {
    const id = newJobId();
    stop.current = false;
    runDispatch({ type: "start", stage, jobId: id });
    const un = await onEngineProgress((p) => { if (p.job_id === id) runDispatch({ type: "progress", prog: p }); });
    try {
      await work(id);
      runDispatch({ type: "end" });
    } catch (e) {
      notify(errText(e), true);
      runDispatch({ type: "end", outcome: isCancelled(e) ? "Stopped." : `Did not finish: ${errText(e)}` });
    } finally {
      un();
    }
  };

  /** One engine call inside a longer run, with its own job id and progress. */
  const step = async <T,>(drawingId: string | null, work: (id: string) => Promise<T>): Promise<T> => {
    const id = newJobId();
    runDispatch({ type: "job", jobId: id, drawingId });
    const un = await onEngineProgress((p) => { if (p.job_id === id) runDispatch({ type: "progress", prog: p }); });
    try { return await work(id); } finally { un(); runDispatch({ type: "job", jobId: null, drawingId: null }); }
  };

  const outcome = (text: string) => runDispatch({ type: "outcome", outcome: text });

  /** The analysis landed: the board is planned and goes to the cast. */
  const adopt = async (book: StoryBook, chapterId: string, panelsIn: Panel[],
                       covered: BoardState["coverage"], extra: { story?: string; title?: string } = {}) => {
    const castNow = castFromBook(book, chapterId);
    act({ type: "analysed", book, chapterId, doc: fromArrays(panelsIn), cast: castNow, coverage: covered ?? null,
          projectId: newJobId(), ...extra });
    referenceCache.current = {};
    setSelected(0); setChoices(null); setPagesStale(false); setEditingStory(false);
  };

  /** Make comic: analyse and plan in one pass, then cast. */
  const makeComic = async () => {
    const s = boardRef.current;
    if (!s.story.trim()) { notify("Write the story first.", true); return; }
    if (!storyLibrary) { notify("The story library has not loaded yet.", true); return; }
    let planned = false;
    await runJob("analysing", async (id) => {
      const chapterNow = s.storyBook?.chapters.find((c) => c.id === s.chapterId);
      const again = !!(s.storyBook && s.chapterId && chapterNow && chapterNow.source === s.story);
      const previousBook = appendChapter
        ? storyLibrary.books.find((b) => b.id === (continuationBookId || storyLibrary.activeBookId)) ?? null : null;
      const r = again
        ? await api.resumeStory(id, s.storyBook!, s.chapterId!, analystId || null, s.format, true, s.pageCount)
        : await api.analyzeStory(id, s.story, s.chapterTitle, previousBook, analystId || null, "full",
                                 s.format, s.pageCount);
      await onStoryAnalyzed(r.book);
      await adopt(r.book, r.chapter_id, r.panels, r.coverage ?? null);
      setAppendChapter(false);
      planned = true;
      outcome(r.unanalysed
        ? `${r.stopped ? "Stopped — " : ""}${r.unanalysed} passage${r.unanalysed === 1 ? " is" : "s are"} not analysed `
          + "yet; continue from the story whenever you like."
        : `Planned ${r.panels.length} panels for ${s.pageCount} page${s.pageCount === 1 ? "" : "s"}. `
          + "Check the cast, then approve it.");
    });
    // The sheets are the next thing needed, and drawing them is not a
    // decision: start at once, and the cast screen fills in as they arrive.
    if (planned && !stop.current) await drawSheets();
  };

  /** A chapter the Audiobook (or an earlier board) already analysed. */
  const openChapter = async (value: string) => {
    if (!storyLibrary) return;
    const [bid, cid] = value.split(":");
    const book = storyLibrary.books.find((b) => b.id === bid);
    const chapter = book?.chapters.find((c) => c.id === cid);
    if (!book || !chapter) return;
    if (chapter.analysis === "voice") {
      act({ type: "story", text: chapter.source }); act({ type: "title", text: chapter.title });
      notify("This chapter was analysed for voices only. Make the comic to analyse it for drawing.");
      return;
    }
    await runJob("analysing", async (id) => {
      const s = boardRef.current;
      const r = await api.directStory(id, book, cid, s.format, false, null, s.pageCount);
      const current = r.book ?? book;
      if (r.book) await onStoryAnalyzed(r.book);
      await adopt(current, cid, r.panels, r.coverage ?? null, { story: chapter.source, title: chapter.title });
      outcome(`Loaded “${chapter.title}” — ${r.panels.length} panel${r.panels.length === 1 ? "" : "s"} planned.`);
    });
  };

  /** Analyse what a stopped analysis left, keeping what was done. */
  const continueAnalysis = async () => {
    const s = boardRef.current;
    if (!s.storyBook || !s.chapterId) return;
    await runJob("analysing", async (id) => {
      const r = await api.resumeStory(id, s.storyBook!, s.chapterId!, analystId || null, s.format, false, s.pageCount);
      await onStoryAnalyzed(r.book);
      act({ type: "book", book: r.book, cast: castFromBook(r.book, s.chapterId) });
      act({ type: "replanned", doc: replan(boardRef.current.doc, r.panels), coverage: r.coverage ?? null });
      outcome(r.unanalysed ? `${r.unanalysed} passage${r.unanalysed === 1 ? " is" : "s are"} still not analysed.`
        : `Analysis complete: ${r.panels.length} panels planned. Drawn panels were kept.`);
    });
  };

  /**
   * A new format or page count plans the panels again, at once and by the
   * rules. A panel that still shows the same moment keeps its brief, so what
   * a writer or the user wrote is not silently replaced; how many briefs
   * changed is said.
   */
  const replanFor = async (format: BookFormat, pages: number) => {
    act({ type: "format", format }); act({ type: "pageCount", pages });
    const s = boardRef.current;
    if (!s.storyBook || !s.chapterId || legacy) return;
    await runJob("analysing", async (id) => {
      const r = await api.directStory(id, s.storyBook!, s.chapterId!, format, false, null, s.pageCount);
      if (r.book) { await onStoryAnalyzed(r.book); act({ type: "book", book: r.book, cast: castFromBook(r.book, s.chapterId) }); }
      const before = new Map(s.doc.panels.map((p) => [p.id, p.description ?? ""]));
      const next = replan(s.doc, r.panels);
      const rewritten = next.panels.filter((p) => before.has(p.id) && before.get(p.id) !== (p.description ?? "")).length;
      act({ type: "replanned", doc: next, coverage: r.coverage ?? null });
      setPagesStale(false);
      outcome(`Planned again: ${r.panels.length} panels for ${s.pageCount} page${s.pageCount === 1 ? "" : "s"}.`
        + (rewritten ? ` ${rewritten} panel${rewritten === 1 ? "" : "s"} now show a different moment and have new briefs.` : ""));
    });
  };

  // ---- the cast ------------------------------------------------------------
  /** Whether a vault item still exists; a sheet deleted in the Vault is drawn again. */
  const vaultHas = async (id: string) => (await api.vaultList()).some((item) => item.id === id);

  /** One person's full-body cast sheet, in their chapter outfit. */
  const drawSheet = async (person: CastMember, regenerate: boolean): Promise<boolean> => {
    if (!model || !person.id) return false;
    const s = boardRef.current;
    const pid = person.id;
    try {
      const saved = s.storyBook?.references?.[pid]
        ?? storyLibrary?.books.find((b) => b.id === s.storyBook?.id)?.references?.[pid];
      let image: string | null = null;
      if (!regenerate && saved?.kind === "cast-sheet-v2" && saved.style_words === s.style.words
          && saved.visual === person.description && (saved.outfit ?? "") === (person.outfit ?? "")
          && await vaultHas(saved.image)) {
        image = saved.image;
      }
      if (!image) {
        const size = panelSize(SHEET_ASPECT, panelPx);
        const res = await step(`sheet:${pid}`, (id) => api.generate({
          job_id: id, model_id: model.id,
          prompt: castSheetPrompt(s.style, person.name, person.description, person.outfit ?? ""),
          negative_prompt: null, width: size.width, height: size.height,
          steps: model.steps_default || 4, guidance: Math.min(1.0, model.guidance_max),
          // Seeded by who they are, not where they stand: by position every
          // first-listed person shared a seed and two came back as one face.
          seed: castSeed(pid, (s.sheetRegens[pid] ?? 0) + (regenerate ? 1 : 0)), count: 1,
          images: [], image_strength: null, i2i_mode: null, low_ram: true, preview: false, cache_limit_gb: null,
          allow_over_budget: false, loras: [], mask: null, outpaint_padding: null, outpaint_fill: null,
          project: s.projectId, project_name: projectName,
        }));
        image = res[0] ?? null;
        if (!image) throw new Error(`No cast sheet came back for ${person.name}`);
        onProduced();
        const book = boardRef.current.storyBook;
        if (book) {
          const reference: CharacterReference = { kind: "cast-sheet-v2", image, style_words: s.style.words,
                                                  visual: person.description, outfit: person.outfit ?? "" };
          await onCharacterReference(book.id, pid, reference);
          act({ type: "book", book: { ...book, references: { ...book.references, [pid]: reference } } });
        }
      }
      act({ type: "sheet", id: pid, image, regenerated: regenerate });
      referenceCache.current = {};
      return true;
    } catch (e) {
      notify(`${person.name}'s cast sheet: ${errText(e)}`, true);
      if (isCancelled(e)) stop.current = true;
      return false;
    }
  };

  /** Every recurring person still without a sheet. */
  const drawSheets = async () => {
    const want = missingSheets(boardRef.current);
    if (!want.length || !model) return;
    stop.current = false;
    runDispatch({ type: "start", stage: "sheets" });
    startTimer();
    let made = 0;
    for (const person of want) {
      if (stop.current) break;
      if (await drawSheet(person, false)) made++;
    }
    stopTimer();
    runDispatch({ type: "end", outcome: stop.current ? "Stopped — draw the rest of the cast when you are ready."
      : `${made} cast sheet${made === 1 ? "" : "s"} drawn. Check them, then approve the cast.` });
  };

  const regenerateSheet = async (person: CastMember) => {
    stop.current = false;
    runDispatch({ type: "start", stage: "sheets" });
    const ok = await drawSheet(person, true);
    runDispatch({ type: "end", outcome: ok ? `${person.name}'s sheet drawn again.` : "The sheet was not drawn." });
  };

  const ownPicture = (person: CastMember, image: string | null) => {
    if (!person.id) return;
    act({ type: "sheet", id: person.id, image, regenerated: !!image });
    referenceCache.current = {};
  };

  /** Approve the cast and draw the comic, a page at a time. */
  const approve = async () => {
    const recasting = boardRef.current.recast.length;
    act({ type: "approve" });
    if (recasting) setPagesStale(true);
    await drawComic();
  };

  // ---- drawing ---------------------------------------------------------------
  /** File the picture a panel is about to lose, before anything overwrites it. */
  const supersede = (panelId: string, previousId: string | null | undefined) => {
    if (!previousId) return;
    const had = artOf(boardRef.current.doc, panelId).variants;
    if (had.includes(previousId)) return;
    putArt(panelId, { variants: [...had, previousId] });
  };
  const putArt = (panelId: string, patch: Partial<PanelArt>) =>
    act({ type: "doc", update: (d) => setArt(d, panelId, patch) });
  const retirePages = () => setPagesStale(true);

  /**
   * One panel drawn against its cast sheets, at its cell's shape. Returns the
   * new picture's id; nothing is written to the board here.
   */
  const renderPanel = async (i: number, seed: number): Promise<string | null> => {
    const s = boardRef.current;
    const panel = s.doc.panels[i];
    if (!model || !panel) return null;
    const note = artOf(s.doc, panel.id).note;
    const people = new Map((s.cast?.people ?? []).map((p) => [p.id, p]));
    // Left to right as they stand, so the reference lineup reads like the panel.
    const actors = [...(panel.cast ?? [])].sort((a, b) =>
      SLOT_ORDER.indexOf(panel.positions?.[a.name] ?? "centre") - SLOT_ORDER.indexOf(panel.positions?.[b.name] ?? "centre"));
    const sheetIds = actors.map((a) => s.sheets[a.character_id]).filter((x): x is string => !!x);
    const place = placesOf(s.doc.panels, s.format, s.layout)[i];
    const size = panelSize(place?.aspect ?? 1, panelPx);
    return step(panel.id, async (id) => {
      let identity: string | null = sheetIds[0] ?? null;
      if (sheetIds.length > 1) {
        const key = sheetIds.join(":");
        identity = referenceCache.current[key] ?? await api.characterReference(id, sheetIds);
        referenceCache.current[key] = identity;
      }
      if (stop.current) return null;
      const refs = panelReferences(panel, identity, null, model.max_edit_images || 1);
      const params = {
        job_id: id, model_id: model.id,
        prompt: panelPrompt({ ...panel, cast: actors.map((a) => ({
          ...a, visual: people.get(a.character_id)?.description || a.visual,
          outfit: people.get(a.character_id)?.outfit ?? "" })) }, s.style, note),
        negative_prompt: null, width: size.width, height: size.height,
        steps: model.steps_default || 4, guidance: Math.min(1.0, model.guidance_max),
        seed, count: 1, images: refs,
        // "edit" is the reference-conditioned route: the sheets and the brief together.
        image_strength: null, i2i_mode: refs.length ? "edit" as const : null,
        low_ram: true, preview: false, cache_limit_gb: null, allow_over_budget: false, loras: [],
        mask: null, outpaint_padding: null, outpaint_fill: null,
        project: s.projectId, project_name: projectName, project_index: i,
      };
      // A panel with no reference is a plain drawing; the edit route needs one.
      const res = refs.length ? await api.editImage(params) : await api.generate(params);
      return res[0] ?? null;
    });
  };

  /** Draw a panel and put it on the board, keeping what it replaces. */
  const drawOne = async (i: number, redrawCount?: number): Promise<boolean> => {
    const panel = boardRef.current.doc.panels[i];
    if (!panel) return false;
    const count = redrawCount ?? artOf(boardRef.current.doc, panel.id).redraws;
    try {
      const image = await renderPanel(i, panelSeed(i, count));
      if (!image) return false;
      const before = artOf(boardRef.current.doc, panel.id).drawn;
      // Before the id is overwritten: the only moment the board still knows it.
      supersede(panel.id, before);
      putArt(panel.id, { drawn: image, redraws: count });
      // A composed page is only as current as the panels in it.
      if (before) retirePages();
      onProduced();
      return true;
    } catch (e) {
      notify(`Panel ${i + 1}: ${errText(e)}`, true);
      if (isCancelled(e)) stop.current = true;
      lastFailure.current = e;
      return false;
    }
  };

  /**
   * Draw the comic a page at a time, composing each page as it is finished,
   * until every panel has a picture or Stop. No "draw the next four": the
   * board carries on by itself, and Stop leaves it ready to continue.
   */
  const drawComic = async () => {
    if (!model || !boardRef.current.castApproved) return;
    stop.current = false;
    runDispatch({ type: "start", stage: "drawing" });
    startTimer();
    const times = (panelTimes.current[panelPx] ??= []);
    const skip = new Set<string>();
    let drew = 0, failed = 0;
    let end: Awaited<ReturnType<typeof drawBatch>> = { end: "done" };
    while (!stop.current) {
      const s = boardRef.current;
      const where = placesOf(s.doc.panels, s.format, s.layout);
      const order = nextPage(s.doc, where).filter((i) => !skip.has(s.doc.panels[i].id));
      if (!order.length) break;
      const page = where[order[0]]?.page ?? 0;
      end = await drawBatch(order, async (i) => {
        lastFailure.current = null;
        const perPanel = times.length ? times.reduce((a, b) => a + b, 0) / times.length : (panelPx >= 768 ? 170 : 90) * 1000;
        runDispatch({ type: "batch", batch: { k: order.indexOf(i) + 1, n: order.length, perPanelMs: perPanel,
                                               startedAt: Date.now(), page } });
        const began = Date.now();
        const ok = await drawOne(i);
        if (ok) { times.push(Date.now() - began); drew++; } else { failed++; skip.add(boardRef.current.doc.panels[i]?.id ?? ""); }
        return ok ? { ok: true } : { ok: false, error: lastFailure.current };
      }, () => stop.current, isEngineDeath);
      if (end.end !== "done") break;
      // The page is finished: show it composed now, not after the last one.
      await compose(true);
    }
    stopTimer();
    runDispatch({ type: "batch", batch: null });
    const now = toArrays(boardRef.current.doc).drawn.filter(Boolean).length;
    const tally = `${now} of ${boardRef.current.doc.panels.length} drawn`;
    const lost = failed ? ` ${failed} panel${failed === 1 ? "" : "s"} failed.` : "";
    runDispatch({ type: "end", outcome:
      end.end === "stopped" || end.end === "cancelled" || stop.current ? `Stopped — ${tally}.${lost} Continue whenever you like.`
        : end.end === "engine" ? `The engine stopped — ${tally}. ${errText(end.error)}`
          : end.end === "repeated" ? `Two panels in a row failed, so drawing stopped — ${tally}.`
            : failed ? `${failed} panel${failed === 1 ? "" : "s"} failed, ${drew} drawn — ${tally}.`
              : `Drew ${drew} panel${drew === 1 ? "" : "s"} — ${tally}.` });
  };

  /**
   * Set every finished page, in the cells its panels were drawn for, with
   * balloons on their speakers' sides. Pages this session made before are
   * discarded once the new ones exist: they were steps on the way, not
   * versions anyone chose.
   */
  const compose = async (quiet = false) => {
    const s = boardRef.current;
    const where = placesOf(s.doc.panels, s.format, s.layout);
    const done = new Set(finishedPages(s.doc, where));
    const idx = s.doc.panels.map((_, i) => i).filter((i) => done.has(where[i]?.page ?? -1));
    if (!idx.length) { if (!quiet) notify("Draw at least one whole page first.", true); return; }
    const outer = runRef.current.stage === "idle";
    if (outer) runDispatch({ type: "start", stage: "composing" });
    try {
      const made = await api.composeBoard(newJobId(),
        idx.map((i) => artOf(s.doc, s.doc.panels[i].id).drawn!),
        idx.map((i) => s.doc.panels[i].caption ?? ""),
        idx.map((i) => shotClass(s.doc.panels[i])),
        idx.map((i) => (s.doc.panels[i].dialogue ?? []).map((d) => ({ speaker: d.speaker, text: d.text,
          side: speakerSide(s.doc.panels[i], d.speaker, d.offscreen) }))),
        idx.map((i) => s.doc.panels[i].scene ?? 1),
        // A webtoon is one scroll, whatever the board's page setting.
        s.format === "webtoon" ? "strip" : s.layout, s.projectId, projectName,
        { pages: idx.map((i) => where[i].page), cells: idx.map((i) => where[i].cell),
          direction: s.format === "manga" ? "rtl" : "ltr" });
      const superseded = autoPages.current.filter((id) => !made.pages.includes(id));
      autoPages.current = made.pages;
      act({ type: "pages", pages: made.pages,
            overflow: anchorOverflow(made.overflow, idx.map((i) => s.doc.panels[i].id), boardRef.current.doc) });
      setPagesStale(false);
      onProduced();
      for (const id of superseded) { try { await api.vaultDelete(id); } catch { /* locked or already gone */ } }
      if (!quiet) outcome(`Composed ${made.pages.length} page${made.pages.length === 1 ? "" : "s"}.`);
    } catch (e) {
      notify(errText(e), true);
    } finally {
      if (outer) runDispatch({ type: "end" });
    }
  };

  /**
   * Alternatives for one panel, each from a new seed, to choose from. The
   * current picture stays until one is picked; the others are kept in the
   * panel's history either way.
   */
  const variants = async (i: number, n: number) => {
    const panel = boardRef.current.doc.panels[i];
    if (!panel) return;
    stop.current = false;
    runDispatch({ type: "start", stage: "variants" });
    const start = artOf(boardRef.current.doc, panel.id).redraws;
    const ids: string[] = [];
    for (let k = 1; k <= n && !stop.current; k++) {
      try {
        const image = await renderPanel(i, panelSeed(i, start + k));
        if (image) { ids.push(image); setChoices({ panelId: panel.id, ids: [...ids] }); onProduced(); }
      } catch (e) {
        notify(`Panel ${i + 1}: ${errText(e)}`, true);
        if (isCancelled(e)) break;
      }
    }
    putArt(panel.id, { redraws: start + n });
    runDispatch({ type: "end", outcome: ids.length ? `${ids.length} version${ids.length === 1 ? "" : "s"} of panel ${i + 1} to choose from.`
      : "No other version was drawn." });
  };

  const pick = async (image: string) => {
    if (!choices) return;
    const art = artOf(boardRef.current.doc, choices.panelId);
    const rest = [art.drawn, ...choices.ids].filter((x): x is string => !!x && x !== image);
    putArt(choices.panelId, { drawn: image, variants: [...art.variants.filter((x) => x !== image),
                                                       ...rest.filter((x) => !art.variants.includes(x))] });
    setChoices(null);
    if (image !== art.drawn) await compose(true);
  };

  /** Redraw one panel with a new seed and its note. */
  const redraw = async (i: number) => {
    stop.current = false;
    runDispatch({ type: "start", stage: "drawing" });
    const panel = boardRef.current.doc.panels[i];
    const ok = panel ? await drawOne(i, artOf(boardRef.current.doc, panel.id).redraws + 1) : false;
    runDispatch({ type: "end", outcome: ok ? `Panel ${i + 1} redrawn.` : `Panel ${i + 1} could not be redrawn.` });
    if (ok) await compose(true);
  };

  /**
   * Repair one painted region of a drawn panel. The panel itself is the
   * input image, not a cast sheet: a masked edit regenerates inside the
   * painted region of the picture it is given.
   */
  const repair = async (i: number) => {
    const s = boardRef.current;
    const panel = s.doc.panels[i];
    const current = panel ? artOf(s.doc, panel.id).drawn : null;
    if (!model || !panel || !current) return;
    if (!maskBytes) { notify("Paint over the part to redraw first.", true); return; }
    stop.current = false;
    runDispatch({ type: "start", stage: "repairing" });
    const nextRedraw = artOf(s.doc, panel.id).redraws + 1;
    let maskId: string | null = null;
    try {
      // The mask is sealed into the vault, handed over by id, and removed once read.
      maskId = await api.vaultImportBytes(maskBytes, "mask.png", "image/png", "mask");
      const note = artOf(s.doc, panel.id).note.trim();
      const res = await step(panel.id, (id) => api.editImage({
        job_id: id, model_id: model.id,
        prompt: note ? `${note}. ${s.style.words}` : panelPrompt(panel, s.style),
        negative_prompt: null, width: panelSize(places[i]?.aspect ?? 1, panelPx).width,
        height: panelSize(places[i]?.aspect ?? 1, panelPx).height,
        steps: model.steps_default || 4, guidance: Math.min(1.0, model.guidance_max),
        seed: panelSeed(i, nextRedraw), count: 1, images: [current],
        image_strength: null, i2i_mode: "edit", low_ram: true, preview: false, cache_limit_gb: null,
        allow_over_budget: false, loras: [], mask: maskId, outpaint_padding: null, outpaint_fill: null,
        project: s.projectId, project_name: projectName, project_index: i,
      }));
      if (!res[0]) { notify("The engine returned no picture.", true); return; }
      supersede(panel.id, current);
      putArt(panel.id, { drawn: res[0], redraws: nextRedraw });
      retirePages();
      setMaskBytes(null);
      onProduced();
      outcome(`Panel ${i + 1} repaired. The picture it replaces is in its history.`);
    } catch (e) {
      notify(isCancelled(e) ? "Cancelled." : `Panel ${i + 1}: ${errText(e)}`, !isCancelled(e));
    } finally {
      if (maskId) { try { await api.vaultDelete(maskId); } catch { /* lock/cleanup race */ } }
      runDispatch({ type: "end" });
    }
    await compose(true);
  };

  /** Put a previous picture of a panel back; the current one joins the history. */
  const restoreVariant = (i: number, id: string) => {
    const panel = boardRef.current.doc.panels[i];
    if (!panel) return;
    const art = artOf(boardRef.current.doc, panel.id);
    const rest = art.variants.filter((x) => x !== id);
    putArt(panel.id, { drawn: id, variants: art.drawn ? [...rest, art.drawn] : rest });
    retirePages();
    notify(`Panel ${i + 1} put back to an earlier picture.`);
  };

  /**
   * Change one field of one panel. What it invalidates is decided by
   * `staleness` in boardDoc: the words on the page never cost a redraw.
   */
  const edit = (i: number, patch: Partial<Panel>) => {
    const panel = boardRef.current.doc.panels[i];
    if (!panel) return;
    if (patch.characters && panel.cast !== undefined) {
      const people = boardRef.current.cast?.people ?? [];
      patch = { ...patch, cast: patch.characters.map((name) => {
        const person = people.find((c) => c.name.toLowerCase() === name.toLowerCase()
          || c.aliases?.some((a) => a.toLowerCase() === name.toLowerCase()));
        const old = panel.cast?.find((a) => a.character_id === person?.id || a.name === name);
        // Someone typed in who is not in the cast is drawn from their name.
        return { character_id: person?.id ?? `named:${name}`, name: person?.name ?? name,
                 visual: person?.description ?? "", expression: old?.expression ?? "",
                 body_language: old?.body_language ?? "", appearance: old?.appearance ?? "" };
      }) };
    }
    let stale = { art: false, pages: false };
    act({ type: "doc", update: (d) => { const r = editPanel(d, panel.id, patch); stale = r.stale; return r.doc; } });
    if (stale.art) setChoices(null);
    if (stale.pages) retirePages();
  };

  /** Move, add, remove, split or join, by identity, keeping the selection. */
  const structural = (op: (d: BoardDoc) => BoardDoc, keep: string | null, at?: number) => {
    act({ type: "doc", update: op });
    const next = boardRef.current.doc;
    const to = keep ? next.panels.findIndex((x) => x.id === keep) : at ?? 0;
    setSelected(Math.max(0, Math.min(to === -1 ? 0 : to, next.panels.length - 1)));
    retirePages();
  };
  const movePanel = (i: number, by: number) => {
    const panel = doc.panels[i];
    if (!panel || i + by < 0 || i + by >= doc.panels.length) return;
    structural((d) => movePanelIn(d, panel.id, i + by), panel.id);
  };
  /** A new panel is a structured one, in the same scene and place as its neighbour. */
  const addPanelAfter = (i: number) => {
    const near = doc.panels[i];
    structural((d) => insertPanel(d, i + 1, {
      shot: "MS", angle: "eye", framing: "single", subject: "", action: "", moment: "",
      setting: near?.setting ?? "", place: near?.place, place_key: near?.place_key,
      location_state: near?.location_state, mood: near?.mood, scene: near?.scene ?? 1,
      scene_title: near?.scene_title, chapter_id: near?.chapter_id,
      character_in_frame: false, characters: [], cast: [], positions: {}, caption: "", dialogue: [],
      description: "", covers: [], folded: [],
    }), null, i + 1);
    notify("An empty panel. Write its brief, then continue drawing.");
  };
  const deletePanel = (i: number) => {
    const panel = doc.panels[i];
    if (!panel) return;
    const had = artOf(doc, panel.id).drawn;
    structural((d) => removePanel(d, panel.id).doc, null, Math.max(0, i - 1));
    notify(had ? `Panel ${i + 1} removed. Its picture stays in the Vault.` : `Panel ${i + 1} removed.`);
  };
  const splitPanel = (i: number) => {
    const panel = doc.panels[i];
    if (!panel) return;
    structural((d) => splitPanelIn(d, panel.id, { action: "" }), panel.id);
  };
  const mergePanel = (i: number) => {
    const panel = doc.panels[i];
    if (!panel || i + 1 >= doc.panels.length) return;
    structural((d) => mergePanelIn(d, panel.id), panel.id);
  };

  const changeStyle = (id: string, words?: string) => {
    const next = id === CUSTOM_STYLE ? lockCustomStyle(words ?? "") : lockStyle(id);
    if (!next.words.trim()) { notify("Describe the look first, then it can be used.", true); return; }
    act({ type: "style", style: next });
    referenceCache.current = {};
    setChoices(null);
  };

  const cancel = async () => {
    stop.current = true;
    const job = runRef.current.jobId;
    if (job) { try { await api.cancelJob(job); } catch { /* already gone */ } }
  };

  const newBoard = () => {
    act({ type: "reset" });
    referenceCache.current = {};
    setSelected(0); setChoices(null); setPagesStale(false);
    runDispatch({ type: "end", outcome: null });
  };

  // ---- the screen ------------------------------------------------------------
  if (usable.length === 0) {
    return (
      <div className="panel">
        <h2>Picture board</h2>
        <div className="notice warn">
          <strong>Needs a model that can both generate and edit</strong>
          A board draws a cast sheet for each person and draws every panel against them, so the image
          model has to do both. FLUX.2 Klein is the smallest that does — add it from the Models tab.
        </div>
      </div>
    );
  }
  if (!draftLoaded) {
    return <div className="empty-state">Opening your encrypted Board draft…</div>;
  }

  const selIndex = panels ? Math.min(selected, Math.max(0, panels.length - 1)) : 0;
  const sel = panels?.[selIndex] ?? null;
  const selLocked = busy && !!sel && run.drawingId === sel.id;
  const currentStep = viewStep ?? workflowStep;
  const stepState: Record<StepName, { reached: boolean; done: boolean; detail: string }> = {
    story: { reached: true, done: board.divided,
             detail: board.divided ? `${panels?.length ?? 0} panels planned` : board.story.trim() ? "ready" : "not started" },
    cast: { reached: board.divided, done: castApproved,
            detail: !board.divided ? "not yet" : castApproved ? "approved"
              : sheetsLeft ? `${casting.length - sheetsLeft} of ${casting.length} sheets` : "ready to approve" },
    draw: { reached: castApproved, done: board.divided && remaining === 0,
            detail: panels ? `${drawnCount} of ${panels.length} drawn` : "not yet" },
    pages: { reached: board.pages.length > 0, done: board.pages.length > 0 && !pagesStale,
             detail: board.pages.length ? `${board.pages.length} page${board.pages.length === 1 ? "" : "s"}` : "not yet" },
  };

  /**
   * The one run button, and what it does now: the next step when idle, Stop
   * while running. A step that cannot run says why (aria-disabled stays
   * focusable and readable; disabled does neither).
   */
  const about = `about ${estimate.low}–${estimate.high} min`;
  const primary: { label: string; go: () => void; blocked?: string } =
    !board.divided || legacy
      ? { label: legacy ? "Make the comic again" : "Make comic", go: () => void makeComic(),
          blocked: !analysts.length ? "Install a story analyst from Models."
            : !storyLibrary ? "The story library is still loading."
              : !board.story.trim() ? "Write or paste the story first." : undefined }
      : sheetsLeft > 0
        ? { label: `Draw the cast sheets (${sheetsLeft}) · ${about}`, go: () => void drawSheets() }
        : !castApproved
          ? { label: board.recast.length
                ? `Approve and redraw ${board.recast.flatMap((id) => doc.panels.filter((p) =>
                    (p.cast ?? []).some((a) => a.character_id === id))).length} panels`
                : `Approve the cast and draw · ${about}`,
              go: () => void approve() }
          : remaining > 0
            ? { label: drawnCount ? `Continue drawing · ${remaining} left · ${about}` : `Draw the comic · ${about}`,
                go: () => void drawComic() }
            : { label: pagesStale || !board.pages.length ? "Update the pages" : "Make the pages again",
                go: () => void compose() };
  const running = run.stage === "analysing" ? "the analysis" : run.stage === "sheets" ? "the cast sheets"
    : run.stage === "composing" ? "composing" : "drawing";
  const progLabel = (() => {
    if (run.stage === "analysing") return "Reading the story and planning the panels";
    if (run.stage === "sheets") {
      const who = cast?.people.find((p) => `sheet:${p.id}` === run.drawingId)?.name;
      return who ? `Drawing ${who}'s cast sheet` : "Drawing the cast sheets";
    }
    if (run.stage === "composing") return "Composing the pages";
    if (run.batch) {
      const left = Math.max(0, (run.batch.n - run.batch.k + 1) * run.batch.perPanelMs - (Date.now() - run.batch.startedAt));
      return `Page ${(run.batch.page ?? 0) + 1}: panel ${run.batch.k} of ${run.batch.n} · `
        + (left < 60000 ? "under a minute left" : `about ${Math.ceil(left / 60000)} min left`);
    }
    const at = doc.panels.findIndex((x) => x.id === run.drawingId);
    return at >= 0 ? `Drawing panel ${at + 1}` : "Drawing";
  })();
  const pageCells = finished.map((page) => places.map((pl, index) => ({ index, cell: pl.cell, page: pl.page }))
    .filter((x) => x.page === page && board.layout !== "strip" && board.format !== "webtoon"));

  return (
    <div className="board" style={{ padding: 14 }}>
      <div className="board-top">
        <ol className="board-steps" aria-label="Board steps">
          {STEPS.map(([name, label], n) => {
            const st = stepState[name];
            const inner = (
              <>
                <span className="step-n" aria-hidden="true">{st.done ? "✓" : n + 1}</span>
                <span className="step-label">{label}</span>
                <span className="step-detail">{st.done && <span className="visually-hidden">done, </span>}{st.detail}</span>
              </>
            );
            return (
              <li key={name} className={[name === currentStep ? "current" : "", st.done ? "done" : "",
                                         st.reached ? "" : "ahead"].filter(Boolean).join(" ")}>
                {st.reached ? (
                  <button type="button" aria-current={name === currentStep ? "step" : undefined}
                          onClick={() => setViewStep(name)}>{inner}</button>
                ) : <span aria-current={name === currentStep ? "step" : undefined}>{inner}</span>}
              </li>
            );
          })}
        </ol>
        <span className="board-save">{unsaved ? "Not being saved" : "Encrypted draft saved automatically"}</span>
      </div>

      <div className="board-run" role="group" aria-label="Run">
        <button ref={runButton} type="button" className={"btn small" + (busy ? " danger" : " primary")}
                aria-disabled={!busy && primary.blocked ? true : undefined}
                aria-describedby={!busy && primary.blocked ? "board-run-why" : undefined}
                aria-label={busy ? `Stop ${running}` : undefined}
                onClick={() => { if (busy) void cancel(); else if (!primary.blocked) primary.go(); }}>
          {busy ? "Stop" : primary.label}
        </button>
        {!busy && primary.blocked && <span id="board-run-why" className="t-help">{primary.blocked}</span>}
        {panels && (
          <span className="board-tally"><b>{drawnCount}</b> of {panels.length} drawn</span>
        )}
        {!busy && remaining > 0 && (
          <select value={panelPx} aria-label="Panel size" onChange={(e) => setPanelPx(Number(e.target.value))}>
            {PANEL_SIZES.map(([label, px]) => <option key={px} value={px}>{label}</option>)}
          </select>
        )}
        <div style={{ flex: 1 }} />
        {!busy && (board.story || panels) && (
          <button className={"btn small" + (confirmReset ? " danger" : "")}
                  onClick={() => {
                    if (drawnCount && !confirmReset) { setConfirmReset(true); return; }
                    setConfirmReset(false); newBoard();
                  }}>
            {confirmReset ? `Start over, discarding ${drawnCount} drawn panel${drawnCount === 1 ? "" : "s"}?` : "New board"}
          </button>
        )}
        {elapsed > 0 && busy && (
          <span className="board-elapsed">{Math.floor(elapsed / 60)}m {elapsed % 60}s</span>
        )}
      </div>

      {busy && (
        <div className="board-progress">
          {run.prog && <div className="board-progress-line">{progLabel}</div>}
          <JobProgress p={run.prog} label={progLabel} />
        </div>
      )}
      <p className="board-outcome" role="status">{run.outcome ?? ""}</p>

      {unsaved && (
        <div className="notice warn">
          <strong>This board is not being saved</strong>
          <div>{unsaved}</div>
          <div>Everything on screen is still here; copy the story out before closing the app if this does not clear.</div>
        </div>
      )}
      {previous && (
        <div className="notice warn">
          <strong>This board opened empty, and an earlier version of it exists</strong>
          <div>{previous.story?.trim() ? `It begins "${previous.story.trim().slice(0, 90)}…"` : "It has panels but no story."}</div>
          <div className="board-structure">
            <button className="btn small primary" disabled={busy}
                    onClick={() => { applyDraft(previous); setPrevious(null); }}>Put it back</button>
            <button className="btn small" onClick={() => setPrevious(null)}>No, this board is meant to be empty</button>
          </div>
        </div>
      )}
      {legacy && (
        <div className="notice warn">
          <strong>This board was made by an older version</strong>
          Its pictures and pages are kept and can be read, but it cannot be drawn further. Make the comic
          again from its story to cast it and draw it the new way; nothing here is deleted.
        </div>
      )}

      {unanalysed > 0 && (
        <div className="notice warn">
          <strong>{unanalysed} passage{unanalysed === 1 ? "" : "s"} not analysed yet</strong>
          <div>Their text is on the board without cast or direction. Finished passages and drawn panels are kept.</div>
          <button className="btn small primary" disabled={busy} onClick={() => void continueAnalysis()}>Continue analysis</button>
        </div>
      )}
      {/* ---- the story and its options, until the comic is planned ---- */}
      {(!board.divided || editingStory || currentStep === "story") && (
        <section className="board-story" aria-label="Story">
          {board.divided && !editingStory && (
            <div className="notice">
              The comic is planned from this story.
              <button type="button" className="btn small" disabled={busy} onClick={() => setEditingStory(true)}>
                Edit the story
              </button>
            </div>
          )}
          {(!board.divided || editingStory) && (
            <>
              {!!storyLibrary?.books.length && (
                <label>Use an analysed chapter
                  <select value="" disabled={busy} onChange={(e) => void openChapter(e.target.value)}>
                    <option value="">Choose a saved chapter…</option>
                    {storyLibrary.books.flatMap((book) => book.chapters.map((chapter) =>
                      <option key={chapter.id} value={`${book.id}:${chapter.id}`}>
                        {book.chapters[0]?.title} · {chapter.title}{chapter.analysis === "voice" ? " (voices only)" : ""}
                      </option>))}
                  </select>
                </label>
              )}
              <label>Chapter title
                <input value={board.chapterTitle} disabled={busy} onChange={(e) => act({ type: "title", text: e.target.value })} />
              </label>
              <textarea value={board.story} rows={14} disabled={busy} aria-label="Manuscript prose"
                        aria-describedby="board-ms-hint" onChange={(e) => act({ type: "story", text: e.target.value })}
                        placeholder={"Mira comes home and finds the door already open. Nothing is taken, but every "
                          + "photograph has been turned to face the wall."} />
              <div id="board-ms-hint" className="t-help">
                Prose, not prompts. {board.story.trim() ? `${board.story.trim().split(/\s+/).length} words.` : ""}
              </div>
              {!!storyLibrary?.books.length && (
                <label className="board-check">
                  <input type="checkbox" checked={appendChapter} disabled={busy}
                         onChange={(e) => setAppendChapter(e.target.checked)} />
                  Add to the current book and reuse its cast
                </label>
              )}
              {appendChapter && storyLibrary && (
                <label>Continue book
                  <select value={continuationBookId || storyLibrary.activeBookId || storyLibrary.books.at(-1)?.id}
                          disabled={busy} onChange={(e) => setContinuationBookId(e.target.value)}>
                    {storyLibrary.books.map((book) => <option key={book.id} value={book.id}>
                      {book.chapters[0]?.title} · {book.chapters.length} chapters</option>)}
                  </select>
                </label>
              )}
            </>
          )}
          <div className="board-options">
            <label>Make it a
              <select value={board.format} disabled={busy}
                      onChange={(e) => void replanFor(e.target.value as BookFormat, board.pageCount)}>
                <option value="manga">Manga — about 4 panels a page, read right to left</option>
                <option value="comic">Comic — about 5 panels a page</option>
                <option value="webtoon">Webtoon — one vertical scroll</option>
                <option value="picture">Picture book — one spread per scene, the full text</option>
              </select>
            </label>
            <label>Pages
              <input type="number" min={1} max={200} value={board.pageCount} disabled={busy}
                     onChange={(e) => act({ type: "pageCount", pages: Number(e.target.value) })}
                     onBlur={() => { if (board.divided) void replanFor(board.format, board.pageCount); }} />
            </label>
            <label>Story analyst
              <select value={analystId} disabled={busy} onChange={(e) => setAnalystId(e.target.value)}>
                <option value="">Best installed model for this Mac</option>
                {analysts.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
              </select>
            </label>
          </div>
          <StylePicker style={board.style} busy={busy}
                       drawn={drawnCount > 0 || Object.keys(board.sheets).length > 0} onChange={changeStyle} />
          <Field label="Model">
            <select value={model?.id ?? ""} disabled={busy} onChange={(e) => setModelId(e.target.value)}>
              {usable.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </Field>
          {editingStory && (
            <button type="button" className="btn small" onClick={() => setEditingStory(false)}>Done editing</button>
          )}
        </section>
      )}

      {/* ---- the cast: the one gate before drawing ---- */}
      {board.divided && !legacy && (currentStep === "cast" || !castApproved) && cast && (
        <CastScreen people={casting} minor={minor} sheets={board.sheets} drawingId={run.drawingId}
                    busy={busy} approved={castApproved} recast={board.recast} notify={notify}
                    onEdit={(id, patch) => act({ type: "person", id, patch })}
                    onLead={(id) => act({ type: "lead", id })}
                    onRegenerate={(p) => void regenerateSheet(p)}
                    onOwnPicture={ownPicture}
                    onPromote={(p) => p.id && act({ type: "person", id: p.id, patch: { recurring: true } })} />
      )}

      {/* ---- the comic, page by page, and the panel being fixed ---- */}
      {panels && (
        <div className="board-work">
          <div className="board-pane">
            <PagesPane pages={board.pages} pageCells={pageCells} overflow={board.overflow.flatMap((o) => {
                         if (!o.panelId) return [o];
                         const at = doc.panels.findIndex((x) => x.id === o.panelId);
                         return at < 0 ? [] : [{ ...o, panel: at }];
                       })}
                       coverage={coverage} stale={pagesStale} busy={busy} selected={selIndex}
                       onPick={setSelected} onUpdate={() => void compose()} headRef={pagesHead} />
            <div className="board-cards" aria-label="Panels">
              {panels.map((p, i) => (
                <Fragment key={p.id}>
                  {(i === 0 || places[i]?.page !== places[i - 1]?.page) && (
                    <div className="board-scene">Page {(places[i]?.page ?? 0) + 1}</div>
                  )}
                  <button className="board-card" aria-current={i === selIndex ? "true" : "false"}
                          onClick={() => setSelected(i)}>
                    <span className="shot">
                      {drawn[i] ? <img src={vaultUrl(drawn[i]!)} alt={`Panel ${i + 1}`} />
                        : run.drawingId === p.id ? "drawing…" : "—"}
                      <span className="n">{String(i + 1).padStart(2, "0")}</span>
                    </span>
                    <span className="cap"><b>{p.shot}</b>{(p.description || p.moment || p.caption || "").slice(0, 60)}</span>
                  </button>
                </Fragment>
              ))}
            </div>
          </div>
          <div className="board-pane">
            <div className="board-pane-head">Panel {selIndex + 1}</div>
            {sel && (
              <PanelInspector panel={sel} index={selIndex} count={panels.length} art={artOf(doc, sel.id)}
                              prompt={panelPrompt(sel, board.style, notes[selIndex])} locked={selLocked}
                              busy={busy} readOnly={legacy}
                              choices={choices?.panelId === sel.id ? choices.ids : []} maskReady={!!maskBytes}
                              onEdit={(patch) => edit(selIndex, patch)}
                              onNote={(note) => putArt(sel.id, { note })}
                              onVariants={(n) => void variants(selIndex, n)}
                              onPick={(id) => void pick(id)} onDiscardChoices={() => setChoices(null)}
                              onRedraw={() => void redraw(selIndex)} onRepair={() => void repair(selIndex)}
                              onMask={setMaskBytes} onRestore={(id) => restoreVariant(selIndex, id)}
                              onMove={(by) => movePanel(selIndex, by)} onAddAfter={() => addPanelAfter(selIndex)}
                              onSplit={() => splitPanel(selIndex)} onMerge={() => mergePanel(selIndex)}
                              onRemove={() => deletePanel(selIndex)} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
