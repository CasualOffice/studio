import type { BookFormat, Cast, CastMember, EngineProgress, Panel } from "./types";
import type { StoryBook } from "./story";
import type { Coverage } from "./boardDraft";
import type { TextOverflow } from "./api";
import { artOf, setArt, type BoardDoc } from "./boardDoc";
import { lockStyle, STYLES, type StyleLock } from "./board";
import { DEFAULT_PAGES, layoutBoard, type PanelPlace } from "./layout";

/**
 * The board's workflow, as one reducer.
 *
 * The component used to hold this in fifty-four separate pieces of state,
 * set from wherever a job happened to end, so the rules of the workflow --
 * nothing is drawn before the cast is approved, a changed sheet redraws only
 * that person's panels, a new story clears everything read from the old one
 * -- lived in no one place and were broken by whichever path forgot one.
 * They are here, as transitions, and tested as such.
 *
 *   story -> (Make comic) -> cast -> (Approve) -> drawing, page by page
 *                              ^                         |
 *                              +----- Recast someone ----+
 */

export interface BoardState {
  story: string;
  chapterTitle: string;
  storyBook: StoryBook | null;
  chapterId: string | null;
  cast: Cast | null;
  /** Who leads, when the person chose; otherwise the cast's own lead. */
  leadId: string | null;
  sheets: Record<string, string>;
  sheetRegens: Record<string, number>;
  castApproved: boolean;
  /** People whose sheet changed after approval, waiting on approval again. */
  recast: string[];
  doc: BoardDoc;
  divided: boolean;
  pages: string[];
  overflow: (TextOverflow & { panelId?: string })[];
  coverage: Coverage | null;
  style: StyleLock;
  format: BookFormat;
  pageCount: number;
  layout: string;
  projectId: string | null;
}

export const emptyBoard = (style: StyleLock = lockStyle(STYLES[0].id),
                           format: BookFormat = "manga", layout = "page"): BoardState => ({
  story: "", chapterTitle: "Chapter 1", storyBook: null, chapterId: null, cast: null, leadId: null,
  sheets: {}, sheetRegens: {}, castApproved: false, recast: [],
  doc: { panels: [], art: {} }, divided: false, pages: [], overflow: [], coverage: null,
  style, format, pageCount: DEFAULT_PAGES, layout, projectId: null,
});

export type BoardAction =
  | { type: "load"; state: BoardState }
  | { type: "story"; text: string }
  | { type: "title"; text: string }
  | { type: "analysed"; book: StoryBook; chapterId: string; doc: BoardDoc; cast: Cast;
      coverage: Coverage | null; projectId: string; story?: string; title?: string }
  | { type: "book"; book: StoryBook; cast?: Cast }
  | { type: "replanned"; doc: BoardDoc; coverage?: Coverage | null }
  | { type: "person"; id: string; patch: Partial<Pick<CastMember, "description" | "outfit" | "name" | "recurring">> }
  | { type: "lead"; id: string }
  | { type: "sheet"; id: string; image: string | null; regenerated?: boolean }
  | { type: "approve" }
  | { type: "doc"; update: (doc: BoardDoc) => BoardDoc }
  | { type: "pages"; pages: string[]; overflow: (TextOverflow & { panelId?: string })[] }
  | { type: "style"; style: StyleLock }
  | { type: "format"; format: BookFormat }
  | { type: "pageCount"; pages: number }
  | { type: "layout"; layout: string }
  | { type: "reset" };

/** The cast with the chosen lead first and marked as such. */
function withLead(cast: Cast, leadId: string | null): Cast {
  if (!leadId || !cast.people.some((p) => p.id === leadId)) return cast;
  const people = cast.people.map((p) => ({
    ...p, tier: p.id === leadId ? 1 : p.recurring ? 2 : 3,
    recurring: p.id === leadId || !!p.recurring,
  }));
  people.sort((a, b) => Number(b.id === leadId) - Number(a.id === leadId));
  return { ...cast, people };
}

/** Panels a person appears in. */
export function panelsWith(doc: BoardDoc, characterId: string): string[] {
  return doc.panels.filter((p) => (p.cast ?? []).some((a) => a.character_id === characterId)).map((p) => p.id);
}

/** Retire the pictures of these panels to their history, to be drawn again. */
function retire(doc: BoardDoc, ids: string[]): BoardDoc {
  let next = doc;
  for (const id of ids) {
    const art = artOf(next, id);
    if (!art.drawn) continue;
    next = setArt(next, id, {
      drawn: null, redraws: art.redraws + 1,
      variants: art.variants.includes(art.drawn) ? art.variants : [...art.variants, art.drawn],
    });
  }
  return next;
}

export function boardReducer(state: BoardState, action: BoardAction): BoardState {
  switch (action.type) {
    case "load":
      return action.state;
    case "story":
      if (action.text === state.story) return state;
      // Every downstream decision was made from the previous manuscript.
      // Keeping any of it would quietly mix two different stories.
      return { ...emptyBoard(state.style, state.format, state.layout),
               story: action.text, chapterTitle: state.chapterTitle, pageCount: state.pageCount };
    case "title":
      return { ...state, chapterTitle: action.text };
    case "analysed":
      return { ...state, storyBook: action.book, chapterId: action.chapterId, doc: action.doc, divided: true,
               cast: action.cast, leadId: null, sheets: {}, sheetRegens: {}, castApproved: false, recast: [],
               pages: [], overflow: [], coverage: action.coverage, projectId: action.projectId,
               ...(action.story !== undefined ? { story: action.story } : {}),
               ...(action.title !== undefined ? { chapterTitle: action.title } : {}) };
    case "book": {
      // The engine settled the cast (a narrator merged, a role made a
      // person): people already here keep their edits and sheets.
      if (!action.cast) return { ...state, storyBook: action.book };
      const known = new Map((state.cast?.people ?? []).map((p) => [p.id, p]));
      const people = action.cast.people.map((p) => {
        const old = p.id ? known.get(p.id) : undefined;
        return old ? { ...p, description: old.description, outfit: old.outfit, recurring: old.recurring } : p;
      });
      return { ...state, storyBook: action.book, cast: withLead({ ...action.cast, people }, state.leadId) };
    }
    case "replanned":
      return { ...state, doc: action.doc, pages: [], overflow: [],
               ...(action.coverage !== undefined ? { coverage: action.coverage } : {}) };
    case "person": {
      if (!state.cast) return state;
      const cast = { ...state.cast, people: state.cast.people.map((p) =>
        (p.id === action.id ? { ...p, ...action.patch } : p)) };
      const storyBook = state.storyBook && (action.patch.description !== undefined || action.patch.name !== undefined)
        ? { ...state.storyBook, characters: state.storyBook.characters.map((c) => (c.id === action.id
            ? { ...c, ...(action.patch.description !== undefined ? { visual: action.patch.description } : {}),
                ...(action.patch.name !== undefined ? { name: action.patch.name } : {}) } : c)) }
        : state.storyBook;
      // A rename reaches the panels too, where the name is in the prompt.
      const doc = action.patch.name !== undefined ? { ...state.doc, panels: state.doc.panels.map((p) => {
        const old = state.cast?.people.find((x) => x.id === action.id)?.name;
        if (!old || !(p.cast ?? []).some((a) => a.character_id === action.id)) return p;
        const name = action.patch.name!;
        return { ...p, cast: p.cast!.map((a) => (a.character_id === action.id ? { ...a, name } : a)),
                 characters: (p.characters ?? []).map((n) => (n === old ? name : n)),
                 focus: p.focus === old ? name : p.focus,
                 positions: Object.fromEntries(Object.entries(p.positions ?? {}).map(([k, v]) => [k === old ? name : k, v])),
                 dialogue: (p.dialogue ?? []).map((d) => (d.speaker === old ? { ...d, speaker: name } : d)) };
      }) } : state.doc;
      return { ...state, cast, storyBook, doc };
    }
    case "lead":
      return { ...state, leadId: action.id, cast: state.cast && withLead(state.cast, action.id) };
    case "sheet": {
      const sheets = { ...state.sheets };
      if (action.image) sheets[action.id] = action.image; else delete sheets[action.id];
      const sheetRegens = action.regenerated
        ? { ...state.sheetRegens, [action.id]: (state.sheetRegens[action.id] ?? 0) + 1 } : state.sheetRegens;
      // A sheet changed after approval is a recast: that person's panels are
      // drawn again once the cast is approved again, and no others.
      const recast = state.castApproved || state.recast.length
        ? Array.from(new Set([...state.recast, action.id])) : state.recast;
      return { ...state, sheets, sheetRegens, recast,
               castApproved: state.castApproved && recast.length === 0 };
    }
    case "approve": {
      const ids = state.recast.flatMap((id) => panelsWith(state.doc, id));
      return { ...state, castApproved: true, recast: [], doc: retire(state.doc, ids),
               ...(ids.length ? { pages: [], overflow: [] } : {}) };
    }
    case "doc":
      return { ...state, doc: action.update(state.doc) };
    case "pages":
      return { ...state, pages: action.pages, overflow: action.overflow };
    case "style":
      // Everything drawn was drawn in the old style; the reading stays.
      return { ...state, style: action.style, sheets: {}, castApproved: false, recast: [],
               doc: { ...state.doc, art: Object.fromEntries(Object.entries(state.doc.art).map(([k, a]) =>
                 [k, { ...a, drawn: null, variants: a.drawn && !a.variants.includes(a.drawn) ? [...a.variants, a.drawn] : a.variants }])) },
               pages: [], overflow: [] };
    case "format":
      return { ...state, format: action.format };
    case "pageCount":
      return { ...state, pageCount: Math.max(1, Math.min(200, Math.round(action.pages) || DEFAULT_PAGES)) };
    case "layout":
      return { ...state, layout: action.layout, pages: [], overflow: [] };
    case "reset":
      return emptyBoard(state.style, state.format, state.layout);
  }
}

/** The people who get a cast sheet: everyone recurring, the lead first. */
export function sheetPeople(state: Pick<BoardState, "cast">): CastMember[] {
  return (state.cast?.people ?? []).filter((p) => p.id && (p.recurring || p.tier === 1));
}

/** Recurring people still without a sheet; the cast cannot be approved until none are. */
export function missingSheets(state: Pick<BoardState, "cast" | "sheets">): CastMember[] {
  return sheetPeople(state).filter((p) => !state.sheets[p.id!]);
}

/** Where each panel goes: its page, cell and drawn shape. */
export function placesOf(panels: Panel[], format: BookFormat, layout: string): PanelPlace[] {
  return layoutBoard(panels, format, layout);
}

/**
 * The panels still to draw, a page at a time: the first page with anything
 * missing, in reading order. Page one first, so the comic is seen early.
 */
export function nextPage(doc: BoardDoc, places: PanelPlace[]): number[] {
  const waiting = doc.panels.map((p, i) => (artOf(doc, p.id).drawn ? -1 : i)).filter((i) => i >= 0);
  if (!waiting.length) return [];
  const page = Math.min(...waiting.map((i) => places[i]?.page ?? 0));
  return waiting.filter((i) => (places[i]?.page ?? 0) === page);
}

/** Pages whose every panel is drawn, in order. */
export function finishedPages(doc: BoardDoc, places: PanelPlace[]): number[] {
  const pages = Array.from(new Set(places.map((p) => p.page))).sort((a, b) => a - b);
  return pages.filter((page) => doc.panels.every((p, i) => places[i]?.page !== page || artOf(doc, p.id).drawn));
}

/**
 * What of the chapter the board carries now, counted the way the engine
 * counts it: a sentence a panel draws or letters, a line a balloon holds.
 * Recounted after every edit; the number shown used to be a hard-coded 100.
 */
export function coverageOf(panels: Panel[], sentences: string[] | undefined, cap = 20): Coverage | null {
  if (!sentences) return null;
  const plain = (t: string) => t.replace(/[^\p{L}\p{N}]+/gu, " ").trim().toLowerCase();
  const shown = new Set<string>();
  let lettered = " ";
  for (const p of panels) {
    for (const s of p.covers ?? []) shown.add(plain(s));
    if (p.moment && !(p.folded ?? []).includes(p.moment) && (p.covers ?? []).includes(p.moment)) shown.add(plain(p.moment));
    lettered += [plain(p.caption ?? ""), ...(p.dialogue ?? []).map((d) => plain(d.text))].join(" ") + " | ";
  }
  const missing = sentences.filter((s) => !shown.has(plain(s)) && !lettered.includes(` ${plain(s)} `));
  const covered = sentences.length - missing.length;
  return { covered, total: sentences.length,
           percent: sentences.length ? Math.round((100 * covered) / sentences.length) : 100,
           missing: missing.slice(0, cap), missing_total: missing.length, sentences };
}

/**
 * How long the rest of the board will take, in minutes, from what it will
 * actually draw: the cast sheets not drawn yet and every panel without a
 * picture. Rooms are not counted -- a director's panel never draws one -- and
 * the sheets are, which the old figure left out.
 */
export function estimateMinutes(sheetsLeft: number, panelsLeft: number, px: number,
                                measuredPanelMs?: number): { low: number; high: number } {
  const perSheet = px >= 768 ? 60 : 36;
  const perPanel = measuredPanelMs ? measuredPanelMs / 1000 : px >= 768 ? 170 : 90;
  const low = sheetsLeft * perSheet + panelsLeft * perPanel;
  return { low: Math.ceil(low / 60), high: Math.ceil((low * 1.6) / 60) };
}

/** Where a job is, for the progress line. */
export interface RunState {
  stage: "idle" | "analysing" | "sheets" | "drawing" | "composing" | "variants" | "repairing";
  prog: EngineProgress | null;
  jobId: string | null;
  outcome: string | null;
  /** The panel id, or `sheet:<character id>`, being drawn now. */
  drawingId: string | null;
  batch: { k: number; n: number; perPanelMs: number; startedAt: number; page?: number } | null;
}

export const idleRun: RunState = { stage: "idle", prog: null, jobId: null, outcome: null, drawingId: null, batch: null };

export type RunAction =
  | { type: "start"; stage: RunState["stage"]; jobId?: string | null }
  | { type: "job"; jobId: string | null; drawingId?: string | null }
  | { type: "progress"; prog: EngineProgress | null }
  | { type: "batch"; batch: RunState["batch"] }
  | { type: "end"; outcome?: string | null }
  | { type: "outcome"; outcome: string | null };

export function runReducer(state: RunState, action: RunAction): RunState {
  switch (action.type) {
    case "start":
      return { ...state, stage: action.stage, jobId: action.jobId ?? null, outcome: null, prog: null };
    case "job":
      return { ...state, jobId: action.jobId,
               ...(action.drawingId !== undefined ? { drawingId: action.drawingId } : {}) };
    case "progress":
      return { ...state, prog: action.prog };
    case "batch":
      return { ...state, batch: action.batch };
    case "end":
      return { ...idleRun, outcome: action.outcome === undefined ? state.outcome : action.outcome };
    case "outcome":
      return { ...state, outcome: action.outcome };
  }
}
