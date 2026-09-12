import { api } from "./api";
import { loadPref, removePref } from "./prefs";
import type { StyleLock } from "./board";
import type { Cast, Panel } from "./types";

export interface Coverage {
  covered: number;
  total: number;
  percent: number;
  /** The unanchored passages, truncated to twenty for display. */
  missing: string[];
  /**
   * How many passages are really unanchored.
   *
   * `missing` is capped at twenty, and the interface was reporting the length
   * of that cap as the number needing review -- so the boards that had lost
   * the most were the ones that under-reported it. Optional because a draft
   * saved by an older build will not carry it.
   */
  missing_total?: number;
}

export interface BoardDraft {
  version: 1;
  story: string;
  cast: Cast | null;
  projectId: string | null;
  /**
   * The exact style words this board is drawn from.
   *
   * The style used to live only in plaintext preferences as a preset id, which
   * made it the one whole-board creative decision the board did not own -- so
   * improving a preset reached back into work already drawn, and the anime
   * words had already been replaced once between two public tags. Optional
   * because a draft written before this carries no lock; the board takes one
   * from the remembered id on first load.
   */
  style: StyleLock | null;
  character: string;
  panels: Panel[] | null;
  sheet: string | null;
  drawn: (string | null)[];
  redraws: number[];
  notes: Record<number, string>;
  pages: string[];
  ownSheet: string[];
  placeSheets: Record<number, string>;
  coverage: Coverage | null;
}

const LEGACY_PREFS = [
  "boardStory", "boardCast", "boardProject", "boardNotes", "boardCharacter",
  "boardPanels", "boardSheet", "boardDrawn", "boardPlaces",
];

/** Move pre-vault Board data out of plaintext localStorage exactly once. */
export async function loadBoardDraft(): Promise<Partial<BoardDraft> | null> {
  const encrypted = await api.boardStateGet();
  if (encrypted) {
    // The migration has already run, but an earlier one may have left the
    // plaintext copies behind -- and a story sitting in localStorage is the
    // thing the encrypted draft exists to avoid. Clearing them here means it
    // is cleaned up on the next launch rather than never.
    for (const key of LEGACY_PREFS) removePref(key);
    return JSON.parse(encrypted) as Partial<BoardDraft>;
  }

  const story = loadPref("boardStory", "");
  const panels = loadPref<Panel[] | null>("boardPanels", null);
  if (!story && !panels) return null;

  const legacy: Partial<BoardDraft> = {
    version: 1,
    story,
    cast: loadPref<Cast | null>("boardCast", null),
    projectId: loadPref<string | null>("boardProject", null),
    character: loadPref("boardCharacter", ""),
    panels,
    sheet: loadPref<string | null>("boardSheet", null),
    drawn: loadPref<(string | null)[]>("boardDrawn", []),
    notes: loadPref<Record<number, string>>("boardNotes", {}),
    placeSheets: loadPref<Record<number, string>>("boardPlaces", {}),
  };
  await api.boardStateSet(JSON.stringify(legacy));
  LEGACY_PREFS.forEach(removePref);
  return legacy;
}

export async function migrateBoardDraft(): Promise<void> {
  await loadBoardDraft();
}
