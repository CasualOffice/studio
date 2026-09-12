import { api } from "./api";
import { loadPref, removePref } from "./prefs";
import type { Cast, Panel } from "./types";

export interface Coverage {
  covered: number;
  total: number;
  percent: number;
  missing: string[];
}

export interface BoardDraft {
  version: 1;
  story: string;
  cast: Cast | null;
  projectId: string | null;
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
  if (encrypted) return JSON.parse(encrypted) as Partial<BoardDraft>;

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
