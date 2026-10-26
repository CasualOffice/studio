import type { Panel } from "./types";
import type { TextOverflow } from "./api";

/**
 * A board's panels and the work attached to them, keyed by identity.
 *
 * The board used to hold five parallel arrays -- panels, drawn, redraws,
 * notes, variants -- lined up by position. Position is not identity. Nothing
 * could move, and the moment anything did, every picture and every hand-typed
 * note attached itself to a different panel: reordering two panels swapped
 * their art, inserting one at the top shifted every note down by one, and
 * deleting one moved the whole board up. The arrays stayed the right length,
 * so nothing was detectably wrong until someone read the pages.
 *
 * So the panels carry ids and the work is stored against them. The board still
 * renders from arrays -- they are derived, and every read site is unchanged --
 * but nothing is written by position any more.
 */

/** What has been made for one panel. */
export interface PanelArt {
  /** Vault id of the current picture, or null if it has not been drawn. */
  drawn: string | null;
  /** How many times it has been drawn again, so a retry gets a new seed. */
  redraws: number;
  /** A written correction, kept with the panel and applied to every redraw. */
  note: string;
  /** Pictures this panel used to have, oldest first. */
  variants: string[];
}

export const EMPTY_ART: PanelArt = {
  drawn: null, redraws: 0, note: "", variants: [],
};

/** A panel that has been given an identity. */
export type IdentifiedPanel = Panel & { id: string };

export interface BoardDoc {
  panels: IdentifiedPanel[];
  art: Record<string, PanelArt>;
}

/**
 * A fresh identity.
 *
 * `crypto.randomUUID` is present in the Tauri webview and in jsdom. The
 * fallback is not for security -- nothing here is a secret -- it is so a panel
 * can still be given an id in an environment that lacks it, rather than the
 * board failing to load.
 */
let fallbackCounter = 0;
export function newPanelId(): string {
  const c = globalThis.crypto;
  if (c && typeof c.randomUUID === "function") return c.randomUUID();
  fallbackCounter += 1;
  return `p-${fallbackCounter}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Give every panel an id, keeping the ones that already have one.
 *
 * Idempotent: running it twice must not renumber a board, or every reopen
 * would detach the work from the panels it belongs to. Duplicate ids are
 * replaced rather than tolerated, because two panels sharing an id share
 * their art.
 */
export function identify(panels: Panel[]): IdentifiedPanel[] {
  const seen = new Set<string>();
  return panels.map((p) => {
    const had = (p as Partial<IdentifiedPanel>).id;
    const id = had && !seen.has(had) ? had : newPanelId();
    seen.add(id);
    return { ...p, id };
  });
}

/**
 * Build a document from what a v1 draft held.
 *
 * The parallel arrays are read by position exactly once -- here, where their
 * positions are still the truth -- and never again.
 */
export function fromArrays(
  panels: Panel[] | null,
  drawn: (string | null)[] = [],
  redraws: number[] = [],
  notes: Record<number, string> = {},
  variants: Record<number, string[]> = {},
): BoardDoc {
  const identified = identify(panels ?? []);
  const art: Record<string, PanelArt> = {};
  identified.forEach((panel, i) => {
    art[panel.id] = {
      drawn: drawn[i] ?? null,
      redraws: redraws[i] ?? 0,
      note: notes[i] ?? "",
      variants: variants[i] ?? [],
    };
  });
  return { panels: identified, art };
}

/**
 * A new panel plan for the same chapter, keeping the work on panels it shares.
 *
 * Continuing a stopped analysis re-plans the chapter, but the passages that
 * were already analysed come back with the same beats. A panel is the same
 * panel when it is the same frame of the same beat; it keeps its identity and
 * so its picture, notes and variants. Panels of newly analysed passages start
 * empty, as they must -- their briefs did not exist before.
 *
 * A panel that still shows the same moment keeps its brief: one a writer or
 * the user wrote would otherwise be replaced by the rules' plainer one every
 * time the format or the page count changed.
 */
export function replan(doc: BoardDoc, panels: Panel[]): BoardDoc {
  const key = (list: Panel[]) => {
    const seen: Record<string, number> = {};
    return list.map((p) => {
      const beat = p.beat_id ?? "";
      seen[beat] = (seen[beat] ?? 0) + 1;
      return beat ? `${beat}#${seen[beat]}` : null;
    });
  };
  const before = new Map<string, IdentifiedPanel>();
  key(doc.panels).forEach((k, i) => { if (k) before.set(k, doc.panels[i]); });
  const next = identify(key(panels).map((k, i) => {
    const old = k ? before.get(k) : undefined;
    if (!old) return panels[i];
    const same = (old.moment ?? "") === (panels[i].moment ?? "") && (old.description ?? "").trim();
    return { ...panels[i], id: old.id, ...(same ? { description: old.description } : {}) };
  }));
  const art: Record<string, PanelArt> = {};
  for (const p of next) if (doc.art[p.id]) art[p.id] = doc.art[p.id];
  return { panels: next, art };
}

/** The art for one panel, never undefined. */
export function artOf(doc: BoardDoc, id: string): PanelArt {
  return doc.art[id] ?? EMPTY_ART;
}

/** The arrays the board renders from, rebuilt from identity. */
export function toArrays(doc: BoardDoc): {
  drawn: (string | null)[];
  redraws: number[];
  notes: Record<number, string>;
  variants: Record<number, string[]>;
} {
  const drawn: (string | null)[] = [];
  const redraws: number[] = [];
  const notes: Record<number, string> = {};
  const variants: Record<number, string[]> = {};
  doc.panels.forEach((panel, i) => {
    const art = artOf(doc, panel.id);
    drawn.push(art.drawn);
    redraws.push(art.redraws);
    if (art.note) notes[i] = art.note;
    if (art.variants.length) variants[i] = art.variants;
  });
  return { drawn, redraws, notes, variants };
}

/** Change one panel's art, leaving every other panel alone. */
export function setArt(
  doc: BoardDoc, id: string, patch: Partial<PanelArt>,
): BoardDoc {
  if (!doc.panels.some((p) => p.id === id)) return doc;
  return { ...doc, art: { ...doc.art, [id]: { ...artOf(doc, id), ...patch } } };
}

/**
 * Which fields of a panel, when changed, make its picture out of date.
 *
 * A picture is a drawing of an action, in a place, of these people. Change any
 * of those and the picture is of something else. Change the caption and the
 * picture is still correct -- it is the lettering that has to be set again,
 * and re-rendering a panel to fix a comma costs minutes and a new face.
 *
 * `description` is the visual brief the prompt is built from, so changing it
 * is changing the picture.
 */
const ART_FIELDS = [
  "shot", "subject", "action", "setting", "character_in_frame",
  "characters", "place", "place_key", "description", "cast", "mood", "tension", "location_state",
] as const;

/** What a change to one panel makes out of date. */
export interface Staleness {
  /** The picture no longer matches the brief. */
  art: boolean;
  /** The worked-up scene was written from fields that have since changed. */
  description: boolean;
  /** The composed pages are no longer current. Any change does this. */
  pages: boolean;
}

/**
 * What changing these fields invalidates, without changing anything.
 *
 * Separate from `edit` so the interface can say what a change will cost before
 * it is made, and so the rule is one testable function rather than a condition
 * repeated at every call site.
 */
export function staleness(patch: Partial<Panel>): Staleness {
  const touched = Object.keys(patch) as (keyof Panel)[];
  const art = touched.some((k) => (ART_FIELDS as readonly string[]).includes(k));
  return {
    art,
    // The brief is the director's, written once from the analysis, and then
    // the user's to edit; it is shown and edited directly, so no other edit
    // throws it away. Clearing it fell back to the raw source sentence.
    description: false,
    pages: touched.length > 0,
  };
}

/**
 * Apply an edit, and say what it invalidated.
 *
 * The picture is not deleted. It goes to the panel's history, where it is one
 * click from coming back -- the board used to set `drawn[i] = null` on any
 * edit, which left the picture in the vault with nothing pointing at it.
 */
export function edit(
  doc: BoardDoc, id: string, patch: Partial<Panel>,
): { doc: BoardDoc; stale: Staleness } {
  const stale = staleness(patch);
  const panels = doc.panels.map(
    (p) => (p.id === id
      ? { ...p, ...patch, ...(stale.description ? { description: "" } : {}),
          ...((patch.characters !== undefined || patch.character_in_frame === false) && patch.cast === undefined
            ? { cast: p.cast?.filter((a) => patch.character_in_frame !== false &&
                (patch.characters ?? p.characters ?? []).includes(a.name)) } : {}) }
      : p));
  let art = doc.art;
  const current = artOf(doc, id);
  if (stale.art && current.drawn) {
    art = {
      ...art,
      [id]: {
        ...current,
        drawn: null,
        variants: current.variants.includes(current.drawn)
          ? current.variants
          : [...current.variants, current.drawn],
      },
    };
  }
  return { doc: { ...doc, panels, art }, stale };
}

/** Put a panel in at `at`, pushing the rest down. Nothing else moves. */
export function insert(doc: BoardDoc, at: number, panel: Panel): BoardDoc {
  const made: IdentifiedPanel = { ...panel, id: newPanelId() };
  const panels = [...doc.panels];
  panels.splice(Math.max(0, Math.min(at, panels.length)), 0, made);
  return { panels, art: { ...doc.art, [made.id]: { ...EMPTY_ART } } };
}

/**
 * Take a panel out.
 *
 * Its art goes with it, which is why this returns what was removed: the
 * pictures stay in the vault and the caller may want to say so.
 */
export function remove(
  doc: BoardDoc, id: string,
): { doc: BoardDoc; removed: { panel: IdentifiedPanel; art: PanelArt } | null } {
  const panel = doc.panels.find((p) => p.id === id);
  if (!panel) return { doc, removed: null };
  const art = { ...doc.art };
  const was = artOf(doc, id);
  delete art[id];
  return {
    doc: { panels: doc.panels.filter((p) => p.id !== id), art },
    removed: { panel, art: was },
  };
}

/** Move a panel to a new position. Every panel keeps its own work. */
export function move(doc: BoardDoc, id: string, to: number): BoardDoc {
  const from = doc.panels.findIndex((p) => p.id === id);
  if (from < 0) return doc;
  const panels = [...doc.panels];
  const [panel] = panels.splice(from, 1);
  panels.splice(Math.max(0, Math.min(to, panels.length)), 0, panel);
  return { ...doc, panels };
}

/**
 * Split one panel into two.
 *
 * The first keeps the panel's identity and its picture; the second is new and
 * undrawn. Which half the existing art belongs to is a judgement about the
 * picture, and the board already has a redraw for making it -- guessing here
 * would attach a drawing to a beat it is not of.
 */
export function split(
  doc: BoardDoc, id: string, second: Partial<Panel> = {},
): BoardDoc {
  const at = doc.panels.findIndex((p) => p.id === id);
  if (at < 0) return doc;
  const from = doc.panels[at];
  const made: IdentifiedPanel = {
    ...from, ...second, id: newPanelId(),
    // The source quote belongs to whichever half actually quotes it, and this
    // cannot know. Left on the first half rather than duplicated, because a
    // quote credited twice is how coverage came to report a board as complete
    // that had dropped three sentences.
    source: second.source ?? "",
  };
  const panels = [...doc.panels];
  panels.splice(at + 1, 0, made);
  return { panels, art: { ...doc.art, [made.id]: { ...EMPTY_ART } } };
}

/**
 * Join two adjacent panels into the first.
 *
 * The second's text is carried over and its picture is kept in the survivor's
 * history rather than discarded -- it was drawn, it may be the better of the
 * two, and after a merge there is no panel left to reach it from.
 */
export function merge(doc: BoardDoc, firstId: string): BoardDoc {
  const at = doc.panels.findIndex((p) => p.id === firstId);
  if (at < 0 || at + 1 >= doc.panels.length) return doc;
  const a = doc.panels[at];
  const b = doc.panels[at + 1];
  const artA = artOf(doc, a.id);
  const artB = artOf(doc, b.id);
  const joined: IdentifiedPanel = {
    ...a,
    action: [a.action, b.action].filter(Boolean).join(", "),
    caption: [a.caption, b.caption].filter(Boolean).join(" "),
    dialogue: [...(a.dialogue ?? []), ...(b.dialogue ?? [])],
    characters: Array.from(
      new Set([...(a.characters ?? []), ...(b.characters ?? [])])),
    source: [a.source, b.source].filter(Boolean).join(" "),
    // Two beats in one panel is not the scene the description was written
    // for, and the merged action is what a new one would be written from.
    description: "",
  };
  const art = { ...doc.art };
  delete art[b.id];
  art[a.id] = {
    ...artA,
    note: [artA.note, artB.note].filter(Boolean).join(" "),
    variants: [...artA.variants, ...artB.variants,
               ...(artB.drawn ? [artB.drawn] : [])]
      .filter((v, i, all) => v !== artA.drawn && all.indexOf(v) === i),
  };
  const panels = [...doc.panels];
  panels.splice(at, 2, joined);
  return { panels, art };
}

/**
 * Drop art belonging to panels that no longer exist.
 *
 * Every operation above already keeps the two halves in step. This is for what
 * arrives from disk, where a draft written by a build with a bug in it, or
 * hand-edited, can carry entries for panels that are gone.
 */
export function prune(doc: BoardDoc): BoardDoc {
  const live = new Set(doc.panels.map((p) => p.id));
  const art: Record<string, PanelArt> = {};
  for (const [id, value] of Object.entries(doc.art)) {
    if (live.has(id)) art[id] = value;
  }
  return { ...doc, art };
}

/**
 * Re-point an overflow report at the panels it is about.
 *
 * The engine counts the panels it was given, which are the drawn ones in the
 * order they were sent. Panels move afterwards, and a report still pointing at
 * position 3 would name whichever panel is there now.
 */
export function anchorOverflow(
  overflow: TextOverflow[], sentIds: string[], doc: BoardDoc,
): (TextOverflow & { panelId: string })[] {
  return overflow.flatMap((o) => {
    const panelId = sentIds[o.panel];
    if (!panelId) return [];
    const at = doc.panels.findIndex((p) => p.id === panelId);
    if (at < 0) return [];
    return [{ ...o, panel: at, panelId }];
  });
}
