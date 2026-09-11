import type { Panel } from "./types";

/**
 * How a picture board turns panels into prompts.
 *
 * Pure, and separate from the component, because this is where the board is
 * won or lost. A panel prompt that quietly drops the subject, or inserts the
 * protagonist into a shot she is not in, produces a picture that looks fine
 * on its own and is wrong in the board -- the kind of mistake that is obvious
 * in a test and invisible in a preview.
 */

/** Style is a whole-board decision: a board whose style drifts is not a board. */
export const STYLES: { id: string; label: string; words: string }[] = [
  { id: "anime", label: "Anime",
    words: "flat cel-shaded anime illustration, clean linework, muted palette" },
  { id: "photo", label: "Photographic",
    words: "photographic, natural skin texture, available light, 35mm, shallow depth of field" },
  { id: "ink", label: "Ink and wash",
    words: "black ink illustration with grey wash, visible brushwork, high contrast" },
  { id: "paint", label: "Painted",
    words: "digital painting, visible brush strokes, soft edges, muted colour" },
];

export function styleWords(id: string): string {
  return STYLES.find((s) => s.id === id)?.words ?? "";
}

/**
 * The prompt for one panel.
 *
 * Order is style, then what is in frame, then what it is doing, then where.
 * The character's description is included only when they are actually in the
 * panel: a close-up of a running tap, handed the protagonist's description,
 * draws her running instead of the tap.
 */
export function panelPrompt(
  panel: Panel, style: string, character: string
): string {
  const who = panel.character_in_frame
    ? [character.trim(), panel.subject.trim()].filter(Boolean).join(", ")
    : panel.subject.trim();
  // A worked-up scene already carries the setting, the surface and the light,
  // so it replaces the terse fields rather than being appended to them --
  // otherwise the panel is described twice, once thinly and once properly,
  // and the model has to reconcile them.
  const described = (panel.description ?? "").trim();
  const body = described
    ? [who, described]
    : [who, panel.action, panel.setting];
  return [styleWords(style), `${panel.shot} shot`, ...body]
    .map((b) => b.trim())
    .filter(Boolean)
    .join(". ") + ".";
}

/** The prompt for the character sheet every panel is drawn against. */
export function sheetPrompt(style: string, character: string): string {
  return `${styleWords(style)}. Character reference sheet, full body, neutral `
       + `pose, plain background. ${character.trim()}.`;
}

/**
 * Seed for a panel, moved on by each redraw.
 *
 * Redrawing with the same seed reproduces the picture that was just rejected,
 * which reads as the button not working.
 */
export function panelSeed(index: number, redraws: number): number {
  return 7 + index + redraws * 1000;
}

/** The prompt for a scene's location sheet: the room, empty. */
export function placePrompt(style: string, place: string): string {
  return `${styleWords(style)}. Empty interior, no people. ${place.trim()}.`;
}

/**
 * Which reference images a panel is drawn against.
 *
 * The character sheet holds the person; the location sheet holds the room.
 * Verified: passing both keeps consecutive panels in the same room rather
 * than a similar one. A panel the character is not in gets the room only --
 * handing it her sheet is what drew her into shots she does not appear in.
 */
export function panelReferences(
  panel: Panel, sheet: string | null, place: string | null
): string[] {
  const refs: string[] = [];
  if (panel.character_in_frame && sheet) refs.push(sheet);
  if (place) refs.push(place);
  return refs.length ? refs : (sheet ? [sheet] : []);
}
