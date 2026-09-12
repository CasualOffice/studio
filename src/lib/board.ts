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

export interface Style {
  id: string;
  /**
   * Bumped whenever `words` changes, so a board can tell that the preset it
   * was drawn under is no longer the preset of the same name.
   */
  version: number;
  label: string;
  words: string;
}

/**
 * Style is a whole-board decision: a board whose style drifts is not a board.
 *
 * These are the defaults for a *new* board. They are not what an existing
 * board is drawn from -- see `lockStyle`. Anime is at version 2 because its
 * words were replaced between v0.1.0 and v0.2.0 ("flat cel-shaded anime
 * illustration, clean linework, muted palette"), which is exactly the drift
 * the lock exists to stop: a board half-drawn under one build and finished
 * under the next had panels 1-6 and 7-12 asking for different things.
 *
 * Bump the version in the same commit that edits the words. A preset may be
 * improved freely -- that is what happened, and it was a real fix -- but it
 * must not reach back into work already drawn.
 */
export const STYLES: Style[] = [
  { id: "anime", version: 2, label: "Anime",
    words: "cel-shaded anime, clean confident linework, flat colour with hard "
         + "shadow shapes, screentone texture" },
  { id: "photo", version: 1, label: "Photographic",
    words: "photographic, natural skin texture, available light, 35mm, shallow depth of field" },
  { id: "ink", version: 1, label: "Ink and wash",
    words: "black ink illustration with grey wash, visible brushwork, high contrast" },
  { id: "paint", version: 1, label: "Painted",
    words: "digital painting, visible brush strokes, soft edges, muted colour" },
];

/**
 * What every panel is, before it is anything else.
 *
 * Nothing in these prompts used to say the picture was a panel, so each one
 * was drawn as a standalone illustration -- a well-composed portrait of the
 * moment, centred and complete, which is the opposite of what a panel does.
 * Panels crop, they hold a single beat, and they are read in sequence. The
 * finished pages looked like comics only because the compositor drew the
 * borders and set the captions afterwards; the pictures inside them had never
 * been asked to be panels at all.
 *
 * Not applied to the character sheet or the room reference. Those are
 * reference material, drawn flat and neutral on purpose, and framing them as
 * panels would put a border and a dramatic crop on the very thing every panel
 * is supposed to match against.
 *
 * Purely compositional, and deliberately free of format words. Measured at a
 * fixed seed: "a single comic book panel, sequential art" drew a heavy black
 * frame into the picture -- inside the frame the compositor then drew around
 * it -- and removing the negations did not stop it, because the words "panel"
 * and "comic book" imply the frame on their own. Saying nothing about panels
 * and asking only for the crop leaves the border to the compositor, which is
 * the only thing that knows how wide the gutters are.
 *
 * Style-neutral on purpose. "manga artwork" produced a slightly crisper result
 * but would fight the Photographic style, and the framing has to hold for all
 * four. Against no framing at all the difference is visible: flatter colour,
 * harder shadow shapes, cleaner edges. The compositional words are what make
 * the chosen style land.
 */
const PANEL_FRAMING =
  "cropped composition, full bleed artwork, edge to edge";

export function styleWords(id: string): string {
  return STYLES.find((s) => s.id === id)?.words ?? "";
}

export function styleLabel(id: string): string {
  return STYLES.find((s) => s.id === id)?.label ?? id;
}

/**
 * The exact words a board is drawn from, resolved once and kept with it.
 *
 * A board used to store only the preset id, in plaintext preferences, and
 * resolve it against whatever `STYLES` said at draw time. So the look -- the
 * one decision that has to hold for the length of a long project -- was the
 * one thing the board did not own. Improving a preset silently changed work
 * already in progress, and the anime words had already been replaced once
 * between two public tags. Chapter eleven could not be made to match
 * chapter one.
 *
 * `framing` is in here for the same reason and is the sharper case: it is
 * prepended to every panel prompt, it is not part of any style, and it did not
 * exist at all in v0.1.0.
 *
 * An unknown id resolves to the first style rather than to nothing. It used to
 * resolve to the empty string, which dropped the style words from every prompt
 * while the board went on looking like it had worked -- the failure the file's
 * own header calls invisible in a preview and obvious in a test. The caller is
 * expected to notice the substitution and say so; `isKnownStyle` is how.
 */
export interface StyleLock {
  id: string;
  version: number;
  words: string;
  framing: string;
}

export function isKnownStyle(id: string): boolean {
  return STYLES.some((s) => s.id === id);
}

export function lockStyle(id: string): StyleLock {
  const style = STYLES.find((s) => s.id === id) ?? STYLES[0];
  return {
    id: style.id,
    version: style.version,
    words: style.words,
    framing: PANEL_FRAMING,
  };
}

/**
 * A lock read back out of a draft, made safe to draw from.
 *
 * The draft is JSON, so every field here is whatever was on disk -- written by
 * an older build, a newer one, or by hand. A missing `framing` is the dangerous
 * one: `panelPrompt` trims each part, and trimming `undefined` throws, which
 * would take the whole Board tab down on load rather than degrading. Anything
 * unusable falls back to the named preset, and to the first preset if the name
 * has gone too.
 */
export function reviveLock(raw: unknown): StyleLock | null {
  if (!raw || typeof raw !== "object") return null;
  const lock = raw as Partial<StyleLock>;
  const words = typeof lock.words === "string" ? lock.words.trim() : "";
  if (!words) return null;
  const base = lockStyle(typeof lock.id === "string" ? lock.id : STYLES[0].id);
  return {
    id: typeof lock.id === "string" && lock.id ? lock.id : base.id,
    version: typeof lock.version === "number" ? lock.version : base.version,
    words,
    framing: typeof lock.framing === "string" ? lock.framing : base.framing,
  };
}

/**
 * Whether a stored lock still asks for what its preset now asks for.
 *
 * Not an error -- a board keeping its own words is the point -- but worth
 * telling someone starting chapter twelve that the Anime of chapter eleven is
 * not today's Anime.
 */
/** The id a look the user described for themselves carries. */
export const CUSTOM_STYLE = "custom";

/**
 * A look that is not on the list.
 *
 * The four presets are a starting point, not the range of what the models can
 * do, and someone who wants "1950s newspaper strip" or "soft pencil, no ink"
 * had no way to say so -- the picker was four words and nothing else, and the
 * only thing the tool ever said about any of them appeared after twenty
 * minutes of drawing. A described look is pinned exactly like a preset one, so
 * it survives a restart and holds for the length of the board.
 *
 * Version 0 marks it as belonging to nobody's preset, so `styleHasMoved` never
 * claims a described look has drifted: there is nothing for it to drift from.
 */
export function lockCustomStyle(words: string): StyleLock {
  return {
    id: CUSTOM_STYLE,
    version: 0,
    words: words.trim(),
    framing: PANEL_FRAMING,
  };
}

export function styleHasMoved(lock: StyleLock): boolean {
  if (lock.id === CUSTOM_STYLE) return false;
  const current = STYLES.find((s) => s.id === lock.id);
  if (!current) return true;
  return current.words !== lock.words || lock.framing !== PANEL_FRAMING;
}

/**
 * The prompt for one panel.
 *
 * Order is style, then what is in frame, then what it is doing, then where.
 * The character's description is included only when they are actually in the
 * panel: a close-up of a running tap, handed the protagonist's description,
 * draws her running instead of the tap.
 */
/**
 * How much of the place belongs in a panel, by shot.
 *
 * A comic page does not draw the room again in every frame, and this one was.
 * Every panel asked for the walls, the floor, the furniture and the light --
 * the enrichment stage is told to state all four -- so a close-up of a face
 * came back as a face in a fully furnished room, and the room was re-invented
 * from prose each time, which is also why it drifted. Two problems with one
 * cause: the place was being drawn twelve times instead of established once.
 *
 * How the form actually works: a wide shot opens a scene and carries the
 * location; the frames after it live inside the place the reader has already
 * been shown, so they carry less and less of it; a close-up carries almost
 * none, because at that distance there is nothing but the subject. Backgrounds
 * are the expensive part of a real page and are spent deliberately.
 *
 * Style-neutral on purpose, like the framing clause: the board has to hold for
 * the photographic style as well as the drawn ones, so this says how much
 * scene is in frame rather than naming a technique. Not measured against
 * fixed seeds the way PANEL_FRAMING was -- this is the composition of the
 * form, and the claim is about what a panel is, not about what one model does
 * with a phrase.
 */
const SHOT_BACKGROUND: Record<string, string> = {
  wide: "the place itself in frame, full depth",
  medium: "only what stands directly behind the figure, kept simple",
  "close-up": "plain ground behind the subject, no scenery",
};

export function shotBackground(shot: string): string {
  return SHOT_BACKGROUND[shot] ?? SHOT_BACKGROUND.medium;
}

export function panelPrompt(
  panel: Panel, style: StyleLock, character: string, note?: string
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
  // A note goes last, where it can correct what came before it. This is the
  // repair for a wrong frame: a sentence in English about that one panel,
  // rather than rolling the dice on a new seed and hoping. It is kept with
  // the panel, so a later redraw is still the panel you asked for.
  return [style.words, style.framing, `${panel.shot} shot`,
          shotBackground(panel.shot), ...body, (note ?? "").trim()]
    .map((b) => b.trim())
    .filter(Boolean)
    .join(". ") + ".";
}

/** The prompt for the character sheet every panel is drawn against. */
export function sheetPrompt(style: StyleLock, character: string): string {
  const cast = character.split("\n").map((line) => line.trim()).filter(Boolean);
  if (cast.length > 1) {
    return `${style.words}. Cast reference lineup, each named character `
      + `shown separately, full body, neutral poses, plain background, distinct `
      + `silhouettes and clothing. ${cast.join("; ")}.`;
  }
  return `${style.words}. Character reference sheet, full body, neutral `
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

/** Words that do not distinguish one room from another. */
const PLACE_NOISE = new Set([
  "the", "a", "an", "in", "at", "on", "of", "into", "inside", "outside",
  "her", "his", "their", "my", "our", "its", "this", "that",
]);

/**
 * Which place a panel is in, by identity rather than by scene number.
 *
 * Mirrors the engine's `_place_key`, and prefers the key the engine already
 * stamped on the panel. Rooms were keyed on the scene number the writer
 * assigned, which fails in both directions: a story the writer puts entirely
 * in scene 1 got one room however far it travelled, and the same kitchen in
 * scenes 1 and 5 got two separately invented kitchens. Articles and
 * possessives are dropped, so "the kitchen", "kitchen" and "her kitchen" are
 * one place.
 */
export function placeKey(panel: Panel): string {
  const stamped = (panel.place_key ?? "").trim();
  if (stamped) return stamped;
  const setting = (panel.setting ?? "").toLowerCase().replace(/[^a-z0-9 ]+/g, " ");
  const words = setting.split(/\s+/).filter((w) => w && !PLACE_NOISE.has(w));
  if (words.length > 0) return words.join(" ");
  const title = (panel.scene_title ?? "").trim().toLowerCase();
  return title || `scene ${panel.scene ?? 1}`;
}

/**
 * A seed for a place, derived from the place itself.
 *
 * It was `21 + scene`, so renumbering the scenes redrew every room, and two
 * scenes in one place drew it twice from different seeds. Derived from the key
 * instead, the same room is the same picture whenever the story comes back
 * to it.
 */
export function placeSeed(key: string): number {
  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    hash = (hash * 31 + key.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % 100000;
}

/** The prompt for a scene's location sheet: the room, empty. */
export function placePrompt(style: StyleLock, place: string): string {
  return `${style.words}. Empty interior, no people. ${place.trim()}.`;
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
  // No room for a close-up. At that distance the room is not in frame, and
  // handing the drawer a picture of it asks for two things at once: it pulls
  // the shot wider to fit the furniture in, which is the opposite of what a
  // close-up is for. The same reasoning as withholding the character sheet
  // from a panel she is not in -- a reference is an instruction, and an
  // instruction for something outside the frame fights the frame.
  if (place && panel.shot !== "close-up") refs.push(place);
  // Nothing, when there is nothing that belongs. This used to fall back to the
  // character sheet so the drawer always had an image to work from, which put
  // her in every panel she is not in whenever the room failed to draw -- the
  // one failure this function exists to prevent. A panel with no reference is
  // drawn from its prompt instead; the style words are in the prompt, so the
  // board still holds together.
  return refs;
}
