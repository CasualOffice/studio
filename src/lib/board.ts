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

/**
 * Whether the cast description already binds this subject.
 *
 * The cast reaches this file in the shape the board builds it in: people
 * separated by "; ", and a name separated from their description by ": ", as in
 * "Mira: young woman, short dark hair; Jonas: tall, grey overcoat". So the names
 * are what stands before the first colon of each person. A cast of one that was
 * typed rather than read out of the story has no colon at all, and then the
 * whole line is the name -- which only matters if the subject is that same
 * whole line, and in that case saying it twice is just as wrong.
 */
function castBinds(character: string, subject: string): boolean {
  const wanted = subject.toLowerCase().replace(/[.,;:]+$/, "").trim();
  if (!wanted) return false;
  return character.split(";").some((person) => {
    const name = person.split(":")[0].toLowerCase().replace(/[.,;:]+$/, "").trim();
    return name !== "" && name === wanted;
  });
}

/**
 * The shot, stated once, by the panel.
 *
 * The enrichment stage writes the shot back into the brief it returns: a real
 * run came back with "Mira stands in the narrow kitchen, wide shot. Pale green
 * walls, ..." for a panel whose prompt had already said "wide shot" two clauses
 * earlier. Said twice it is merely noise, but the enricher and the writer do not
 * always agree, and then one prompt asked for two distances at once. The panel
 * owns the shot -- it is what the shot background and the references are chosen
 * from -- so the echo comes back out of the brief.
 *
 * Only the two shapes the echo actually takes are removed: a clause tacked onto
 * the end of a sentence, and a sentence of its own opening the brief. A shot
 * named in the middle of a sentence ("a close-up shot of her hands") is left
 * alone, because cutting it out there would take the grammar with it and leave
 * "a of her hands".
 */
function stripShotEcho(described: string): string {
  return described
    .replace(/,\s*(?:wide|medium|full|close[\s-]?up)\s+shots?(?=\s*[.;]|$)/gi, "")
    .replace(/(^|[.;]\s*)(?:wide|medium|full|close[\s-]?up)\s+shots?\s*[.;]\s*/gi, "$1")
    .replace(/\s{2,}/g, " ")
    .trim();
}

export function panelPrompt(
  panel: Panel, style: StyleLock, character: string, note?: string
): string {
  const cast = character.trim();
  const subject = panel.subject.trim();
  // The subject used to be joined onto the cast description with ", ", which
  // welded the name onto the clothing: a panel of Mira read "... oversized grey
  // wool coat, dark green scarf, Mira", and the drawer took the name for one
  // more garment. A person the description already names is named once. A thing
  // -- "a kitchen tap" -- still has to be said, and gets a clause of its own
  // rather than the next slot in the wardrobe.
  const who: string[] = [];
  if (panel.character_in_frame && cast) who.push(cast);
  if (subject && !(panel.character_in_frame && castBinds(cast, subject))) {
    who.push(subject);
  }
  // A worked-up scene already carries the setting, the surface and the light,
  // so it replaces the terse fields rather than being appended to them --
  // otherwise the panel is described twice, once thinly and once properly,
  // and the model has to reconcile them.
  const described = stripShotEcho((panel.description ?? "").trim());
  const body = described
    ? [...who, described]
    : [...who, panel.action, panel.setting];
  // How much of the place is in frame is settled after the scene, not before
  // it. The enrichment stage lists the walls, the floor and the light in every
  // brief whatever the shot is, so a close-up used to say "plain ground behind
  // the subject, no scenery" and then list the room anyway, in the same prompt,
  // a clause later. The brief cannot be edited from here -- it is prose the
  // model wrote -- but the shot can have the last word on the distance, which
  // is the same reason a note goes after everything it corrects.
  //
  // A note goes last of all. This is the repair for a wrong frame: a sentence
  // in English about that one panel, rather than rolling the dice on a new seed
  // and hoping. It is kept with the panel, so a later redraw is still the panel
  // you asked for.
  return [style.words, style.framing, `${panel.shot} shot`,
          ...body, shotBackground(panel.shot), (note ?? "").trim()]
    // Trailing punctuation is dropped because the parts are joined with ". ".
    // A brief is whole sentences and ends in a full stop, so the join produced
    // "dim light from a corner lamp.. plain ground behind the subject" -- and a
    // part ending in a comma ran the next clause on as if it belonged to it.
    .map((b) => b.trim().replace(/[.,;:]+$/, "").trim())
    .filter(Boolean)
    .join(". ") + ".";
}

/** The prompt for the character sheet every panel is drawn against. */
export function sheetPrompt(style: StyleLock, character: string): string {
  const cast = character.split("\n").map((line) => line.trim()).filter(Boolean);
  if (cast.length > 1) {
    // The count, stated. Asked for a lineup of "each named character" the
    // drawer put three figures on a sheet for a cast of two, and that extra
    // person is then the reference every panel of the board is conditioned
    // against. Saying how many there are is the one instruction that fixes it.
    //
    // And no longer "named": the word invited a caption under each figure, and
    // a diffusion model cannot letter. A real run came back with "Referace
    // Mekwarird" and "L0naONomirs" printed on the sheet -- noise baked into the
    // one image every frame is drawn from. The names stay in the description,
    // where they bind identity; nothing asks for them to be written down.
    const count = ["one", "two", "three", "four"][cast.length - 1] ?? `${cast.length}`;
    return `${style.words}. Character reference sheet of ${count} people, `
      + `standing apart in a row, full body, neutral poses, plain background, `
      + `distinct silhouettes and clothing. ${cast.join("; ")}.`;
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

/**
 * The prompt for a place reference: the location, with nobody in it.
 *
 * It used to open "Empty interior", which decided the answer before the place
 * was named. A story that goes to a railway platform got a sealed shed with no
 * platform and no rails -- the iron roof it asked for, inside four walls it
 * never mentioned -- and a street or a field would have come back as a room
 * just as surely. Drawn from a real run: the story says "at the station the
 * roof was iron and very high", and the reference came back as a warehouse.
 *
 * Nothing here says indoors or out. The place says which it is, and the only
 * thing asserted is what a reference sheet is for: the location itself, with
 * nobody standing in it, so every panel of the scene can be drawn against the
 * same one.
 */
export function placePrompt(style: StyleLock, place: string): string {
  return `${style.words}. Establishing view of the location itself, `
    + `unoccupied, no people present. ${place.trim()}.`;
}

/**
 * How often a panel is re-anchored to the character sheet alone.
 *
 * Chaining each panel to the one before it carries continuity forward, and
 * carries error forward with it: every generation is a lossy copy, so by the
 * tenth panel the character is ten copies from the sheet that defined her.
 * The sheet is the fixed point and cannot drift, so the chain is broken at
 * intervals and re-anchored to it.
 */
export const CHAIN_LIMIT = 4;

/**
 * Which reference images a panel is drawn against.
 *
 * Three anchors, in falling order of what they cost to lose:
 *
 *   the character sheet, which holds who this is and never drifts
 *   the previous panel, which holds what the last moment looked like
 *   the location sheet, which holds the room
 *
 * The previous panel is the continuity that was missing: within a scene it
 * carries the time of day, the weather, the state of the clothes and where
 * the light is coming from -- none of which the sheet or the room knows, and
 * all of which reset between panels without it. It is passed only inside a
 * scene, because the panel before a scene change is the wrong room.
 *
 * Order matters beyond taste: most models in the catalogue take exactly one
 * reference, so this list is trimmed to `budget` and the first entry is the
 * one that survives. Identity wins, because a character who changes face
 * between panels is the failure everyone sees first.
 */
export function panelReferences(
  panel: Panel, sheet: string | null, place: string | null,
  previous: string | null = null, budget = 3
): string[] {
  const refs: string[] = [];
  if (panel.character_in_frame && sheet) refs.push(sheet);
  if (previous) refs.push(previous);
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
  return refs.slice(0, Math.max(1, budget));
}

/**
 * The panel to carry forward from, if any.
 *
 * Only within a scene, and only for a few panels before the chain is broken.
 * `drawn` holds one id per panel, null where nothing has been drawn yet, so a
 * redrawn panel in the middle of a batch is followed correctly rather than
 * skipped.
 */
export function previousPanel(
  index: number, panels: Panel[], drawn: (string | null)[]
): string | null {
  const scene = panels[index]?.scene ?? 1;
  let back = 0;
  for (let i = index - 1; i >= 0 && back < CHAIN_LIMIT; i--, back++) {
    if ((panels[i]?.scene ?? 1) !== scene) return null;
    if (drawn[i]) {
      // Re-anchor on the interval boundary: this panel is far enough from the
      // sheet that another copy of a copy is worth less than going back to it.
      return index % CHAIN_LIMIT === 0 ? null : drawn[i];
    }
  }
  return null;
}
