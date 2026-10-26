import type { Actor, Panel, VaultItem } from "./types";
import { sizeClass } from "./layout";

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
  medium: "the setting behind the figure, kept simple",
  // Not "plain ground, no scenery": the same prompt names the setting, and a
  // close-up told both drew one or the other at random.
  "close-up": "the setting only as a soft, out-of-focus background",
};

/** How a shot is said to an image model: distance, then what it holds. */
const SHOT_WORDS: Record<string, string> = {
  EWS: "extreme wide establishing shot, the whole place in view, any people small within it",
  WS: "wide shot, full figures head to toe with their surroundings",
  MS: "medium shot, framed from the waist up",
  MCU: "medium close-up, head and shoulders",
  CU: "close-up on the face, filling the frame",
  ECU: "extreme close-up on a single detail of the face or hands",
  INSERT: "insert shot, a close view of one object or detail",
  wide: "wide shot", medium: "medium shot", "close-up": "close-up",
};
// Said as where the camera physically is: image models follow a described
// position far better than an angle's name.
const ANGLE_WORDS: Record<string, string> = {
  eye: "camera at eye level",
  high: "high-angle shot, camera above the subject looking down at them, they look small",
  low: "low-angle shot, camera near the ground looking up at the subject, they tower over the viewer",
  bird: "bird's-eye view, camera directly overhead looking straight down",
  worm: "worm's-eye view, camera on the ground looking steeply up",
  dutch: "dutch angle, the whole frame tilted, horizon on a diagonal",
};

/** The three layout classes the page compositor knows, from any shot. A
 *  large panel takes a whole tier, as an establishing shot or a Peak should. */
export function shotClass(panel: Pick<Panel, "shot" | "weight">): "wide" | "medium" | "close-up" {
  return sizeClass(panel);
}

/** The camera, said first: how far, from where, composed how. */
export function cameraWords(panel: Panel): string[] {
  const words = [SHOT_WORDS[panel.shot] ?? `${panel.shot} shot`];
  if (panel.angle && ANGLE_WORDS[panel.angle]) words.push(ANGLE_WORDS[panel.angle]);
  const cast = panel.cast ?? [];
  const focus = panel.focus || cast[0]?.name || "";
  const other = cast.find((a) => a.name !== focus)?.name;
  if (panel.framing === "ots" && other) words.push(`over-the-shoulder shot from behind ${other}, looking at ${focus}`);
  else if (panel.framing === "pov") words.push("first-person point of view; the viewer is the narrator, seen at most as their own hands");
  else if (panel.framing === "two" && cast.length >= 2) words.push(`two-shot of ${cast.map((a) => a.name).join(" and ")}`);
  else if (panel.framing === "insert" && !cast.length) words.push("no people in frame");
  for (const [name, where] of Object.entries(panel.positions ?? {})) {
    if (!cast.some((a) => a.name === name)) continue;
    const facing = panel.facing?.[name];
    words.push(`${name} at the ${where} of the frame${facing ? `, facing ${facing}` : ""}`);
  }
  return words;
}

export function shotBackground(shot: string): string {
  return SHOT_BACKGROUND[shot] ?? SHOT_BACKGROUND[shotClass({ shot: shot as Panel["shot"] })];
}

/**
 * One person as the drawer needs them: name, look, and what they wear when
 * it is not what their cast sheet shows.
 */
function actorWords(actor: Actor): string {
  const changed = (actor.appearance ?? "").trim()
    && (actor.appearance ?? "").trim() !== (actor.outfit ?? "").trim();
  return [actor.name, actor.visual,
    changed ? `now wearing ${actor.appearance} instead of the outfit on their reference`
      : actor.outfit && `wearing ${actor.outfit}, as on their reference`,
    actor.expression && `expression: ${actor.expression}`,
    actor.body_language && `body language: ${actor.body_language}`].filter(Boolean).join(", ");
}

/**
 * The prompt for one panel.
 *
 * The camera leads -- a model weighs what it reads first, and the shot was a
 * lone "medium shot" lost in the middle, which is why panels came out alike.
 * The brief comes next: who, doing what, where, how they look, written by the
 * director from the analysis. The source sentence is used only for a panel
 * with no brief (a board saved before briefs), because a sentence like "My
 * name is Mira Vale" is not something a drawer can draw.
 *
 * The references are full-body cast sheets in the chapter's outfit, so the
 * drawing is told to match them -- face and costume -- rather than to borrow
 * identity only, as it was when a sheet was a face study.
 */
export function panelPrompt(panel: Panel, style: StyleLock, note?: string): string {
  const cast = panel.cast ?? [];
  const reference = cast.length > 1
    ? `The reference shows ${cast.map((a) => a.name).join(", ")} from left to right, full figures`
    : "";
  const brief = (panel.description ?? "").trim() || panel.moment || panel.action;
  return [style.words, style.framing, ...cameraWords(panel), brief,
    panel.subject && panel.subject !== panel.focus && !cast.length && `Subject: ${panel.subject}`, reference,
    cast.length && "Each person keeps the face, hair, build and outfit of their reference, in this scene's pose",
    cast.length ? `Only visible people: ${cast.map((a) => a.name).join(", ")}` : "No people visible in frame",
    ...cast.map(actorWords), panel.setting && `Setting: ${panel.setting}`,
    panel.location_state && shotClass(panel) !== "close-up" && `Current location state: ${panel.location_state}`,
    panel.mood && `The feeling of the moment: ${panel.mood}`,
    shotBackground(panel.shot), note]
    .filter(Boolean).map((part) => String(part).trim().replace(/[.,;:]+$/, "")).filter(Boolean).join(". ") + ".";
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
 * stamped on the panel. Articles and possessives are dropped, so "the
 * kitchen", "kitchen" and "her kitchen" are one place.
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
 * A seed derived from a key -- a place, or a character's id -- so the same
 * thing is drawn from the same seed wherever it comes back, and two things
 * never share one by standing in the same position.
 */
export function placeSeed(key: string): number {
  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    hash = (hash * 31 + key.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % 100000;
}

/** A cast sheet's seed: the character's own, moved on by each regeneration. */
export function castSeed(characterId: string, regenerations = 0): number {
  return placeSeed(characterId) + regenerations * 1000;
}

/**
 * Which reference images a panel is drawn against: who is in it, and for a
 * frame with nobody in it, the room.
 *
 * Never a previous panel. Every panel is drawn on the edit route, and handed
 * a finished frame the drawer edits that frame: the same composition, the
 * same pose, a moment that does not move. With a one-reference model the
 * last frame also displaced the character sheet, so a chained run was a copy
 * of a copy with the identity anchor gone. Outfit, light and the state of the
 * room carry between panels in the brief -- `appearance` and `location_state`
 * -- where they describe the moment instead of dictating its picture.
 */
export function panelReferences(
  panel: Panel, sheet: string | null, place: string | null, budget = 3
): string[] {
  const refs: string[] = [];
  if (panel.character_in_frame && sheet) refs.push(sheet);
  // The room picture goes to the frames that are about the room.
  //
  // Handing it to a panel with somebody in it does not place them in the room;
  // it composites the room's most salient object onto them. Measured, same
  // seed, same prompt, one reference changed: with the kitchen sheet attached,
  // the wooden window frame was painted across Mira's face in one panel and
  // across Jonas's in the next -- a window worn like a mask. Without it, both
  // came back as ordinary panels, faces visible, and the kitchen still right:
  // green walls, the dark counter, the sugar tin, the letter face up. The room
  // survives because it is in the brief, which is built from the story's own
  // words, and prose cannot be composited onto a face.
  //
  // It is also cheaper. One reference instead of two took 406s against 504s,
  // and 353s against 540s, on the same two panels.
  //
  // So the room reference is for establishing frames -- the ones with nobody
  // in them, where the room IS the subject. A reference is an instruction, and
  // an instruction to draw the room competes with the instruction to draw the
  // person.
  // And still not for a close-up, for the older reason that has not changed:
  // at that distance the room is not in frame at all, so a picture of it pulls
  // the shot wider to fit the furniture in.
  if (place && !panel.character_in_frame && shotClass(panel) !== "close-up") {
    refs.push(place);
  }
  // Nothing, when there is nothing that belongs. This used to fall back to the
  // character sheet so the drawer always had an image to work from, which put
  // her in every panel she is not in whenever the room failed to draw -- the
  // one failure this function exists to prevent. A panel with no reference is
  // drawn from its prompt instead; the style words are in the prompt, so the
  // board still holds together.
  return refs.slice(0, Math.max(1, budget));
}

/** The shape a cast sheet is drawn at: a standing figure, head to toe. */
export const SHEET_ASPECT = 2 / 3;

/**
 * The prompt for one character's cast sheet: the whole figure, in the outfit
 * they wear in this chapter.
 *
 * It used to be a face study with "no full-body outfit", so nothing held the
 * costume and every panel dressed the person afresh. The sheet now carries
 * the clothes too, and panels are told to follow it; a costume change within
 * the chapter is said in the panel itself.
 */
export function castSheetPrompt(style: StyleLock, name: string, look: string, outfit: string): string {
  return [style.words,
    "Character reference sheet of one person: full body, head to toe, standing, front view, "
      + "neutral pose and expression, arms relaxed, plain light background",
    `${name}${look.trim() ? `: ${look.trim()}` : ""}`,
    outfit.trim() ? `Wearing ${outfit.trim()}` : "Wearing one plain everyday outfit that suits them",
    "No other people, no scenery, no props, no lettering"]
    .map((part) => part.trim().replace(/[.,;:]+$/, "")).filter(Boolean).join(". ") + ".";
}

/**
 * One board's composed pages, split into the runs that made them.
 *
 * Composing a board does not replace the pages composed last time. Those are
 * sealed vault items and they stay -- which is right, because the previous
 * version of a comic is worth keeping -- but it means "every composed item in
 * this project" is not the current chapter. It is every chapter this board has
 * ever had.
 *
 * The library grouped them exactly that way and sorted them by page number, so
 * a board composed twice showed, exported and bound as page one old, page one
 * new, page two old, page two new: the comic twice over, interleaved, half of
 * it the version the user had already rejected, and nothing anywhere said so.
 *
 * `items` must already be in project order (page number, then time), which is
 * the order the library holds a project in. Runs come back oldest first, so
 * the last one is the current comic.
 *
 * A page composed before runs were recorded carries no id. Those are grouped
 * together under the empty id, which is unknown provenance rather than
 * membership -- they are not the current composition merely because nothing
 * contradicts it.
 *
 * The bound PDF is deliberately not a page. It is stamped `composed` like the
 * pages are and sorts after all of them, so it was counted as the comic's last
 * page, exported alongside them as a PNG, and handed to the next bind as a
 * page to bind -- a comic ending with a PDF of itself.
 */
export function compositions(items: VaultItem[]): {
  id: string; pages: VaultItem[];
}[] {
  const runs: { id: string; pages: VaultItem[] }[] = [];
  for (const item of items) {
    if (item.model !== "composed" || !item.mime.startsWith("image/")) continue;
    const id = item.composition ?? "";
    const run = runs.find((r) => r.id === id);
    if (run) run.pages.push(item);
    else runs.push({ id, pages: [item] });
  }
  return runs;
}

/** How a batch of panels ended, and the failure behind it when there was one. */
export type BatchEnd =
  | { end: "done" }
  | { end: "stopped" }
  | { end: "cancelled" }
  | { end: "engine"; error: unknown }
  | { end: "repeated"; error: unknown };

/** Two failures in a row is a pattern, not bad luck. */
export const BATCH_FAILURE_LIMIT = 2;

/**
 * Draw panels one after another, and know when to give up.
 *
 * The loop used to stop only for the user's own Stop. When the engine died
 * -- usually macOS killing it for memory -- every remaining panel started a
 * fresh worker, loaded the model again and died the same way, so a board of
 * twelve was twelve crashes and twelve red toasts. A dead engine ends the
 * batch at once; any other failure ends it on the second in a row.
 */
export async function drawBatch(
  indices: number[],
  drawOne: (index: number) => Promise<{ ok: true } | { ok: false; error: unknown }>,
  stopped: () => boolean,
  engineDied: (error: unknown) => boolean,
): Promise<BatchEnd> {
  let failures = 0;
  for (const i of indices) {
    if (stopped()) return { end: "stopped" };
    const outcome = await drawOne(i);
    if (outcome.ok) { failures = 0; continue; }
    if (stopped()) return { end: "cancelled" };
    if (engineDied(outcome.error)) return { end: "engine", error: outcome.error };
    if (++failures >= BATCH_FAILURE_LIMIT) return { end: "repeated", error: outcome.error };
  }
  return { end: "done" };
}
