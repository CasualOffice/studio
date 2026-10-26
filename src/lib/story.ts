import type { Panel } from "./types";
import type { Coverage } from "./boardDraft";

export interface StoryCharacter {
  id: string;
  name: string;
  aliases: string[];
  visual: string;
  voice: string;
}
export interface StoryActor {
  character_id: string;
  expression: string;
  body_language: string;
  appearance?: string;
}
export interface StoryDialogue {
  speaker_id: string | null;
  text: string;
  offscreen: boolean;
  start: number;
  end: number;
}
export interface StoryBeat {
  id: string;
  start: number;
  end: number;
  source: string;
  importance: number;
  tension: number;
  action: string;
  setting: string;
  location_id?: string;
  location_state?: string;
  mood: string;
  visible: StoryActor[];
  dialogue: StoryDialogue[];
}
export interface StoryScene {
  id: string;
  title: string;
  summary: string;
  start: number;
  end: number;
  beats: StoryBeat[];
  /** Left by a stopped or failed analysis; "Continue analysis" picks it up. */
  unanalysed?: boolean;
}
export interface StoryChapter {
  id: string;
  title: string;
  source: string;
  source_hash: string;
  scenes: StoryScene[];
  /** "voice": speakers and mood only, from the Audiobook's quick analysis. */
  analysis?: "voice";
}
export interface CharacterReference {
  /** "cast-sheet-v2": a full-body sheet in the chapter's outfit. The older
   *  face study ("identity-face-v1") carried no costume and is drawn again. */
  kind?: "identity-face-v1" | "cast-sheet-v2";
  image: string;
  style_words: string;
  visual: string;
  outfit?: string;
}
export interface StoryBook {
  references?: Record<string, CharacterReference>;
  version: 1;
  id: string;
  characters: StoryCharacter[];
  chapters: StoryChapter[];
}
export const SPEECH_MODEL_ID = "qwen3-tts-1.7b-custom-6bit";
export const KOKORO_MODEL_ID = "kokoro-82m";

/** "en_IN" -> "English (India)"; Kokoro's labels are already readable. */
export function voiceLanguage(code: string): string {
  if (!/^[a-z]{2,3}[_-][A-Z]{2}$/.test(code)) return code;
  try {
    return new Intl.DisplayNames(["en"], { type: "language" }).of(code.replace("_", "-")) ?? code;
  } catch { return code; }
}

/** Voices grouped by language, Indian voices first, then alphabetically. */
export function voiceGroups(voices: { id: string; language: string }[]) {
  const groups = new Map<string, { id: string; language: string }[]>();
  for (const v of voices) {
    const label = voiceLanguage(v.language);
    groups.set(label, [...(groups.get(label) ?? []), v]);
  }
  const indian = (label: string) => /India|Hindi/.test(label) ? 0 : 1;
  return [...groups.entries()].sort(([a], [b]) => indian(a) - indian(b) || a.localeCompare(b));
}
export interface AudioSettings {
  model_id?: string;
  language?: string;
  delivery?: string;
  directions?: Record<string, string>;
  narrator: string;
  voices: Record<string, string>;
  rate: number;
  outputs: Record<string, string[]>;
  /** Narrator and character voices last chosen on each speech engine, so
   *  switching engines and back does not lose a cast. */
  casts?: Record<string, { narrator: string; voices: Record<string, string> }>;
  /** "storyteller": one voice tells everything (the default); "cast": a voice per character. */
  telling?: "storyteller" | "cast";
  /** A prepared narration script per chapter, read instead of the chapter text. */
  scripts?: Record<string, string>;
  /** Prepare scripts in Devanagari, for Hindi voices reading Hinglish. */
  devanagari?: boolean;
}
export interface StoryLibrary {
  version: 1;
  books: StoryBook[];
  activeBookId: string | null;
  audio?: Record<string, AudioSettings>;
}
export const EMPTY_LIBRARY: StoryLibrary = { version: 1, books: [], activeBookId: null };
export const ANALYST_IDS = ["qwen3.5-9b-story-4bit", "qwen3-14b-story-4bit", "qwen3-8b-story-4bit", "qwen3-4b-instruct-4bit"];

export interface StoryAnalysisResult {
  book: StoryBook;
  chapter_id: string;
  panels: Panel[];
  /** What of the chapter the panels carry, sentence by sentence. */
  coverage?: Coverage;
  /** Stop was pressed: finished passages are kept, the rest left unanalysed. */
  stopped?: boolean;
  unanalysed?: number;
}
export function unanalysedIn(book: StoryBook | null, chapterId: string | null): number {
  return book?.chapters.find((c) => c.id === chapterId)?.scenes.filter((s) => s.unanalysed).length ?? 0;
}
export function keepBook(library: StoryLibrary, book: StoryBook): StoryLibrary {
  const existing = library.books.find((b) => b.id === book.id);
  const chapters = new Map((existing?.chapters ?? []).map((c) => [c.id, c]));
  for (const chapter of book.chapters) chapters.set(chapter.id, chapter);
  const characters = new Map((existing?.characters ?? []).map((c) => [c.id, c]));
  for (const character of book.characters) {
    const old = characters.get(character.id);
    characters.set(character.id, { ...character,
      aliases: Array.from(new Set([...(old?.aliases ?? []), ...character.aliases])),
    });
  }
  const merged = { ...book, chapters: [...chapters.values()], characters: [...characters.values()],
    references: { ...existing?.references, ...book.references } };
  // Dividing the same text again made a new book every time and kept the old
  // one, so retrying a chapter filled the library until it refused to save at
  // all. A book holding nothing but text this book now holds is superseded.
  // Never replaced by a quicker analysis than its own, and never while
  // narrations are filed under it.
  const held = new Map(merged.chapters.map((c) => [c.source_hash, c.analysis]));
  const narrated = (id: string) => Object.keys(library.audio?.[id]?.outputs ?? {}).length > 0;
  const superseded = new Set(existing ? [] : library.books
    .filter((b) => b.id !== book.id && b.chapters.length > 0 && !narrated(b.id)
      && b.chapters.every((c) => held.has(c.source_hash)
        && !(held.get(c.source_hash) === "voice" && c.analysis !== "voice")))
    .map((b) => b.id));
  const audio = library.audio && Object.fromEntries(
    Object.entries(library.audio).filter(([id]) => !superseded.has(id)));
  return { ...library, activeBookId: book.id, ...(audio ? { audio } : {}),
    books: [...library.books.filter((b) => b.id !== book.id && !superseded.has(b.id)), merged] };
}
/** A first-person narrator: the generic Narrator, or a named person who says "I". */
export function isNarrator(c: Pick<StoryCharacter, "name" | "aliases">): boolean {
  return c.name === "Narrator" || c.aliases.some((a) => ["I", "me", "my"].includes(a));
}

/** How often a name or alias appears in the text, as a whole word. */
function namedIn(text: string, c: StoryCharacter): number {
  let n = 0;
  for (const name of [c.name, ...c.aliases]) {
    if (name.length < 3) continue;
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    n += (text.match(new RegExp(`(?<![\\p{L}\\p{N}])${escaped}(?![\\p{L}\\p{N}])`, "gu")) ?? []).length;
  }
  return n;
}

/**
 * The Board's cast for one chapter: who leads, who recurs, and what each
 * wears in it.
 *
 * The lead used to be whoever the analysis declared first. It is the
 * protagonist: the first-person narrator when the story has one, otherwise
 * whoever speaks most and is seen and named most. The board lets a person
 * change it. Someone seen in two beats or more recurs, and gets a cast sheet;
 * a voice heard once -- the guard calling the last train -- does not.
 */
export function castFromBook(book: StoryBook, chapterId?: string | null) {
  const chapter = book.chapters.find((c) => c.id === chapterId) ?? book.chapters.at(-1);
  const beats = chapter?.scenes.flatMap((s) => s.beats) ?? [];
  const text = chapter?.source ?? "";
  const stats = book.characters.map((c) => {
    const seen = beats.filter((b) => b.visible.some((a) => a.character_id === c.id)).length;
    const lines = beats.flatMap((b) => b.dialogue).filter((d) => d.speaker_id === c.id).length;
    const outfit = beats.flatMap((b) => b.visible)
      .find((a) => a.character_id === c.id && (a.appearance ?? "").trim())?.appearance?.trim() ?? "";
    return { c, seen, lines, named: namedIn(text, c), outfit };
  });
  const score = (s: typeof stats[number]) => s.lines * 2 + s.seen + s.named;
  const present = stats.filter((s) => s.seen || s.lines || s.named);
  const lead = present.find((s) => isNarrator(s.c))
    ?? [...present].sort((a, b) => score(b) - score(a))[0];
  const ordered = [...stats].sort((a, b) =>
    Number(b === lead) - Number(a === lead) || score(b) - score(a));
  return { people: ordered.map((s) => {
    const recurring = s === lead || s.seen >= 2;
    return {
      id: s.c.id, name: s.c.name, aliases: s.c.aliases, description: s.c.visual,
      outfit: s.outfit, mentions: s.seen, recurring,
      tier: s === lead ? 1 : recurring ? 2 : 3,
    };
  }), places: [], words: text.split(/\s+/).filter(Boolean).length };
}

const QWEN_VOICES: Record<string, string> = {
  ryan: "male", aiden: "male", dylan: "male", eric: "male", uncle_fu: "male",
  vivian: "female", serena: "female", ono_anna: "female", sohee: "female",
};
const KOKORO_ACCENT: Record<string, string> = {
  a: "US", b: "UK", h: "Hindi", e: "Spanish", f: "French", i: "Italian", p: "Portuguese",
};
/** "hm_omega" -> "Omega · Hindi, male"; "vivian" -> "Vivian · female". */
export function voiceLabel(id: string): string {
  const title = (s: string) => s.split("_").map((w) => w[0]?.toUpperCase() + w.slice(1)).join(" ");
  const kokoro = /^([abhefip])([fm])_(.+)$/.exec(id);
  if (kokoro) return `${title(kokoro[3])} · ${KOKORO_ACCENT[kokoro[1]]}, ${kokoro[2] === "f" ? "female" : "male"}`;
  if (QWEN_VOICES[id]) return `${title(id)} · ${QWEN_VOICES[id]}`;
  return id;
}
