import type { Panel } from "./types";

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
  attribution_review?: boolean;
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
}
export interface StoryChapter {
  id: string;
  title: string;
  source: string;
  source_hash: string;
  scenes: StoryScene[];
}
export interface CharacterReference {
  kind?: "identity-face-v1";
  image: string;
  style_words: string;
  visual: string;
}
export interface StoryBook {
  references?: Record<string, CharacterReference>;
  version: 1;
  id: string;
  characters: StoryCharacter[];
  chapters: StoryChapter[];
}
export const SPEECH_MODEL_ID = "qwen3-tts-1.7b-custom-6bit";
export interface AudioSettings {
  model_id?: string;
  language?: string;
  delivery?: string;
  directions?: Record<string, string>;
  narrator: string;
  voices: Record<string, string>;
  rate: number;
  outputs: Record<string, string[]>;
}
export interface StoryLibrary {
  version: 1;
  books: StoryBook[];
  activeBookId: string | null;
  audio?: Record<string, AudioSettings>;
}
export const EMPTY_LIBRARY: StoryLibrary = { version: 1, books: [], activeBookId: null };
export const ANALYST_IDS = ["qwen3-14b-story-4bit", "qwen3-8b-story-4bit", "qwen3-4b-instruct-4bit"];

export interface StoryAnalysisResult {
  book: StoryBook;
  chapter_id: string;
  panels: Panel[];
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
  return { ...library, activeBookId: book.id,
    books: [...library.books.filter((b) => b.id !== book.id), merged] };
}
export function castFromBook(book: StoryBook) {
  return { people: book.characters.map((c, i) => ({
    id: c.id, name: c.name, aliases: c.aliases, description: c.visual,
    mentions: book.chapters.flatMap((ch) => ch.scenes.flatMap((s) => s.beats))
      .filter((b) => b.visible.some((a) => a.character_id === c.id)).length,
    tier: i === 0 ? 1 : 2,
  })), places: [], words: book.chapters.at(-1)?.source.split(/\s+/).length ?? 0 };
}
