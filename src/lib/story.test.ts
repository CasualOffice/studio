import { describe, expect, it } from "vitest";
import { castFromBook, EMPTY_LIBRARY, keepBook, voiceGroups, voiceLabel, voiceLanguage, type StoryBook, type StoryLibrary } from "./story";
import { lockStyle, panelPrompt, identityPrompt } from "./board";
import { staleness } from "./boardDoc";
import type { Panel } from "./types";

const book: StoryBook = { version: 1, id: "book", chapters: [], characters: [
  { id: "mira", name: "Mira", aliases: ["Mi"], visual: "red coat", voice: "quiet" },
] };
describe("shared story model", () => {
  it("keeps other books and voice assignments when another chapter is saved", () => {
    const library: StoryLibrary = { version: 1, activeBookId: "other", books: [book, { ...book, id: "other" }],
      audio: { book: { narrator: "Samantha", voices: { mira: "Karen" }, rate: 160, outputs: {} } } };
    const result = keepBook(library, { ...book, characters: [...book.characters] });
    expect(result.books.map((b) => b.id)).toEqual(["other", "book"]);
    expect(result.audio).toEqual(library.audio);
    expect(library.activeBookId).toBe("other");
  });
  it("replaces a book when the same text is divided again, and keeps books with other chapters", () => {
    const chapter = (hash: string) => ({ id: hash, title: hash, source: hash, source_hash: hash, scenes: [] });
    const first: StoryBook = { ...book, id: "first", chapters: [chapter("h1")] };
    const longer: StoryBook = { ...book, id: "longer", chapters: [chapter("h3"), chapter("h1")] };
    let library = keepBook(keepBook(EMPTY_LIBRARY, longer), first);
    library = { ...library, audio: { first: { narrator: "ryan", voices: {}, rate: 160, outputs: {} } } };
    const retry = keepBook(library, { ...book, id: "retry", chapters: [chapter("h1")] });
    expect(retry.books.map((b) => b.id)).toEqual(["longer", "retry"]);
    expect(retry.audio).toEqual({});
  });
  it("never lets a voice-only analysis replace a full one, or replace a narrated book", () => {
    const chapter = (hash: string, analysis?: "voice") => ({ id: hash, title: hash, source: hash, source_hash: hash, scenes: [], analysis });
    const full: StoryBook = { ...book, id: "full", chapters: [chapter("h1")] };
    const quick = keepBook(keepBook(EMPTY_LIBRARY, full), { ...book, id: "quick", chapters: [chapter("h1", "voice")] });
    expect(quick.books.map((b) => b.id)).toEqual(["full", "quick"]);
    const narrated = { ...quick, audio: { quick: { narrator: "ryan", voices: {}, rate: 160, outputs: { h1: ["audio"] } } } };
    const redone = keepBook(narrated, { ...book, id: "board", chapters: [chapter("h1")] });
    expect(redone.books.map((b) => b.id)).toEqual(["quick", "board"]);
  });
  it("carries canonical IDs and aliases into the editable Board cast", () => {
    expect(castFromBook(book).people[0]).toMatchObject({ id: "mira", aliases: ["Mi"], description: "red coat" });
    const told = { ...book, characters: [...book.characters,
      { id: "x", name: "Xan", aliases: [], visual: "", voice: "" }, { id: "y", name: "Yul", aliases: [], visual: "", voice: "" },
      { id: "n", name: "Narrator", aliases: [], visual: "", voice: "" }] };
    expect(castFromBook(told).people[1].name).toBe("Narrator");
  });
  it("keeps actions and emotion even when surface enrichment is present", () => {
    const panel: Panel = { shot: "close-up", subject: "Mira", action: "releases the key",
      setting: "station", character_in_frame: true, caption: "", description: "cold blue light",
      cast: [{ character_id: "mira", name: "Mira", visual: "red coat", expression: "grief",
        body_language: "clenched hands" }], mood: "loss", tension: 5 };
    const prompt = panelPrompt(panel, lockStyle("ink"), "ignored legacy binding");
    for (const text of ["releases the key", "cold blue light", "grief", "clenched hands", "The feeling of the moment: loss"]) {
      expect(prompt).toContain(text);
    }
    expect(prompt).not.toContain("ignored legacy binding");
  });
  it("does not invite an offscreen speaker into the image", () => {
    const panel: Panel = { shot: "wide", subject: "door", action: "A guard calls from behind the locked door",
      setting: "station", character_in_frame: false, caption: "", cast: [],
      dialogue: [{ speaker: "Guard", text: "Last train!", offscreen: true }] };
    expect(panelPrompt(panel, lockStyle("ink"), "")).toContain("No people visible in frame");
  });
  it("changing expression or mood invalidates drawn art, while lettering does not", () => {
    expect(staleness({ mood: "fear" }).art).toBe(true);
    expect(staleness({ cast: [] }).art).toBe(true);
    expect(staleness({ caption: "Original words" }).art).toBe(false);
  });
});

describe("story appearance continuity", () => {
  const panel = (appearance: string, location_state = "dry floor; dawn"): Panel => ({
    shot: "medium", subject: "Mira", action: "walks", setting: "station", place_key: "station",
    chapter_id: "chapter", scene: 1, character_in_frame: true, caption: "", location_state,
    cast: [{ character_id: "mira", name: "Mira", visual: "curly black hair", appearance,
      expression: "calm", body_language: "walking" }],
  });
  it("uses story clothing as a replacement and separates the identity reference", () => {
    const prompt = panelPrompt(panel("blue dress"), lockStyle("ink"), "");
    expect(prompt).toContain("replaces earlier clothing");
    expect(prompt).toContain("blue dress");
    expect(prompt).toContain("Current location state: dry floor; dawn");
    expect(identityPrompt(lockStyle("ink"), "Mira: curly black hair")).toContain("No full-body outfit");
  });
});

describe("voice choice", () => {
  it("names system voice languages and lists Indian voices first", () => {
    expect(voiceLanguage("en_IN")).toBe("English (India)");
    const groups = voiceGroups([{ id: "Samantha", language: "en_US" }, { id: "Rishi", language: "en_IN" },
      { id: "Lekha", language: "hi_IN" }, { id: "hm_omega", language: "Hindi / Indian English" }]);
    expect(groups.map(([label]) => label)).toEqual(["English (India)", "Hindi (India)", "Hindi / Indian English", "American English"]);
  });
});

describe("voice labels", () => {
  it("names engine voices by person, accent and gender", () => {
    expect(voiceLabel("hm_omega")).toBe("Omega · Hindi, male");
    expect(voiceLabel("bf_isabella")).toBe("Isabella · UK, female");
    expect(voiceLabel("uncle_fu")).toBe("Uncle Fu · male");
    expect(voiceLabel("Rishi")).toBe("Rishi");
  });
});
