import { describe, expect, it } from "vitest";
import { castFromBook, EMPTY_LIBRARY, keepBook, voiceGroups, voiceLabel, voiceLanguage, type StoryBook, type StoryLibrary } from "./story";
import { lockStyle, panelPrompt, castSheetPrompt } from "./board";
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
  });
  it("leads with the protagonist, not whoever was declared first", () => {
    const beat = (id: string, visible: string[], speakers: string[] = [], appearance = "") => ({
      id, start: 0, end: 1, source: "x", importance: 2, tension: 2, action: "", setting: "", mood: "",
      visible: visible.map((c) => ({ character_id: c, expression: "", body_language: "", appearance })),
      dialogue: speakers.map((s) => ({ speaker_id: s, text: "x", offscreen: false, start: 0, end: 1 })) });
    const chapter = (beats: ReturnType<typeof beat>[], source = "x") => ({ id: "c", title: "c", source,
      source_hash: "h", scenes: [{ id: "s", title: "s", summary: "", start: 0, end: 1, beats }] });
    const people = [{ id: "g", name: "Guard", aliases: [], visual: "", voice: "" },
                    { id: "j", name: "Jonas", aliases: [], visual: "", voice: "" },
                    { id: "m", name: "Mira Vale", aliases: ["Mira"], visual: "", voice: "" }];
    // Most spoken and seen leads, wherever the analysis listed them.
    const talky = { ...book, characters: people,
      chapters: [chapter([beat("1", ["j", "m"], ["m", "m"], "red coat"), beat("2", ["m"]), beat("3", [], ["g"])])] };
    const cast = castFromBook(talky, "c");
    expect(cast.people[0]).toMatchObject({ id: "m", tier: 1, recurring: true, outfit: "red coat" });
    // Heard once, never seen: no cast sheet.
    expect(cast.people.find((p) => p.id === "g")).toMatchObject({ recurring: false, tier: 3 });
    // A first-person narrator is the protagonist even when others speak more.
    const told = { ...talky, characters: people.map((c) => (c.id === "j" ? { ...c, aliases: ["I", "me", "my"] } : c)) };
    expect(castFromBook(told, "c").people[0].id).toBe("j");
  });
  it("keeps emotion and the brief in a directed panel's prompt", () => {
    const panel: Panel = { shot: "close-up", subject: "Mira", action: "releases the key",
      setting: "station", character_in_frame: true, caption: "", description: "cold blue light",
      cast: [{ character_id: "mira", name: "Mira", visual: "red coat", expression: "grief",
        body_language: "clenched hands" }], mood: "loss", tension: 5 };
    const prompt = panelPrompt(panel, lockStyle("ink"));
    for (const text of ["cold blue light", "grief", "clenched hands", "The feeling of the moment: loss"]) {
      expect(prompt).toContain(text);
    }
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
  it("says a change of clothes against the outfit on the cast sheet", () => {
    const prompt = panelPrompt(panel("blue dress"), lockStyle("ink"), "");
    expect(prompt).toContain("now wearing blue dress instead of the outfit on their reference");
    expect(prompt).toContain("Current location state: dry floor; dawn");
    expect(castSheetPrompt(lockStyle("ink"), "Mira", "curly black hair", "grey coat")).toContain("Wearing grey coat");
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
