import { describe, expect, it } from "vitest";
import { boardReducer, coverageOf, emptyBoard, estimateMinutes, finishedPages, missingSheets, nextPage,
         placesOf, runReducer, idleRun, type BoardState } from "./boardFlow";
import { artOf, fromArrays } from "./boardDoc";
import { lockStyle } from "./board";
import type { Cast, Panel } from "./types";
import type { StoryBook } from "./story";

const actor = (id: string, name: string) => ({ character_id: id, name, visual: "", expression: "", body_language: "" });
const panel = (over: Partial<Panel> = {}): Panel => ({
  shot: "MS", subject: "", action: "", setting: "pier", character_in_frame: true, caption: "", scene: 1,
  cast: [actor("m", "Mira")], ...over,
});
const cast: Cast = { places: [], words: 100, people: [
  { id: "m", name: "Mira", description: "short hair", outfit: "red coat", mentions: 3, tier: 1, recurring: true },
  { id: "j", name: "Jonas", description: "", outfit: "", mentions: 2, tier: 2, recurring: true },
  { id: "g", name: "The guard", description: "", outfit: "", mentions: 0, tier: 3, recurring: false },
] };
const book: StoryBook = { version: 1, id: "b", chapters: [], characters: [
  { id: "m", name: "Mira", aliases: [], visual: "short hair", voice: "" },
  { id: "j", name: "Jonas", aliases: [], visual: "", voice: "" }] };

function analysed(panels: Panel[] = [panel(), panel({ cast: [actor("j", "Jonas")] }), panel({ cast: [] })],
                  drawn: (string | null)[] = []): BoardState {
  const s = boardReducer(emptyBoard(), { type: "analysed", book, chapterId: "c", doc: fromArrays(panels, drawn),
                                         cast, coverage: null, projectId: "p" });
  return s;
}

describe("the cast is the one gate", () => {
  it("starts unapproved, and only recurring people need a sheet", () => {
    const s = analysed();
    expect(s.castApproved).toBe(false);
    expect(missingSheets(s).map((p) => p.name)).toEqual(["Mira", "Jonas"]);
    const sheeted = boardReducer(boardReducer(s, { type: "sheet", id: "m", image: "sm" }),
                                 { type: "sheet", id: "j", image: "sj" });
    expect(missingSheets(sheeted)).toEqual([]);
    // Drawing sheets before approval is not a recast.
    expect(sheeted.recast).toEqual([]);
    expect(boardReducer(sheeted, { type: "approve" }).castApproved).toBe(true);
  });

  it("recasting one person redraws only their panels, after approval again", () => {
    let s = analysed(undefined, ["a1", "a2", "a3"]);
    s = boardReducer(s, { type: "sheet", id: "m", image: "sm" });
    s = boardReducer(s, { type: "sheet", id: "j", image: "sj" });
    s = boardReducer(s, { type: "approve" });
    s = boardReducer(s, { type: "sheet", id: "j", image: "sj2", regenerated: true });
    expect(s.castApproved).toBe(false);
    expect(s.recast).toEqual(["j"]);
    expect(s.sheetRegens.j).toBe(1);
    // Nothing is retired until the new sheet is approved.
    expect(s.doc.panels.map((p) => artOf(s.doc, p.id).drawn)).toEqual(["a1", "a2", "a3"]);
    s = boardReducer(s, { type: "approve" });
    expect(s.castApproved).toBe(true);
    expect(s.doc.panels.map((p) => artOf(s.doc, p.id).drawn)).toEqual(["a1", null, "a3"]);
    expect(artOf(s.doc, s.doc.panels[1].id).variants).toEqual(["a2"]);
  });

  it("lets the person choose the lead", () => {
    const s = boardReducer(analysed(), { type: "lead", id: "j" });
    expect(s.cast!.people[0]).toMatchObject({ id: "j", tier: 1 });
    expect(s.cast!.people.find((p) => p.id === "m")!.tier).toBe(2);
  });

  it("renames a person everywhere their name is drawn or lettered", () => {
    const s = boardReducer(analysed([panel({ dialogue: [{ speaker: "Mira", text: "Go." }], positions: { Mira: "left third" } })]),
                           { type: "person", id: "m", patch: { name: "Mira Vale" } });
    const p = s.doc.panels[0];
    expect(p.cast![0].name).toBe("Mira Vale");
    expect(p.dialogue![0].speaker).toBe("Mira Vale");
    expect(p.positions).toEqual({ "Mira Vale": "left third" });
    expect(s.storyBook!.characters[0].name).toBe("Mira Vale");
  });
});

describe("what a change throws away", () => {
  it("a new story clears everything read from the old one", () => {
    const s = boardReducer(analysed(), { type: "story", text: "Something else entirely." });
    expect(s.divided).toBe(false);
    expect(s.cast).toBe(null);
    expect(s.doc.panels).toEqual([]);
  });

  it("a new style retires the pictures and sheets but keeps the plan", () => {
    let s = analysed(undefined, ["a1", null, null]);
    s = boardReducer(s, { type: "sheet", id: "m", image: "sm" });
    s = boardReducer(s, { type: "style", style: lockStyle("ink") });
    expect(s.sheets).toEqual({});
    expect(s.castApproved).toBe(false);
    expect(s.doc.panels).toHaveLength(3);
    expect(artOf(s.doc, s.doc.panels[0].id)).toMatchObject({ drawn: null, variants: ["a1"] });
  });
});

describe("drawing goes a page at a time", () => {
  const many = Array.from({ length: 9 }, (_, i) => panel({ scene: i < 5 ? 1 : 2 }));
  it("draws the first page with anything missing, in reading order", () => {
    const s = analysed(many, ["x", null, "y"]);
    const places = placesOf(s.doc.panels, "manga", "page");
    expect(nextPage(s.doc, places)).toEqual([1, 3]);
    const later = analysed(many, ["x", "x", "x", "x"]);
    expect(nextPage(later.doc, placesOf(later.doc.panels, "manga", "page"))[0]).toBe(4);
  });

  it("knows which pages are finished, to compose them at once", () => {
    const s = analysed(many, ["x", "x", "x", "x"]);
    expect(finishedPages(s.doc, placesOf(s.doc.panels, "manga", "page"))).toEqual([0]);
  });
});

describe("honest coverage", () => {
  const sentences = ["Mira opened the door.", "The street was wet.", "Go home.", "Rain began."];
  it("counts what a panel draws, letters or says, and names the rest", () => {
    const panels = [panel({ moment: "Mira opened the door.", covers: ["Mira opened the door."] }),
                    panel({ caption: "The street was wet.", dialogue: [{ speaker: "Mira", text: "Go home." }] })];
    const c = coverageOf(panels, sentences)!;
    expect(c).toMatchObject({ covered: 3, total: 4, percent: 75, missing: ["Rain began."], missing_total: 1 });
  });

  it("drops when a panel carrying a sentence is removed", () => {
    const one = [panel({ covers: ["Mira opened the door."] })];
    expect(coverageOf(one, sentences)!.percent).toBe(25);
    expect(coverageOf([], sentences)!.percent).toBe(0);
  });

  it("does not count a sentence that was only folded into a panel", () => {
    const c = coverageOf([panel({ folded: ["Rain began."] })], ["Rain began."])!;
    expect(c.missing).toEqual(["Rain began."]);
  });

  it("has nothing to say without the engine's sentences", () => {
    expect(coverageOf([panel()], undefined)).toBe(null);
  });
});

describe("the time a board will take", () => {
  it("counts the cast sheets and every panel left, and no rooms", () => {
    expect(estimateMinutes(2, 0, 768).low).toBe(2);
    expect(estimateMinutes(0, 6, 768).low).toBe(17);
    expect(estimateMinutes(2, 6, 768).low).toBe(19);
    // Once panels have been timed here, that is what is said.
    expect(estimateMinutes(0, 6, 768, 60_000).low).toBe(6);
  });
});

describe("the run", () => {
  it("keeps how the last job ended until the next one starts", () => {
    let r = runReducer(idleRun, { type: "start", stage: "drawing" });
    r = runReducer(r, { type: "end", outcome: "Drew 4 panels." });
    expect(r).toMatchObject({ stage: "idle", outcome: "Drew 4 panels." });
    expect(runReducer(r, { type: "start", stage: "sheets" }).outcome).toBe(null);
  });
});
