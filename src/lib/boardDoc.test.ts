import { describe, expect, it } from "vitest";
import {
  anchorOverflow, artOf, edit, fromArrays, identify, insert, merge, move,
  prune, remove, replan, setArt, split, staleness, toArrays,
} from "./boardDoc";
import type { Panel } from "./types";

/**
 * Position is not identity.
 *
 * The board held five arrays lined up by index. Nothing could move, and the
 * moment anything did, every picture and note attached itself to a different
 * panel -- with the arrays still the right length, so nothing was detectably
 * wrong until someone read the pages. These are the cases that were silently
 * wrong before, written as assertions.
 */
const panel = (over: Partial<Panel> = {}): Panel => ({
  shot: "medium", subject: "Mira", action: "hands over the key",
  setting: "a waiting room", character_in_frame: true, caption: "", ...over,
});

/** A three-panel board where every panel is drawn and noted. */
const board = () => fromArrays(
  [panel({ action: "first" }), panel({ action: "second" }),
   panel({ action: "third" })],
  ["art-1", "art-2", "art-3"],
  [0, 2, 1],
  { 0: "note one", 1: "note two", 2: "note three" },
  { 1: ["old-2"] },
);

describe("identity survives the board moving", () => {
  it("gives every panel an id", () => {
    const doc = board();
    const ids = doc.panels.map((p) => p.id);
    expect(new Set(ids).size).toBe(3);
    expect(ids.every(Boolean)).toBe(true);
  });

  it("does not renumber a board it has already seen", () => {
    // Running on every reopen, so renumbering would detach all the work.
    const once = identify([panel(), panel()]);
    const twice = identify(once);
    expect(twice.map((p) => p.id)).toEqual(once.map((p) => p.id));
  });

  it("replaces a duplicate id rather than tolerating it", () => {
    // Two panels sharing an id share their art.
    const dup = identify([{ ...panel(), id: "same" } as Panel,
                          { ...panel(), id: "same" } as Panel]);
    expect(dup[0].id).not.toBe(dup[1].id);
  });

  it("reads the old arrays by position exactly once", () => {
    const doc = board();
    expect(artOf(doc, doc.panels[1].id)).toEqual({
      drawn: "art-2", redraws: 2, note: "note two", variants: ["old-2"],
    });
  });

  it("rebuilds the arrays the board renders from", () => {
    const { drawn, redraws, notes, variants } = toArrays(board());
    expect(drawn).toEqual(["art-1", "art-2", "art-3"]);
    expect(redraws).toEqual([0, 2, 1]);
    expect(notes).toEqual({ 0: "note one", 1: "note two", 2: "note three" });
    expect(variants).toEqual({ 1: ["old-2"] });
  });
});

describe("moving a panel takes its work with it", () => {
  it("does not swap two panels' pictures when they change places", () => {
    // The defect. Reordering used to leave the arrays untouched, so the two
    // panels exchanged pictures and notes while keeping their own text.
    const doc = board();
    const third = doc.panels[2].id;
    const moved = move(doc, third, 0);
    expect(moved.panels[0].action).toBe("third");
    expect(artOf(moved, third).drawn).toBe("art-3");
    expect(artOf(moved, third).note).toBe("note three");
    expect(toArrays(moved).drawn).toEqual(["art-3", "art-1", "art-2"]);
  });

  it("does not shift every note down when a panel is inserted at the top", () => {
    const doc = insert(board(), 0, panel({ action: "new opening" }));
    expect(doc.panels[0].action).toBe("new opening");
    expect(artOf(doc, doc.panels[0].id).drawn).toBeNull();
    expect(toArrays(doc).notes).toEqual({
      1: "note one", 2: "note two", 3: "note three",
    });
  });

  it("does not move the whole board up when one panel is deleted", () => {
    const doc = board();
    const { doc: after, removed } = remove(doc, doc.panels[0].id);
    expect(removed?.art.drawn).toBe("art-1");
    expect(after.panels.map((p) => p.action)).toEqual(["second", "third"]);
    expect(toArrays(after).drawn).toEqual(["art-2", "art-3"]);
    expect(toArrays(after).notes).toEqual({ 0: "note two", 1: "note three" });
  });

  it("leaves a board alone when asked to move a panel that is not in it", () => {
    const doc = board();
    expect(move(doc, "not-a-panel", 0)).toBe(doc);
    expect(remove(doc, "not-a-panel").doc).toBe(doc);
  });

  it("clamps a position rather than losing the panel off the end", () => {
    const doc = insert(board(), 99, panel({ action: "last" }));
    expect(doc.panels).toHaveLength(4);
    expect(doc.panels[3].action).toBe("last");
  });
});

describe("what an edit makes stale", () => {
  it("retires the picture when the action changes", () => {
    const doc = board();
    const id = doc.panels[0].id;
    const { doc: after, stale } = edit(doc, id, { action: "takes it back" });
    expect(stale.art).toBe(true);
    expect(artOf(after, id).drawn).toBeNull();
  });

  it("keeps the picture when only the caption changes", () => {
    // Re-rendering a panel to fix a comma costs minutes and a new face.
    const doc = board();
    const id = doc.panels[0].id;
    const { doc: after, stale } = edit(doc, id, { caption: "Later that day." });
    expect(stale.art).toBe(false);
    expect(stale.pages).toBe(true);
    expect(artOf(after, id).drawn).toBe("art-1");
  });

  it("keeps the picture when only the dialogue changes", () => {
    const doc = board();
    const id = doc.panels[0].id;
    const { stale } = edit(doc, id, {
      dialogue: [{ speaker: "Mira", text: "Take it." }],
    });
    expect(stale.art).toBe(false);
  });

  it("does not delete the retired picture, it files it", () => {
    // The board used to set drawn[i] = null, leaving the picture in the vault
    // with nothing pointing at it.
    const doc = board();
    const id = doc.panels[0].id;
    const { doc: after } = edit(doc, id, { action: "takes it back" });
    expect(artOf(after, id).variants).toContain("art-1");
  });

  it("keeps the brief when another field changes", () => {
    // The brief is edited directly. Clearing it on any other edit fell back
    // to drawing the raw source sentence.
    const doc = fromArrays([panel({ description: "Mira at the green kitchen door, dawn." })]);
    const id = doc.panels[0].id;
    const { doc: after, stale } = edit(doc, id, { action: "slams the door" });
    expect(stale.description).toBe(false);
    expect(after.panels[0].description).toBe("Mira at the green kitchen door, dawn.");
  });

  it("does not clear a description the user typed themselves", () => {
    const doc = fromArrays([panel({ description: "old" })]);
    const id = doc.panels[0].id;
    const { doc: after } = edit(doc, id, { description: "mine" });
    expect(after.panels[0].description).toBe("mine");
  });

  it("names the fields that change what is drawn", () => {
    for (const field of ["shot", "subject", "action", "setting", "place",
                         "character_in_frame", "characters", "description"]) {
      expect(staleness({ [field]: "x" } as Partial<Panel>).art,
             `${field} changes the picture`).toBe(true);
    }
    for (const field of ["caption", "dialogue", "scene_title", "purpose"]) {
      expect(staleness({ [field]: "x" } as Partial<Panel>).art,
             `${field} does not change the picture`).toBe(false);
    }
  });

  it("says nothing is stale when nothing changed", () => {
    expect(staleness({})).toEqual({ art: false, description: false, pages: false });
  });
});

describe("splitting and merging keep the work reachable", () => {
  it("leaves the picture on the half that kept the identity", () => {
    const doc = board();
    const id = doc.panels[0].id;
    const after = split(doc, id, { action: "and closes his fingers on it" });
    expect(after.panels).toHaveLength(4);
    expect(artOf(after, id).drawn).toBe("art-1");
    expect(artOf(after, after.panels[1].id).drawn).toBeNull();
  });

  it("does not credit one source quote to both halves", () => {
    // A quote credited twice is how coverage reported a board complete that
    // had dropped three sentences.
    const doc = fromArrays([panel({ source: "He closed his fingers around it." })]);
    const after = split(doc, doc.panels[0].id);
    expect(after.panels[0].source).toBe("He closed his fingers around it.");
    expect(after.panels[1].source).toBe("");
  });

  it("keeps the absorbed panel's picture in the survivor's history", () => {
    // After a merge there is no panel left to reach it from.
    const doc = board();
    const first = doc.panels[0].id;
    const after = merge(doc, first);
    expect(after.panels).toHaveLength(2);
    expect(artOf(after, first).variants).toContain("art-2");
    expect(artOf(after, first).variants).toContain("old-2");
    expect(artOf(after, first).drawn).toBe("art-1");
  });

  it("carries the absorbed panel's text and notes over", () => {
    const doc = board();
    const first = doc.panels[0].id;
    const after = merge(doc, first);
    expect(after.panels[0].action).toBe("first, second");
    expect(artOf(after, first).note).toBe("note one note two");
  });

  it("refuses to merge the last panel with nothing", () => {
    const doc = board();
    expect(merge(doc, doc.panels[2].id)).toBe(doc);
  });

  it("does not leave the survivor's own picture in its own history", () => {
    const doc = board();
    const first = doc.panels[0].id;
    expect(artOf(merge(doc, first), first).variants).not.toContain("art-1");
  });
});

describe("what arrives from disk", () => {
  it("drops art belonging to panels that are gone", () => {
    const doc = board();
    const orphaned = { ...doc, art: { ...doc.art, ghost: {
      drawn: "x", redraws: 0, note: "", variants: [] } } };
    expect(Object.keys(prune(orphaned).art)).toHaveLength(3);
  });

  it("survives a draft with no panels at all", () => {
    const doc = fromArrays(null);
    expect(doc.panels).toEqual([]);
    expect(toArrays(doc).drawn).toEqual([]);
  });

  it("survives arrays shorter than the board", () => {
    // A v1 draft saved mid-draw has exactly this shape.
    const doc = fromArrays([panel(), panel(), panel()], ["only-one"]);
    expect(toArrays(doc).drawn).toEqual(["only-one", null, null]);
  });

  it("ignores art for a panel that is not there", () => {
    const doc = board();
    expect(setArt(doc, "ghost", { note: "x" })).toBe(doc);
  });
});

describe("an overflow report follows its panel", () => {
  const report = (panelIndex: number) => ({
    panel: panelIndex, page: 1, spoken: 3, lettered: 2, caption_clipped: false,
  });

  it("points at the board panel, not the position it was sent in", () => {
    // The engine is given the drawn panels only, so its index 1 is the second
    // drawn panel, which may be anywhere on the board.
    const doc = fromArrays(
      [panel({ action: "a" }), panel({ action: "b" }), panel({ action: "c" })],
      [null, "art-b", "art-c"]);
    const sent = [doc.panels[1].id, doc.panels[2].id];
    expect(anchorOverflow([report(1)], sent, doc)[0].panel).toBe(2);
  });

  it("follows the panel after the board is reordered", () => {
    const doc = fromArrays([panel(), panel()], ["a", "b"]);
    const sent = doc.panels.map((p) => p.id);
    const moved = move(doc, doc.panels[1].id, 0);
    expect(anchorOverflow([report(1)], sent, moved)[0].panel).toBe(0);
  });

  it("drops a report about a panel that has been deleted", () => {
    // Better than naming whichever panel is at that position now.
    const doc = fromArrays([panel(), panel()], ["a", "b"]);
    const sent = doc.panels.map((p) => p.id);
    const { doc: after } = remove(doc, doc.panels[1].id);
    expect(anchorOverflow([report(1)], sent, after)).toEqual([]);
  });
});


describe("replan", () => {
  const p = (beat: string, caption = ""): Panel => ({ shot: "wide", subject: "", action: "", setting: "",
    character_in_frame: false, caption, beat_id: beat });
  it("keeps the picture and note of every unchanged beat's frames and starts new ones empty", () => {
    let doc = fromArrays([p("b1"), p("b1"), p("stub")], ["img1", "img2", "img3"], [], { 1: "keep" });
    doc = replan(doc, [p("b1", "new caption"), p("b1"), p("b2"), p("b3")]);
    const arrays = toArrays(doc);
    expect(arrays.drawn).toEqual(["img1", "img2", null, null]);
    expect(arrays.notes[1]).toBe("keep");
    expect(doc.panels[0].caption).toBe("new caption");
    expect(new Set(doc.panels.map((x) => x.id)).size).toBe(4);
  });

  it("keeps a written brief for a panel that still shows the same moment", () => {
    // Changing the format or the page count re-plans the chapter; a brief a
    // writer or the user wrote must not be replaced by the rules' one.
    const old = { ...p("b1"), moment: "Mira opens the door.", description: "written by hand" };
    let doc = fromArrays([old, { ...p("b2"), moment: "Rain.", description: "also mine" }]);
    doc = replan(doc, [{ ...p("b1"), moment: "Mira opens the door.", description: "rules" },
                       { ...p("b2"), moment: "Something else.", description: "rules" }]);
    expect(doc.panels.map((x) => x.description)).toEqual(["written by hand", "rules"]);
  });
});
