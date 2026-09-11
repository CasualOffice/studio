import { describe, expect, it } from "vitest";
import { panelPrompt, panelReferences, panelSeed, placePrompt, sheetPrompt,
         styleWords, STYLES } from "./board";
import type { Panel } from "./types";

const CHAR = "a young woman with short black hair and a red scarf";

const panel = (over: Partial<Panel> = {}): Panel => ({
  shot: "wide",
  subject: "Mira",
  action: "stands in the hallway",
  setting: "a narrow flat",
  character_in_frame: true,
  caption: "",
  ...over,
});

describe("panelPrompt", () => {
  it("keeps the subject, which is not the same as the action", () => {
    // The first version dropped `subject` entirely, so a panel about a tap
    // was described only as "water running" and drawn as the character.
    const out = panelPrompt(panel({ subject: "a kitchen tap" }), "anime", CHAR);
    expect(out).toContain("a kitchen tap");
  });

  it("names the character when they are in frame", () => {
    expect(panelPrompt(panel(), "anime", CHAR)).toContain("red scarf");
  });

  it("leaves the character out of a panel they are not in", () => {
    // A close-up of a running tap, handed the protagonist's description,
    // draws her running instead of the tap.
    const out = panelPrompt(
      panel({ subject: "a kitchen tap", action: "water running",
              character_in_frame: false, shot: "close-up" }),
      "anime", CHAR);
    expect(out).not.toContain("red scarf");
    expect(out).toContain("a kitchen tap");
    expect(out).toContain("close-up shot");
  });

  it("leads with the style, so every panel matches the others", () => {
    for (const s of STYLES) {
      expect(panelPrompt(panel(), s.id, CHAR).startsWith(s.words)).toBe(true);
    }
  });

  it("drops empty parts rather than leaving stray punctuation", () => {
    const out = panelPrompt(
      panel({ action: "", setting: "" }), "anime", CHAR);
    expect(out).not.toContain("..");
    expect(out).not.toContain(". .");
    expect(out.endsWith(".")).toBe(true);
  });

  it("survives an empty character description", () => {
    const out = panelPrompt(panel(), "anime", "   ");
    expect(out).toContain("Mira");
    expect(out).not.toContain(", .");
  });

  it("falls back to no style words for an unknown style", () => {
    expect(styleWords("no-such-style")).toBe("");
    expect(panelPrompt(panel(), "no-such-style", CHAR).startsWith("wide shot")).toBe(true);
  });
});

describe("panelPrompt with a worked-up scene", () => {
  it("uses the scene instead of the terse fields", () => {
    // A worked-up scene already carries the setting and the light. Appending
    // the thin version too describes the panel twice and leaves the model to
    // reconcile them.
    const out = panelPrompt(
      panel({ action: "stands in the hallway", setting: "a flat",
              description: "A narrow hallway, faded linoleum, dim light from a corner lamp." }),
      "anime", CHAR);
    expect(out).toContain("faded linoleum");
    expect(out).not.toContain("a flat");
  });

  it("falls back to the terse fields when the scene is empty", () => {
    const out = panelPrompt(panel({ description: "" }), "anime", CHAR);
    expect(out).toContain("stands in the hallway");
    expect(out).toContain("a narrow flat");
  });

  it("falls back when the scene is absent entirely", () => {
    const p = panel();
    delete (p as { description?: string }).description;
    expect(panelPrompt(p, "anime", CHAR)).toContain("a narrow flat");
  });

  it("still leaves the character out of a panel they are not in", () => {
    const out = panelPrompt(
      panel({ character_in_frame: false, subject: "a kitchen tap",
              description: "Water running into a steel sink, grey daylight." }),
      "anime", CHAR);
    expect(out).not.toContain("red scarf");
    expect(out).toContain("a kitchen tap");
  });
});

describe("sheetPrompt", () => {
  it("asks for a neutral reference, not a scene", () => {
    const out = sheetPrompt("anime", CHAR);
    expect(out).toContain("reference sheet");
    expect(out).toContain("plain background");
    expect(out).toContain(CHAR);
  });
});

describe("panelReferences", () => {
  it("hands the drawer both the character and the room", () => {
    // Verified against the model: two references keep consecutive panels in
    // the same room rather than a similar one.
    expect(panelReferences(panel(), "sheet", "room")).toEqual(["sheet", "room"]);
  });

  it("leaves the character out of a panel she is not in", () => {
    // Handing her sheet to a shot of an empty tap is what drew her into it.
    expect(panelReferences(panel({ character_in_frame: false }), "sheet", "room"))
      .toEqual(["room"]);
  });

  it("falls back to the character alone when no room was built", () => {
    expect(panelReferences(panel(), "sheet", null)).toEqual(["sheet"]);
  });

  it("never returns nothing when a sheet exists", () => {
    // A panel with no references is drawn from the prompt alone, which is
    // exactly the inconsistency this is here to prevent.
    const out = panelReferences(
      panel({ character_in_frame: false }), "sheet", null);
    expect(out).toEqual(["sheet"]);
  });
});

describe("placePrompt", () => {
  it("asks for the room without anyone in it", () => {
    const out = placePrompt("ink", "faded blue walls, worn pine boards");
    expect(out).toContain("no people");
    expect(out).toContain("worn pine boards");
  });
});

describe("panelSeed", () => {
  it("gives each panel its own seed", () => {
    expect(panelSeed(0, 0)).not.toBe(panelSeed(1, 0));
  });

  it("moves on when a panel is redrawn", () => {
    // Redrawing with the same seed reproduces the picture just rejected,
    // which reads as the button not working.
    expect(panelSeed(2, 1)).not.toBe(panelSeed(2, 0));
    expect(panelSeed(2, 2)).not.toBe(panelSeed(2, 1));
  });

  it("does not collide with a neighbouring panel after redraws", () => {
    const seen = new Set<number>();
    for (let i = 0; i < 12; i++) {
      for (let r = 0; r < 5; r++) seen.add(panelSeed(i, r));
    }
    expect(seen.size).toBe(12 * 5);
  });
});
