import { describe, expect, it } from "vitest";
import { panelPrompt, panelSeed, sheetPrompt, styleWords, STYLES } from "./board";
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

describe("sheetPrompt", () => {
  it("asks for a neutral reference, not a scene", () => {
    const out = sheetPrompt("anime", CHAR);
    expect(out).toContain("reference sheet");
    expect(out).toContain("plain background");
    expect(out).toContain(CHAR);
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
