import { describe, expect, it } from "vitest";
import { isKnownStyle, lockStyle, panelPrompt, panelReferences, panelSeed,
         placePrompt, reviveLock, sheetPrompt, styleHasMoved, styleWords,
         STYLES } from "./board";
import type { Panel } from "./types";

const CHAR = "a young woman with short black hair and a red scarf";

// A board is drawn from the words it pinned, not from a preset name, so these
// are what the prompt builders take.
const ANIME = lockStyle("anime");
const INK = lockStyle("ink");

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
    const out = panelPrompt(panel({ subject: "a kitchen tap" }), ANIME, CHAR);
    expect(out).toContain("a kitchen tap");
  });

  it("carries a note, and puts it last so it can correct what came before", () => {
    // The repair for a wrong frame is a sentence about that frame, not a new
    // seed. It goes at the end because a correction has to follow the thing
    // it corrects.
    const out = panelPrompt(panel(), ANIME, CHAR, "older, grey at the temples");
    expect(out).toContain("older, grey at the temples");
    const noteAt = out.indexOf("older, grey");
    expect(noteAt).toBeGreaterThan(out.indexOf("red scarf"));
    expect(out.trimEnd().endsWith("temples.")).toBe(true);
  });

  it("is unchanged when there is no note", () => {
    // A panel nobody has complained about must produce exactly what it did
    // before notes existed, or every undrawn panel changes the day this ships.
    expect(panelPrompt(panel(), ANIME, CHAR, "")).toBe(
      panelPrompt(panel(), ANIME, CHAR));
    expect(panelPrompt(panel(), ANIME, CHAR, "   ")).toBe(
      panelPrompt(panel(), ANIME, CHAR));
    expect(panelPrompt(panel(), ANIME, CHAR, undefined)).toBe(
      panelPrompt(panel(), ANIME, CHAR));
  });

  it("names the character when they are in frame", () => {
    expect(panelPrompt(panel(), ANIME, CHAR)).toContain("red scarf");
  });

  it("leaves the character out of a panel they are not in", () => {
    // A close-up of a running tap, handed the protagonist's description,
    // draws her running instead of the tap.
    const out = panelPrompt(
      panel({ subject: "a kitchen tap", action: "water running",
              character_in_frame: false, shot: "close-up" }),
      ANIME, CHAR);
    expect(out).not.toContain("red scarf");
    expect(out).toContain("a kitchen tap");
    expect(out).toContain("close-up shot");
  });

  it("leads with the style, so every panel matches the others", () => {
    for (const s of STYLES) {
      expect(panelPrompt(panel(), lockStyle(s.id), CHAR).startsWith(s.words)).toBe(true);
    }
  });

  it("drops empty parts rather than leaving stray punctuation", () => {
    const out = panelPrompt(
      panel({ action: "", setting: "" }), ANIME, CHAR);
    expect(out).not.toContain("..");
    expect(out).not.toContain(". .");
    expect(out.endsWith(".")).toBe(true);
  });

  it("survives an empty character description", () => {
    const out = panelPrompt(panel(), ANIME, "   ");
    expect(out).toContain("Mira");
    expect(out).not.toContain(", .");
  });

  it("never leaves a board with no style at all", () => {
    // An unknown id used to resolve to the empty string, so every prompt lost
    // the one thing that makes a board a board while the run went on looking
    // like it had worked. It resolves to a real style now, and the caller is
    // expected to say so rather than let the substitution pass unseen.
    expect(styleWords("no-such-style")).toBe("");
    expect(isKnownStyle("no-such-style")).toBe(false);
    expect(isKnownStyle("anime")).toBe(true);

    const lock = lockStyle("no-such-style");
    expect(lock.id).toBe(STYLES[0].id);
    expect(lock.words).toBe(STYLES[0].words);
    expect(panelPrompt(panel(), lock, CHAR).startsWith(STYLES[0].words)).toBe(true);
  });

  it("leaves no stray punctuation when a lock carries no words", () => {
    // Not reachable through `lockStyle`, but a draft is JSON and can hold
    // anything. The framing still leads, because every panel is a panel
    // whatever it is drawn in.
    const out = panelPrompt(panel(), { ...ANIME, words: "" }, CHAR);
    expect(out.startsWith("cropped composition")).toBe(true);
    expect(out).not.toContain("..");
    expect(out).not.toMatch(/^[.,;\s]/);
  });

  it("asks for a panel, not a picture of the moment", () => {
    // Nothing used to say the picture was a panel, so each one came back as a
    // standalone illustration: centred, complete, and the opposite of what a
    // panel does. Pages looked like comics only because the compositor drew
    // the borders afterwards.
    const out = panelPrompt(panel(), ANIME, CHAR);
    expect(out).toContain("cropped composition");
    expect(out).toContain("full bleed");
    // Measured: naming the format draws the frame. "a single comic book panel,
    // sequential art" put a heavy black border inside the one the compositor
    // draws, and removing the negations did not stop it.
    expect(out).not.toContain("comic book panel");
    expect(out).not.toContain("sequential art");
    // Klein has no negative conditioning at guidance 1, so "no border" reads
    // as "border" and draws one -- inside the border the compositor then draws.
    // The way to not get a border is to never mention one.
    expect(out).not.toContain("no border");
    expect(out).not.toContain("no text");
    expect(out).not.toContain("no speech");
  });

  it("does not frame the reference sheets as panels", () => {
    // The sheet and the room are what every panel matches against. Framing
    // them as panels would crop and dramatise the one thing that has to stay
    // flat and neutral.
    expect(sheetPrompt(ANIME, CHAR)).not.toContain("cropped composition");
    expect(placePrompt(ANIME, "a parlour at dusk")).not.toContain("cropped composition");
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
      ANIME, CHAR);
    expect(out).toContain("faded linoleum");
    expect(out).not.toContain("a flat");
  });

  it("falls back to the terse fields when the scene is empty", () => {
    const out = panelPrompt(panel({ description: "" }), ANIME, CHAR);
    expect(out).toContain("stands in the hallway");
    expect(out).toContain("a narrow flat");
  });

  it("falls back when the scene is absent entirely", () => {
    const p = panel();
    delete (p as { description?: string }).description;
    expect(panelPrompt(p, ANIME, CHAR)).toContain("a narrow flat");
  });

  it("still leaves the character out of a panel they are not in", () => {
    const out = panelPrompt(
      panel({ character_in_frame: false, subject: "a kitchen tap",
              description: "Water running into a steel sink, grey daylight." }),
      ANIME, CHAR);
    expect(out).not.toContain("red scarf");
    expect(out).toContain("a kitchen tap");
  });
});

describe("sheetPrompt", () => {
  it("asks for a neutral reference, not a scene", () => {
    const out = sheetPrompt(ANIME, CHAR);
    expect(out).toContain("reference sheet");
    expect(out).toContain("plain background");
    expect(out).toContain(CHAR);
  });

  it("builds one labelled lineup when the story has several major characters", () => {
    const out = sheetPrompt(INK, "Mira: red scarf\nJon: blue coat");
    expect(out).toContain("Cast reference lineup");
    expect(out).toContain("Mira: red scarf");
    expect(out).toContain("Jon: blue coat");
    expect(out).toContain("distinct silhouettes");
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

  it("hands nothing to a panel the character is not in", () => {
    // It used to fall back to the character sheet so the drawer always had an
    // image. A panel about a running tap, handed her sheet, draws her running
    // -- and that happens to every such panel as soon as the room fails to
    // draw. The panel is drawn from its prompt instead.
    const out = panelReferences(
      panel({ character_in_frame: false }), "sheet", null);
    expect(out).toEqual([]);
  });
});

describe("placePrompt", () => {
  it("asks for the room without anyone in it", () => {
    const out = placePrompt(INK, "faded blue walls, worn pine boards");
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

describe("a board keeps the words it was drawn with", () => {
  it("pins the words and the framing, not the preset name", () => {
    const lock = lockStyle("anime");
    expect(lock.words).toBe(STYLES[0].words);
    expect(lock.framing).toContain("cropped composition");
    expect(lock.version).toBe(STYLES[0].version);
  });

  it("draws from the pinned words even after the preset moves under it", () => {
    // This is the drift the lock exists to stop: the anime words were replaced
    // between v0.1.0 and v0.2.0, so a board half-drawn under one build asked
    // for something different in the other half under the next.
    const old = { ...lockStyle("anime"),
                  words: "flat cel-shaded anime illustration, clean linework, muted palette" };
    const out = panelPrompt(panel(), old, CHAR);
    expect(out.startsWith("flat cel-shaded anime illustration")).toBe(true);
    expect(out).not.toContain(STYLES[0].words);
    expect(sheetPrompt(old, CHAR).startsWith("flat cel-shaded")).toBe(true);
    expect(placePrompt(old, "a kitchen").startsWith("flat cel-shaded")).toBe(true);
  });

  it("reports that a preset has moved, without changing what is drawn", () => {
    expect(styleHasMoved(lockStyle("anime"))).toBe(false);
    expect(styleHasMoved({ ...lockStyle("anime"), words: "something else" })).toBe(true);
    expect(styleHasMoved({ ...lockStyle("anime"), framing: "other framing" })).toBe(true);
    expect(styleHasMoved({ ...lockStyle("anime"), id: "gone" })).toBe(true);
  });
});

describe("reviveLock", () => {
  it("keeps a well-formed lock exactly as it was stored", () => {
    const stored = { ...ANIME, words: "old anime words" };
    expect(reviveLock(stored)).toEqual(stored);
  });

  it("fills a missing framing rather than crashing the board", () => {
    // A draft is JSON. `panelPrompt` trims every part, and trimming undefined
    // throws -- which would take the whole tab down on load instead of
    // degrading to something drawable.
    const lock = reviveLock({ id: "anime", words: "old anime words" });
    expect(lock?.framing).toContain("cropped composition");
    expect(() => panelPrompt(panel(), lock!, CHAR)).not.toThrow();
    expect(panelPrompt(panel(), lock!, CHAR).startsWith("old anime words")).toBe(true);
  });

  it("keeps an id this build no longer has, so the drift is still reportable", () => {
    const lock = reviveLock({ id: "woodcut", words: "woodcut, heavy black" });
    expect(lock?.id).toBe("woodcut");
    expect(lock?.words).toBe("woodcut, heavy black");
    expect(styleHasMoved(lock!)).toBe(true);
  });

  it("refuses a lock with nothing to draw from", () => {
    for (const bad of [null, undefined, 0, "anime", {}, { id: "anime" },
                       { words: "" }, { words: "   " }]) {
      expect(reviveLock(bad)).toBe(null);
    }
  });
});
