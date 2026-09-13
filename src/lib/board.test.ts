import { describe, expect, it } from "vitest";
import { CHAIN_LIMIT, CUSTOM_STYLE, isKnownStyle, lockCustomStyle, lockStyle,
         panelPrompt, previousPanel,
         panelReferences, panelSeed, placeKey, placePrompt, placeSeed,
         reviveLock, sheetPrompt, shotBackground, styleHasMoved, styleWords,
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

describe("the cast is described once, and the subject is not a garment", () => {
  // Read off a real run. The subject was joined to the cast description with
  // ", ", so a panel of Mira asked for "... oversized grey wool coat, dark
  // green scarf, Mira" -- the name arriving in the middle of a list of clothes,
  // and drawn as one more of them.
  const MIRA = "Mira: young woman, short dark hair, thin face, "
             + "oversized grey wool coat, dark green scarf";
  const BOTH = `${MIRA}; Jonas: tall, stooped, grey overcoat, flat cap`;
  const count = (text: string, needle: string) => text.split(needle).length - 1;

  it("does not weld the subject's name onto the clothing", () => {
    const out = panelPrompt(
      panel({ subject: "Mira", action: "finds letter under sugar tin",
              setting: "a narrow kitchen" }), ANIME, MIRA);
    expect(out).not.toContain("scarf, Mira");
    expect(count(out, "Mira")).toBe(1);
    expect(out).toContain("dark green scarf");
    expect(out).toContain("finds letter under sugar tin");
  });

  it("names a second person once as well", () => {
    // The multi-character cast is written "Mira: ...; Jonas: ...", so the names
    // are what stands before each colon.
    const out = panelPrompt(panel({ subject: "Jonas" }), ANIME, BOTH);
    expect(count(out, "Jonas")).toBe(1);
    expect(out).not.toContain("flat cap, Jonas");
    expect(out).toContain("dark green scarf");
    expect(out).toContain("grey overcoat");
  });

  it("matches a name whatever its case or trailing punctuation", () => {
    for (const subject of ["mira", "MIRA", "Mira."]) {
      expect(count(panelPrompt(panel({ subject }), ANIME, MIRA), "ira")).toBe(1);
    }
  });

  it("still says a subject the cast does not name", () => {
    // Not everyone in a panel is in the cast description, and dropping a name
    // the prompt is the only statement of would lose the person entirely.
    const out = panelPrompt(panel({ subject: "the postman" }), ANIME, MIRA);
    expect(out).toContain("the postman");
  });

  it("gives a thing in frame a clause of its own, not a slot in the wardrobe", () => {
    const out = panelPrompt(
      panel({ subject: "a kitchen tap", action: "water running" }), ANIME, MIRA);
    expect(out).toContain("a kitchen tap");
    expect(out).not.toContain("scarf, a kitchen tap");
    expect(out).toContain("dark green scarf. a kitchen tap.");
  });

  it("still leaves the whole cast out of a panel nobody is in", () => {
    const out = panelPrompt(
      panel({ subject: "a kitchen tap", character_in_frame: false }), ANIME, BOTH);
    expect(out).not.toContain("dark green scarf");
    expect(out).not.toContain("Jonas");
    expect(out).toContain("a kitchen tap");
  });
});

describe("the prompt does not contradict or repeat itself", () => {
  it("states the shot once, even when the brief says it again", () => {
    // A real brief came back as "Mira stands in the narrow kitchen, wide shot.
    // Pale green walls, ..." for a panel whose prompt already said "wide shot".
    // Said twice it is noise; said differently by the writer and the enricher
    // it asks for two distances at once.
    const out = panelPrompt(
      panel({ description: "Mira stands in the narrow kitchen, wide shot. "
                         + "Pale green walls, scuffed lino." }), ANIME, CHAR);
    expect(out.split("wide shot").length - 1).toBe(1);
    expect(out).toContain("Mira stands in the narrow kitchen. Pale green walls");
  });

  it("takes the echo out of the front of a brief too", () => {
    const out = panelPrompt(
      panel({ shot: "close-up",
              description: "Close-up shot. Her hands on the sugar tin." }),
      ANIME, CHAR);
    expect(out).toContain("Her hands on the sugar tin");
    expect(out.split(" shot").length - 1).toBe(1);
    expect(out).not.toContain("Close-up shot.");
  });

  it("leaves a shot named mid-sentence alone rather than mangling the grammar", () => {
    // Cutting it out here would take the grammar with it: "a of her hands".
    const out = panelPrompt(
      panel({ description: "A close-up shot of her hands, steam rising." }),
      ANIME, CHAR);
    expect(out).toContain("A close-up shot of her hands, steam rising");
  });

  it("falls back to the terse fields when the brief was only the shot", () => {
    const out = panelPrompt(panel({ description: "Wide shot." }), ANIME, CHAR);
    expect(out).toContain("stands in the hallway");
    expect(out).toContain("a narrow flat");
  });

  it("does not double the full stop the brief already ends with", () => {
    // The parts are joined with ". " and a brief is whole sentences, so this
    // used to read "dim light from a corner lamp.. plain ground behind ...".
    const out = panelPrompt(
      panel({ shot: "close-up",
              description: "Her hands, lit by a corner lamp." }), ANIME, CHAR);
    expect(out).not.toContain("..");
    expect(out).toContain("lit by a corner lamp. plain ground");
  });

  it("does not run the next clause on from a part ending in a comma", () => {
    const out = panelPrompt(
      panel({ description: "A narrow hallway, faded linoleum," }), ANIME, CHAR);
    expect(out).not.toContain("linoleum, the place itself");
    expect(out).toContain("faded linoleum. the place itself");
  });

  it("lets the shot have the last word on how much place is in frame", () => {
    // The enrichment stage lists the walls, the floor and the light whatever
    // the shot is, so a close-up asked for plain ground and then listed the
    // room anyway, a clause later in the same prompt. The brief cannot be
    // rewritten from here, but the distance can be settled after it.
    const out = panelPrompt(
      panel({ shot: "close-up",
              description: "Pale green walls, scuffed lino, a bare bulb." }),
      ANIME, CHAR);
    expect(out.indexOf("no scenery")).toBeGreaterThan(out.indexOf("bare bulb"));
    // And the same order without a brief, so there is one rule, not two.
    const terse = panelPrompt(panel({ shot: "medium" }), ANIME, CHAR);
    expect(terse.indexOf("kept simple")).toBeGreaterThan(terse.indexOf("a narrow flat"));
  });
});

describe("sheetPrompt", () => {
  it("asks for a neutral reference, not a scene", () => {
    const out = sheetPrompt(ANIME, CHAR);
    expect(out).toContain("reference sheet");
    expect(out).toContain("plain background");
    expect(out).toContain(CHAR);
  });

  it("builds one sheet for the whole cast, and says how many there are", () => {
    // Asked for a lineup of "each named character", the drawer put three
    // figures on a sheet for a cast of two -- and that invented person is then
    // the reference every panel of the board is drawn against. Stating the
    // count is what fixes it.
    const out = sheetPrompt(INK, "Mira: red scarf\nJon: blue coat");
    expect(out).toContain("two people");
    expect(out).toContain("Mira: red scarf");
    expect(out).toContain("Jon: blue coat");
    expect(out).toContain("distinct silhouettes");
  });

  it("does not ask for the names to be written on the sheet", () => {
    // A diffusion model cannot letter. Asking for a "labelled" lineup printed
    // "Referace Mekwarird" across the one image every frame is conditioned
    // against. The names bind identity in the description, not in ink.
    const out = sheetPrompt(INK, "Mira: red scarf\nJon: blue coat");
    expect(out).not.toContain("labelled");
    expect(out).not.toContain("named");
  });

  it("counts a larger cast correctly", () => {
    const four = sheetPrompt(INK, "A: a\nB: b\nC: c\nD: d");
    expect(four).toContain("four people");
  });
});

describe("continuity between panels", () => {
  const scene = (n: number): Panel => panel({ scene: n });

  it("carries the previous panel forward inside a scene", () => {
    // The board had no continuity at all: every panel was drawn from the
    // sheet and the room, so the light, the weather and the state of the
    // character's clothes reset between one panel and the next.
    const panels = [scene(1), scene(1), scene(1)];
    expect(previousPanel(1, panels, ["a", null, null])).toBe("a");
  });

  it("does not carry across a scene change", () => {
    // The panel before a scene change is the wrong room.
    const panels = [scene(1), scene(2)];
    expect(previousPanel(1, panels, ["a", null])).toBeNull();
  });

  it("re-anchors to the sheet at the chain limit", () => {
    // Every generation is a lossy copy. Chained without limit, the character
    // is N copies from the sheet that defined her by panel N.
    const panels = Array.from({ length: 9 }, () => scene(1));
    const drawn = panels.map((_, i) => `p${i}`);
    expect(previousPanel(CHAIN_LIMIT, panels, drawn)).toBeNull();
    expect(previousPanel(CHAIN_LIMIT + 1, panels, drawn)).toBe(`p${CHAIN_LIMIT}`);
  });

  it("skips a panel that has not been drawn", () => {
    const panels = [scene(1), scene(1), scene(1)];
    expect(previousPanel(2, panels, ["a", null, null])).toBe("a");
  });

  it("identity wins when only one reference fits", () => {
    // Most models in the catalogue take exactly one. A character whose face
    // changes between panels is the failure everyone sees first.
    const refs = panelReferences(panel(), "sheet", "room", "previous", 1);
    expect(refs).toEqual(["sheet"]);
  });

  it("orders identity, then continuity, then the room", () => {
    const refs = panelReferences(panel(), "sheet", "room", "previous", 3);
    expect(refs).toEqual(["sheet", "previous", "room"]);
  });

  it("gives a panel without the character its continuity instead", () => {
    const refs = panelReferences(
      panel({ character_in_frame: false }), "sheet", "room", "previous", 1);
    expect(refs).toEqual(["previous"]);
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

describe("how much place a panel asks for", () => {
  // A comic page establishes a location once and then lives inside it. Every
  // panel here used to ask for the walls, the floor, the furniture and the
  // light, so a close-up of a face came back as a face in a furnished room --
  // and the room was re-invented from prose each time, which is why it
  // drifted. The shot decides how much of the place is in frame.
  it("asks for the whole place only in a wide shot", () => {
    const wide = panelPrompt(panel({ shot: "wide" }), ANIME, CHAR);
    expect(wide).toContain("the place itself in frame");
  });

  it("keeps the background simple behind a medium shot", () => {
    const mid = panelPrompt(panel({ shot: "medium" }), ANIME, CHAR);
    expect(mid).toContain("only what stands directly behind the figure");
    expect(mid).not.toContain("the place itself in frame");
  });

  it("asks for no scenery at all in a close-up", () => {
    const close = panelPrompt(panel({ shot: "close-up" }), ANIME, CHAR);
    expect(close).toContain("no scenery");
    expect(close).not.toContain("the place itself in frame");
  });

  it("still leads with the style and the framing", () => {
    for (const shot of ["wide", "medium", "close-up"] as const) {
      const out = panelPrompt(panel({ shot }), ANIME, CHAR);
      expect(out.startsWith(ANIME.words)).toBe(true);
      expect(out).toContain("cropped composition");
      expect(out).toContain(`${shot} shot`);
    }
  });

  it("falls back to the medium treatment for an unknown shot", () => {
    expect(shotBackground("establishing")).toBe(shotBackground("medium"));
  });

  it("does not hand a close-up the room reference", () => {
    // A picture of the room is an instruction, and at this distance the room
    // is not in frame: it pulls the shot wider to fit the furniture in, which
    // is the opposite of what a close-up is for.
    const close = panel({ shot: "close-up", character_in_frame: true });
    expect(panelReferences(close, "sheet-id", "place-id")).toEqual(["sheet-id"]);
  });

  it("still hands the room to the shots that show it", () => {
    for (const shot of ["wide", "medium"] as const) {
      expect(panelReferences(panel({ shot, character_in_frame: true }),
                             "sheet-id", "place-id"))
        .toEqual(["sheet-id", "place-id"]);
    }
  });

  it("leaves a close-up she is not in with no reference at all", () => {
    const close = panel({ shot: "close-up", character_in_frame: false });
    expect(panelReferences(close, "sheet-id", "place-id")).toEqual([]);
  });
});

describe("a look described in your own words", () => {
  it("is pinned exactly like a preset", () => {
    const lock = lockCustomStyle("  1950s newspaper strip, coarse halftone  ");
    expect(lock.id).toBe(CUSTOM_STYLE);
    expect(lock.words).toBe("1950s newspaper strip, coarse halftone");
    expect(lock.framing).toContain("cropped composition");
  });

  it("leads every prompt, the same way a preset does", () => {
    const lock = lockCustomStyle("soft pencil, no ink");
    expect(panelPrompt(panel(), lock, CHAR).startsWith("soft pencil, no ink")).toBe(true);
    expect(sheetPrompt(lock, CHAR).startsWith("soft pencil, no ink")).toBe(true);
    expect(placePrompt(lock, "a kitchen").startsWith("soft pencil, no ink")).toBe(true);
  });

  it("is never reported as having drifted", () => {
    // There is no preset behind it for it to drift from.
    expect(styleHasMoved(lockCustomStyle("soft pencil"))).toBe(false);
  });

  it("survives a restart", () => {
    const lock = lockCustomStyle("woodcut, heavy black, no grey");
    expect(reviveLock(JSON.parse(JSON.stringify(lock)))).toEqual(lock);
  });
});

describe("one room per place, not per scene number", () => {
  it("gives the same place one key however the scenes are numbered", () => {
    // A story the writer leaves in scene 1 still changes rooms when it moves,
    // and coming back to the kitchen comes back to the same kitchen.
    expect(placeKey(panel({ setting: "the kitchen", scene: 1 })))
      .toBe(placeKey(panel({ setting: "Kitchen", scene: 5 })));
    expect(placeKey(panel({ setting: "her kitchen", scene: 9 })))
      .toBe(placeKey(panel({ setting: "the kitchen", scene: 1 })));
  });

  it("gives different places different keys inside one scene", () => {
    expect(placeKey(panel({ setting: "the kitchen", scene: 1 })))
      .not.toBe(placeKey(panel({ setting: "the station platform", scene: 1 })));
  });

  it("prefers the key the engine stamped", () => {
    expect(placeKey(panel({ setting: "anything", place_key: "iron roof" })))
      .toBe("iron roof");
  });

  it("falls back to the scene title, then to the scene", () => {
    expect(placeKey(panel({ setting: "", scene_title: "On the road", scene: 2 })))
      .toBe("on the road");
    expect(placeKey(panel({ setting: "", scene: 7 }))).toBe("scene 7");
  });

  it("draws one room from one seed wherever the story returns to it", () => {
    expect(placeSeed("kitchen")).toBe(placeSeed("kitchen"));
    expect(placeSeed("kitchen")).not.toBe(placeSeed("station platform"));
  });
});
