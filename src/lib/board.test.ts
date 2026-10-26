import { describe, expect, it } from "vitest";
import { cameraWords, castSeed, castSheetPrompt, shotClass, compositions, drawBatch, CUSTOM_STYLE,
         lockCustomStyle, lockStyle, panelPrompt, panelReferences, panelSeed, placeKey, placeSeed,
         reviveLock, shotBackground, styleHasMoved, STYLES } from "./board";
import type { Panel, VaultItem } from "./types";

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

describe("panelReferences", () => {
  it("never hands a panel a previous frame, which it would edit into a copy", () => {
    // A one-reference model was given the last frame in place of the sheet,
    // on the edit route: each panel came back as the one before it.
    expect(panelReferences(panel(), "sheet", "room", 1)).toEqual(["sheet"]);
    expect(panelReferences(panel({ character_in_frame: false }), "sheet", "room", 1)).toEqual(["room"]);
  });

  it("hands a frame with somebody in it her sheet, and not the room", () => {
    // This asserted both references, on a measurement that was real but only
    // half the question: two references do keep consecutive panels in the same
    // room, and nobody checked what the room picture did to the person in
    // front of it. Measured again, same seed, same prompt, one reference
    // changed: with the kitchen attached the window frame was painted across
    // Mira's face, and across Jonas's in the next panel. Without it both came
    // back as ordinary panels -- and the kitchen was still right, because the
    // room is in the brief and prose cannot be composited onto a face.
    expect(panelReferences(panel(), "sheet", "room")).toEqual(["sheet"]);
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
    const out = panelPrompt(panel(), old);
    expect(out.startsWith("flat cel-shaded anime illustration")).toBe(true);
    expect(out).not.toContain(STYLES[0].words);
    expect(castSheetPrompt(old, "Mira", "", "").startsWith("flat cel-shaded")).toBe(true);
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
    expect(() => panelPrompt(panel(), lock!)).not.toThrow();
    expect(panelPrompt(panel(), lock!).startsWith("old anime words")).toBe(true);
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

describe("a look described in your own words", () => {
  it("is pinned exactly like a preset", () => {
    const lock = lockCustomStyle("  1950s newspaper strip, coarse halftone  ");
    expect(lock.id).toBe(CUSTOM_STYLE);
    expect(lock.words).toBe("1950s newspaper strip, coarse halftone");
    expect(lock.framing).toContain("cropped composition");
  });

  it("leads every prompt, the same way a preset does", () => {
    const lock = lockCustomStyle("soft pencil, no ink");
    expect(panelPrompt(panel(), lock).startsWith("soft pencil, no ink")).toBe(true);
    expect(castSheetPrompt(lock, "Mira", "", "").startsWith("soft pencil, no ink")).toBe(true);
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


describe("a composed chapter is exactly one composition", () => {
  /** A vault item with only the fields the grouping reads. */
  const page = (
    index: number, composition: string | null, at: string,
    over: Partial<VaultItem> = {},
  ): VaultItem => ({
    id: `${composition ?? "old"}-${index}`,
    kind: "image", name: `page-${index}.png`, mime: "image/png", bytes: 1,
    model: "composed", prompt: "", seed: 0, inputs: [],
    created_at: at, duration_ms: 0,
    project: "board", project_index: index, composition,
    ...over,
  } as VaultItem);

  /** How the library holds a project: page number, then time. */
  const inProjectOrder = (items: VaultItem[]) => [...items].sort(
    (a, b) => (a.project_index ?? 1e9) - (b.project_index ?? 1e9)
      || a.created_at.localeCompare(b.created_at));

  it("keeps two runs of the same board apart", () => {
    // The defect this exists for: composing twice used to export page one old,
    // page one new, page two old, page two new.
    const items = inProjectOrder([
      page(0, "first", "2026-01-01T10:00:00"),
      page(1, "first", "2026-01-01T10:01:00"),
      page(0, "second", "2026-02-02T10:00:00"),
      page(1, "second", "2026-02-02T10:01:00"),
    ]);
    const runs = compositions(items);
    expect(runs.map((r) => r.id)).toEqual(["first", "second"]);
    expect(runs.every((r) => r.pages.length === 2)).toBe(true);
  });

  it("puts the newest run last, so it is the one shown", () => {
    const items = inProjectOrder([
      page(0, "first", "2026-01-01T10:00:00"),
      page(0, "second", "2026-02-02T10:00:00"),
    ]);
    expect(compositions(items).at(-1)!.id).toBe("second");
  });

  it("keeps each run's pages in reading order", () => {
    const items = inProjectOrder([
      page(2, "one", "2026-01-01T10:02:00"),
      page(0, "one", "2026-01-01T10:00:00"),
      page(1, "one", "2026-01-01T10:01:00"),
    ]);
    expect(compositions(items)[0].pages.map((p) => p.project_index))
      .toEqual([0, 1, 2]);
  });

  it("does not let a run with fewer pages absorb the longer one", () => {
    // Three pages became two after the board was re-divided. The short run
    // must not inherit the third page of the long one.
    const items = inProjectOrder([
      page(0, "long", "2026-01-01T10:00:00"),
      page(1, "long", "2026-01-01T10:01:00"),
      page(2, "long", "2026-01-01T10:02:00"),
      page(0, "short", "2026-02-02T10:00:00"),
      page(1, "short", "2026-02-02T10:01:00"),
    ]);
    const runs = compositions(items);
    expect(runs.map((r) => r.pages.length)).toEqual([3, 2]);
  });

  it("groups pages composed before runs were recorded, without claiming them", () => {
    // Not members of the current composition merely because nothing
    // contradicts it. They are their own group, and the empty id is what the
    // interface labels as unknown.
    const items = inProjectOrder([
      page(0, null, "2026-01-01T10:00:00"),
      page(0, "now", "2026-02-02T10:00:00"),
    ]);
    const runs = compositions(items);
    expect(runs.map((r) => r.id)).toEqual(["", "now"]);
  });

  it("does not count the bound PDF as a page of the comic", () => {
    // It is stamped `composed` and sorts after every page, so it was the
    // comic's last page, exported as a PNG, and bound into the next PDF.
    const items = inProjectOrder([
      page(0, "one", "2026-01-01T10:00:00"),
      page(1, "one", "2026-01-01T10:01:00"),
      page(4294967295, "one", "2026-01-01T10:05:00",
           { kind: "document", mime: "application/pdf", name: "comic.pdf" }),
    ]);
    const runs = compositions(items);
    expect(runs).toHaveLength(1);
    expect(runs[0].pages.map((p) => p.mime)).toEqual(["image/png", "image/png"]);
  });

  it("ignores the panels and sheets the pages were made from", () => {
    const items = inProjectOrder([
      page(0, null, "2026-01-01T09:00:00", { model: "flux", composition: null }),
      page(0, "one", "2026-01-01T10:00:00"),
    ]);
    expect(compositions(items)).toEqual([
      { id: "one", pages: [expect.objectContaining({ model: "composed" })] },
    ]);
  });

  it("has nothing to show for a board that was never composed", () => {
    expect(compositions([])).toEqual([]);
  });
});

describe("drawBatch", () => {
  const died = (e: unknown) => String(e).includes("exited");
  const run = (outcomes: (true | string)[], stopped = () => false) => {
    const tried: number[] = [];
    const done = drawBatch(outcomes.map((_, i) => i), async (i) => {
      tried.push(i);
      const o = outcomes[i];
      return o === true ? { ok: true } : { ok: false, error: o };
    }, stopped, died);
    return done.then((end) => ({ end, tried }));
  };

  // A dead engine used to be restarted for every remaining panel, each one
  // loading the model again and dying the same way.
  it("stops at once when the engine dies", async () => {
    const { end, tried } = await run([true, "the engine process exited unexpectedly", true, true]);
    expect(end).toEqual({ end: "engine", error: "the engine process exited unexpectedly" });
    expect(tried).toEqual([0, 1]);
  });

  it("stops after two failures in a row, but not after two apart", async () => {
    const twice = await run(["bad prompt", "bad prompt", true]);
    expect(twice.end.end).toBe("repeated");
    expect(twice.tried).toEqual([0, 1]);
    const apart = await run(["bad prompt", true, "bad prompt", true]);
    expect(apart.end.end).toBe("done");
    expect(apart.tried).toEqual([0, 1, 2, 3]);
  });

  it("still honours Stop", async () => {
    let stop = false;
    const tried: number[] = [];
    const end = await drawBatch([0, 1, 2], async (i) => { tried.push(i); stop = true; return { ok: true }; },
      () => stop, died);
    expect(end.end).toBe("stopped");
    expect(tried).toEqual([0]);
  });
});


describe("director's camera in the prompt", () => {
  const cast = [{ character_id: "a", name: "Agnes", visual: "", expression: "", body_language: "" },
                { character_id: "n", name: "Narrator", visual: "", expression: "", body_language: "" }];
  it("leads with distance, angle, framing and blocking", () => {
    const words = cameraWords(panel({ shot: "MCU", angle: "low", framing: "ots", focus: "Agnes", cast,
      positions: { Agnes: "left third", Narrator: "right third" }, facing: { Agnes: "right", Narrator: "left" } }));
    expect(words[0]).toMatch(/medium close-up/);
    expect(words.some((w) => w.startsWith("low-angle shot, camera near the ground"))).toBe(true);
    expect(words).toContain("over-the-shoulder shot from behind Narrator, looking at Agnes");
    expect(words).toContain("Agnes at the left third of the frame, facing right");
  });
  it("puts the camera before the moment in a directed panel's prompt", () => {
    const prompt = panelPrompt(panel({ shot: "EWS", angle: "bird", framing: "establishing", cast: [],
      moment: "The harbour under low cloud" }), INK, "");
    expect(prompt.indexOf("extreme wide establishing shot")).toBeLessThan(prompt.indexOf("The harbour under low cloud"));
    expect(prompt).toContain("bird's-eye view");
  });
  it("lays a Peak or an establishing shot out as a full tier", () => {
    expect(shotClass({ shot: "CU", weight: 3 })).toBe("wide");
    expect(shotClass({ shot: "INSERT", weight: 1 })).toBe("close-up");
    expect(shotClass({ shot: "MCU" })).toBe("medium");
    expect(shotClass({ shot: "wide" })).toBe("wide");
  });
});


describe("the panel prompt is built from the brief", () => {
  const mira = { character_id: "m", name: "Mira", visual: "short black hair", expression: "grief",
                 body_language: "still", outfit: "a red wool coat" };
  it("draws the brief, never the raw source sentence, when there is one", () => {
    const out = panelPrompt(panel({ cast: [mira], moment: "My name is Mira Vale, though Jonas calls me Mi.",
                                    description: "Mira stands by the bench, the key in her palm" }), INK);
    expect(out).toContain("Mira stands by the bench, the key in her palm");
    expect(out).not.toContain("My name is");
  });

  it("falls back to the moment only for a board saved before briefs", () => {
    expect(panelPrompt(panel({ cast: [], moment: "The harbour under low cloud" }), INK))
      .toContain("The harbour under low cloud");
  });

  it("leads with the style and the framing, and carries a note last", () => {
    const out = panelPrompt(panel({ cast: [mira], description: "Mira waits" }), ANIME, "older, grey at the temples");
    expect(out.startsWith(ANIME.words)).toBe(true);
    expect(out).toContain("cropped composition");
    expect(out.trim().endsWith("older, grey at the temples.")).toBe(true);
  });

  it("tells the drawing to follow the full-body reference, costume included", () => {
    const out = panelPrompt(panel({ cast: [mira], description: "Mira waits" }), INK);
    expect(out).toContain("outfit of their reference");
    expect(out).toContain("wearing a red wool coat, as on their reference");
    expect(out).not.toContain("identity only");
  });

  it("says a costume change in the panel itself", () => {
    const out = panelPrompt(panel({ cast: [{ ...mira, appearance: "a soaked raincoat" }], description: "Mira waits" }), INK);
    expect(out).toContain("now wearing a soaked raincoat instead of the outfit on their reference");
  });

  it("names a lineup of references as full figures, left to right", () => {
    const jonas = { ...mira, character_id: "j", name: "Jonas", outfit: "" };
    const out = panelPrompt(panel({ cast: [mira, jonas], description: "They argue" }), INK);
    expect(out).toContain("The reference shows Mira, Jonas from left to right, full figures");
  });

  it("does not invite an offscreen speaker into the picture", () => {
    expect(panelPrompt(panel({ cast: [], dialogue: [{ speaker: "The guard", text: "Last train!", offscreen: true }] }), INK))
      .toContain("No people visible in frame");
  });

  it("keeps the setting behind a close-up instead of contradicting it", () => {
    const close = panelPrompt(panel({ shot: "CU", cast: [mira], setting: "station waiting room",
                                      location_state: "benches, a locked door" }), INK);
    expect(close).toContain("Setting: station waiting room");
    expect(close).toContain("soft, out-of-focus background");
    expect(close).not.toContain("no scenery");
    expect(close).not.toContain("Current location state");
    expect(panelPrompt(panel({ shot: "WS", cast: [mira], location_state: "benches" }), INK))
      .toContain("Current location state: benches");
  });

  it("asks for the whole place only in a wide shot", () => {
    expect(shotBackground("wide")).toContain("the place itself in frame");
    expect(shotBackground("MS")).toContain("the setting behind the figure");
    expect(shotBackground("establishing")).toBe(shotBackground("medium"));
  });
});

describe("the cast sheet", () => {
  it("is one whole figure in the chapter's outfit", () => {
    const out = castSheetPrompt(INK, "Mira", "short black hair, thin", "a red wool coat");
    expect(out).toContain("full body, head to toe");
    expect(out).toContain("Mira: short black hair, thin");
    expect(out).toContain("Wearing a red wool coat");
    expect(out).not.toContain("No full-body outfit");
    expect(out).toContain("no lettering");
  });

  it("still dresses someone the story never dresses", () => {
    expect(castSheetPrompt(INK, "Jonas", "", "")).toContain("one plain everyday outfit");
  });

  it("is seeded by who it is, and moves on when drawn again", () => {
    expect(castSeed("mira")).toBe(castSeed("mira"));
    expect(castSeed("mira")).not.toBe(castSeed("jonas"));
    expect(castSeed("mira", 1)).not.toBe(castSeed("mira"));
  });
});
