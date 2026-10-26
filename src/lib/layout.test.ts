import { describe, expect, it } from "vitest";
import { INNER_ASPECT, layoutBoard, pageCells, paginate, panelSize, speakerSide, tiers,
         type ShotClass } from "./layout";
import type { Panel } from "./types";

/**
 * The compositor's own fallback layout (worker.py `_page_cells`) is checked
 * against the same file, so the shape a panel is drawn at and the cell it is
 * set in come from one rule.
 */
const CASES = JSON.parse(Object.values(import.meta.glob("../../engine/fixtures/layout-cases.json", {
  query: "?raw", import: "default", eager: true,
}) as Record<string, string>)[0]) as { shots: ShotClass[]; rtl: boolean; cells: number[][] }[];

describe("the page layout", () => {
  it("agrees with the compositor cell for cell", () => {
    expect(CASES.length).toBeGreaterThan(10);
    for (const c of CASES) {
      const got = pageCells(c.shots, c.rtl);
      got.forEach((cell, i) => cell.forEach((v, k) => expect(v).toBeCloseTo(c.cells[i][k], 4)));
    }
  });

  it("puts a wide shot alone and two tight shots together", () => {
    expect(tiers(["wide", "medium", "close-up"])).toEqual([[0], [1, 2]]);
    expect(tiers(["medium", "medium", "medium"])).toEqual([[0, 1], [2]]);
    expect(tiers(Array(6).fill("wide"))).toHaveLength(4);
  });

  it("fills one fixed page shape, so every page of a board matches", () => {
    for (const shots of [["wide"], ["medium", "close-up"], Array(5).fill("medium")] as ShotClass[][]) {
      const cells = pageCells(shots);
      expect(Math.max(...cells.map(([, y, , h]) => y + h))).toBeCloseTo(1, 5);
    }
  });

  it("runs manga right to left", () => {
    const [first, second] = pageCells(["medium", "medium"], true);
    expect(first[0]).toBeGreaterThan(second[0]);
  });
});

describe("drawing at the cell's shape", () => {
  it("draws each panel at its cell's aspect, in multiples of 16, within the measured memory", () => {
    const panel = (shot: Panel["shot"], weight = 2, scene = 1) => ({ shot, weight, scene });
    const placed = layoutBoard([panel("EWS", 3), panel("MS"), panel("CU"), panel("MCU")], "comic", "page");
    for (const p of placed) {
      const { width, height } = panelSize(p.aspect, 768);
      expect(width % 16).toBe(0);
      expect(height % 16).toBe(0);
      expect(width * height).toBeLessThanOrEqual(768 * 768);
      expect(width / height).toBeCloseTo(p.aspect, 0);
    }
    // The establishing panel takes a wide cell and is drawn wide, not square.
    expect(placed[0].aspect).toBeGreaterThan(1.5);
    const wide = panelSize(placed[0].aspect, 768);
    expect(wide.width).toBeGreaterThan(wide.height);
  });

  it("works out the aspect from the cell and the page shape", () => {
    const [place] = layoutBoard([{ shot: "MS", weight: 2, scene: 1 }], "comic", "page");
    expect(place.aspect).toBeCloseTo(INNER_ASPECT, 5);
  });

  it("draws a scroll's panels at their shot's own shape", () => {
    const placed = layoutBoard([{ shot: "EWS", weight: 3, scene: 1 }, { shot: "CU", weight: 2, scene: 1 }],
      "webtoon", "strip");
    expect(placed[0].aspect).toBeGreaterThan(placed[1].aspect);
  });
});

describe("pages", () => {
  it("holds about a format's panels a page and breaks at a scene change", () => {
    expect(paginate([1, 1, 1, 1, 1, 1, 1, 1], 4)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
    expect(paginate([1, 1, 2, 2, 2], 4)).toEqual([0, 0, 1, 1, 1]);
  });

  it("does not break a page at a scene change that would leave it nearly empty", () => {
    expect(paginate([1, 2, 2, 2], 4)).toEqual([0, 0, 0, 0]);
  });

  it("never leaves one panel alone on the last page", () => {
    expect(paginate([1, 1, 1, 1, 1], 4)).toEqual([0, 0, 0, 0, 0]);
  });
});

describe("where a balloon goes", () => {
  const panel = { cast: [{ character_id: "m", name: "Mira", visual: "", expression: "", body_language: "" }],
                  positions: { Mira: "right third" } };
  it("sits on the speaker's side when they are in frame", () => {
    expect(speakerSide(panel, "Mira")).toBe("right");
    expect(speakerSide({ ...panel, positions: { Mira: "far left" } }, "Mira")).toBe("left");
  });
  it("has no side for a voice off panel", () => {
    expect(speakerSide(panel, "The guard")).toBe("off");
    expect(speakerSide(panel, "Mira", true)).toBe("off");
  });
});
