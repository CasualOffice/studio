import { describe, expect, it, vi } from "vitest";
import { dropLandedIn, type DropRect } from "./shared";

// A drop zone as `getBoundingClientRect` reports one: CSS pixels, relative to
// the top left of the window, which is the same frame wry reports a drop in.
const zone = (over: Partial<DropRect> = {}): DropRect => ({
  left: 200, top: 240, right: 600, bottom: 440,
  width: 400, height: 200,
  ...over,
});

describe("dropLandedIn", () => {
  it("matches a point inside the zone", () => {
    expect(dropLandedIn(zone(), { x: 400, y: 300 })).toBe(true);
  });

  it("accepts the edges, because a drop on the border is a drop on the zone", () => {
    expect(dropLandedIn(zone(), { x: 200, y: 240 })).toBe(true);
    expect(dropLandedIn(zone(), { x: 600, y: 440 })).toBe(true);
  });

  it("rejects a point outside on any side", () => {
    expect(dropLandedIn(zone(), { x: 199, y: 300 })).toBe(false);
    expect(dropLandedIn(zone(), { x: 601, y: 300 })).toBe(false);
    expect(dropLandedIn(zone(), { x: 400, y: 239 })).toBe(false);
    expect(dropLandedIn(zone(), { x: 400, y: 441 })).toBe(false);
  });

  it("does not halve the position on a Retina display", () => {
    // The whole defect: the position was divided by the device pixel ratio,
    // so with a ratio of 2 this drop was tested at (200, 150) -- above the
    // zone -- and dragging an image in never worked on a Retina Mac. The
    // ratio must not enter into it at all, so reading it fails the test.
    let readRatio = false;
    vi.stubGlobal("window", {
      get devicePixelRatio() { readRatio = true; return 2; },
    });
    try {
      expect(dropLandedIn(zone(), { x: 400, y: 300 })).toBe(true);
      // And the halved point is genuinely outside, so the assertion above is
      // about the scaling rather than about a zone too big to miss.
      expect(dropLandedIn(zone(), { x: 200, y: 150 })).toBe(false);
      expect(readRatio).toBe(false);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("ignores a drop on a zero-sized zone, even at the origin", () => {
    // Hidden tabs keep their drop zones mounted and listening. A zero-sized
    // rect sits at (0, 0) and would otherwise claim a drop in the corner of
    // the window, which is how one drop landed in every tab at once.
    const hidden: DropRect = { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 };
    expect(dropLandedIn(hidden, { x: 0, y: 0 })).toBe(false);
    expect(dropLandedIn(zone({ width: 0 }), { x: 400, y: 300 })).toBe(false);
    expect(dropLandedIn(zone({ height: 0 }), { x: 400, y: 300 })).toBe(false);
  });
});
