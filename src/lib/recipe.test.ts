import { describe, expect, it } from "vitest";
import { editKindFor, isOutpaintFill, OUTPAINT_FILLS, parseOutpaintPadding } from "./recipe";

describe("editKindFor", () => {
  it("leaves the control alone for an item that recorded no mode", () => {
    // Every item in the vault from before the run details were recorded looks
    // like this. Guessing a kind for them would change what Reuse does to
    // years of existing library items.
    expect(editKindFor(undefined, undefined)).toBeNull();
    expect(editKindFor(null, null)).toBeNull();
  });

  it("leaves the control alone for a plain generation", () => {
    // A text-to-image run sends no i2i_mode at all.
    expect(editKindFor(null, null)).toBeNull();
  });

  it("recognises a reinterpret, which the engine names outright", () => {
    expect(editKindFor("latent", null)).toBe("latent");
  });

  it("recognises an expand by its padding, which nothing else sets", () => {
    expect(editKindFor("edit", "0%,25%,0%,25%")).toBe("expand");
  });

  it("does not call an empty padding string an expand", () => {
    // The engine-side filter drops a blank padding before the run, so an item
    // carrying one was never actually outpainted.
    expect(editKindFor("edit", "")).toBe("instruct");
    expect(editKindFor("edit", "   ")).toBe("instruct");
  });

  it("lands a masked edit on instruct rather than an unpaintable mask tab", () => {
    // A mask is deleted the moment its run ends, so "mask" would restore to a
    // paint tab with nothing painted and a Run button that refuses to start.
    // Same engine route, so instruct reproduces the picture either way.
    expect(editKindFor("edit", null)).toBe("instruct");
  });

  it("ignores a padding that came with a mode it cannot belong to", () => {
    // Defensive: latent never carries padding, and if some future writer pairs
    // them the mode is the more specific fact.
    expect(editKindFor("latent", "10%,10%,10%,10%")).toBe("latent");
  });

  it("refuses to guess at a mode it does not know", () => {
    expect(editKindFor("inpaint", null)).toBeNull();
  });
});

describe("parseOutpaintPadding", () => {
  it("reads back exactly what the sliders wrote", () => {
    expect(parseOutpaintPadding("0%,25%,0%,25%"))
      .toEqual({ top: 0, right: 25, bottom: 0, left: 25 });
  });

  it("keeps the sides in CSS order rather than alphabetical", () => {
    // top, right, bottom, left. Getting this wrong mirrors an expand and is
    // invisible in any test that uses a symmetric padding.
    expect(parseOutpaintPadding("10%,20%,30%,40%"))
      .toEqual({ top: 10, right: 20, bottom: 30, left: 40 });
  });

  it("honours the one, two and three value shorthands", () => {
    expect(parseOutpaintPadding("15%"))
      .toEqual({ top: 15, right: 15, bottom: 15, left: 15 });
    expect(parseOutpaintPadding("10%, 20%"))
      .toEqual({ top: 10, right: 20, bottom: 10, left: 20 });
    expect(parseOutpaintPadding("10%,20%,30%"))
      .toEqual({ top: 10, right: 20, bottom: 30, left: 20 });
  });

  it("has nothing to restore for an item that recorded no padding", () => {
    expect(parseOutpaintPadding(null)).toBeNull();
    expect(parseOutpaintPadding(undefined)).toBeNull();
    expect(parseOutpaintPadding("")).toBeNull();
  });

  it("rejects pixel padding instead of restoring a wrong percentage", () => {
    // mflux accepts pixels; the sliders are percentages of the original, so
    // there is no honest slider position for "64px" and the default is the
    // less misleading answer.
    expect(parseOutpaintPadding("64px,64px,64px,64px")).toBeNull();
    expect(parseOutpaintPadding("25")).toBeNull();
  });

  it("rejects a value the sliders could not have produced", () => {
    // Clamping would present a number the run never used as if it were the
    // one it did.
    expect(parseOutpaintPadding("0%,250%,0%,0%")).toBeNull();
    expect(parseOutpaintPadding("-10%,0%,0%,0%")).toBeNull();
  });

  it("rejects malformed strings rather than half-reading them", () => {
    expect(parseOutpaintPadding("10%,,10%")).toBeNull();
    expect(parseOutpaintPadding("10%,20%,30%,40%,50%")).toBeNull();
    expect(parseOutpaintPadding("a lot")).toBeNull();
  });
});

describe("isOutpaintFill", () => {
  it("accepts every fill the dropdown offers", () => {
    for (const f of OUTPAINT_FILLS) expect(isOutpaintFill(f.id)).toBe(true);
  });

  it("rejects a fill this build dropped, so the dropdown never blanks", () => {
    expect(isOutpaintFill("mirror")).toBe(false);
    expect(isOutpaintFill(null)).toBe(false);
    expect(isOutpaintFill(undefined)).toBe(false);
    expect(isOutpaintFill("")).toBe(false);
  });
});
