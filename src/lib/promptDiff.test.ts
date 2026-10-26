import { describe, expect, it } from "vitest";
import { diffWords, lostWords, rebuild } from "./promptDiff";

const added = (original: string, variant: string) =>
  diffWords(original, variant).filter((t) => t.added).map((t) => t.text);

describe("diffWords marks what the variant added", () => {
  it("marks new words and leaves the request's own", () => {
    expect(added("a cat on a chair", "A grey cat sits on a wooden chair."))
      .toEqual(["grey", "sits", "wooden"]);
  });

  it("is not fooled by inflection or punctuation", () => {
    expect(added("2 red cars", "Two red car, side by side.")).toEqual(["side", "side."]);
    expect(added("a man runs", "A man running.")).toEqual([]);
  });

  it("keeps the text exactly, whitespace and all", () => {
    const variant = "A grey cat,  on a chair.";
    expect(diffWords("a cat", variant).map((t) => t.text).join("")).toBe(variant);
  });
});

describe("lostWords names what the variant no longer says", () => {
  it("flags a lost word but not filler", () => {
    expect(lostWords("a red car with no sunroof", "A red car.")).toEqual(["no", "sunroof"]);
  });

  it("does not count what the cleaner removed as lost", () => {
    expect(lostWords("a masterpiece 8k cat", "A cat.", ["masterpiece", "8k"])).toEqual([]);
  });
});

describe("rebuild leaves additions out cleanly", () => {
  const text = "A grey cat sits on an old wooden chair by a window. Watercolour illustration.";

  it("returns the text untouched with nothing left out", () => {
    expect(rebuild(text, [])).toBe(text);
  });

  it("removes a phrase and tidies the seam", () => {
    expect(rebuild(text, ["by a window"]))
      .toBe("A grey cat sits on an old wooden chair. Watercolour illustration.");
  });

  it("removes a trailing style sentence", () => {
    expect(rebuild(text, ["Watercolour illustration"]))
      .toBe("A grey cat sits on an old wooden chair by a window.");
  });

  it("mends the article a removed word leaves behind", () => {
    expect(rebuild(text, ["old"])).toContain("on a wooden chair");
    expect(rebuild("A cat on a wooden old chair.", ["wooden"])).toBe("A cat on an old chair.");
  });

  it("closes a comma left with nothing between", () => {
    expect(rebuild("A cat, with grey fur, sits by the fire.", ["with grey fur"]))
      .toBe("A cat, sits by the fire.");
  });

  it("recapitalises when the first phrase goes", () => {
    expect(rebuild("Soft light falls on a cat.", ["Soft light falls on"]))
      .toBe("A cat.");
  });
});
