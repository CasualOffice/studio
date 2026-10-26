import { describe, it, expect } from "vitest";

/**
 * The board's two repairs must stay two repairs.
 *
 * A note and a redraw reconsider the whole frame: the seed moves, every
 * reference is read again, and a panel that was right except for one hand
 * comes back with a different face. That is the correct move when the panel is
 * wrong and the wrong one when it is nearly right, and for a long time it was
 * the only move the board had -- it sent `mask: null` on every request while
 * the engine, the Rust command and Studio all supported masking.
 *
 * These read the source, in the manner of `hookOrder.test.ts`, because there is
 * no harness in this project that renders a component. Crude, and it is the
 * check that holds the distinction in place: a masked repair that quietly
 * stopped sending its mask would still type-check, still run, and still return
 * a picture -- just the whole frame redrawn, which is the thing it exists not
 * to do.
 */
const SOURCE = Object.values(
  import.meta.glob("../components/Storyboard.tsx", {
    query: "?raw", import: "default", eager: true,
  }) as Record<string, string>,
)[0];

/** The body of one `const <name> = (...) => { ... }` in the component. */
function fn(name: string): string {
  const at = [`const ${name} = async (`, `const ${name} = (`]
    .map((decl) => SOURCE.indexOf(decl))
    .find((i) => i > -1) ?? -1;
  expect(at, `${name} is declared`).toBeGreaterThan(-1);
  const open = SOURCE.indexOf("{", at);
  let depth = 0;
  for (let i = open; i < SOURCE.length; i++) {
    if (SOURCE[i] === "{") depth++;
    else if (SOURCE[i] === "}" && --depth === 0) return SOURCE.slice(open, i + 1);
  }
  throw new Error(`${name} is not balanced`);
}

describe("the board repairs a region without redrawing the frame", () => {
  it("has a repair that is separate from redraw", () => {
    expect(SOURCE).toContain("const repair = async (");
    expect(SOURCE).toContain("const redraw = async (");
  });

  it("sends the painted mask, not null", () => {
    const body = fn("repair");
    expect(body).toContain("mask: maskId");
    expect(body).not.toContain("mask: null");
  });

  it("repairs the panel's own picture, not the cast sheet", () => {
    // A masked edit regenerates inside the painted region of the image it is
    // handed. Handed the reference sheet instead, it would paint over the
    // sheet and return something that is not this panel at all.
    expect(fn("repair")).toContain("images: [current]");
  });

  it("leaves the whole-frame redraw unmasked", () => {
    // `renderPanel` draws a panel from its references; there is nothing to
    // mask. A mask arriving here would confine a first draw to a painted region.
    expect(fn("renderPanel")).toContain("mask: null");
  });

  it("removes the mask from the vault once the engine has read it", () => {
    // Working material, not a library item, and it is a picture of what the
    // user wanted changed about their own manuscript.
    const body = fn("repair");
    expect(body).toContain("api.vaultDelete(maskId)");
    expect(body).toContain("} finally {");
  });

  it("files the previous picture before overwriting the panel", () => {
    // `putArt` is the write, since the board keeps its work by panel identity
    // rather than in an array indexed by position.
    for (const name of ["repair", "drawOne"]) {
      const body = fn(name);
      const filed = body.indexOf("supersede(");
      const replaced = body.indexOf("putArt(");
      expect(filed, `${name} files the old picture`).toBeGreaterThan(-1);
      expect(replaced, `${name} writes the new one`).toBeGreaterThan(-1);
      expect(filed, `${name} files it before replacing it`).toBeLessThan(replaced);
    }
  });

  it("writes a picture against the panel, never against a position", () => {
    // The defect the document exists for. An index written into a shared
    // array is correct only until something moves.
    for (const name of ["repair", "drawOne", "restoreVariant"]) {
      expect(fn(name), `${name} writes by position`)
        .not.toMatch(/setDrawn\(|setRedraws\(|setVariants\(/);
    }
  });

  it("retires the composed pages when a panel changes", () => {
    // A page is only as current as the panels in it.
    for (const name of ["repair", "drawOne", "restoreVariant"]) {
      expect(fn(name)).toContain("retirePages()");
    }
  });
});

describe("the board edits a panel without destroying its picture", () => {
  it("decides staleness in one place rather than at the call site", () => {
    // Every edit used to retire the picture, including a caption fix, so
    // correcting a comma cost a five-minute render and came back with a
    // different face. The rule is `staleness` in boardDoc and is tested there.
    const body = fn("edit");
    expect(body).toContain("editPanel(d, panel.id, patch)");
    expect(body).toContain("stale.art");
    expect(body).not.toMatch(/setDrawn\(|next\[i\] = null/);
  });

  it("retires the pages on any edit and the art only on some", () => {
    const body = fn("edit");
    const art = body.indexOf("if (stale.art)");
    const pages = body.indexOf("if (stale.pages)");
    expect(art).toBeGreaterThan(-1);
    expect(pages).toBeGreaterThan(-1);
  });

  it("moves a panel by identity, through the tested operations", () => {
    for (const [name, op] of [
      ["movePanel", "movePanelIn"], ["addPanelAfter", "insertPanel"],
      ["deletePanel", "removePanel"], ["splitPanel", "splitPanelIn"],
      ["mergePanel", "mergePanelIn"],
    ]) {
      expect(fn(name), `${name} uses ${op}`).toContain(op);
    }
  });

  it("renders the panel list keyed by identity", () => {
    // React reuses a row by key. Keyed by position, a reordered board reused
    // the wrong row for the panel now standing there.
    expect(SOURCE).toContain("key={p.id}");
  });
});
