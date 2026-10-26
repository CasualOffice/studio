import type { BookFormat, Panel } from "./types";

/**
 * Where every panel goes on the page, decided before anything is drawn.
 *
 * Panels used to be drawn square and then cut to whatever cell the
 * compositor gave them: a wide establishing cell took 42% off the height of
 * a square drawing, which on a face is the top of the head and the chin. The
 * layout is now settled first, here, and each panel is drawn at its own
 * cell's shape (`panelSize`); the compositor sets it in exactly that cell.
 *
 * The engine keeps a copy of the tiering for callers that send no cells
 * (`_page_cells` in worker.py). Both read engine/fixtures/layout-cases.json
 * in their tests, so the two cannot drift apart unnoticed.
 */

/** A page's inner area, width over height: a comic page is about 2:3. */
export const INNER_ASPECT = 2 / 3;
/** The gutter between cells, as a fraction of the inner width. */
export const GUTTER = 0.02;
/** A tier holding one large panel is this much taller than the others. */
export const WIDE_TIER = 1.3;

/** Panels a page, by format: about five in a Western comic, four in manga. */
export const PANELS_PER_PAGE: Record<BookFormat, number> = {
  comic: 5, manga: 4, webtoon: 6, picture: 1,
};
/** The page count a new board is planned for. */
export const DEFAULT_PAGES = 4;

export type Cell = [number, number, number, number];

/** The three size classes a page is built from. */
export type ShotClass = "wide" | "medium" | "close-up";

/**
 * One page's panels as tiers, the way worker.py's `_tiers` does it: a wide
 * shot alone, two to a tier (three on a crowded page), never more than four
 * tiers.
 */
export function tiers(shots: ShotClass[]): number[][] {
  const per = shots.length <= 8 ? 2 : 3;
  const out: number[][] = [];
  let i = 0;
  while (i < shots.length) {
    if (shots[i] === "wide") { out.push([i]); i++; continue; }
    const group = [i++];
    while (i < shots.length && shots[i] !== "wide" && group.length < per) {
      let remaining = 0;
      for (let j = i; j < shots.length; j++) if (shots[j] !== "wide") remaining++;
      // Leave two for the next tier rather than three and one.
      if (per === 3 && group.length === 2 && remaining === 2) break;
      group.push(i++);
    }
    out.push(group);
  }
  while (out.length > 4) {
    let k = 0;
    for (let t = 1; t < out.length - 1; t++) {
      if (out[t].length + out[t + 1].length < out[k].length + out[k + 1].length) k = t;
    }
    out.splice(k, 2, [...out[k], ...out[k + 1]]);
  }
  return out;
}

const round6 = (n: number) => Math.round(n * 1e6) / 1e6;

/**
 * Each panel's cell on one page as [x, y, w, h] fractions of the inner
 * area. The tiers fill the page, so every page of a board is the same
 * shape. Manga runs right to left, so a tier's cells start at the right.
 */
export function pageCells(shots: ShotClass[], rtl = false): Cell[] {
  const bands = tiers(shots);
  const gy = GUTTER * INNER_ASPECT;
  const weights = bands.map((t) => (t.length === 1 && shots[t[0]] === "wide" ? WIDE_TIER : 1));
  const total = weights.reduce((a, b) => a + b, 0);
  const room = 1 - gy * (bands.length - 1);
  const cells: Cell[] = new Array(shots.length);
  let y = 0;
  bands.forEach((band, b) => {
    const h = room * weights[b] / total;
    const w = (1 - GUTTER * (band.length - 1)) / band.length;
    band.forEach((i, k) => {
      const x = k * (w + GUTTER);
      cells[i] = [round6(rtl ? 1 - x - w : x), round6(y), round6(w), round6(h)];
    });
    y += h + gy;
  });
  return cells;
}

/** The layout class of a panel: a large panel takes a whole tier. */
export function sizeClass(panel: Pick<Panel, "shot" | "weight">): ShotClass {
  if ((panel.weight ?? 2) >= 3) return "wide";
  if (["EWS", "WS", "wide"].includes(panel.shot)) return "wide";
  if (["CU", "ECU", "INSERT", "close-up"].includes(panel.shot)) return "close-up";
  return "medium";
}

/**
 * Which page each panel is on.
 *
 * About `perPage` panels a page, breaking where the scene changes when the
 * page has enough on it already -- a scene starting halfway down a page
 * reads as a jump -- and never leaving one panel alone on the last page.
 */
export function paginate(scenes: number[], perPage: number): number[] {
  const cap = Math.max(1, perPage);
  const page: number[] = [];
  let at = 0, count = 0;
  scenes.forEach((scene, i) => {
    const sceneChanged = i > 0 && scene !== scenes[i - 1];
    if (count >= cap || (sceneChanged && count >= Math.ceil(cap / 2))) { at++; count = 0; }
    page.push(at);
    count++;
  });
  // A lone panel at the end joins the page before it, a little over.
  const last = page.length - 1;
  if (last > 0 && page[last] !== page[last - 1] && count === 1
      && page.filter((p) => p === page[last - 1]).length < cap + 1) {
    page[last] = page[last - 1];
  }
  return page;
}

export interface PanelPlace {
  /** Which page, from 0. */
  page: number;
  /** The cell on that page; for a scroll, the whole width. */
  cell: Cell;
  /** Width over height of what is drawn: the cell's own shape. */
  aspect: number;
}

/** The shape of a scroll panel: wider for a wide shot, square otherwise. */
export function stripAspect(cls: ShotClass): number {
  return cls === "wide" ? 1.5 : cls === "close-up" ? 1 : 1.2;
}

/**
 * Every panel's page, cell and drawn shape, for a board in this format.
 * `layout` "strip" is a vertical scroll: one panel a row, each at its shot's
 * shape, paged only so drawing can go a page at a time.
 */
export function layoutBoard(
  panels: Pick<Panel, "shot" | "weight" | "scene">[], format: BookFormat, layout: string,
): PanelPlace[] {
  const per = PANELS_PER_PAGE[format] ?? 5;
  const pages = paginate(panels.map((p) => Number(p.scene ?? 1) || 1), per);
  const out: PanelPlace[] = new Array(panels.length);
  const strip = layout === "strip" || format === "webtoon";
  const rtl = format === "manga";
  for (const page of new Set(pages)) {
    const idx = pages.map((p, i) => (p === page ? i : -1)).filter((i) => i >= 0);
    const classes = idx.map((i) => sizeClass(panels[i]));
    const cells = strip ? classes.map((): Cell => [0, 0, 1, 1]) : pageCells(classes, rtl);
    idx.forEach((i, k) => {
      const [, , w, h] = cells[k];
      out[i] = { page, cell: cells[k],
                 aspect: strip ? stripAspect(classes[k]) : (w / h) * INNER_ASPECT };
    });
  }
  return out;
}

/**
 * The size to draw a panel of this shape at: about `px` x `px` worth of
 * pixels -- the memory a square panel at that size was measured to need --
 * each side a multiple of 16, which the image models require.
 */
export function panelSize(aspect: number, px: number): { width: number; height: number } {
  const a = Math.min(3, Math.max(1 / 3, aspect || 1));
  const area = px * px;
  const snap = (n: number) => Math.max(256, Math.floor(n / 16) * 16);
  let width = snap(Math.sqrt(area * a));
  let height = snap(Math.sqrt(area / a));
  // Rounding both sides down keeps the drawing within the measured memory.
  while (width * height > area && width > 256) width -= 16;
  width = Math.min(width, 1536);
  height = Math.min(height, 1536);
  return { width, height };
}

/**
 * Which side of the panel a speaker stands on, for their balloon: from the
 * director's blocking when they are in frame, "off" when they are heard and
 * not seen.
 */
export function speakerSide(panel: Pick<Panel, "cast" | "positions">, speaker: string,
                            offscreen?: boolean): "left" | "right" | "centre" | "off" {
  if (offscreen) return "off";
  if (!(panel.cast ?? []).some((a) => a.name === speaker)) return "off";
  const where = (panel.positions?.[speaker] ?? "").toLowerCase();
  if (where.includes("left")) return "left";
  if (where.includes("right")) return "right";
  return "centre";
}
