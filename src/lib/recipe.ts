/**
 * Turning what a run recorded back into what the controls showed.
 *
 * The vault stores a run the way the engine received it, which is not the way
 * the Studio offered it: the engine knows two image-to-image modes, the UI
 * offers five kinds of edit. Reproducing a picture means walking that back,
 * and the walk is lossy in places, so it lives here as plain functions with
 * the losses written down and tested rather than buried in an effect.
 */

/**
 * What kind of edit the Studio is offering.
 *
 * Two of these exist only in the UI. "adjust" is crop and rotate, done
 * locally, and never reaches the engine at all. "mask" is the instruct route
 * with a painted region attached, and the region is deleted the moment the run
 * ends because it is working material rather than something anyone wants in
 * their library.
 */
export type EditKind = "instruct" | "mask" | "expand" | "latent" | "adjust";

/** How the area added by an outpaint starts out, before the model fills it. */
export const OUTPAINT_FILLS: { id: string; label: string }[] = [
  { id: "auto", label: "Choose for me" },
  { id: "edge", label: "Continue the edges outward" },
  { id: "neutral", label: "Flat colour, invent new subject matter" },
  { id: "blur", label: "Blurred copy of the original" },
];

export interface Padding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/**
 * Which edit control produced a run, read back from what the run actually sent.
 *
 * `i2i_mode` is the engine's word for this and has two values; `editKind` is
 * the UI's and has five, so the inverse cannot be exact and this is where the
 * inexactness is decided:
 *
 *  - "latent" is the one case the engine names outright.
 *  - "expand" left a fingerprint. Nothing but the expand control ever sets
 *    outpaint padding, so its presence identifies the kind on its own.
 *  - "instruct" and "mask" both travel as i2i_mode "edit" and are otherwise
 *    indistinguishable in the record. Restoring "mask" would open the paint
 *    tab with nothing painted -- the mask was deleted when the run finished --
 *    and Run would then refuse with "paint over the part you want changed".
 *    "instruct" is the honest landing spot: same engine route, same prompt,
 *    and the paint tab is one click away if that is what was wanted.
 *  - "adjust" never reaches the engine, so no record can ever name it.
 *
 * Returns null when the run carried no i2i_mode: a plain generation, or an
 * item written before any of this was recorded. Callers must then leave the
 * current choice alone rather than guessing at one.
 */
export function editKindFor(
  i2iMode: string | null | undefined,
  outpaintPadding: string | null | undefined,
): EditKind | null {
  if (i2iMode === "latent") return "latent";
  if (i2iMode !== "edit") return null;
  return outpaintPadding && outpaintPadding.trim() !== "" ? "expand" : "instruct";
}

/**
 * Read outpaint padding back into the four sliders that wrote it.
 *
 * The Studio writes "top%,right%,bottom%,left%", but the field is CSS-like by
 * contract and mflux accepts the shorthands, so all four lengths are honoured
 * here -- an item that came from anywhere else still restores.
 *
 * Percentages only, deliberately. mflux also takes pixel padding, and the
 * sliders are percentages of the original, so a pixel recipe has no honest
 * position to restore to; null leaves the sliders at their defaults, which is
 * visibly a default rather than a wrong number presented as a faithful one.
 */
export function parseOutpaintPadding(s: string | null | undefined): Padding | null {
  if (!s) return null;
  const parts = s.split(",").map((p) => p.trim());
  if (parts.length < 1 || parts.length > 4 || parts.some((p) => p === "")) return null;

  const nums: number[] = [];
  for (const p of parts) {
    const m = /^(\d+(?:\.\d+)?)%$/.exec(p);
    if (!m) return null;
    const n = Number(m[1]);
    // The slider tops out at 100% of the original per side. A larger value
    // came from somewhere else and would silently clamp on the way in, so the
    // whole string is rejected instead of half-honoured.
    if (!Number.isFinite(n) || n < 0 || n > 100) return null;
    nums.push(Math.round(n));
  }

  // CSS shorthand: 1 value is all sides, 2 is vertical then horizontal, 3 adds
  // a distinct bottom, 4 is top-right-bottom-left.
  const [top, right = top, bottom = top, left = right] = nums;
  return { top, right, bottom, left };
}

/** Whether a recorded fill is one this build still offers in the dropdown. */
export function isOutpaintFill(s: string | null | undefined): s is string {
  return !!s && OUTPAINT_FILLS.some((f) => f.id === s);
}
