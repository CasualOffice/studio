import { useCallback, useEffect, useRef, useState } from "react";
import { api, errText, vaultUrl } from "../lib/api";

type Rect = { x: number; y: number; w: number; h: number };

const ASPECTS: [string, number | null][] = [
  ["Free", null], ["Square", 1], ["4:3", 4 / 3], ["3:4", 3 / 4],
  ["16:9", 16 / 9], ["9:16", 9 / 16],
];

/**
 * Crop, rotate, straighten and flip.
 *
 * None of this needs a model. Doing it here is instant and lossless where the
 * operation allows, instead of asking a diffusion model to reproduce a picture
 * it was only meant to reframe.
 */
export default function Adjust({
  sourceId, onSaved, notify,
}: {
  sourceId: string;
  onSaved: (newId: string) => void;
  notify: (m: string, bad?: boolean) => void;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ ox: number; oy: number; start: Rect } | null>(null);

  const [natural, setNatural] = useState({ w: 0, h: 0 });
  const [crop, setCrop] = useState<Rect | null>(null);
  const [aspect, setAspect] = useState<number | null>(null);
  const [quarter, setQuarter] = useState(0);      // 90-degree steps
  const [straighten, setStraighten] = useState(0); // fine angle, degrees
  const [flipH, setFlipH] = useState(false);
  const [flipV, setFlipV] = useState(false);
  const [busy, setBusy] = useState(false);

  const onLoad = useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    setNatural({ w: img.naturalWidth, h: img.naturalHeight });
    setCrop(null);
  }, []);

  useEffect(() => {
    setQuarter(0); setStraighten(0); setFlipH(false); setFlipV(false); setCrop(null);
  }, [sourceId]);

  const applyAspect = (r: Rect, a: number | null): Rect => {
    const availableW = Math.max(0, 1 - r.x);
    const availableH = Math.max(0, 1 - r.y);
    if (!a || !natural.w || !natural.h) {
      return { ...r, w: Math.min(r.w, availableW), h: Math.min(r.h, availableH) };
    }
    // The selection is stored in normalised image coordinates. Correct for
    // the source's own aspect ratio before enforcing the requested output.
    const normalisedAspect = a * natural.h / natural.w;
    let w = Math.min(r.w, availableW);
    let h = w / normalisedAspect;
    if (h > availableH) {
      h = availableH;
      w = Math.min(availableW, h * normalisedAspect);
    }
    return { ...r, w, h };
  };

  const pointIn = (e: React.PointerEvent) => {
    const box = boxRef.current!;
    const r = box.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    };
  };

  const reset = () => {
    setCrop(null); setQuarter(0); setStraighten(0); setFlipH(false); setFlipV(false);
  };

  const dirty = !!crop || quarter !== 0 || straighten !== 0 || flipH || flipV;

  const save = async () => {
    const img = imgRef.current;
    if (!img) return;
    setBusy(true);
    try {
      // Turn first, then cut what was actually selected.
      //
      // This was the other way round -- crop the source, rotate the result --
      // while the preview turns the picture on screen with a CSS transform. So
      // the rectangle was drawn over a rotated image and applied to an
      // unrotated one: after a quarter turn a horizontal drag took a vertical
      // band out of the source, and the saved picture was of somewhere else.
      // The selection is normalised against what is on screen, so the thing it
      // has to be applied to is the turned picture.
      const angle = (quarter * 90 + straighten) * (Math.PI / 180);
      const cos = Math.abs(Math.cos(angle)), sin = Math.abs(Math.sin(angle));
      const rotW = Math.max(1, Math.round(natural.w * cos + natural.h * sin));
      const rotH = Math.max(1, Math.round(natural.w * sin + natural.h * cos));

      const turned = document.createElement("canvas");
      turned.width = rotW; turned.height = rotH;
      const tctx = turned.getContext("2d")!;
      tctx.translate(rotW / 2, rotH / 2);
      tctx.rotate(angle);
      tctx.scale(flipH ? -1 : 1, flipV ? -1 : 1);
      tctx.drawImage(img, -natural.w / 2, -natural.h / 2);

      const sx = crop ? crop.x * rotW : 0;
      const sy = crop ? crop.y * rotH : 0;
      const sw = crop ? crop.w * rotW : rotW;
      const sh = crop ? crop.h * rotH : rotH;

      const out = document.createElement("canvas");
      const outW = Math.max(1, Math.round(sw));
      const outH = Math.max(1, Math.round(sh));
      out.width = outW; out.height = outH;
      out.getContext("2d")!.drawImage(turned, sx, sy, sw, sh, 0, 0, outW, outH);

      const blob = await new Promise<Blob | null>((r) => out.toBlob(r, "image/png"));
      if (!blob) throw new Error("could not render the result");
      const bytes = new Uint8Array(await blob.arrayBuffer());
      const id = await api.vaultImportBytes(bytes, "adjusted.png", "image/png", "adjust");
      notify(`Saved ${outW}×${outH}.`);
      onSaved(id);
      reset();
    } catch (e) {
      notify(errText(e), true);
    } finally {
      setBusy(false);
    }
  };

  /**
   * The turned picture, scaled to stay inside the frame it is turned in.
   *
   * A CSS transform does not affect layout, so the container kept the height
   * of the unturned image while `overflow: hidden` cut off the taller turned
   * one -- at a quarter turn the top and bottom of the picture were simply not
   * on screen, which is the one thing the crop rectangle is drawn against.
   *
   * Worked in units of the container's width, so the width itself cancels:
   * the unturned picture is 1 wide by `ratio` tall, and its turned bounding
   * box is the usual cos/sin combination of the two.
   */
  const preview = (() => {
    const ratio = natural.w > 0 ? natural.h / natural.w : 1;
    const angle = (quarter * 90 + straighten) * (Math.PI / 180);
    const cos = Math.abs(Math.cos(angle)), sin = Math.abs(Math.sin(angle));
    const turnedW = cos + ratio * sin;
    const turnedH = sin + ratio * cos;
    const fit = Math.min(1 / (turnedW || 1), ratio / (turnedH || 1), 1);
    return {
      transform: `rotate(${quarter * 90 + straighten}deg) `
        + `scale(${fit * (flipH ? -1 : 1)}, ${fit * (flipV ? -1 : 1)})`,
      transition: "transform .12s",
    };
  })();

  return (
    <div>
      <div
        ref={boxRef}
        style={{
          position: "relative", overflow: "hidden", borderRadius: 8,
          border: "1px solid var(--border)", background: "#0b0d12",
          cursor: "crosshair", touchAction: "none",
        }}
        onPointerDown={(e) => {
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
          const p = pointIn(e);
          drag.current = { ox: p.x, oy: p.y, start: { x: p.x, y: p.y, w: 0, h: 0 } };
          setCrop({ x: p.x, y: p.y, w: 0, h: 0 });
        }}
        onPointerMove={(e) => {
          if (!drag.current) return;
          const p = pointIn(e);
          const { ox, oy } = drag.current;
          setCrop(applyAspect({
            x: Math.min(ox, p.x), y: Math.min(oy, p.y),
            w: Math.abs(p.x - ox), h: Math.abs(p.y - oy),
          }, aspect));
        }}
        onPointerUp={() => {
          drag.current = null;
          // A stray click is not a crop.
          setCrop((c) => (c && (c.w < 0.02 || c.h < 0.02) ? null : c));
        }}
      >
        <img
          ref={imgRef} src={vaultUrl(sourceId)} alt="" onLoad={onLoad}
          /* Fetched as a CORS request so the canvas that saves the result is
             not tainted by it. `vault://` is a different origin from the page,
             and without this the rotate-and-crop save threw SecurityError --
             "The operation is insecure." -- at `toBlob`, which is the last
             step, after the work. Needs the matching
             `Access-Control-Allow-Origin` on the protocol in lib.rs: with
             `crossOrigin` set and no such header the image does not load at
             all, so the two only work together. */
          crossOrigin="anonymous"
          style={{ display: "block", width: "100%", ...preview }}
        />
        {crop && (
          <>
            {/* Dim everything outside the selection. */}
            <div style={{
              position: "absolute", inset: 0, pointerEvents: "none",
              boxShadow: `0 0 0 9999px rgba(8,9,13,.62)`,
              clipPath: `inset(${crop.y * 100}% ${(1 - crop.x - crop.w) * 100}% ${(1 - crop.y - crop.h) * 100}% ${crop.x * 100}%)`,
            }} />
            <div style={{
              position: "absolute", pointerEvents: "none",
              left: `${crop.x * 100}%`, top: `${crop.y * 100}%`,
              width: `${crop.w * 100}%`, height: `${crop.h * 100}%`,
              outline: "1px solid var(--accent)",
            }} />
          </>
        )}
      </div>

      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        <button className="btn small" onClick={() => setQuarter((q) => (q + 3) % 4)}>↺ Left</button>
        <button className="btn small" onClick={() => setQuarter((q) => (q + 1) % 4)}>↻ Right</button>
        <button className={"btn small" + (flipH ? " primary" : "")}
          onClick={() => setFlipH(!flipH)}>Flip ↔</button>
        <button className={"btn small" + (flipV ? " primary" : "")}
          onClick={() => setFlipV(!flipV)}>Flip ↕</button>
        <button className="btn small" disabled={!dirty} onClick={reset}>Reset</button>
      </div>

      <div className="field" style={{ marginTop: 10 }}>
        <label>Straighten <em>{straighten.toFixed(1)}°</em></label>
        <input type="range" min={-15} max={15} step={0.5} value={straighten}
          onChange={(e) => setStraighten(+e.target.value)} />
      </div>

      <div className="field">
        <label>Crop shape</label>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {ASPECTS.map(([label, a]) => (
            <button key={label}
              className={"btn small" + (aspect === a ? " primary" : "")}
              onClick={() => { setAspect(a); setCrop((c) => (c ? applyAspect(c, a) : c)); }}
            >{label}</button>
          ))}
        </div>
        <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5 }}>
          {crop
            ? `Cropping to ${Math.round(crop.w * natural.w)}×${Math.round(crop.h * natural.h)}`
            : "Drag on the picture to crop. Nothing selected keeps the whole frame."}
        </div>
      </div>

      <button className="btn primary full" disabled={busy || !dirty} onClick={save}>
        {busy ? "Saving…" : "Save as a new picture"}
      </button>
      <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 6 }}>
        Saved into your vault as a new item. The original is untouched.
      </div>
    </div>
  );
}
