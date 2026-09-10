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
    if (!a) return r;
    // Keep the taller/wider dimension and derive the other, so dragging feels
    // like it follows the pointer rather than fighting it.
    const h = r.w / a;
    return h <= 1 ? { ...r, h: r.w / a } : { ...r, h };
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
      // Crop first, in source pixels, then rotate and flip the result.
      const sx = crop ? crop.x * natural.w : 0;
      const sy = crop ? crop.y * natural.h : 0;
      const sw = crop ? crop.w * natural.w : natural.w;
      const sh = crop ? crop.h * natural.h : natural.h;

      const cut = document.createElement("canvas");
      cut.width = Math.max(1, Math.round(sw));
      cut.height = Math.max(1, Math.round(sh));
      cut.getContext("2d")!.drawImage(img, sx, sy, sw, sh, 0, 0, cut.width, cut.height);

      const angle = (quarter * 90 + straighten) * (Math.PI / 180);
      const cos = Math.abs(Math.cos(angle)), sin = Math.abs(Math.sin(angle));
      const outW = Math.round(cut.width * cos + cut.height * sin);
      const outH = Math.round(cut.width * sin + cut.height * cos);

      const out = document.createElement("canvas");
      out.width = outW; out.height = outH;
      const ctx = out.getContext("2d")!;
      ctx.translate(outW / 2, outH / 2);
      ctx.rotate(angle);
      ctx.scale(flipH ? -1 : 1, flipV ? -1 : 1);
      ctx.drawImage(cut, -cut.width / 2, -cut.height / 2);

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

  const preview = {
    transform: `rotate(${quarter * 90 + straighten}deg) scale(${flipH ? -1 : 1}, ${flipV ? -1 : 1})`,
    transition: "transform .12s",
  };

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
