import { useCallback, useEffect, useRef, useState } from "react";
import { vaultUrl } from "../lib/api";

/**
 * Paint the region a model is allowed to change.
 *
 * Two canvases sit on top of the picture: one the user paints on, in a colour
 * they can see, and one kept in black and white that is what actually gets
 * sent. Deriving the second from the first at export time would mean
 * round-tripping the display colour back to a threshold, which is lossy and
 * fiddly; keeping both is simpler and exact.
 */
export default function MaskCanvas({
  sourceId, onMaskChange, disabled,
}: {
  sourceId: string;
  /** PNG bytes of the mask, or null when nothing is painted. */
  onMaskChange: (png: Uint8Array | null) => void;
  disabled?: boolean;
}) {
  const displayRef = useRef<HTMLCanvasElement>(null);
  const maskRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);

  const [brush, setBrush] = useState(48);
  const [erasing, setErasing] = useState(false);
  const [painted, setPainted] = useState(false);
  const [size, setSize] = useState({ w: 0, h: 0 });

  // Size both canvases to the image's natural pixels, so the exported mask
  // lines up with the source exactly regardless of how it is displayed.
  const onLoad = useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    const w = img.naturalWidth, h = img.naturalHeight;
    setSize({ w, h });

    const mask = document.createElement("canvas");
    mask.width = w; mask.height = h;
    const mctx = mask.getContext("2d")!;
    mctx.fillStyle = "#000";
    mctx.fillRect(0, 0, w, h);
    maskRef.current = mask;

    const disp = displayRef.current;
    if (disp) {
      disp.width = w; disp.height = h;
      disp.getContext("2d")!.clearRect(0, 0, w, h);
    }
    setPainted(false);
    onMaskChange(null);
  }, [onMaskChange]);

  const exportMask = useCallback(() => {
    const mask = maskRef.current;
    if (!mask || !painted) { onMaskChange(null); return; }
    mask.toBlob(async (blob) => {
      if (!blob) return;
      onMaskChange(new Uint8Array(await blob.arrayBuffer()));
    }, "image/png");
  }, [painted, onMaskChange]);

  const pointAt = (e: React.PointerEvent) => {
    const disp = displayRef.current!;
    const r = disp.getBoundingClientRect();
    // Displayed size differs from natural size; map back into image pixels.
    return {
      x: ((e.clientX - r.left) / r.width) * disp.width,
      y: ((e.clientY - r.top) / r.height) * disp.height,
    };
  };

  const stroke = (from: { x: number; y: number } | null, to: { x: number; y: number }) => {
    const disp = displayRef.current, mask = maskRef.current;
    if (!disp || !mask) return;
    for (const [ctx, colour] of [
      [disp.getContext("2d")!, erasing ? null : "rgba(124,140,255,0.55)"],
      [mask.getContext("2d")!, erasing ? "#000" : "#fff"],
    ] as [CanvasRenderingContext2D, string | null][]) {
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = brush;
      if (colour === null) {
        // Erasing the visible layer means clearing it, not painting black.
        ctx.globalCompositeOperation = "destination-out";
        ctx.strokeStyle = "rgba(0,0,0,1)";
      } else {
        ctx.globalCompositeOperation = "source-over";
        ctx.strokeStyle = colour;
      }
      ctx.beginPath();
      ctx.moveTo(from?.x ?? to.x, from?.y ?? to.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
      ctx.globalCompositeOperation = "source-over";
    }
  };

  const clear = () => {
    const mask = maskRef.current, disp = displayRef.current;
    if (mask) {
      const c = mask.getContext("2d")!;
      c.fillStyle = "#000";
      c.fillRect(0, 0, mask.width, mask.height);
    }
    if (disp) disp.getContext("2d")!.clearRect(0, 0, disp.width, disp.height);
    setPainted(false);
    onMaskChange(null);
  };

  useEffect(() => { clear(); }, [sourceId]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div>
      <div style={{
        position: "relative", borderRadius: 8, overflow: "hidden",
        border: "1px solid var(--border)", background: "#0b0d12",
        cursor: disabled ? "default" : "crosshair", touchAction: "none",
      }}>
        <img
          ref={imgRef}
          src={vaultUrl(sourceId)}
          alt=""
          onLoad={onLoad}
          style={{ display: "block", width: "100%" }}
        />
        <canvas
          ref={displayRef}
          style={{
            position: "absolute", inset: 0, width: "100%", height: "100%",
            pointerEvents: disabled ? "none" : "auto",
          }}
          onPointerDown={(e) => {
            if (disabled) return;
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
            drawing.current = true;
            const p = pointAt(e);
            last.current = p;
            stroke(null, p);
            setPainted(true);
          }}
          onPointerMove={(e) => {
            if (!drawing.current || disabled) return;
            const p = pointAt(e);
            stroke(last.current, p);
            last.current = p;
          }}
          onPointerUp={() => {
            drawing.current = false;
            last.current = null;
            exportMask();
          }}
          onPointerLeave={() => {
            if (drawing.current) { drawing.current = false; last.current = null; exportMask(); }
          }}
        />
      </div>

      <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
        <button
          className={"btn small" + (erasing ? "" : " primary")}
          onClick={() => setErasing(false)}
        >Paint</button>
        <button
          className={"btn small" + (erasing ? " primary" : "")}
          onClick={() => setErasing(true)}
        >Erase</button>
        <button className="btn small" onClick={clear} disabled={!painted}>Clear</button>
        <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 7 }}>
          <span style={{ fontSize: 10.5, color: "var(--text-faint)" }}>Brush</span>
          <input
            type="range" min={8} max={160} value={brush}
            onChange={(e) => setBrush(+e.target.value)}
            style={{ flex: 1 }}
          />
        </div>
      </div>
      <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 6, lineHeight: 1.55 }}>
        {painted
          ? `Painted area will be regenerated${size.w ? ` · mask is ${size.w}×${size.h}` : ""}. Everything else is kept.`
          : "Paint over what you want changed. Without a mask the whole picture is edited."}
      </div>
    </div>
  );
}
