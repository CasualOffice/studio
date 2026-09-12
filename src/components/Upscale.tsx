import { useMemo, useState } from "react";
import { api, errText, isCancelled, newJobId, onEngineProgress, onEnginePreview, vaultUrl } from "../lib/api";
import type { EngineProgress, ModelStatus } from "../lib/types";
import { exportItem, ImageDrop, JobProgress } from "./shared";
import SendTo, { type Destination } from "./SendTo";

export default function Upscale({
  models, notify, onProduced, images: controlledImages, onImagesChange, send,
}: {
  models: ModelStatus[];
  notify: (m: string, bad?: boolean) => void;
  onProduced: () => void;
  images?: string[];
  onImagesChange?: (ids: string[]) => void;
  send?: (dest: Destination, ids: string[]) => void;
}) {
  // What the run looks like part-way. A tiled upscale sends one of these per
  // finished piece, so a multi-minute job shows the picture filling in
  // instead of a bar over a blank frame.
  const [preview, setPreview] = useState<string | null>(null);

  const usable = useMemo(
    () => models.filter((m) => m.installed && m.fit !== "too_much_memory"
      && m.fit !== "broken" && m.tasks.includes("upscale" as never)),
    [models]
  );
  const [modelId, setModelId] = useState("");
  const [localImages, setLocalImages] = useState<string[]>([]);
  const images = controlledImages ?? localImages;
  const setImages = onImagesChange ?? setLocalImages;
  const [resolution, setResolution] = useState("2x");
  const [lowRam, setLowRam] = useState(true);
  const [running, setRunning] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [prog, setProg] = useState<EngineProgress | null>(null);
  const [out, setOut] = useState<string | null>(null);

  const model = usable.find((m) => m.id === modelId) ?? usable[0];

  // Distinguish "not installed" from "cannot work at all": the only upscaler
  // in the catalog is blocked by a version conflict, and saying "install one"
  // would send the user to download 2.5 GiB that will never run.
  const blocked = models.filter(
    (m) => m.tasks.includes("upscale" as never) && m.fit === "broken"
  );

  if (usable.length === 0) {
    return (
      <div className="empty-state">
        <span className="big">⤢</span>
        {blocked.length > 0 ? "Upscaling is unavailable in this build." : "No upscaler installed."}
        {blocked.length > 0 ? (
          <div style={{
            marginTop: 12, fontSize: 12, maxWidth: 520, marginLeft: "auto",
            marginRight: "auto", lineHeight: 1.7, textAlign: "left",
          }}>
            SeedVR2 needs MLX 0.31.0, while the prompt assistant needs 0.31.2, and
            MLX-Gen caps MLX below 0.32 — so no single version runs both. This
            build is pinned to 0.31.2, which keeps generation, editing and prompt
            help working.
            <div style={{ marginTop: 10, color: "var(--text-faint)", fontSize: 11 }}>
              If upscaling matters more than prompt help, downgrading MLX to
              0.31.0 reverses the trade.
            </div>
          </div>
        ) : (
          <div style={{ marginTop: 10, fontSize: 12 }}>
            Install one from <b>Models</b>.
          </div>
        )}
      </div>
    );
  }

  const run = async () => {
    if (!model || images.length === 0) { notify("Add an image first.", true); return; }
    const id = newJobId();
    // Clear the last result as the new run starts. Leaving it up made a
    // second run look like nothing was happening.
    setOut(null);
    setRunning(true); setJobId(id); setProg(null);
    setPreview(null);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    // If this subscription throws, the progress listener above would
    // never be released. Failing to get a preview is not worth leaking one.
    let unp: (() => void) | undefined;
    try { unp = await onEnginePreview((f) => { if (f.jobId === id) setPreview(f.src); }); }
    catch { unp = undefined; }
    try {
      const res = await api.upscale(id, model.id, images[0], resolution, lowRam);
      setOut(res[0] ?? null);
      onProduced();
    } catch (e) {
      notify(isCancelled(e) ? "Cancelled." : errText(e), !isCancelled(e));
    } finally {
      un(); unp?.(); setPreview(null); setRunning(false); setJobId(null); setProg(null);
    }
  };

  return (
    <div className="content split" style={{ padding: 0 }}>
      <div className="panel">
        <h2>Upscale</h2>
        <div className="field">
          <label>Image</label>
          <ImageDrop images={images} onChange={setImages} max={1}
            onError={(m) => notify(m, true)} />
        </div>
        <div className="field">
          <label>Model</label>
          <select value={model?.id ?? ""} onChange={(e) => setModelId(e.target.value)}>
            {usable.map((m) => (
              <option key={m.id} value={m.id}>{m.name} — ~{m.peak_gib.toFixed(1)} GiB</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Scale</label>
          <select value={resolution} onChange={(e) => setResolution(e.target.value)}>
            {["2x", "3x", "4x", "5x"].map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
          <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5 }}>
            Memory scales with the output canvas, not just the model. Large scales from a large
            source can exceed what a 16 GB machine has.
          </div>
        </div>
        <div className="field">
          <label style={{ cursor: "pointer" }}>
            <span>
              <input
                type="checkbox" checked={lowRam} style={{ width: "auto", marginRight: 6 }}
                onChange={(e) => setLowRam(e.target.checked)}
              />
              Low-RAM mode
            </span>
          </label>
          <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5 }}>
            Restoration allocates a large output canvas. The public video path enables this
            automatically; leaving it on here is the safe default.
          </div>
        </div>
        {running ? (
          <>
            <JobProgress p={prog} label="Upscaling" />
            <button className="btn danger full" style={{ marginTop: 10 }}
              onClick={() => jobId && api.cancelJob(jobId).catch(() => {})}>
              Cancel
            </button>
          </>
        ) : (
          <button className="btn primary full" onClick={run}>Upscale</button>
        )}
      </div>
      <div>
        <div className="canvas">
          {out ? <img src={vaultUrl(out)} alt="" />
           : preview ? (
            <div style={{ textAlign: "center" }}>
              {/* Soft on purpose: pieces are still landing, and this should
                  not be mistaken for the finished picture. */}
              <img src={preview} alt="" style={{
                maxWidth: "100%", maxHeight: "58vh", borderRadius: 6, opacity: 0.92,
              }} />
              <div style={{ fontSize: 11, color: "var(--text-faint)", marginTop: 8 }}>
                {prog?.message ?? "Working\u2026"}
              </div>
            </div>
           ) : <div className="empty"><span className="big">⤢</span>Restored output appears here.</div>}
        </div>
        {out && (
          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            <button className="btn small" onClick={() => exportItem({ id: out, name: "" }, notify)}>
              Save a copy…
            </button>
            {send && <SendTo ids={[out]} send={send} models={models} exclude={["upscale"]} />}
          </div>
        )}
      </div>
    </div>
  );
}
