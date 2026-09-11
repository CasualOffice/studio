import { useEffect, useMemo, useState } from "react";
import { api, errText, fmtDuration, newJobId, onEngineProgress, vaultUrl } from "../lib/api";
import type { EngineProgress, ModelStatus } from "../lib/types";
import { humanDuration } from "../lib/presets";
import { exportItem, ImageDrop, JobProgress } from "./shared";
import { loadPref, savePref } from "../lib/prefs";

/** Small canvases only: attention cost is quadratic in sequence length. */
const SIZES: [string, number, number][] = [
  ["Small landscape", 384, 224],
  ["Small portrait", 224, 384],
  ["Square", 320, 320],
  ["Wider", 480, 272],
];

export default function Video({
  models, notify, onProduced, firstFrame: controlledFrame, onFirstFrameChange,
}: {
  models: ModelStatus[];
  notify: (m: string, bad?: boolean) => void;
  onProduced: () => void;
  /** A picture carried in from another tab, to animate. */
  firstFrame?: string[];
  onFirstFrameChange?: (ids: string[]) => void;
}) {
  const usable = useMemo(
    () => models.filter(
      (m) => m.installed && !m.hopeless && m.fit !== "broken"
        && m.tasks.includes("video" as never)
    ),
    [models]
  );
  const notInstalled = useMemo(
    () => models.filter((m) => m.tasks.includes("video" as never) && !m.installed && !m.hopeless),
    [models]
  );

  const [modelId, setModelId] = useState("");
  const [prompt, setPrompt] = useState("");
  const [sizeIdx, setSizeIdx] = useState(() => loadPref("videoSize", 0));
  const [frames, setFrames] = useState(() => loadPref("videoFrames", 33));
  const [fps, setFps] = useState(() => loadPref("videoFps", 16));
  const [steps, setSteps] = useState(20);
  const [seed, setSeed] = useState(0);
  const [randomSeed, setRandomSeed] = useState(true);
  const [localFrame, setLocalFrame] = useState<string[]>([]);
  const firstFrame = controlledFrame ?? localFrame;
  const setFirstFrame = onFirstFrameChange ?? setLocalFrame;

  const [assisting, setAssisting] = useState(false);
  const [running, setRunning] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [prog, setProg] = useState<EngineProgress | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [out, setOut] = useState<string | null>(null);
  const [took, setTook] = useState<number | null>(null);

  const model = usable.find((m) => m.id === modelId) ?? usable[0];

  const assistantReady = useMemo(
    // The writer, specifically. There are two assist models now, and `.some`
    // was satisfied by the picture-reader alone -- so the button looked ready
    // and then failed with "the prompt writer is not installed".
    () => models.some((m) => m.id === "qwen3-4b-instruct-4bit" && m.installed),
    [models]
  );

  /** Add the motion a video model is steered by, keeping the user's words. */
  const improve = async () => {
    if (!prompt.trim()) { notify("Write a rough idea first.", true); return; }
    const id = newJobId();
    setAssisting(true);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    try {
      const r = await api.assistPrompt(id, prompt, "video", []);
      if (r.unclear) {
        notify(r.note ?? "Say what should be in the clip first.", true);
        return;
      }
      setPrompt(r.prompt);
      notify("Made your prompt specific, and said how it moves.");
    } catch (e) {
      notify(errText(e), true);
    } finally {
      un(); setAssisting(false); setProg(null);
    }
  };

  useEffect(() => { if (model) setSteps(model.steps_default || 20); }, [model?.id]); // eslint-disable-line
  useEffect(() => { savePref("videoSize", sizeIdx); }, [sizeIdx]);
  useEffect(() => { savePref("videoFrames", frames); }, [frames]);
  useEffect(() => { savePref("videoFps", fps); }, [fps]);

  useEffect(() => {
    if (!running) { setElapsed(0); return; }
    const t0 = Date.now();
    const t = window.setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 500);
    return () => window.clearInterval(t);
  }, [running]);

  const [w, h] = [SIZES[sizeIdx][1], SIZES[sizeIdx][2]];
  const seconds = (frames / fps).toFixed(1);

  // Video is far slower per pixel than a still, and cost grows faster than
  // linearly with frames. This is a rough shape, not a measurement.
  const eta = model
    ? Math.round((steps * (w * h / 1_000_000) * frames * 900) / 1000 + 40)
    : 0;

  if (usable.length === 0) {
    return (
      <div className="empty-state">
        <span className="big">▷</span>
        No video model installed.
        <div style={{ marginTop: 12, fontSize: 12, maxWidth: 520, marginLeft: "auto",
                      marginRight: "auto", lineHeight: 1.7, textAlign: "left" }}>
          {notInstalled.length > 0 ? (
            <>
              Install one from <b>Models</b>. On this Mac only the 1.3B class fits:
              {notInstalled.map((m) => (
                <div key={m.id} style={{ marginTop: 6 }}>
                  <b>{m.name}</b> — {m.package_gib.toFixed(1)} GiB download,
                  ~{m.peak_gib.toFixed(1)} GiB memory
                </div>
              ))}
              <div style={{ marginTop: 10, color: "var(--text-faint)", fontSize: 11 }}>
                Wan 2.2 and larger need 33 GiB or more and cannot run here. Wan 2.5
                and 2.6 have no public weights at all.
              </div>
            </>
          ) : "No video model in the catalog can run on this Mac."}
        </div>
      </div>
    );
  }

  const run = async () => {
    if (!prompt.trim()) { notify("Describe the clip first.", true); return; }
    const id = newJobId();
    const s = randomSeed ? Math.floor(Math.random() * 2_000_000_000) : seed;
    // Clear the last result as the new run starts. Leaving it up made a
    // second run look like nothing was happening.
    setOut(null); setTook(null);
    setRunning(true); setJobId(id); setProg(null); setTook(null);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    const t0 = performance.now();
    try {
      const res = await api.generateVideo({
        jobId: id, modelId: model.id, prompt, negativePrompt: null,
        width: w, height: h, frames, fps, steps,
        guidance: model.guidance_default, seed: s,
        firstFrame: firstFrame[0] ?? null,
      });
      setOut(res[0] ?? null);
      setTook(performance.now() - t0);
      setSeed(s);
      onProduced();
    } catch (e) {
      const msg = errText(e);
      notify(msg.includes("cancelled") ? "Cancelled." : msg, !msg.includes("cancelled"));
    } finally {
      un(); setRunning(false); setJobId(null); setProg(null);
    }
  };

  return (
    <div className="content split" style={{ padding: 0 }}>
      <div>
        <div className="panel">
          <h2>Make a clip</h2>

          {model?.video_from_image === false ? (
            <div className="notice" style={{ marginBottom: 12 }}>
              <strong>{model.name} makes clips from a description only</strong>
              It transforms existing footage rather than animating a still, so
              it has no way to use a starting picture. Pick a model listed as
              image to video to animate a photo.
            </div>
          ) : (
            <div className="field">
              <label>
                Starting picture <em>optional</em>
              </label>
              <ImageDrop
                images={firstFrame}
                onChange={setFirstFrame}
                max={model?.max_edit_images ?? 1}
                onError={(m) => notify(m, true)}
              />
              <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5 }}>
                {controlledFrame && controlledFrame.length > 0
                  ? "Carried over from another tab. This picture becomes the first frame."
                  : "With a picture the clip animates it. Without one it is made from the description alone."}
              </div>
            </div>
          )}

          <div className="field">
            <label>What should happen?</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="steam rises slowly from the teapot, the light shifts"
            />
            <button
              className="btn small full"
              style={{ marginTop: 7 }}
              disabled={assisting || running || !prompt.trim()}
              onClick={improve}
              title={assistantReady
                ? "Keeps your words and adds the motion a video model needs"
                : "Needs the prompt assistant from the Models tab"}
            >
              {assisting ? "Thinking…" : "\u2728 Make my prompt precise"}
            </button>
            {!assistantReady && (
              <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5 }}>
                Needs the prompt writer — a 2.1 GB download in the Models tab.
                It runs on this Mac; nothing is sent anywhere.
              </div>
            )}
          </div>

          <div className="field">
            <label>Size</label>
            <select value={sizeIdx} onChange={(e) => setSizeIdx(+e.target.value)}>
              {SIZES.map(([label, sw, sh], i) => (
                <option key={label} value={i}>{label} — {sw}×{sh}</option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>
              Frames <em>{frames} · {seconds}s at {fps} fps</em>
            </label>
            <input
              type="range" min={9} max={81} step={4} value={frames}
              onChange={(e) => setFrames(+e.target.value)}
            />
            <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5, lineHeight: 1.55 }}>
              Frames are the only real cost. Memory and time grow faster than
              linearly with them, so halving the count saves more than half.
            </div>
          </div>

          <div className="row">
            <div className="field">
              <label>Playback speed <em>{fps} fps</em></label>
              <input type="range" min={8} max={30} step={1} value={fps}
                onChange={(e) => setFps(+e.target.value)} />
              <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5 }}>
                Changes duration only, not how long it takes to make.
              </div>
            </div>
            <div className="field">
              <label>Steps <em>{steps}</em></label>
              <input type="range" min={4} max={40} value={steps}
                onChange={(e) => setSteps(+e.target.value)} />
            </div>
          </div>

          <div className="field">
            <label style={{ cursor: "pointer" }}>
              <span>
                <input type="checkbox" checked={randomSeed}
                  style={{ width: "auto", marginRight: 6 }}
                  onChange={(e) => setRandomSeed(e.target.checked)} />
                Different every time
              </span>
            </label>
          </div>

          {usable.length > 1 ? (
            <div className="field">
              <label>Model</label>
              <select value={model?.id ?? ""} onChange={(e) => setModelId(e.target.value)}>
                {usable.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} — ~{m.peak_gib.toFixed(1)} GiB
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginBottom: 8 }}>
              Using <b>{model?.name}</b>
            </div>
          )}

          {running ? (
            <>
              <JobProgress p={prog} label="Rendering" />
              <div style={{ display: "flex", justifyContent: "space-between",
                            fontSize: 11, color: "var(--text-faint)", marginTop: 6 }}>
                <span>{elapsed}s elapsed</span>
                <span>video takes minutes, not seconds</span>
              </div>
              <button className="btn danger full" style={{ marginTop: 10 }}
                onClick={() => jobId && api.cancelJob(jobId).catch(() => {})}>
                Cancel
              </button>
            </>
          ) : (
            <>
              <button className="btn primary full" onClick={run}>Make the clip</button>
              <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 6,
                            textAlign: "center" }}>
                Roughly {humanDuration(eta)} — a rough shape, not a measurement
              </div>
            </>
          )}
        </div>
      </div>

      <div>
        <div className="canvas">
          {out ? (
            <video
              src={vaultUrl(out)}
              controls
              loop
              autoPlay
              muted
              style={{ maxWidth: "100%", maxHeight: "64vh", borderRadius: 6 }}
            />
          ) : (
            <div className="empty">
              <span className="big">▷</span>
              {running
                ? "Rendering. The model loads first, then denoises every frame."
                : "Describe a clip, or drop in a picture to animate."}
            </div>
          )}
        </div>
        {out && (
          <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
            <button className="btn small" onClick={() => exportItem({ id: out, name: "clip.mp4" }, notify)}>
              Save a copy…
            </button>
            {took != null && (
              <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-faint)" }}>
                Took {fmtDuration(Math.round(took))} · seed {seed}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
