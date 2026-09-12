import { useEffect, useMemo, useState } from "react";
import { api, errText, isCancelled, fmtDuration, newJobId, onEngineProgress, onEnginePreview, vaultUrl } from "../lib/api";
import type { AssistResult, EngineProgress, ModelStatus } from "../lib/types";
import { humanDuration } from "../lib/presets";
import { exportItem, ImageDrop, JobProgress } from "./shared";
import { loadPref, savePref } from "../lib/prefs";
import PromptProposal from "./PromptProposal";

/**
 * Canvas sizes, up to what these models were actually trained at.
 *
 * The list stopped at 480x272 while Wan VACE is trained at 832x480 -- a third
 * of the pixels -- and running a video model far below its native size is a
 * quality problem of its own, not a safe economy. The larger ones are offered
 * with what they cost, because refusing to show them is not the same as them
 * not existing.
 *
 * Attention is quadratic in sequence length, and sequence length is pixels
 * times frames, so cost climbs fast in both.
 */
// Only sizes the model can actually resolve on.
//
// Measured on Wan 2.2 TI2V-5B, same seed and prompt: 320x192 and 480x272 come
// back as coloured noise, 640x368 and 704x384 come back as pictures. Five of
// the seven sizes offered here were below that floor, and every clip anyone
// made with them was noise.
//
// The small ones are not kept as a "fast" option because they are not a worse
// picture, they are no picture. And they are barely cheaper in memory: 704x384
// peaks 0.5 GiB above 320x192. What a larger canvas costs is time.
const SIZES: [string, number, number][] = [
  ["Large (640x368)", 640, 368],
  ["Wide (704x384)", 704, 384],
  ["Native (832x480)", 832, 480],
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
  // Clamped: a saved index from the old list can point past the end of this
  // one, and an out-of-range index reads as undefined rather than as an error.
  const [sizeIdx, setSizeIdx] = useState(
    () => Math.min(Math.max(0, loadPref("videoSize", 0)), SIZES.length - 1));
  const [frames, setFrames] = useState(() => loadPref("videoFrames", 33));
  const [fps, setFps] = useState(() => loadPref("videoFps", 16));
  const [steps, setSteps] = useState(20);
  const [seed, setSeed] = useState(0);
  const [randomSeed, setRandomSeed] = useState(true);
  const [localFrame, setLocalFrame] = useState<string[]>([]);
  const firstFrame = controlledFrame ?? localFrame;
  const setFirstFrame = onFirstFrameChange ?? setLocalFrame;

  const [assisting, setAssisting] = useState(false);
  /** What the enhancer suggests, pending your decision. */
  const [proposal, setProposal] = useState<AssistResult | null>(null);
  const [undoPrompt, setUndoPrompt] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [prog, setProg] = useState<EngineProgress | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [out, setOut] = useState<string | null>(null);
  // A clip is the longest wait in the app. If the route exposes a step
  // callback this shows it forming, so a wrong composition can be abandoned
  // at step two instead of at the end.
  const [preview, setPreview] = useState<string | null>(null);
  const [took, setTook] = useState<number | null>(null);

  const model = usable.find((m) => m.id === modelId) ?? usable[0];

  // What a larger canvas costs is time, not memory.
  //
  // Measured on this machine, same seed and prompt, 9 frames at 20 steps:
  // 704x384 peaks at 10.20 GiB against 9.72 at 320x192 -- 4.4 times the pixels
  // for half a gigabyte. The time went from 126 seconds to 1571. The old
  // warning here said the opposite, that large sizes would run out of memory,
  // which is what pushed everyone onto a canvas too small to draw on.
  const [, sw, sh] = SIZES[sizeIdx];
  const MEASURED_PIXELS = 640 * 368;
  const MEASURED_FRAMES = 9;
  const MEASURED_SECONDS = 4193;
  const estSeconds = Math.round(
    MEASURED_SECONDS * ((sw * sh * frames) / (MEASURED_PIXELS * MEASURED_FRAMES)));
  const slow = estSeconds > 20 * 60;

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
      // Hand it the starting picture when there is one: a clip that animates
      // a photograph should describe how that photograph moves, not invent a
      // scene from the words alone.
      // Shown, not applied -- the same as in the studio. Overwriting the
      // prompt the moment a rewrite existed hid what changed, and the two ways
      // it could decline without rewriting anything both reported
      // "made your prompt specific", which was untrue and is most of why this
      // button reads as doing nothing.
      setProposal(await api.assistPrompt(id, prompt, "video", firstFrame));
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
    // If this subscription throws, the progress listener above would
    // never be released. Failing to get a preview is not worth leaking one.
    let unp: (() => void) | undefined;
    try { unp = await onEnginePreview((f) => { if (f.jobId === id) setPreview(f.src); }); }
    catch { unp = undefined; }
    const t0 = performance.now();
    try {
      const res = await api.generateVideo({
        jobId: id, modelId: model.id, prompt, negativePrompt: null,
        width: w, height: h, frames, fps, steps,
        guidance: model.guidance_default, seed: s,
        // A picture carried in from another tab stays in state when the model
        // is switched to one that cannot start from a still, and the drop zone
        // that would show it is hidden for exactly that model -- so it was
        // sent invisibly and the run failed with "this model cannot start
        // from a picture", about a picture the screen was not showing.
        firstFrame: model.video_from_image === false
          ? null : (firstFrame[0] ?? null),
      });
      setOut(res[0] ?? null);
      setTook(performance.now() - t0);
      setSeed(s);
      onProduced();
    } catch (e) {
      notify(isCancelled(e) ? "Cancelled." : errText(e), !isCancelled(e));
    } finally {
      un(); unp?.(); setRunning(false); setJobId(null); setProg(null);
      setPreview(null);
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
              {firstFrame.length > 0 && " The picture you carried over is not "
                + "being used; it stays where it came from."}
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
              onChange={(e) => { setPrompt(e.target.value); setUndoPrompt(null); setProposal(null); }}
              placeholder="steam rises slowly from the teapot, the light shifts"
            />
            {undoPrompt !== null && (
              <button className="btn small" style={{ marginTop: 6 }}
                      onClick={() => { setPrompt(undoPrompt); setUndoPrompt(null); }}>
                Undo the rewrite
              </button>
            )}
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
            {proposal && (
              <PromptProposal
                result={proposal}
                busy={assisting}
                onAccept={(text) => {
                  setUndoPrompt(prompt);
                  setPrompt(text);
                  setProposal(null);
                  notify("Prompt updated. Undo is next to the box.");
                }}
                onDismiss={() => setProposal(null)}
              />
            )}
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
            {model && (
              <div style={{ fontSize: 10.5, marginTop: 5, lineHeight: 1.55,
                            color: slow ? "var(--warn)" : "var(--text-faint)" }}>
                {`Roughly ${estSeconds < 90 ? `${estSeconds} seconds`
                    : `${Math.round(estSeconds / 60)} minutes`}, `
                  + `from a measured 640×368 run at 9 frames. Memory barely `
                  + `moves with size — about ${model.peak_gib.toFixed(1)} GiB `
                  + `either way. Time is what a larger canvas costs.`}
                {slow ? " This one is a long wait." : ""}
                {" "}Running a video model well below the size it was trained
                at is its own quality problem, so prefer the largest that runs.
              </div>
            )}
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
          ) : preview ? (
            <div style={{ textAlign: "center" }}>
              <img src={preview} alt="" style={{
                maxWidth: "100%", maxHeight: "64vh", borderRadius: 6, opacity: 0.92,
              }} />
              <div style={{ fontSize: 11, color: "var(--text-faint)", marginTop: 8 }}>
                Taking shape{prog?.step && prog?.total_steps
                  ? ` \u2014 step ${prog.step} of ${prog.total_steps}` : "\u2026"}
              </div>
            </div>
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
