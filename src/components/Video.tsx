import { useEffect, useMemo, useState } from "react";
import { api, errText, isCancelled, fmtDuration, newJobId, onEngineProgress, onEnginePreview, vaultUrl } from "../lib/api";
import type { AssistGoal, AssistResult, EngineProgress, ModelStatus } from "../lib/types";
import { humanDuration } from "../lib/presets";
import { exportItem, ImageDrop, JobProgress } from "./shared";
import { loadPref, savePref } from "../lib/prefs";
import PromptProposal, { PromptGoal } from "./PromptProposal";
import { ANALYST_IDS } from "../lib/story";
import Field from "./Field";
import { LiveStatus, ResultStrip, RunError, useRunShortcut, useStatus, type Made } from "./creation";

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

/**
 * How long a clip takes, from one measured run.
 *
 * There were two estimators on this screen: the size hint scaled a measured
 * run, and the button quoted a made-up shape that ignored the measurement, so
 * the same settings were promised two different waits a few lines apart.
 * This is the only one now. The measurement was taken at 20 steps, and the
 * denoise cost is linear in steps.
 */
export function estimateClipSeconds(w: number, h: number, frames: number, steps: number): number {
  const MEASURED_PIXELS = 640 * 368;
  const MEASURED_FRAMES = 9;
  const MEASURED_STEPS = 20;
  const MEASURED_SECONDS = 4193;
  return Math.max(1, Math.round(
    MEASURED_SECONDS * ((w * h * frames) / (MEASURED_PIXELS * MEASURED_FRAMES))
      * (steps / MEASURED_STEPS)));
}

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
  /** The rewrite as it is being written. See `improve`. */
  const [draft, setDraft] = useState("");
  const [undoPrompt, setUndoPrompt] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [prog, setProg] = useState<EngineProgress | null>(null);
  const [elapsed, setElapsed] = useState(0);
  /** Clips made this session, newest first. Each run used to replace the last. */
  const [made, setMade] = useState<(Made & { took: number })[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  /** Why the last run failed, kept on the canvas until the next one. */
  const [failure, setFailure] = useState<string | null>(null);
  const [promptError, setPromptError] = useState<string | null>(null);
  const [status, setStatus] = useStatus(prog);
  // A clip is the longest wait in the app. If the route exposes a step
  // callback this shows it forming, so a wrong composition can be abandoned
  // at step two instead of at the end.
  const [preview, setPreview] = useState<string | null>(null);

  const model = usable.find((m) => m.id === modelId) ?? usable[0];

  // What a larger canvas costs is time, not memory.
  //
  // Measured on this machine, same seed and prompt, 9 frames at 20 steps:
  // 704x384 peaks at 10.20 GiB against 9.72 at 320x192 -- 4.4 times the pixels
  // for half a gigabyte. The time went from 126 seconds to 1571. The old
  // warning here said the opposite, that large sizes would run out of memory,
  // which is what pushed everyone onto a canvas too small to draw on.
  const [, sw, sh] = SIZES[sizeIdx];
  const estSeconds = estimateClipSeconds(sw, sh, frames, steps);
  const slow = estSeconds > 20 * 60;

  const assistantReady = useMemo(
    // The writer, specifically. There are two assist models now, and `.some`
    // was satisfied by the picture-reader alone -- so the button looked ready
    // and then failed with "the prompt writer is not installed".
    // Any writer: the engine uses the best one installed.
    () => models.some((m) => ANALYST_IDS.includes(m.id) && m.installed && !m.hopeless),
    [models]
  );
  const [goal, setGoal] = useState<AssistGoal>(() => loadPref("promptGoal", "enhance" as AssistGoal));
  useEffect(() => { savePref("promptGoal", goal); }, [goal]);

  /** Add the motion a video model is steered by, keeping the user's words. */
  const improve = async () => {
    if (!prompt.trim()) { setPromptError("Write a rough idea first."); return; }
    const id = newJobId();
    setAssisting(true);
    setDraft("");
    const un = await onEngineProgress((p) => {
      if (p.job_id !== id) return;
      setProg(p);
      // The words as they arrive, the same as in the studio. The engine has
      // streamed them all along and this screen registered the listener and
      // discarded `p.text` -- so the video assist button was still the spinner
      // that was removed from the studio for being one, and this rewrite takes
      // longer than that one did.
      if (typeof p.text === "string") setDraft(p.text);
    });
    try {
      // Hand it the starting picture when there is one: a clip that animates
      // a photograph should describe how that photograph moves, not invent a
      // scene from the words alone.
      // Shown, not applied -- the same as in the studio. Overwriting the
      // prompt the moment a rewrite existed hid what changed, and the two ways
      // it could decline without rewriting anything both reported
      // "made your prompt specific", which was untrue and is most of why this
      // button reads as doing nothing.
      setProposal(await api.assistPrompt(id, prompt, "video", firstFrame,
                                         { modelId: model?.id ?? null, goal }));
      setStatus("Suggestion ready, below the prompt.");
    } catch (e) {
      notify(errText(e), true);
    } finally {
      un(); setAssisting(false); setProg(null); setDraft("");
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

  const run = async () => {
    if (!model || running) return;
    if (!prompt.trim()) { setPromptError("Describe the clip first."); return; }
    setPromptError(null);
    const id = newJobId();
    const s = randomSeed ? Math.floor(Math.random() * 2_000_000_000) : seed;
    // The last clip stays, dimmed, until the new one lands: clearing it at
    // the start left a failed run with nothing on screen at all. The live
    // preview still takes precedence, so a second run does not look idle.
    setFailure(null); setStatus("Starting");
    setRunning(true); setJobId(id); setProg(null);
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
      const took = performance.now() - t0;
      if (res[0]) {
        setMade((m) => [{ id: res[0], prompt, seed: s, took }, ...m]);
        setSelectedId(res[0]);
        setStatus("Done, 1 clip.");
      } else {
        setFailure("The engine finished but returned no clip.");
      }
      setSeed(s);
      onProduced();
    } catch (e) {
      if (isCancelled(e)) { setStatus("Cancelled."); notify("Cancelled."); }
      else { setStatus(""); setFailure(errText(e)); }
    } finally {
      un(); unp?.(); setRunning(false); setJobId(null); setProg(null);
      setPreview(null);
    }
  };

  const shortcut = useRunShortcut(() => { void run(); }, !running && usable.length > 0);

  if (usable.length === 0) {
    return (
      <div className="empty-state">
        <span className="big" aria-hidden="true">▷</span>
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

  const current = made.find((m) => m.id === selectedId) ?? made[0] ?? null;
  const currentLabel = current
    ? `Clip ${made.indexOf(current) + 1} of ${made.length}: ${current.prompt}` : "";
  const remaining = Math.max(0, estSeconds - elapsed);

  return (
    <div className="content split creation" style={{ padding: 0 }}
         ref={shortcut.ref} onKeyDown={shortcut.onKeyDown}>
      <LiveStatus text={status} />
      <div className="creation-params">
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
              <div className="choice-label">
                Starting picture <em>optional</em>
              </div>
              <ImageDrop
                images={firstFrame}
                onChange={setFirstFrame}
                max={model?.max_edit_images ?? 1}
                onError={(m) => notify(m, true)}
              />
              <div className="t-help" style={{ marginTop: 5 }}>
                {controlledFrame && controlledFrame.length > 0
                  ? "Carried over from another tab. This picture becomes the first frame."
                  : "With a picture the clip animates it. Without one it is made from the description alone."}
              </div>
            </div>
          )}

          <div className="field">
            <Field className="prompt-field" label="What should happen?" error={promptError ?? undefined}>
              <textarea
                value={prompt}
                onChange={(e) => {
                  setPrompt(e.target.value); setUndoPrompt(null); setProposal(null);
                  setPromptError(null);
                }}
                placeholder="steam rises slowly from the teapot, the light shifts"
              />
            </Field>
            {undoPrompt !== null && (
              <button className="btn small" style={{ marginTop: 6 }}
                      onClick={() => { setPrompt(undoPrompt); setUndoPrompt(null); }}>
                Undo the rewrite
              </button>
            )}
            <PromptGoal value={goal} onChange={(g) => { setGoal(g); setProposal(null); }}
                        disabled={assisting} />
            <button
              className="btn small full"
              style={{ marginTop: 7 }}
              disabled={assisting || running || !prompt.trim()}
              onClick={improve}
              title={assistantReady
                ? "Keeps the pace you asked for and says how the subject and the "
                  + "camera move. Shown next to your words before anything changes."
                : "Needs the prompt writer from the Models tab"}
            >
              {assisting ? "Thinking…"
                : goal === "style" ? "Suggest styles"
                : `${goal === "tighten" ? "Tighten" : "Enhance"} my prompt`}
            </button>
            {assisting && draft && (
              // Silent on purpose: read aloud, every token would interrupt
              // the last. Completion is announced once instead.
              <div className="proposal plain" aria-live="off">
                <b>Writing…</b>
                <div className="text">{draft}</div>
              </div>
            )}
            {proposal && (
              <PromptProposal
                result={proposal}
                busy={assisting}
                onAccept={(text) => {
                  setUndoPrompt(prompt);
                  setPrompt(text);
                  setProposal(null);
                  notify("Prompt updated. Undo is below the box.");
                }}
                onDismiss={() => setProposal(null)}
              />
            )}
            {!assistantReady && (
              <div className="t-help" style={{ marginTop: 5 }}>
                Needs the prompt writer — a 2.1 GB download in the Models tab.
                It runs on this Mac; nothing is sent anywhere.
              </div>
            )}
          </div>

          <Field
            label="Size"
            hint={model && (
              <span className="t-note" style={{ color: slow ? "var(--warn)" : undefined }}>
                {`Takes ${humanDuration(estSeconds)}. `
                  + `Memory barely moves with size — about ${model.peak_gib.toFixed(1)} GiB `
                  + `either way; time is what a larger canvas costs.`}
                {slow ? " This one is a long wait." : ""}
                {" "}Prefer the largest size that runs: below the size it was
                trained at, a video model loses quality.
              </span>
            )}
          >
            <select value={sizeIdx} onChange={(e) => setSizeIdx(+e.target.value)}>
              {SIZES.map(([label, sw, sh], i) => (
                <option key={label} value={i}>{label} — {sw}×{sh}</option>
              ))}
            </select>
          </Field>

          <Field
            label={<>Frames <em aria-hidden="true">{frames} · {seconds}s at {fps} fps</em></>}
            hint="The main cost. Halving the frames saves more than half the time."
          >
            <input
              type="range" min={9} max={81} step={4} value={frames}
              aria-valuetext={`${frames} frames, ${seconds} seconds`}
              onChange={(e) => setFrames(+e.target.value)}
            />
          </Field>

          <div className="row">
            <Field label={<>Playback speed <em aria-hidden="true">{fps} fps</em></>}
              hint="Changes the clip's length only, not the wait.">
              <input type="range" min={8} max={30} step={1} value={fps}
                aria-valuetext={`${fps} frames per second`}
                onChange={(e) => setFps(+e.target.value)} />
            </Field>
            <Field label={<>Steps <em aria-hidden="true">{steps}</em></>}>
              <input type="range" min={4} max={40} value={steps}
                aria-valuetext={`${steps} steps`}
                onChange={(e) => setSteps(+e.target.value)} />
            </Field>
          </div>

          {/* The seed was always sent and always reported, but never
              settable, so a clip worth refining could not be made again. */}
          <div className="row" style={{ alignItems: "flex-end" }}>
            <Field label="Seed" className="field">
              <input type="number" value={seed} disabled={randomSeed}
                onChange={(e) => setSeed(+e.target.value)} />
            </Field>
            <div className="field">
              <label className="check">
                <input type="checkbox" checked={randomSeed}
                  onChange={(e) => setRandomSeed(e.target.checked)} />
                Different every time
              </label>
            </div>
          </div>

          {usable.length > 1 ? (
            <Field label="Model">
              <select value={model?.id ?? ""} onChange={(e) => setModelId(e.target.value)}>
                {usable.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} — ~{m.peak_gib.toFixed(1)} GiB
                  </option>
                ))}
              </select>
            </Field>
          ) : (
            <div className="t-note" style={{ marginBottom: 8 }}>
              Using <b>{model?.name}</b>
            </div>
          )}
        </div>

        <div className="run-footer">
          {running ? (
            <>
              <JobProgress p={prog} label="Rendering" />
              <div className="t-note run-footer-times">
                <span>{elapsed}s elapsed</span>
                <span>
                  {remaining > 0
                    ? `${humanDuration(remaining)} left, roughly`
                    : "longer than expected"}
                </span>
              </div>
              <button className="btn danger full" style={{ marginTop: 8 }}
                onClick={() => jobId && api.cancelJob(jobId).catch(() => {})}>
                Cancel
              </button>
            </>
          ) : (
            <>
              <button className="btn primary full" onClick={() => void run()}
                aria-keyshortcuts="Meta+Enter">
                Make the clip<span className="kbd" aria-hidden="true">⌘↵</span>
              </button>
              <div className="t-note" style={{ marginTop: 6, textAlign: "center" }}>
                Takes {humanDuration(estSeconds)}, scaled from a measured run
              </div>
            </>
          )}
        </div>
      </div>

      <div className="creation-canvas">
        <div className="canvas">
          {failure && !running ? (
            <div className="canvas-stack">
              <RunError message={failure} onRetry={() => void run()} />
            </div>
          ) : running && preview ? (
            <div style={{ textAlign: "center" }}>
              <img src={preview} alt="Preview of the clip forming" style={{
                maxWidth: "100%", maxHeight: "64vh", borderRadius: 7, opacity: 0.92,
              }} />
              <div className="t-note" style={{ marginTop: 8 }}>
                Taking shape{prog?.step && prog?.total_steps
                  ? ` — step ${prog.step} of ${prog.total_steps}` : "…"}
              </div>
            </div>
          ) : current ? (
            <div className="canvas-stack">
              <video
                key={current.id}
                className={running ? "stale" : undefined}
                src={vaultUrl(current.id)}
                aria-label={currentLabel}
                controls
                loop
                autoPlay={!running}
                muted
                style={{ maxWidth: "100%", maxHeight: "64vh", borderRadius: 7 }}
              />
              {running && (
                <div className="t-note">
                  Rendering — the last clip stays until the new one lands.
                </div>
              )}
            </div>
          ) : (
            <div className="empty">
              <span className="big" aria-hidden="true">▷</span>
              {running
                ? "Rendering. The model loads first, then denoises every frame."
                : "Describe a clip, or drop in a picture to animate."}
            </div>
          )}
        </div>
        {current && (
          <>
            <ResultStrip items={made} selected={current.id} onSelect={setSelectedId} video />
            <div className="result-actions">
              <button className="btn small"
                onClick={() => exportItem({ id: current.id, name: "clip.mp4" }, notify)}>
                Save a copy…
              </button>
              <button className="btn small"
                onClick={() => {
                  setSeed(current.seed); setRandomSeed(false);
                  notify(`Seed ${current.seed} will be used for the next clip.`);
                }}>
                Reuse seed
              </button>
              <button className="btn small"
                disabled={current.prompt === prompt}
                onClick={() => { setUndoPrompt(prompt); setPrompt(current.prompt); }}>
                Use prompt
              </button>
              <span className="t-note" style={{ marginLeft: "auto" }}>
                Took {fmtDuration(Math.round(current.took))} · seed {current.seed}
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
