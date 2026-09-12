import { useMemo, useState } from "react";
import { api, errText, fmtBytes, newJobId, onEngineProgress } from "../lib/api";
import type { EngineProgress, ModelStatus } from "../lib/types";
import { JobProgress } from "./shared";

/**
 * First run, for someone who has not come here to choose a model.
 *
 * The Models tab lists twenty-six of them with memory ceilings and
 * quantisation levels, which is the right screen for the second visit and the
 * wrong one for the first. This asks for a single decision and then does the
 * rest: install the few that make every tab work, in the order that makes the
 * app usable soonest.
 */

/** What each one buys, in the order they are installed. */
const ESSENTIALS: { id: string; gives: string }[] = [
  { id: "flux2-klein-4b-4bit",
    gives: "makes and edits pictures, and draws picture boards" },
  { id: "qwen3-4b-instruct-4bit",
    gives: "turns a story into panels, and makes prompts precise" },
  { id: "seedvr2-3b-4bit",
    gives: "enlarges a picture without smearing it" },
];

export default function Welcome({
  models, notify, onChanged, onSkip,
}: {
  models: ModelStatus[];
  notify: (m: string, bad?: boolean) => void;
  onChanged: () => void;
  onSkip: () => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [prog, setProg] = useState<EngineProgress | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [doneIds, setDoneIds] = useState<string[]>([]);

  // Only offer what this Mac can actually run. A recommendation that fails on
  // the machine it was recommended for is worse than no recommendation.
  const picks = useMemo(
    () => ESSENTIALS
      .map((e) => ({ ...e, m: models.find((m) => m.id === e.id) }))
      .filter((e): e is typeof e & { m: ModelStatus } =>
        Boolean(e.m) && !e.m!.hopeless && e.m!.fit !== "broken"),
    [models]
  );
  const pending = picks.filter((p) => !p.m.installed);
  const totalBytes = pending.reduce(
    (n, p) => n + p.m.package_gib * 1024 ** 3, 0);

  const installAll = async () => {
    let failures = 0;
    let cancelled = false;
    for (const p of pending) {
      const id = newJobId();
      setBusy(p.m.id); setJobId(id); setProg(null);
      const un = await onEngineProgress((e) => { if (e.job_id === id) setProg(e); });
      try {
        await api.downloadModel(p.m.id, id);
        setDoneIds((d) => [...d, p.m.id]);
        onChanged();
      } catch (e) {
        const msg = errText(e);
        notify(`${p.m.name}: ${msg}`, true);
        // One failure should not abandon the rest; the others still help.
        if (msg.includes("ancelled")) { cancelled = true; break; }
        failures++;
      } finally {
        un(); setBusy(null); setJobId(null); setProg(null);
      }
    }
    if (cancelled) {
      notify("Download stopped. You can continue with the models already installed.");
    } else if (failures > 0) {
      notify(`${failures} model download${failures === 1 ? "" : "s"} failed. Retry before continuing.`, true);
    } else {
      notify("Ready. Everything is on this Mac; nothing gets sent anywhere.");
      onSkip();
    }
  };

  const cancel = async () => {
    if (jobId) { try { await api.cancelJob(jobId); } catch { /* gone */ } }
  };

  return (
    <div className="content" style={{ maxWidth: 680, margin: "0 auto" }}>
      <div className="panel">
        <h2>Welcome</h2>
        <div style={{ fontSize: 13, lineHeight: 1.65, color: "var(--text-dim)",
                      marginBottom: 16 }}>
          This app makes and edits pictures on your Mac. Nothing is uploaded,
          there is no account, and everything it makes is encrypted on disk.
          <br /><br />
          It needs a few models first. They are large, they download once, and
          they stay on this machine.
        </div>

        {picks.map((p) => {
          const installed = p.m.installed || doneIds.includes(p.m.id);
          return (
            <div key={p.id} style={{ display: "flex", gap: 10, marginBottom: 12,
                                     alignItems: "flex-start" }}>
              <span style={{ width: 16, flex: "0 0 16px", marginTop: 1,
                             color: installed ? "var(--good)" : "var(--text-faint)" }}>
                {installed ? "✓" : busy === p.m.id ? "◐" : "○"}
              </span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 12.5 }}>
                  {p.gives}
                  {!installed && (
                    <span style={{ color: "var(--text-faint)" }}>
                      {" "}· {p.m.package_gib.toFixed(1)} GB
                    </span>
                  )}
                </div>
                <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 1 }}>
                  {p.m.name}
                </div>
              </div>
            </div>
          );
        })}

        {busy && <JobProgress p={prog} label="Downloading" />}

        <div style={{ display: "flex", gap: 7, marginTop: 16, alignItems: "center" }}>
          {busy ? (
            <button className="btn small" onClick={cancel}>Cancel</button>
          ) : (
            <>
              <button className="btn primary small"
                      disabled={pending.length === 0}
                      onClick={() => void installAll()}>
                {pending.length === 0
                  ? "Everything is installed"
                  : `Install these — ${fmtBytes(totalBytes)}`}
              </button>
              <button className="btn small" onClick={onSkip}>
                I&rsquo;ll choose myself
              </button>
            </>
          )}
        </div>

        <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 12,
                      lineHeight: 1.55 }}>
          You can change any of this later in <b>Models</b>, and delete anything
          you stop using. Downloads resume if they are interrupted.
        </div>
      </div>
    </div>
  );
}
