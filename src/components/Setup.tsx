import { useEffect, useRef, useState } from "react";
import { api, errText, onSetupProgress } from "../lib/api";
import type { HostInfo, SetupProgress, SetupState } from "../lib/types";
import { Bar } from "./shared";

const STEPS: [string, string][] = [
  ["python", "Install a private Python 3.12 runtime"],
  ["venv", "Create an isolated environment"],
  ["engine", "Install mlx-gen, MLX and PyTorch (~3 GB)"],
  ["verify", "Verify the installation"],
];

export default function Setup({
  host, state, onDone,
}: {
  host: HostInfo | null;
  state: SetupState;
  onDone: () => void;
}) {
  const [running, setRunning] = useState(false);
  const [prog, setProg] = useState<SetupProgress | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const logRef = useRef<string[]>([]);

  useEffect(() => {
    let un: (() => void) | undefined;
    // Subscribing is async. Unmounting before it resolves would run the
    // cleanup against nothing, and the listener would attach afterwards with
    // nothing left to remove it.
    let live = true;
    void onSetupProgress((p) => {
      setProg(p);
      if (p.detail) {
        logRef.current = [...logRef.current.slice(-40), p.detail];
      }
      if (p.done) {
        setRunning(false);
        if (p.error) setErr(p.error);
        else onDone();
      }
    }).then((u) => { if (live) un = u; else u(); });
    return () => { live = false; un?.(); };
  }, [onDone]);

  const start = async (force = false) => {
    setErr(null);
    setRunning(true);
    logRef.current = [];
    try {
      await api.runSetup(force);
    } catch (e) {
      setErr(errText(e));
      setRunning(false);
    }
  };

  const stepIndex = STEPS.findIndex(([k]) => k === prog?.step);
  const notSilicon = host && !host.apple_silicon;
  const lowDisk = host && host.free_disk_gib < 12;

  return (
    <div className="setup-wrap">
      <div className="setup-card">
        <h1>Set up the local engine</h1>
        <p className="lede">
          Image models run through Python and Apple&rsquo;s MLX framework. Model Studio installs
          its own private runtime so nothing touches your system Python. Everything lands in
          one folder you can delete later.
        </p>

        {notSilicon && (
          <div className="notice bad" style={{ marginBottom: 14 }}>
            <strong>Apple Silicon required</strong>
            MLX has no Intel Mac backend, so local generation is not possible on this machine.
          </div>
        )}

        {host && !notSilicon && (
          <div className="notice" style={{ marginBottom: 14 }}>
            <strong>{host.chip} · {host.total_ram_gib.toFixed(0)} GB unified memory</strong>
            About {host.usable_ram_gib.toFixed(1)} GB is realistically available to a model, and{" "}
            {host.free_disk_gib.toFixed(0)} GB of disk is free. The runtime itself needs roughly
            3 GB before any model is downloaded.
          </div>
        )}

        {lowDisk && !notSilicon && (
          <div className="notice warn" style={{ marginBottom: 14 }}>
            <strong>Free disk is very low</strong>
            The engine alone needs about 3 GB, and the smallest useful model adds 3.6 GB more.
            Free up space before continuing.
          </div>
        )}

        <div className="panel">
          <h2>What gets installed</h2>
          <div className="steps">
            {STEPS.map(([key, label], i) => {
              const done = running ? i < stepIndex : state.ready;
              const active = running && i === stepIndex;
              return (
                <div key={key} className={"step" + (done ? " done" : active ? " active" : "")}>
                  <span className="mark">
                    {done ? "✓" : active ? <span className="spin">◐</span> : "○"}
                  </span>
                  {label}
                </div>
              );
            })}
          </div>

          {running && (
            <>
              <Bar value={prog?.progress ?? null} />
              <div className="setup-log">
                {logRef.current.slice(-5).join("\n") || "Starting…"}
              </div>
            </>
          )}

          {err && (
            <div className="notice bad" style={{ marginTop: 12 }}>
              <strong>Setup failed</strong>
              <span style={{ whiteSpace: "pre-wrap" }}>{err}</span>
            </div>
          )}

          <button
            className="btn primary full"
            style={{ marginTop: 14 }}
            disabled={running || !!notSilicon}
            onClick={() => start(false)}
          >
            {running ? "Installing…" : err ? "Try again" : "Install runtime"}
          </button>
          {err && !running && (
            <button
              className="btn full"
              style={{ marginTop: 8 }}
              onClick={() => start(true)}
              title="Discard what was installed and start over"
            >
              Reinstall from scratch
            </button>
          )}
          {err && !running && (
            <div style={{ marginTop: 8, fontSize: 10.5, color: "var(--text-faint)", lineHeight: 1.6 }}>
              &ldquo;Try again&rdquo; resumes and keeps whatever downloaded correctly.
              Reinstalling throws that away, which is what you want if a previous
              attempt left something half-written.
            </div>
          )}
          <div style={{ marginTop: 9, fontSize: 11, color: "var(--text-faint)" }}>
            Installs to <code style={{ userSelect: "text" }}>{state.root}</code>
          </div>
        </div>
      </div>
    </div>
  );
}
