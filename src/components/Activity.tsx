import { useEffect, useState } from "react";
import { api, errText, fmtBytes, onEngineLog, onEngineStderr } from "../lib/api";
import type { EngineLog } from "../lib/api";

interface Health {
  python?: string;
  mlxgen?: string;
  mlx?: string;
  resident_model?: string | null;
  active_bytes?: number | null;
  peak_bytes?: number | null;
  threads?: string;
  memory_budget_gib?: string | null;
}

/**
 * What the engine is doing and what it is holding.
 *
 * The worker degrades quietly by design — a parameter a route rejects is
 * dropped rather than failing the run — so there has to be somewhere those
 * decisions are visible, or a silently different result looks like a bug.
 */
export default function Activity({ notify }: { notify: (m: string, bad?: boolean) => void }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [busy, setBusy] = useState(false);
  const [logs, setLogs] = useState<EngineLog[]>([]);
  const [showStderr, setShowStderr] = useState(false);
  // State, not a ref. The tail was held in a ref, and writing to one does not
  // re-render: the raw output panel -- the thing you open precisely because a
  // run appears to have hung -- showed only the lines that had arrived before
  // it was opened and then never moved while you watched it.
  const [stderr, setStderr] = useState<string[]>([]);

  useEffect(() => {
    // Subscribing is async, so the component can unmount before either
    // listener is registered. Without the flag the cleanup runs against an
    // empty list, the listener attaches afterwards, and nothing ever removes
    // it -- one leaked listener per visit to this tab.
    let live = true;
    const uns: (() => void)[] = [];
    const keep = (u: () => void) => { live ? uns.push(u) : u(); };
    void onEngineLog((l) => setLogs((prev) => [...prev.slice(-199), l])).then(keep);
    void onEngineStderr((line) => {
      // Keep a bounded tail: PyTorch and tokenizers are extremely chatty.
      setStderr((prev) => [...prev.slice(-299), line]);
    }).then(keep);
    return () => { live = false; uns.forEach((u) => u()); };
  }, []);

  const refresh = async () => {
    setBusy(true);
    try {
      setHealth((await api.enginePing()) as Health);
    } catch (e) {
      notify(errText(e), true);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => { void refresh(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div style={{ maxWidth: 720 }}>
      <div className="panel">
        <div style={{ display: "flex", alignItems: "center", marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>Engine</h2>
          <button className="btn small" style={{ marginLeft: "auto" }}
            disabled={busy} onClick={refresh}>
            {busy ? "Checking…" : "Refresh"}
          </button>
        </div>
        {health ? (
          <>
            <div className="stats" style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
              <span><b>Python</b> {health.python ?? "—"}</span>
              <span><b>mlx-gen</b> {health.mlxgen ?? "—"}</span>
              <span><b>MLX</b> {health.mlx ?? "—"}</span>
            </div>
            <div className="field" style={{ marginTop: 12 }}>
              <label>Model held in memory</label>
              <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
                {health.resident_model ?? "none — the next run will load one"}
              </div>
            </div>
            <div className="notice" style={{ marginTop: 12, marginBottom: 12 }}>
              <strong>
                Limited to {health.memory_budget_gib ?? "?"} GB of memory
                {health.threads ? ` and ${health.threads} threads` : ""}
              </strong>
              Past that ceiling MLX refuses the allocation instead of letting
              macOS swap, and the engine runs at lower priority so the rest of
              the machine stays responsive. Models are released after 10 minutes
              idle.
            </div>
            {(health.active_bytes ?? 0) > 0 && (
              <div className="stats" style={{ display: "flex", gap: 18 }}>
                <span><b>{fmtBytes(health.active_bytes ?? 0)}</b> in use by MLX</span>
                <span><b>{fmtBytes(health.peak_bytes ?? 0)}</b> peak this session</span>
              </div>
            )}
          </>
        ) : (
          <div style={{ fontSize: 12, color: "var(--text-faint)" }}>
            Starting the engine…
          </div>
        )}
      </div>

      <div className="panel">
        <div style={{ display: "flex", alignItems: "center", marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>Activity</h2>
          <button className="btn small" style={{ marginLeft: "auto" }}
            onClick={() => setShowStderr(!showStderr)}>
            {showStderr ? "Hide raw output" : "Raw output"}
          </button>
        </div>
        <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginBottom: 10, lineHeight: 1.6 }}>
          Models disagree about which settings they accept. Rather than fail a run
          after the weights are already loaded, the engine drops a setting the
          model rejects and says so here — so a result that came out differently
          than you expected has an explanation.
        </div>
        {logs.length === 0 ? (
          <div style={{ fontSize: 12, color: "var(--text-faint)" }}>
            Nothing yet. Generate something and it will appear here.
          </div>
        ) : (
          <div className="setup-log" style={{ maxHeight: 260 }}>
            {logs.slice().reverse().map((l, i) => (
              <div key={i} style={{ color: l.level === "warn" ? "var(--warn)" : undefined }}>
                {new Date(l.at).toLocaleTimeString()}  {l.message}
              </div>
            ))}
          </div>
        )}
        {showStderr && (
          <div className="setup-log" style={{ maxHeight: 200, marginTop: 8 }}>
            {stderr.slice(-60).join("\n") || "(nothing)"}
          </div>
        )}
      </div>
    </div>
  );
}
