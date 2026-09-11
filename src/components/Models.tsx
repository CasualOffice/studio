import { useState } from "react";
import { api, errText, fmtBytes, newJobId, onEngineProgress } from "../lib/api";
import type { EngineProgress, Fit, HostInfo, ModelStatus } from "../lib/types";
import { JobProgress } from "./shared";
import AddModel from "./AddModel";

const FIT_PILL: Record<Fit, { cls: string; label: string }> = {
  good: { cls: "good", label: "Fits" },
  tight: { cls: "tight", label: "Tight" },
  too_much_memory: { cls: "bad", label: "Too much memory" },
  too_much_disk: { cls: "bad", label: "Too large for disk" },
  broken: { cls: "bad", label: "Unavailable" },
};

export default function Models({
  models, host, onChanged, notify,
}: {
  models: ModelStatus[];
  host: HostInfo | null;
  onChanged: () => void;
  notify: (m: string, bad?: boolean) => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [prog, setProg] = useState<EngineProgress | null>(null);

  const download = async (m: ModelStatus) => {
    const id = newJobId();
    setBusy(m.id);
    setJobId(id);
    setProg(null);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    try {
      await api.downloadModel(m.id, id);
      notify(`${m.name} is ready.`);
      onChanged();
    } catch (e) {
      notify(errText(e), true);
    } finally {
      un();
      setBusy(null);
      setJobId(null);
      setProg(null);
    }
  };

  const remove = async (m: ModelStatus) => {
    try {
      const freed = await api.deleteModel(m.id);
      notify(`Removed ${m.name} — freed ${fmtBytes(freed)}.`);
      onChanged();
    } catch (e) {
      notify(errText(e), true);
    }
  };

  // Forgetting a custom entry is separate from deleting its weights, so the
  // user can drop it from the list without re-downloading later.
  const forget = async (m: ModelStatus) => {
    try {
      await api.removeCustomModel(m.id);
      notify(`Removed ${m.name} from your list.`);
      onChanged();
    } catch (e) {
      notify(errText(e), true);
    }
  };

  const cancel = async () => {
    if (jobId) {
      try { await api.cancelJob(jobId); } catch { /* the job may already be gone */ }
    }
  };

  const runnable = models.filter((m) => m.fit === "good" || m.fit === "tight");
  const withLowRam = models.filter((m) => m.low_ram_may_help && !m.hopeless);
  const blocked = models.filter(
    (m) => m.fit !== "good" && m.fit !== "tight" && !(m.low_ram_may_help && !m.hopeless)
  );

  const card = (m: ModelStatus) => {
    const fit = FIT_PILL[m.fit];
    // Only a weight floor above physical memory is genuinely unavailable.
    // A high transient peak is something low-RAM mode can attack.
    const unavailable =
      m.fit === "broken" || m.hopeless || (m.fit === "too_much_disk" && !m.installed);
    return (
      <div className={"model-card" + (unavailable ? " unavailable" : "")} key={m.id}>
        <div className="model-card-head">
          <span className="name">{m.name}</span>
          {m.installed && <span className="pill installed"><span className="dot" />Installed</span>}
          {m.custom && <span className="pill">Added by you</span>}
          <span className={"pill " + fit.cls}>
            {m.low_ram_may_help && !m.hopeless ? "Needs low-RAM mode" : fit.label}
          </span>
          {m.tasks_str.map((t) => <span className="pill" key={t}>{t}</span>)}
        </div>
        <div className="repo">{m.repo}</div>
        <div className="notes">{m.notes}</div>
        <div className="stats">
          <span><b>{m.package_gib.toFixed(1)} GiB</b> download</span>
          <span>
            <b>~{m.peak_gib.toFixed(1)} GiB</b> peak memory
            {m.peak_estimated && <span title="Interpolated, not a published benchmark"> *</span>}
          </span>
          <span><b>~{m.package_gib.toFixed(1)} GiB</b> weight floor</span>
          {m.installed && m.installed_bytes > 0 && (
            <span><b>{fmtBytes(m.installed_bytes)}</b> on disk</span>
          )}
        </div>
        <div style={{ fontSize: 11, color: "var(--text-faint)", marginTop: 6 }}>{m.fit_reason}</div>
        {m.gated && !m.installed && (
          <div style={{ fontSize: 10.5, color: "var(--warn)", marginTop: 5,
                        lineHeight: 1.5 }}>
            Behind a licence. Accept it on the model&rsquo;s Hugging Face page,
            then add an access token under <b>Security</b> — otherwise the
            download stops with a 401 partway through.
          </div>
        )}

        {busy === m.id ? (
          <>
            <JobProgress p={prog} label="Downloading" />
            <div className="actions">
              <button className="btn small danger" onClick={cancel}>Cancel</button>
            </div>
          </>
        ) : (
          <div className="actions">
            {m.installed ? (
              <button className="btn small danger" onClick={() => remove(m)}>
                Delete weights
              </button>
            ) : (
              <button
                className="btn small primary"
                disabled={unavailable || busy !== null}
                onClick={() => download(m)}
              >
                Download {m.package_gib.toFixed(1)} GiB
              </button>
            )}
            {m.custom && (
              <button className="btn small" onClick={() => forget(m)}>
                Forget
              </button>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div>
      {host && (
        <div className="notice" style={{ marginBottom: 16 }}>
          <strong>{host.chip} · {host.total_ram_gib.toFixed(0)} GB memory · {host.free_disk_gib.toFixed(0)} GB disk free</strong>
          A model has to fit twice over: its package on disk, and its peak memory while running.
          Peak memory is the figure that actually decides whether a model works, and it is often
          nothing like the download size.
        </div>
      )}

      <AddModel onAdded={onChanged} notify={notify} host={host} />

      <div className="group-label">Runs on this Mac</div>
      <div className="model-list">{runnable.map(card)}</div>

      {withLowRam.length > 0 && (
        <>
          <div className="group-label">Only with low-RAM mode</div>
          <div className="notice warn" style={{ marginBottom: 10 }}>
            <strong>Above the ceiling, but not out of reach</strong>
            Peak memory here exceeds what this Mac can offer, yet the resident weights fit.
            Low-RAM mode clears the MLX cache at transformer-block boundaries and between
            denoise steps, which is exactly the memory that overflows. Expect slower runs, one
            image at a time, and no guarantee — the published benchmarks put these above this tier.
          </div>
          <div className="model-list">{withLowRam.map(card)}</div>
        </>
      )}

      {blocked.length > 0 && (
        <>
          <div className="group-label">Beyond this Mac</div>
          <div className="model-list">{blocked.map(card)}</div>
        </>
      )}

      <div style={{ marginTop: 18, fontSize: 11, color: "var(--text-faint)", lineHeight: 1.7 }}>
        * Peak memory is interpolated where MLX-Gen has not published a benchmark for that exact
        package. Figures without a star are measured.
        <br />
        The weight floor is what has to be resident throughout a run. Low-RAM mode and a tighter
        buffer cache reduce the gap between that floor and the peak; neither reduces the floor
        itself, so a model whose weights exceed memory is never offered.
        <br />
        Wan 2.5 and 2.6 are not listed because Alibaba never released their weights — they exist
        only as a cloud API. Wan 2.1 and 2.2 are the open ones, and 2.2 is what appears above.
      </div>
    </div>
  );
}
