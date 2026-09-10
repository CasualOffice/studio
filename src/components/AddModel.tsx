import { useState } from "react";
import { api, errText, fmtBytes } from "../lib/api";
import type { ResolvedModel } from "../lib/types";

const TASK_LABEL: Record<string, string> = {
  text_to_image: "Generate",
  edit: "Edit",
  upscale: "Upscale",
  video: "Video",
};

/**
 * Add any Hugging Face repo MLX-Gen can route. Resolution happens before the
 * download is offered, so an unroutable or oversized repo is caught without
 * spending a byte.
 */
export default function AddModel({
  onAdded, notify,
}: {
  onAdded: () => void;
  notify: (m: string, bad?: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
  const [repo, setRepo] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [resolved, setResolved] = useState<ResolvedModel | null>(null);

  const reset = () => {
    setRepo(""); setName(""); setResolved(null); setOpen(false);
  };

  const check = async () => {
    setBusy(true); setResolved(null);
    try {
      const r = await api.resolveModel(repo);
      setResolved(r);
      if (!name.trim()) setName(repo.split("/").pop() ?? repo);
    } catch (e) {
      notify(errText(e), true);
    } finally {
      setBusy(false);
    }
  };

  const add = async () => {
    if (!resolved) return;
    setBusy(true);
    try {
      await api.addCustomModel(
        resolved.model, name, resolved.tasks, resolved.bytes, null, resolved.family
      );
      notify(`Added ${name}. Download it from the list below.`);
      onAdded();
      reset();
    } catch (e) {
      notify(errText(e), true);
    } finally {
      setBusy(false);
    }
  };

  if (!open) {
    return (
      <button className="btn small" style={{ marginBottom: 16 }} onClick={() => setOpen(true)}>
        + Add a model by repo id
      </button>
    );
  }

  const runsHere = resolved ? resolved.fit === "good" || resolved.fit === "tight" : false;

  return (
    <div className="panel" style={{ marginBottom: 16 }}>
      <h2>Add a model</h2>
      <div className="field">
        <label>Hugging Face model</label>
        <input
          type="text"
          value={repo}
          autoFocus
          spellCheck={false}
          placeholder="Paste a link, or type owner/name"
          onChange={(e) => { setRepo(e.target.value); setResolved(null); }}
          onKeyDown={(e) => { if (e.key === "Enter" && repo.trim().length > 2) void check(); }}
        />
        <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5, lineHeight: 1.55 }}>
          Paste a model link from huggingface.co, or type <code>owner/name</code>.
          It is checked against this Mac before anything downloads.
          <div style={{ marginTop: 6 }}>
            The engine runs the <b>FLUX.2</b>, <b>Qwen-Image</b>, <b>Z-Image</b>,
            <b> ERNIE</b>, <b>FIBO</b>, <b>Bonsai</b> and <b>Wan</b> families.
            Stable Diffusion, SDXL and FLUX.1 are different architectures and
            will not run. Pre-quantized MLX packages work best — a
            full-precision repository is usually far too large for this Mac.
          </div>
        </div>
      </div>

      {resolved && (
        <div
          className={"notice" + (!resolved.routable ? " bad" : runsHere ? "" : " warn")}
          style={{ marginBottom: 12 }}
        >
          {!resolved.routable ? (
            <>
              <strong>MLX-Gen cannot route this repository</strong>
              {resolved.error
                ? <span style={{ whiteSpace: "pre-wrap" }}>{resolved.error}</span>
                : "It reports no usable generation modes, so it cannot be generated from."}
            </>
          ) : (
            <>
              <strong>
                {resolved.tasks.map((t) => TASK_LABEL[t] ?? t).join(" · ")}
                {" · "}{fmtBytes(resolved.bytes)} download
                {" · "}~{resolved.peak_gib.toFixed(1)} GiB memory
                {runsHere ? " · runs here" : " · will not run here"}
              </strong>
              {resolved.fit_reason}
              {resolved.gated && " This repo is gated — accept its licence on Hugging Face first."}
              {" "}Memory is estimated from the download size, not a published benchmark.
            </>
          )}
        </div>
      )}

      {resolved?.routable && (
        <div className="field">
          <label>Display name</label>
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
        </div>
      )}

      <div style={{ display: "flex", gap: 7 }}>
        {!resolved ? (
          <button
            className="btn primary small"
            disabled={busy || repo.trim().length < 3}
            onClick={check}
          >
            {busy ? "Checking…" : "Check"}
          </button>
        ) : (
          <button
            className="btn primary small"
            disabled={busy || !resolved.routable}
            onClick={add}
          >
            {runsHere ? "Add to list" : "Add anyway"}
          </button>
        )}
        <button className="btn small" onClick={reset}>Cancel</button>
      </div>
    </div>
  );
}
