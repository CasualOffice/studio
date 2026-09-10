import { useState } from "react";
import { api, errText, fmtBytes } from "../lib/api";
import type { HostInfo, ResolvedModel } from "../lib/types";

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
  onAdded, notify, host,
}: {
  onAdded: () => void;
  notify: (m: string, bad?: boolean) => void;
  host: HostInfo | null;
}) {
  const [open, setOpen] = useState(false);
  const [repo, setRepo] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [resolved, setResolved] = useState<ResolvedModel | null>(null);
  const [token, setToken] = useState("");

  const reset = () => {
    setRepo(""); setName(""); setResolved(null); setToken(""); setOpen(false);
  };

  /** Save a token, then re-check: gating is the only thing that was in the way. */
  const saveToken = async () => {
    setBusy(true);
    try {
      await api.setHfToken(token);
      notify("Token saved. Re-checking the model.");
      const r = await api.resolveModel(repo);
      setResolved(r);
      if (!name.trim()) setName(repo.split("/").pop() ?? repo);
    } catch (e) {
      notify(errText(e), true);
    } finally {
      setBusy(false);
    }
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
        resolved.model, name, resolved.tasks, resolved.bytes, null,
        resolved.family, resolved.backend
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
            {host && (
              <>
                This Mac fits roughly <b>{host.max_params_4bit.toFixed(0)}B parameters
                at 4-bit</b> or <b>{host.max_params_8bit.toFixed(0)}B at 8-bit</b>.
                {" "}Anything larger will not run, whatever its file size says.
                <br />
              </>
            )}
            It also has to be an architecture the engine knows: <b>FLUX.1</b> and
            {" "}<b>FLUX.2</b>, <b>Qwen-Image</b>, <b>Z-Image</b>, <b>ERNIE</b>,
            {" "}<b>FIBO</b>, <b>Bonsai</b> or <b>Wan</b>. Stable Diffusion and
            SDXL are a different architecture and will not run here.
          </div>
        </div>
      </div>

      {resolved && (
        <div
          className={"notice" + (!resolved.routable ? " bad" : runsHere ? "" : " warn")}
          style={{ marginBottom: 12 }}
        >
          {resolved.kind === "lora" ? (
            <>
              <strong>This is a LoRA, not a model</strong>
              An adapter refines a model you already have; it cannot generate on
              its own. Add it under <b>Style adapters</b> on the Generate panel,
              using the same repo id.
            </>
          ) : resolved.needs_token ? (
            <>
              <strong>This model is gated</strong>
              <span style={{ whiteSpace: "pre-wrap" }}>{resolved.error}</span>
              <div style={{ marginTop: 9 }}>
                <input
                  type="password"
                  value={token}
                  spellCheck={false}
                  placeholder="hf_..."
                  style={{ width: "100%", marginBottom: 7 }}
                  onChange={(e) => setToken(e.target.value)}
                />
                <button
                  className="btn primary small"
                  disabled={busy || !token.trim().startsWith("hf_")}
                  onClick={saveToken}
                >
                  {busy ? "Saving…" : "Save token and re-check"}
                </button>
                <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 6 }}>
                  Create one at huggingface.co/settings/tokens with read access.
                  It is stored on this Mac only, readable by your account alone,
                  and is sent to Hugging Face and nowhere else.
                </div>
              </div>
            </>
          ) : !resolved.routable ? (
            <>
              <strong>The engine cannot run this repository</strong>
              {resolved.error
                ? <span style={{ whiteSpace: "pre-wrap" }}>{resolved.error}</span>
                : "It reports no usable generation modes, so it cannot be generated from."}
            </>
          ) : (
            <>
              <strong>
                {resolved.tasks.map((t) => TASK_LABEL[t] ?? t).join(" · ")}
                {resolved.params > 0 && ` · ${(resolved.params / 1e9).toFixed(1)}B parameters`}
                {" · "}{fmtBytes(resolved.bytes)} download
                {" · "}~{resolved.peak_gib.toFixed(1)} GiB memory
                {runsHere ? " · runs here" : " · will not run here"}
              </strong>
              {resolved.params > 0 && resolved.params / 1e9 > resolved.max_params_4bit && (
                <div style={{ marginBottom: 6 }}>
                  At {(resolved.params / 1e9).toFixed(1)}B it is beyond this
                  Mac&rsquo;s ceiling of about {resolved.max_params_4bit.toFixed(0)}B,
                  even quantized to 4-bit.
                </div>
              )}
              {resolved.fit_reason}
              {resolved.gated && " Gated, and your saved token has access."}
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
