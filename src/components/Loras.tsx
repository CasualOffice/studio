import { useEffect, useState } from "react";
import { api, errText, fmtBytes, newJobId } from "../lib/api";
import type { Lora, ResolvedLora } from "../lib/types";

/**
 * Style and task adapters.
 *
 * A LoRA is a small set of weight deltas layered onto a base model, so a few
 * tens of megabytes buys a whole look or a specialised behaviour. They are tied
 * to a specific base: one trained for FLUX.2 Klein 9B will not load against the
 * 4B, and there is no way to tell from the file itself.
 */
export default function Loras({
  selected, onSelectedChange, notify,
}: {
  /** Handles paired with strengths, applied to the next run. */
  selected: [string, number][];
  onSelectedChange: (next: [string, number][]) => void;
  notify: (m: string, bad?: boolean) => void;
}) {
  const [installed, setInstalled] = useState<Lora[]>([]);
  const [adding, setAdding] = useState(false);
  const [repo, setRepo] = useState("");
  const [found, setFound] = useState<ResolvedLora | null>(null);
  const [file, setFile] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try { setInstalled(await api.listLoras()); } catch { /* none yet */ }
  };
  useEffect(() => { void refresh(); }, []);

  const check = async () => {
    setBusy(true); setFound(null);
    try {
      const r = await api.resolveLora(repo);
      setFound(r);
      setFile(r.files[0]?.name ?? "");
      if (r.files.length === 0) notify("No adapter weights in that repository.", true);
    } catch (e) { notify(errText(e), true); } finally { setBusy(false); }
  };

  const install = async () => {
    if (!found) return;
    setBusy(true);
    try {
      const name = file.replace(/\.safetensors$/, "") || found.repo.split("/").pop()!;
      setInstalled(await api.addLora(newJobId(), found.repo, file, name, found.bytes));
      notify(`Added ${name}.`);
      setRepo(""); setFound(null); setAdding(false);
    } catch (e) { notify(errText(e), true); } finally { setBusy(false); }
  };

  const toggle = (l: Lora) => {
    const at = selected.findIndex(([h]) => h === l.handle);
    if (at >= 0) onSelectedChange(selected.filter((_, i) => i !== at));
    else onSelectedChange([...selected, [l.handle, l.scale]]);
  };

  const setStrength = (handle: string, scale: number) =>
    onSelectedChange(selected.map(([h, s]) => (h === handle ? [h, scale] : [h, s])));

  return (
    <div className="panel">
      <h2>Style adapters</h2>

      {installed.length === 0 && !adding && (
        <div style={{ fontSize: 11.5, color: "var(--text-dim)", lineHeight: 1.6, marginBottom: 10 }}>
          None yet. Adapters are usually tens of megabytes and change how a
          model draws — a look, a medium, or a specific task.
        </div>
      )}

      {installed.map((l) => {
        const active = selected.find(([h]) => h === l.handle);
        return (
          <div key={l.handle} style={{ marginBottom: 10 }}>
            <label style={{ cursor: "pointer" }}>
              <span>
                <input
                  type="checkbox" checked={!!active} style={{ width: "auto", marginRight: 6 }}
                  onChange={() => toggle(l)}
                />
                {l.name}
              </span>
              <em>{fmtBytes(l.bytes)}</em>
            </label>
            {active && (
              <>
                <input
                  type="range" min={0} max={1.5} step={0.05} value={active[1]}
                  onChange={(e) => setStrength(l.handle, +e.target.value)}
                  style={{ marginTop: 5 }}
                />
                <div style={{ display: "flex", justifyContent: "space-between",
                              fontSize: 10.5, color: "var(--text-faint)" }}>
                  <span>strength {active[1].toFixed(2)}</span>
                  <button
                    className="btn small"
                    style={{ padding: "1px 7px", fontSize: 10 }}
                    onClick={async () => {
                      try {
                        setInstalled(await api.removeLora(l.handle));
                        onSelectedChange(selected.filter(([h]) => h !== l.handle));
                      } catch (e) { notify(errText(e), true); }
                    }}
                  >Forget</button>
                </div>
              </>
            )}
          </div>
        );
      })}

      {adding ? (
        <>
          <div className="field">
            <label>Adapter repository</label>
            <input
              type="text" value={repo} spellCheck={false} autoFocus
              placeholder="fal/flux-2-klein-4B-outpaint-lora"
              onChange={(e) => { setRepo(e.target.value); setFound(null); }}
              onKeyDown={(e) => { if (e.key === "Enter" && repo.length > 2) void check(); }}
            />
          </div>

          {found && found.files.length > 0 && (
            <>
              {found.files.length > 1 && (
                <div className="field">
                  <label>Which adapter</label>
                  <select value={file} onChange={(e) => setFile(e.target.value)}>
                    {found.files.map((f) => (
                      <option key={f.name} value={f.name}>
                        {f.name} — {fmtBytes(f.bytes)}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              {found.not_an_adapter ? (
                // Checked before the download, not after. Nothing used to
                // check at all: a VAE repository looked exactly like an
                // adapter repository, downloaded, appeared installed, could be
                // selected and given a strength, and then changed nothing --
                // because there is no adapter in it to apply.
                <div className="notice bad" style={{ marginBottom: 10 }}>
                  <strong>This is not an adapter</strong>
                  {found.files[0]?.name} holds {found.not_an_adapter}. Installing
                  it would spend {fmtBytes(found.bytes)} on something that cannot
                  change a picture.
                </div>
              ) : (
                <div className="notice warn" style={{ marginBottom: 10 }}>
                  <strong>{fmtBytes(found.bytes)}{found.base_model ? ` · for ${found.base_model}` : ""}</strong>
                  An adapter only works on the base model it was trained for. If it
                  was made for a different size or family, generation will fail or
                  the result will be unaffected.
                </div>
              )}
            </>
          )}

          <div style={{ display: "flex", gap: 7 }}>
            {!found ? (
              <button className="btn small primary" disabled={busy || repo.length < 3} onClick={check}>
                {busy ? "Checking…" : "Check"}
              </button>
            ) : (
              <button className="btn small primary"
                      disabled={busy || !found.files.length || !!found.not_an_adapter}
                      onClick={install}>
                {busy ? "Downloading…" : `Install ${fmtBytes(found.bytes)}`}
              </button>
            )}
            <button className="btn small" onClick={() => { setAdding(false); setFound(null); }}>
              Cancel
            </button>
          </div>
        </>
      ) : (
        <button className="btn small" onClick={() => setAdding(true)}>+ Add an adapter</button>
      )}
    </div>
  );
}
