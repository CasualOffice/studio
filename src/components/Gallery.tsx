import { useState } from "react";
import { api, errText, fmtBytes, fmtDuration, vaultUrl } from "../lib/api";
import type { ModelStatus, VaultItem } from "../lib/types";
import { exportItem } from "./shared";
import SendTo, { type Destination } from "./SendTo";

export default function Gallery({
  items, onChanged, notify, send, models,
}: {
  items: VaultItem[];
  onChanged: () => void;
  notify: (m: string, bad?: boolean) => void;
  /** Reuse a stored item anywhere else, without exporting it first. */
  send?: (dest: Destination, ids: string[]) => void;
  models: ModelStatus[];
}) {
  const [open, setOpen] = useState<VaultItem | null>(null);

  // Painted masks live in the vault so they are never written in the clear,
  // but they are working data and would only clutter the library.
  const all = items;
  items = items.filter((i) => i.kind !== "mask");

  /**
   * Walk an item back to where it started.
   *
   * Each item records the ids it was made from, so a picture that was
   * generated, edited, then animated carries its whole history. Following it
   * is what makes a chain of tabs legible after the fact.
   */
  const lineage = (item: VaultItem, depth = 0): VaultItem[] => {
    if (depth > 8) return []; // a cycle should not hang the interface
    const parents = item.inputs
      .map((id) => all.find((x) => x.id === id))
      .filter((x): x is VaultItem => !!x);
    return parents.flatMap((p) => [...lineage(p, depth + 1), p]);
  };

  const KIND_LABEL: Record<string, string> = {
    generate: "Generated", edit: "Edited", upscale: "Enlarged",
    video: "Animated", import: "Imported", doc: "Document",
    recovered: "Recovered", mask: "Mask",
  };

  if (items.length === 0) {
    return (
      <div className="empty-state">
        <span className="big">▦</span>
        The vault is empty.
        <div style={{ marginTop: 10, fontSize: 12 }}>
          Anything you generate or import is sealed here automatically.
        </div>
      </div>
    );
  }

  const remove = async (item: VaultItem) => {
    try {
      await api.vaultDelete(item.id);
      if (open?.id === item.id) setOpen(null);
      onChanged();
      notify("Deleted from the vault.");
    } catch (e) {
      notify(errText(e), true);
    }
  };

  const isImage = (i: VaultItem) => i.mime.startsWith("image/");
  const isVideo = (i: VaultItem) => i.mime.startsWith("video/");

  if (open) {
    return (
      <div>
        <button className="btn small" onClick={() => setOpen(null)} style={{ marginBottom: 14 }}>
          ← Back to vault
        </button>
        <div className="content split" style={{ padding: 0 }}>
          <div className="panel">
            <h2>Details</h2>
            {open.prompt && (
              <div className="field">
                <label>Prompt</label>
                <div style={{ fontSize: 12, userSelect: "text", lineHeight: 1.6 }}>{open.prompt}</div>
              </div>
            )}
            {([
              ["Name", open.name],
              ["Model", open.model || "—"],
              ["Kind", open.kind],
              ["Type", open.mime],
              ["Size", fmtBytes(open.bytes)],
              ["Seed", open.seed ? String(open.seed) : "—"],
              ["Dimensions", open.width && open.height ? `${open.width}×${open.height}` : "—"],
              ["Steps", open.steps != null ? String(open.steps) : "—"],
              ["Guidance", open.guidance != null ? open.guidance.toFixed(1) : "—"],
              ["Took", open.duration_ms ? fmtDuration(open.duration_ms) : "—"],
              ["Created", new Date(open.created_at).toLocaleString()],
            ] as [string, string][]).map(([k, v]) => (
              <div className="field" key={k}>
                <label>{k}</label>
                <div style={{ fontSize: 12, color: "var(--text-dim)", userSelect: "text" }}>{v}</div>
              </div>
            ))}
            {open.inputs.length > 0 && (() => {
              const chain = lineage(open);
              return (
                <div className="field">
                  <label>How this was made</label>
                  <div style={{ display: "flex", alignItems: "center", gap: 6,
                                flexWrap: "wrap", marginTop: 4 }}>
                    {chain.map((step) => (
                      <div key={step.id} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <div
                          className="thumb"
                          title={`${KIND_LABEL[step.kind] ?? step.kind}: ${step.prompt || step.name}`}
                          style={{ cursor: "pointer" }}
                          onClick={() => setOpen(step)}
                        >
                          <img src={vaultUrl(step.id)} alt="" />
                        </div>
                        <span style={{ color: "var(--text-faint)" }}>&rarr;</span>
                      </div>
                    ))}
                    <div className="thumb" style={{ outline: "2px solid var(--accent)",
                                                    borderRadius: 6 }}>
                      {open.mime.startsWith("video/")
                        ? <video src={vaultUrl(open.id)} muted
                            style={{ width: 62, height: 62, objectFit: "cover",
                                     borderRadius: 6, display: "block" }} />
                        : <img src={vaultUrl(open.id)} alt="" />}
                    </div>
                  </div>
                  <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 6 }}>
                    {chain.map((c) => KIND_LABEL[c.kind] ?? c.kind).concat(
                      KIND_LABEL[open.kind] ?? open.kind
                    ).join(" → ")}
                    {" · click a step to open it"}
                  </div>
                </div>
              );
            })()}
            <div style={{ display: "flex", gap: 7, marginTop: 12, flexWrap: "wrap" }}>
              {isImage(open) && send && (
                <SendTo ids={[open.id]} send={send} models={models} />
              )}
              <button className="btn small" onClick={() => exportItem(open, notify)}>
                Export a copy…
              </button>
              <button className="btn small danger" onClick={() => remove(open)}>Delete</button>
            </div>
            <div style={{ marginTop: 10, fontSize: 10.5, color: "var(--text-faint)", lineHeight: 1.6 }}>
              Exporting writes an unencrypted copy wherever you choose. Everything
              inside the vault stays sealed.
            </div>
          </div>
          <div className="canvas">
            {isImage(open) ? (
              <img src={vaultUrl(open.id)} alt="" />
            ) : isVideo(open) ? (
              <video src={vaultUrl(open.id)} controls loop autoPlay muted
                style={{ maxWidth: "100%", maxHeight: "64vh", borderRadius: 6 }} />
            ) : (
              <div className="empty"><span className="big">▤</span>{open.name}<br />
                No preview for this file type.</div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="gallery-grid">
      {items.map((it) => (
        <div className="gallery-card" key={it.id} onClick={() => setOpen(it)}>
          {isImage(it)
            ? <img src={vaultUrl(it.id)} alt="" loading="lazy" />
            : isVideo(it)
            ? <video src={vaultUrl(it.id)} muted loop
                onMouseEnter={(e) => void (e.target as HTMLVideoElement).play()}
                onMouseLeave={(e) => (e.target as HTMLVideoElement).pause()}
                style={{ width: "100%", aspectRatio: "1", objectFit: "cover",
                         display: "block", background: "#0b0d12" }} />
            : <div style={{
                aspectRatio: "1", display: "flex", alignItems: "center",
                justifyContent: "center", fontSize: 28, color: "var(--text-faint)",
                background: "#0b0d12",
              }}>▤</div>}
          <div className="meta">
            <div className="p">{it.prompt || it.name}</div>
            <div className="sub">
              {KIND_LABEL[it.kind] ?? it.kind}
              {it.inputs.length > 0 ? " · from another" : ""}
              {it.duration_ms ? ` · ${fmtDuration(it.duration_ms)}` : ""}
            </div>
            {isImage(it) && send && (
              <div style={{ display: "flex", gap: 5, marginTop: 7, flexWrap: "wrap" }}>
                <SendTo ids={[it.id]} send={send} models={models} compact />
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
