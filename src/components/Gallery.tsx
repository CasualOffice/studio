import { useState } from "react";
import { api, errText, fmtBytes, fmtDuration, vaultUrl } from "../lib/api";
import type { VaultItem } from "../lib/types";
import { exportItem } from "./shared";

export default function Gallery({
  items, onChanged, notify, onSendToEdit,
}: {
  items: VaultItem[];
  onChanged: () => void;
  notify: (m: string, bad?: boolean) => void;
  /** Reuse a stored image as an edit source, without exporting it first. */
  onSendToEdit?: (ids: string[]) => void;
}) {
  const [open, setOpen] = useState<VaultItem | null>(null);

  // Painted masks live in the vault so they are never written in the clear,
  // but they are working data and would only clutter the library.
  items = items.filter((i) => i.kind !== "mask");

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
            {open.inputs.length > 0 && (
              <div className="field">
                <label>Source images</label>
                <div className="thumbs">
                  {open.inputs.map((id) => (
                    <div className="thumb" key={id}><img src={vaultUrl(id)} alt="" /></div>
                  ))}
                </div>
              </div>
            )}
            <div style={{ display: "flex", gap: 7, marginTop: 12, flexWrap: "wrap" }}>
              {isImage(open) && onSendToEdit && (
                <button className="btn small primary" onClick={() => onSendToEdit([open.id])}>
                  Edit this
                </button>
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
              {it.model || it.mime}
              {it.duration_ms ? ` · ${fmtDuration(it.duration_ms)}` : ""}
            </div>
            {isImage(it) && onSendToEdit && (
              <button
                className="btn small"
                style={{ marginTop: 7, width: "100%", padding: "3px 0", fontSize: 11 }}
                onClick={(e) => { e.stopPropagation(); onSendToEdit([it.id]); }}
              >
                Edit this
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
