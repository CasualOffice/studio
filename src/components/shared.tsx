import { useCallback, useEffect, useRef, useState } from "react";
import { open, save } from "@tauri-apps/plugin-dialog";
import { api, errText, vaultUrl } from "../lib/api";
import type { EngineProgress, VaultItem } from "../lib/types";

export function Bar({ value }: { value: number | null }) {
  return (
    <div className="bar">
      <div
        className={"bar-fill" + (value === null ? " indeterminate" : "")}
        style={{ width: `${Math.round((value ?? 0) * 100)}%` }}
      />
    </div>
  );
}

const PHASE_LABEL: Record<string, string> = {
  load: "Loading model",
  start: "Starting",
  denoise: "Generating",
  decode: "Decoding",
  generated: "Finishing",
  save: "Saving",
  complete: "Complete",
  download: "Downloading",
};

export function JobProgress({ p, label }: { p: EngineProgress | null; label: string }) {
  if (!p) {
    return (
      <div className="progress">
        <div className="progress-head"><span>{label}</span></div>
        <Bar value={null} />
      </div>
    );
  }
  const phase = PHASE_LABEL[p.phase] ?? p.phase;
  let right = "";
  let note: string | null = null;
  if (p.phase === "download" && p.total_bytes) {
    const gb = (n: number) => (n / 1024 ** 3).toFixed(2);
    right = `${gb(p.done_bytes ?? 0)} / ${gb(p.total_bytes)} GB`;
    // Slow and stuck look identical without this. One download ran for
    // seventy-nine minutes before anyone noticed it had stopped.
    const stalled = p.stalled_seconds ?? 0;
    if (stalled >= 45) {
      note = `no data for ${Math.round(stalled)}s — it may have stalled`;
    } else {
      const bps = p.bytes_per_second ?? 0;
      const parts: string[] = [];
      if (bps > 0) parts.push(`${(bps / 1024 ** 2).toFixed(1)} MB/s`);
      if (p.eta_seconds != null && p.eta_seconds > 0) {
        const m = Math.round(p.eta_seconds / 60);
        parts.push(m >= 1 ? `about ${m} min left` : "less than a minute left");
      }
      note = parts.join(" · ") || null;
    }
  } else if (p.step != null && p.total_steps) {
    right = `step ${p.step}/${p.total_steps}`;
  } else if (p.progress != null) {
    right = `${Math.round(p.progress * 100)}%`;
  }
  const multi = p.item_count && p.item_count > 1
    ? ` · image ${(p.item_index ?? 0) + 1}/${p.item_count}` : "";

  return (
    <div className="progress">
      <div className="progress-head">
        <span>{phase}{multi}{p.message ? ` — ${p.message}` : ""}</span>
        <span>{right}</span>
      </div>
      {note && (
        <div style={{
          fontSize: 10.5, marginTop: 3,
          color: (p.stalled_seconds ?? 0) >= 45 ? "var(--warn)" : "var(--text-faint)",
        }}>
          {note}
        </div>
      )}
      <Bar value={p.progress} />
    </div>
  );
}

/** File picker + drag-and-drop that copies sources into the app's own inputs. */
export function ImageDrop({
  images, onChange, max = 3, onError,
}: {
  images: string[];
  onChange: (next: string[]) => void;
  max?: number;
  onError: (m: string) => void;
}) {
  const [over, setOver] = useState(false);
  const [picking, setPicking] = useState(false);
  const [vaultItems, setVaultItems] = useState<VaultItem[] | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  const add = useCallback(async (paths: string[]) => {
    const next = [...images];
    for (const p of paths) {
      if (next.length >= max) break;
      try {
        // Importing seals a copy into the vault and returns its id. The file
        // the user picked is left exactly where it was.
        next.push(await api.vaultImport(p, "image"));
      } catch (e) {
        onError(errText(e));
      }
    }
    onChange(next);
  }, [images, max, onChange, onError]);

  useEffect(() => {
    // Tauri v2 delivers OS file drops as a webview event, not an HTML5 one.
    let unlisten: (() => void) | undefined;
    (async () => {
      const { getCurrentWebview } = await import("@tauri-apps/api/webview");
      unlisten = await getCurrentWebview().onDragDropEvent((event) => {
        const el = ref.current;
        if (!el) return;
        if (event.payload.type === "over") {
          const { x, y } = event.payload.position;
          const r = el.getBoundingClientRect();
          const scale = window.devicePixelRatio || 1;
          const px = x / scale, py = y / scale;
          setOver(px >= r.left && px <= r.right && py >= r.top && py <= r.bottom);
        } else if (event.payload.type === "drop") {
          const { x, y } = event.payload.position;
          const r = el.getBoundingClientRect();
          const scale = window.devicePixelRatio || 1;
          const px = x / scale, py = y / scale;
          if (px >= r.left && px <= r.right && py >= r.top && py <= r.bottom) {
            void add(event.payload.paths);
          }
          setOver(false);
        } else {
          setOver(false);
        }
      });
    })();
    return () => unlisten?.();
  }, [add]);

  const pick = async () => {
    const sel = await open({
      multiple: true,
      filters: [{
        name: "Images",
        extensions: [
          "png", "jpg", "jpeg", "jpe", "webp", "avif",
          "heic", "heif", "tif", "tiff", "bmp", "gif",
        ],
      }],
    });
    if (!sel) return;
    await add(Array.isArray(sel) ? sel : [sel]);
  };

  // Everything already made or imported lives in the vault, so reusing one
  // should not require exporting it and importing it back.
  const openVault = async () => {
    setPicking(true);
    if (vaultItems === null) {
      try {
        setVaultItems((await api.vaultList()).filter(
          (i) => i.mime.startsWith("image/") && i.kind !== "mask"
        ));
      } catch (e) {
        onError(errText(e));
        setVaultItems([]);
      }
    }
  };

  return (
    <div>
      <div
        ref={ref}
        className={"dropzone" + (over ? " over" : "")}
        onClick={pick}
      >
        {images.length >= max
          ? `Maximum of ${max} image${max === 1 ? "" : "s"}`
          : "Drop an image here, or click to choose a file"}
      </div>

      {images.length < max && (
        <button
          className="btn small full"
          style={{ marginTop: 7 }}
          onClick={() => (picking ? setPicking(false) : void openVault())}
        >
          {picking ? "Close" : "Use one from your vault"}
        </button>
      )}

      {picking && (
        <div style={{
          marginTop: 8, maxHeight: 190, overflowY: "auto",
          border: "1px solid var(--border)", borderRadius: 8, padding: 8,
        }}>
          {vaultItems === null ? (
            <div style={{ fontSize: 11, color: "var(--text-faint)" }}>Loading…</div>
          ) : vaultItems.length === 0 ? (
            <div style={{ fontSize: 11, color: "var(--text-faint)" }}>
              No images in the vault yet.
            </div>
          ) : (
            <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
              {vaultItems.map((it) => (
                <img
                  key={it.id}
                  src={vaultUrl(it.id)}
                  alt=""
                  title={it.prompt || it.name}
                  style={{
                    width: 62, height: 62, objectFit: "cover", borderRadius: 6,
                    border: images.includes(it.id)
                      ? "2px solid var(--accent)" : "1px solid var(--border)",
                    cursor: "pointer",
                  }}
                  onClick={() => {
                    if (images.includes(it.id)) {
                      onChange(images.filter((x) => x !== it.id));
                    } else if (images.length < max) {
                      onChange([...images, it.id]);
                      setPicking(false);
                    }
                  }}
                />
              ))}
            </div>
          )}
        </div>
      )}
      {images.length > 0 && (
        <div className="thumbs">
          {images.map((p, i) => (
            <div className="thumb" key={p}>
              <img src={vaultUrl(p)} alt="" />
              <button
                onClick={() => onChange(images.filter((_, j) => j !== i))}
                title="Remove"
              >×</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * A message at the bottom of the window.
 *
 * An error stays until it is dismissed, and clicking it does not dismiss it.
 * Both of those were wrong before: a failure vanished after eight seconds and
 * the click you made to select the text closed it, so the one thing a person
 * needs from an error -- to copy it and send it to someone -- was impossible.
 * Good news still fades on its own, because nobody needs to keep it.
 */
export function Toast({ msg, bad, onDone }: { msg: string; bad?: boolean; onDone: () => void }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (bad) return;               // a failure waits for the reader
    const t = setTimeout(onDone, 3200);
    return () => clearTimeout(t);
  }, [msg, bad, onDone]);
  useEffect(() => setCopied(false), [msg]);
  return (
    <div className={"toast" + (bad ? " bad" : "")}>
      <span style={{ userSelect: "text", WebkitUserSelect: "text" }}>{msg}</span>
      {bad && (
        <span style={{ display: "inline-flex", gap: 6, marginLeft: 10 }}>
          <button
            className="btn small"
            onClick={() => {
              void navigator.clipboard.writeText(msg)
                .then(() => setCopied(true))
                .catch(() => setCopied(false));
            }}
          >
            {copied ? "Copied" : "Copy"}
          </button>
          <button className="btn small" onClick={onDone}>Dismiss</button>
        </span>
      )}
    </div>
  );
}

/**
 * Export several items into one folder.
 *
 * Exporting one at a time asks for a destination per file, which for eight
 * pictures is eight dialogs. This asks once for a folder and writes every
 * selection into it, keeping each item's own name and disambiguating the
 * collisions that follow from two runs of the same prompt.
 */
export async function exportMany(
  items: Pick<VaultItem, "id" | "name">[],
  notify: (m: string, bad?: boolean) => void
): Promise<void> {
  if (items.length === 0) return;
  if (items.length === 1) return exportItem(items[0], notify);
  try {
    const dir = await open({ directory: true, multiple: false,
                             title: `Export ${items.length} items` });
    if (typeof dir !== "string") return;
    const used = new Set<string>();
    let bytes = 0;
    let failed = 0;
    for (const it of items) {
      let name = it.name || `${it.id}.png`;
      if (used.has(name)) {
        // Two runs of one prompt produce two files with one name. Numbering
        // the later one keeps both rather than silently overwriting.
        const dot = name.lastIndexOf(".");
        const stem = dot > 0 ? name.slice(0, dot) : name;
        const ext = dot > 0 ? name.slice(dot) : "";
        let n = 2;
        while (used.has(`${stem} ${n}${ext}`)) n++;
        name = `${stem} ${n}${ext}`;
      }
      used.add(name);
      try {
        bytes += await api.vaultExport(it.id, `${dir}/${name}`);
      } catch {
        failed++;
      }
    }
    const ok = items.length - failed;
    notify(
      `Exported ${ok} of ${items.length} — ${(bytes / 1024 / 1024).toFixed(1)} MB, ` +
      "and these copies are not encrypted." +
      (failed ? ` ${failed} could not be written.` : ""),
      failed > 0
    );
  } catch (e) {
    notify(errText(e), true);
  }
}

export async function exportItem(
  item: Pick<VaultItem, "id" | "name">,
  notify: (m: string, bad?: boolean) => void
): Promise<void> {
  try {
    const dest = await save({
      defaultPath: item.name || `${item.id}.png`,
      title: "Export a decrypted copy",
    });
    if (!dest) return;
    const bytes = await api.vaultExport(item.id, dest);
    notify(`Exported ${(bytes / 1024).toFixed(0)} KB — this copy is not encrypted.`);
  } catch (e) {
    notify(errText(e), true);
  }
}
