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

/** The part of a drop zone's on-screen rect that the hit test needs. */
export interface DropRect {
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
}

/**
 * Whether an OS file drop landed on a drop zone.
 *
 * Tauri names the payload's field a `PhysicalPosition`, so this used to divide
 * the position by `window.devicePixelRatio` before comparing it against the
 * zone's CSS-pixel rect. On macOS wry actually reports logical points, so the
 * division halved a coordinate that was already correct: on a Retina display a
 * drop three hundred points down the window was tested against a point a
 * hundred and fifty down and fell outside every zone, which is why dragging an
 * image in had never once worked on these machines. The position is compared
 * exactly as it arrives, and nothing here consults the device pixel ratio.
 *
 * A zero-sized rect matches nothing. Every tab stays mounted, so every drop
 * zone in the app hears every drop, and a hidden one measures as an empty rect
 * at the origin -- which contains the point (0, 0), so a drop in the very
 * corner of the window used to land in all of them at the same time.
 */
export function dropLandedIn(rect: DropRect, point: { x: number; y: number }): boolean {
  if (rect.width === 0 || rect.height === 0) return false;
  return point.x >= rect.left && point.x <= rect.right
    && point.y >= rect.top && point.y <= rect.bottom;
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
    let over = 0;
    const failed: string[] = [];
    for (const p of paths) {
      // Say so rather than dropping them. Five files into a three-image slot
      // took three and discarded two in silence, which reads as the drop
      // having half worked for no reason.
      if (next.length >= max) { over++; continue; }
      try {
        // Importing seals a copy into the vault and returns its id. The file
        // the user picked is left exactly where it was.
        next.push(await api.vaultImport(p, "image"));
      } catch (e) {
        failed.push(errText(e));
      }
    }
    onChange(next);
    if (failed.length) {
      // One message, not one per file: a toast per failure shows you the last
      // one and hides how many there were.
      onError(failed.length === 1 ? failed[0]
        : `${failed.length} of ${paths.length} could not be read.`);
    } else if (over) {
      onError(`This takes ${max} image${max === 1 ? "" : "s"}; ` +
              `${over} ${over === 1 ? "was" : "were"} not added.`);
    }
  }, [images, max, onChange, onError]);

  // Every call site passes a fresh inline `onError`, and `add` closes over the
  // images it is adding to, so `add` was a new function on every parent render.
  // The parent re-renders on every progress event, so during a run the effect
  // below tore down and re-registered its four OS listeners several times a
  // second, and a file dropped in one of those gaps was ignored. Reading the
  // latest `add` through a ref lets that effect subscribe once per mount and
  // still see the current images and callbacks.
  const addRef = useRef(add);
  useEffect(() => { addRef.current = add; }, [add]);

  useEffect(() => {
    // Tauri v2 delivers OS file drops as a webview event, not an HTML5 one.
    let unlisten: (() => void) | undefined;
    let live = true;
    (async () => {
      const { getCurrentWebview } = await import("@tauri-apps/api/webview");
      const remove = await getCurrentWebview().onDragDropEvent((event) => {
        const el = ref.current;
        if (!el) return;
        // `dropLandedIn` owns the whole hit test, including the zero-sized
        // rect a hidden tab's zone reports.
        const rect = el.getBoundingClientRect();
        if (event.payload.type === "over") {
          setOver(dropLandedIn(rect, event.payload.position));
        } else if (event.payload.type === "drop") {
          if (dropLandedIn(rect, event.payload.position)) {
            void addRef.current(event.payload.paths);
          }
          setOver(false);
        } else {
          setOver(false);
        }
      });
      if (live) unlisten = remove;
      else remove();
    })();
    return () => { live = false; unlisten?.(); };
  }, []);

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
    try {
      // Refresh on every open. Generations can finish while this component is
      // mounted, and a cached first snapshot hid those new images.
      setVaultItems((await api.vaultList()).filter(
        (i) => i.mime.startsWith("image/") && i.kind !== "mask"
      ));
    } catch (e) {
      onError(errText(e));
      setVaultItems([]);
    }
  };

  return (
    <div>
      <div
        ref={ref}
        className={"dropzone" + (over ? " over" : "")}
        // Full means full. It said "Maximum of 3 images" and still opened the
        // file picker when clicked, so choosing a file there did nothing.
        onClick={images.length >= max ? undefined : pick}
        style={images.length >= max ? { cursor: "default" } : undefined}
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
  // App passes a fresh inline `onDone` on every render, so depending on it here
  // restarted the timer each time -- and App re-renders on every progress
  // event, which during a run arrive far more often than every 3.2 seconds. A
  // success toast therefore never reached the end of its own countdown and sat
  // on screen for the whole generation.
  const onDoneRef = useRef(onDone);
  useEffect(() => { onDoneRef.current = onDone; }, [onDone]);
  useEffect(() => {
    if (bad) return;               // a failure waits for the reader
    const t = setTimeout(() => onDoneRef.current(), 3200);
    return () => clearTimeout(t);
  }, [msg, bad]);
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
    const failures: string[] = [];
    let bytes = 0;
    let failed = 0;
    for (const it of items) {
      let name = (it.name || `${it.id}.png`).split(/[\\/]/).pop() || `${it.id}.png`;
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
      let written = false;
      let lastError = "";
      for (let attempt = 1; attempt <= 100 && !written; attempt++) {
        const dot = name.lastIndexOf(".");
        const stem = dot > 0 ? name.slice(0, dot) : name;
        const ext = dot > 0 ? name.slice(dot) : "";
        const candidate = attempt === 1 ? name : `${stem} ${attempt}${ext}`;
        try {
          bytes += await api.vaultExport(it.id, `${dir}/${candidate}`, false);
          used.add(candidate);
          written = true;
        } catch (e) {
          // Only a name that is already taken is worth another attempt. A
          // read-only folder or a full disk fails identically every time, and
          // treating it as a collision meant a hundred silent retries and then
          // a count of failures with no reason attached -- the one thing the
          // person needed in order to fix it.
          lastError = errText(e);
          if (!/exists|file exists|already|in use/i.test(lastError)) break;
        }
      }
      if (!written) {
        failed++;
        if (lastError && !failures.includes(lastError)) failures.push(lastError);
      }
    }
    const ok = items.length - failed;
    notify(
      `Exported ${ok} of ${items.length} — ${(bytes / 1024 / 1024).toFixed(1)} MB, ` +
      "and these copies are not encrypted." +
      (failed
        ? ` ${failed} could not be written${failures.length ? `: ${failures[0]}` : "."}`
        : ""),
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
    const bytes = await api.vaultExport(item.id, dest, true);
    notify(`Exported ${(bytes / 1024).toFixed(0)} KB — this copy is not encrypted.`);
  } catch (e) {
    notify(errText(e), true);
  }
}
