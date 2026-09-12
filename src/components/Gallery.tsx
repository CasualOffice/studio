import { useEffect, useMemo, useState } from "react";
import { api, errText, fmtBytes, fmtDuration, vaultUrl } from "../lib/api";
import type { ModelStatus, VaultItem } from "../lib/types";
import { exportItem, exportMany } from "./shared";
import SendTo, { type Destination } from "./SendTo";

export default function Gallery({
  items, onChanged, notify, send, models, onReuse,
}: {
  items: VaultItem[];
  onChanged: () => void;
  notify: (m: string, bad?: boolean) => void;
  /** Reuse a stored item anywhere else, without exporting it first. */
  send?: (dest: Destination, ids: string[]) => void;
  models: ModelStatus[];
  /** Load this run's settings back into the tab that produced it. */
  onReuse?: (item: VaultItem, reuseSeed: boolean) => void;
}) {
  const [open, setOpen] = useState<VaultItem | null>(null);
  const [query, setQuery] = useState("");
  const [kindFilter, setKindFilter] = useState<string>("all");
  // Selection is by id, not by index: filtering and searching reorder the
  // grid underneath, and a selection that survives a search is the point.
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [anchor, setAnchor] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Bulk delete is irreversible and the vault is the only copy, so the button
  // asks once before it does it. The single-item delete is one item in front
  // of you; twelve selected across a filtered grid is not.
  //
  // Two buttons, two flags. One flag was shared by the grid's bulk delete and
  // the board view's delete-everything, and nothing disarmed it when you moved
  // between the two screens: arming one and backing out left the other already
  // reading "Really delete?", so the next click deleted without asking.
  const [confirmBulk, setConfirmBulk] = useState(false);
  const [confirmBoard, setConfirmBoard] = useState(false);

  // Escape clears a selection the way it does everywhere else. This is
  // registered above the early returns because hooks cannot be conditional.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setPicked(new Set()); setAnchor(null); setConfirmBulk(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Leaving a screen disarms whatever was armed on it. Backing out of a board
  // is how someone changes their mind, and a mind changed on one screen must
  // not leave a loaded delete waiting on the next one.
  useEffect(() => { setConfirmBoard(false); setConfirmBulk(false); }, [open]);

  // Painted masks live in the vault so they are never written in the clear,
  // but they are working data and would only clutter the library.
  const all = items;
  items = items.filter((i) => i.kind !== "mask");

  // What the library holds, before the kind filter and the search box narrow
  // it down. "The vault is empty" and "nothing matches that" are different
  // statements, and the empty-vault branch below was answering both: a search
  // that matched nothing unmounted the grid *and* the search box with it, so
  // the only way out of the query was to leave the tab and come back.
  const libraryCount = items.length;

  const kinds = Array.from(new Set(items.map((i) => i.kind))).sort();
  if (kindFilter !== "all") items = items.filter((i) => i.kind === kindFilter);
  if (query.trim()) {
    // Search what a person would remember: what they asked for, what made it,
    // and the file name.
    const q = query.toLowerCase();
    items = items.filter((i) =>
      i.prompt.toLowerCase().includes(q) ||
      i.name.toLowerCase().includes(q) ||
      i.model.toLowerCase().includes(q)
    );
  }

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

  /**
   * Collapse a board into one entry.
   *
   * A picture board writes a character sheet, a room reference per scene and
   * one picture per panel. Left flat that is seventeen items in the library
   * for one comic, which is what made the vault unusable after a board run.
   * Members are folded into their first panel, and the card says how many are
   * inside; opening it still reaches every one.
   */
  const groups = useMemo(() => {
    const byProject = new Map<string, VaultItem[]>();
    const flat: VaultItem[] = [];
    for (const it of items) {
      if (it.project) {
        const g = byProject.get(it.project);
        if (g) g.push(it); else byProject.set(it.project, [it]);
      } else {
        flat.push(it);
      }
    }
    const heads: { item: VaultItem; members: VaultItem[] }[] = [];
    for (const [, members] of byProject) {
      members.sort((a, b) =>
        (a.project_index ?? 1e9) - (b.project_index ?? 1e9) ||
        a.created_at.localeCompare(b.created_at));
      heads.push({ item: members[0], members });
    }
    for (const it of flat) heads.push({ item: it, members: [it] });
    heads.sort((a, b) => b.item.created_at.localeCompare(a.item.created_at));
    return heads;
  }, [items]);

  const shown = groups.map((g) => g.item);

  const togglePick = (id: string, range: boolean) => {
    setPicked((prev) => {
      const next = new Set(prev);
      if (range && anchor) {
        const a = shown.findIndex((i) => i.id === anchor);
        const b = shown.findIndex((i) => i.id === id);
        if (a >= 0 && b >= 0) {
          for (let k = Math.min(a, b); k <= Math.max(a, b); k++) next.add(shown[k].id);
          return next;
        }
      }
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
    setAnchor(id);
    setConfirmBulk(false);
  };

  const clearPicks = () => {
    setPicked(new Set()); setAnchor(null); setConfirmBulk(false);
  };

  /**
   * Everything a selection actually refers to.
   *
   * The grid shows one card per board, and that card is the board's first
   * panel. Selecting it and pressing Delete must delete the board, not strip
   * one picture out of it and leave the rest orphaned -- which is what
   * matching on picked ids alone would do. So a picked head expands to every
   * member of its project.
   */
  const pickedItems = () => {
    const ids = new Set(picked);
    const projects = new Set(
      all.filter((i) => ids.has(i.id) && i.project).map((i) => i.project));
    return all.filter((i) => ids.has(i.id)
      || (i.project != null && projects.has(i.project)));
  };

  const exportPicked = async () => {
    setBusy(true);
    try { await exportMany(pickedItems(), notify); clearPicks(); }
    finally { setBusy(false); }
  };

  const deletePicked = async () => {
    const chosen = pickedItems();
    setBusy(true);
    let failed = 0;
    for (const it of chosen) {
      try { await api.vaultDelete(it.id); } catch { failed++; }
    }
    setBusy(false);
    clearPicks();
    if (open && chosen.some((c) => c.id === open.id)) setOpen(null);
    onChanged();
    notify(
      failed
        ? `Deleted ${chosen.length - failed} of ${chosen.length}; ${failed} could not be removed.`
        : `Deleted ${chosen.length} from the vault.`,
      failed > 0
    );
  };

  if (libraryCount === 0) {
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

  // Opening a board shows the comic, not the first panel's metadata. A board
  // is a thing someone made; its parts are pages and sheets, and the actions
  // that matter are on the whole of it.
  const openProject = open?.project
    ? all.filter((i) => i.project === open.project)
        .sort((a, b) => (a.project_index ?? 1e9) - (b.project_index ?? 1e9) ||
                        a.created_at.localeCompare(b.created_at))
    : null;

  if (open && openProject && openProject.length > 1) {
    const pagesOf = openProject.filter((i) => i.model === "composed");
    const panelsOf = openProject.filter(
      (i) => i.project_index != null && i.model !== "composed");
    const sheets = openProject.filter(
      (i) => i.project_index == null && i.model !== "composed");
    const total = openProject.reduce((n, i) => n + i.bytes, 0);
    return (
      <div>
        <button className="btn small" onClick={() => setOpen(null)} style={{ marginBottom: 14 }}>
          ← Back to vault
        </button>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
          <h2 style={{ margin: 0 }}>{open.project_name || "Picture board"}</h2>
          {/* Pages first, because the pages are what was made. The panels and
              the sheets are how it was made, and leading with "12 panels"
              described the machinery to someone looking for their comic. */}
          <span style={{ fontSize: 12, color: "var(--text-faint)" }}>
            {pagesOf.length
              ? `${pagesOf.length} page${pagesOf.length === 1 ? "" : "s"}`
              : "no pages composed yet"}
            {` · ${panelsOf.length} panel${panelsOf.length === 1 ? "" : "s"}`}
            {" · "}{fmtBytes(total)}
            {" · "}{new Date(open.created_at).toLocaleDateString()}
          </span>
        </div>

        <div style={{ display: "flex", gap: 8, margin: "14px 0", flexWrap: "wrap" }}>
          {/* The comic, in one action, named and in order.
              Exporting used to mean picking the two finished pages out of
              seventeen lookalike items and hoping they were the right ones. */}
          {pagesOf.length > 0 && (
            <button className="btn small primary" disabled={busy}
                    onClick={() => void exportMany(
                      pagesOf.map((it, i) => ({
                        id: it.id,
                        name: `${(open.project_name || "Picture board")
                          .replace(/[\\/:*?"<>|]/g, "-")} — page ${i + 1}.png`,
                      })), notify)}>
              Export the comic ({pagesOf.length} page{pagesOf.length === 1 ? "" : "s"})
            </button>
          )}
          <button className="btn small" disabled={busy}
                  onClick={() => void exportMany(openProject, notify)}>
            Export everything ({openProject.length})
          </button>
          {send && (
            <SendTo ids={openProject.filter((i) => i.mime.startsWith("image/")).map((i) => i.id)}
                    send={send} models={models} compact />
          )}
          <button
            className="btn small danger"
            disabled={busy}
            onClick={async () => {
              if (!confirmBoard) { setConfirmBoard(true); return; }
              setBusy(true);
              let failed = 0;
              for (const it of openProject) {
                try { await api.vaultDelete(it.id); } catch { failed++; }
              }
              setBusy(false); setConfirmBoard(false); setOpen(null); onChanged();
              notify(failed
                ? `Deleted ${openProject.length - failed} of ${openProject.length}.`
                : `Deleted the board and all ${openProject.length} pictures.`,
                failed > 0);
            }}
          >
            {confirmBoard
              ? "Really delete this whole board? This cannot be undone"
              : "Delete board"}
          </button>
        </div>

        {/* Three peer grids of square thumbnails used to sit here, and the
            heading "The comic, in order" sat above the *panels* -- so the two
            pictures the whole board exists to produce were labelled with the
            software's word for how it made them, centre-cropped to squares,
            and placed below twelve of their own ingredients. A person opening
            this is looking for their comic. It reads in order, full width, at
            the shape it was drawn in; the materials are still here, folded
            away, because chapter twelve needs the same cast as chapter
            eleven. */}
        {pagesOf.length > 0 ? (
          <>
            <div className="sub-head">
              {pagesOf.length === 1 ? "The page" : "The comic, in order"}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 16,
                          marginBottom: 22 }}>
              {pagesOf.map((it, i) => (
                <div className="page-sheet" key={it.id}
                     onClick={() => setOpen({ ...it, project: null })}>
                  <img src={vaultUrl(it.id)} alt={`Page ${i + 1}`} loading="lazy" />
                  <div className="page-no">Page {i + 1} of {pagesOf.length}</div>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="notice" style={{ marginBottom: 18 }}>
            <strong>No pages yet.</strong> The panels are drawn; composing them
            in the Board tab turns them into the pages you read.
          </div>
        )}

        <details style={{ marginBottom: 12 }} open={pagesOf.length === 0}>
          <summary style={{ cursor: "pointer", fontSize: 12,
                            color: "var(--text-dim)", padding: "4px 0" }}>
            Panels ({panelsOf.length})
          </summary>
          <div className="gallery-grid" style={{ marginTop: 10 }}>
            {panelsOf.map((it) => (
              <div className="gallery-card" key={it.id}
                   onClick={() => setOpen({ ...it, project: null })}>
                <span className="count">{(it.project_index ?? 0) + 1}</span>
                <img src={vaultUrl(it.id)} alt="" loading="lazy" />
                <div className="meta">
                  <div className="sub">{it.prompt.slice(0, 60)}</div>
                </div>
              </div>
            ))}
          </div>
        </details>

        {sheets.length > 0 && (
          <details>
            <summary style={{ cursor: "pointer", fontSize: 12,
                              color: "var(--text-dim)", padding: "4px 0" }}>
              Cast and places ({sheets.length})
            </summary>
            <div className="gallery-grid" style={{ marginTop: 10 }}>
              {sheets.map((it) => (
                <div className="gallery-card" key={it.id}
                     onClick={() => setOpen({ ...it, project: null })}>
                  <img src={vaultUrl(it.id)} alt="" loading="lazy" />
                  <div className="meta"><div className="sub">{it.prompt.slice(0, 48)}</div></div>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    );
  }

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
              {onReuse && open.prompt && (
                <>
                  <button className="btn small primary" onClick={() => onReuse(open, false)}>
                    Make another like this
                  </button>
                  <button
                    className="btn small"
                    title="Same prompt, same seed, same settings"
                    onClick={() => onReuse(open, true)}
                  >
                    Recreate exactly
                  </button>
                </>
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
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
        <input
          type="text"
          value={query}
          placeholder="Search prompts, models, names"
          onChange={(e) => setQuery(e.target.value)}
          style={{ flex: 1, minWidth: 200 }}
        />
        <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value)}
          style={{ width: "auto" }}>
          <option value="all">Everything</option>
          {kinds.map((k) => (
            <option key={k} value={k}>{KIND_LABEL[k] ?? k}</option>
          ))}
        </select>
      </div>
      {picked.size > 0 && (
        <div className="bulk-bar">
          <b>
            {(() => {
              const n = pickedItems().length;
              return n === picked.size
                ? `${n} selected`
                : `${picked.size} selected \u00b7 ${n} pictures`;
            })()}
          </b>
          <button className="btn small" disabled={busy} onClick={exportPicked}>
            {busy ? "Working…" : picked.size === 1 ? "Export" : `Export ${picked.size}`}
          </button>
          {send && pickedItems().every((i) => i.mime.startsWith("image/")) && (
            <SendTo ids={[...picked]} send={send} models={models} compact />
          )}
          <button
            className="btn small danger"
            disabled={busy}
            onClick={() => {
              if (confirmBulk) { void deletePicked(); return; }
              setConfirmBulk(true);
            }}
          >
            {confirmBulk
              ? `Really delete ${pickedItems().length}? This cannot be undone`
              : `Delete ${pickedItems().length}`}
          </button>
          <button className="btn small" style={{ marginLeft: "auto" }}
            onClick={clearPicks}>Clear</button>
        </div>
      )}
      {items.length === 0 ? (
        <div className="empty-state">
          <span className="big">⌕</span>
          Nothing matches that.
        </div>
      ) : (
      <div className="gallery-grid">
      {groups.map(({ item: it, members }) => (
        <div
          className={"gallery-card" + (picked.has(it.id) ? " picked" : "")}
          key={it.id}
          onClick={(e) => {
            // The tick selects; a modifier selects; anything else opens. Once
            // something is selected a plain click keeps selecting, because
            // that is what a file list does and opening would lose the set.
            if ((e.target as HTMLElement).closest(".pick") || e.shiftKey || e.metaKey) {
              e.preventDefault();
              togglePick(it.id, e.shiftKey);
            } else if (picked.size > 0) {
              togglePick(it.id, false);
            } else {
              setOpen(it);
            }
          }}
        >
          <button
            className="pick"
            aria-pressed={picked.has(it.id)}
            aria-label={picked.has(it.id) ? "Deselect" : "Select"}
            onClick={(e) => { e.stopPropagation(); togglePick(it.id, e.shiftKey); }}
          >✓</button>
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
          {members.length > 1 && (
            <span className="count" title={`${members.length} pictures in this board`}>
              {members.length}
            </span>
          )}
          <div className="meta">
            <div className="p">
              {members.length > 1
                ? (it.project_name || "Picture board")
                : (it.prompt || it.name)}
            </div>
            <div className="sub">
              {/* What was made, not how many files it took. "board \u00b7 17
                  pictures" is the tool's internals reaching the surface: the
                  person made one comic of two pages, and the seventeen are the
                  panels and sheets it was built from. */}
              {members.length > 1
                ? (() => {
                    const pages = members.filter((m) => m.model === "composed").length;
                    const panels = members.filter(
                      (m) => m.project_index != null && m.model !== "composed").length;
                    return [
                      pages ? `${pages} page${pages === 1 ? "" : "s"}` : null,
                      panels ? `${panels} panel${panels === 1 ? "" : "s"}` : null,
                    ].filter(Boolean).join(" \u00b7 ") || `${members.length} pictures`;
                  })()
                : (KIND_LABEL[it.kind] ?? it.kind)}
              {members.length === 1 && it.inputs.length > 0 ? " · from another" : ""}
              {members.length === 1 && it.duration_ms
                ? ` · ${fmtDuration(it.duration_ms)}` : ""}
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
      )}
    </div>
  );
}
