import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { api, errText, fmtBytes } from "../lib/api";
import type { Orphans, StorageInfo } from "../lib/types";

/**
 * Where the weights live.
 *
 * Models are by far the largest thing this app stores, and a 256 GB Mac runs
 * out long before the models get interesting. Keeping them on an external
 * drive while keys, the vault and the runtime stay internal is the difference
 * between two models and a library.
 */
export default function Storage({
  notify, onMoved,
}: {
  notify: (m: string, bad?: boolean) => void;
  onMoved: () => void;
}) {
  const [info, setInfo] = useState<StorageInfo | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [orphans, setOrphans] = useState<Orphans | null>(null);

  const refresh = async () => {
    try { setInfo(await api.storageInfo()); } catch (e) { notify(errText(e), true); }
    // Cheap: a directory walk, no hashing. Worth doing on every visit so the
    // waste is visible rather than something you have to suspect.
    try { setOrphans(await api.findOrphans()); } catch { /* not fatal */ }
  };

  const sweep = async () => {
    setBusy("Clearing…");
    try {
      const freed = await api.sweepOrphans();
      notify(`Recovered ${fmtBytes(freed)}.`);
      await refresh();
    } catch (e) {
      notify(errText(e), true);
    } finally {
      setBusy(null);
    }
  };
  useEffect(() => { void refresh(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const relocate = async (moveExisting: boolean) => {
    const picked = await open({ directory: true, title: "Choose where models are stored" });
    if (!picked || Array.isArray(picked)) return;
    setBusy(moveExisting ? "Moving models…" : "Switching…");
    try {
      setInfo(await api.setModelsLocation(picked, moveExisting));
      notify(moveExisting ? "Models moved." : "New downloads will go there.");
      onMoved();
    } catch (e) {
      notify(errText(e), true);
    } finally {
      setBusy(null);
    }
  };

  return (
    <>
    {orphans && orphans.bytes > 50 * 1024 * 1024 && (
      <div className="panel">
        <h2>Leftover files</h2>
        <div className="notice warn" style={{ marginBottom: 12 }}>
          <strong>{fmtBytes(orphans.bytes)} is being held by nothing</strong>
          {orphans.files} cached weight files that no installed model points
          at. Interrupted and repeated downloads leave these behind, and
          nothing collects them — the model looks correct, so the waste is
          invisible.
        </div>
        {orphans.repos.slice(0, 4).map(([repo, bytes]) => (
          <div key={repo} style={{ display: "flex", justifyContent: "space-between",
                                   fontSize: 11.5, marginBottom: 3 }}>
            <span style={{ color: "var(--text-dim)" }}>{repo}</span>
            <span>{fmtBytes(bytes)}</span>
          </div>
        ))}
        <div className="actions" style={{ marginTop: 12 }}>
          <button className="btn primary small" disabled={!!busy} onClick={sweep}>
            {busy ?? `Recover ${fmtBytes(orphans.bytes)}`}
          </button>
        </div>
        <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 8,
                      lineHeight: 1.55 }}>
          Safe: a file no snapshot links to cannot be reached through the cache,
          so removing it cannot break a model you have installed. Anything
          still downloading is left alone.
        </div>
      </div>
    )}
    <div className="panel">
      <h2>Model storage</h2>
      {info && (
        <>
          <div className="notice" style={{ marginBottom: 12 }}>
            <strong>
              {fmtBytes(info.models_bytes)} stored
              {" · "}{info.volume_free_gib.toFixed(0)} GB free on that volume
              {info.is_external ? " · external drive" : ""}
            </strong>
            Weights are the bulk of what this app keeps. Moving them to another
            drive leaves your keys, vault and runtime where they are.
          </div>
          <div style={{
            fontSize: 10.5, color: "var(--text-faint)", userSelect: "text",
            marginBottom: 12, wordBreak: "break-all",
          }}>
            {info.models_root}
          </div>
        </>
      )}

      {info?.is_external && (
        <div className="notice warn" style={{ marginBottom: 12 }}>
          <strong>Models are on a removable drive</strong>
          If it is not connected, downloaded models will not be found and
          generation will fail until you reconnect it or move them back.
        </div>
      )}

      <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
        <button className="btn small primary" disabled={!!busy} onClick={() => relocate(true)}>
          {busy ?? "Move models to another drive…"}
        </button>
        <button className="btn small" disabled={!!busy} onClick={() => relocate(false)}>
          Change location without moving
        </button>
        <button className="btn small" disabled={!!busy} onClick={refresh}>Refresh</button>
      </div>
      <div style={{ marginTop: 9, fontSize: 10.5, color: "var(--text-faint)", lineHeight: 1.6 }}>
        Moving copies every file first and only removes the originals once they
        have all landed. The engine restarts afterwards so it picks up the new
        location; nothing needs re-downloading.
      </div>
    </div>
    </>
  );
}
