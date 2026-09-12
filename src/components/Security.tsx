import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { api, errText } from "../lib/api";
import type { VaultStatus } from "../lib/types";
import { MIN_PASSPHRASE } from "../lib/passphrase";
import Storage from "./Storage";
import HuggingFaceToken from "./HuggingFaceToken";

export default function Security({
  status, onStatus, notify, onImported, onStorageMoved,
}: {
  status: VaultStatus;
  onStatus: (s: VaultStatus) => void;
  notify: (m: string, bad?: boolean) => void;
  onImported: () => void;
  onStorageMoved: () => void;
}) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);

  const wrap = async (fn: () => Promise<void>) => {
    setBusy(true);
    try { await fn(); } catch (e) { notify(errText(e), true); } finally { setBusy(false); }
  };

  const importFiles = () => wrap(async () => {
    const sel = await open({ multiple: true, title: "Import into the vault" });
    if (!sel) return;
    const paths = Array.isArray(sel) ? sel : [sel];
    let n = 0;
    const failed: string[] = [];
    for (const p of paths) {
      try { await api.vaultImport(p, "any"); n++; }
      // Collected rather than announced one at a time: a toast per failure
      // means importing twenty files shows you the last one and hides the
      // other nineteen, including how many there were.
      catch (e) { failed.push(errText(e)); }
    }
    if (n) onImported();
    if (failed.length) {
      notify(
        `Sealed ${n} of ${paths.length}. ${failed.length} could not be read` +
        (failed.length === 1 ? `: ${failed[0]}` : "."),
        true
      );
    } else if (n) {
      notify(`Sealed ${n} file${n === 1 ? "" : "s"} into the vault.`);
    }
  });

  return (
    <div style={{ maxWidth: 620 }}>
      <div className="panel">
        <h2>Vault</h2>
        <div className="notice" style={{ marginBottom: 12 }}>
          <strong>
            {status.item_count} item{status.item_count === 1 ? "" : "s"} · unlocked
            {status.via_biometry ? " with Touch ID" : " with your passphrase"}
          </strong>
          Content is sealed with ChaCha20-Poly1305 before it reaches the disk.
          Filenames are random and carry no information; names, prompts and seeds
          live in an encrypted index.
        </div>
        <div style={{ fontSize: 10.5, color: "var(--text-faint)", userSelect: "text", marginBottom: 12 }}>
          {status.root}
        </div>
        <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginBottom: 10, lineHeight: 1.6 }}>
          Every stored file carries its own identifier, so if the index is ever
          lost the content can be identified and re-listed. <b>Check and repair</b>
          reconciles the two.
        </div>
        <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
          <button className="btn small" disabled={busy} onClick={importFiles}>
            Import files…
          </button>
          <button
            className="btn small"
            disabled={busy}
            onClick={() => wrap(async () => {
              onStatus(await api.vaultLock());
              notify("Vault locked and model memory released.");
            })}
          >
            Lock now
          </button>
          <button
            className="btn small"
            disabled={busy}
            title="Reconcile the index against what is actually stored"
            onClick={() => wrap(async () => {
              const r = await api.vaultRepair();
              onImported();
              notify(
                r.recovered || r.dropped || r.unreadable
                  ? `Recovered ${r.recovered}, removed ${r.dropped} dead entries` +
                    (r.unreadable ? `, ${r.unreadable} unreadable` : "")
                  : "Everything already consistent."
              );
            })}
          >
            Check and repair
          </button>
        </div>
      </div>

      <Storage notify={notify} onMoved={onStorageMoved} />
      <HuggingFaceToken notify={notify} />

      <div className="panel">
        <h2>Touch ID</h2>
        {!status.biometry_available ? (
          <div style={{ fontSize: 12, color: "var(--text-dim)", lineHeight: 1.6 }}>
            Not available in this build. Biometric unlock stores its key in the
            data-protection Keychain, which macOS only allows for apps signed
            with an Apple Developer team identifier. This app is built and run
            locally without one.
            <div style={{ marginTop: 8, color: "var(--text-faint)", fontSize: 11 }}>
              Your passphrase is unaffected — it is the root of trust either way,
              and Touch ID would only ever have been a shortcut to it.
            </div>
          </div>
        ) : status.biometry_enrolled ? (
          <>
            <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 11, lineHeight: 1.6 }}>
              A second key is stored in the macOS Keychain, marked
              &ldquo;this device only&rdquo;, and released only after Touch ID or
              your login password. Your passphrase always keeps working.
            </div>
            <button
              className="btn small danger"
              disabled={busy}
              onClick={() => wrap(async () => {
                onStatus(await api.vaultDisableBiometry());
                notify("Touch ID unlock removed. Your passphrase still works.");
              })}
            >
              Remove Touch ID unlock
            </button>
          </>
        ) : (
          <>
            <div style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 11, lineHeight: 1.6 }}>
              Add a Keychain-held key so you can unlock without typing your
              passphrase. It never syncs to iCloud and never leaves this Mac.
            </div>
            <button
              className="btn small primary"
              disabled={busy}
              onClick={() => wrap(async () => {
                onStatus(await api.vaultEnableBiometry());
                notify("Touch ID unlock enabled.");
              })}
            >
              Enable Touch ID unlock
            </button>
          </>
        )}
      </div>

      <div className="panel">
        <h2>Change passphrase</h2>
        <div className="field">
          <label>Current passphrase</label>
          <input type="password" autoComplete="off" value={current}
            onChange={(e) => setCurrent(e.target.value)} />
        </div>
        <div className="field">
          <label>New passphrase <em>{MIN_PASSPHRASE} characters minimum</em></label>
          <input type="password" autoComplete="off" value={next}
            onChange={(e) => setNext(e.target.value)} />
        </div>
        <div className="field">
          <label>Confirm new passphrase</label>
          <input type="password" autoComplete="off" value={confirm}
            onChange={(e) => setConfirm(e.target.value)} />
        </div>
        <button
          className="btn primary"
          disabled={busy || next.length < MIN_PASSPHRASE || !current}
          onClick={() => wrap(async () => {
            if (next !== confirm) throw new Error("The two new passphrases do not match.");
            await api.vaultChangePassphrase(current, next);
            setCurrent(""); setNext(""); setConfirm("");
            notify("Passphrase changed.");
          })}
        >
          Change passphrase
        </button>
        <div style={{ marginTop: 9, fontSize: 10.5, color: "var(--text-faint)", lineHeight: 1.6 }}>
          This re-wraps the same encryption key with a new salt, so your existing
          content is not re-encrypted and nothing has to be rewritten.
        </div>
      </div>
    </div>
  );
}
