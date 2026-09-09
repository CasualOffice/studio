import { useEffect, useState } from "react";
import { api, errText } from "../lib/api";
import type { VaultStatus } from "../lib/types";
import { MIN_PASSPHRASE, strength } from "../lib/passphrase";

/**
 * Creates or unlocks the vault. Nothing else in the app renders until this
 * resolves, because every path that could produce or read content needs the
 * data key.
 */
export default function VaultGate({
  status, onOpen,
}: {
  status: VaultStatus;
  onOpen: (s: VaultStatus) => void;
}) {
  const creating = !status.exists;

  const [pass, setPass] = useState("");
  const [confirm, setConfirm] = useState("");
  const [useBiometry, setUseBiometry] = useState(true);
  const [ack, setAck] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Offer Touch ID first when it is set up: it is the fast path.
  useEffect(() => {
    if (creating || !status.biometry_enrolled) return;
    let cancelled = false;
    (async () => {
      try {
        const s = await api.vaultUnlockBiometry();
        if (!cancelled) onOpen(s);
      } catch {
        // Cancelled or unavailable: the passphrase field is already there.
      }
    })();
    return () => { cancelled = true; };
  }, [creating, status.biometry_enrolled, onOpen]);

  const submit = async () => {
    setErr(null);
    setBusy(true);
    try {
      if (creating) {
        if (pass !== confirm) throw new Error("The two passphrases do not match.");
        if (!ack) throw new Error("Please confirm you understand the recovery terms.");
        onOpen(await api.vaultCreate(pass, useBiometry));
      } else {
        onOpen(await api.vaultUnlockPassphrase(pass));
      }
      setPass(""); setConfirm("");
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };

  const s = strength(pass);

  return (
    <div className="setup-wrap">
      <div className="setup-card">
        <h1>{creating ? "Create your vault" : "Unlock"}</h1>
        <p className="lede">
          {creating
            ? "Everything this app produces or imports is encrypted before it touches the disk. Choose the passphrase that protects it."
            : "Your images and documents are encrypted at rest. Unlock to continue."}
        </p>

        <div className="panel">
          {creating && (
            <div className="notice warn" style={{ marginBottom: 14 }}>
              <strong>There is no way to reset this passphrase</strong>
              The key is derived from what you type; it is never sent anywhere and
              never stored. If you forget it and Touch ID is unavailable, the
              encrypted content cannot be recovered by anyone, including us.
            </div>
          )}

          <div className="field">
            <label>
              Passphrase
              {creating && <em>{pass.length} / {MIN_PASSPHRASE} minimum</em>}
            </label>
            <input
              type="password"
              value={pass}
              autoFocus
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
              onChange={(e) => setPass(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !creating) void submit(); }}
              placeholder={creating ? "Four or more unrelated words works well" : ""}
            />
            {creating && pass.length > 0 && (
              <>
                <div className="meter-track" style={{ marginTop: 7 }}>
                  <div
                    className={"meter-fill" + (s.score < 2 ? " bad" : s.score < 3 ? " warn" : "")}
                    style={{ width: `${(s.score / 4) * 100}%` }}
                  />
                </div>
                <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5 }}>
                  {s.label}
                </div>
              </>
            )}
          </div>

          {creating && (
            <>
              <div className="field">
                <label>Confirm passphrase</label>
                <input
                  type="password"
                  value={confirm}
                  autoComplete="off"
                  autoCorrect="off"
                  autoCapitalize="off"
                  spellCheck={false}
                  onChange={(e) => setConfirm(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") void submit(); }}
                />
              </div>

              {status.biometry_available && (
                <div className="field">
                  <label style={{ cursor: "pointer" }}>
                    <span>
                      <input
                        type="checkbox" checked={useBiometry} style={{ width: "auto", marginRight: 6 }}
                        onChange={(e) => setUseBiometry(e.target.checked)}
                      />
                      Also unlock with Touch ID
                    </span>
                  </label>
                  <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5, lineHeight: 1.55 }}>
                    Stores a second key in the macOS Keychain, marked
                    &ldquo;this device only&rdquo; so it never syncs to iCloud. Your
                    passphrase keeps working if the fingerprint set changes or you
                    move to another Mac.
                  </div>
                </div>
              )}

              <div className="field">
                <label style={{ cursor: "pointer" }}>
                  <span>
                    <input
                      type="checkbox" checked={ack} style={{ width: "auto", marginRight: 6 }}
                      onChange={(e) => setAck(e.target.checked)}
                    />
                    I understand this passphrase cannot be recovered
                  </span>
                </label>
              </div>
            </>
          )}

          {err && (
            <div className="notice bad" style={{ marginBottom: 12 }}>
              <span style={{ whiteSpace: "pre-wrap" }}>{err}</span>
            </div>
          )}

          <button
            className="btn primary full"
            disabled={busy || pass.length < (creating ? MIN_PASSPHRASE : 1)}
            onClick={submit}
          >
            {busy ? "Working…" : creating ? "Create vault" : "Unlock"}
          </button>

          {!creating && status.biometry_enrolled && (
            <button
              className="btn full"
              style={{ marginTop: 8 }}
              disabled={busy}
              onClick={async () => {
                setErr(null); setBusy(true);
                try { onOpen(await api.vaultUnlockBiometry()); }
                catch (e) { setErr(errText(e)); }
                finally { setBusy(false); }
              }}
            >
              Use Touch ID
            </button>
          )}

          <div style={{ marginTop: 10, fontSize: 10.5, color: "var(--text-faint)", lineHeight: 1.6 }}>
            Encrypted with ChaCha20-Poly1305. The key is derived with Argon2id
            (64 MiB, 3 passes) and held only in memory while unlocked.
          </div>
        </div>
      </div>
    </div>
  );
}
