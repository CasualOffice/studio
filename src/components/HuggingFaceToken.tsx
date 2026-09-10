import { useEffect, useState } from "react";
import { api, errText } from "../lib/api";

/**
 * A Hugging Face access token, for gated repositories.
 *
 * Some of the best models -- every official FLUX.1 among them -- sit behind a
 * licence you accept on Hugging Face. Without a token their download fails
 * with a 401 that looks like the model is broken, so this is the difference
 * between "unsupported" and "one field away from working".
 */
export default function HuggingFaceToken({
  notify,
}: {
  notify: (m: string, bad?: boolean) => void;
}) {
  const [status, setStatus] = useState<{ present: boolean; hint: string | null } | null>(null);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try { setStatus(await api.hfTokenStatus()); } catch { /* not fatal */ }
  };
  useEffect(() => { void refresh(); }, []);

  const save = async (value: string) => {
    setBusy(true);
    try {
      await api.setHfToken(value);
      setToken("");
      await refresh();
      notify(value ? "Token saved." : "Token removed.");
    } catch (e) {
      notify(errText(e), true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel">
      <h2>Hugging Face account</h2>
      <div style={{ fontSize: 11.5, color: "var(--text-dim)", lineHeight: 1.6, marginBottom: 12 }}>
        Optional. Some models are gated: you accept their licence on Hugging
        Face, and downloads then need a token proving it is you. Without one
        those models fail to download.
      </div>

      {status?.present ? (
        <div className="notice" style={{ marginBottom: 12 }}>
          <strong>Token saved</strong>
          <code>{status.hint}</code> — gated models you have access to will download.
        </div>
      ) : null}

      <div className="field">
        <label>{status?.present ? "Replace token" : "Access token"}</label>
        <input
          type="password"
          value={token}
          spellCheck={false}
          placeholder="hf_..."
          onChange={(e) => setToken(e.target.value)}
        />
        <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5, lineHeight: 1.55 }}>
          Create one at huggingface.co/settings/tokens with read access. It is
          stored on this Mac, readable by your account alone, and sent to
          Hugging Face and nowhere else.
        </div>
      </div>

      <div style={{ display: "flex", gap: 7 }}>
        <button
          className="btn primary small"
          disabled={busy || !token.trim().startsWith("hf_")}
          onClick={() => void save(token.trim())}
        >
          {busy ? "Saving…" : "Save token"}
        </button>
        {status?.present && (
          <button className="btn small" disabled={busy} onClick={() => void save("")}>
            Remove
          </button>
        )}
      </div>
    </div>
  );
}
