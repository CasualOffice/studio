import { useState } from "react";
import { CUSTOM_STYLE, STYLES, styleHasMoved, styleLabel, type StyleLock } from "../../lib/board";

/**
 * The look of the whole board: four presets or the user's own words.
 *
 * Changing it once anything is drawn asks twice, because everything drawn
 * was drawn in the old one; the reading of the story -- the panels, briefs
 * and notes -- stays.
 */
export default function StylePicker({ style, busy, drawn, onChange }: {
  style: StyleLock;
  busy: boolean;
  /** Whether anything has been drawn in the current style. */
  drawn: boolean;
  onChange: (id: string, words?: string) => void;
}) {
  const [confirm, setConfirm] = useState<string | null>(null);
  const [customOpen, setCustomOpen] = useState(false);
  const [customWords, setCustomWords] = useState(style.id === CUSTOM_STYLE ? style.words : "");

  const choose = (id: string) => {
    const words = id === CUSTOM_STYLE ? customWords.trim() : undefined;
    if (id === style.id && (id !== CUSTOM_STYLE || words === style.words)) return;
    if (drawn && confirm !== id) { setConfirm(id); return; }
    setConfirm(null);
    onChange(id, words);
  };

  return (
    <div className="field">
      <label id="board-style-label">Style</label>
      <div role="group" aria-labelledby="board-style-label" className="board-style-pills">
        {STYLES.map(({ id, label, words }) => (
          <button key={id} type="button"
                  className={"pill" + (style.id === id ? " installed" : "") + (confirm === id ? " danger" : "")}
                  aria-pressed={style.id === id} title={words} disabled={busy}
                  onClick={() => choose(id)}>
            {style.id === id && <span aria-hidden="true">✓ </span>}
            {confirm === id ? `Restyle? ${label}` : label}
          </button>
        ))}
        <button type="button" className={"pill" + (style.id === CUSTOM_STYLE ? " installed" : "")}
                aria-pressed={style.id === CUSTOM_STYLE}
                aria-expanded={customOpen || style.id === CUSTOM_STYLE}
                title="Describe a look in your own words" disabled={busy}
                onClick={() => setCustomOpen((v) => !v)}>
          {style.id === CUSTOM_STYLE && <span aria-hidden="true">✓ </span>}
          Describe it…
        </button>
      </div>
      {(customOpen || style.id === CUSTOM_STYLE) && (
        <div className="board-style-custom">
          <textarea rows={2} disabled={busy} aria-label="Your own look" value={customWords}
                    onChange={(e) => setCustomWords(e.target.value)}
                    placeholder="1950s newspaper strip, coarse halftone, three colours" />
          <button className="btn small" disabled={busy || !customWords.trim() || customWords.trim() === style.words}
                  onClick={() => choose(CUSTOM_STYLE)}>
            {drawn && confirm === CUSTOM_STYLE ? "Press again to restyle" : "Use this look"}
          </button>
        </div>
      )}
      {confirm && (
        <div className="t-help board-warn-text">
          Press again to restyle. The cast sheets and every drawn panel go, because they were drawn
          in {styleLabel(style.id)}; the panels, briefs and notes stay, and the old pictures remain in
          the Vault.
        </div>
      )}
      {styleHasMoved(style) && (
        <div className="t-help">
          This board keeps the {styleLabel(style.id)} it was started in. The preset has changed since;
          picking it again here takes the current version.
        </div>
      )}
    </div>
  );
}
