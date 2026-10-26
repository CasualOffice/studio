import { useState } from "react";
import type { CastMember } from "../../lib/types";
import { vaultUrl } from "../../lib/api";
import { ImageDrop } from "../shared";

/**
 * The one gate before drawing: who the comic is about, and what they look
 * like and wear in this chapter.
 *
 * Each recurring person gets a full-body cast sheet in their outfit, seeded
 * by who they are, and every panel they are in is drawn against it. Nothing
 * is drawn before this is approved -- sheets used to be drawn in the middle of
 * the first batch, so a wrong face was found out four panels later.
 */
export default function CastScreen({
  people, minor, sheets, drawingId, busy, approved, recast, onEdit, onLead, onRegenerate, onOwnPicture,
  onPromote, notify,
}: {
  people: CastMember[];
  /** Named once or only heard: no sheet unless the person asks for one. */
  minor: CastMember[];
  sheets: Record<string, string>;
  drawingId: string | null;
  busy: boolean;
  approved: boolean;
  recast: string[];
  onEdit: (id: string, patch: Partial<Pick<CastMember, "description" | "outfit" | "name">>) => void;
  onLead: (id: string) => void;
  onRegenerate: (person: CastMember) => void;
  onOwnPicture: (person: CastMember, image: string | null) => void;
  onPromote: (person: CastMember) => void;
  notify: (m: string, bad?: boolean) => void;
}) {
  const [showMinor, setShowMinor] = useState(false);
  /** Whose own-picture drop is open; one at a time. */
  const [picking, setPicking] = useState<string | null>(null);
  return (
    <section className="board-cast" aria-label="Cast" tabIndex={-1}>
      <div className="board-cast-head">
        <h2>Cast</h2>
        <p className="t-help">
          {approved
            ? "Approved. Change someone and draw their sheet again to recast them: only their panels are drawn again."
            : "One sheet per recurring person, in the outfit they wear in this chapter. Correct the words, draw a "
              + "sheet again until it is right, then approve. No panel is drawn before you do."}
        </p>
      </div>
      {people.length === 0 && (
        <p className="t-help">Nobody recurs in this chapter, so there is nothing to cast. Approve to draw.</p>
      )}
      <div className="board-cast-cards">
        {people.map((p) => {
          const id = p.id!;
          const sheet = sheets[id];
          const drawing = drawingId === `sheet:${id}`;
          return (
            <article key={id} className={"board-cast-card" + (recast.includes(id) ? " recast" : "")}
                     aria-label={p.name}>
              <div className="board-cast-sheet">
                {sheet ? <img src={vaultUrl(sheet)} alt={`Cast sheet for ${p.name}`} />
                  : <span>{drawing ? "drawing…" : "no sheet yet"}</span>}
              </div>
              <label className="board-cast-lead">
                <input type="radio" name="board-lead" checked={p.tier === 1} disabled={busy}
                       onChange={() => onLead(id)} />
                Lead
              </label>
              <label>Name
                <input value={p.name} disabled={busy}
                       onChange={(e) => onEdit(id, { name: e.target.value })} />
              </label>
              <label>What {p.name} looks like
                <textarea rows={2} value={p.description} disabled={busy}
                          placeholder="The story does not say: face, hair, build, age"
                          onChange={(e) => onEdit(id, { description: e.target.value })} />
              </label>
              <label>What {p.name} wears in this chapter
                <textarea rows={2} value={p.outfit ?? ""} disabled={busy}
                          placeholder="An outfit for the whole chapter"
                          onChange={(e) => onEdit(id, { outfit: e.target.value })} />
              </label>
              <div className="board-cast-actions">
                <button type="button" className="btn small" disabled={busy}
                        onClick={() => onRegenerate(p)}>
                  {sheet ? "Draw the sheet again" : "Draw the sheet"}
                </button>
              </div>
              <button type="button" className="link-btn" aria-expanded={picking === id}
                      onClick={() => setPicking(picking === id ? null : id)}>
                Use a picture of them instead
              </button>
              {picking === id && (
                <ImageDrop images={sheet ? [sheet] : []} max={1} onError={(m) => notify(m, true)}
                           onChange={(images) => onOwnPicture(p, images[0] ?? null)} />
              )}
            </article>
          );
        })}
      </div>
      {minor.length > 0 && (
        <div className="board-cast-minor">
          <button type="button" className="link-btn" aria-expanded={showMinor}
                  onClick={() => setShowMinor((v) => !v)}>
            {showMinor ? "Hide" : "Show"} {minor.length} minor {minor.length === 1 ? "person" : "people"}
          </button>
          {showMinor && minor.map((p) => (
            <div key={p.id} className="board-cast-minor-row">
              <span>{p.name}</span>
              <span className="t-help">{p.mentions ? `seen in ${p.mentions} moment${p.mentions === 1 ? "" : "s"}` : "heard, not seen"}</span>
              <button type="button" className="btn small" disabled={busy} onClick={() => onPromote(p)}>
                Give them a sheet
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
