import { useState } from "react";
import type { Panel } from "../../lib/types";
import type { PanelArt } from "../../lib/boardDoc";
import { vaultUrl } from "../../lib/api";
import Field from "../Field";
import MaskCanvas from "../MaskCanvas";

/**
 * One panel, and every way to fix it in place.
 *
 * The brief is what is drawn; the words -- caption and balloons -- are set on
 * the page afterwards, so changing them never redraws the picture. A wrong
 * picture gets a few alternatives to choose from, each from a new seed, or a
 * painted repair of just the part that is wrong.
 */
export default function PanelInspector({
  panel, index, count, art, prompt, locked, busy, readOnly, choices, maskReady,
  onEdit, onNote, onVariants, onPick, onDiscardChoices, onRedraw, onRepair, onMask, onRestore,
  onMove, onAddAfter, onSplit, onMerge, onRemove,
}: {
  panel: Panel & { id: string };
  index: number;
  count: number;
  art: PanelArt;
  /** The exact words the drawer is given. */
  prompt: string;
  locked: boolean;
  busy: boolean;
  readOnly: boolean;
  /** Alternatives drawn for this panel, waiting to be chosen from. */
  choices: string[];
  maskReady: boolean;
  onEdit: (patch: Partial<Panel>) => void;
  onNote: (note: string) => void;
  onVariants: (n: number) => void;
  onPick: (id: string) => void;
  onDiscardChoices: () => void;
  onRedraw: () => void;
  onRepair: () => void;
  onMask: (bytes: Uint8Array | null) => void;
  onRestore: (id: string) => void;
  onMove: (by: number) => void;
  onAddAfter: () => void;
  onSplit: () => void;
  onMerge: () => void;
  onRemove: () => void;
}) {
  const [repairing, setRepairing] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const [count3, setCount3] = useState(3);
  const off = locked || readOnly;
  const cast = panel.cast ?? [];
  const editActor = (at: number, patch: Partial<NonNullable<Panel["cast"]>[number]>) =>
    onEdit({ cast: cast.map((a, k) => (k === at ? { ...a, ...patch } : a)) });

  return (
    <div className="board-inspector">
      {locked && !readOnly && (
        <div className="t-help board-locked" role="note">
          This panel is being drawn, so it is locked until it finishes. The others can still be edited,
          and a change to one still waiting is what gets drawn.
        </div>
      )}

      {art.drawn && !readOnly && (
        <div className="field board-fix">
          <label>Fix this panel <em>the rest of the comic stays</em></label>
          {choices.length > 0 ? (
            <>
              <div className="board-choices" role="group" aria-label="Choose a version">
                {[art.drawn, ...choices].map((id, k) => (
                  <button key={id} type="button" className="board-choice" disabled={busy}
                          aria-label={k === 0 ? "Keep the current picture" : `Use version ${k}`}
                          onClick={() => onPick(id)}>
                    <img src={vaultUrl(id)} alt="" />
                    <span>{k === 0 ? "Current" : `Version ${k}`}</span>
                  </button>
                ))}
              </div>
              <button type="button" className="btn small" disabled={busy} onClick={onDiscardChoices}>
                Keep the current one
              </button>
            </>
          ) : (
            <div className="board-fix-row">
              <button type="button" className="btn small primary" disabled={busy} onClick={() => onVariants(count3)}>
                Try {count3} other versions
              </button>
              <select aria-label="How many versions" value={count3} disabled={busy}
                      onChange={(e) => setCount3(Number(e.target.value))}>
                {[2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
              <button type="button" className="btn small" disabled={busy}
                      onClick={() => { onMask(null); setRepairing((v) => !v); }}>
                {repairing ? "Cancel the painted repair" : "Repair one part of it"}
              </button>
            </div>
          )}
          {repairing && (
            <div className="board-repair">
              <div className="t-help">
                Paint over what is wrong. Only the painted part is drawn again; your note below says
                what should be there.
              </div>
              <MaskCanvas sourceId={art.drawn} onMaskChange={onMask} disabled={busy} />
              <button type="button" className="btn small full primary" disabled={busy || !maskReady}
                      onClick={() => { onRepair(); setRepairing(false); }}>
                {maskReady ? "Redraw the painted region" : "Paint a region first"}
              </button>
            </div>
          )}
          <Field label={<>Your note <em>applied to every redraw of this panel</em></>}>
            <input type="text" value={art.note} disabled={off} placeholder="older, grey at the temples"
                   onChange={(e) => onNote(e.target.value)} />
          </Field>
          {art.note.trim() && (
            <button type="button" className="btn small" disabled={busy} onClick={onRedraw}>
              Redraw with the note
            </button>
          )}
        </div>
      )}

      <Field label={<>Brief <em>what is drawn: who, doing what, where</em></>}>
        <textarea rows={3} value={panel.description ?? ""} disabled={off}
                  placeholder={panel.moment ?? ""}
                  onChange={(e) => onEdit({ description: e.target.value })} />
      </Field>
      <Field label="Caption" hint="Narration on the page. Changing it never redraws the picture.">
        <input type="text" value={panel.caption} disabled={off}
               onChange={(e) => onEdit({ caption: e.target.value })} />
      </Field>
      <fieldset className="board-balloons" disabled={off}>
        <legend>Balloons <em>changing them never redraws the picture</em></legend>
        {(panel.dialogue ?? []).map((d, k) => (
          <div key={k} className="board-balloon-row">
            <input aria-label={`Balloon ${k + 1} speaker`} value={d.speaker}
                   onChange={(e) => onEdit({ dialogue: (panel.dialogue ?? []).map((x, j) =>
                     (j === k ? { ...x, speaker: e.target.value } : x)) })} />
            <input aria-label={`Balloon ${k + 1} text`} value={d.text}
                   onChange={(e) => onEdit({ dialogue: (panel.dialogue ?? []).map((x, j) =>
                     (j === k ? { ...x, text: e.target.value } : x)) })} />
            <button type="button" className="btn small" aria-label={`Remove balloon ${k + 1}`}
                    onClick={() => onEdit({ dialogue: (panel.dialogue ?? []).filter((_, j) => j !== k) })}>×</button>
          </div>
        ))}
        <button type="button" className="btn small"
                onClick={() => onEdit({ dialogue: [...(panel.dialogue ?? []), { speaker: "", text: "" }] })}>
          + Balloon
        </button>
      </fieldset>

      <details className="board-camera">
        <summary>Camera and people</summary>
        <Field label="Shot">
          <select value={panel.shot} disabled={off}
                  onChange={(e) => onEdit({ shot: e.target.value as Panel["shot"] })}>
            <option value="EWS">Extreme wide — the whole place</option>
            <option value="WS">Wide — full figures in their setting</option>
            <option value="MS">Medium — waist up</option>
            <option value="MCU">Medium close-up — head and shoulders</option>
            <option value="CU">Close-up — the face</option>
            <option value="ECU">Extreme close-up — eyes, hands</option>
            <option value="INSERT">Insert — an object or detail</option>
            {["wide", "medium", "close-up"].includes(panel.shot)
              && <option value={panel.shot}>{panel.shot} (older board)</option>}
          </select>
        </Field>
        <Field label="Camera angle">
          <select value={panel.angle ?? "eye"} disabled={off} onChange={(e) => onEdit({ angle: e.target.value })}>
            <option value="eye">Eye level — neutral</option>
            <option value="high">High — looking down, vulnerable</option>
            <option value="low">Low — looking up, powerful</option>
            <option value="bird">Bird's-eye — from above, small in the world</option>
            <option value="worm">Worm's-eye — from the ground</option>
            <option value="dutch">Dutch — tilted, uneasy</option>
          </select>
        </Field>
        <Field label="Framing">
          <select value={panel.framing ?? "single"} disabled={off} onChange={(e) => onEdit({ framing: e.target.value })}>
            <option value="establishing">Establishing</option>
            <option value="single">Single</option>
            <option value="two">Two-shot</option>
            <option value="group">Group</option>
            <option value="ots">Over the shoulder</option>
            <option value="pov">Point of view</option>
            <option value="insert">Insert</option>
          </select>
        </Field>
        <Field label={<>People in frame <em>names from the cast, comma separated</em></>}>
          <input type="text" value={(panel.characters ?? []).join(", ")} disabled={off}
                 onChange={(e) => onEdit({
                   characters: e.target.value.split(",").map((v) => v.trim()).filter(Boolean),
                   character_in_frame: Boolean(e.target.value.trim()),
                 })} />
        </Field>
        <Field label="Setting">
          <input type="text" value={panel.setting} disabled={off} onChange={(e) => onEdit({ setting: e.target.value })} />
        </Field>
        <Field label="Mood">
          <input value={panel.mood ?? ""} disabled={off} onChange={(e) => onEdit({ mood: e.target.value })} />
        </Field>
        <Field label="Current location state">
          <input value={panel.location_state ?? ""} disabled={off}
                 onChange={(e) => onEdit({ location_state: e.target.value })} />
        </Field>
        {cast.map((actor, k) => (
          <div className="field" key={actor.character_id} role="group" aria-labelledby={`board-actor-${k}`}>
            <label id={`board-actor-${k}`}>{actor.name}</label>
            <input aria-label={`${actor.name} expression`} value={actor.expression} disabled={off}
                   onChange={(e) => editActor(k, { expression: e.target.value })} />
            <input aria-label={`${actor.name} body language`} value={actor.body_language} disabled={off}
                   onChange={(e) => editActor(k, { body_language: e.target.value })} />
            <input aria-label={`${actor.name} outfit here`} value={actor.appearance ?? ""} disabled={off}
                   placeholder="Only if it differs from the cast sheet"
                   onChange={(e) => editActor(k, { appearance: e.target.value })} />
          </div>
        ))}
      </details>

      <div className="field">
        <label>Story source <em>what this panel tells</em></label>
        <div className="board-source">
          {[...(panel.covers ?? []), ...(panel.folded ?? []).map((s) => `(not shown) ${s}`)].join(" ")
            || panel.source || "No source sentence: this panel was added by hand."}
        </div>
      </div>

      {!readOnly && (
        <div className="field">
          <label>This panel's place in the comic</label>
          <div className="board-structure">
            <button className="btn small" disabled={busy || index === 0} onClick={() => onMove(-1)}>↑ Earlier</button>
            <button className="btn small" disabled={busy || index >= count - 1} onClick={() => onMove(1)}>↓ Later</button>
            <button className="btn small" disabled={busy} onClick={onAddAfter}>+ After</button>
            <button className="btn small" disabled={busy} onClick={onSplit}>Split</button>
            <button className="btn small" disabled={busy || index >= count - 1} onClick={onMerge}>Join next</button>
            <button className={"btn small" + (confirmRemove ? " danger" : "")} disabled={busy || count < 2}
                    onClick={() => {
                      if (!confirmRemove) { setConfirmRemove(true); return; }
                      setConfirmRemove(false); onRemove();
                    }}>
              {confirmRemove ? (art.drawn ? "Remove it? The picture stays in the Vault" : "Remove it?") : "Remove"}
            </button>
          </div>
        </div>
      )}

      <details className="field">
        <summary>The exact words asked for</summary>
        <div className="board-source">{prompt}</div>
      </details>

      {art.variants.length > 0 && (
        <div className="field">
          <label>Earlier pictures <em>click one to put it back</em></label>
          <div className="board-history">
            {art.variants.map((id) => (
              <button key={id} className="btn small" disabled={busy} title="Put this picture back"
                      onClick={() => onRestore(id)}>
                <img src={vaultUrl(id)} alt="An earlier version" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
