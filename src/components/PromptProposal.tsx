import { useMemo, useState } from "react";
import type { AssistGoal, AssistResult, AssistVariant } from "../lib/types";
import { diffWords, lostWords, rebuild } from "../lib/promptDiff";
import { Choice } from "./creation";

/** What each goal asks the writer for, said where the choice is made. */
export const GOALS: readonly (readonly [AssistGoal, string])[] = [
  ["enhance", "Enhance"],
  ["tighten", "Tighten"],
  ["style", "Style"],
] as const;

export const GOAL_HINT: Record<AssistGoal, string> = {
  enhance: "A faithful version and a fuller description, for you to compare.",
  tighten: "Fixes spelling and clarity only. Adds nothing.",
  style: "The same picture in three different looks.",
};

/** The goal picker beside the prompt-help button, in Studio and Video. */
export function PromptGoal({ value, onChange, disabled }: {
  value: AssistGoal;
  onChange: (g: AssistGoal) => void;
  disabled?: boolean;
}) {
  return (
    <Choice label="Prompt help" options={GOALS} value={value} onChange={onChange}
            hint={GOAL_HINT[value]} disabled={disabled} chip />
  );
}

/**
 * The versions a reply offers. A reply from an engine older than variants
 * carries one suggestion; it is shown as one, with what it said it added.
 */
function variantsOf(result: AssistResult): AssistVariant[] {
  if (result.variants?.length) return result.variants;
  return [{
    kind: "enriched", label: "Suggested", prompt: result.prompt,
    additions: [], missing: result.dropped ?? [], reworded: [],
  }];
}

/**
 * What the enhancer wants to do, shown before it does it.
 *
 * It used to overwrite the prompt the moment it had a rewrite, and when it had
 * none it said "your prompt is already specific enough" — which was often a
 * lie, because a rewrite had been attempted and thrown away for dropping a
 * word. Three outcomes looked identical from the outside: nothing needed
 * doing, something was tried and rejected, and the request named nothing at
 * all. All three read as the button not working.
 *
 * So each one says which it is, the suggestion is a proposal rather than an
 * edit, and now there are several to choose between: a faithful version, a
 * fuller one and a stylistic one. Each is shown against your own words --
 * what it added highlighted, what it lost named -- and every phrase the writer
 * added is a switch, so the fuller version can be taken without the parts you
 * did not want.
 */
export default function PromptProposal({
  result, busy, onAccept, onDismiss,
}: {
  result: AssistResult;
  busy?: boolean;
  onAccept: (text: string) => void;
  onDismiss: () => void;
}) {
  const outcome = result.outcome
    ?? (result.unclear ? "unclear"
      : result.changed === false ? "already_specific" : "proposal");
  const variants = useMemo(() => variantsOf(result), [result]);
  // The one the engine put first: the richest version that lost nothing.
  const first = Math.max(0, variants.findIndex((v) => v.prompt === result.prompt));
  const [at, setAt] = useState(first);
  // Per variant, the additions switched off.
  const [off, setOff] = useState<Record<number, string[]>>({});
  // A new reply into the same card starts over: the tab and the switches
  // belonged to the last one.
  const [shown, setShown] = useState(result);
  if (shown !== result) { setShown(result); setAt(first); setOff({}); }

  if (outcome === "already_specific") {
    return (
      <div className="proposal plain">
        <b>Nothing to add.</b> Your prompt already says this clearly, so it was
        left alone.
        {result.note && <div className="body">{result.note}</div>}
        <div className="row">
          <button className="btn small" onClick={onDismiss}>Close</button>
        </div>
      </div>
    );
  }

  if (outcome === "unclear") {
    return (
      <div className="proposal warn">
        <b>This does not say what to draw yet.</b>
        <div className="body">{result.note}</div>
        <div className="row">
          <button className="btn small" onClick={onDismiss}>Close</button>
        </div>
      </div>
    );
  }

  if (outcome === "rewrite_rejected") {
    return (
      <div className="proposal warn">
        <b>Kept your words.</b>
        <div className="body">{result.note}</div>
        {result.attempt && (
          <>
            {/* Shown on purpose. If the rewrite was good and the check was
                wrong, that should be visible and usable, not discarded. */}
            <div className="label">What it wrote, if you want it anyway</div>
            <div className="text rejected">{result.attempt}</div>
            <div className="row">
              <button className="btn small" disabled={busy}
                      onClick={() => onAccept(result.attempt!)}>
                Use it anyway
              </button>
              <button className="btn small" onClick={onDismiss}>Keep mine</button>
            </div>
          </>
        )}
        {!result.attempt && (
          <div className="row">
            <button className="btn small" onClick={onDismiss}>Close</button>
          </div>
        )}
      </div>
    );
  }

  const index = Math.min(at, variants.length - 1);
  const variant = variants[index];
  const leftOut = off[index] ?? [];
  const text = rebuild(variant.prompt, leftOut);
  const tokens = diffWords(result.original, text);
  const lost = lostWords(result.original, text, [
    ...(result.removed ?? []),
    // Spelling fixes are changes, not losses: "hous" is still there as "house".
    ...(result.corrected ?? []).map(([typed]) => typed),
  ]);
  const toggle = (phrase: string) => setOff((prev) => {
    const now = prev[index] ?? [];
    return {
      ...prev,
      [index]: now.includes(phrase) ? now.filter((p) => p !== phrase) : [...now, phrase],
    };
  });
  const heading = result.target_label ? `Suggested for ${result.target_label}` : "Suggested prompt";

  return (
    <div className="proposal">
      <b>{heading}</b>
      {/* The engine writes a note on the success reply too -- it is how a
          rewrite made without looking at the picture says so -- and an edit
          is routed as an edit whether or not a reader is installed, which
          makes "your picture was not looked at" the common case. */}
      {result.note && <div className="body">{result.note}</div>}

      {variants.length > 1 && (
        <div className="variant-tabs" role="tablist" aria-label="Versions">
          {variants.map((v, i) => (
            <button key={v.kind + i} role="tab" type="button"
                    aria-selected={i === index}
                    className={"variant-tab" + (v.missing.length ? " lossy" : "")}
                    title={v.missing.length ? `May have lost: ${v.missing.join(", ")}` : undefined}
                    onClick={() => setAt(i)}>
              {v.label}{v.missing.length ? " ⚠" : ""}
            </button>
          ))}
        </div>
      )}

      <div role={variants.length > 1 ? "tabpanel" : undefined}>
        {/* Your words plain, its words marked: the first question anyone
            asks of a suggestion is what it added. */}
        <div className="text diff" data-testid="proposal-text">
          {tokens.map((t, i) => t.added
            ? <ins key={i}>{t.text}</ins>
            : <span key={i}>{t.text}</span>)}
        </div>

        {!!variant.missing.length && (
          <div className="chips">
            {/* From the engine's constraint checks: a count, an exclusion,
                quoted text or a pace this version does not keep. */}
            <span className="label">no longer says</span>
            {variant.missing.map((w) => (
              <span className="chip drop" key={w}>{w}</span>
            ))}
          </div>
        )}
        {!!lost.length && (
          <div className="chips">
            <span className="label">not in this version</span>
            {lost.map((w) => <span className="chip lost" key={w}>{w}</span>)}
          </div>
        )}
        {!!variant.additions.length && (
          <div className="chips">
            <span className="label">added — tap to leave out</span>
            {variant.additions.map((phrase) => {
              const on = !leftOut.includes(phrase);
              return (
                <button key={phrase} type="button" aria-pressed={on}
                        className={"chip add toggle" + (on ? "" : " off")}
                        disabled={busy} onClick={() => toggle(phrase)}>
                  {phrase}
                </button>
              );
            })}
          </div>
        )}
        {!!variant.reworded.length && (
          <div className="seen">
            Reworded rather than repeated: {variant.reworded.join(", ")}. The
            writer read it as the same request; check it says what you meant.
          </div>
        )}
      </div>

      {!!result.corrected?.length && (
        <div className="chips">
          <span className="label">spelling</span>
          {result.corrected.map(([typed, written]) => (
            <span className="chip add" key={typed}>{typed} → {written}</span>
          ))}
        </div>
      )}
      {!!result.removed?.length && (
        <div className="chips">
          {/* A different fact from "no longer says": these were taken out on
              purpose, before anything was written. Quoted text is never
              among them. */}
          <span className="label">removed</span>
          {result.removed.map((w) => (
            <span className="chip drop" key={w}>{w}</span>
          ))}
        </div>
      )}
      {result.saw_image && (result.regions?.length || result.description) && (
        <div className="seen">
          <span className="label">read from your picture</span>{" "}
          {result.regions?.length ? result.regions.join(" · ") : result.description}
          {result.target_region && <> — “it” read as {result.target_region}</>}
        </div>
      )}
      {result.released && (
        <div className="seen">
          {result.released} was unloaded to make room for the writer, so your
          next run loads it again first.
        </div>
      )}

      <div className="row">
        <button className="btn small primary" disabled={busy}
                onClick={() => onAccept(text)}>
          Use this
        </button>
        <button className="btn small" onClick={onDismiss}>Keep mine</button>
      </div>
    </div>
  );
}
