import type { AssistResult } from "../lib/types";

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
 * edit, and a rejected attempt is shown too — if the enhancer wrote something
 * good and the checker was wrong, that is visible instead of silently lost.
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

  if (outcome === "already_specific") {
    return (
      <div className="proposal plain">
        <b>Nothing to add.</b> Your prompt already names what to draw, so it was
        left alone.
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

  return (
    <div className="proposal">
      <b>Suggested prompt</b>
      <div className="text">{result.prompt}</div>

      {!!result.added?.length && (
        <div className="chips">
          <span className="label">added</span>
          {result.added.slice(0, 14).map((w) => (
            <span className="chip add" key={w}>{w}</span>
          ))}
        </div>
      )}
      {!!result.removed?.length && (
        <div className="chips">
          <span className="label">removed</span>
          {result.removed.map((w) => (
            <span className="chip drop" key={w}>{w}</span>
          ))}
        </div>
      )}
      {result.saw_image && result.description && (
        <div className="seen">
          <span className="label">read from your picture</span> {result.description}
        </div>
      )}

      <div className="row">
        <button className="btn small primary" disabled={busy}
                onClick={() => onAccept(result.prompt)}>
          Use this
        </button>
        <button className="btn small" onClick={onDismiss}>Keep mine</button>
      </div>
    </div>
  );
}
