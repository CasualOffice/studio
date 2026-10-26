import { useState } from "react";
import { vaultUrl, type TextOverflow } from "../../lib/api";
import type { Coverage } from "../../lib/boardDraft";
import type { Cell } from "../../lib/layout";

/** The compositor's margin around a page's panels, in pixels. */
const MARGIN = 34;

/**
 * A composed page with every panel on it clickable, so a panel is fixed
 * where it is seen: click it on the page, and its fixes open beside it.
 */
function PagePicture({ id, n, total, cells, onPick, selected }: {
  id: string; n: number; total: number;
  cells: { index: number; cell: Cell }[];
  onPick: (index: number) => void;
  selected: number;
}) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  return (
    <div className="board-page">
      <img src={vaultUrl(id)} alt={`Page ${n} of ${total}`}
           onLoad={(e) => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
      {size && cells.map(({ index, cell: [x, y, w, h] }) => {
        const iw = size.w - MARGIN * 2, ih = size.h - MARGIN * 2;
        const style = {
          left: `${((MARGIN + x * iw) / size.w) * 100}%`, top: `${((MARGIN + y * ih) / size.h) * 100}%`,
          width: `${((w * iw) / size.w) * 100}%`, height: `${((h * ih) / size.h) * 100}%`,
        };
        return (
          <button key={index} type="button" className="board-page-cell" style={style}
                  aria-label={`Panel ${index + 1}`} aria-current={selected === index ? "true" : undefined}
                  onClick={() => onPick(index)} />
        );
      })}
    </div>
  );
}

export default function PagesPane({ pages, pageCells, overflow, coverage, stale, busy, selected, onPick,
                                    onUpdate, headRef }: {
  pages: string[];
  /** For each composed page, its panels' board indices and cells. */
  pageCells: { index: number; cell: Cell }[][];
  overflow: (TextOverflow & { panelId?: string })[];
  coverage: Coverage | null;
  /** The pages no longer match the panels: words or pictures changed. */
  stale: boolean;
  busy: boolean;
  selected: number;
  onPick: (index: number) => void;
  onUpdate: () => void;
  headRef: React.RefObject<HTMLDivElement | null>;
}) {
  const lost = coverage ? coverage.missing_total ?? coverage.missing.length : 0;
  return (
    <section className="board-pages" aria-label="Pages">
      <div className="sub-head" ref={headRef} tabIndex={-1}>
        {pages.length === 0 ? "Pages" : pages.length === 1 ? "The page" : `${pages.length} pages`}
      </div>
      {coverage && coverage.total > 0 && (
        <div className={"notice" + (lost ? " warn" : " good")}>
          <strong>{coverage.percent}% of the story is on the page</strong>
          {lost > 0 ? (
            <details>
              <summary>
                {lost} sentence{lost === 1 ? " is" : "s are"} neither drawn nor lettered
                {coverage.missing.length < lost ? ` (first ${coverage.missing.length} shown)` : ""}. More pages
                make room for them.
              </summary>
              {coverage.missing.map((text, i) => <p key={i}>{text}</p>)}
            </details>
          ) : " Every sentence is drawn or lettered, and every line is in a balloon."}
        </div>
      )}
      {stale && pages.length > 0 && (
        <div className="notice">
          The pages are older than the panels.
          <button type="button" className="btn small" disabled={busy} onClick={onUpdate}>Update the pages</button>
        </div>
      )}
      {overflow.length > 0 && (
        <div className="notice warn">
          <strong>{overflow.length} panel{overflow.length === 1 ? "" : "s"} lost text on the page</strong>
          <details>
            <summary>Shorten the words, then update the pages.</summary>
            {overflow.map((o) => {
              const missed = o.spoken - o.lettered;
              const parts = [missed > 0 ? `${missed} of ${o.spoken} spoken line${o.spoken === 1 ? "" : "s"} not drawn` : "",
                             o.caption_clipped ? "caption cut short" : ""].filter(Boolean);
              return (
                <p key={`${o.panel}-${o.page}`}>
                  <button className="btn small" onClick={() => onPick(o.panel)}>Panel {o.panel + 1}</button>
                  {" "}{parts.join(", ")}{o.page ? ` (page ${o.page})` : ""}
                </p>
              );
            })}
          </details>
        </div>
      )}
      {pages.map((id, k) => (
        <PagePicture key={id} id={id} n={k + 1} total={pages.length} cells={pageCells[k] ?? []}
                     onPick={onPick} selected={selected} />
      ))}
    </section>
  );
}
