import { useEffect, useState, useSyncExternalStore } from "react";
import { clock } from "../../lib/library";
import type { LiveQueue } from "../../lib/liveAudio";

const SKIP = 15;

/**
 * The now-playing bar for a narration that is still being read.
 *
 * Same controls as the file player, over what has arrived so far: the bar's
 * end grows as sentences come in, and "reading" shows the voice is still
 * going. Speed is not offered: Web Audio changes pitch with rate.
 */
export default function LivePlayer({ queue, title }: { queue: LiveQueue; title: string }) {
  useSyncExternalStore(queue.subscribe, queue.snapshot);
  // The clock moves on its own; redraw a few times a second to follow it.
  const [, tick] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => tick((n) => n + 1), 250);
    return () => window.clearInterval(timer);
  }, []);
  const state = queue.state();
  const time = queue.position();
  const duration = queue.duration;
  const pct = duration ? (time / duration) * 100 : 0;
  const playing = state === "playing" || state === "waiting";

  return (
    <div className="audio-player live" role="group" aria-label={`Player: ${title}`}>
      <div className="ap-title" title={title}>{title}</div>
      <div className="ap-chapter" aria-live="polite">
        {state === "waiting" ? (queue.done ? "" : "Reading the next sentence…")
          : queue.done ? "" : "Reading aloud · the rest follows as it is read"}
      </div>
      <div className="ap-scrub">
        <span className="ap-time">{clock(time)}</span>
        <div className="ap-track" style={{ ["--pct" as string]: `${pct}%` }}>
          <input type="range" min={0} max={duration || 0} step={0.1} value={Math.min(time, duration || 0)}
            onChange={(e) => queue.seek(Number(e.target.value))} aria-label="Seek"
            aria-valuetext={`${clock(time)} of ${clock(duration)}`} disabled={!duration} />
        </div>
        <span className="ap-time">{clock(duration)}{queue.done ? "" : "+"}</span>
      </div>
      <div className="ap-controls">
        <button className="ap-btn" onClick={() => queue.seek(time - SKIP)} aria-label="Back 15 seconds">↺ 15</button>
        <button className="ap-btn ap-play" onClick={() => (playing ? queue.pause() : queue.play())}
          aria-label={playing ? "Pause" : "Play"}>{playing ? "❚❚" : "▶"}</button>
        <button className="ap-btn" onClick={() => queue.seek(time + SKIP)} aria-label="Forward 15 seconds">15 ↻</button>
      </div>
    </div>
  );
}
