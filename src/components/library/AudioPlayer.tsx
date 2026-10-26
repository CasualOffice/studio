import { useCallback, useEffect, useRef, useState } from "react";
import { loadPref, savePref } from "../../lib/prefs";
import { clock, markerAt } from "../../lib/library";
import { playhead } from "../../lib/nowPlaying";

export interface AudioMarker { time: number; label: string }

// Up to 4x for listening through a draft; pitch is preserved at every rate.
export const RATES = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4];
const SKIP = 15;

/**
 * A player for listening, not for checking that a file exists.
 *
 * The bare `<audio controls>` WebKit draws is a thin strip with no skip, no
 * speed and a scrubber too small to hit, which is fine for a ding and useless
 * for a forty-minute chapter. This is the player a book needs: big play
 * button, skip back to catch a missed line, speed, and chapters on the bar.
 *
 * Shortcuts work while focus is inside the player, or anywhere on the page
 * with `shortcuts="global"`, which the full-screen viewer uses because there
 * is nothing else on screen they could mean. Space plays and pauses, J and L
 * skip fifteen seconds, comma and full stop change speed.
 *
 * The speed is remembered across items and launches; a listener who likes
 * 1.25x likes it for every chapter.
 *
 * With `follow`, the app-wide player reports where it is and takes seeks
 * through `playhead`, so a narration's transcript can highlight the sentence
 * being read and jump to the one clicked. `start` opens partway in.
 */
export default function AudioPlayer({
  src, title, markers = [], autoPlay = false, shortcuts = "focus", follow = false, start,
}: {
  src: string;
  title?: string;
  markers?: AudioMarker[];
  autoPlay?: boolean;
  shortcuts?: "focus" | "global";
  follow?: boolean;
  start?: number;
}) {
  const audio = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rate, setRate] = useState(() => loadPref("audio.rate", 1));
  const [failed, setFailed] = useState(false);

  useEffect(() => { setTime(0); setDuration(0); setFailed(false); setPlaying(false); }, [src]);
  useEffect(() => {
    if (audio.current) { audio.current.playbackRate = rate; audio.current.preservesPitch = true; }
    savePref("audio.rate", rate);
  }, [rate, src]);

  const toggle = useCallback(() => {
    const a = audio.current;
    if (!a) return;
    if (a.paused) void a.play()?.catch(() => setPlaying(false));
    else a.pause();
  }, []);

  const seek = useCallback((t: number) => {
    const a = audio.current;
    if (!a) return;
    const end = Number.isFinite(a.duration) ? a.duration : duration;
    const next = Math.max(0, end ? Math.min(end, t) : t);
    a.currentTime = next;
    setTime(next);
  }, [duration]);

  const skip = useCallback((by: number) => seek((audio.current?.currentTime ?? 0) + by), [seek]);

  useEffect(() => {
    if (!follow) return;
    return playhead.attach((at, t) => {
      if (at !== src) return false;
      seek(t);
      if (audio.current?.paused) void audio.current.play()?.catch(() => setPlaying(false));
      return true;
    });
  }, [follow, src, seek]);

  const stepRate = useCallback((dir: 1 | -1) => {
    setRate((r: number) => {
      const i = RATES.indexOf(r);
      const at = i < 0 ? RATES.indexOf(1) : i;
      return RATES[Math.max(0, Math.min(RATES.length - 1, at + dir))];
    });
  }, []);

  /** True when the key was one of ours and has been handled. */
  const onShortcut = useCallback((e: KeyboardEvent | React.KeyboardEvent): boolean => {
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === "INPUT" && (t as HTMLInputElement).type !== "range"
              || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return false;
    if (e.metaKey || e.ctrlKey || e.altKey) return false;
    // Space and Enter belong to a focused button or link: it was being
    // swallowed as play/pause, so "Back 15", a chapter, Close or Delete could
    // not be pressed from the keyboard.
    if ((e.key === " " || e.key === "Enter") && t?.closest("button, a, [role=button], summary")) return false;
    switch (e.key) {
      case " ": case "k": case "K": toggle(); break;
      case "j": case "J": skip(-SKIP); break;
      case "l": case "L": skip(SKIP); break;
      case ",": case "<": stepRate(-1); break;
      case ".": case ">": stepRate(1); break;
      default: return false;
    }
    e.preventDefault();
    e.stopPropagation();
    return true;
  }, [toggle, skip, stepRate]);

  useEffect(() => {
    if (shortcuts !== "global") return;
    const onKey = (e: KeyboardEvent) => { onShortcut(e); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [shortcuts, onShortcut]);

  const chapter = markerAt(markers, time);
  const pct = duration ? (time / duration) * 100 : 0;

  return (
    <div className="audio-player" role="group" aria-label={title ? `Player: ${title}` : "Audio player"}
         onKeyDown={shortcuts === "focus" ? (e) => { onShortcut(e); } : undefined}>
      <audio
        ref={audio}
        src={src}
        preload="metadata"
        autoPlay={autoPlay}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(e) => {
          setTime(e.currentTarget.currentTime);
          if (follow) playhead.publish(src, e.currentTarget.currentTime);
        }}
        onLoadedMetadata={(e) => {
          setDuration(e.currentTarget.duration);
          e.currentTarget.playbackRate = rate;
          if (start) e.currentTarget.currentTime = start;
        }}
        onDurationChange={(e) => {
          if (Number.isFinite(e.currentTarget.duration)) setDuration(e.currentTarget.duration);
        }}
        onError={() => setFailed(true)}
      />
      {title && <div className="ap-title" title={title}>{title}</div>}
      {chapter && <div className="ap-chapter">{chapter.label}</div>}

      <div className="ap-scrub">
        <span className="ap-time">{clock(time)}</span>
        <div className="ap-track" style={{ ["--pct" as string]: `${pct}%` }}>
          <input
            type="range"
            min={0}
            max={duration || 0}
            step={0.1}
            value={Math.min(time, duration || 0)}
            onChange={(e) => seek(Number(e.target.value))}
            aria-label="Seek"
            aria-valuetext={`${clock(time)} of ${clock(duration)}`}
            disabled={!duration}
          />
          {duration > 0 && markers.map((m) => (
            <span key={`${m.time}-${m.label}`} className="ap-marker" title={m.label}
                  style={{ left: `${(m.time / duration) * 100}%` }} />
          ))}
        </div>
        <span className="ap-time">{duration ? clock(duration) : "–:––"}</span>
      </div>

      <div className="ap-controls">
        <button className="ap-btn" onClick={() => skip(-SKIP)}
                aria-label="Back 15 seconds" title="Back 15 seconds (J)">↺ 15</button>
        <button className="ap-btn ap-play" onClick={toggle}
                aria-label={playing ? "Pause" : "Play"} title={playing ? "Pause (Space)" : "Play (Space)"}>
          {playing ? "❚❚" : "▶"}
        </button>
        <button className="ap-btn" onClick={() => skip(SKIP)}
                aria-label="Forward 15 seconds" title="Forward 15 seconds (L)">15 ↻</button>
        <select className="ap-rate" value={rate} aria-label="Playback speed"
                title="Playback speed (, and .)"
                onChange={(e) => setRate(Number(e.target.value))}>
          {RATES.map((r) => <option key={r} value={r}>{r}×</option>)}
        </select>
      </div>

      {failed && (
        <div className="ap-error" role="alert">
          This recording could not be played. If the vault was locked, unlock it
          and open it again.
        </div>
      )}

      {markers.length > 1 && (
        <ol className="ap-chapters">
          {[...markers].sort((a, b) => a.time - b.time).map((m) => (
            <li key={`${m.time}-${m.label}`}>
              <button className={m === chapter ? "on" : ""} aria-current={m === chapter ? "true" : undefined}
                onClick={() => seek(m.time)}>
                <span>{m.label}</span><span className="ap-time">{clock(m.time)}</span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
