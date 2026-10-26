import { createContext, useContext } from "react";
import type { LiveQueue } from "./liveAudio";

/**
 * The one recording playing across the whole app.
 *
 * A player inside a tab or the viewer stopped the moment you went anywhere
 * else, so listening to a chapter meant not touching the rest of the app.
 * This one lives in the shell and keeps playing while you work.
 */
export interface Track {
  src: string;
  title: string;
  markers?: { time: number; label: string }[];
  /** Seconds to start from, as when a transcript sentence is clicked. */
  start?: number;
  /** A narration still being read: played from its queue, not from `src`. */
  live?: LiveQueue;
}

export const NowPlaying = createContext<{
  track: Track | null;
  play: (track: Track) => void;
  stop: () => void;
}>({ track: null, play: () => {}, stop: () => {} });

export const useNowPlaying = () => useContext(NowPlaying);

/**
 * Where the app-wide player is, for a transcript to follow and to seek.
 *
 * Kept outside React state: the time changes several times a second, and in
 * the shell's state it re-rendered the whole app with it.
 */
function createPlayhead() {
  let state = { src: "", time: 0 };
  const listeners = new Set<() => void>();
  let seeker: ((src: string, time: number) => boolean) | null = null;
  return {
    get: () => state,
    publish(src: string, time: number) {
      state = { src, time };
      for (const listener of listeners) listener();
    },
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    },
    /** The player registers how to seek; the transcript asks it to. */
    attach(fn: (src: string, time: number) => boolean) {
      seeker = fn;
      return () => { if (seeker === fn) seeker = null; };
    },
    seek: (src: string, time: number) => seeker?.(src, time) ?? false,
  };
}

export const playhead = createPlayhead();
