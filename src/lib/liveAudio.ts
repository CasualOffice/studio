import type { AudioSpan } from "./types";

/**
 * Narration played while it is still being read.
 *
 * The engine sends each sentence as 16-bit PCM the moment it exists. WebKit's
 * Media Source does not take WAV, and a blob URL only plays what it already
 * holds, so sentences are queued as Web Audio buffers instead, each scheduled
 * to start where the last one ends. Listening starts after the first
 * sentence rather than after the chapter.
 *
 * Everything received is kept, so the listener can pause, skip back and
 * click any sentence already read. When the reader falls behind the voice,
 * playback waits at the end of what has arrived and carries on from there.
 */

/** The parts of an AudioContext the queue uses; a fake clock in tests. */
export interface AudioClock {
  readonly currentTime: number;
  readonly destination: unknown;
  createBuffer(channels: number, length: number, sampleRate: number): { getChannelData(channel: number): Float32Array };
  createBufferSource(): {
    buffer: unknown;
    onended: (() => void) | null;
    connect(destination: unknown): unknown;
    start(when?: number, offset?: number): void;
    stop(): void;
  };
  resume(): Promise<void>;
  suspend(): Promise<void>;
  close(): Promise<void>;
}

export interface LiveChunk { index: number; start: number; pcm: Int16Array; span: AudioSpan; sampleRate?: number }

export type LiveState = "waiting" | "playing" | "paused" | "ended";

// Sources are started this far ahead of the clock so the first samples are
// never in the past by the time the audio thread sees them.
const LEAD = 0.05;

export class LiveQueue {
  readonly rate: number;
  /** Sentences received so far, by their index in the recording. */
  readonly spans: AudioSpan[] = [];
  done = false;
  private chunks: { start: number; samples: Float32Array; buffer?: unknown }[] = [];
  private received = 0;
  private paused = false;
  private closed = false;
  // The clock time at which sample `anchorSample` plays; null when stopped.
  private anchor: number | null = null;
  private anchorSample = 0;
  private scheduledTo = 0;
  private sources: ReturnType<AudioClock["createBufferSource"]>[] = [];
  private listeners = new Set<() => void>();
  private version = 0;

  constructor(private clock: AudioClock, readonly title: string, rate = 24000) {
    this.rate = rate;
  }

  /** Seconds of narration received so far. */
  get duration() { return this.received / this.rate; }

  push(chunk: LiveChunk) {
    if (this.closed) return;
    this.spans[chunk.index] = chunk.span;
    // A repeat of audio already held (a retried event) is ignored; a gap
    // cannot happen on one ordered pipe, but is filled with silence if it does.
    if (chunk.start + chunk.pcm.length <= this.received) return this.changed();
    const skip = Math.max(0, this.received - chunk.start);
    const gap = Math.max(0, chunk.start - this.received);
    const samples = new Float32Array(gap + chunk.pcm.length - skip);
    for (let i = skip; i < chunk.pcm.length; i++) samples[gap + i - skip] = chunk.pcm[i] / 32768;
    this.chunks.push({ start: this.received, samples });
    this.received += samples.length;
    if (this.anchor === null && !this.paused && this.chunks.length === 1) this.anchor = this.clock.currentTime + LEAD;
    this.schedule();
    this.changed();
  }

  /** The engine has finished reading: what has arrived is the whole of it. */
  finish() { this.done = true; this.changed(); }

  /** Samples heard so far. */
  positionSamples(): number {
    if (this.anchor === null) return this.anchorSample;
    const at = this.anchorSample + Math.floor((this.clock.currentTime - this.anchor) * this.rate);
    return Math.max(this.anchorSample, Math.min(at, this.scheduledTo));
  }

  position() { return this.positionSamples() / this.rate; }

  state(): LiveState {
    if (this.paused) return "paused";
    const at = this.positionSamples();
    if (at >= this.received) return this.done ? "ended" : "waiting";
    return this.anchor === null ? "paused" : "playing";
  }

  /** The sentence being heard: the last one that has started. */
  current(): number {
    const at = this.positionSamples();
    let found = -1;
    this.spans.forEach((span, i) => { if (span && span.start <= at) found = i; });
    return found;
  }

  pause() {
    if (this.paused) return;
    this.anchorSample = this.positionSamples();
    this.stopSources();
    this.paused = true;
    void this.clock.suspend().catch(() => {});
    this.changed();
  }

  play() {
    if (!this.paused && this.anchor !== null) return;
    this.paused = false;
    // Played to the end: start again from the beginning.
    if (this.done && this.anchorSample >= this.received) this.anchorSample = 0;
    void this.clock.resume().catch(() => {});
    this.restart(this.anchorSample);
  }

  /** Jump within what has arrived. */
  seek(seconds: number) {
    const sample = Math.max(0, Math.min(Math.round(seconds * this.rate), this.received));
    if (this.paused) {
      this.anchorSample = sample;
      this.changed();
      return;
    }
    this.restart(sample);
  }

  close() {
    this.closed = true;
    this.stopSources();
    this.listeners.clear();
    void this.clock.close().catch(() => {});
  }

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  /** Changes whenever anything but the clock does, for useSyncExternalStore. */
  snapshot = () => this.version;

  private restart(sample: number) {
    this.stopSources();
    this.anchorSample = sample;
    this.scheduledTo = sample;
    this.anchor = this.clock.currentTime + LEAD;
    this.schedule();
    this.changed();
  }

  private schedule() {
    if (this.paused || this.anchor === null || this.closed) return;
    while (this.scheduledTo < this.received) {
      const chunk = this.chunks.find((c) => c.start <= this.scheduledTo && this.scheduledTo < c.start + c.samples.length);
      if (!chunk) break;
      let when = this.anchor + (this.scheduledTo - this.anchorSample) / this.rate;
      if (when < this.clock.currentTime + LEAD / 2) {
        // The voice fell behind and playback ran dry: carry on from here,
        // now, rather than starting a sentence partway through.
        this.anchorSample = this.scheduledTo;
        this.anchor = this.clock.currentTime + LEAD;
        when = this.anchor;
      }
      if (!chunk.buffer) {
        const buffer = this.clock.createBuffer(1, chunk.samples.length, this.rate);
        buffer.getChannelData(0).set(chunk.samples);
        chunk.buffer = buffer;
      }
      const source = this.clock.createBufferSource();
      source.buffer = chunk.buffer;
      source.connect(this.clock.destination);
      source.onended = () => this.changed();
      source.start(when, (this.scheduledTo - chunk.start) / this.rate);
      this.sources.push(source);
      this.scheduledTo = chunk.start + chunk.samples.length;
    }
  }

  private stopSources() {
    for (const source of this.sources) {
      source.onended = null;
      try { source.stop(); } catch { /* never started, or already ended */ }
    }
    this.sources = [];
    this.anchor = null;
  }

  private changed() {
    this.version++;
    for (const listener of this.listeners) listener();
  }
}

/** An AudioContext, or null where there is none (tests, a locked-down page). */
export function audioClock(): AudioClock | null {
  const Context = (globalThis as { AudioContext?: new () => AudioClock; webkitAudioContext?: new () => AudioClock }).AudioContext
    ?? (globalThis as { webkitAudioContext?: new () => AudioClock }).webkitAudioContext;
  try { return Context ? new Context() : null; } catch { return null; }
}
