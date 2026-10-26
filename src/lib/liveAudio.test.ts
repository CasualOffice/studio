import { describe, expect, it } from "vitest";
import { LiveQueue, type AudioClock } from "./liveAudio";

/** A clock that only moves when told to, recording what was scheduled. */
function fakeClock() {
  const started: { when: number; offset: number; length: number; stopped: boolean }[] = [];
  const clock = {
    currentTime: 0,
    destination: {},
    suspended: false,
    closed: false,
    createBuffer: (_c: number, length: number) => {
      const data = new Float32Array(length);
      return { length, getChannelData: () => data };
    },
    createBufferSource() {
      const entry = { when: 0, offset: 0, length: 0, stopped: false };
      return {
        buffer: null as unknown,
        onended: null as (() => void) | null,
        connect: () => {},
        start(when = 0, offset = 0) {
          Object.assign(entry, { when, offset, length: (this.buffer as { length: number }).length });
          started.push(entry);
        },
        stop() { entry.stopped = true; },
      };
    },
    resume: async () => { clock.suspended = false; },
    suspend: async () => { clock.suspended = true; },
    close: async () => { clock.closed = true; },
  };
  return { clock: clock as typeof clock & AudioClock, started };
}

const RATE = 24000;
const chunk = (index: number, start: number, seconds: number) => ({
  index, start, pcm: new Int16Array(Math.round(seconds * RATE)).fill(1000),
  span: { start, end: start + Math.round(seconds * RATE), text: `Sentence ${index}.`, voice: "af_heart" },
});

describe("live narration", () => {
  it("plays each sentence straight after the one before", () => {
    const { clock, started } = fakeClock();
    const queue = new LiveQueue(clock, "Story");
    queue.push(chunk(0, 0, 1));
    queue.push(chunk(1, RATE, 2));
    expect(started.map((s) => s.when)).toEqual([0.05, 1.05]);
    clock.currentTime = 1.55;
    expect(queue.position()).toBeCloseTo(1.5);
    expect(queue.current()).toBe(1);
    expect(queue.state()).toBe("playing");
    expect(queue.duration).toBe(3);
  });

  it("waits at the end of what has arrived and carries on when the next sentence comes", () => {
    const { clock, started } = fakeClock();
    const queue = new LiveQueue(clock, "Story");
    queue.push(chunk(0, 0, 1));
    clock.currentTime = 3;
    expect(queue.state()).toBe("waiting");
    expect(queue.position()).toBe(1);
    queue.push(chunk(1, RATE, 1));
    // Scheduled from now, not in the past, so its start is not cut off.
    expect(started[1].when).toBeCloseTo(3.05);
    expect(started[1].offset).toBe(0);
    clock.currentTime = 3.55;
    expect(queue.position()).toBeCloseTo(1.5);
  });

  it("pauses where it is and resumes from there", () => {
    const { clock, started } = fakeClock();
    const queue = new LiveQueue(clock, "Story");
    queue.push(chunk(0, 0, 2));
    clock.currentTime = 1.05;
    queue.pause();
    expect(started[0].stopped).toBe(true);
    expect(queue.state()).toBe("paused");
    clock.currentTime = 10;
    expect(queue.position()).toBeCloseTo(1);
    queue.play();
    expect(started[1].offset).toBeCloseTo(1);
    expect(started[1].when).toBeCloseTo(10.05);
  });

  it("seeks to any sentence already read", () => {
    const { clock, started } = fakeClock();
    const queue = new LiveQueue(clock, "Story");
    queue.push(chunk(0, 0, 1));
    queue.push(chunk(1, RATE, 1));
    queue.seek(1.5);
    expect(started.slice(0, 2).every((s) => s.stopped)).toBe(true);
    expect(started[2]).toMatchObject({ offset: 0.5 });
    expect(started[2].when).toBeCloseTo(0.05);
    clock.currentTime = 0.05;
    expect(queue.current()).toBe(1);
    // Never past what has arrived.
    queue.seek(99);
    expect(queue.position()).toBe(2);
  });

  it("ends when the reader has finished and everything has played, and restarts on play", () => {
    const { clock, started } = fakeClock();
    const queue = new LiveQueue(clock, "Story");
    queue.push(chunk(0, 0, 1));
    queue.finish();
    clock.currentTime = 2;
    expect(queue.state()).toBe("ended");
    queue.pause();
    queue.play();
    expect(started.at(-1)?.offset).toBe(0);
  });

  it("ignores a repeated chunk and stops sounding once closed", () => {
    const { clock, started } = fakeClock();
    const queue = new LiveQueue(clock, "Story");
    queue.push(chunk(0, 0, 1));
    queue.push(chunk(0, 0, 1));
    expect(queue.duration).toBe(1);
    queue.close();
    queue.push(chunk(1, RATE, 1));
    expect(started).toHaveLength(1);
    expect(started[0].stopped).toBe(true);
    expect(clock.closed).toBe(true);
  });

  it("tells subscribers when anything changes", () => {
    const { clock } = fakeClock();
    const queue = new LiveQueue(clock, "Story");
    let calls = 0;
    const unsubscribe = queue.subscribe(() => { calls++; });
    const before = queue.snapshot();
    queue.push(chunk(0, 0, 1));
    expect(calls).toBeGreaterThan(0);
    expect(queue.snapshot()).not.toBe(before);
    unsubscribe();
  });
});
