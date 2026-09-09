import type { ModelStatus } from "./types";

/** Plain-language framing, so nobody has to think in pixels. */
export const SHAPES: { key: string; label: string; w: number; h: number }[] = [
  { key: "square", label: "Square", w: 1024, h: 1024 },
  { key: "wide", label: "Wide", w: 1024, h: 576 },
  { key: "tall", label: "Tall", w: 576, h: 1024 },
  { key: "photo", label: "Photo", w: 1024, h: 768 },
];

export type Quality = "fast" | "balanced" | "best";

export const QUALITY_LABEL: Record<Quality, string> = {
  fast: "Quick look",
  balanced: "Balanced",
  best: "Take your time",
};

/**
 * Steps for a quality setting.
 *
 * Anchored on the model's own default rather than a fixed number: a distilled
 * checkpoint is tuned for ~4 steps and gains almost nothing from 40, while a
 * base model at 4 steps produces mush. Multiplying the model's default
 * respects both.
 */
export function stepsFor(model: ModelStatus | undefined, quality: Quality): number {
  const base = Math.max(2, model?.steps_default || 8);
  const factor = quality === "fast" ? 1 : quality === "balanced" ? 1.5 : 2.5;
  return Math.min(50, Math.max(2, Math.round(base * factor)));
}

/**
 * Pick a model on the user's behalf.
 *
 * Prefers the heaviest model that still comfortably fits, on the rough
 * assumption that more resident weights means better output — but never picks
 * something that only runs under memory pressure when a comfortable option
 * exists.
 */
export function autoPick(models: ModelStatus[], task: string): ModelStatus | undefined {
  const usable = models.filter(
    (m) => m.installed && m.fit !== "broken" && !m.hopeless && m.tasks.includes(task as never)
  );
  const comfortable = usable.filter((m) => m.fit === "good");
  const pool = comfortable.length > 0 ? comfortable : usable;
  return [...pool].sort((a, b) => b.peak_gib - a.peak_gib)[0];
}

/**
 * How long a run will take, in seconds.
 *
 * Prefers what this machine actually measured for this model. Until a run has
 * completed there is nothing to go on, so it falls back to a figure scaled
 * from resident size — calibrated on an M4, where FLUX.2 Klein 4B q4 took
 * ~38s for 4 steps at 512x512.
 */
export function estimateSeconds(
  model: ModelStatus | undefined,
  steps: number,
  px: number,
  count = 1,
  /** Whether the model is already in memory. A cold run pays the load once. */
  resident = false
): { seconds: number; measured: boolean; loadSeconds: number } {
  if (!model) return { seconds: 0, measured: false, loadSeconds: 0 };
  const mpx = Math.max(px / 1_000_000, 0.01);

  // Loading is a fixed cost paid once per session, not per step or per pixel,
  // so it is added on rather than folded into the rate.
  const loadMs = resident
    ? 0
    : model.measured_load_ms > 0
      ? model.measured_load_ms
      : model.peak_gib * 2200;

  const measured = !!model.measured_ms_per_step_mpx && model.measured_runs > 0;

  // Measured on an M4 with FLUX.2 Klein 4B q4 (5.6 GiB peak): 106s for 4 steps
  // at 1024x1024 is ~25,300 ms per step per megapixel, checked against three
  // other shapes within ~10%. Scaled by resident size for other models.
  const rate = measured
    ? (model.measured_ms_per_step_mpx as number)
    : 25300 * (model.peak_gib / 5.6);

  const ms = rate * steps * mpx * count + loadMs;
  return {
    seconds: Math.max(1, Math.round(ms / 1000)),
    measured,
    loadSeconds: Math.round(loadMs / 1000),
  };
}

/** "about 40 seconds", "about 3 minutes" — no false precision. */
export function humanDuration(seconds: number): string {
  if (seconds < 15) return "a few seconds";
  if (seconds < 90) return `about ${Math.round(seconds / 5) * 5} seconds`;
  const mins = seconds / 60;
  if (mins < 10) return `about ${Math.round(mins * 2) / 2} minutes`;
  return `about ${Math.round(mins)} minutes`;
}
