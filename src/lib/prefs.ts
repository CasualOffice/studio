/**
 * Small, forgiving preference store.
 *
 * Every setting reset on each launch, which made the app feel like it forgot
 * you between sessions. Nothing here is sensitive — no vault content, no keys,
 * only interface choices — so localStorage is the right home for it.
 */
const PREFIX = "modelstudio:";

export function loadPref<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    if (raw === null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    // Corrupt or unavailable storage must never stop the app from starting.
    return fallback;
  }
}

export function savePref(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // Full or disabled storage is not worth interrupting the user for.
  }
}
