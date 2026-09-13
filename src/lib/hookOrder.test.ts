import { describe, it, expect } from "vitest";

/**
 * Hooks must not run after an early return.
 *
 * React counts hooks per render. A component that returns early while its data
 * is still loading runs fewer hooks on that pass than on the next, and React
 * answers by tearing the tree down: "Rendered more hooks than during the
 * previous render" — minified to error #310, which names no component and
 * reads to the user as the app failing to open.
 *
 * That shipped. `vaultCount` was added to App as a useMemo below four early
 * returns, so every launch crashed the moment the vault finished unlocking.
 * Nothing caught it — it type-checks, and no test renders the shell. This one
 * reads the source instead, which is crude but is the check that would have
 * stopped it.
 *
 * Sources come through Vite rather than node:fs so the suite needs no node
 * types, which the app's tsconfig does not carry.
 */
const SOURCES = import.meta.glob("../**/*.tsx", {
  query: "?raw", import: "default", eager: true,
}) as Record<string, string>;

const HOOK =
  /^ {2}(?:const .*=\s*)?(?:useState|useEffect|useMemo|useCallback|useRef|useReducer|useLayoutEffect)\s*\(|^ {2}const \[[^\]]*\]\s*=\s*use[A-Z]/;
const GUARD = /^ {2}if \(.*\) \{\s*$/;
const RETURN = /^ {2,4}return[\s(;]/;
const COMPONENT = /^(?:export\s+)?(?:default\s+)?function\s+[A-Z]/;

describe("hook order", () => {
  const paths = Object.keys(SOURCES).sort();

  it("finds the components to check", () => {
    expect(paths.length).toBeGreaterThan(8);
  });

  for (const path of paths) {
    it(`${path} runs every hook before any early return`, () => {
      const lines = SOURCES[path].split("\n");
      // Scoped per component: one file can hold several, and a return in the
      // first says nothing about a hook in the second.
      const starts = lines
        .map((l, i) => (COMPONENT.test(l) ? i : -1))
        .filter((i) => i >= 0);
      const bounds = [...starts, lines.length];
      const bad: string[] = [];

      starts.forEach((from, c) => {
        const to = bounds[c + 1];
        const name = lines[from].match(/function\s+(\w+)/)?.[1] ?? "?";
        let firstReturn = Infinity;
        let inGuard = false;
        for (let i = from; i < to; i++) {
          if (GUARD.test(lines[i])) inGuard = true;
          else if (/^ {2}\}/.test(lines[i])) inGuard = false;
          if (inGuard && RETURN.test(lines[i]) && i < firstReturn) firstReturn = i;
        }
        for (let i = from; i < to; i++) {
          if (HOOK.test(lines[i]) && i > firstReturn) {
            bad.push(
              `${name}: hook on line ${i + 1}, after the early return on line ` +
              `${firstReturn + 1}`
            );
          }
        }
      });

      expect(bad).toEqual([]);
    });
  }
});
