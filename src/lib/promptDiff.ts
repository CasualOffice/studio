/**
 * Reading a suggested prompt against the words that were typed.
 *
 * A rewrite is judged by what it added and what it lost, and both used to be
 * found by eye: the old card showed the rewrite as a block of text with the
 * first fourteen added words underneath, and nothing said which of your own
 * words were gone. These helpers mark both, and rebuild a prompt with any of
 * the writer's additions left out.
 */

/** One word of a variant as shown, and whether the request said it. */
export interface DiffToken {
  text: string;
  added: boolean;
}

// Words whose presence or absence says nothing about the picture. Losing "the"
// is not a loss worth flagging; losing "no" is, so it is not in here.
const FILLER = new Set([
  "a", "an", "the", "of", "and", "or", "to", "in", "on", "at", "with", "for",
  "is", "are", "be", "it", "its", "this", "that", "my", "me", "i", "some",
  "very", "please", "make", "into", "as", "by", "from", "but", "look", "like",
  "just", "picture", "image", "photo",
]);

const NUMBERS = ["two", "three", "four", "five", "six", "seven", "eight", "nine",
  "ten", "eleven", "twelve"];

const key = (word: string) => word.toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");

/**
 * The shapes an inflected word could have come from, so "cars" meets "car"
 * and "running" meets "run". The same rule as the engine's `_forms`.
 */
export function forms(word: string): Set<string> {
  const w = key(word);
  const out = new Set([w]);
  // "2" and "two" are the same count.
  const n = NUMBERS.indexOf(w);
  if (n >= 0) out.add(String(n + 2));
  if (/^\d+$/.test(w) && NUMBERS[+w - 2]) out.add(NUMBERS[+w - 2]);
  if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) out.add(w.slice(0, -1));
  if (w.length > 4 && w.endsWith("es")) out.add(w.slice(0, -2));
  if (w.length > 4 && w.endsWith("ies")) out.add(w.slice(0, -3) + "y");
  for (const suffix of ["ing", "ed", "ly"]) {
    if (w.length > suffix.length + 2 && w.endsWith(suffix)) {
      const base = w.slice(0, -suffix.length);
      out.add(base); out.add(base + "e");
      if (base.length > 2 && base[base.length - 1] === base[base.length - 2]) out.add(base.slice(0, -1));
    }
  }
  return out;
}

function stems(text: string): Set<string> {
  const all = new Set<string>();
  for (const w of text.split(/\s+/)) for (const f of forms(w)) if (f) all.add(f);
  return all;
}

const shares = (word: string, pool: Set<string>) =>
  [...forms(word)].some((f) => pool.has(f));

/**
 * The variant word by word, each marked added or not.
 *
 * By vocabulary rather than by position: a rewrite reorders as a matter of
 * course, and a positional diff would mark "a red car" as entirely new when it
 * appears after the setting instead of before it.
 */
export function diffWords(original: string, variant: string): DiffToken[] {
  const had = stems(original);
  return variant.split(/(\s+)/).filter((t) => t !== "").map((t) => ({
    text: t,
    added: /\S/.test(t) && key(t) !== "" && !FILLER.has(key(t)) && !shares(t, had),
  }));
}

/**
 * Words of the request that the variant no longer says.
 *
 * `ignore` is what the cleaner removed on purpose -- quality folklore -- which
 * is reported once, as removed, and not a second time as lost.
 */
export function lostWords(original: string, variant: string, ignore: string[] = []): string[] {
  const kept = stems(variant);
  const skip = new Set(ignore.flatMap((p) => p.split(/\s+/)).map(key));
  const out: string[] = [];
  for (const w of original.split(/\s+/)) {
    const k = key(w);
    if (!k || FILLER.has(k) || skip.has(k) || out.includes(k)) continue;
    if (!shares(w, kept)) out.push(k);
  }
  return out;
}

/** Tidy the seams a removed phrase leaves behind. */
function tidy(text: string, capitalise: boolean): string {
  let s = text
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/([,;:])(\s*[,;:])+/g, "$1")
    .replace(/[,;:]\s*([.!?])/g, "$1")
    .replace(/([.!?])(\s*\.)+/g, "$1")
    .replace(/\(\s*\)/g, "")
    .replace(/\s{2,}/g, " ")
    .replace(/^[\s,;:.]+/, "")
    .trim()
    // An article left before a word it no longer fits: "an old chair" with
    // "old" left out.
    .replace(/\b([Aa])n (?=[bcdfghjklmnpqrstvwxyz])/g, "$1 ")
    .replace(/\b([Aa]) (?=[aio]|e(?!u))(?!one\b)/g, "$1n ");
  s = s.replace(/([.!?] )([a-z])/g, (_m, p: string, c: string) => p + c.toUpperCase());
  if (capitalise && s) s = s[0].toUpperCase() + s.slice(1);
  return s;
}

/**
 * The prompt with the chosen additions left out.
 *
 * Each addition is a phrase the engine checked is in the prompt verbatim, so
 * removing it is a string operation and cannot touch the request's own words
 * unless the phrase repeats them -- in which case the first occurrence goes,
 * which is the one the writer added.
 */
export function rebuild(text: string, leaveOut: string[]): string {
  if (!leaveOut.length) return text;
  let out = text;
  for (const phrase of leaveOut) {
    const at = out.toLowerCase().indexOf(phrase.toLowerCase());
    if (at < 0 || !phrase.trim()) continue;
    out = out.slice(0, at) + out.slice(at + phrase.length);
  }
  return tidy(out, /^[A-Z]/.test(text));
}
