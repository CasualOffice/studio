import { KOKORO_MODEL_ID, SPEECH_MODEL_ID } from "./story";

/**
 * What a pasted text needs from a voice, decided before anything is read.
 *
 * The first listen must not ask which voice to use: a Hindi voice for Hindi
 * or Hinglish, an English one otherwise. Devanagari is counted directly;
 * Hinglish in the Latin alphabet by how many of its words are the common
 * Hindi function words no English text uses.
 */
export type Script = "english" | "hinglish" | "devanagari";

const HINDI_WORDS = new Set(`hai hain nahi nhi nahin kya main mein mujhe mujhse tum tumhe tumko aap aapko ka ki ke ko ne se
bhi toh yaar kal aaj kuch kuchh bahut bohot accha acha achha theek thik gaya gayi gaye raha rahi rahe tha thi kar karo
karna kiya kyun kyu kaise haan abhi phir fir chalo bolo bola boli socha mera meri mere tera teri tere apna apni wala wali
hum humne maine usne unhone sab koi sirf matlab pata chahiye dekho suno arre bhai ghar dost pyaar pyar dil baat kahan
kab kaun jab hoon hu hun wo woh ye yeh lekin aur magar isliye kyunki zara shayad bilkul`.split(/\s+/));

export function detectScript(text: string): Script {
  const devanagari = (text.match(/[ऀ-ॿ]/g) ?? []).length;
  const latin = (text.match(/[A-Za-z]/g) ?? []).length;
  if (devanagari && devanagari / (devanagari + latin) > 0.2) return "devanagari";
  const words = text.toLowerCase().match(/[a-z]+/g) ?? [];
  const hindi = words.filter((w) => HINDI_WORDS.has(w)).length;
  return words.length && hindi / words.length >= 0.15 ? "hinglish" : "english";
}

/** The voice to read with when none was chosen, for this engine and text. */
export function autoNarrator(engine: string, voices: { id: string; language: string }[], script: Script): string {
  const has = (id: string) => voices.some((v) => v.id === id);
  const hindi = script !== "english";
  if (engine === KOKORO_MODEL_ID) {
    const pick = hindi ? "hf_alpha" : "af_heart";
    return has(pick) ? pick : voices[0]?.id ?? "";
  }
  if (engine === SPEECH_MODEL_ID) return has("ryan") ? "ryan" : voices[0]?.id ?? "";
  const by = (test: (language: string) => boolean) => voices.find((v) => test(v.language))?.id;
  return (hindi ? by((l) => l.startsWith("hi")) ?? by((l) => l === "en_IN") : undefined)
    ?? by((l) => l === "en_US") ?? by((l) => l.startsWith("en")) ?? voices[0]?.id ?? "";
}

export interface DiffPart { kind: "same" | "del" | "ins"; text: string }

/**
 * A word-level diff, for showing a suggested retelling against the author's
 * own words. Whitespace travels with the word before it.
 */
export function wordDiff(before: string, after: string): DiffPart[] {
  const a = before.match(/\S+\s*/g) ?? [];
  const b = after.match(/\S+\s*/g) ?? [];
  const same = (x: string, y: string) => x.trim() === y.trim();
  // Longest common subsequence, filled from the end so the walk runs forward.
  const lcs = Array.from({ length: a.length + 1 }, () => new Uint16Array(b.length + 1));
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      lcs[i][j] = same(a[i], b[j]) ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const parts: DiffPart[] = [];
  const add = (kind: DiffPart["kind"], text: string) => {
    const last = parts.at(-1);
    if (last?.kind === kind) last.text += text; else parts.push({ kind, text });
  };
  let i = 0, j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && same(a[i], b[j])) { add("same", b[j]); i++; j++; }
    // What was removed reads before what replaced it.
    else if (i < a.length && (j >= b.length || lcs[i + 1][j] >= lcs[i][j + 1])) { add("del", a[i]); i++; }
    else { add("ins", b[j]); j++; }
  }
  return parts;
}

/** Seconds from a position in samples at 24 kHz. */
export const sampleSeconds = (samples: number) => samples / 24000;
