import { describe, expect, it } from "vitest";
import { autoNarrator, detectScript, wordDiff } from "./narration";
import { KOKORO_MODEL_ID, SPEECH_MODEL_ID } from "./story";

describe("script detection", () => {
  it("tells English, Hinglish and Devanagari apart", () => {
    expect(detectScript("The main thing is to tell us what you want.")).toBe("english");
    expect(detectScript("Yaar, main kal office gaya tha, but boss ne bola ki meeting cancel ho gayi.")).toBe("hinglish");
    expect(detectScript("मैं कल ऑफिस गया था, but the boss was late.")).toBe("devanagari");
    expect(detectScript("")).toBe("english");
  });
});

describe("automatic narrator", () => {
  const kokoro = [{ id: "af_heart", language: "English (US)" }, { id: "hf_alpha", language: "Hindi / Indian English" }];
  const mac = [{ id: "Karen", language: "en_AU" }, { id: "Samantha", language: "en_US" }, { id: "Rishi", language: "en_IN" },
               { id: "Lekha", language: "hi_IN" }];
  it("reads Hindi with a Hindi voice and English with an English one", () => {
    expect(autoNarrator(KOKORO_MODEL_ID, kokoro, "hinglish")).toBe("hf_alpha");
    expect(autoNarrator(KOKORO_MODEL_ID, kokoro, "english")).toBe("af_heart");
    expect(autoNarrator("macos", mac, "devanagari")).toBe("Lekha");
    expect(autoNarrator("macos", mac.filter((v) => v.id !== "Lekha"), "hinglish")).toBe("Rishi");
    expect(autoNarrator("macos", mac, "english")).toBe("Samantha");
    expect(autoNarrator(SPEECH_MODEL_ID, [{ id: "ryan", language: "multilingual" }], "english")).toBe("ryan");
    expect(autoNarrator(KOKORO_MODEL_ID, [], "english")).toBe("");
  });
});

describe("word diff", () => {
  it("marks what a retelling removed and added, keeping the rest", () => {
    expect(wordDiff("Arjun: u r late", "Arjun wrote, you are late")).toEqual([
      { kind: "del", text: "Arjun: u r " }, { kind: "ins", text: "Arjun wrote, you are " }, { kind: "same", text: "late" }]);
    expect(wordDiff("same words", "same words")).toEqual([{ kind: "same", text: "same words" }]);
    const parts = wordDiff("She went home.", "She went back home.");
    expect(parts.map((p) => p.kind)).toEqual(["same", "ins", "same"]);
    expect(parts.filter((p) => p.kind !== "del").map((p) => p.text).join("")).toBe("She went back home.");
  });
});
