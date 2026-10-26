/** @vitest-environment jsdom */
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import Audiobook, { LISTEN, replaceOutput, sentenceAt, titleFor } from "./Audiobook";
import { api } from "../lib/api";
import { KOKORO_MODEL_ID, SPEECH_MODEL_ID, type StoryChapter, type StoryLibrary } from "../lib/story";
import { NowPlaying } from "../lib/nowPlaying";
import type { ModelStatus, VaultItem } from "../lib/types";

vi.mock("../lib/api", () => ({
  api: { audioVoices: vi.fn(), generateAudiobook: vi.fn(), cancelJob: vi.fn(), analyzeStory: vi.fn(), resumeStory: vi.fn(),
         audioPreview: vi.fn(), vaultList: vi.fn(async () => []), narrationScript: vi.fn(), revoiceAudiobook: vi.fn() },
  errText: (e: unknown) => String(e), isCancelled: () => false,
  newJobId: () => "job", onEngineProgress: async () => () => {}, onAudioChunk: async () => () => {},
  vaultUrl: (id: string) => `vault://localhost/${id}`,
}));

const analysed: StoryChapter = { id: "chapter", title: "The train", source: "Mira said, “Wait.”", source_hash: "hash", analysis: "voice",
  scenes: [{ id: "s", title: "Passage 1", summary: "", start: 0, end: 18, beats: [{ id: "b", start: 0, end: 18, source: "Mira said, “Wait.”",
    importance: 2, tension: 2, action: "", setting: "", mood: "", visible: [],
    dialogue: [{ speaker_id: "mira", text: "“Wait.”", offscreen: false, start: 10, end: 18 }] }] }] };
const withBook: StoryLibrary = { version: 1, activeBookId: "book", audio: { book: { narrator: "", voices: {}, rate: 160, outputs: {} } },
  books: [{ version: 1, id: "book", characters: [{ id: "mira", name: "Mira", aliases: ["Mi"], visual: "red coat", voice: "quiet" }],
    chapters: [analysed] }] };
const empty: StoryLibrary = { version: 1, books: [], activeBookId: null };
const kokoro = { id: KOKORO_MODEL_ID, installed: true, hopeless: false } as ModelStatus;
const qwen = { id: SPEECH_MODEL_ID, installed: true, hopeless: false } as ModelStatus;
const writer = { id: "qwen3-4b-instruct-4bit", installed: true, broken: null, hopeless: false } as unknown as ModelStatus;
const KOKORO_VOICES = [{ id: "hf_alpha", language: "Hindi / Indian English" }, { id: "af_heart", language: "English (US)" },
  { id: "am_adam", language: "English (US)" }];
const MAC_VOICES = [{ id: "Samantha", language: "en_US" }, { id: "Karen", language: "en_AU" }];

const produced = vi.fn();
const play = vi.fn();
const notify = vi.fn();
let latest: StoryLibrary = empty;
function Harness({ models = [], start = empty }: { models?: ModelStatus[]; start?: StoryLibrary }) {
  const [library, setLibrary] = useState(start);
  return <NowPlaying.Provider value={{ track: null, play, stop: () => {} }}>
    <Audiobook library={library} models={models} notify={notify} onProduced={produced}
      onLibraryChange={async (update) => { setLibrary((current) => (latest = update(current))); }} />
  </NowPlaying.Provider>;
}
const paste = (text: string) => fireEvent.change(screen.getByLabelText("Text to read"), { target: { value: text } });
const listenButton = () => screen.getByRole("button", { name: "▶ Listen" }) as HTMLButtonElement;
const item = (over: Partial<VaultItem>): VaultItem => ({ id: "x", kind: "audiobook", name: "x.wav", mime: "audio/wav", bytes: 1,
  model: "Kokoro", prompt: "Hello there", seed: 0, width: null, height: null, steps: null, guidance: null, inputs: [],
  created_at: "now", duration_ms: 0, ...over });

afterEach(cleanup);
beforeEach(() => {
  vi.clearAllMocks();
  latest = empty;
  vi.mocked(api.audioVoices).mockImplementation(async (model) => ({ voices: model === KOKORO_MODEL_ID ? KOKORO_VOICES
    : model === SPEECH_MODEL_ID ? [{ id: "ryan", language: "multilingual" }] : MAC_VOICES }));
  vi.mocked(api.generateAudiobook).mockResolvedValue({ id: "sealed-audio", stopped: false, error: null, seconds: 4 });
});

describe("Listen", () => {
  it("reads pasted text at once, with no analysis, no book and no story model", async () => {
    render(<Harness models={[kokoro]} />);
    paste("The rain had stopped. She opened the door.");
    await waitFor(() => expect(listenButton().disabled).toBe(false));
    fireEvent.click(listenButton());
    await waitFor(() => expect(api.generateAudiobook).toHaveBeenCalledWith("job", null, null, "af_heart", {}, 160,
      expect.objectContaining({ model_id: KOKORO_MODEL_ID, source: "The rain had stopped. She opened the door.", single: true })));
    expect(api.analyzeStory).not.toHaveBeenCalled();
    // No Web Audio here, so the saved recording plays once it is ready.
    await waitFor(() => expect(play).toHaveBeenCalledWith(expect.objectContaining({ src: "vault://localhost/sealed-audio" })));
    await waitFor(() => expect(latest.audio?.[LISTEN]?.outputs[LISTEN]).toEqual(["sealed-audio"]));
    expect(produced).toHaveBeenCalledOnce();
    expect(await screen.findByRole("button", { name: "▶ Play" })).toBeTruthy();
  });

  it("chooses a Hindi voice for Hinglish and Devanagari, an English one otherwise", async () => {
    render(<Harness models={[kokoro, qwen]} />);
    paste("Yaar, main kal office gaya tha. Boss ne bola ki meeting cancel ho gayi.");
    await waitFor(() => expect(listenButton().disabled).toBe(false));
    expect(screen.getByText(/Hindi detected/)).toBeTruthy();
    // Qwen3-TTS has no Hindi, so it is not offered for this text.
    expect((screen.getByRole("option", { name: /Qwen3-TTS/ }) as HTMLOptionElement).disabled).toBe(true);
    fireEvent.click(listenButton());
    await waitFor(() => expect(api.generateAudiobook).toHaveBeenCalledWith("job", null, null, "hf_alpha", {}, 160, expect.anything()));
    vi.mocked(api.generateAudiobook).mockClear();
    paste("मैं कल ऑफिस गया था।");
    await waitFor(() => expect(listenButton().disabled).toBe(false));
    fireEvent.click(listenButton());
    await waitFor(() => expect(api.generateAudiobook).toHaveBeenCalledWith("job", null, null, "hf_alpha", {}, 160, expect.anything()));
  });

  it("falls back to macOS voices when Kokoro is not downloaded, so Listen never waits on a download", async () => {
    render(<Harness />);
    paste("Hello there.");
    await waitFor(() => expect(listenButton().disabled).toBe(false));
    fireEvent.click(listenButton());
    await waitFor(() => expect(api.generateAudiobook).toHaveBeenCalledWith("job", null, null, "Samantha", {}, 160,
      expect.objectContaining({ model_id: "macos" })));
  });

  it("keeps a stopped narration and says how much was saved", async () => {
    vi.mocked(api.generateAudiobook).mockResolvedValue({ id: "partial", stopped: true, error: null, seconds: 75 });
    render(<Harness models={[kokoro]} />);
    paste("A long story.");
    await waitFor(() => expect(listenButton().disabled).toBe(false));
    fireEvent.click(listenButton());
    await waitFor(() => expect(notify).toHaveBeenCalledWith(expect.stringMatching(/Stopped\. The 1:15 read so far is saved/), false));
    expect(latest.audio?.[LISTEN]?.outputs[LISTEN]).toEqual(["partial"]);
  });

  it("reports a failure with nothing read and keeps the text", async () => {
    vi.mocked(api.generateAudiobook).mockRejectedValue(new Error("Voice unavailable"));
    render(<Harness models={[kokoro]} />);
    paste("Exact original prose.");
    await waitFor(() => expect(listenButton().disabled).toBe(false));
    fireEvent.click(listenButton());
    await waitFor(() => expect(notify).toHaveBeenCalledWith("Error: Voice unavailable", true));
    expect((screen.getByLabelText("Text to read") as HTMLTextAreaElement).value).toBe("Exact original prose.");
  });

  it("does not enable Listen when no local speech voices are available", async () => {
    vi.mocked(api.audioVoices).mockResolvedValue({ voices: [] });
    render(<Harness />);
    paste("Hello.");
    await waitFor(() => expect(screen.getByText(/Installed macOS speech voices are required/)).toBeTruthy());
    expect(listenButton().disabled).toBe(true);
  });

  it("offers language and direction only for Qwen3-TTS", async () => {
    render(<Harness models={[kokoro, qwen]} />);
    paste("The train waits.");
    await waitFor(() => expect(listenButton().disabled).toBe(false));
    expect(screen.queryByLabelText("Narration direction")).toBeNull();
    fireEvent.change(screen.getByLabelText("Speech engine"), { target: { value: SPEECH_MODEL_ID } });
    await waitFor(() => expect(api.audioVoices).toHaveBeenCalledWith(SPEECH_MODEL_ID));
    fireEvent.change(screen.getByLabelText("Narration direction"), { target: { value: "Warm" } });
    fireEvent.change(screen.getByLabelText("Language"), { target: { value: "english" } });
    await waitFor(() => expect(listenButton().disabled).toBe(false));
    fireEvent.click(listenButton());
    await waitFor(() => expect(api.generateAudiobook).toHaveBeenCalledWith("job", null, null, "ryan", {}, 160,
      expect.objectContaining({ model_id: SPEECH_MODEL_ID, language: "english", delivery: "Warm" })));
  });

  it("keeps every typed letter of a setting", async () => {
    render(<Harness models={[kokoro, qwen]} />);
    paste("The train waits.");
    fireEvent.change(screen.getByLabelText("Speech engine"), { target: { value: SPEECH_MODEL_ID } });
    const direction = await screen.findByLabelText("Narration direction") as HTMLTextAreaElement;
    for (const text of ["W", "Wa", "War", "Warm"]) fireEvent.change(direction, { target: { value: text } });
    expect(direction.value).toBe("Warm");
  });
});

describe("Character voices", () => {
  it("are an optional upgrade: casting runs the quick voice-only analysis on the pasted text", async () => {
    const book = { ...withBook.books[0] };
    vi.mocked(api.analyzeStory).mockResolvedValue({ book, chapter_id: "chapter", panels: [] });
    render(<Harness models={[kokoro, writer]} />);
    paste("Mira said, “Wait.”");
    fireEvent.click(screen.getByRole("button", { name: "Cast character voices" }));
    await waitFor(() => expect(api.analyzeStory).toHaveBeenCalledWith("job", "Mira said, “Wait.”", expect.any(String), null, null, "voice"));
    expect(await screen.findByLabelText("Mira voice")).toBeTruthy();
  });

  it("reads an opened chapter with each character's voice", async () => {
    render(<Harness models={[kokoro]} start={withBook} />);
    fireEvent.click(screen.getByRole("button", { name: "Open The train" }));
    expect((screen.getByLabelText("Text to read") as HTMLTextAreaElement).value).toBe("Mira said, “Wait.”");
    await waitFor(() => expect(screen.getAllByRole("option", { name: /Adam/ }).length).toBeGreaterThan(1));
    fireEvent.change(screen.getByLabelText("Mira voice"), { target: { value: "am_adam" } });
    fireEvent.click(screen.getByRole("button", { name: "Listen with character voices" }));
    await waitFor(() => expect(api.generateAudiobook).toHaveBeenCalledWith("job", withBook.books[0], "chapter", "af_heart",
      { mira: "am_adam" }, 160, expect.objectContaining({ model_id: KOKORO_MODEL_ID })));
    // Filed under the chapter, not with the pasted listens.
    await waitFor(() => expect(latest.audio?.book.outputs.chapter).toEqual(["sealed-audio"]));
  });

  it("plays a sample of a character's chosen voice", async () => {
    const sample = vi.fn(async () => {});
    vi.stubGlobal("Audio", class { play = sample; pause = vi.fn(); onended = null; });
    URL.createObjectURL = vi.fn(() => "blob:sample");
    vi.mocked(api.audioPreview).mockResolvedValue({ wav_base64: btoa("RIFF") });
    render(<Harness models={[kokoro]} start={withBook} />);
    fireEvent.click(screen.getByRole("button", { name: "Open The train" }));
    await waitFor(() => expect(screen.getAllByRole("option", { name: /Adam/ }).length).toBeGreaterThan(1));
    fireEvent.change(screen.getByLabelText("Mira voice"), { target: { value: "am_adam" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview Mira voice" }));
    await waitFor(() => expect(sample).toHaveBeenCalled());
    expect(api.audioPreview).toHaveBeenCalledWith(expect.objectContaining({ voice: "am_adam", text: "“Wait.”" }));
    vi.unstubAllGlobals();
  });
});

describe("Transcript", () => {
  const recorded: StoryLibrary = { ...empty, audio: { [LISTEN]: { narrator: "", voices: {}, rate: 160, outputs: { [LISTEN]: ["rec"] } } } };
  const spans = [{ start: 0, end: 24000, text: "The rain had stopped.", voice: "af_heart" },
                 { start: 26880, end: 48000, text: "She opened the door.", voice: "af_heart" }];
  beforeEach(() => {
    vi.mocked(api.vaultList).mockResolvedValue([item({ id: "rec", spans, markers: [{ time: 0, label: "The rain" }] })]);
  });

  it("plays from the sentence clicked", async () => {
    render(<Harness models={[kokoro]} start={recorded} />);
    fireEvent.click(await screen.findByRole("button", { name: "She opened the door." }));
    expect(play).toHaveBeenCalledWith(expect.objectContaining({ src: "vault://localhost/rec", start: 1.12 }));
  });

  it("reads one sentence again, with new words and another voice, and replaces the recording", async () => {
    vi.mocked(api.revoiceAudiobook).mockResolvedValue("rec2");
    render(<Harness models={[kokoro]} start={recorded} />);
    await screen.findByRole("button", { name: "She opened the door." });
    await waitFor(() => expect(screen.getAllByRole("option", { name: /Adam/ }).length).toBeGreaterThan(0));
    vi.mocked(api.vaultList).mockResolvedValue([item({ id: "rec", spans }), item({ id: "rec2", spans })]);
    fireEvent.click(screen.getByRole("button", { name: "Re-read sentence 1" }));
    await waitFor(() => expect(api.revoiceAudiobook).toHaveBeenCalledWith("job", expect.objectContaining({
      item_id: "rec", index: 0, voice: "af_heart", model_id: KOKORO_MODEL_ID })));
    await waitFor(() => expect(latest.audio?.[LISTEN]?.outputs[LISTEN]).toEqual(["rec2"]));
    vi.mocked(api.revoiceAudiobook).mockResolvedValue("rec3");
    fireEvent.click(await screen.findByRole("button", { name: "Edit sentence 2" }));
    fireEvent.change(screen.getByLabelText("Sentence text"), { target: { value: "She opened the old door." } });
    fireEvent.change(screen.getByLabelText("Voice for this sentence"), { target: { value: "am_adam" } });
    fireEvent.click(screen.getByRole("button", { name: "Re-read" }));
    await waitFor(() => expect(api.revoiceAudiobook).toHaveBeenLastCalledWith("job", expect.objectContaining({
      item_id: "rec2", index: 1, text: "She opened the old door.", voice: "am_adam" })));
    // The new recording plays from the sentence that changed.
    await waitFor(() => expect(play).toHaveBeenLastCalledWith(expect.objectContaining({ src: "vault://localhost/rec3", start: 1.12 })));
  });
});

describe("Suggested retellings", () => {
  it("are shown as a diff and change the text only when accepted", async () => {
    vi.mocked(api.narrationScript).mockResolvedValue({ script: "", stopped: false,
      suggestions: [{ index: 0, original: "Arjun: u late", suggestion: "Arjun wrote, you are late" }] });
    render(<Harness models={[kokoro, writer]} />);
    paste("Arjun: u late\n\nThe end.");
    fireEvent.click(screen.getByRole("button", { name: "Suggest rewrites" }));
    await waitFor(() => expect(api.narrationScript).toHaveBeenCalledWith("job", "Arjun: u late\n\nThe end.", false));
    await screen.findByRole("button", { name: "Accept" });
    expect(document.querySelector(".ab-diff del")?.textContent).toBe("Arjun: u ");
    expect(document.querySelector(".ab-diff ins")?.textContent).toBe("Arjun wrote, you are ");
    expect((screen.getByLabelText("Text to read") as HTMLTextAreaElement).value).toBe("Arjun: u late\n\nThe end.");
    fireEvent.click(screen.getByRole("button", { name: "Accept" }));
    expect((screen.getByLabelText("Text to read") as HTMLTextAreaElement).value).toBe("Arjun wrote, you are late\n\nThe end.");
  });
});

describe("helpers", () => {
  it("names pasted text by its first line or words", () => {
    expect(titleFor("# Chapter One\n\nIt began.")).toBe("Chapter One");
    expect(titleFor("The rain had stopped and the street was quiet again.")).toBe("The rain had stopped and the…");
  });
  it("finds the sentence being heard", () => {
    const spans = [{ start: 0, end: 10, text: "a", voice: "" }, { start: 24000, end: 30000, text: "b", voice: "" }];
    expect(sentenceAt(spans, 0.5)).toBe(0);
    expect(sentenceAt(spans, 1.0)).toBe(1);
  });
  it("points every list of recordings at a re-read one", () => {
    const next = replaceOutput({ ...empty, audio: { a: { narrator: "", voices: {}, rate: 160, outputs: { c: ["x", "old"] } } } }, "old", "new");
    expect(next.audio?.a.outputs.c).toEqual(["x", "new"]);
  });
});
