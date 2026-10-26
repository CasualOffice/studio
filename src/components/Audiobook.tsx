import { useEffect, useMemo, useRef, useState } from "react";
import { api, errText, isCancelled, newJobId, onAudioChunk, onEngineProgress, vaultUrl, type ScriptSuggestion } from "../lib/api";
import { ANALYST_IDS, KOKORO_MODEL_ID, SPEECH_MODEL_ID, keepBook, unanalysedIn, voiceGroups, voiceLabel, type AudioSettings, type StoryLibrary } from "../lib/story";
import type { AudioSpan, EngineProgress, ModelStatus, VaultItem } from "../lib/types";
import { JobProgress } from "./shared";
import { playhead, useNowPlaying } from "../lib/nowPlaying";
import { LiveQueue, audioClock } from "../lib/liveAudio";
import { autoNarrator, detectScript, sampleSeconds, wordDiff } from "../lib/narration";
import { clock } from "../lib/library";

/**
 * Settings and recordings for listening to pasted text, kept beside the
 * books' own under a key no book id can take.
 */
export const LISTEN = "listen";
const DEFAULTS: AudioSettings = { narrator: "", voices: {}, rate: 160, outputs: {} };
// Pasted listens are filed so they can be played again; older ones stay in
// the Vault but leave this list.
const KEEP_LISTENS = 20;

/** A recording's name for text with no chapter: its first line or words. */
export function titleFor(text: string): string {
  const line = text.trim().split("\n")[0].trim().replace(/^#+\s*/, "");
  if (line && line.length <= 60 && !/[.!?।]$/.test(line)) return line;
  const words = text.trim().split(/\s+/);
  return words.slice(0, 6).join(" ") + (words.length > 6 ? "…" : "");
}

/** Every list of recordings with `old` in it, pointing at `next` instead. */
export function replaceOutput(library: StoryLibrary, old: string, next: string): StoryLibrary {
  const audio = Object.fromEntries(Object.entries(library.audio ?? {}).map(([key, settings]) => [key, {
    ...settings, outputs: Object.fromEntries(Object.entries(settings.outputs ?? {})
      .map(([slot, ids]) => [slot, ids.map((id) => (id === old ? next : id))])),
  }]));
  return { ...library, audio };
}

export default function Audiobook({ library, models, notify, onLibraryChange, onProduced }: {
  library: StoryLibrary | null;
  models: ModelStatus[];
  notify: (message: string, bad?: boolean) => void;
  onLibraryChange: (update: (current: StoryLibrary) => StoryLibrary) => Promise<void>;
  onProduced: () => void;
}) {
  const nowPlaying = useNowPlaying();
  // The text box is the whole first step: paste, press Listen. A saved
  // chapter, its analysis and character voices are all optional.
  const [text, setText] = useState("");
  const [opened, setOpened] = useState<{ book: string; chapter: string } | null>(null);
  const [append, setAppend] = useState(false);
  const [voices, setVoices] = useState<{ id: string; language: string }[]>([]);
  const [voiceError, setVoiceError] = useState("");
  const [job, setJob] = useState<string | null>(null);
  const [progress, setProgress] = useState<EngineProgress | null>(null);
  const [live, setLive] = useState<{ job: string; queue: LiveQueue; id?: string } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [items, setItems] = useState<Record<string, VaultItem>>({});
  const [suggestions, setSuggestions] = useState<ScriptSuggestion[] | null>(null);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  const book = opened ? library?.books.find((b) => b.id === opened.book) : undefined;
  const chapter = book?.chapters.find((c) => c.id === opened?.chapter);
  const analysed = !!chapter && chapter.scenes.length > 0;

  // Edits show at once and are saved shortly after. Bound straight to the
  // library, every keystroke waited on an encrypted write, so typed letters
  // were dropped and the pace slider snapped back.
  const [drafts, setDrafts] = useState<Record<string, Partial<AudioSettings>>>({});
  const stored = (key: string): AudioSettings => ({ ...DEFAULTS, ...library?.audio?.[key], ...drafts[key] });
  // Engine, narrator, pace and delivery are the listener's, whatever is read;
  // character voices and directions belong to a book.
  const settings = stored(LISTEN);
  const cast = book ? stored(book.id) : DEFAULTS;

  const script = detectScript(text);
  const hindiText = script !== "english";
  const installed = (id: string) => models.some((m) => m.id === id && m.installed && !m.hopeless);
  // Kokoro is the default: fast, and with Hindi voices. Without it, the
  // macOS voices are always there, so Listen never waits on a download.
  const fallback = installed(KOKORO_MODEL_ID) ? KOKORO_MODEL_ID : "macos";
  const chosen = settings.model_id || fallback;
  // Qwen3-TTS has no Hindi; Hinglish or Devanagari goes to a voice that does.
  const speechModel = chosen === SPEECH_MODEL_ID && hindiText ? fallback : chosen;
  const neural = speechModel === SPEECH_MODEL_ID;
  const speechReady = speechModel === "macos" || installed(speechModel);
  useEffect(() => {
    let live = true;
    setVoices([]); setVoiceError("");
    void api.audioVoices(speechModel).then((r) => { if (live) setVoices(r.voices); })
      .catch((e) => { if (live) setVoiceError(errText(e)); });
    return () => { live = false; };
  }, [speechModel]);
  const automatic = autoNarrator(speechModel, voices, script);
  const narrator = voices.some((v) => v.id === settings.narrator) ? settings.narrator : automatic;
  const assignedVoices = Object.fromEntries(Object.entries(cast.voices).filter(([, voice]) => voices.some((v) => v.id === voice)));
  const voiceOptions = voiceGroups(voices).map(([label, list]) => <optgroup key={label} label={label}>
    {list.map((v) => <option key={v.id} value={v.id}>{voiceLabel(v.id)}</option>)}</optgroup>);
  const writerReady = models.some((m) => ANALYST_IDS.includes(m.id) && m.installed && !m.hopeless);

  // Recordings are filed under the open chapter, or with the pasted listens.
  const outputs = (chapter ? cast.outputs[chapter.id] : settings.outputs[LISTEN]) ?? [];
  const current = selected && outputs.includes(selected) ? selected : outputs.at(-1) ?? null;
  const outputKey = outputs.join(",");
  const loadItems = () => api.vaultList().then((list) => {
    if (mounted.current) setItems(Object.fromEntries(list.filter((i) => i.kind === "audiobook").map((i) => [i.id, i])));
  }).catch(() => { /* the player still works without markers and spans */ });
  useEffect(() => { if (outputKey) void loadItems(); }, [outputKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const pending = useRef<{ key: string; patch: Partial<AudioSettings>; timer?: number }>({ key: "", patch: {} });
  const flush = () => {
    const { key, patch, timer } = pending.current;
    window.clearTimeout(timer);
    if (!key || !Object.keys(patch).length) return;
    pending.current = { key, patch: {} };
    void onLibraryChange((current) => ({ ...current, audio: { ...current.audio,
      [key]: { ...DEFAULTS, ...current.audio?.[key], ...patch } } }))
      .catch((e) => notify(`Could not save voice settings: ${errText(e)}`, true));
  };
  useEffect(() => flush, []); // eslint-disable-line react-hooks/exhaustive-deps
  const save = (key: string, patch: Partial<AudioSettings> | ((latest: AudioSettings) => Partial<AudioSettings>)) => {
    const resolved = typeof patch === "function" ? patch(stored(key)) : patch;
    if (pending.current.key !== key) flush();
    pending.current = { key, patch: { ...pending.current.patch, ...resolved } };
    setDrafts((d) => ({ ...d, [key]: { ...d[key], ...resolved } }));
    window.clearTimeout(pending.current.timer);
    pending.current.timer = window.setTimeout(flush, 400);
  };
  const saveSettings = (patch: Partial<AudioSettings>) => save(LISTEN, patch);
  const saveCast = (patch: Partial<AudioSettings> | ((latest: AudioSettings) => Partial<AudioSettings>)) => {
    if (book) save(book.id, patch);
  };
  // Each engine keeps its own narrator and cast; switching back restores them.
  const switchEngine = (next: string) => {
    saveSettings({ model_id: next, narrator: settings.casts?.[next]?.narrator ?? "",
      casts: { ...settings.casts, [speechModel]: { narrator: settings.narrator, voices: {} } } });
    if (book) saveCast({ voices: cast.casts?.[next]?.voices ?? {},
      casts: { ...cast.casts, [speechModel]: { narrator: "", voices: cast.voices } } });
  };

  // Hear a voice before choosing it. The sample is the character's own first
  // line when there is one, so the choice is made on words from this story.
  const [previewing, setPreviewing] = useState<string | null>(null);
  const [sounding, setSounding] = useState<string | null>(null);
  const player = useRef<HTMLAudioElement | null>(null);
  const stopPreview = () => { player.current?.pause(); setSounding(null); };
  useEffect(() => () => { player.current?.pause(); }, []);
  const sampleFor = (characterId: string | null) => {
    const lines = chapter?.scenes.flatMap((s) => s.beats.flatMap((b) => b.dialogue)) ?? [];
    const line = characterId ? lines.find((d) => d.speaker_id === characterId)?.text : null;
    if (line) return line;
    const opening = text.trim().slice(0, 300);
    return characterId ? "Hello. This is how I will sound in your story."
      : opening.slice(0, Math.max(opening.lastIndexOf(". ") + 1, 120)) || "This is how I will read your story.";
  };
  const preview = async (key: string, voice: string, characterId: string | null) => {
    setPreviewing(key);
    try {
      const { wav_base64 } = await api.audioPreview({ model_id: speechModel, voice, text: sampleFor(characterId),
        language: settings.language || "auto", delivery: settings.delivery || "",
        direction: characterId ? (cast.directions?.[characterId]
          ?? book?.characters.find((c) => c.id === characterId)?.voice ?? "") : "",
        rate: settings.rate });
      const bytes = Uint8Array.from(atob(wav_base64), (c) => c.charCodeAt(0));
      const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
      player.current?.pause();
      const audio = new Audio(url);
      audio.onended = () => { URL.revokeObjectURL(url); setSounding(null); };
      setSounding(key);
      player.current = audio;
      await audio.play();
    } catch (e) {
      notify(`Could not preview ${voice}: ${errText(e)}`, true);
    } finally {
      if (mounted.current) setPreviewing(null);
    }
  };
  // One button per voice: play a sample, and stop it while it plays.
  const PreviewButton = ({ id, voice, character, label }: { id: string; voice: string; character: string | null; label: string }) =>
    sounding === id
      ? <button type="button" className="btn small" aria-label={`Stop ${label.replace(/^Preview /, "")} preview`}
          onClick={stopPreview}>■</button>
      : <button type="button" className="btn small" aria-label={label} title={`Hear ${voiceLabel(voice)}`}
          aria-busy={previewing === id} disabled={!!job || !!previewing || !voice || !speechReady}
          onClick={() => { stopPreview(); void preview(id, voice, character); }}>{previewing === id ? "…" : "▶"}</button>;

  const perform = async (action: (id: string) => Promise<void>, id = newJobId()) => {
    setJob(id); setProgress(null);
    let un: (() => void) | undefined;
    try {
      un = await onEngineProgress((p) => { if (p.job_id === id && mounted.current) setProgress(p); });
      await action(id);
    } catch (e) {
      notify(isCancelled(e) ? "Stopped before anything was read." : errText(e), !isCancelled(e));
    } finally {
      un?.();
      if (mounted.current) { setJob(null); setProgress(null); }
    }
  };

  const fileRecording = async (id: string) => {
    const [key, slot] = chapter && book ? [book.id, chapter.id] : [LISTEN, LISTEN];
    await onLibraryChange((current) => {
      const latest = { ...DEFAULTS, ...current.audio?.[key] };
      const list = [...(latest.outputs[slot] ?? []), id];
      return { ...current, audio: { ...current.audio, [key]: { ...latest,
        outputs: { ...latest.outputs, [slot]: key === LISTEN ? list.slice(-KEEP_LISTENS) : list } } } };
    });
  };

  /**
   * Read aloud, hearing each sentence as soon as it is read. With `withCast`
   * the analysed chapter is read with its character voices; otherwise the
   * text box is read by one storyteller.
   */
  const listen = (withCast: boolean) => {
    const title = chapter?.title ?? titleFor(text);
    // Made inside the click: WebKit only lets a page start audio from one.
    const audio = audioClock();
    void audio?.resume().catch(() => {});
    const id = newJobId();
    const queue = audio ? new LiveQueue(audio, title) : null;
    void perform(async () => {
      setLive(queue ? { job: id, queue } : null);
      const unlisten = await onAudioChunk((chunk) => { if (chunk.jobId === id) queue?.push(chunk); });
      if (queue) nowPlaying.play({ src: `live:${id}`, title, live: queue });
      let outcome;
      try {
        const options = { model_id: speechModel, language: neural ? settings.language || "auto" : undefined,
          delivery: neural ? settings.delivery || "" : undefined };
        outcome = withCast && book && chapter
          ? await api.generateAudiobook(id, book, chapter.id, narrator, assignedVoices, settings.rate,
              { ...options, directions: neural ? cast.directions || {} : {} })
          : await api.generateAudiobook(id, chapter ? book ?? null : null, chapter?.id ?? null, narrator, {}, settings.rate,
              { ...options, source: text, single: true, ...(chapter ? {} : { title }) });
      } finally {
        unlisten();
        queue?.finish();
        // Nothing was read: no empty player left behind.
        if (queue && !queue.duration) nowPlaying.stop();
      }
      onProduced();
      if (mounted.current) {
        setLive((l) => (l?.job === id ? { ...l, id: outcome.id } : l));
        setSelected(outcome.id);
      }
      if (!queue) nowPlaying.play({ src: vaultUrl(outcome.id), title });
      try {
        await fileRecording(outcome.id);
      } catch (error) {
        notify(`The recording is saved in the Vault, but could not be listed here: ${errText(error)}`, true);
        return;
      }
      void loadItems();
      notify(outcome.error ? `The voice stopped partway (${outcome.error}); the ${clock(outcome.seconds)} read so far is saved in the Vault.`
        : outcome.stopped ? `Stopped. The ${clock(outcome.seconds)} read so far is saved in the Vault.`
        : "Saved in the encrypted Vault. Click any sentence to hear it again.", !!outcome.error);
    }, id);
  };

  // Character voices: the optional upgrade after a first listen.
  const castVoices = () => perform(async (id) => {
    // Speakers and mood only: the audiobook needs nothing about shots, outfits
    // or rooms, and describing them for every beat made this take hours.
    const result = await api.analyzeStory(id, text, titleFor(text), append ? book ?? null : null, null, "voice");
    await onLibraryChange((current) => keepBook(current, result.book));
    setOpened({ book: result.book.id, chapter: result.chapter_id }); setAppend(false);
    notify("Voices cast and saved. Choose a voice for each character, then listen with character voices.");
  });
  // Stop keeps the passages already cast; this casts only the rest.
  const resume = () => perform(async (id) => {
    if (!book || !chapter) return;
    const result = await api.resumeStory(id, book, chapter.id);
    await onLibraryChange((current) => keepBook(current, result.book));
    notify(result.unanalysed ? `${result.unanalysed} passages still to cast; the rest is saved.` : "Every passage is cast.");
  });
  // Going back: the chapter analysed again in place, so the book, its
  // characters and the recordings filed under the chapter all stay.
  const analyseAgain = () => perform(async (id) => {
    if (!book || !chapter) return;
    const result = await api.resumeStory(id, book, chapter.id, null, "manga", true);
    await onLibraryChange((current) => keepBook(current, result.book));
    notify("Analysed again. Speakers and moods are updated; recordings are kept.");
  });
  // The story model's retellings, offered paragraph by paragraph as a diff.
  // Never needed: the text is already read by rule (chat lines, shorthand,
  // emoji, Hinglish spelled for Hindi voices).
  const suggest = () => perform(async (id) => {
    const result = await api.narrationScript(id, text, false);
    setSuggestions(result.suggestions ?? []);
    notify(result.suggestions?.length ? "Suggestions ready. Accept the ones you like."
      : "The story model has nothing to change.");
  });

  const open = (bookId: string, chapterId: string) => {
    const source = library?.books.find((b) => b.id === bookId)?.chapters.find((c) => c.id === chapterId)?.source ?? "";
    setOpened({ book: bookId, chapter: chapterId }); setText(source); setSuggestions(null); setSelected(null);
  };

  // Read one sentence again, as it was or edited, and splice it in.
  const reread = (index: number, words?: string, voice?: string) => {
    const item = current ? items[current] : undefined;
    const span = item?.spans?.[index];
    if (!item || !span) return;
    void perform(async (id) => {
      const next = await api.revoiceAudiobook(id, { item_id: item.id, index, text: words,
        voice: voice ?? (voices.some((v) => v.id === span.voice) ? span.voice : narrator),
        model_id: speechModel, rate: settings.rate,
        language: neural ? settings.language || "auto" : undefined, delivery: neural ? settings.delivery || "" : undefined });
      await onLibraryChange((library) => replaceOutput(library, item.id, next)).catch((e) =>
        notify(`The new reading is in the Vault, but could not be listed here: ${errText(e)}`, true));
      setSelected(next);
      onProduced();
      await loadItems();
      nowPlaying.play({ src: vaultUrl(next), title: item.prompt || "Narration", markers: item.markers ?? [],
        start: sampleSeconds(span.start) });
      notify("Sentence read again; the recording is updated.");
    });
  };

  // What the transcript shows: the narration being read, else the chosen recording.
  // Until the saved recording's sentences load, the ones streamed stand in.
  const reading = live && !live.id ? live : null;
  const saved = (current && items[current]?.spans) || [];
  const streamed = live && (!live.id || live.id === current) && !saved.length ? live : null;
  const spans: AudioSpan[] = streamed ? streamed.queue.spans.filter(Boolean) : saved;
  const liveQueue = live && (!live.id || live.id === current) && nowPlaying.track?.live === live.queue ? live.queue : null;
  const playingFile = !liveQueue && !!current && nowPlaying.track?.src === vaultUrl(current);
  const now = useMemo(() => liveQueue ? () => liveQueue.position() : playingFile ? () => playhead.get().time : null,
    [liveQueue, playingFile]);
  const seekTo = (seconds: number) => {
    if (liveQueue) { liveQueue.seek(seconds); liveQueue.play(); return; }
    if (!current) return;
    if (playingFile && playhead.seek(vaultUrl(current), seconds)) return;
    const item = items[current];
    nowPlaying.play({ src: vaultUrl(current), title: item?.prompt || "Narration", markers: item?.markers ?? [], start: seconds });
  };

  const engineName = speechModel === KOKORO_MODEL_ID ? "Kokoro" : neural ? "Qwen3-TTS" : "macOS voices";
  const narratorName = narrator ? voiceLabel(narrator) : "no voice";
  const listenDisabled = !!job || !text.trim() || !speechReady || !narrator;

  return <div className="audiobook-workspace">
    <div className="pane">
      <h2>Listen</h2>
      <p className="muted">Paste any text and press Listen. It starts reading within seconds, sentence by sentence;
        everything else here is optional.</p>
      {chapter && <p className="t-meta ab-opened">From your saved chapter <strong>{chapter.title}</strong>{" "}
        <button type="button" className="btn small" disabled={!!job}
          onClick={() => { setOpened(null); setText(""); setSuggestions(null); setSelected(null); }}>New text</button></p>}
      <label className="ab-text">Text to read<textarea rows={12} value={text} disabled={!!job}
        onChange={(e) => { setText(e.target.value); setSuggestions(null); }}
        placeholder="Paste a story, a chapter, a chat, an article… English, Hindi or Hinglish." /></label>
      <div className="control-row ab-listen-row">
        <button className="btn primary ab-listen" disabled={listenDisabled} onClick={() => listen(false)}>
          {job && live && !live.id ? "Reading…" : "▶ Listen"}</button>
        {job && <button className="btn" onClick={() => void api.cancelJob(job).catch((e) => notify(errText(e), true))}>Stop</button>}
        <span className="t-meta">{narratorName} · {engineName}{hindiText ? " · Hindi detected" : ""}</span>
      </div>
      {!speechReady && <p className="notice warn">Download {speechModel === KOKORO_MODEL_ID
        ? "Fast audiobook voices (Kokoro 82M)" : "Audiobook voices (Qwen3-TTS 1.7B)"} in Models to use it, or choose macOS voices below.</p>}
      {speechModel === "macos" && !installed(KOKORO_MODEL_ID) && <p className="t-help">
        For natural voices, download Fast audiobook voices (Kokoro 82M) in Models.</p>}
      {!voices.length && <p className="notice warn">{voiceError || (speechModel === "macos"
        ? "Installed macOS speech voices are required for narration. Manage voices in macOS Accessibility → Spoken Content."
        : `${engineName} voices are loading.`)}</p>}
      {progress && <JobProgress p={progress} label="Audiobook" />}

      <details className="ab-drawer">
        <summary>Voice · {narratorName} · {settings.rate === 160 ? "normal speed" : `speed ${settings.rate}`}</summary>
        <label>Speech engine<select value={speechModel} disabled={!!job} onChange={(e) => switchEngine(e.target.value)}>
          <option value={KOKORO_MODEL_ID}>Kokoro · fast, Hindi and Indian English voices</option>
          <option value={SPEECH_MODEL_ID} disabled={hindiText}>Qwen3-TTS · expressive, slower{hindiText ? " (no Hindi)" : ""}</option>
          <option value="macos">macOS system voices · includes Indian voices</option>
        </select></label>
        {chosen === SPEECH_MODEL_ID && hindiText && <p className="t-help">Qwen3-TTS has no Hindi voice, so this text is read with {engineName}.</p>}
        <label>Narrator<span className="control-row"><select value={voices.some((v) => v.id === settings.narrator) ? settings.narrator : ""}
          disabled={!!job || !voices.length} onChange={(e) => saveSettings({ narrator: e.target.value })}>
          <option value="">Automatic · {automatic ? voiceLabel(automatic) : "…"}</option>
          {voiceOptions}
        </select><PreviewButton id="narrator" voice={narrator} character={null} label="Preview narrator voice" /></span></label>
        <label>{neural ? `Reading pace · ${settings.rate} (approximate)` : `Reading speed · ${settings.rate} words/minute`}<input type="range" min={80} max={300}
          value={settings.rate} disabled={!!job} onChange={(e) => saveSettings({ rate: Number(e.target.value) })} /></label>
        {neural && <>
          <label>Language<select value={settings.language || "auto"} disabled={!!job}
            onChange={(e) => saveSettings({ language: e.target.value })}>
            {["auto", "english", "chinese", "japanese", "korean", "german", "french", "russian", "portuguese", "spanish", "italian"].map((l) => <option key={l} value={l}>{l}</option>)}
          </select></label>
          <label>Narration direction<textarea rows={2} maxLength={1500} value={settings.delivery || ""} disabled={!!job}
            placeholder="Warm, intimate storytelling with restrained emotion and natural pauses."
            onChange={(e) => saveSettings({ delivery: e.target.value })} /></label>
        </>}
      </details>

      <details className="ab-drawer" open={analysed || undefined}>
        <summary>Character voices{analysed ? ` · ${book?.characters.length ?? 0} characters` : " · optional"}</summary>
        {!analysed && <>
          <p className="t-help">A quick analysis finds who speaks each line, so each character can have their own voice.
            Board shares the characters it finds.</p>
          {book && <label><input type="checkbox" checked={append} disabled={!!job}
            onChange={(e) => setAppend(e.target.checked)} /> Add to this book and reuse its characters</label>}
          <button className="btn small" disabled={!!job || !text.trim() || !writerReady || !library}
            onClick={() => void castVoices()}>Cast character voices</button>
          {!writerReady && <p className="notice warn">Install a story analyst or the prompt writer in Models to cast character voices.</p>}
        </>}
        {analysed && book && chapter && <>
          {unanalysedIn(book, chapter.id) > 0 && <div className="notice warn">
            {unanalysedIn(book, chapter.id)} passages are not cast yet and would be read by the narrator.
            <button className="btn small" disabled={!!job || !writerReady} onClick={() => void resume()}>Continue analysis</button>
          </div>}
          <CastList book={book} chapter={chapter} cast={cast} voices={voices} narrator={narrator} job={!!job}
            neural={neural} voiceOptions={voiceOptions} saveCast={saveCast} PreviewButton={PreviewButton} />
          <span className="control-row">
            <button className="btn small primary" disabled={listenDisabled} onClick={() => listen(true)}>Listen with character voices</button>
            <button type="button" className="btn small" disabled={!!job || !writerReady}
              onClick={() => void analyseAgain()}>Analyse this chapter again</button>
          </span>
        </>}
      </details>

      <details className="ab-drawer">
        <summary>Smoother read-aloud text · optional</summary>
        <p className="t-help">Chat lines, texting shorthand and emoji are already read sensibly. The story model can
          also suggest retellings of awkward paragraphs, keeping your words; accept only the ones you like.</p>
        <button className="btn small" disabled={!!job || !text.trim() || !writerReady} onClick={() => void suggest()}>Suggest rewrites</button>
        {!writerReady && <p className="t-help">Needs a story analyst or the prompt writer from Models.</p>}
        {suggestions?.map((s) => <div key={`${s.index}-${s.original.slice(0, 20)}`} className="ab-suggestion">
          <p className="ab-diff">{wordDiff(s.original, s.suggestion).map((part, i) =>
            part.kind === "same" ? <span key={i}>{part.text}</span>
              : part.kind === "del" ? <del key={i}>{part.text}</del> : <ins key={i}>{part.text}</ins>)}</p>
          <span className="control-row">
            <button className="btn small primary" disabled={!!job || !text.includes(s.original)}
              onClick={() => { setText((t) => t.replace(s.original, s.suggestion)); setSuggestions((l) => l?.filter((x) => x !== s) ?? null); }}>Accept</button>
            <button className="btn small" onClick={() => setSuggestions((l) => l?.filter((x) => x !== s) ?? null)}>Dismiss</button>
          </span>
        </div>)}
      </details>

      {!!library?.books.length && <details className="ab-drawer">
        <summary>Saved chapters · {library.books.reduce((n, b) => n + b.chapters.length, 0)}</summary>
        {library.books.map((b) => <div key={b.id} className="ab-book">
          <div className="t-meta">{b.chapters[0]?.title ?? "Book"} · {b.chapters.length} chapter{b.chapters.length === 1 ? "" : "s"}</div>
          {b.chapters.map((c) => <button key={c.id} type="button" className="btn small" disabled={!!job}
            aria-pressed={opened?.chapter === c.id} onClick={() => open(b.id, c.id)}>Open {c.title}</button>)}
        </div>)}
      </details>}
    </div>
    <div className="pane">
      <h3>{chapter?.title ?? "Recordings"}</h3>
      {/* Recordings play in the app-wide bar, so a chapter keeps playing
          while you work elsewhere. */}
      {outputs.map((id, i) => {
        const title = items[id]?.prompt || `${chapter?.title ?? "Listen"} · recording ${i + 1}`;
        const playing = nowPlaying.track?.src === vaultUrl(id);
        const markers = items[id]?.markers ?? [];
        return <div className={"audio-result recording" + (id === current ? " on" : "")} key={id}>
          <button type="button" className="ab-recording-name" onClick={() => setSelected(id)} aria-pressed={id === current}>
            {title}{markers.length ? <span className="t-meta"> · {markers.length} sections</span> : null}</button>
          <button type="button" className={"btn small" + (playing ? "" : " primary")} aria-pressed={playing}
            onClick={() => playing ? nowPlaying.stop()
              : nowPlaying.play({ src: vaultUrl(id), title, markers })}>
            {playing ? "■ Stop" : "▶ Play"}</button>
        </div>;
      })}
      {outputs.length > 0 && <p className="t-help">Recordings are in the Vault too, where they can be exported as WAV.</p>}
      {spans.length > 0 && <Transcript key={reading ? `live-${reading.job}` : current ?? ""} spans={spans} now={now}
        onSeek={seekTo} editable={!reading && !job} voices={voices} voiceOptions={voiceOptions} narrator={narrator}
        onReread={reread} />}
      {!spans.length && current && items[current] && <p className="t-help">This recording was made before sentences were tracked;
        play it above.</p>}
      {chapter && analysed && <details><summary>Cast lines · {chapter.scenes.length} passage{chapter.scenes.length === 1 ? "" : "s"}</summary>
        {chapter.scenes.map((scene) => <section key={scene.id} className="notice">
          {scene.beats.flatMap((beat) => beat.dialogue).map((line, i) => <p key={i}>
            <strong>{book?.characters.find((c) => c.id === line.speaker_id)?.name ?? "Unattributed"}{line.offscreen ? " (offscreen)" : ""}:</strong> {line.text}
          </p>)}
        </section>)}
      </details>}
    </div>
  </div>;
}

/** Each character with the voice that reads their lines. */
function CastList({ book, chapter, cast, voices, narrator, job, neural, voiceOptions, saveCast, PreviewButton }: {
  book: NonNullable<StoryLibrary["books"][number]>;
  chapter: NonNullable<StoryLibrary["books"][number]["chapters"][number]>;
  cast: AudioSettings;
  voices: { id: string; language: string }[];
  narrator: string;
  job: boolean;
  neural: boolean;
  voiceOptions: React.ReactNode;
  saveCast: (patch: (latest: AudioSettings) => Partial<AudioSettings>) => void;
  PreviewButton: (props: { id: string; voice: string; character: string | null; label: string }) => React.ReactElement;
}) {
  const linesBy: Record<string, number> = {};
  for (const line of chapter.scenes.flatMap((s) => s.beats.flatMap((b) => b.dialogue))) {
    if (line.speaker_id) linesBy[line.speaker_id] = (linesBy[line.speaker_id] ?? 0) + 1;
  }
  const assigned = (id: string) => voices.some((v) => v.id === cast.voices[id]) ? cast.voices[id] : "";
  // Who speaks most in this chapter comes first, with their line count; those
  // silent here follow, so a long bible does not bury the cast.
  return <>{[...book.characters].sort((a, b) => (linesBy[b.id] ?? 0) - (linesBy[a.id] ?? 0)).map((c) => <label key={c.id}>
    <span>{c.name} <span className="t-meta">· {linesBy[c.id] ? `${linesBy[c.id]} line${linesBy[c.id] === 1 ? "" : "s"}` : "silent in this chapter"}</span></span>
    {c.voice && <small className="t-help">{c.voice}</small>}
    <span className="control-row"><select aria-label={`${c.name} voice`} value={assigned(c.id)} disabled={job}
      onChange={(e) => {
        const value = e.target.value;
        saveCast((latest) => {
          const next = { ...latest.voices };
          if (value) next[c.id] = value; else delete next[c.id];
          return { voices: next };
        });
      }}>
      <option value="">Use narrator</option>
      {voiceOptions}
    </select>
    <PreviewButton id={c.id} voice={assigned(c.id) || narrator} character={c.id} label={`Preview ${c.name} voice`} /></span>
    {neural && <input aria-label={`${c.name} voice direction`} maxLength={1500}
      value={cast.directions?.[c.id] ?? c.voice} disabled={job} placeholder="Quiet, guarded; a slight tremor when afraid"
      onChange={(e) => { const value = e.target.value;
        saveCast((latest) => ({ directions: { ...latest.directions, [c.id]: value } }));
      }} />}
  </label>)}</>;
}

/** The sentence playing at `seconds`: the last one that has started. */
export function sentenceAt(spans: AudioSpan[], seconds: number): number {
  let found = -1;
  spans.forEach((span, i) => { if (sampleSeconds(span.start) <= seconds + 0.02) found = i; });
  return found;
}

/**
 * The transcript is the player: the sentence being read is highlighted,
 * clicking one plays from there, and any one can be read again, as it was or
 * with new words or another voice.
 */
function Transcript({ spans, now, onSeek, editable, voices, voiceOptions, narrator, onReread }: {
  spans: AudioSpan[];
  now: (() => number) | null;
  onSeek: (seconds: number) => void;
  editable: boolean;
  voices: { id: string; language: string }[];
  voiceOptions: React.ReactNode;
  narrator: string;
  onReread: (index: number, text?: string, voice?: string) => void;
}) {
  const [time, setTime] = useState(0);
  useEffect(() => {
    if (!now) return;
    setTime(now());
    const timer = window.setInterval(() => setTime(now()), 200);
    return () => window.clearInterval(timer);
  }, [now]);
  const active = now ? sentenceAt(spans, time) : -1;
  const list = useRef<HTMLOListElement>(null);
  useEffect(() => {
    list.current?.querySelector<HTMLElement>("li.on")?.scrollIntoView?.({ block: "nearest" });
  }, [active]);
  const [editing, setEditing] = useState<{ index: number; text: string; voice: string } | null>(null);
  return <div className="ab-transcript">
    <div className="sub-head">Transcript · click a sentence to play from there</div>
    <ol ref={list}>
      {spans.map((span, i) => <li key={`${i}-${span.start}`} className={i === active ? "on" : undefined}>
        <button type="button" className="ab-sentence" aria-current={i === active ? "true" : undefined}
          title={clock(sampleSeconds(span.start))} onClick={() => onSeek(sampleSeconds(span.start))}>{span.text}</button>
        {editable && <span className="ab-sentence-tools">
          <button type="button" className="btn small" aria-label={`Re-read sentence ${i + 1}`} title="Read this sentence again"
            onClick={() => onReread(i)}>↻</button>
          <button type="button" className="btn small" aria-label={`Edit sentence ${i + 1}`} title="Change the words or the voice"
            onClick={() => setEditing({ index: i, text: span.text,
              voice: voices.some((v) => v.id === span.voice) ? span.voice : narrator })}>✎</button>
        </span>}
        {editable && editing?.index === i && <div className="ab-edit">
          <input aria-label="Sentence text" value={editing.text} onChange={(e) => setEditing({ ...editing, text: e.target.value })} />
          <select aria-label="Voice for this sentence" value={editing.voice}
            onChange={(e) => setEditing({ ...editing, voice: e.target.value })}>{voiceOptions}</select>
          <button type="button" className="btn small primary" disabled={!editing.text.trim()}
            onClick={() => { onReread(i, editing.text, editing.voice); setEditing(null); }}>Re-read</button>
          <button type="button" className="btn small" onClick={() => setEditing(null)}>Cancel</button>
        </div>}
      </li>)}
    </ol>
  </div>;
}
