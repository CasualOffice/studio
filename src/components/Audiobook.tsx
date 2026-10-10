import { useEffect, useRef, useState } from "react";
import { api, errText, isCancelled, newJobId, onEngineProgress, vaultUrl } from "../lib/api";
import { ANALYST_IDS, SPEECH_MODEL_ID, keepBook, type AudioSettings, type StoryLibrary } from "../lib/story";
import type { EngineProgress, ModelStatus } from "../lib/types";
import { JobProgress } from "./shared";

export default function Audiobook({ library, models, notify, onLibraryChange, onProduced }: {
  library: StoryLibrary | null;
  models: ModelStatus[];
  notify: (message: string, bad?: boolean) => void;
  onLibraryChange: (update: (current: StoryLibrary) => StoryLibrary) => Promise<void>;
  onProduced: () => void;
}) {
  const [bookId, setBookId] = useState("");
  const [chapterId, setChapterId] = useState("");
  const [source, setSource] = useState("");
  const [title, setTitle] = useState("Chapter 1");
  const [append, setAppend] = useState(false);
  const [voices, setVoices] = useState<{ id: string; language: string }[]>([]);
  const [voiceError, setVoiceError] = useState("");
  const [job, setJob] = useState<string | null>(null);
  const [progress, setProgress] = useState<EngineProgress | null>(null);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  const book = library?.books.find((b) => b.id === (bookId || library.activeBookId)) ?? library?.books.at(-1);
  const chapter = book?.chapters.find((c) => c.id === chapterId) ?? book?.chapters.at(-1);
  const saved = book && library?.audio?.[book.id];
  const settings: AudioSettings = saved ?? { narrator: "", voices: {}, rate: 160, outputs: {} };
  const speechModel = settings.model_id || SPEECH_MODEL_ID;
  const neural = speechModel !== "macos";
  const speechReady = !neural || models.some((m) => m.id === speechModel && m.installed && !m.hopeless);
  useEffect(() => {
    let live = true;
    setVoices([]); setVoiceError("");
    void api.audioVoices(speechModel).then((r) => { if (live) setVoices(r.voices); })
      .catch((e) => { if (live) setVoiceError(errText(e)); });
    return () => { live = false; };
  }, [speechModel]);
  const narrator = voices.some((v) => v.id === settings.narrator) ? settings.narrator : voices[0]?.id || "";
  const assignedVoices = Object.fromEntries(Object.entries(settings.voices).filter(([, voice]) => voices.some((v) => v.id === voice)));
  const outputs = chapter ? settings.outputs[chapter.id] ?? [] : [];
  const writerReady = models.some((m) => ANALYST_IDS.includes(m.id) && m.installed && !m.hopeless);

  const saveSettings = (patch: Partial<AudioSettings> | ((latest: AudioSettings) => Partial<AudioSettings>)) => {
    if (!book) return;
    const id = book.id;
    void onLibraryChange((current) => {
      const latest = current.audio?.[id] ?? settings;
      return { ...current, audio: { ...current.audio,
        [id]: { ...latest, ...(typeof patch === "function" ? patch(latest) : patch) },
      } };
    }).catch((e) => notify(`Could not save voice settings: ${errText(e)}`, true));
  };
  const setSpeaker = (sceneId: string, beatId: string, start: number, speakerId: string) => {
    if (!book || !chapter) return;
    const selectedBook = book.id, selectedChapter = chapter.id;
    void onLibraryChange((current) => ({ ...current, books: current.books.map((b) => b.id !== selectedBook ? b : {
      ...b, chapters: b.chapters.map((c) => c.id !== selectedChapter ? c : { ...c,
        scenes: c.scenes.map((scene) => scene.id !== sceneId ? scene : { ...scene,
          beats: scene.beats.map((beat) => beat.id !== beatId ? beat : { ...beat,
            dialogue: beat.dialogue.map((line) => line.start !== start ? line : {
              ...line, speaker_id: speakerId || null, attribution_review: false,
            }),
          }),
        }),
      }),
    }) })).catch((e) => notify(`Could not save speaker attribution: ${errText(e)}`, true));
  };
  const perform = async (action: (id: string) => Promise<void>) => {
    const id = newJobId(); setJob(id); setProgress(null);
    let un: (() => void) | undefined;
    try {
      un = await onEngineProgress((p) => { if (p.job_id === id && mounted.current) setProgress(p); });
      await action(id);
    } catch (e) {
      notify(isCancelled(e) ? "Audiobook stopped." : errText(e), !isCancelled(e));
    } finally {
      un?.();
      if (mounted.current) { setJob(null); setProgress(null); }
    }
  };
  const analyze = () => perform(async (id) => {
    const result = await api.analyzeStory(id, source, title, append ? book ?? null : null);
    await onLibraryChange((current) => keepBook(current, result.book));
    setBookId(result.book.id); setChapterId(result.chapter_id); setSource(""); setAppend(false);
    notify("Chapter analyzed and saved. Board and Audiobook share its character bible.");
  });
  const narrate = () => perform(async (id) => {
    if (!book || !chapter) return;
    const audio = await api.generateAudiobook(id, book, chapter.id, narrator, assignedVoices, settings.rate, {
      model_id: speechModel, language: settings.language || "auto",
      delivery: settings.delivery || "", directions: settings.directions || {},
    });
    onProduced();
    // Audio remains a Vault item even if the follow-up state save fails.
    try { await onLibraryChange((current) => {
      const latest = current.audio?.[book.id] ?? settings;
      return { ...current, audio: { ...current.audio, [book.id]: {
        ...latest, narrator, outputs: { ...latest.outputs,
          [chapter.id]: [...(latest.outputs[chapter.id] ?? []), audio] },
      } } };
    });
    } catch (error) {
      notify(`Audio is saved in Vault, but its chapter link could not be saved: ${errText(error)}`, true);
      return;
    }
    notify("Chapter audio saved in the encrypted Vault.");
  });

  return <div className="audiobook-workspace">
    <div className="pane">
      <h2>Audiobook</h2>
      <p className="muted">Read the original chapter with local neural voices and character delivery. Characters and dialogue come from the same story analysis as Board.</p>
      {!library && <div className="notice warn">The encrypted story library is loading.</div>}
      {!!library?.books.length && <>
        <label>Book<select value={book?.id ?? ""} disabled={!!job}
          onChange={(e) => { setBookId(e.target.value); setChapterId(""); }}>
          {library.books.map((b) => <option key={b.id} value={b.id}>
            {b.chapters[0]?.title ?? "Book"} · {b.chapters.length} chapters
          </option>)}
        </select></label>
        <label>Chapter<select value={chapter?.id ?? ""} disabled={!!job}
          onChange={(e) => setChapterId(e.target.value)}>
          {book?.chapters.map((c) => <option key={c.id} value={c.id}>{c.title}</option>)}
        </select></label>
      </>}
      <details open={!book}>
        <summary>{book ? "Add a chapter or start another book" : "Start with your chapter"}</summary>
        <label>Chapter title<input value={title} disabled={!!job} onChange={(e) => setTitle(e.target.value)} /></label>
        <label>Prose<textarea rows={10} value={source} disabled={!!job}
          onChange={(e) => setSource(e.target.value)} placeholder="Paste the original prose…" /></label>
        {book && <label><input type="checkbox" checked={append} disabled={!!job}
          onChange={(e) => setAppend(e.target.checked)} /> Add to this book and reuse its characters</label>}
        <button className="btn primary" disabled={!!job || !source.trim() || !writerReady || !library}
          onClick={() => void analyze()}>Analyze chapter</button>
        {!writerReady && <p className="notice warn">Install a story analyst or the prompt writer in Models to analyze prose.</p>}
      </details>
      {book && chapter && <>
        <label>Speech engine<select value={speechModel} disabled={!!job}
          onChange={(e) => saveSettings({ model_id: e.target.value, narrator: "", voices: {} })}>
          <option value={SPEECH_MODEL_ID}>Qwen3-TTS · neural voices</option>
          <option value="macos">macOS system voices · fallback</option>
        </select></label>
        {neural && !speechReady && <p className="notice warn">Download Audiobook voices (Qwen3-TTS 1.7B) in Models to use neural narration.</p>}
        {neural && <>
          <label>Language<select value={settings.language || "auto"} disabled={!!job}
            onChange={(e) => saveSettings({ language: e.target.value })}>
            {["auto", "english", "chinese", "japanese", "korean", "german", "french", "russian", "portuguese", "spanish", "italian"].map((l) => <option key={l} value={l}>{l}</option>)}
          </select></label>
          <label>Narration direction<textarea rows={2} maxLength={1500} value={settings.delivery || ""} disabled={!!job}
            placeholder="Warm, intimate storytelling with restrained emotion and natural pauses."
            onChange={(e) => saveSettings({ delivery: e.target.value })} /></label>
        </>}
        <label>Narrator<select value={narrator} disabled={!!job || !voices.length}
          onChange={(e) => saveSettings({ narrator: e.target.value })}>
          {voices.map((v) => <option key={v.id} value={v.id}>{v.id} · {v.language}</option>)}
        </select></label>
        <label>{neural ? `Reading pace · ${settings.rate} (approximate)` : `Reading speed · ${settings.rate} words/minute`}<input type="range" min={80} max={300}
          value={settings.rate} disabled={!!job} onChange={(e) => saveSettings({ rate: Number(e.target.value) })} /></label>
        <div className="sub-head">Character voices</div>
        {book.characters.map((c) => <label key={c.id}>{c.name}
          {c.voice && <small className="muted">{c.voice}</small>}
          <select aria-label={`${c.name} voice`} value={assignedVoices[c.id] ?? ""} disabled={!!job}
            onChange={(e) => {
              const value = e.target.value;
              saveSettings((latest) => {
                const next = { ...latest.voices };
                if (value) next[c.id] = value; else delete next[c.id];
                return { voices: next };
              });
            }}>
            <option value="">Use narrator</option>
            {voices.map((v) => <option key={v.id} value={v.id}>{v.id} · {v.language}</option>)}
          </select>
          {neural && <input aria-label={`${c.name} voice direction`} maxLength={1500}
            value={settings.directions?.[c.id] ?? c.voice} disabled={!!job} placeholder="Quiet, guarded; a slight tremor when afraid"
            onChange={(e) => { const value = e.target.value;
              saveSettings((latest) => ({ directions: { ...latest.directions, [c.id]: value } }));
            }} />}
        </label>)}
        <button className="btn primary" disabled={!!job || !speechReady || !narrator || !voices.some((v) => v.id === narrator)}
          onClick={() => void narrate()}>Narrate chapter</button>
      </>}
      {!voices.length && <p className="notice warn">{voiceError || (neural ? "Neural voices are loading." : "Installed macOS speech voices are required for narration. Manage voices in macOS Accessibility → Spoken Content.")}</p>}
      {job && <button className="btn" onClick={() => void api.cancelJob(job).catch((e) => notify(errText(e), true))}>Stop</button>}
      {progress && <JobProgress p={progress} label="Audiobook" />}
    </div>
    <div className="pane">
      <h3>{chapter?.title ?? "Your narrated chapter"}</h3>
      {outputs.map((id, i) => <div className="audio-result" key={id}>
        <label>Recording {i + 1}</label><audio controls preload="metadata" src={vaultUrl(id)} />
        <p className="muted">Export the WAV file from Vault.</p>
      </div>)}
      {chapter && <details><summary>Original text · {chapter.scenes.length} analyzed scenes</summary>
        <p style={{ whiteSpace: "pre-wrap" }}>{chapter.source}</p></details>}
      {chapter?.scenes.map((scene) => <section key={scene.id} className="notice">
        <h4>{scene.title}</h4><p>{scene.summary}</p>
        {scene.beats.flatMap((beat) => beat.dialogue.map((line) => <div key={`${beat.id}-${line.start}`}>
          <p><strong>{book?.characters.find((c) => c.id === line.speaker_id)?.name ?? "Narrator / unattributed"}{line.offscreen ? " (offscreen)" : ""}:</strong> {line.text}</p>
          {line.attribution_review && <p className="notice warn">Speaker attribution needs review. The original words are preserved; narration uses the narrator until you assign a speaker.</p>}
          <label>Speaker<select aria-label={`Speaker for ${line.text}`} value={line.speaker_id ?? ""} disabled={!!job}
            onChange={(e) => setSpeaker(scene.id, beat.id, line.start, e.target.value)}>
            <option value="">Narrator / unknown speaker</option>
            {book?.characters.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select></label>
        </div>))}
      </section>)}
    </div>
  </div>;
}
