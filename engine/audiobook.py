"""Offline narration. Intermediates live only in private staging."""
import io
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import hinglish

SAMPLE_RATE = 24000

# Pauses a storyteller leaves before a sentence, in seconds; see
# narration.PAUSES, which says which one each sentence gets. Kept here too so
# a segment without a pause (an older caller, a preview) still gets a breath.
PAUSES = {"sentence": 0.12, "speaker": 0.3, "paragraph": 0.5, "scene": 0.9}

# Loudness: each sentence is brought to about -20 dBFS RMS with its peak held
# under -1 dBFS. Voices differ by several decibels between sentences, and
# more between a narrator and a character; clipped and butted together, a
# chapter swelled and faded line by line.
TARGET_RMS = 0.1
PEAK = 0.89


def voices():
    if sys.platform != "darwin":
        return []
    out = subprocess.run(["/usr/bin/say", "-v", "?"], check=True,
                         capture_output=True, text=True, timeout=20).stdout
    result = []
    for line in out.splitlines():
        match = re.match(r'(.+?)\s+([a-z]{2,3}_[A-Za-z_]+)\s+#', line)
        if match:
            result.append({"id": match[1].strip(), "language": match[2]})
    return result


def run_process(args, check_cancel):
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            while proc.poll() is None:
                check_cancel()
                time.sleep(0.05)
            check_cancel()
            if proc.returncode:
                raise RuntimeError(f"Local speech command failed (exit {proc.returncode}). Check the installed voice.")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def render_segment(text, voice, rate, directory, check_cancel):
    text_path, aiff, wav = [os.path.join(directory, n) for n in ("speech.txt", "speech.aiff", "speech.wav")]
    Path(aiff).unlink(missing_ok=True)
    Path(wav).unlink(missing_ok=True)
    Path(text_path).write_text(text, encoding="utf-8")
    run_process(["/usr/bin/say", "-v", voice, "-r", str(rate), "-f", text_path, "-o", aiff], check_cancel)
    run_process(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1", aiff, wav], check_cancel)
    return Path(wav).read_bytes()


def _mark(markers, segment, seconds):
    """A chapter marker where a new scene's narration begins."""
    if markers is not None and segment.get("scene") and (not markers or markers[-1]["label"] != segment["scene"]):
        markers.append({"time": round(seconds, 2), "label": segment["scene"]})


def wav_header(frames):
    """The 44-byte header of a mono 16-bit 24 kHz WAV of `frames` samples."""
    size = frames * 2
    return (b"RIFF" + struct.pack("<I", 36 + size) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
            + b"data" + struct.pack("<I", size))


def read_wav(data):
    """A narrated WAV's samples as int16, checking it is what narration writes."""
    import numpy as np
    with wave.open(io.BytesIO(data), 'rb') as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError("voice output must be mono 24 kHz PCM")
        frames = audio.getnframes()
    # A view of the samples where they lie, not a copy: a chapter is large.
    at = data.find(b"data", 12)
    if at < 0 or at + 8 + frames * 2 > len(data):
        raise ValueError("voice output is not a complete WAV")
    return np.frombuffer(data, dtype='<i2', count=frames, offset=at + 8)


def level(samples):
    """One sentence at a steady loudness: RMS toward TARGET_RMS, peak under PEAK.

    Measured over the voiced samples only, so a sentence with long silences
    is not boosted until its words clip. A clip with almost no voice in it is
    left as it is rather than amplifying noise."""
    import numpy as np
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    voiced = samples[np.abs(samples) > 0.01]
    if voiced.size < SAMPLE_RATE // 50:
        return np.clip(samples, -1, 1)
    rms = float(np.sqrt(np.mean(np.square(voiced, dtype=np.float64))))
    peak = float(np.max(np.abs(samples)))
    gain = min(TARGET_RMS / rms, PEAK / peak, 8.0)
    return np.clip(samples * gain, -1, 1)


class Recording:
    """Narrated audio as it is made, a sentence at a time.

    It keeps the samples once, as int16 -- the WAV is only assembled while it
    is sealed -- with where each sentence starts and ends, so the transcript
    can follow the voice and one sentence can be re-read and spliced in later.
    `on_sentence` receives each sentence's PCM, the pause before it included,
    as soon as it exists: the first one plays while the rest are still being
    read. Whatever was recorded survives a Stop or a failure; the caller
    decides to keep it."""

    def __init__(self, on_sentence=None, markers=None):
        self.parts = []
        self.frames = 0
        self.spans = []
        self.markers = markers if markers is not None else []
        self.skipped = 0
        self.on_sentence = on_sentence
        self._scene = None

    @property
    def seconds(self):
        return self.frames / SAMPLE_RATE

    def pause_before(self, segment):
        if not self.frames:
            return 0.0
        pause = segment.get("pause")
        if pause is None:
            pause = "scene" if segment.get("scene") and segment.get("scene") != self._scene else "sentence"
        return PAUSES.get(pause, PAUSES["sentence"])

    def add(self, segment, samples, voice):
        import numpy as np
        gap = np.zeros(int(round(self.pause_before(segment) * SAMPLE_RATE)), dtype='<i2')
        clip = (level(samples) * 32767).astype('<i2')
        start = self.frames
        if gap.size:
            self.parts.append(gap)
            self.frames += gap.size
        _mark(self.markers, segment, self.frames / SAMPLE_RATE)
        self._scene = segment.get("scene")
        span = {"start": self.frames, "end": self.frames + clip.size,
                "text": segment.get("display", segment["text"]).strip(), "voice": voice}
        self.parts.append(clip)
        self.frames += clip.size
        self.spans.append(span)
        if self.on_sentence:
            self.on_sentence(gap.tobytes() + clip.tobytes(), start, span, len(self.spans) - 1)

    def chunks(self):
        """The WAV, header first, without joining it into one buffer."""
        yield wav_header(self.frames)
        for part in self.parts:
            yield memoryview(part).cast("B")

    def wav_bytes(self):
        if not self.frames:
            raise ValueError("chapter produced no spoken audio")
        return b"".join(self.chunks())


def splice(samples, spans, markers, index, clip, text, voice):
    """Replace sentence `index` with a newly read `clip`.

    Later sentences and markers move by the change in length, so the
    transcript and the chapter list still point at the right words. Returns
    (samples, spans, markers); nothing passed in is changed."""
    import numpy as np
    if not 0 <= index < len(spans):
        raise ValueError("no such sentence in this recording")
    old = spans[index]
    start, end = int(old["start"]), int(old["end"])
    if not 0 <= start <= end <= len(samples):
        raise ValueError("this recording's sentence positions do not match its audio")
    clip = np.asarray(clip, dtype='<i2')
    shift = clip.size - (end - start)
    joined = np.concatenate([samples[:start], clip, samples[end:]])
    moved = []
    for i, span in enumerate(spans):
        if i == index:
            moved.append({**span, "start": start, "end": start + clip.size, "text": text, "voice": voice})
        elif int(span["start"]) >= end:
            moved.append({**span, "start": int(span["start"]) + shift, "end": int(span["end"]) + shift})
        else:
            moved.append(dict(span))
    after = end / SAMPLE_RATE
    marks = [{**m, "time": round(m["time"] + shift / SAMPLE_RATE, 2)} if m["time"] >= after else dict(m)
             for m in markers]
    return joined, moved, marks


def spoken(text, voice, languages):
    """The words a voice should receive: Hindi voices read Hinglish properly
    only from Devanagari (an M4 test: 22% of words misheard from Latin
    script, 7% from Devanagari). Kokoro's h-voices and macOS hi_IN voices
    alike; English voices get the text as written."""
    language = languages.get(voice, "") if languages else ""
    hindi = voice in KOKORO_VOICES and voice[0] == "h" or language.startswith("hi")
    return hinglish.devanagari(text) if hindi else text


def _script(segments, narrator, assignments, languages=None):
    """Each segment with its voice and the text that voice reads.

    Converted up front so progress counts the characters actually read:
    measured on the source, it ran past 100% once Devanagari changed the
    length."""
    result = []
    for segment in segments:
        voice = assignments.get(segment["speaker_id"], narrator)
        result.append((segment, voice, spoken(segment["text"], voice, languages)))
    return result


def narrate_system(segments, narrator, assignments, rate, staging, check_cancel, recording,
                   progress=lambda *_: None, render=render_segment, available=None):
    import numpy as np
    listed = voices() if available is None else [{"id": v, "language": ""} for v in available]
    languages = {v["id"]: v["language"] for v in listed}
    if narrator not in languages or any(v not in languages for v in assignments.values()):
        raise ValueError("choose an installed macOS voice for narrator and characters")
    if type(rate) is not int or not 80 <= rate <= 300:
        raise ValueError("speech rate must be 80..300 words per minute")
    with tempfile.TemporaryDirectory(prefix="audiobook-", dir=staging) as directory:
        os.chmod(directory, 0o700)
        script = _script(segments, narrator, assignments, languages)
        for i, (segment, voice, text) in enumerate(script):
            check_cancel()
            progress(i, len(script))
            pieces = []
            for part in re.findall(r'[\s\S]{1,1800}(?:\s|$)|[\s\S]{1,1800}', text):
                if not part.strip() or not any(c.isalnum() for c in part):
                    continue
                check_cancel()
                pieces.append(read_wav(render(part, voice, rate, directory, check_cancel)))
            if not sum(p.size for p in pieces):
                # One silent sentence used to lose the whole chapter.
                recording.skipped += 1
                continue
            recording.add(segment, np.concatenate(pieces).astype(np.float32) / 32768, voice)


def synthesize(segments, narrator, assignments, rate, staging, check_cancel,
               progress=lambda *_: None, render=render_segment, available=None, markers=None):
    recording = Recording(markers=markers)
    narrate_system(segments, narrator, assignments, rate, staging, check_cancel, recording,
                   progress, render, available)
    return recording.wav_bytes(), recording.seconds


NEURAL_MODEL_ID = "qwen3-tts-1.7b-custom-6bit"
NEURAL_REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit"
NEURAL_SPEAKERS = ("ryan", "aiden", "vivian", "serena", "dylan", "eric", "uncle_fu", "ono_anna", "sohee")
NEURAL_LANGUAGES = ("auto", "english", "chinese", "japanese", "korean", "german", "french", "russian", "portuguese", "spanish", "italian")


def neural_voices():
    # Speaker identities are multilingual; language is selected separately.
    return [{"id": speaker, "language": "multilingual"} for speaker in NEURAL_SPEAKERS]


# Kokoro: an 82M-parameter voice model, measured on an M4 at ~14x real time
# against ~1.8x for Qwen3-TTS, with Hindi voices. It takes no delivery
# instructions; pace is its speed setting.
KOKORO_MODEL_ID = "kokoro-82m"
KOKORO_REPO = "mlx-community/Kokoro-82M-bf16"
# Voice-name prefix -> the language its pipeline reads. Japanese and Chinese
# need further packages and are not offered.
KOKORO_LANGUAGES = {"a": "English (US)", "b": "English (UK)", "h": "Hindi / Indian English",
                    "e": "Spanish", "f": "French", "i": "Italian", "p": "Portuguese"}
KOKORO_VOICES = (
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_aoede", "af_kore", "af_nova", "af_river",
    "am_adam", "am_michael", "am_echo", "am_eric", "am_liam", "am_onyx", "am_puck", "am_fenrir",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "ff_siwis", "if_sara", "im_nicola", "pf_dora", "pm_alex")


def kokoro_voices():
    return [{"id": v, "language": KOKORO_LANGUAGES[v[0]]} for v in KOKORO_VOICES]


def load_kokoro_model():
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import load_model
    try:
        local = Path(snapshot_download(KOKORO_REPO, local_files_only=True))
    except Exception as exc:
        raise ValueError("Download Kokoro voices in Models first") from exc
    try:
        import misaki  # noqa: F401
        import en_core_web_sm  # noqa: F401
    except ImportError as exc:
        raise ValueError("Kokoro's text packages are missing; download Kokoro again in Models") from exc
    model = load_model(local)
    # Voices otherwise load lazily from another repository over the network;
    # the downloaded one already holds them under voices/.
    model.repo_id = KOKORO_REPO
    return model


def kokoro_language(voice, text):
    """Hindi voices read Devanagari as Hindi and English with their accent."""
    if voice[0] == "h" and not re.search(r'[ऀ-ॿ]', text):
        return "a"
    return voice[0]


def _release():
    import gc
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except ImportError:
        pass


def narrate_kokoro(segments, narrator, assignments, rate, check_cancel, recording,
                   progress=lambda *_: None, load=load_kokoro_model):
    if narrator not in KOKORO_VOICES or any(v not in KOKORO_VOICES for v in assignments.values()):
        raise ValueError("choose Kokoro voices for narrator and characters")
    if type(rate) is not int or not 80 <= rate <= 300:
        raise ValueError("reading pace must be 80..300")
    check_cancel()
    model = load()
    speed = min(2.0, max(0.5, rate / 160))
    try:
        import numpy as np
        script = _script(segments, narrator, assignments)
        total, done = sum(len(text) for *_, text in script) or 1, 0
        for segment, voice, text in script:
            pieces = []
            for part in speech_chunks(text):
                check_cancel()
                progress(done, total)
                done += len(part)
                if not any(c.isalnum() for c in part):
                    continue
                for result in model.generate(text=part, voice=voice, speed=speed,
                                             lang_code=kokoro_language(voice, part)):
                    check_cancel()
                    if result.sample_rate != SAMPLE_RATE:
                        raise ValueError("Kokoro output must be 24 kHz")
                    samples = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                    if not np.isfinite(samples).all():
                        raise ValueError("Kokoro produced invalid audio samples")
                    pieces.append(samples)
            if not sum(p.size for p in pieces):
                # One silent sentence used to lose the whole chapter.
                recording.skipped += 1
                continue
            recording.add(segment, np.concatenate(pieces), voice)
    finally:
        del model
        _release()


def synthesize_kokoro(segments, narrator, assignments, rate, check_cancel,
                      progress=lambda *_: None, load=load_kokoro_model, markers=None):
    recording = Recording(markers=markers)
    narrate_kokoro(segments, narrator, assignments, rate, check_cancel, recording, progress, load)
    check_cancel()
    return recording.wav_bytes(), recording.seconds


def speech_chunks(text, limit=350):
    """Bound inference without dropping, duplicating or rewriting source text.

    Each chunk is generated without the others' context, so a cut mid-sentence
    is heard as a broken pause. Prefer the last sentence end, then a line break,
    then clause punctuation, and a bare space only when there is none."""
    while len(text) > limit:
        head = text[:limit]
        for pattern in (r'[.!?…][”"’)\]]*\s+|[。！？][」』）]*', r'\n+', r'[,;:—–、，；：]\s*', r'\s+'):
            boundaries = [m.end() for m in re.finditer(pattern, head) if m.end() > limit // 3]
            if boundaries:
                end = boundaries[-1]
                break
        else:
            end = limit
        yield text[:end]
        text = text[end:]
    if text:
        yield text


def load_neural_model():
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import load_model
    try:
        local = Path(snapshot_download(NEURAL_REPO, local_files_only=True))
    except Exception as exc:
        raise ValueError("Download Audiobook voices (Qwen3-TTS 1.7B) in Models first") from exc
    if not (local / "speech_tokenizer" / "config.json").is_file():
        raise ValueError("The audiobook model download is incomplete; download it again in Models")
    # Use the local snapshot, never a repo ID, during synthesis.
    model = load_model(local)
    if model.tokenizer is None or model.speech_tokenizer is None:
        raise ValueError("The audiobook model tokenizer could not be loaded; check its installation")
    return model


def narrate_neural(segments, narrator, assignments, rate, check_cancel, recording,
                   progress=lambda *_: None, language="auto", delivery="",
                   directions=None, character_directions=None, load=load_neural_model):
    if narrator not in NEURAL_SPEAKERS or any(v not in NEURAL_SPEAKERS for v in assignments.values()):
        raise ValueError("choose a neural narrator and neural character voices")
    if language not in NEURAL_LANGUAGES:
        raise ValueError("unsupported narration language")
    if type(rate) is not int or not 80 <= rate <= 300:
        raise ValueError("reading pace must be 80..300")
    directions, character_directions = directions or {}, character_directions or {}
    if not isinstance(delivery, str) or len(delivery) > 1500 or any(
            not isinstance(v, str) or len(v) > 1500 for v in directions.values()):
        raise ValueError("voice directions must be text under 1,500 characters")
    check_cancel()
    model = load()
    try:
        import numpy as np
        supported = {s.lower() for s in model.get_supported_speakers()}
        if narrator not in supported or any(v not in supported for v in assignments.values()):
            raise ValueError("a selected voice is not supported by the downloaded model")
        total, done = sum(len(s["text"]) for s in segments) or 1, 0
        for segment in segments:
            check_cancel()
            speaker_id = segment["speaker_id"]
            voice = assignments.get(speaker_id, narrator)
            # Bible text is only bounded at 4,000 characters, and the
            # instruction is sent again with every chunk.
            direction = directions.get(speaker_id, character_directions.get(speaker_id, ""))[:600]
            mood = (segment.get("mood") or "")[:200]
            pace = "Read slowly with measured pauses." if rate < 140 else "Read briskly but clearly." if rate > 190 else "Read at a natural conversational pace."
            instruct = " ".join(filter(None, ["Read the supplied text verbatim. Do not add words.", delivery, direction, f"Scene mood: {mood}." if mood else "", pace]))
            pieces = []
            for part in speech_chunks(segment["text"]):
                # Per chunk, by characters: a chapter without dialogue is
                # one long segment and sat at 0% for the whole run.
                progress(done, total)
                done += len(part)
                if not any(c.isalnum() for c in part):
                    continue
                token_count = 0
                check_cancel()
                results = model.generate_custom_voice(text=part, speaker=voice,
                    language=language, instruct=instruct, stream=True,
                    streaming_interval=0.5, max_tokens=2048, verbose=False)
                try:
                    for result in results:
                        check_cancel()
                        token_count += getattr(result, "token_count", 0)
                        if token_count >= 2048:
                            raise ValueError("neural narration reached its token limit; the sentences before it were kept")
                        if result.sample_rate != SAMPLE_RATE:
                            raise ValueError("neural voice output must be 24 kHz")
                        samples = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                        if not np.isfinite(samples).all():
                            raise ValueError("neural voice produced invalid audio samples")
                        pieces.append(samples)
                finally:
                    if hasattr(results, "close"):
                        results.close()
            if not sum(p.size for p in pieces):
                # One silent sentence used to lose the whole chapter.
                recording.skipped += 1
                continue
            recording.add(segment, np.concatenate(pieces), voice)
    finally:
        # The caller decides whether the model stays loaded for previews and
        # re-reads; this releases only what synthesis itself allocated.
        del model
        _release()


def synthesize_neural(segments, narrator, assignments, rate, check_cancel,
                      progress=lambda *_: None, language="auto", delivery="",
                      directions=None, character_directions=None, load=load_neural_model, markers=None):
    recording = Recording(markers=markers)
    narrate_neural(segments, narrator, assignments, rate, check_cancel, recording, progress,
                   language, delivery, directions, character_directions, load)
    check_cancel()
    return recording.wav_bytes(), recording.seconds
