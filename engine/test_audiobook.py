"""Narration audio: pauses, loudness, streaming, partial chapters and re-reading.

Every voice here is a stand-in producing synthetic samples; no speech model
is loaded."""
import io
import secrets
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import audiobook
import vaultcrypto as vc

SR = audiobook.SAMPLE_RATE


def tone(seconds, amplitude):
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def kokoro(lengths=None, calls=None):
    """A stand-in Kokoro: one quiet tone per chunk, or silence for ""."""
    class Model:
        def generate(self, **kwargs):
            if calls is not None:
                calls.append(kwargs)
            if kwargs["text"].strip() == "silent.":
                return
            yield SimpleNamespace(audio=tone(0.2, 0.05), sample_rate=SR)
    return Model()


def wav_frames(data):
    with wave.open(io.BytesIO(data)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2')


class Pauses(unittest.TestCase):
    def test_sentences_paragraphs_and_scenes_get_their_own_breath(self):
        recording = audiobook.Recording()
        for pause in ("scene", "sentence", "paragraph", "scene", "speaker"):
            recording.add({"text": pause, "pause": pause}, tone(0.1, 0.2), "v")
        gaps = [b["start"] - a["end"] for a, b in zip(recording.spans, recording.spans[1:])]
        self.assertEqual(gaps, [int(SR * s) for s in (0.12, 0.5, 0.9, 0.3)])
        # The first sentence starts at once: nothing to pause after.
        self.assertEqual(recording.spans[0]["start"], 0)

    def test_a_segment_without_a_pause_breathes_longer_at_a_new_scene(self):
        recording = audiobook.Recording()
        for scene in ("One", "One", "Two"):
            recording.add({"text": "x", "scene": scene}, tone(0.1, 0.2), "v")
        gaps = [b["start"] - a["end"] for a, b in zip(recording.spans, recording.spans[1:])]
        self.assertEqual(gaps, [int(SR * 0.12), int(SR * 0.9)])
        self.assertEqual([m["label"] for m in recording.markers], ["One", "Two"])
        self.assertEqual(recording.markers[1]["time"], round(recording.spans[2]["start"] / SR, 2))


class Loudness(unittest.TestCase):
    def rms(self, x):
        return float(np.sqrt(np.mean(np.square(x[np.abs(x) > 0.01]))))

    def test_quiet_and_loud_sentences_come_out_at_the_same_level(self):
        quiet, loud = audiobook.level(tone(1, 0.05)), audiobook.level(tone(1, 0.6))
        self.assertAlmostEqual(self.rms(quiet), self.rms(loud), delta=0.01)
        self.assertAlmostEqual(self.rms(loud), audiobook.TARGET_RMS, delta=0.01)

    def test_peaks_are_held_below_full_scale(self):
        spiky = tone(1, 0.05)
        spiky[100] = 0.9
        self.assertLessEqual(float(np.max(np.abs(audiobook.level(spiky)))), audiobook.PEAK + 1e-6)

    def test_near_silence_is_not_amplified_into_noise(self):
        hiss = np.full(SR, 0.001, dtype=np.float32)
        np.testing.assert_array_equal(audiobook.level(hiss), hiss)


class Streaming(unittest.TestCase):
    def test_each_sentence_streams_as_pcm_with_its_pause_and_matches_the_file(self):
        sent = []
        recording = audiobook.Recording(on_sentence=lambda pcm, start, span, i: sent.append((pcm, start, span, i)))
        segments = [{"speaker_id": None, "text": "One.", "pause": "scene"},
                    {"speaker_id": None, "text": "Two.", "pause": "paragraph"}]
        audiobook.narrate_kokoro(segments, "af_heart", {}, 160, lambda: None, recording, load=kokoro)
        self.assertEqual([s[3] for s in sent], [0, 1])
        self.assertEqual(sent[1][1], sent[0][2]["end"])
        self.assertEqual(sent[1][2]["start"] - sent[1][1], int(SR * 0.5))
        streamed = b"".join(s[0] for s in sent)
        self.assertEqual(streamed, recording.wav_bytes()[44:])
        self.assertEqual([s[2]["text"] for s in sent], ["One.", "Two."])

    def test_kokoro_progress_counts_the_text_actually_read(self):
        seen = []
        segments = [{"speaker_id": None, "text": "Main kal office gaya tha."}]
        with patch("hinglish.devanagari", lambda text: text * 3):
            audiobook.synthesize_kokoro(segments, "hf_alpha", {}, 160, lambda: None,
                                        lambda done, total: seen.append((done, total)), load=kokoro)
        self.assertEqual(seen[-1][1], len("Main kal office gaya tha.") * 3)
        self.assertTrue(all(done <= total for done, total in seen))


class PartialChapters(unittest.TestCase):
    def test_a_silent_sentence_is_skipped_not_fatal(self):
        segments = [{"speaker_id": None, "text": t} for t in ("One.", "silent.", "Three.")]
        recording = audiobook.Recording()
        audiobook.narrate_kokoro(segments, "af_heart", {}, 160, lambda: None, recording, load=kokoro)
        self.assertEqual([s["text"] for s in recording.spans], ["One.", "Three."])
        self.assertEqual(recording.skipped, 1)

    def test_nothing_spoken_at_all_is_still_an_error(self):
        with self.assertRaisesRegex(ValueError, "no spoken audio"):
            audiobook.synthesize_kokoro([{"speaker_id": None, "text": "silent."}], "af_heart", {}, 160,
                                        lambda: None, load=kokoro)

    def worker_run(self, fail_after, error):
        import worker
        def narrate(segments, narrator, voices, rate, check, recording, progress, **kw):
            for i, s in enumerate(segments):
                if i == fail_after:
                    raise error
                recording.add(s, tone(0.1, 0.2), narrator)
        with tempfile.TemporaryDirectory() as d:
            slot = {"id": "s", "path": str(Path(d, "out")), "key": secrets.token_hex(32), "file_id": secrets.token_hex(16)}
            with patch.object(audiobook, "narrate_kokoro", narrate), patch.object(worker, "_speech_model"), \
                    patch.object(worker, "emit"):
                result = worker.op_audiobook("job", {"source": "One. Two. Three.", "narrator": "af_heart",
                                                     "model_id": audiobook.KOKORO_MODEL_ID, "vault_slots": [slot]})
            data = vc.open_with_file_key(bytes.fromhex(slot["key"]), Path(slot["path"]).read_bytes())
        return result, data

    def test_stop_keeps_the_sentences_already_read(self):
        import worker
        result, data = self.worker_run(2, worker.Cancelled())
        self.assertTrue(result["stopped"])
        self.assertIsNone(result["error"])
        self.assertEqual([s["text"] for s in result["spans"]], ["One.", "Two."])
        self.assertEqual(len(wav_frames(data)), result["spans"][-1]["end"])

    def test_a_failure_partway_keeps_what_was_read_and_says_why(self):
        result, data = self.worker_run(1, RuntimeError("voice crashed"))
        self.assertTrue(result["stopped"])
        self.assertEqual(result["error"], "voice crashed")
        self.assertEqual(len(result["spans"]), 1)

    def test_a_stop_before_any_audio_is_still_a_cancellation(self):
        import worker
        with self.assertRaises(worker.Cancelled):
            self.worker_run(0, worker.Cancelled())


class Sealing(unittest.TestCase):
    def test_sealing_in_pieces_is_byte_identical_to_sealing_whole(self):
        key, fid = secrets.token_bytes(32), secrets.token_bytes(16)
        for size in (0, 5, vc.CHUNK_SIZE, vc.CHUNK_SIZE + 1, 3 * vc.CHUNK_SIZE - 7):
            data = secrets.token_bytes(size)
            pieces = [data[i:i + 1000] for i in range(0, size, 1000)] or [b""]
            self.assertEqual(b"".join(vc.seal_parts(key, fid, pieces)), vc.seal_with_file_key(key, fid, data))

    def test_a_recording_seals_without_being_joined_first(self):
        recording = audiobook.Recording()
        for _ in range(3):
            recording.add({"text": "x"}, tone(0.5, 0.2), "v")
        key, fid = secrets.token_bytes(32), secrets.token_bytes(16)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d, "x"))
            size = vc.write_sealed(path, key, fid, recording.chunks())
            sealed = Path(path).read_bytes()
        self.assertEqual(size, len(sealed))
        self.assertEqual(vc.open_with_file_key(key, sealed), recording.wav_bytes())


class Splicing(unittest.TestCase):
    def spans(self):
        return [{"start": 0, "end": 100, "text": "One.", "voice": "a"},
                {"start": 150, "end": 250, "text": "Two.", "voice": "a"},
                {"start": 300, "end": 400, "text": "Three.", "voice": "a"}]

    def test_a_longer_reading_moves_everything_after_it(self):
        samples = np.arange(400, dtype='<i2')
        markers = [{"time": 0.0, "label": "Start"}, {"time": 300 / SR, "label": "Later"}]
        joined, spans, marks = audiobook.splice(samples, self.spans(), markers, 1,
                                                np.full(160, 7, dtype='<i2'), "Two, again.", "b")
        self.assertEqual(joined.size, 460)
        self.assertEqual(spans[1], {"start": 150, "end": 310, "text": "Two, again.", "voice": "b"})
        self.assertEqual((spans[2]["start"], spans[2]["end"]), (360, 460))
        self.assertEqual(spans[0], self.spans()[0])
        np.testing.assert_array_equal(joined[150:310], 7)
        np.testing.assert_array_equal(joined[360:], np.arange(300, 400))
        self.assertEqual(marks[0]["time"], 0.0)
        self.assertAlmostEqual(marks[1]["time"], 360 / SR, delta=0.006)

    def test_a_shorter_reading_pulls_everything_back(self):
        joined, spans, _ = audiobook.splice(np.zeros(400, dtype='<i2'), self.spans(), [], 0,
                                            np.ones(40, dtype='<i2'), "One.", "a")
        self.assertEqual(joined.size, 340)
        self.assertEqual([(s["start"], s["end"]) for s in spans], [(0, 40), (90, 190), (240, 340)])

    def test_positions_that_do_not_fit_the_audio_are_refused(self):
        with self.assertRaises(ValueError):
            audiobook.splice(np.zeros(200, dtype='<i2'), self.spans(), [], 2, np.ones(4, dtype='<i2'), "x", "a")
        with self.assertRaises(ValueError):
            audiobook.splice(np.zeros(400, dtype='<i2'), self.spans(), [], 3, np.ones(4, dtype='<i2'), "x", "a")

    def test_the_worker_rereads_one_sentence_into_a_new_sealed_recording(self):
        import worker
        recording = audiobook.Recording()
        for text in ("One.", "Two.", "Three."):
            recording.add({"text": text, "pause": "sentence"}, tone(0.2, 0.2), "af_heart")
        key, fid = secrets.token_bytes(32), secrets.token_bytes(16)
        read = []
        def narrate(segments, narrator, voices, rate, check, take, progress, **kw):
            read.append((segments[0]["text"], narrator))
            take.add(segments[0], tone(0.5, 0.2), narrator)
        with tempfile.TemporaryDirectory() as d:
            source = str(Path(d, "old"))
            vc.write_sealed(source, key, fid, recording.chunks())
            slot = {"id": "new", "path": str(Path(d, "new")), "key": key.hex(), "file_id": fid.hex()}
            with patch.object(audiobook, "narrate_kokoro", narrate), patch.object(worker, "_speech_model"):
                result = worker.op_audiobook_revoice("job", {
                    "vault_inputs": [{"path": source, "key": key.hex()}], "vault_slots": [slot],
                    "spans": recording.spans, "markers": [], "index": 1, "text": "Two,  again.",
                    "voice": "hf_alpha", "model_id": audiobook.KOKORO_MODEL_ID, "rate": 160})
            data = vc.open_with_file_key(key, Path(slot["path"]).read_bytes())
        self.assertEqual(read, [("Two, again.", "hf_alpha")])
        frames = wav_frames(data)
        self.assertEqual(len(frames), result["spans"][-1]["end"])
        self.assertEqual(result["spans"][1]["end"] - result["spans"][1]["start"], int(0.5 * SR))
        self.assertEqual(result["spans"][1]["voice"], "hf_alpha")
        self.assertAlmostEqual(result["seconds"], len(frames) / SR)


class HindiVoices(unittest.TestCase):
    def test_macos_hindi_voices_get_devanagari_and_english_voices_do_not(self):
        self.assertEqual(audiobook.spoken("Main ghar ja raha hu.", "Lekha", {"Lekha": "hi_IN"}), "मैं घर जा रहा हूँ।")
        self.assertEqual(audiobook.spoken("Main ghar ja raha hu.", "Samantha", {"Samantha": "en_US"}),
                         "Main ghar ja raha hu.")
        self.assertEqual(audiobook.spoken("Main ghar ja raha hu.", "hf_alpha", None), "मैं घर जा रहा हूँ।")


class KeptVoice(unittest.TestCase):
    def test_a_previewed_model_reads_the_chapter_without_loading_again(self):
        import worker
        loads, model = [], object()
        worker._VOICE.update(model=model, id=audiobook.KOKORO_MODEL_ID)
        try:
            with patch.object(audiobook, "load_kokoro_model", lambda: loads.append(1)), \
                    patch.object(worker.CACHE, "unload"), patch.object(worker, "emit"):
                load = worker._speech_model("job", audiobook.KOKORO_MODEL_ID)
                self.assertIs(load(), model)
            self.assertEqual(loads, [])
            self.assertIn("audiobook", worker._KEEPS_VOICE)
            self.assertIn("audiobook_revoice", worker._KEEPS_VOICE)
        finally:
            worker._VOICE.update(model=None, id=None)

    def test_another_engines_model_is_released_first(self):
        import worker
        worker._VOICE.update(model=object(), id=audiobook.NEURAL_MODEL_ID)
        fresh = object()
        try:
            with patch.object(audiobook, "load_kokoro_model", lambda: fresh), \
                    patch.object(worker.CACHE, "unload"), patch.object(worker, "emit"):
                self.assertIs(worker._speech_model("job", audiobook.KOKORO_MODEL_ID)(), fresh)
        finally:
            worker._VOICE.update(model=None, id=None)


if __name__ == "__main__":
    unittest.main()
