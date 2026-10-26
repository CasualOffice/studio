"""The narration script: a story retold for one storyteller, never translated."""
import json
import unittest
from unittest.mock import patch

import narration

HINGLISH = "Yaar, main kal office gaya tha. Boss ne bola ki meeting cancel ho gayi 😅. Phir maine socha, chalo chai peete hain."


class Script(unittest.TestCase):
    def test_the_retelling_is_spelled_in_devanagari_by_the_word_list_not_the_model(self):
        retold = "Yaar, main kal office gaya tha. Boss ne bola ki meeting cancel ho gayi. Phir maine socha, chalo chai peete hain."
        prompts = []
        def write(system, prompt):
            prompts.append(system)
            return retold
        text, stopped = narration.script(HINGLISH, write, devanagari=True)
        self.assertFalse(stopped)
        self.assertNotIn("Devanagari", prompts[0])
        self.assertEqual(text, "यार, मैं कल office गया था। Boss ने बोला कि meeting cancel हो गई। फिर मैंने सोचा, चलो चाय पीते हैं।")

    def test_a_reply_that_is_not_a_retelling_is_retried_then_the_original_kept(self):
        calls = []
        def empty(*_):
            calls.append(1)
            return ""
        text, _ = narration.script(HINGLISH, empty, devanagari=False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(text, HINGLISH.strip())

    def test_a_reply_far_shorter_than_the_passage_is_not_a_retelling(self):
        text, _ = narration.script(HINGLISH, lambda *_: "Office.", devanagari=False)
        self.assertEqual(text, HINGLISH.strip())

    def test_stop_keeps_what_is_written_and_the_original_for_the_rest(self):
        source = "One. " * 200 + "Two. " * 200
        calls = []
        def write(system, prompt):
            calls.append(1)
            if len(calls) > 1:
                raise narration.Stop()
            return prompt.split("Passage:\n")[1].upper()
        text, stopped = narration.script(source, write, words=250)
        self.assertTrue(stopped)
        self.assertTrue(text.startswith("ONE."))
        self.assertIn("Two.", text)

    def test_a_script_is_narrated_by_one_voice_paragraph_by_paragraph(self):
        segs = narration.segments("पहला पैराग्राफ।\n\nदूसरा पैराग्राफ।")
        self.assertEqual([s["speaker_id"] for s in segs], [None, None])
        self.assertEqual(len(segs), 2)


class Speech(unittest.TestCase):
    def test_emoji_and_symbols_never_reach_the_voice(self):
        self.assertEqual(narration.speakable("Kahan ho?? 😤 *bas* 5 min 👍🏽").strip(), "Kahan ho?? bas 5 min")

    def test_texting_is_said_in_full_by_rule(self):
        self.assertEqual(narration.normalize("u r late, pls msg me in 5 min 😤").strip(),
                         "you are late, please message me in 5 minutes")
        self.assertEqual(narration.normalize("Sooo sorry!!! 1 hr, lol."), "So sorry! 1 hour.")
        # Initials and ordinary words are left alone.
        self.assertEqual(narration.normalize("U. K. Rao said ok."), "U. K. Rao said ok.")


class Segments(unittest.TestCase):
    def test_one_storyteller_reads_sentence_by_sentence_with_pauses(self):
        segs = narration.segments("One. Two.\n\nThree.\n\n* * *\n\nFour.")
        self.assertEqual([(s["text"], s["pause"]) for s in segs],
                         [("One.", "scene"), ("Two.", "sentence"), ("Three.", "paragraph"), ("Four.", "scene")])
        self.assertTrue(all(s["speaker_id"] is None for s in segs))

    def test_quotes_are_not_split_from_their_sentence_for_one_voice(self):
        segs = narration.segments('“Wait,” she said. “Don\'t go.”')
        self.assertEqual([s["text"] for s in segs], ['“Wait,” she said.', '“Don\'t go.”'])

    def test_a_chat_line_is_the_name_a_pause_then_the_message(self):
        segs = narration.segments("[10:42] Arjun: kahan ho?? 😤\nMira: 5 min")
        self.assertEqual([(s["text"], s["pause"]) for s in segs],
                         [("Arjun.", "scene"), ("kahan ho?", "speaker"), ("Mira.", "paragraph"), ("5 minutes", "speaker")])

    def test_headings_name_the_markers_and_long_text_gets_more_of_them(self):
        segs = narration.segments("# Chapter One\n\nThe pier was empty.\n\nChapter 2\n\nRain.")
        self.assertEqual([s["scene"] for s in segs], ["Chapter One", "Chapter One", "Chapter 2", "Chapter 2"])
        self.assertEqual(segs[0]["text"], "Chapter One")
        long = "\n\n".join(f"Paragraph {n} begins here and goes on. " + "word " * 120 for n in range(6))
        labels = list(dict.fromkeys(s["scene"] for s in narration.segments(long)))
        self.assertEqual(labels[0], "Paragraph 0 begins here and goes…")
        self.assertGreater(len(labels), 1)
        self.assertNotIn("Passage", " ".join(labels))

    def test_character_segments_become_sentences_and_passage_numbers_become_words(self):
        segs = narration.refine([
            {"speaker_id": None, "text": "The pier was empty. He waited.\n\nNight came. ", "mood": "calm", "scene": "Passage 1"},
            {"speaker_id": "mira", "text": "“Hi.”", "mood": "calm", "scene": "Passage 1"},
            {"speaker_id": None, "text": "Morning.", "mood": "", "scene": "The kitchen"}])
        self.assertEqual([(s["text"], s["speaker_id"], s["pause"]) for s in segs],
                         [("The pier was empty.", None, "scene"), ("He waited.", None, "sentence"),
                          ("Night came.", None, "paragraph"), ("“Hi.”", "mira", "sentence"),
                          ("Morning.", None, "scene")])
        self.assertEqual([s["scene"] for s in segs][::4], ["The pier was empty. He waited.…", "The kitchen"])


class Worker(unittest.TestCase):
    def test_script_op_suggests_retellings_that_keep_the_authors_words(self):
        import worker
        retold = "Yaar, main kal office gaya tha. Boss ne bola ki meeting cancel ho gayi, thodi hansi aayi. Phir maine socha, chalo chai peete hain."
        with patch.object(worker, '_write', return_value=retold):
            out = worker.op_narration_script("n", {"source": HINGLISH})
        self.assertEqual([(s["original"], s["suggestion"]) for s in out["suggestions"]], [(HINGLISH, retold)])
        self.assertEqual(out["script"], retold)
        # A "retelling" that translates is not one, however fluent.
        with patch.object(worker, '_write', return_value="यार, मैं कल ऑफिस गया था। बॉस ने बोला कि मीटिंग कैंसल हो गई। फिर मैंने सोचा, चलो चाय पीते हैं।"):
            self.assertEqual(worker.op_narration_script("n", {"source": HINGLISH})["suggestions"], [])

    def test_pasted_text_is_read_by_one_voice_without_any_analysis(self):
        import worker
        import audiobook
        captured = {}
        def fake(segments, narrator, assignments, rate, check, recording, progress, **kw):
            captured.update(segments=segments, assignments=assignments)
            import numpy as np
            for s in segments:
                recording.add(s, np.zeros(10, dtype=np.float32), narrator)
        with patch.object(audiobook, "narrate_kokoro", fake), patch.object(worker, "_speech_model"), \
                patch("vaultcrypto.write_sealed", lambda *a: 4), patch.object(worker, "emit"):
            result = worker.op_audiobook("a", {"source": HINGLISH, "narrator": "hf_alpha",
                                               "voices": {"x": "am_puck"}, "model_id": audiobook.KOKORO_MODEL_ID,
                                               "vault_slots": [{"id": "s", "path": "/tmp/x", "key": "00" * 32, "file_id": "00" * 16}]})
        self.assertTrue(result["sealed"])
        self.assertEqual(captured.get("assignments"), {})
        # Sentences, not quote-split passages, with no emoji left in them.
        self.assertEqual([s["text"] for s in captured["segments"]][:2],
                         ["Yaar, main kal office gaya tha.", "Boss ne bola ki meeting cancel ho gayi."])
        self.assertEqual(len(result["spans"]), 3)


if __name__ == "__main__":
    unittest.main()
