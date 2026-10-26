"""Romanized Hinglish is spelled in Devanagari for a Hindi voice; English stays English."""
import unittest
from unittest.mock import patch

import hinglish

# The English lexicon comes with Kokoro (misaki); the tests pin a small one so
# they mean the same with or without it installed.
REAL_ENGLISH = hinglish.english
ENGLISH = hinglish.ENGLISH_CORE | {"car", "min", "thing", "tell", "want", "market", "going"}


def setUpModule():
    patcher = patch.object(hinglish, "english", lambda: ENGLISH)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class Spelling(unittest.TestCase):
    def test_hindi_words_go_to_devanagari_and_english_words_stay(self):
        self.assertEqual(
            hinglish.devanagari("Yaar, main kal office gaya tha, but boss ne bola ki meeting cancel ho gayi."),
            "यार, मैं कल office गया था, but boss ने बोला कि meeting cancel हो गई।")

    def test_an_english_sentence_is_left_alone(self):
        text = "The main thing is to tell us what you want. I'm in the car."
        self.assertEqual(hinglish.devanagari(text), text)

    def test_shared_words_follow_their_clause(self):
        self.assertEqual(hinglish.devanagari("Main to ghar ja raha tha."), "मैं तो घर जा रहा था।")
        self.assertEqual(hinglish.devanagari("Arjun: tum late ho, msg kyun nahi kiya?"),
                         "अर्जुन: तुम late हो, msg क्यों नहीं किया?")
        self.assertEqual(hinglish.devanagari("I'm in the car, aa raha hu."), "I'm in the car, आ रहा हूँ।")
        self.assertEqual(hinglish.devanagari("Maine phone use kiya."), "मैंने phone use किया।")
        self.assertEqual(hinglish.devanagari("Maine use dekha."), "मैंने उसे देखा।")

    def test_the_is_the_english_article_unless_it_completes_a_hindi_verb(self):
        # Regressions: "the" leaned Hindi, so with no English word beside it
        # every shared word in the clause fell back to Hindi together.
        self.assertEqual(hinglish.devanagari("The boss ne kaha"), "The boss ने कहा")
        self.assertEqual(hinglish.devanagari("Use the door, yaar"), "Use the door, यार")
        self.assertEqual(hinglish.devanagari("Wo log aaye the."), "वो लोग आये थे।")
        self.assertEqual(hinglish.devanagari("Hum ghar pe the to boss aaya."), "हम घर पे थे तो boss आया।")

    def test_the_word_list_corrects_the_lexicons_wrong_favourites(self):
        # Dakshina's most attested spellings of these are a name or a loanword.
        for roman, deva in (("me", "में"), ("tab", "तब"), ("us", "उस"), ("kuch", "कुछ"), ("haan", "हाँ")):
            self.assertEqual(hinglish._hindi(roman), deva)

    def test_ki_after_saying_or_thinking_is_the_conjunction(self):
        self.assertIn("सोचा कि", hinglish.devanagari("Usne socha ki wo aayegi."))
        self.assertIn("घर की", hinglish.devanagari("Wo ghar ki chabi bhool gaya."))

    def test_an_unknown_word_among_hindi_is_spelled_by_sound(self):
        self.assertEqual(hinglish.spell("bolegi"), "बोलेगी")
        self.assertEqual(hinglish.spell("hain"), "हैं")
        self.assertEqual(hinglish.spell("dukaan"), "दुकान")

    def test_devanagari_and_numbers_pass_through(self):
        for text in ("यार, मैं आ रहा हूँ।", "5 min", "", "...!"):
            self.assertEqual(hinglish.devanagari(text), text)


class RealLexicon(unittest.TestCase):
    def test_kokoros_english_lexicon_keeps_ordinary_english_english(self):
        try:
            import misaki  # noqa: F401
        except ImportError:
            self.skipTest("Kokoro's packages are not installed")
        with patch.object(hinglish, "english", REAL_ENGLISH):
            self.assertEqual(hinglish.devanagari("Bas 5 min, traffic mein phasi hu, I'm in the car."),
                             "बस 5 min, traffic में फँसी हूँ, I'm in the car।")


class Voice(unittest.TestCase):
    def test_hindi_voices_get_devanagari_and_english_voices_the_text_as_written(self):
        import audiobook
        heard = []

        class Model:
            def generate(self, text, voice, speed, lang_code):
                heard.append((voice, lang_code, text))
                import numpy as np
                yield type("R", (), {"sample_rate": 24000, "audio": np.zeros(240, dtype="float32")})()

        segment = [{"speaker_id": None, "text": "Main kal office gaya tha."}]
        audiobook.synthesize_kokoro(segment, "hf_alpha", {}, 160, lambda: None, load=Model)
        audiobook.synthesize_kokoro(segment, "af_heart", {}, 160, lambda: None, load=Model)
        self.assertEqual(heard[0], ("hf_alpha", "h", "मैं कल office गया था।"))
        self.assertEqual(heard[1], ("af_heart", "a", "Main kal office gaya tha."))


if __name__ == "__main__":
    unittest.main()
