"""Prompt help without a model: the rules op_assist relies on.

The measured cases from docs/prompt-enhancer-audit.md (PL-01 to PL-12 and the
PD diagnostics) are the regression set here. Each one is a request a real
writer got wrong, and the assertion is about what the checks now see.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prompt_assist as pa  # noqa: E402


def missing(request: str, candidate: str) -> list[str]:
    return [c["text"] for c in pa.extract_constraints(request)
            if not pa.satisfied(c, candidate)]


class TheCleanerNeverTouchesQuotes(unittest.TestCase):
    def test_lettering_in_quotes_survives(self):
        # PL-04: the sign's lettering was stripped as quality folklore.
        cleaned, removed = pa.clean_request('A sign reading exactly "BEST QUALITY 4K"')
        self.assertEqual(cleaned, 'A sign reading exactly "BEST QUALITY 4K"')
        self.assertEqual(removed, [])

    def test_folklore_outside_quotes_still_goes(self):
        cleaned, removed = pa.clean_request('a masterpiece 8k sign that says "8k ready"')
        self.assertEqual(cleaned, 'a sign that says "8k ready"')
        self.assertEqual(sorted(removed), ["8k", "masterpiece"])

    def test_curly_and_single_quotes_are_quotes_too(self):
        for text in ("a banner reading “HDR 4K”", "a mug that says 'BEST QUALITY'"):
            with self.subTest(text=text):
                self.assertEqual(pa.clean_request(text)[0], text)

    def test_an_apostrophe_does_not_open_a_quote(self):
        cleaned, removed = pa.clean_request("the cat's 4k toy and the dog's bowl")
        self.assertEqual(removed, ["4k"])
        self.assertEqual(cleaned, "the cat's toy and the dog's bowl")

    def test_pd08_pd09_a_folklore_word_that_is_the_subject_stays(self):
        for text in ("a masterpiece hanging in a gallery",
                     "Unreal Engine editor interface on a laptop"):
            with self.subTest(text=text):
                self.assertEqual(pa.clean_request(text), (text, []))
        # ... while the same word as a tag still goes.
        self.assertEqual(pa.clean_request("a masterpiece 8k cat")[0], "a cat")

    def test_a_request_of_nothing_but_folklore_is_empty(self):
        self.assertEqual(pa.clean_request("masterpiece, best quality, 8k")[0], "")


class WhatTheRequestFixes(unittest.TestCase):
    """Constraints read from the words, so the writer cannot leave one out."""

    def kinds(self, request):
        return {(c["kind"], c["text"]) for c in pa.extract_constraints(request)}

    def test_pl02_count_negation_and_colour(self):
        got = self.kinds("2 red cars with no sunroof, plain white background")
        self.assertIn(("count", "2 red cars"), got)
        self.assertIn(("negation", "no sunroof"), got)
        self.assertIn(("colour", "red cars"), got)
        self.assertIn(("colour", "white background"), got)

    def test_pl03_position_and_bindings(self):
        got = self.kinds("a red cube left of a blue sphere")
        self.assertIn(("relation", "left of"), got)
        self.assertIn(("colour", "red cube"), got)
        self.assertIn(("colour", "blue sphere"), got)

    def test_pl05_a_colour_belongs_to_the_person_named(self):
        got = self.kinds("Mira gives Jonas a brass key. Mira wears red; Jonas wears blue.")
        self.assertIn(("colour", "Mira in red"), got)
        self.assertIn(("colour", "Jonas in blue"), got)
        self.assertNotIn(("colour", "red jonas"), got,
                         "a colour ending one clause is not bound to the next")

    def test_pl07_two_exclusions(self):
        got = self.kinds("Black ink line drawing of a cat on a white background. "
                         "No shading, no color.")
        self.assertIn(("negation", "no shading"), got)
        self.assertIn(("negation", "no colour"), got)

    def test_one_negation_covering_two_things(self):
        got = self.kinds("a portrait with no shading or colour")
        self.assertIn(("negation", "no shading"), got)
        self.assertIn(("negation", "no colour"), got)

    def test_pl11_pace_is_held(self):
        got = self.kinds("A runner sprints quickly across the track. Locked camera.")
        self.assertIn(("pace", "sprints"), got)
        self.assertIn(("pace", "quickly"), got)
        self.assertIn(("camera", "locked camera"), got)

    def test_quoted_text_is_not_read_as_a_negation(self):
        got = self.kinds('a sign that says "NO PARKING"')
        self.assertEqual(got, {("quoted", '"NO PARKING"')})

    def test_pd06_what_the_subject_does_is_held_softly(self):
        got = {(c["kind"], c["text"], c["hard"])
               for c in pa.extract_constraints("a cat walking through a forest")}
        self.assertIn(("action", "walking", False), got)
        self.assertEqual(missing("a cat walking through a forest",
                                 "a cat sitting in a forest"), ["walking"])
        self.assertEqual(missing("a cat walking through a forest",
                                 "a cat walks through a forest"), [])

    def test_a_plain_request_fixes_nothing(self):
        self.assertEqual(pa.extract_constraints("a hous at nite"), [])


class TheChecksSeeTheLosses(unittest.TestCase):
    """PD-01 to PD-06: mutations the word-matching validators accepted."""

    def test_pd01_a_negation_flipped(self):
        self.assertEqual(missing("a red car with no sunroof",
                                 "a red car with a sunroof"), ["no sunroof"])

    def test_pd02_a_count_changed(self):
        self.assertIn("2 red cars", missing("2 red cars", "3 red cars"))

    def test_pd03_a_position_swapped(self):
        self.assertIn("left of", missing("a red cube left of a blue sphere",
                                         "a red cube right of a blue sphere"))

    def test_pd04_giver_and_receiver_swapped(self):
        self.assertEqual(missing("Mira gives Jonas the key", "Jonas gives Mira the key"),
                         ["Mira before Jonas"])
        self.assertEqual(missing("Mira gives Jonas the key",
                                 "Mira hands Jonas a small brass key"), [])

    def test_pd05_colours_swapped_between_objects(self):
        lost = missing("a red cube and a blue sphere", "a blue cube and a red sphere")
        self.assertIn("red cube", lost)
        self.assertIn("blue sphere", lost)

    def test_pl05_a_person_lost_their_colour(self):
        self.assertEqual(
            missing("Mira gives Jonas a brass key. Mira wears red; Jonas wears blue.",
                    "Mira hands Jonas a brass key in a hallway, Jonas in blue."),
            ["Mira in red"])

    def test_pl11_a_sprint_slowed(self):
        self.assertEqual(
            missing("A runner sprints quickly across the track.",
                    "A runner drifts slowly across the track."),
            ["sprints", "quickly"])


class GoodParaphrasesPass(unittest.TestCase):
    """The checks are literal, but not about wording that keeps the meaning."""

    CASES = (
        ("2 red cars with no sunroof", "Two red cars, neither with a sunroof."),
        ("a red car with no sunroof", "A red car without a sunroof."),
        ("a red cube left of a blue sphere",
         "A glossy red cube sits to the left of a matte blue sphere."),
        ("a red cube", "a cube painted red"),
        ("a red cube", "a cube of deep red glass"),
        ("Mira wears red; Jonas wears blue",
         "Mira, in a long red coat, stands beside Jonas, dressed in blue."),
        ("No shading, no color.", "Without shading or colour."),
        ("a dog with no collar", "a collarless dog"),
        ("A runner sprints quickly", "A runner sprinting quickly"),
    )

    def test_each_paraphrase_keeps_every_constraint(self):
        for request, candidate in self.CASES:
            with self.subTest(request=request):
                self.assertEqual(missing(request, candidate), [])


class ReadingTheWriter(unittest.TestCase):
    def test_clean_json(self):
        parsed, how = pa.parse_reply('{"prompt": "A cat.", "additions": []}')
        self.assertEqual((parsed["prompt"], how), ("A cat.", "json"))

    def test_think_block_and_code_fence(self):
        parsed, _ = pa.parse_reply(
            '<think>\nhmm\n</think>\n```json\n{"prompt": "A cat."}\n```')
        self.assertEqual(parsed["prompt"], "A cat.")

    def test_a_reply_cut_off_inside_a_list(self):
        parsed, how = pa.parse_reply(
            '{"prompt": "A cat on a chair.", "additions": ["on a ch')
        self.assertEqual(parsed["prompt"], "A cat on a chair.")
        self.assertEqual(how, "repaired")

    def test_a_reply_that_is_not_json_at_all(self):
        parsed, how = pa.parse_reply("Here is a cat.\nIt sits on a chair.")
        self.assertEqual(how, "prose")
        self.assertEqual(parsed["prompt"], "Here is a cat. It sits on a chair.")

    def test_garbage_is_none(self):
        self.assertEqual(pa.parse_reply("{{{")[0], None)
        self.assertEqual(pa.parse_reply("<think>only thinking")[0], None)

    def test_wrong_types_are_coerced(self):
        parsed, _ = pa.parse_reply('{"prompt": "A cat.", "additions": "by a window", '
                                   '"styles": null, "unclear": "yes"}')
        self.assertEqual(parsed["additions"], ["by a window"])
        self.assertEqual(parsed["styles"], [])
        self.assertFalse(parsed["unclear"], "only a real true is unclear")

    def test_an_echoed_placeholder_is_not_content(self):
        parsed, _ = pa.parse_reply('{"prompt": "A cat.", "styles": ["<style one>"], '
                                   '"faithful": "<the request said clearly>"}')
        self.assertEqual(parsed["styles"], [])
        self.assertEqual(parsed["faithful"], "")

    def test_preview_shows_the_prose_being_written(self):
        self.assertEqual(pa.preview('{"subject": "x", "faithful": "A ca'), "A ca")
        self.assertEqual(pa.preview('{"faithful": "A cat.", "prompt": "A grey'), "A grey")
        self.assertEqual(pa.preview('{"subject": "x"'), "")


class TheDescriptionIsTidied(unittest.TestCase):
    # Real Qwen3.5-9B endings, from the live audit run.
    def test_folklore_the_writer_adds_comes_out(self):
        self.assertEqual(pa.tidy_description(
            "A dog dressed as a king. Rendered in a photorealistic style with 8k resolution.",
            "my dog but make him look like a king"), "A dog dressed as a king.")

    def test_an_unasked_style_comes_out_as_a_sentence_or_a_clause(self):
        self.assertEqual(pa.tidy_description(
            "A cat sits on a chair. The image is rendered in a photorealistic style with sharp focus.",
            "a cat on a chair"), "A cat sits on a chair.")
        self.assertEqual(pa.tidy_description(
            "Framed close on the key, rendered in a realistic photographic style.",
            "Mira gives Jonas a brass key."), "Framed close on the key.")

    def test_quotes_and_asked_for_styles_stay(self):
        self.assertEqual(pa.tidy_description(
            "A sign reading 'BEST QUALITY 4K', rendered in a clean style with high definition.",
            'A sign reading exactly "BEST QUALITY 4K"'), "A sign reading 'BEST QUALITY 4K'.")
        kept = "A cat in black ink, a clean vector-style line drawing, rendered in an ink style."
        self.assertEqual(pa.tidy_description(kept, "Black ink line drawing of a cat"), kept)


class Variants(unittest.TestCase):
    PARSED = {"faithful": "A cat on a chair.",
              "prompt": "A grey cat sits on a chair by a window.",
              "additions": ["grey", "by a window", "a fireplace"],
              "styles": ["pencil sketch", "oil painting"]}

    def test_enhance_gives_faithful_enriched_stylistic(self):
        out = pa.build_variants(self.PARSED, "enhance", "generate")
        self.assertEqual([v["kind"] for v in out], ["faithful", "enriched", "stylistic"])
        self.assertEqual(out[1]["additions"], ["grey", "by a window"])
        self.assertEqual(out[2]["prompt"],
                         "A grey cat sits on a chair by a window. Pencil sketch.")

    def test_style_gives_one_per_style(self):
        out = pa.build_variants(self.PARSED, "style", "generate")
        self.assertEqual([v["label"] for v in out], ["Pencil sketch", "Oil painting"])

    def test_tighten_gives_one(self):
        out = pa.build_variants({"prompt": "A cat on a chair."}, "tighten", "generate")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["additions"], [])

    def test_an_edit_gets_no_stylistic_rewrite_unasked(self):
        out = pa.build_variants(self.PARSED, "enhance", "edit")
        self.assertNotIn("stylistic", [v["kind"] for v in out])

    def test_duplicates_collapse(self):
        out = pa.build_variants({"faithful": "A cat.", "prompt": "a cat."},
                                "enhance", "generate")
        self.assertEqual(len(out), 1)

    def test_the_primary_is_the_richest_that_lost_nothing(self):
        out = pa.build_variants(self.PARSED, "enhance", "generate")
        out[1]["missing"] = ["no sunroof"]
        self.assertEqual(pa.primary(out, "enhance")["kind"], "stylistic")


class Profiles(unittest.TestCase):
    def test_each_model_family_gets_its_template(self):
        for mode, mid, fam, want in (
            ("generate", "flux2-klein-9b-4bit", "flux2", "flux2"),
            ("generate", "qwen-image-2512-8bit", "qwen", "qwen_image"),
            ("generate", "z-image-turbo-4bit", "z-image", "z_image"),
            ("generate", "flux1-schnell-4bit", "flux", "flux1"),
            ("edit", "flux1-kontext-4bit", "flux", "kontext"),
            ("edit", "qwen-image-edit-2511-4bit", "qwen", "qwen_edit"),
            ("edit", "flux2-klein-4b-4bit", "flux2", "flux2_edit"),
            ("video", "wan22-ti2v-5b-8bit", "wan", "wan"),
            ("video", "ltx-video-2b", "", "ltx"),
            ("generate", "", "", "generic"),
        ):
            with self.subTest(model=mid, mode=mode):
                self.assertEqual(pa.profile_for(mode, mid, fam), want)

    def test_lengths_follow_the_model(self):
        self.assertIn("60 to 120 words", pa.system_prompt("enhance", "generate", "flux2"))
        self.assertIn("80 to 200 words", pa.system_prompt("enhance", "generate", "qwen_image"))

    def test_no_style_is_chosen_for_an_enhance(self):
        text = pa.system_prompt("enhance", "generate", "flux2")
        self.assertIn("Choose no art style or medium unless the request implies one", text)

    def test_the_budget_fits_the_goal(self):
        tight = pa.token_budget("tighten", "flux2", "a cat on a chair")
        full = pa.token_budget("enhance", "flux2", "a cat on a chair")
        qwen = pa.token_budget("enhance", "qwen_image", "a cat on a chair")
        self.assertLess(tight, full)
        self.assertLess(full, qwen)
        # About half a minute at fifteen tokens a second, at most.
        self.assertLess(qwen, 15 * 40)


class ThePicture(unittest.TestCase):
    TEXT = ("Here is what I see:\n"
            "1. man in a jacket | centre, foreground | red\n"
            "- woman with an umbrella | left, background | yellow\n"
            "car | right | silver, black\n"
            "a sentence with no columns\n")

    def test_regions_are_read_with_places_and_colours(self):
        regions = pa.parse_regions(self.TEXT)
        self.assertEqual([r["thing"] for r in regions],
                         ["man in a jacket", "woman with an umbrella", "car"])
        self.assertEqual(regions[1]["where"], "left, background")
        self.assertEqual(regions[2]["colours"], "silver, black")

    def test_the_old_labelled_shape_still_reads(self):
        self.assertEqual(pa.parse_regions("SUBJECT: a beige teapot\nLIGHT: soft")[0]["thing"],
                         "a beige teapot")

    def test_it_means_the_prominent_thing(self):
        regions = pa.parse_regions(self.TEXT)
        self.assertEqual(pa.resolve_target("make it blue", regions)["thing"],
                         "man in a jacket")

    def test_a_named_thing_is_that_thing(self):
        regions = pa.parse_regions(self.TEXT)
        self.assertEqual(pa.resolve_target("remove the umbrella", regions)["thing"],
                         "woman with an umbrella")

    def test_the_place_settles_two_of_a_kind(self):
        regions = [{"thing": "person", "where": "left, foreground", "colours": ""},
                   {"thing": "person", "where": "right, foreground", "colours": ""}]
        # PL-10.
        self.assertEqual(pa.resolve_target("remove the person on the right",
                                           regions)["where"], "right, foreground")

    def test_nothing_named_and_no_pronoun_is_no_guess(self):
        regions = pa.parse_regions(self.TEXT)
        self.assertIsNone(pa.resolve_target("brighten the whole scene", regions))
        self.assertIsNone(pa.resolve_target("make it blue", []))


if __name__ == "__main__":
    unittest.main()
