"""The director turns an analysed chapter into a shot list worth drawing."""
import json
import unittest
from pathlib import Path

import director
import story

SOURCE = ('The harbour lay grey under low cloud. Gulls wheeled over the empty pier.\n\n'
          'Agnes Thorne waited by the ticket hut. “You’re late,” Agnes said.\n\n'
          '“The ferry was early,” I said. “Never in forty years,” Agnes said, and laughed.\n\n'
          'Hours passed. The clock above the hut ticked on. Rain began.\n\n'
          'Agnes slammed the hut door and ran for the boat as the storm broke.')


def book():
    """A chapter analysed by a stand-in writer: three scenes, two people."""
    replies = iter([
        {"title": "The pier", "characters": [{"id": "a", "name": "Agnes Thorne", "aliases": ["Agnes"], "visual": "broad, grey hair", "voice": ""},
                                              {"id": "n", "name": "Narrator", "aliases": [], "visual": "", "voice": ""}],
         "looks": [], "places": [{"place": "pier", "state": "grey, empty"}],
         "beats": [{"units": [0, 1], "importance": 2, "action": "The empty harbour", "place": "pier", "mood": "bleak", "cast": []},
                   {"units": [2, 3], "importance": 3, "action": "Agnes waits", "place": "pier", "mood": "wry",
                    "cast": [{"id": "a", "feel": "arms folded"}]},
                   {"units": [4, 5, 6], "importance": 4, "action": "They argue about the ferry", "place": "pier", "mood": "wry",
                    "cast": [{"id": "a", "feel": "laughing"}, {"id": "n", "feel": "stung"}]}],
         "speakers": []},
        {"title": "Waiting", "characters": [], "looks": [], "places": [],
         "beats": [{"units": [0, 1, 2], "importance": 2, "action": "Time passes", "place": "pier", "mood": "slow", "cast": []}],
         "speakers": []},
        {"title": "The storm", "characters": [], "looks": [], "places": [],
         "beats": [{"units": [0], "importance": 5, "action": "Agnes runs for the boat", "place": "pier", "mood": "violent storm",
                    "cast": [{"id": "Agnes", "feel": "fierce"}]}],
         "speakers": []},
    ])
    def write(_, prompt):
        group = json.loads(prompt.split('Numbered source units:\n')[1].split('\nYour previous reply')[0])
        reply = next(replies)
        reply["beats"] = [dict(b, units=[u for u in b["units"] if u < len(group)]) for b in reply["beats"]]
        reply["beats"] = [b for b in reply["beats"] if b["units"]]
        return json.dumps(reply)
    return story.analyze(SOURCE.replace('\n\nHours', '\n\n* * *\n\nHours').replace('\n\nAgnes slammed', '\n\n* * *\n\nAgnes slammed'),
                         write, passage_words=999, passage_units=99)


def one_beat(source, characters, cast, importance=3, place="pier", speakers=()):
    """`source` analysed as a single beat by a stand-in writer."""
    def write(_, prompt):
        group = json.loads(prompt.split('Numbered source units:\n')[1].split('\nYour previous reply')[0])
        return json.dumps({"title": "t", "characters": characters, "looks": [], "places": [],
                           "speakers": list(speakers),
                           "beats": [{"units": [u["id"] for u in group], "importance": importance, "action": "a",
                                      "place": place, "mood": "calm", "cast": [{"id": c, "feel": ""} for c in cast]}]})
    return story.analyze(source, write, passage_words=999, passage_units=99)


class Plan(unittest.TestCase):
    """The structure the rules give a chapter, before any budget is applied."""

    def setUp(self):
        self.book = book()
        self.chapter = self.book['chapters'][0]
        self.panels = director.plan(self.book, self.chapter['id'], "manga", limit=99)

    def test_no_shot_is_repeated_and_every_panel_shows_its_own_instant(self):
        for a, b in zip(self.panels, self.panels[1:]):
            if b["transition"] != "aspect":
                self.assertNotEqual((a["shot"], a["angle"], a["focus"]), (b["shot"], b["angle"], b["focus"]))
            self.assertNotEqual(a["moment"], b["moment"])

    def test_the_place_is_established_first_and_nothing_is_padded_on(self):
        self.assertEqual((self.panels[0]["role"], self.panels[0]["framing"]), ("E", "establishing"))
        self.assertIn(self.panels[0]["shot"], ("EWS", "WS"))
        # No filler: a "first sight" panel per person and a pull-back at the
        # end turned 69 words into twelve panels.
        for p in self.panels:
            self.assertFalse(p["moment"].startswith(("First sight", "Pulling far back")), p["moment"])

    def test_first_person_sentences_draw_the_narrator_and_looking_is_point_of_view(self):
        b = one_beat('I carried my suitcase down the long pier. I saw the ferry leave without me.',
                     [{"id": "n", "name": "Narrator", "aliases": [], "visual": "tall", "voice": ""}], ["n"])
        panels = director.plan(b, b['chapters'][0]['id'], "manga", limit=99)
        carry = next(p for p in panels if p["moment"].startswith("I carried"))
        self.assertIn("Narrator", carry["characters"])
        self.assertNotEqual(carry["framing"], "pov")
        saw = next(p for p in panels if p["moment"].startswith("I saw"))
        self.assertEqual(saw["framing"], "pov")

    def test_every_line_of_dialogue_is_placed_once_with_shot_reverse_coverage(self):
        said = [d["text"] for p in self.panels for d in p["dialogue"]]
        expected = [d["text"] for s in self.chapter["scenes"] for b in s["beats"] for d in b["dialogue"]]
        self.assertEqual(said, expected)
        # The exchange between two visible people opens on a two-shot, then
        # goes over the shoulder.
        talk = [p for p in self.panels if p["dialogue"] and len(p["cast"]) >= 2]
        self.assertEqual(talk[0]["framing"], "two")
        self.assertIn("ots", [p["framing"] for p in talk])

    def test_waiting_is_shown_as_quiet_aspect_panels(self):
        wait = [p for p in self.panels if p["transition"] == "aspect"]
        self.assertGreaterEqual(len(wait), 2)
        self.assertIn("INSERT", [p["shot"] for p in wait])

    def test_each_scene_has_one_peak_and_violence_gets_a_dramatic_angle(self):
        peaks = [p for p in self.panels if p["role"] == "P"]
        self.assertEqual(len({p["scene"] for p in peaks}), len(peaks))
        storm = next(p for p in peaks if "storm" in p["moment"] or "ran" in p["moment"])
        self.assertIn(storm["angle"], ("low", "dutch"))
        self.assertEqual(storm["weight"], 3)

    def test_people_keep_their_side_of_the_frame_and_face_each_other(self):
        sides = {}
        for p in self.panels:
            for name, pos in p["positions"].items():
                sides.setdefault(name, set()).add(pos)
                self.assertEqual(p["facing"][name], "right" if "left" in pos else "left" if "right" in pos else "the camera")
        self.assertTrue(all(len(s) == 1 for s in sides.values()), sides)

    def test_manga_stands_the_first_person_on_the_right_where_the_eye_starts(self):
        western = director.plan(self.book, self.chapter['id'], "comic", limit=99)
        manga = self.panels
        first = lambda panels: next(p["positions"]["Agnes Thorne"] for p in panels if "Agnes Thorne" in p["positions"])  # noqa: E731
        self.assertEqual((first(western), first(manga)), ("left third", "right third"))

    def test_close_shots_are_capped(self):
        for i in range(len(self.panels)):
            window = self.panels[max(0, i - 9):i + 1]
            self.assertLessEqual(sum(p["shot"] in director.CLOSE for p in window), 5)

    def test_a_picture_book_is_one_spread_per_scene_with_the_whole_text(self):
        spreads = director.plan(self.book, self.chapter['id'], "picture", pages=99)
        self.assertEqual(len(spreads), len(self.chapter['scenes']))
        self.assertEqual(''.join(p['source'] for p in spreads), self.chapter['source'])
        self.assertTrue(all(p['weight'] == 3 and not p['dialogue'] for p in spreads))

    def test_a_picture_book_with_fewer_pages_than_scenes_keeps_every_word(self):
        spreads = director.plan(self.book, self.chapter['id'], "picture", pages=2)
        self.assertEqual(len(spreads), 2)
        self.assertEqual(''.join(p['source'] for p in spreads), self.chapter['source'])

    def test_scenes_are_the_storys_own_not_the_analysis_passages(self):
        source = 'Mira waited on the pier. ' * 3 + '\n\nLater, Jonas made tea in the kitchen.'
        b = one_beat(source, [{"id": "m", "name": "Mira", "aliases": [], "visual": "", "voice": ""},
                              {"id": "j", "name": "Jonas", "aliases": [], "visual": "", "voice": ""}], ["m"])
        # The stand-in writer puts everything in one analysed scene.
        self.assertEqual(len(b['chapters'][0]['scenes']), 2)
        panels = director.plan(b, b['chapters'][0]['id'], "manga", limit=99)
        self.assertEqual(sorted({p["scene"] for p in panels}), [1, 2])


class Budget(unittest.TestCase):
    """The panel count follows the words and the pages, not the sentences."""

    def test_about_one_panel_per_55_words_capped_by_the_pages(self):
        self.assertEqual(director.budget(69, "manga", 4), 3)
        self.assertEqual(director.budget(1500, "comic", 4), 20)
        self.assertEqual(director.budget(1500, "manga", 4), 16)
        self.assertEqual(director.budget(1500, "manga", 10), 27)
        self.assertEqual(director.budget(10, "comic", 1), 3)

    def test_a_short_story_is_not_exploded_into_a_dozen_panels(self):
        b = book()
        panels = director.plan(b, b['chapters'][0]['id'], "manga")
        self.assertLessEqual(len(panels), director.budget(len(SOURCE.split()), "manga", 4))

    def test_a_long_story_fits_the_pages_asked_for(self):
        source = ('Mira walked along the harbour wall. The boats knocked together in the swell. '
                  'She stopped at the ticket hut and read the timetable. ') * 40
        b = one_beat(source, [{"id": "m", "name": "Mira", "aliases": [], "visual": "", "voice": ""}], ["m"])
        for pages in (2, 4, 6):
            panels = director.plan(b, b['chapters'][0]['id'], "comic", pages=pages)
            self.assertLessEqual(len(panels), pages * director.PANELS_PER_PAGE["comic"])

    def test_merging_never_loses_a_spoken_line(self):
        b = book()
        panels = director.plan(b, b['chapters'][0]['id'], "manga", pages=1)
        expected = [d["text"] for s in b['chapters'][0]["scenes"] for bt in s["beats"] for d in bt["dialogue"]]
        said = " ".join(d["text"] for p in panels for d in p["dialogue"])
        for line in expected:
            self.assertIn(line, said)


class DroppedText(unittest.TestCase):
    """No sentence disappears without the coverage report saying so."""

    def test_bare_speech_tags_are_not_moments(self):
        self.assertEqual(director._sentences('“Late,” she said. “Early,” I said. Agnes laughed, a short bark.'),
                         ['Agnes laughed, a short bark.'])

    def test_a_sentence_without_a_quote_is_never_a_tag(self):
        # "began" and "laughed" are speech verbs, and these were dropped.
        self.assertEqual(director._sentences('Rain began. Mira laughed. “Run!” she cried.'),
                         ['Rain began.', 'Mira laughed.'])

    def test_every_sentence_of_a_long_beat_is_drawn_when_the_budget_allows(self):
        source = ('Mira opened the shutters. The street below was wet. A cart rolled past. '
                  'She leaned out. The baker waved. She waved back. The bells began to ring.')
        b = one_beat(source, [{"id": "m", "name": "Mira", "aliases": [], "visual": "", "voice": ""}], ["m"])
        panels = director.plan(b, b['chapters'][0]['id'], "manga", limit=99)
        report = director.coverage(director.script(b, b['chapters'][0]['id']), panels)
        self.assertEqual((report["percent"], report["missing"]), (100, []))

    def test_coverage_is_honest_when_the_budget_folds_sentences(self):
        b = book()
        cid = b['chapters'][0]['id']
        panels = director.plan(b, cid, "manga", pages=1)
        lines = director.script(b, cid)
        report = director.coverage(lines, panels)
        self.assertLess(report["percent"], 100)
        self.assertEqual(report["covered"] + report["missing_total"], report["total"])
        self.assertTrue(set(report["missing"]) <= set(lines))
        # What is missing is exactly what no panel draws or letters.
        shown = {s for p in panels for s in p["covers"]}
        for sentence in report["missing"]:
            self.assertNotIn(sentence, shown)

    def test_folded_sentences_are_lettered_while_the_panel_has_room(self):
        b = book()
        panels = director.plan(b, b['chapters'][0]['id'], "comic", pages=1)
        captioned = [p for p in panels if p["caption"]]
        self.assertTrue(captioned)
        for p in panels:
            words = sum(director._words(d["text"]) for d in p["dialogue"]) + director._words(p["caption"])
            self.assertLessEqual(words, max(director.PANEL_WORDS["comic"],
                                            sum(director._words(d["text"]) for d in p["dialogue"])))

    def test_a_long_line_is_lettered_as_several_balloons_with_every_word(self):
        speech = ' '.join(f'Word{i}.' for i in range(60))
        balloons = director._balloons(speech)
        self.assertTrue(all(director._words(x) <= director.BALLOON_WORDS for x in balloons))
        self.assertEqual(' '.join(balloons).split(), speech.split())


class DialogueAlignment(unittest.TestCase):
    def test_a_line_is_drawn_over_its_own_action_never_the_next_sentence(self):
        source = ('Agnes waited by the hut. “You’re late,” Agnes said, folding her arms. '
                  'The gulls cried overhead. “Sorry,” I said.')
        b = one_beat(source, [{"id": "a", "name": "Agnes", "aliases": [], "visual": "", "voice": ""},
                              {"id": "n", "name": "Narrator", "aliases": [], "visual": "", "voice": ""}], ["a", "n"])
        panels = director.plan(b, b['chapters'][0]['id'], "comic", limit=99)
        late = next(p for p in panels if any(d["text"] == "You’re late," for d in p["dialogue"]))
        self.assertIn("folding her arms", late["moment"])
        sorry = next(p for p in panels if any(d["text"] == "Sorry," for d in p["dialogue"]))
        self.assertNotIn("gulls", sorry["moment"])
        self.assertTrue(any(p["moment"] == "The gulls cried overhead." for p in panels))


class Briefs(unittest.TestCase):
    """A panel is drawn from a visual brief, never from the raw sentence."""

    def fx01(self):
        root = Path(__file__).parent / 'fixtures'
        fixture = json.loads((root / 'fx01.json').read_text())
        return story.analyze(fixture['source'], lambda *_: (root / 'fx01.analysis.json').read_text())

    def test_first_person_is_drawn_as_the_narrator_by_name(self):
        b = self.fx01()
        panels = director.plan(b, b['chapters'][0]['id'], "manga", limit=99)
        key = next(p for p in panels if p["moment"].startswith("In the station waiting room, I unhooked"))
        self.assertIn("Mira unhooked the brass key from Mira’s red coat", key["description"])
        # Said once, not again as a clause on the end.
        self.assertEqual(key["description"].count("station waiting room"), 1)
        self.assertNotRegex(key["description"], r"\bI\b|\bmy\b")

    def test_a_sentence_that_tells_is_briefed_from_the_analysed_action(self):
        b = self.fx01()
        panels = director.plan(b, b['chapters'][0]['id'], "manga", limit=99)
        name = next(p for p in panels if p["moment"].startswith("My name is"))
        self.assertNotIn("My name is", name["description"])
        self.assertTrue(name["brief_weak"])
        self.assertIn("Mira", name["description"])

    def test_every_panel_has_a_brief_and_the_guard_is_heard_not_drawn(self):
        b = self.fx01()
        panels = director.plan(b, b['chapters'][0]['id'], "manga")
        self.assertLessEqual(len(panels), director.budget(len(b['chapters'][0]['source'].split()), "manga"))
        self.assertTrue(all(p["description"] for p in panels))
        speech = next(p for p in panels if p["dialogue"])
        line = next(d for d in speech["dialogue"] if d["text"] == "Last train!")
        self.assertEqual(line["speaker"], "A guard")
        self.assertTrue(line["offscreen"])
        self.assertNotIn("A guard", [n for p in panels for n in p["characters"]])
        self.assertNotIn("Narrator", [c["name"] for c in b["characters"]])

    def test_the_establishing_brief_has_nobody_in_it(self):
        b = book()
        first = director.plan(b, b['chapters'][0]['id'], "manga", limit=99)[0]
        self.assertIn("Nobody in view", first["description"])
        self.assertIn("Grey, empty", first["description"])


class Refinement(unittest.TestCase):
    """The writer only rewrites weak briefs; the rules keep the camera."""

    def setUp(self):
        root = Path(__file__).parent / 'fixtures'
        fixture = json.loads((root / 'fx01.json').read_text())
        self.book = story.analyze(fixture['source'], lambda *_: (root / 'fx01.analysis.json').read_text())
        self.cid = self.book['chapters'][0]['id']
        self.planned = director.plan(self.book, self.cid, "manga", limit=99)
        self.weak = [i for i, p in enumerate(self.planned) if p["brief_weak"]]

    def test_only_weak_briefs_are_rewritten_and_shots_stay_the_rules(self):
        asked = []
        def write(system, prompt):
            rows = json.loads(prompt.split("Panels:\n")[1])
            asked.extend(r["id"] for r in rows)
            return json.dumps({"panels": [{"id": r["id"], "visual": "Mira stands, the key in her palm", "shot": "ECU"}
                                          for r in rows] + [{"id": 0, "visual": "not asked"}]})
        panels = director.direct(self.book, self.cid, "manga", write=write, limit=99)
        self.assertTrue(self.weak)
        self.assertEqual(asked, self.weak)
        for i, (p, q) in enumerate(zip(self.planned, panels)):
            self.assertEqual((p["shot"], p["angle"], p["framing"]), (q["shot"], q["angle"], q["framing"]))
            if i in self.weak:
                self.assertEqual(q["description"], "Mira stands, the key in her palm")
            else:
                self.assertEqual(q["description"], p["description"])

    def test_a_failing_writer_keeps_the_rules_briefs_and_stop_keeps_what_is_done(self):
        broken = director.direct(self.book, self.cid, "manga", limit=99,
                                 write=lambda *_: (_ for _ in ()).throw(RuntimeError("x")))
        self.assertEqual([p["description"] for p in broken], [p["description"] for p in self.planned])
        def stop(*_):
            raise director.Stop()
        stopped = director.direct(self.book, self.cid, "manga", write=stop, limit=99)
        self.assertEqual(len(stopped), len(self.planned))

    def test_weak_briefs_go_in_batches(self):
        calls = []
        def write(system, prompt):
            calls.append(len(json.loads(prompt.split("Panels:\n")[1])))
            return "{}"
        many = self.planned * 3
        weak = [i for i, p in enumerate(many) if p["brief_weak"]]
        from unittest.mock import patch
        with patch.object(director, "plan", return_value=many):
            director.direct(self.book, self.cid, "manga", write=write)
        self.assertEqual(sum(calls), len(weak))
        self.assertTrue(all(n <= director.REFINE_BATCH for n in calls))


class WorkerDirection(unittest.TestCase):
    def test_analysis_returns_the_directors_plan_in_the_chosen_format(self):
        import worker
        from unittest.mock import patch
        b = book()
        with patch.object(worker, '_write', side_effect=AssertionError("no writer without one named")):
            out = worker.op_direct_story('d', {'book': b, 'chapter_id': b['chapters'][0]['id'], 'format': 'picture',
                                               'pages': 99})
        self.assertEqual(len(out['panels']), len(b['chapters'][0]['scenes']))
        self.assertEqual(out['coverage']['percent'], 100)

    def test_the_page_count_reaches_the_director(self):
        import worker
        b = book()
        few = worker.op_direct_story('d', {'book': b, 'chapter_id': b['chapters'][0]['id'], 'format': 'manga', 'pages': 1})
        self.assertLessEqual(len(few['panels']), director.PANELS_PER_PAGE["manga"])
        self.assertIn('sentences', few['coverage'])


class PlaceMoves(unittest.TestCase):
    def test_a_scene_moved_into_a_named_room_is_drawn_there(self):
        self.assertEqual(director._named_place(["Her kitchen smelled of tar."], "pier"), "kitchen")
        self.assertEqual(director._named_place(["We went into the hall."], "street"), "hall")
        self.assertEqual(director._named_place(["She walked back to her house."], "pier"), "house")
        self.assertIsNone(director._named_place(["The harbour lay grey."], "pier"))
        self.assertIsNone(director._named_place(["Her kitchen was warm."], "the kitchen"))

    def test_a_place_only_mentioned_is_not_a_move(self):
        self.assertIsNone(director._named_place(["She thought of her house."], "pier"))
        self.assertIsNone(director._named_place(["I left my car at the station."], "pier"))
        self.assertIsNone(director._named_place(["Mira locked her car and ran."], "street"))

    def test_the_move_lasts_while_the_analysis_repeats_the_old_place(self):
        source = 'Agnes stood on the pier.\n\n* * *\n\nHer kitchen smelled of tar.\n\n* * *\n\nAgnes poured the tea.'
        def write(_, prompt):
            group = json.loads(prompt.split('Numbered source units:\n')[1].split('\nYour previous reply')[0])
            return json.dumps({"title": "t", "characters": [{"id": "a", "name": "Agnes", "aliases": [], "visual": "", "voice": ""}],
                               "looks": [], "places": [], "speakers": [],
                               "beats": [{"units": [u["id"] for u in group], "importance": 2, "action": "a",
                                          "place": "pier", "mood": "calm", "cast": [{"id": "Agnes", "feel": ""}]}]})
        b = story.analyze(source, write, passage_words=999, passage_units=99)
        panels = director.plan(b, b['chapters'][0]['id'], "manga", limit=99)
        last_scene = max(p["scene"] for p in panels)
        self.assertEqual({p["setting"] for p in panels if p["scene"] == last_scene}, {"kitchen"})


class HinglishNarrator(unittest.TestCase):
    SOURCE = ('Main subah office ke liye nikla. Maine Riya ko station par dekha.\n\n'
              'Mujhe laga wo naraz thi. Maine uska haath pakda aur hum chal pade.')

    def analysed(self):
        return one_beat(self.SOURCE, [{"id": "r", "name": "Riya", "aliases": [], "visual": "", "voice": ""}], ["r"],
                        importance=4, place="station")

    def test_a_story_told_with_main_and_maine_has_a_narrator_who_is_drawn(self):
        b = self.analysed()
        self.assertIn("Narrator", [c["name"] for c in b["characters"]])
        panels = director.plan(b, b['chapters'][0]['id'], "manga", limit=99)
        held = next(p for p in panels if p["moment"].startswith("Main subah"))
        self.assertIn("Narrator", held["characters"])
        saw = next(p for p in panels if p["moment"].startswith("Maine Riya ko"))
        self.assertIn("Riya", saw["characters"])

    def test_a_book_analysed_without_its_narrator_gains_one_when_planned_again(self):
        import worker
        b = self.analysed()
        old = {**b, "characters": [c for c in b["characters"] if c["name"] != "Narrator"]}
        out = worker.op_direct_story('d', {'book': old, 'chapter_id': b['chapters'][0]['id'], 'format': 'manga'})
        self.assertIn("Narrator", [c["name"] for c in out["book"]["characters"]])
        again = worker.op_direct_story('d', {'book': out["book"], 'chapter_id': b['chapters'][0]['id'], 'format': 'manga'})
        self.assertNotIn("book", again)

    def test_a_third_person_story_has_no_narrator(self):
        self.assertEqual(story._ensure_narrator([], 'Riya ne kaha, "main aa rahi hoon." Wo chali gayi.'), [])


class HinglishNarratorDeclared(unittest.TestCase):
    def test_a_narrator_the_writer_declares_in_a_hinglish_scene_is_kept_and_visible(self):
        source = 'Maine darwaza khola. Riya bahar khadi thi.'
        b = one_beat(source, [{"id": "n", "name": "Narrator", "aliases": [], "visual": "tall", "voice": ""},
                              {"id": "r", "name": "Riya", "aliases": [], "visual": "", "voice": ""}], ["n", "r"], place="door")
        narrator = next(c for c in b["characters"] if c["name"] == "Narrator")
        self.assertEqual(narrator["visual"], "tall")
        visible = {a["character_id"] for s in b["chapters"][0]["scenes"] for beat in s["beats"] for a in beat["visible"]}
        self.assertIn(narrator["id"], visible)


if __name__ == "__main__":
    unittest.main()
