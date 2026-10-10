"""Regression tests for adaptation, identity, source preservation and audio safety."""
import copy
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch
import wave
from pathlib import Path
import story
import audiobook


def reply(group, characters=None, importance=2):
    return {"title": "The scene", "summary": "A causal change in the story",
            "characters": characters or [], "beats": [{
                "units": [u["id"] for u in group], "importance": importance,
                "action": "The moment unfolds", "setting": "station", "mood": "anxious",
                "tension": 4, "visible": [], "dialogue": []}]}


def example():
    source = 'Mira gives Jonas a key. A guard calls, “Last train! The door is closing.” Jonas finds the platform empty.'
    group = story.passages(source)[0]
    data = reply(group, [{"id": "mira", "name": "Mira", "aliases": [], "visual": "", "voice": "gentle"},
                         {"id": "jonas", "name": "Jonas", "aliases": [], "visual": "", "voice": "low"},
                         {"id": "guard", "name": "A guard", "aliases": [], "visual": "", "voice": "loud"}])
    data["beats"] = []
    for i, unit in enumerate(group):
        beat = reply([unit], importance=5 if i == len(group) - 1 else 2)["beats"][0]
        beat["visible"] = [{"character_id": "jonas", "expression": "disbelief", "body_language": "frozen"}]
        if '“' in unit["text"]:
            beat["visible"] = []
            beat["dialogue"] = [{"speaker_id": "guard", "text": "Last train! The door is closing.", "offscreen": True}]
        data["beats"].append(beat)
    return story.analyze(source, lambda *_: json.dumps(data))


class StoryAnalysis(unittest.TestCase):
    def test_fx01_preserves_every_essential_causal_beat_and_the_final_reveal(self):
        root = Path(__file__).parent / 'fixtures'
        fixture = json.loads((root / 'fx01.json').read_text())
        response = (root / 'fx01.analysis.json').read_text()
        book = story.analyze(fixture['source'], lambda *_: response)
        panels = story.direct(book, book['chapters'][0]['id'])
        sources = [p['source'] for p in panels if p['source']]
        for beat in fixture['beats']:
            if beat['essential']:
                start, end = beat['span']
                self.assertTrue(any(fixture['source'][start:end] in source for source in sources), beat['id'])
        self.assertIn('The train had gone.', sources[-1])
        speech = next(p for p in panels if p['dialogue'])
        self.assertEqual(speech['dialogue'][0]['text'], 'Last train!')
        self.assertFalse(speech['character_in_frame'])
        self.assertEqual(len(book['characters']), 3)

    def test_scene_calls_cover_a_long_chapter_without_truncating_the_ending(self):
        source = 'A quiet morning. ' * 600 + 'Then the killer appeared.'
        calls = []
        def write(_, prompt):
            group = json.loads(prompt.split('Numbered source units:\n')[1].split('\nYour previous reply')[0])
            calls.append(group)
            return json.dumps(reply(group))
        book = story.analyze(source, write)
        self.assertGreater(len(calls), 60)
        chapter = book['chapters'][0]
        self.assertEqual(''.join(b['source'] for s in chapter['scenes'] for b in s['beats']), source)
        panels = story.direct(book, chapter['id'])
        self.assertGreater(len(panels), 60)
        self.assertIn('killer appeared', panels[-1]['source'])
        self.assertTrue(all(len(g) <= 8 for g in calls))

    def test_multi_sentence_dialogue_stays_in_one_source_unit(self):
        book = example()
        lines = [d for s in book['chapters'][0]['scenes'] for b in s['beats'] for d in b['dialogue']]
        self.assertEqual([d['text'] for d in lines], ['Last train! The door is closing.'])
        self.assertTrue(lines[0]['offscreen'])

    def test_alias_keeps_identity_and_appearance_across_chapters(self):
        book = example()
        jonas = next(c for c in book['characters'] if c['name'] == 'Jonas')
        jonas['aliases'] = ['Jo']
        jonas['visual'] = 'grey coat'
        source = 'Jo returned to the station.'
        data = reply(story.passages(source)[0], [{"id": "local", "name": "Jo", "aliases": [], "visual": "", "voice": ""}])
        data['beats'][0]['visible'] = [{"character_id": "local", "expression": "tired", "body_language": "slumped"}]
        old = copy.deepcopy(book)
        new = story.analyze(source, lambda *_: json.dumps(data), previous=book)
        self.assertEqual(book, old)
        self.assertEqual(len(new['characters']), len(book['characters']))
        self.assertEqual(new['chapters'][-1]['scenes'][0]['beats'][0]['visible'][0]['character_id'], jonas['id'])
        self.assertEqual(next(c for c in new['characters'] if c['id'] == jonas['id'])['visual'], 'grey coat')

    def test_incomplete_json_is_retried_then_refused_instead_of_salvaged(self):
        calls = []
        def write(*_):
            calls.append(1)
            return '{"beats": ['
        with self.assertRaisesRegex(ValueError, 'could not be validated'):
            story.analyze('The ending matters.', write)
        self.assertEqual(len(calls), 2)

    def test_missing_source_unit_and_invented_dialogue_are_rejected(self):
        source = 'Mira waited. Jonas said “Come home.”'
        group = story.passages(source)[0]
        data = reply(group)
        scene, _ = story.parse_scene(json.dumps(data), group, [], 's1')
        self.assertEqual(scene['beats'][0]['dialogue'][0]['text'], 'Come home.')
        self.assertTrue(scene['beats'][0]['dialogue'][0]['attribution_review'])
        data['beats'][0]['dialogue'] = [{"speaker_id": None, "text": "Come home.", "offscreen": True}]
        data['beats'][0]['units'] = [0]
        with self.assertRaisesRegex(ValueError, 'every source unit'):
            story.parse_scene(json.dumps(data), group, [], 's1')

    def test_source_mutation_and_overlapping_dialogue_fail_before_rendering(self):
        book = example()
        book['chapters'][0]['source'] += ' Changed.'
        with self.assertRaisesRegex(ValueError, 'source changed'):
            story.direct(book, book['chapters'][0]['id'])
        book = example()
        beat = next(b for b in book['chapters'][0]['scenes'][0]['beats'] if b['dialogue'])
        beat['dialogue'].append(copy.deepcopy(beat['dialogue'][0]))
        with self.assertRaisesRegex(ValueError, 'not overlap'):
            story.narration_segments(book, book['chapters'][0]['id'])

    def test_director_expands_the_reveal_and_keeps_offscreen_speaker_out_of_frame(self):
        book = example()
        panels = story.direct(book, book['chapters'][0]['id'])
        reveal = [p for p in panels if p['importance'] == 5]
        self.assertEqual(len(reveal), 3)
        self.assertEqual(reveal[-1]['purpose'], 'reaction')
        speech = next(p for p in panels if p['dialogue'])
        self.assertFalse(speech['character_in_frame'])
        self.assertEqual(speech['cast'], [])
        self.assertEqual(speech['dialogue'][0]['text'], 'Last train! The door is closing.')

    def test_long_dialogue_is_split_without_losing_words(self):
        text = ' '.join(f'word{i}' for i in range(90))
        source = f'Jonas says “{text}”'
        group = story.passages(source)[0]
        data = reply(group, [{"id": "j", "name": "Jonas", "aliases": [], "visual": "", "voice": ""}])
        data['beats'][0]['dialogue'] = [{"speaker_id": "j", "text": text, "offscreen": False}]
        book = story.analyze(source, lambda *_: json.dumps(data))
        panels = story.direct(book, book['chapters'][0]['id'])
        self.assertEqual(' '.join(d['text'] for p in panels for d in p['dialogue']), text)

    def test_audio_segments_reconstruct_every_character_of_original_prose(self):
        book = example()
        segments = story.narration_segments(book, book['chapters'][0]['id'])
        self.assertEqual(''.join(s['text'] for s in segments), book['chapters'][0]['source'])
        self.assertEqual(next(s for s in segments if s['speaker_id'])['text'], 'Last train! The door is closing.')

    def test_shared_pronouns_do_not_merge_two_characters(self):
        source = 'Mira saw him. Jonas saw him.'
        data = reply(story.passages(source)[0], [
            {"id": "a", "name": "Mira", "aliases": ['him'], "visual": '', "voice": ''},
            {"id": "b", "name": "Jonas", "aliases": ['him'], "visual": '', "voice": ''}])
        book = story.analyze(source, lambda *_: json.dumps(data))
        self.assertEqual(len(book['characters']), 2)
        self.assertTrue(all('him' not in c['aliases'] for c in book['characters']))

    def test_unnamed_first_person_narrator_is_supported_without_inventing_a_name(self):
        source = 'I opened the door.'
        data = reply(story.passages(source)[0], [{"id": "n", "name": "Narrator", "aliases": ['I'], "visual": '', "voice": ''}])
        book = story.analyze(source, lambda *_: json.dumps(data))
        self.assertEqual(book['characters'][0]['name'], 'Narrator')

    def test_explicit_scene_breaks_and_giant_sentences_preserve_exact_source(self):
        source = 'First scene.\n\n***\n\n' + 'word ' * 1100 + 'ending'
        groups = story.passages(source)
        self.assertEqual(''.join(u['text'] for group in groups for u in group), source)
        self.assertGreater(len(groups), 3)


class AudioSafety(unittest.TestCase):
    def wav(self):
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
            wav.writeframes(b'\x01\x00' * 240)
        return buf.getvalue()

    def test_each_character_uses_its_voice_and_output_is_valid_pcm(self):
        calls = []
        def render(text, voice, _rate, directory, _check):
            calls.append((text, voice))
            Path(directory, 'speech.txt').write_text(text)
            return self.wav()
        with tempfile.TemporaryDirectory() as stage:
            data, seconds = audiobook.synthesize([
                {'speaker_id': None, 'text': 'Narration.'}, {'speaker_id': 'j', 'text': 'Dialogue.'}],
                'Narrator', {'j': 'Jonas'}, 160, stage, lambda: None, render=render,
                available={'Narrator', 'Jonas'})
            self.assertEqual(os.listdir(stage), [])
        self.assertEqual(calls, [('Narration.', 'Narrator'), ('Dialogue.', 'Jonas')])
        with wave.open(io.BytesIO(data)) as wav:
            self.assertEqual(wav.getnframes(), 480)
        self.assertAlmostEqual(seconds, .02)

    def test_failure_and_cancellation_delete_plaintext(self):
        for exception in [RuntimeError('voice failed'), KeyboardInterrupt()]:
            def render(text, _voice, _rate, directory, _check):
                Path(directory, 'speech.txt').write_text(text)
                raise exception
            with tempfile.TemporaryDirectory() as stage:
                with self.assertRaises(type(exception)):
                    audiobook.synthesize([{'speaker_id': None, 'text': 'private'}],
                        'Voice', {}, 160, stage, lambda: None, render=render, available={'Voice'})
                self.assertEqual(os.listdir(stage), [])

    def test_unknown_voice_and_invalid_rate_do_not_start_speech(self):
        with tempfile.TemporaryDirectory() as stage:
            for voice, rate in [('missing', 160), ('Voice', 10)]:
                with self.assertRaises(ValueError):
                    audiobook.synthesize([], voice, {}, rate, stage, lambda: None, available={'Voice'})
                self.assertEqual(os.listdir(stage), [])

    def test_running_speech_process_is_terminated_on_cancellation(self):
        proc = unittest.mock.Mock()
        proc.poll.return_value = None
        def check():
            raise RuntimeError('cancelled')
        with patch('audiobook.subprocess.Popen', return_value=proc):
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                audiobook.run_process(['say'], check)
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once()

    def test_long_speech_input_is_bounded_without_losing_source_characters(self):
        text = 'private chapter ' * 500
        calls = []
        with tempfile.TemporaryDirectory() as stage:
            audiobook.synthesize([{'speaker_id': None, 'text': text}], 'Voice', {}, 160,
                stage, lambda: None, available={'Voice'}, render=lambda t, *_: calls.append(t) or self.wav())
        self.assertEqual(''.join(calls), text)
        self.assertTrue(all(len(t) <= 1801 for t in calls))


class WorkerIntegration(unittest.TestCase):
    def slot(self, directory, name):
        return {"id": name, "path": str(Path(directory, name)),
                "key": "01" * 32, "file_id": "02" * 16}

    def test_composed_reference_uses_only_requested_characters_in_order_and_is_sealed(self):
        from PIL import Image
        import worker
        import vaultcrypto as vc
        with tempfile.TemporaryDirectory() as directory:
            inputs = []
            for name, color in [('mira', 'red'), ('jonas', 'blue')]:
                slot = self.slot(directory, name)
                buf = io.BytesIO()
                Image.new('RGB', (20, 40), color).save(buf, format='PNG')
                vc.write_sealed(slot['path'], bytes.fromhex(slot['key']), bytes.fromhex(slot['file_id']), buf.getvalue())
                inputs.append(slot)
            slot = self.slot(directory, 'reference')
            result = worker.op_character_reference('test-reference', {'vault_inputs': inputs, 'vault_slots': [slot]})
            self.assertTrue(result['sealed'])
            encrypted = Path(slot['path']).read_bytes()
            self.assertFalse(encrypted.startswith(b'\x89PNG'))
            with Image.open(io.BytesIO(vc.open_with_file_key(bytes.fromhex(slot['key']), encrypted))) as image:
                self.assertEqual(image.size, (1024, 768))
                self.assertEqual(image.getpixel((256, 384)), (255, 0, 0))
                self.assertEqual(image.getpixel((768, 384)), (0, 0, 255))

    def test_audiobook_worker_evicts_models_and_seals_a_playable_wav(self):
        import worker
        import vaultcrypto as vc
        book = example()
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as audio:
            audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(24000)
            audio.writeframes(b'\x01\x00' * 240)
        with tempfile.TemporaryDirectory() as directory:
            slot = self.slot(directory, 'audio')
            with patch.object(worker.CACHE, 'unload') as release, patch.object(worker, '_unload_assistant') as release_text, \
                 patch('audiobook.synthesize', return_value=(buf.getvalue(), .01)):
                result = worker.op_audiobook('test-audio', {'book': book, 'chapter_id': book['chapters'][0]['id'],
                    'narrator': 'Voice', 'voices': {}, 'rate': 160, 'vault_slots': [slot]})
            release.assert_called_once(); release_text.assert_called_once()
            self.assertTrue(result['sealed'])
            sealed = Path(slot['path']).read_bytes()
            self.assertNotIn(b'RIFF', sealed[:4])
            with wave.open(io.BytesIO(vc.open_with_file_key(bytes.fromhex(slot['key']), sealed))) as audio:
                self.assertEqual(audio.getnframes(), 240)

    def test_neural_worker_routes_directions_and_seals_output_without_system_speech(self):
        import worker
        import vaultcrypto as vc
        book = example()
        with tempfile.TemporaryDirectory() as directory:
            slot = self.slot(directory, 'neural')
            with patch.object(worker.CACHE, 'unload'), patch.object(worker, '_unload_assistant'), \
                 patch.object(worker, '_apply_policy_once'), patch('audiobook.synthesize') as system, \
                 patch('audiobook.synthesize_neural', return_value=(b'neural audio bytes', .01)) as neural:
                result = worker.op_audiobook('test-neural', {'book': book, 'chapter_id': book['chapters'][0]['id'],
                    'model_id': audiobook.NEURAL_MODEL_ID, 'narrator': 'ryan', 'voices': {}, 'rate': 160,
                    'language': 'english', 'delivery': 'Warm', 'directions': {'mira': 'Quiet'}, 'vault_slots': [slot]})
            system.assert_not_called()
            self.assertEqual(neural.call_args.kwargs['language'], 'english')
            self.assertEqual(neural.call_args.kwargs['delivery'], 'Warm')
            self.assertEqual(neural.call_args.kwargs['directions'], {'mira': 'Quiet'})
            self.assertTrue(result['sealed'])
            self.assertEqual(vc.open_with_file_key(bytes.fromhex(slot['key']), Path(slot['path']).read_bytes()), b'neural audio bytes')

    def test_audio_failure_never_writes_output(self):
        import worker
        book = example()
        with tempfile.TemporaryDirectory() as directory:
            slot = self.slot(directory, 'failed')
            with patch.object(worker.CACHE, 'unload'), patch.object(worker, '_unload_assistant'), \
                 patch('audiobook.synthesize', side_effect=RuntimeError('speech failed')):
                with self.assertRaisesRegex(RuntimeError, 'speech failed'):
                    worker.op_audiobook('test-audio', {'book': book, 'chapter_id': book['chapters'][0]['id'],
                        'narrator': 'Voice', 'vault_slots': [slot]})
            self.assertFalse(Path(slot['path']).exists())

    def test_surface_enrichment_cannot_erase_action_or_emotional_direction(self):
        import worker
        book = example()
        panel = story.direct(book, book['chapters'][0]['id'])[-1]
        with patch.object(worker, '_write', return_value='{"lighting":"cold light","surfaces":"stone"}'):
            result = worker.op_enrich_panels('test-enrichment', {'panels': [panel]})['panels'][0]
        for field in ('action', 'cast', 'mood', 'tension', 'dialogue', 'caption'):
            self.assertEqual(result[field], panel[field])
        self.assertIn('cold light', result['description'])

    def test_switching_analysts_unloads_the_previous_writer_before_loading(self):
        import sys
        import types
        import worker
        events = []
        module = types.ModuleType('mlx_lm')
        module.load = lambda repo: events.append(('load', repo)) or ('new-model', 'tokenizer')
        old = copy.copy(worker._WRITER)
        worker._WRITER.update(key='old', model='old-model', tokenizer='old-tokenizer')
        def unload():
            events.append(('unload', 'old'))
            worker._WRITER.update(key=None, model=None, tokenizer=None)
        try:
            with patch.dict(sys.modules, {'mlx_lm': module}), patch.object(worker, '_unload_writer', side_effect=unload), \
                 patch.object(worker, '_apply_policy_once'), patch.object(worker.CACHE, 'unload'):
                worker._load_writer('test-switch', 'new-repo')
            self.assertEqual(events[:2], [('unload', 'old'), ('load', 'new-repo')])
        finally:
            worker._WRITER.update(old)


if __name__ == '__main__':
    unittest.main()


class AppearanceContinuity(unittest.TestCase):
    def test_outfit_replaces_previous_state_without_changing_identity(self):
        source = "Mira wears a red dress. Mira changes into a blue dress."
        calls = []
        def write(system, prompt):
            calls.append(prompt)
            group = story.passages(source)[0]
            data = reply(group, [{"id": "mira", "name": "Mira", "aliases": [],
                                  "visual": "curly black hair", "voice": ""}])
            data["beats"] = []
            for unit, outfit in zip(group, ["red dress", "blue dress"]):
                beat = reply([unit])["beats"][0]
                beat.update(location_id="bedroom", location_state="window left; daylight")
                beat["visible"] = [{"character_id": "mira", "expression": "calm",
                                    "body_language": "standing", "appearance": outfit}]
                data["beats"].append(beat)
            return json.dumps(data)
        book = story.analyze(source, write)
        panels = story.direct(book, book["chapters"][0]["id"])
        self.assertEqual([p["cast"][0]["appearance"] for p in panels], ["red dress", "blue dress"])
        self.assertEqual(book["characters"][0]["visual"], "curly black hair")
        self.assertTrue(all(p["place_key"] == "bedroom" for p in panels))
        state = {"characters": {}, "locations": {}}
        for beat in book["chapters"][0]["scenes"][0]["beats"]:
            story._remember_state(state, beat)
        self.assertEqual(list(state["characters"].values()), ["blue dress"])
        self.assertIn("blue dress", story.instruction(story.passages("Mira leaves.")[0], book["characters"], state))


class NeuralNarration(unittest.TestCase):
    def model(self, calls, empty=False, invalid=False):
        from types import SimpleNamespace
        class Model:
            def get_supported_speakers(self): return ["ryan", "vivian"]
            def generate_custom_voice(self, **kwargs):
                calls.append(kwargs)
                if not empty:
                    yield SimpleNamespace(audio=[0.0, 0.5, -0.5] if not invalid else [float('nan')],
                                          sample_rate=24000, token_count=3)
        return Model()

    def test_voice_direction_streaming_and_pcm_output(self):
        calls = []
        data, seconds = audiobook.synthesize_neural([
            {"speaker_id": None, "text": "The train waits."},
            {"speaker_id": "mira", "text": "Don't leave."}], "ryan", {"mira": "vivian"}, 160,
            lambda: None, language="english", delivery="Warm narration", directions={"mira": "Whisper"},
            load=lambda: self.model(calls))
        self.assertEqual([c["speaker"] for c in calls], ["ryan", "vivian"])
        self.assertIn("Whisper", calls[1]["instruct"])
        self.assertTrue(all(c["stream"] for c in calls))
        with wave.open(io.BytesIO(data), 'rb') as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()), (1, 2, 24000, 6))
        self.assertEqual(seconds, 6 / 24000)

    def test_preserves_text_across_bounded_chunks(self):
        text = ('  A sentence! Then another sentence.\n' * 100) + 'tail'
        parts = list(audiobook.speech_chunks(text))
        self.assertEqual(''.join(parts), text)
        self.assertTrue(all(len(p) <= 350 for p in parts))

    def test_rejects_invalid_voices_before_loading(self):
        load = unittest.mock.Mock()
        with self.assertRaises(ValueError):
            audiobook.synthesize_neural([], "Samantha", {}, 160, lambda: None, load=load)
        load.assert_not_called()

    def test_refuses_empty_and_invalid_audio(self):
        for mode in ("empty", "invalid"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                audiobook.synthesize_neural([{"speaker_id": None, "text": "Hello"}], "ryan", {}, 160,
                    lambda: None, load=lambda: self.model([], empty=mode == "empty", invalid=mode == "invalid"))

    def test_cancels_between_streaming_chunks(self):
        calls, checks = [], []
        def check():
            checks.append(1)
            if len(checks) == 5: raise RuntimeError("cancelled")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            audiobook.synthesize_neural([{"speaker_id": None, "text": "Hello"}], "ryan", {}, 160,
                check, load=lambda: self.model(calls))
        self.assertEqual(len(calls), 1)


class CharacterReferenceBindings(unittest.TestCase):
    def data(self):
        source = 'Mira says, “Wait.”'
        group = story.passages(source)[0]
        data = reply(group, [{"id": "c1", "name": "Mira", "aliases": [], "visual": "", "voice": ""}])
        data["beats"][0]["visible"] = [{"character_id": "Mira", "expression": "worried", "body_language": "still"}]
        data["beats"][0]["dialogue"] = [{"speaker_id": "Mira", "text": "Wait.", "offscreen": False}]
        return source, group, data

    def test_names_resolve_to_the_same_canonical_id_as_local_ids(self):
        source, group, data = self.data()
        book = story.analyze(source, lambda *_: json.dumps(data))
        cid = book['characters'][0]['id']
        beat = book['chapters'][0]['scenes'][0]['beats'][0]
        self.assertEqual(beat['visible'][0]['character_id'], cid)
        self.assertEqual(beat['dialogue'][0]['speaker_id'], cid)

    def test_prior_alias_resolves_without_redeclaring_the_character(self):
        source, group, data = self.data()
        bible = [{'id': 'uuid-mira', 'name': 'Mira', 'aliases': ['Mi'], 'visual': '', 'voice': ''}]
        data['characters'] = []
        data['beats'][0]['visible'][0]['character_id'] = ' mi '
        data['beats'][0]['dialogue'][0]['speaker_id'] = 'uuid-mira'
        scene, characters = story.parse_scene(json.dumps(data), group, bible, 's1')
        self.assertEqual(scene['beats'][0]['visible'][0]['character_id'], 'uuid-mira')
        self.assertEqual(characters, bible)

    def test_undeclared_ids_are_rejected_with_repair_information(self):
        source, group, data = self.data()
        data['beats'][0]['visible'][0]['character_id'] = 'c9'
        with self.assertRaisesRegex(ValueError, "c9.*not declared.*c1=Mira"):
            story.parse_scene(json.dumps(data), group, [], 's1')

    def test_ambiguous_aliases_do_not_merge_two_characters(self):
        source, group, data = self.data()
        bible = [{'id': name.lower(), 'name': name, 'aliases': ['Captain'], 'visual': '', 'voice': ''}
                 for name in ['Mira', 'Jonas']]
        data['characters'] = []
        data['beats'][0]['visible'][0]['character_id'] = 'Captain'
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            story.parse_scene(json.dumps(data), group, bible, 's1')

    def test_retry_includes_the_failed_reply_and_valid_ids(self):
        source, group, valid = self.data()
        broken = copy.deepcopy(valid)
        broken['beats'][0]['visible'][0]['character_id'] = 'missing-id'
        prompts = []
        def write(system, prompt):
            prompts.append(prompt)
            return json.dumps(broken if len(prompts) == 1 else valid)
        book = story.analyze(source, write)
        self.assertEqual(len(prompts), 2)
        self.assertIn(json.dumps(broken), prompts[1])
        self.assertIn('c1=Mira', prompts[1])
        self.assertEqual(len(book['characters']), 1)


class SmallAnalystPassages(unittest.TestCase):
    def test_smaller_calls_preserve_full_source_and_ending(self):
        source = 'The train waits. ' * 15 + 'The final door closes.'
        groups = story.passages(source, max_words=250, max_units=4)
        responses = iter(json.dumps(reply(group)) for group in groups)
        book = story.analyze(source, lambda *_: next(responses), passage_words=250, passage_units=4)
        chapter = book['chapters'][0]
        self.assertEqual(len(chapter['scenes']), len(groups))
        self.assertEqual(''.join(beat['source'] for scene in chapter['scenes'] for beat in scene['beats']), source)
        self.assertTrue(chapter['scenes'][-1]['beats'][-1]['source'].endswith('The final door closes.'))


class NarratorBindings(unittest.TestCase):
    def analyze(self, source, characters, visible=None, previous=None):
        data = reply(story.passages(source)[0], characters)
        data['beats'][0]['visible'] = visible or []
        return story.analyze(source, lambda *_: json.dumps(data), previous=previous)

    def person(self, name='narrator', identifier='n', aliases=None):
        return {'id': identifier, 'name': name, 'aliases': aliases or [], 'visual': '', 'voice': ''}

    def actor(self, identifier='n'):
        return {'character_id': identifier, 'expression': 'calm', 'body_language': 'standing'}

    def test_lowercase_first_person_narrator_is_valid_and_canonical(self):
        book = self.analyze('I opened the door.', [self.person()], [self.actor()])
        self.assertEqual(book['characters'][0]['name'], 'Narrator')
        self.assertIn('I', book['characters'][0]['aliases'])
        self.assertEqual(book['chapters'][0]['scenes'][0]['beats'][0]['visible'][0]['character_id'], book['characters'][0]['id'])

    def test_known_named_narrator_is_reused_instead_of_rejected(self):
        previous = self.analyze('My name is Mira Vale.', [self.person('Mira Vale', 'm')])
        book = self.analyze('I opened the door.', [self.person('NARRATOR')], [self.actor()], previous)
        self.assertEqual(len(book['characters']), 1)
        self.assertEqual(book['characters'][0]['id'], previous['characters'][0]['id'])
        self.assertEqual(book['characters'][0]['name'], 'Mira Vale')

    def test_named_narrator_and_role_do_not_duplicate_in_the_same_reply(self):
        book = self.analyze('My name is Mira Vale. I opened the door.',
            [self.person(), self.person('Mira Vale', 'm')], [self.actor()])
        self.assertEqual(len(book['characters']), 1)
        self.assertEqual(book['characters'][0]['name'], 'Mira Vale')

    def test_third_person_narration_role_does_not_add_a_visible_person(self):
        book = self.analyze('Mira opened the door.', [self.person(), self.person('Mira', 'm')],
            [self.actor(), self.actor('m')])
        self.assertEqual([c['name'] for c in book['characters']], ['Mira'])
        self.assertEqual(len(book['chapters'][0]['scenes'][0]['beats'][0]['visible']), 1)

    def test_anonymous_narrator_can_gain_the_stated_name_in_a_later_chapter(self):
        previous = self.analyze('I opened the door.', [self.person()])
        book = self.analyze('My name is Mira Vale.', [self.person('Mira Vale', 'm', ['I'])], previous=previous)
        self.assertEqual(len(book['characters']), 1)
        self.assertEqual(book['characters'][0]['name'], 'Mira Vale')
        self.assertEqual(book['characters'][0]['id'], previous['characters'][0]['id'])

    def test_unrelated_invented_character_still_fails_validation(self):
        with self.assertRaisesRegex(ValueError, 'Invented.*not in this scene'):
            self.analyze('Mira opened the door.', [self.person('Invented', 'x')])


class SourceDialogueExtraction(unittest.TestCase):
    def parse(self, source, lines):
        group = story.passages(source)[0]
        data = reply(group, [{'id': 'm', 'name': 'Mira', 'aliases': [], 'visual': '', 'voice': ''}])
        data['beats'][0]['dialogue'] = lines
        scene, characters = story.parse_scene(json.dumps(data), group, [], 's')
        return scene['beats'][0]['dialogue'], characters

    def test_missing_and_paraphrased_dialogue_is_preserved_from_source(self):
        lines, _ = self.parse('Mira said, “Wait for me.” Then “Come back.”',
            [{'text': 'Please wait!', 'speaker_id': 'm', 'offscreen': False}])
        self.assertEqual([d['text'] for d in lines], ['Wait for me.', 'Come back.'])
        self.assertTrue(all(d['speaker_id'] is None and d['attribution_review'] for d in lines))

    def test_wrapped_quotes_use_exact_source_spans_and_valid_attribution(self):
        source = 'Mira said, “  Wait for me.  ”'
        lines, characters = self.parse(source, [{'text': '“Wait for me.”', 'speaker_id': 'Mira', 'offscreen': False}])
        line = lines[0]
        self.assertEqual(source[line['start']:line['end']], 'Wait for me.')
        self.assertEqual(line['speaker_id'], characters[0]['id'])
        self.assertFalse(line['attribution_review'])

    def test_invented_lines_are_not_added_and_unknown_speakers_require_review(self):
        lines, _ = self.parse('Mira said, “Wait.”', [
            {'text': 'Invented words', 'speaker_id': 'm', 'offscreen': False},
            {'text': 'Wait.', 'speaker_id': 'missing', 'offscreen': False}])
        self.assertEqual([d['text'] for d in lines], ['Wait.'])
        self.assertIsNone(lines[0]['speaker_id'])
        self.assertTrue(lines[0]['attribution_review'])

    def test_repeated_utterances_keep_distinct_spans(self):
        source = 'Mira said, “Wait.” Mira repeated, “Wait.”'
        lines, _ = self.parse(source, [{'text': 'Wait.', 'speaker_id': 'm', 'offscreen': False}])
        self.assertEqual(len(lines), 2)
        self.assertNotEqual(lines[0]['start'], lines[1]['start'])
        self.assertFalse(lines[0]['attribution_review'])
        self.assertTrue(lines[1]['attribution_review'])
