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

    def test_missing_source_unit_is_rejected_and_invented_dialogue_never_enters(self):
        source = 'Mira waited. Jonas said “Come home.”'
        group = story.passages(source)[0]
        data = reply(group)
        data['beats'][0]['units'] = [0]
        with self.assertRaisesRegex(ValueError, 'every source unit'):
            story.parse_scene(json.dumps(data), group, [], 's1')
        data = reply(group)
        data['beats'][0]['dialogue'] = [{"speaker_id": None, "text": "I never said this.", "offscreen": True}]
        scene, _ = story.parse_scene(json.dumps(data), group, [], 's1')
        self.assertEqual([d['text'] for d in scene['beats'][0]['dialogue']], ['Come home.'])

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

    def test_writer_tokens_keep_the_scene_progress_and_a_repair_gets_more_room(self):
        import worker
        source = 'Mira ran. Jonas waited.'
        calls = []
        def write(req_id, system, user, **kw):
            calls.append(kw)
            group = json.loads(user.split('Numbered source units:\n')[1].split('\nYour previous reply')[0])
            return '{' if len(calls) == 1 else json.dumps(reply(group))
        with patch.object(worker, '_write', side_effect=write), \
                patch.object(story, 'passages', lambda s, **_: [[{**u, 'id': 0}] for u in story.units(s)]):
            worker.op_story_analyze('test-progress', {'source': source})
        self.assertEqual([(c['label'].split(' · ')[0], c['fraction'], c['max_tokens']) for c in calls],
                         [('Understanding scene 1 of 2', 0.0, 1800), ('Understanding scene 1 of 2', 0.0, 2700),
                          ('Understanding scene 2 of 2', 0.5, 1800)])

    def test_a_hand_added_panel_keeps_the_board_on_the_story_path(self):
        import worker
        book = example()
        panels = story.direct(book, book['chapters'][0]['id'])
        plain = {"shot": "medium", "action": "Jonas sits down", "setting": "station"}
        with patch.object(worker, '_write', return_value='{"lighting":"cold light","surfaces":"stone"}'):
            result = worker.op_enrich_panels('test-enrichment', {'panels': panels + [plain]})['panels']
        self.assertEqual([p.get('place_key') for p in result[:-1]], [p['place_key'] for p in panels])
        self.assertIn('cold light', result[-1]['description'])

    def test_lighting_is_written_once_per_place_and_mood(self):
        import worker
        book = example()
        panels = story.direct(book, book['chapters'][0]['id'])
        places = {(p['place_key'], p['mood']) for p in panels}
        with patch.object(worker, '_write', return_value='{"lighting":"cold light","surfaces":"stone"}') as write:
            result = worker.op_enrich_panels('test-enrichment', {'panels': panels})['panels']
        self.assertEqual(write.call_count, len(places))
        self.assertLess(len(places), len(panels))
        self.assertTrue(all('cold light' in p['description'] for p in result))

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
        self.assertIn("blue dress", story.compact_instruction(story.passages("Mira leaves.")[0], book["characters"], state))


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

    def test_chunks_end_at_sentences_not_mid_sentence(self):
        text = 'The rain fell all night on the old harbor town. ' * 12
        chunks = list(audiobook.speech_chunks(text))
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(c.rstrip().endswith('.') for c in chunks), chunks)
        japanese = '雨が一晩中降り続いた。' * 50
        chunks = list(audiobook.speech_chunks(japanese))
        self.assertEqual(''.join(chunks), japanese)
        self.assertTrue(all(c.endswith('。') for c in chunks))

    def test_progress_advances_within_one_long_narration(self):
        calls, seen = [], []
        text = 'The rain fell all night on the old harbor town. ' * 20
        audiobook.synthesize_neural([{"speaker_id": None, "text": text}], "ryan", {}, 160,
                                    lambda: None, lambda done, total: seen.append(done / total),
                                    load=lambda: self.model(calls))
        self.assertGreater(len(seen), 2)
        self.assertEqual(seen, sorted(seen))
        self.assertLess(seen[-1], 1)

    def test_preview_returns_a_playable_sample_and_keeps_the_model_only_between_previews(self):
        import base64
        import worker
        calls, loads = [], []
        def load():
            loads.append(1)
            return self.model(calls)
        worker._VOICE.update(model=None)
        req = {"model_id": audiobook.NEURAL_MODEL_ID, "voice": "vivian", "text": "Wait for me.",
               "direction": "breathless", "rate": 160}
        with patch.object(audiobook, 'load_neural_model', load):
            first = worker.op_audio_preview('test-preview', req)
            worker.op_audio_preview('test-preview', {**req, "voice": "ryan"})
        with wave.open(io.BytesIO(base64.b64decode(first['wav_base64']))) as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels()), (24000, 1))
            self.assertGreater(wav.getnframes(), 0)
        self.assertEqual(len(loads), 1)
        self.assertEqual([c['speaker'] for c in calls], ['vivian', 'ryan'])
        self.assertIn('breathless', calls[0]['instruct'])
        with patch.object(worker, 'emit'):
            worker.handle({"id": "other", "op": "ping"})
            self.assertIsNotNone(worker._VOICE['model'])
            worker.handle({"id": "other", "op": "unload"})
        self.assertIsNone(worker._VOICE['model'])

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


class NarratorIdentity(unittest.TestCase):
    """The in-app failure 'character narrator is not in this scene'."""

    def scene(self, source, name, bible=(), visible=True, speaker=False):
        group = story.passages(source)[0]
        data = reply(group, [{"id": "n", "name": name, "aliases": [], "visual": "", "voice": ""}])
        data['beats'][0]['visible'] = [{"character_id": "n", "expression": "", "body_language": ""}] if visible else []
        if speaker:
            data['beats'][0]['dialogue'] = [{"speaker_id": "n", "text": "Go.", "offscreen": False}]
        return story.parse_scene(json.dumps(data), group, list(bible), 's1')

    def test_narrator_name_is_case_and_article_insensitive(self):
        for name in ('narrator', 'The Narrator', 'NARRATOR'):
            _, characters = self.scene('I walked to the station.', name)
            self.assertEqual([c['name'] for c in characters], ['Narrator'])

    def test_known_narrator_is_reused_in_passages_without_first_person(self):
        _, bible = self.scene('I saw the train leave. I was alone.', 'Narrator')
        for source in ('I ran home.', 'The station was empty. Rain fell.'):
            scene, characters = self.scene(source, 'narrator', bible)
            self.assertEqual(len(characters), 1)
            self.assertEqual(scene['beats'][0]['visible'][0]['character_id'], bible[0]['id'])

    def test_named_first_person_narrator_absorbs_a_generic_narrator(self):
        bible = [{"id": "ellen", "name": "Ellen", "aliases": ["I"], "visual": "", "voice": ""}]
        scene, characters = self.scene('I ran home.', 'Narrator', bible)
        self.assertEqual(characters, bible)
        self.assertEqual(scene['beats'][0]['visible'][0]['character_id'], 'ellen')

    def test_third_person_narrating_voice_is_not_a_character(self):
        scene, characters = self.scene('Mira said “Go.”', 'Narrator', speaker=True)
        self.assertEqual(characters, [])
        self.assertEqual(scene['beats'][0]['visible'], [])
        self.assertIsNone(scene['beats'][0]['dialogue'][0]['speaker_id'])

    def test_undeclared_narrator_reference_is_the_narrating_voice(self):
        group = story.passages('The station was empty.')[0]
        data = reply(group)
        data['beats'][0]['visible'] = [{"character_id": "narrator", "expression": "", "body_language": ""}]
        scene, _ = story.parse_scene(json.dumps(data), group, [], 's1')
        self.assertEqual(scene['beats'][0]['visible'], [])


class DialogueFromSource(unittest.TestCase):
    """The in-app failure 'dialogue must include every complete quoted utterance'."""

    def lines(self, source, said):
        group = story.passages(source)[0]
        data = reply(group, [{"id": "m", "name": "Mira", "aliases": [], "visual": "", "voice": ""}])
        data['beats'][0]['dialogue'] = [{"speaker_id": "m", "text": t, "offscreen": False} for t in said]
        scene, _ = story.parse_scene(json.dumps(data), group, [], 's1')
        mira = scene['beats'][0]
        return [(d['text'], d['speaker_id'] is not None, source[d['start']:d['end']] == d['text'])
                for d in mira['dialogue']]

    def test_retyped_quotes_are_attributed_but_text_comes_from_source(self):
        for said in (['Go home'], ['“Go home.”'], ['go home.']):
            self.assertEqual(self.lines('Mira said, “Go home.”', said), [('Go home.', True, True)])

    def test_a_skipped_quote_is_kept_without_a_guessed_speaker(self):
        self.assertEqual(self.lines('Mira read the sign: “Closed.” She said, “Go home.”', ['Go home.']),
                         [('Closed.', False, True), ('Go home.', True, True)])

    def test_one_reply_line_attributes_an_interrupted_utterance(self):
        self.assertEqual(self.lines('Mira said, “Go,” and then, “home.”', ['Go, home.']),
                         [('Go,', True, True), ('home.', True, True)])

    def test_continued_speech_across_paragraphs_is_dialogue(self):
        for q in ('“', '"'):
            close = '”' if q == '“' else '"'
            source = f'Mira said, {q}Go home.\n\n{q}Now, please.{close} Mira wept.'
            self.assertEqual(self.lines(source, ['Go home.', 'Now, please.']),
                             [('Go home.', True, True), ('Now, please.', True, True)])

    def test_whole_book_still_validates_and_reconstructs_for_audio(self):
        source = 'Mira read the sign: “Closed.” She said, “Go,” and then, “home.”'
        data = reply(story.passages(source)[0], [{"id": "m", "name": "Mira", "aliases": [], "visual": "", "voice": ""}])
        data['beats'][0]['dialogue'] = [{"speaker_id": "m", "text": "Go home", "offscreen": False}]
        book = story.analyze(source, lambda *_: json.dumps(data))
        segments = story.narration_segments(book, book['chapters'][0]['id'])
        self.assertEqual(''.join(s['text'] for s in segments), source)


class LongChapterRobustness(unittest.TestCase):
    def test_a_placeholder_visual_is_no_description(self):
        # Qwen3.5 answers "none defined" for someone the story never describes.
        source = 'Jonas waited.'
        for placeholder in ('none', 'None defined.', 'not described', 'N/A'):
            data = reply(story.passages(source)[0], [{"id": "j", "name": "Jonas", "aliases": [],
                                                      "visual": placeholder, "voice": ""}])
            book = story.analyze(source, lambda *_: json.dumps(data))
            self.assertEqual(book['characters'][0]['visual'], '', placeholder)

    def test_character_visual_stays_bounded_across_many_scenes(self):
        source = ' '.join(f'Mira walked {i}.' for i in range(90))
        def write(_, prompt):
            group = json.loads(prompt.split('Numbered source units:\n')[1].split('\nYour previous reply')[0])
            n = group[0]['text'].split()[-1]
            return json.dumps(reply(group, [{"id": "m", "name": "Mira", "aliases": [], "voice": "",
                                             "visual": f"Tall, grey hair. A scar marks her chin. Detail {n}"}]))
        book = story.analyze(source, write, passage_words=10, passage_units=1)
        mira = book['characters'][0]
        self.assertLessEqual(len(mira['visual']), story.VISUAL_LIMIT)
        self.assertEqual(mira['visual'].count('grey hair'), 1)

    def test_saved_book_with_an_oversized_visual_still_opens(self):
        book = example()
        book['characters'][0]['visual'] = 'x' * 9000
        panels = story.direct(book, book['chapters'][0]['id'])
        self.assertTrue(all(len(a['visual']) <= story.VISUAL_LIMIT for p in panels for a in p['cast']))

    def test_unattributed_dialogue_has_no_invented_speaker_label(self):
        source = 'A voice called “Wait!”'
        data = reply(story.passages(source)[0])
        data['beats'][0]['dialogue'] = [{"speaker_id": None, "text": "Wait!", "offscreen": True}]
        book = story.analyze(source, lambda *_: json.dumps(data))
        lines = [d for p in story.direct(book, book['chapters'][0]['id']) for d in p['dialogue']]
        self.assertEqual([(d['speaker'], d['text']) for d in lines], [('', 'Wait!')])

    def test_one_failing_scene_keeps_the_rest_of_the_chapter(self):
        source = 'Mira ran. Jonas said “Stop.” The train left. Silence followed.'
        warnings = []
        def write(_, prompt):
            group = json.loads(prompt.split('Numbered source units:\n')[1].split('\nYour previous reply')[0])
            return '{"broken": ' if 'Jonas' in group[0]['text'] else json.dumps(reply(group))
        book = story.analyze(source, write, passage_words=3, passage_units=1, warn=warnings.append)
        scenes = book['chapters'][0]['scenes']
        self.assertEqual([bool(s.get('unanalysed')) for s in scenes], [False, True, False, False])
        self.assertEqual(scenes[1]['beats'][0]['dialogue'][0]['text'], 'Stop.')
        self.assertEqual(len(warnings), 1)
        segments = story.narration_segments(book, book['chapters'][0]['id'])
        self.assertEqual(''.join(s['text'] for s in segments), source)

    def test_a_writer_that_never_validates_is_still_refused(self):
        with self.assertRaisesRegex(ValueError, 'Scene 3 could not be validated'):
            story.analyze('One. Two. Three. Four.', lambda *_: '{', passage_words=1, passage_units=1)


class NarrationMood(unittest.TestCase):
    def test_narration_carries_its_beat_mood_and_still_reconstructs_the_source(self):
        book = example()
        chapter = book['chapters'][0]
        segments = story.narration_segments(book, chapter['id'])
        self.assertEqual(''.join(s['text'] for s in segments), chapter['source'])
        self.assertTrue(all(s.get('mood') for s in segments if s['speaker_id'] is None))


class PassageGrouping(unittest.TestCase):
    def test_paragraphs_share_a_call_but_scene_breaks_end_one(self):
        source = '“Hello,” she said.\n\n“Hi.”\n\n“Tea?”\n\n* * *\n\nThe kitchen was cold.'
        groups = story.passages(source, max_words=250, max_units=4)
        self.assertEqual(len(groups), 2)
        self.assertEqual(''.join(u['text'] for g in groups for u in g), source)
        self.assertIn('* * *', groups[1][0]['text'])


class VoiceAnalysis(unittest.TestCase):
    """Audiobook-only analysis: speakers and mood, one short reply per long passage."""
    SOURCE = ('I waited by the pier. Agnes Thorne said, “You’re late.”\n\n'
              '“The ferry was early,” I said.\n\nShe laughed. “Never in forty years.”')

    def test_quotes_are_numbered_in_the_passage_given_to_the_writer(self):
        group = story.passages(self.SOURCE, max_words=700, max_units=60)[0]
        prompt = story.voice_instruction(group, [])
        self.assertIn('said, [1]“You’re late.”', prompt)
        self.assertIn('[2]“The ferry was early,”', prompt)
        self.assertIn('[3]“Never in forty years.”', prompt)

    def test_a_short_reply_yields_a_valid_attributed_book(self):
        reply = {"mood": "wry", "characters": [{"id": "c1", "name": "Narrator", "aliases": [], "voice": "dry"},
                                               {"id": "c2", "name": "Agnes Thorne", "aliases": [], "voice": "gruff"}],
                 "speakers": [{"quote": 1, "speaker_id": "c2"}, {"quote": 2, "speaker_id": "c1"},
                              {"quote": 3, "speaker_id": "c2"}]}
        calls = []
        def write(system, prompt):
            calls.append(system)
            return json.dumps(reply)
        book = story.analyze(self.SOURCE, write, mode="voice", passage_words=700, passage_units=60)
        chapter = book['chapters'][0]
        self.assertEqual(chapter['analysis'], 'voice')
        self.assertEqual(calls, [story.VOICE_SYSTEM])
        people = {c['id']: c['name'] for c in book['characters']}
        lines = [(people[d['speaker_id']], d['text']) for s in chapter['scenes'] for b in s['beats'] for d in b['dialogue']]
        self.assertEqual(lines, [('Agnes Thorne', 'You’re late.'), ('Narrator', 'The ferry was early,'),
                                 ('Agnes Thorne', 'Never in forty years.')])
        segments = story.narration_segments(book, chapter['id'])
        self.assertEqual(''.join(s['text'] for s in segments), self.SOURCE)
        self.assertEqual({s['mood'] for s in segments}, {'wry'})

    def test_an_undeclared_speaker_is_repaired_and_untagged_quotes_stay_unattributed(self):
        replies = iter([{"mood": "", "characters": [], "speakers": [{"quote": 1, "speaker_id": "ghost"}]},
                        {"mood": "", "characters": [{"id": "c1", "name": "Agnes Thorne", "aliases": [], "voice": ""}],
                         "speakers": [{"quote": 1, "speaker_id": "c1"}]}])
        book = story.analyze(self.SOURCE, lambda *_: json.dumps(next(replies)), mode="voice",
                             passage_words=700, passage_units=60)
        lines = [d['speaker_id'] is not None for s in book['chapters'][0]['scenes'] for b in s['beats'] for d in b['dialogue']]
        # The second line is tagged "I said": a Narrator is added for it.
        self.assertEqual(lines, [True, True, False])
        self.assertIn('Narrator', [c['name'] for c in book['characters']])

    def test_worker_casts_voices_in_long_passages_with_a_small_budget(self):
        import worker
        source = ' '.join(f'Line {i} was spoken.' for i in range(120))
        budgets = []
        def write(req_id, system, user, **kw):
            budgets.append(kw['max_tokens'])
            return json.dumps({"mood": "calm", "characters": [], "speakers": []})
        with patch.object(worker, '_write', side_effect=write):
            result = worker.op_story_analyze('test-voice', {'source': source, 'mode': 'voice'})
        # 120 sentences: two voice calls, where full analysis makes fifteen.
        self.assertEqual(budgets, [1000, 1000])
        self.assertEqual(len(story.passages(source, max_words=450, max_units=8)), 15)
        self.assertEqual(result['book']['chapters'][0]['analysis'], 'voice')


class CompactBoardAnalysis(unittest.TestCase):
    SOURCE = ('Agnes Thorne waited in her kitchen in a grey apron. “You’re late,” she said.\n\n'
              '“The ferry was early,” I said. Tam poured tea without a word.')

    def reply(self):
        return {"title": "Arrival",
                "characters": [{"id": "c1", "name": "Agnes Thorne", "aliases": [], "visual": "broad", "voice": "gruff"},
                               {"id": "c2", "name": "Narrator", "aliases": [], "visual": "", "voice": ""},
                               {"id": "c3", "name": "Tam", "aliases": [], "visual": "boy", "voice": ""}],
                "looks": [{"id": "c1", "appearance": "grey apron"}],
                "places": [{"place": "Agnes's kitchen", "state": "tar and oranges, storm outside"}],
                "beats": [{"units": [0, 1], "importance": 3, "action": "Agnes scolds", "place": "Agnes's kitchen",
                           "mood": "wry", "cast": [{"id": "c1", "feel": "arms folded"}]},
                          {"units": [2, 3], "importance": 2, "action": "Tam pours tea", "place": "Agnes's kitchen",
                           "mood": "quiet", "cast": [{"id": "c3", "feel": "eyes down"}]}],
                "speakers": [{"quote": 1, "id": "c1"}, {"quote": 2, "id": "c2"}]}

    def test_quotes_are_numbered_inside_their_units(self):
        group = story.passages(self.SOURCE)[0]
        prompt = story.compact_instruction(group, [])
        units = json.loads(prompt.split('Numbered source units:\n')[1])
        self.assertIn('[1]“You’re late,”', units[1]['text'])
        self.assertIn('[2]“The ferry was early,”', ''.join(u['text'] for u in units))

    def test_compact_reply_becomes_a_full_valid_scene(self):
        book = story.analyze(self.SOURCE, lambda *_: json.dumps(self.reply()))
        chapter = book['chapters'][0]
        people = {c['id']: c['name'] for c in book['characters']}
        beats = chapter['scenes'][0]['beats']
        self.assertEqual([[(people[d['speaker_id']], d['text'], d['offscreen']) for d in b['dialogue']] for b in beats],
                         [[('Agnes Thorne', 'You’re late,', False)], [('Narrator', 'The ferry was early,', True)]])
        self.assertEqual(beats[0]['visible'][0]['appearance'], 'grey apron')
        self.assertEqual(beats[1]['location_state'], 'tar and oranges, storm outside')
        self.assertEqual({b['location_id'] for b in beats}, {'agnes-s-kitchen'})
        panels = story.direct(book, chapter['id'])
        self.assertEqual(''.join(p['source'] for p in panels), self.SOURCE)
        self.assertEqual(panels[0]['cast'][0]['expression'], 'arms folded')

    def test_unchanged_looks_and_places_carry_over_from_continuity(self):
        book = story.analyze(self.SOURCE, lambda *_: json.dumps(self.reply()))
        agnes = next(c['id'] for c in book['characters'] if c['name'] == 'Agnes Thorne')
        later = {"title": "Later", "characters": [], "looks": [], "places": [],
                 "beats": [{"units": [0], "importance": 2, "action": "Agnes sits", "place": "Agnes's kitchen",
                            "mood": "calm", "cast": [{"id": agnes, "feel": "tired"}]}], "speakers": []}
        book = story.analyze('Agnes sat down.', lambda *_: json.dumps(later), previous=book)
        beat = book['chapters'][1]['scenes'][0]['beats'][0]
        self.assertEqual(beat['visible'][0]['appearance'], 'grey apron')
        self.assertEqual(beat['location_state'], 'tar and oranges, storm outside')


class SmallWriterSlips(unittest.TestCase):
    SOURCE = 'Agnes waved. Tam ran. “Wait!” Agnes called. The boat left. Silence.'

    def test_an_invented_person_is_dropped_instead_of_losing_the_passage(self):
        reply = {"title": "Pier", "characters": [{"id": "c1", "name": "Agnes", "aliases": [], "visual": "", "voice": ""},
                                                 {"id": "c9", "name": "The harbourmaster", "aliases": [], "visual": "", "voice": ""}],
                 "beats": [{"units": [0, 1, 2, 3, 4], "importance": 2, "action": "a", "place": "pier", "mood": "m",
                            "cast": [{"id": "c1", "feel": "calm"}, {"id": "c9", "feel": "stern"}]}],
                 "speakers": [{"quote": 1, "id": "c9"}]}
        book = story.analyze(self.SOURCE, lambda *_: json.dumps(reply), passage_words=99, passage_units=9)
        scene = book['chapters'][0]['scenes'][0]
        self.assertFalse(scene.get('unanalysed'))
        self.assertEqual([c['name'] for c in book['characters']], ['Agnes'])
        # Not the invented person: the speech tag names Agnes.
        self.assertEqual(scene['beats'][0]['dialogue'][0]['speaker_id'], book['characters'][0]['id'])

    def test_skipped_and_repeated_units_are_regrouped_in_order(self):
        reply = {"title": "Pier", "characters": [],
                 "beats": [{"units": [0, 1], "importance": 2, "action": "a", "place": "pier", "mood": "m", "cast": []},
                           {"units": [1, 4], "importance": 3, "action": "b", "place": "pier", "mood": "m", "cast": []}],
                 "speakers": []}
        book = story.analyze(self.SOURCE, lambda *_: json.dumps(reply), passage_words=99, passage_units=9)
        chapter = book['chapters'][0]
        self.assertFalse(chapter['scenes'][0].get('unanalysed'))
        self.assertEqual(''.join(b['source'] for s in chapter['scenes'] for b in s['beats']), self.SOURCE)


class StopAndResume(unittest.TestCase):
    SOURCE = ' '.join(f'Mira walked past house {i}.' for i in range(12))

    def write_until(self, limit, calls):
        def write(_, prompt):
            calls.append(prompt)
            if len(calls) > limit:
                raise story.Stop()
            group = json.loads(prompt.split('Numbered source units:\n')[1].split('\nYour previous reply')[0])
            return json.dumps({"title": "Walk", "characters": [], "looks": [], "places": [], "speakers": [],
                               "beats": [{"units": [u["id"] for u in group], "importance": 2, "action": "walks",
                                          "place": "street", "mood": "calm", "cast": []}]})
        return write

    def test_stopping_keeps_finished_passages_and_resume_finishes_only_the_rest(self):
        calls = []
        book = story.analyze(self.SOURCE, self.write_until(2, calls), passage_words=99, passage_units=3)
        chapter = book['chapters'][0]
        self.assertEqual([bool(s.get('unanalysed')) for s in chapter['scenes']], [False, False, True, True])
        self.assertEqual(''.join(b['source'] for s in chapter['scenes'] for b in s['beats']), self.SOURCE)
        calls.clear()
        done = story.resume(book, chapter['id'], self.write_until(99, calls), passage_words=99, passage_units=3)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all('house 0.' not in c for c in calls))
        scenes = done['chapters'][0]['scenes']
        self.assertFalse(any(s.get('unanalysed') for s in scenes))
        self.assertEqual(len({s['id'] for s in scenes}), len(scenes))
        self.assertEqual(''.join(b['source'] for s in scenes for b in s['beats']), self.SOURCE)

    def test_stopping_a_resume_keeps_what_it_finished(self):
        calls = []
        book = story.analyze(self.SOURCE, self.write_until(1, calls), passage_words=99, passage_units=3)
        calls.clear()
        more = story.resume(book, book['chapters'][0]['id'], self.write_until(1, calls), passage_words=99, passage_units=3)
        self.assertEqual([bool(s.get('unanalysed')) for s in more['chapters'][0]['scenes']], [False, False, True, True])

    def test_worker_stop_returns_the_finished_part_instead_of_an_error(self):
        import worker
        calls = []
        def write(req_id, system, user, **kw):
            calls.append(1)
            if len(calls) == 2:
                raise worker.Cancelled()
            group = json.loads(user.split('Numbered source units:\n')[1].split('\nYour previous reply')[0])
            return json.dumps({"title": "Walk", "characters": [], "speakers": [],
                               "beats": [{"units": [u["id"] for u in group], "importance": 2, "action": "walks",
                                          "place": "street", "mood": "calm", "cast": []}]})
        with patch.object(worker, '_write', side_effect=write), \
                patch.object(worker, 'is_cancelled', lambda _id: len(calls) >= 2):
            result = worker.op_story_analyze('test-stop', {'source': self.SOURCE * 3})
        self.assertTrue(result['stopped'])
        self.assertGreater(result['unanalysed'], 0)
        scenes = result['book']['chapters'][0]['scenes']
        self.assertFalse(scenes[0].get('unanalysed'))


class WriterJsonSlips(unittest.TestCase):
    def test_a_field_closed_by_a_curly_quote_is_repaired_and_valid_quotes_are_untouched(self):
        slip = '{"title": "Pier", "beats": [{"action": "Agnes says ‘too late.’”, "place": "pier"}], "speakers": [],}'
        self.assertEqual(story._loads(slip)['beats'][0]['place'], 'pier')
        self.assertEqual(story._loads('{"a": "He said “hi”, then left"}')['a'], 'He said “hi”, then left')
        with self.assertRaises(ValueError):
            story._loads('{"beats": [')


class SpeechTagAttribution(unittest.TestCase):
    """Lines the writer leaves unattributed get the speaker their tag names."""

    def speakers(self, source, cast):
        group = story.passages(source, max_words=999, max_units=99)[0]
        reply = {"title": "t", "characters": [{"id": n.lower(), "name": n, "aliases": [], "visual": "", "voice": ""} for n in cast],
                 "beats": [{"units": [u["id"] for u in group], "importance": 2, "action": "a", "place": "p",
                            "mood": "m", "cast": []}], "speakers": []}
        book = story.analyze(source, lambda *_: json.dumps(reply), passage_words=999, passage_units=99)
        people = {c['id']: c['name'] for c in book['characters']}
        return [people.get(d['speaker_id']) for s in book['chapters'][0]['scenes'] for b in s['beats'] for d in b['dialogue']]

    def test_named_first_person_and_interrupted_tags(self):
        source = ('Agnes said, “Come in.” “I’m not late,” I said. “The ferry was early.” '
                  '“The ferry,” Agnes said, “has never been early.”')
        self.assertEqual(self.speakers(source, ['Narrator', 'Agnes']), ['Agnes', 'Narrator', 'Narrator', 'Agnes', 'Agnes'])

    def test_pronoun_tag_takes_the_person_last_named_in_narration(self):
        source = 'Tam finally looked at me. “It won’t drop,” he said.'
        self.assertEqual(self.speakers(source, ['Narrator', 'Tam']), ['Tam'])

    def test_a_tag_for_the_next_quote_and_untagged_lines_are_not_guessed(self):
        source = 'Mira read the sign: “Closed.” She said, “Go home.” Later: “Who knows?”'
        self.assertEqual(self.speakers(source, ['Mira']), [None, 'Mira', None])

    def test_speech_continued_into_the_next_paragraph(self):
        source = 'Agnes said, “You’re the Pell girl.\n\n“Your grandmother said you’d come.”\n\n“Hello.”'
        self.assertEqual(self.speakers(source, ['Agnes']), ['Agnes', 'Agnes', None])


class SavedBooksGetTags(unittest.TestCase):
    def test_a_book_analysed_before_tags_is_narrated_with_them(self):
        source = 'Agnes said, “Come in.”'
        group = story.passages(source)[0]
        data = reply(group, [{"id": "a", "name": "Agnes", "aliases": [], "visual": "", "voice": ""}])
        book = story.analyze(source, lambda *_: json.dumps(data))
        line = book['chapters'][0]['scenes'][0]['beats'][0]['dialogue'][0]
        line['speaker_id'] = None  # as saved by an older analysis
        segments = story.narration_segments(book, book['chapters'][0]['id'])
        self.assertEqual([s['speaker_id'] for s in segments if s['text'] == 'Come in.'], [book['characters'][0]['id']])
        self.assertIsNone(line['speaker_id'])  # the saved book itself is not changed


class KokoroNarration(unittest.TestCase):
    def model(self, calls):
        from types import SimpleNamespace
        class Model:
            def generate(self, **kwargs):
                calls.append(kwargs)
                yield SimpleNamespace(audio=[0.0, 0.25, -0.25], sample_rate=24000)
        return Model()

    def test_hindi_voices_read_devanagari_as_hindi_and_english_with_their_accent(self):
        self.assertEqual(audiobook.kokoro_language("hf_alpha", "नमस्ते, Agnes."), "h")
        self.assertEqual(audiobook.kokoro_language("hm_omega", "The ferry was early."), "a")
        self.assertEqual(audiobook.kokoro_language("bf_emma", "The ferry was early."), "b")

    def test_characters_use_their_voices_at_the_chosen_pace_and_output_is_pcm(self):
        calls = []
        segments = [{"speaker_id": None, "text": "The pier was empty."},
                    {"speaker_id": "agnes", "text": "You are late."}]
        data, seconds = audiobook.synthesize_kokoro(segments, "af_heart", {"agnes": "hf_beta"}, 200,
                                                    lambda: None, load=lambda: self.model(calls))
        self.assertEqual([(c['voice'], c['lang_code'], c['speed']) for c in calls],
                         [('af_heart', 'a', 1.25), ('hf_beta', 'a', 1.25)])
        with wave.open(io.BytesIO(data)) as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getnframes()), (24000, 1, 6))

    def test_scene_markers_fall_where_each_scene_begins(self):
        segments = [{"speaker_id": None, "text": "One.", "scene": "Arrival"},
                    {"speaker_id": None, "text": "Two.", "scene": "Arrival"},
                    {"speaker_id": None, "text": "Three.", "scene": "The kitchen"}]
        markers = []
        audiobook.synthesize_kokoro(segments, "af_heart", {}, 160, lambda: None,
                                    load=lambda: self.model([]), markers=markers)
        # One marker per scene, not per segment; the stand-in voice is a few samples long.
        self.assertEqual([m["label"] for m in markers], ["Arrival", "The kitchen"])
        self.assertEqual(markers[0]["time"], 0.0)

    def test_unknown_voices_are_refused_before_loading(self):
        with self.assertRaisesRegex(ValueError, 'Kokoro voices'):
            audiobook.synthesize_kokoro([{"speaker_id": None, "text": "x"}], "ryan", {}, 160,
                                        lambda: None, load=lambda: self.fail("loaded"))

    def test_worker_lists_kokoro_voices_with_indian_ones(self):
        import worker
        voices = worker.op_audio_voices('v', {"model_id": audiobook.KOKORO_MODEL_ID})['voices']
        self.assertIn({"id": "hm_omega", "language": "Hindi / Indian English"}, voices)


class TagsOverrideTheWriter(unittest.TestCase):
    def test_an_explicit_tag_corrects_a_wrong_attribution(self):
        source = 'Agnes said, “Come in.” “I’m not late,” I said.'
        group = story.passages(source)[0]
        data = reply(group, [{"id": "a", "name": "Agnes", "aliases": [], "visual": "", "voice": ""}])
        data['beats'][0]['dialogue'] = [{"speaker_id": "a", "text": "Come in.", "offscreen": False},
                                        {"speaker_id": "a", "text": "I’m not late,", "offscreen": False}]
        book = story.analyze(source, lambda *_: json.dumps(data))
        people = {c['id']: c['name'] for c in book['characters']}
        said = [people[d['speaker_id']] for d in book['chapters'][0]['scenes'][0]['beats'][0]['dialogue']]
        self.assertEqual(said, ['Agnes', 'Narrator'])

    def test_speech_continued_past_its_tag_keeps_the_tagged_speaker(self):
        source = 'Agnes said, “Come in.” “I’m not late,” I said. “The ferry was early.”'
        group = story.passages(source)[0]
        data = reply(group, [{"id": "a", "name": "Agnes", "aliases": [], "visual": "", "voice": ""}])
        data['beats'][0]['dialogue'] = [{"speaker_id": "a", "text": t, "offscreen": False}
                                        for t in ("Come in.", "I’m not late,", "The ferry was early.")]
        book = story.analyze(source, lambda *_: json.dumps(data))
        people = {c['id']: c['name'] for c in book['characters']}
        said = [people[d['speaker_id']] for d in book['chapters'][0]['scenes'][0]['beats'][0]['dialogue']]
        self.assertEqual(said, ['Agnes', 'Narrator', 'Narrator'])

    def test_a_stray_straight_quote_inside_a_field_is_escaped(self):
        self.assertEqual(story._loads('{"action": "Agnes says "hello" twice", "place": "pier"}')['place'], 'pier')


class SceneDetection(unittest.TestCase):
    """Scenes come from the story's own breaks, not from a sentence count."""

    def test_separators_headings_and_time_or_place_shifts_start_scenes(self):
        source = ('Mira waited on the pier. The ferry was late.\n\n'
                  'She checked her watch.\n\n'
                  'Hours later, the ferry came in.\n\n'
                  'Back at the house, Jonas made tea.\n\n'
                  '* * *\n\n'
                  'Rain fell all night.\n\n'
                  'Morning\n\n'
                  'The sun came up.')
        starts = story.scene_starts(source)
        opened = [source[at:at + 12] for at in starts]
        self.assertEqual(opened, ['Hours later,', 'Back at the ', '* * *\n\nRain ', 'Morning\n\nThe'])

    def test_two_blank_lines_end_a_scene_and_a_plain_paragraph_does_not(self):
        source = 'Mira left.\n\nShe walked home.\n\n\n\nJonas slept.'
        self.assertEqual([source[at:at + 5] for at in story.scene_starts(source)], ['Jonas'])

    def test_a_passage_never_crosses_a_scene_and_numbers_it(self):
        source = 'Mira waited. ' * 3 + '\n\nThe next morning the ferry came. It was empty.'
        groups = story.passages(source, max_words=999, max_units=99, scenes=True)
        self.assertEqual(len(groups), 2)
        self.assertEqual([g[0]['section'] for g in groups], [0, 1])
        self.assertEqual(''.join(u['text'] for g in groups for u in g), source)
        # The audiobook's passages are unchanged.
        self.assertEqual(len(story.passages(source, max_words=999, max_units=99)), 1)

    def test_analysed_scenes_carry_their_story_scene(self):
        source = 'Mira waited.\n\nLater, the ferry came.'
        def write(_, prompt):
            group = json.loads(prompt.split('Numbered source units:\n')[1].split('\nYour previous reply')[0])
            return json.dumps({"title": "t", "characters": [], "looks": [], "places": [], "speakers": [],
                               "beats": [{"units": [u["id"] for u in group], "importance": 2, "action": "a",
                                          "place": "pier", "mood": "m", "cast": []}]})
        book = story.analyze(source, write)
        self.assertEqual([s['section'] for s in book['chapters'][0]['scenes']], [0, 1])


class SpeakingRoles(unittest.TestCase):
    """A tag naming a role gives the line to that role, as a minor character."""

    def analysed(self, source, characters, speakers):
        group = story.passages(source, max_words=999, max_units=99)[0]
        reply = {"title": "t", "characters": characters,
                 "beats": [{"units": [u["id"] for u in group], "importance": 2, "action": "a", "place": "p",
                            "mood": "m", "cast": []}], "speakers": speakers}
        return story.analyze(source, lambda *_: json.dumps(reply), passage_words=999, passage_units=99)

    def test_fx01_the_guards_call_is_the_guards_even_when_the_writer_says_mira(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' / 'fx01.json').read_text())
        book = self.analysed(fixture['source'], [
            {"id": "m", "name": "Mira Vale", "aliases": ["Mira", "Mi"], "visual": "", "voice": ""},
            {"id": "j", "name": "Jonas", "aliases": [], "visual": "", "voice": ""}],
            [{"quote": 1, "id": "m"}])
        people = {c['id']: c for c in book['characters']}
        line = next(d for s in book['chapters'][0]['scenes'] for b in s['beats'] for d in b['dialogue'])
        self.assertEqual(line['text'], 'Last train!')
        self.assertEqual(people[line['speaker_id']]['name'], 'The guard')
        self.assertIn('guard', people[line['speaker_id']]['aliases'])
        # Mira named herself, so she is the narrator and no Narrator is added.
        self.assertEqual(sorted(c['name'] for c in book['characters']), ['Jonas', 'Mira Vale', 'The guard'])

    def test_a_declared_role_is_reused_not_duplicated(self):
        source = 'A guard called, “Last train!” The guard called, “Doors closing!”'
        book = self.analysed(source, [{"id": "g", "name": "A guard", "aliases": ["guard"], "visual": "", "voice": ""}], [])
        self.assertEqual([c['name'] for c in book['characters']], ['A guard'])
        lines = [d['speaker_id'] for s in book['chapters'][0]['scenes'] for b in s['beats'] for d in b['dialogue']]
        self.assertEqual(lines, [book['characters'][0]['id']] * 2)

    def test_a_sound_or_an_action_verb_makes_no_person(self):
        source = 'The rain began. A voice called, “Hello?” The radio said, “Storm warning.”'
        book = self.analysed(source, [], [])
        self.assertEqual(book['characters'], [])


class SelfNamedNarrator(unittest.TestCase):
    """A narrator who names themselves is that character, not a second person."""

    SOURCE = 'My name is Mira Vale. I unhooked the key from my coat. “Take it,” I said. Jonas took it.'

    def analysed(self, characters, cast):
        group = story.passages(self.SOURCE, max_words=999, max_units=99)[0]
        reply = {"title": "t", "characters": characters,
                 "beats": [{"units": [u["id"] for u in group], "importance": 2, "action": "a", "place": "p",
                            "mood": "m", "cast": [{"id": c, "feel": ""} for c in cast]}],
                 "speakers": [{"quote": 1, "id": "n"}]}
        return story.analyze(self.SOURCE, lambda *_: json.dumps(reply), passage_words=999, passage_units=99)

    def test_a_generic_narrator_beside_the_named_one_is_merged_into_her(self):
        book = self.analysed([{"id": "m", "name": "Mira Vale", "aliases": ["Mira"], "visual": "", "voice": ""},
                              {"id": "n", "name": "Narrator", "aliases": [], "visual": "tall", "voice": ""},
                              {"id": "j", "name": "Jonas", "aliases": [], "visual": "", "voice": ""}], ["m", "n", "j"])
        names = [c['name'] for c in book['characters']]
        self.assertEqual(sorted(names), ['Jonas', 'Mira Vale'])
        mira = next(c for c in book['characters'] if c['name'] == 'Mira Vale')
        self.assertTrue(story.is_narrator(mira))
        self.assertEqual(mira['visual'], 'tall')
        beat = book['chapters'][0]['scenes'][0]['beats'][0]
        self.assertEqual([a['character_id'] for a in beat['visible']].count(mira['id']), 1)
        self.assertEqual(beat['dialogue'][0]['speaker_id'], mira['id'])

    def test_an_unnamed_narrator_takes_the_name_the_text_gives(self):
        book = self.analysed([{"id": "n", "name": "Narrator", "aliases": [], "visual": "", "voice": ""}], ["n"])
        self.assertEqual([c['name'] for c in book['characters']], ['Mira Vale'])

    def test_a_saved_book_with_both_is_put_right_for_the_board(self):
        book = self.analysed([{"id": "n", "name": "Narrator", "aliases": [], "visual": "", "voice": ""}], ["n"])
        old = copy.deepcopy(book)
        narrator = old['characters'][0]
        old['characters'] = [{**narrator, "name": "Narrator", "aliases": []},
                             {"id": "mira", "name": "Mira Vale", "aliases": [], "visual": "", "voice": ""}]
        fixed = story.ensure_cast(old)
        self.assertIsNot(fixed, old)
        self.assertEqual([c['name'] for c in fixed['characters']], ['Mira Vale'])
        self.assertIs(story.ensure_cast(fixed), fixed)

    def test_i_am_with_a_capital_is_not_a_name(self):
        characters = [{"id": "n", "name": "Narrator", "aliases": [], "visual": "", "voice": ""}]
        story._settle_narrator(characters, [], 'I am Tired of waiting. I sat down. I slept.')
        self.assertEqual(characters[0]['name'], 'Narrator')
