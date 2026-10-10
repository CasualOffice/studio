#!/usr/bin/env python3
"""Live Apple Silicon worker smoke test; downloads real analyst and speech models.

Uses redistributable fixture text only. Exports a short WAV for listening review.
This exercises engine protocol, analysis, Board planning and encrypted narration;
it does not drive the desktop UI or render Board images.
"""
import argparse
import io
import json
import os
from pathlib import Path
import platform
import queue
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
WRITER = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
SPEECH = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit"


def run(output):
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("Live inference requires Apple Silicon macOS; no mocked fallback is used")
    output.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    for repo in [WRITER, SPEECH]:
        print(f"Downloading live test model: {repo}", flush=True)
        snapshot_download(repo_id=repo)
    sys.path.insert(0, str(ROOT / "engine"))
    import story
    import vaultcrypto
    events = queue.Queue()
    report = {"mode": "real-model-inference", "writer": WRITER, "speech": SPEECH, "checks": []}
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               MODELSTUDIO_MEMORY_BUDGET_GIB="6", OMP_NUM_THREADS="4",
               MODELSTUDIO_IDLE_UNLOAD_SECONDS="600")
    with tempfile.TemporaryDirectory(prefix="studio-live-smoke-") as stage, \
            (output / "worker.stderr.log").open('w') as errors, \
            (output / "worker.events.jsonl").open('w') as log:
        child = subprocess.Popen([sys.executable, '-u', str(ROOT / 'engine' / 'worker.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True, env=env)
        def read():
            for line in child.stdout:
                try:
                    event = json.loads(line)
                    log.write(line); log.flush(); events.put(event)
                except json.JSONDecodeError:
                    events.put({"type": "error", "error": "Worker emitted invalid protocol JSON"})
            events.put({"type": "error", "error": f"Worker exited ({child.poll()})"})
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        def request(op, **args):
            identifier = secrets.token_hex(8)
            print(f"Live check: {op}", flush=True)
            child.stdin.write(json.dumps({"id": identifier, "op": op, **args}) + '\n'); child.stdin.flush()
            deadline = time.monotonic() + 1800
            last_message = None
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Live {op} timed out")
                event = events.get(timeout=max(1, deadline - time.monotonic()))
                if event.get('type') == 'error' and event.get('id') in (None, identifier):
                    raise RuntimeError(event['error'])
                if event.get('id') != identifier:
                    continue
                if event['type'] == 'result':
                    return event['result']
                if event['type'] == 'progress' and event.get('message') != last_message:
                    last_message = event.get('message')
                    if last_message: print(last_message, flush=True)
        try:
            fixture = json.loads((ROOT / 'engine/fixtures/fx01.json').read_text())
            result = request('story_analyze', source=fixture['source'], title='FX-01 live analysis', writer=WRITER)
            book, chapter_id = result['book'], result['chapter_id']
            story.validate_book(book)
            assert ''.join(s['text'] for s in story.narration_segments(book, chapter_id)) == fixture['source']
            assert result['panels'] and book['characters'], 'Analysis must produce characters and panels'
            report['checks'].append('Qwen3 4B analysis, complete source coverage and canonical character binding')
            direct = request('direct_story', book=book, chapter_id=chapter_id)
            assert direct['panels'] == result['panels'], 'Board and Audiobook must share the same saved analysis'
            report['checks'].append('Board planning from saved chapter without reanalysis')
            original = json.loads(json.dumps(book))
            continuation = 'Mira Vale took off her red coat. Mira Vale put on a blue dress. Jonas waited beside the bench.'
            followup = request('story_analyze', source=continuation, title='Continuation', previous=book, writer=WRITER)
            story.validate_book(followup['book'])
            assert followup['book']['chapters'][0] == original['chapters'][0], 'Appending changed the original chapter'
            assert len(followup['book']['chapters']) == 2
            report['checks'].append('Second chapter appends without changing original source or analysis')
            # Exercise encrypted story round-trip with a disposable key, like vault state.
            key, file_id = secrets.token_bytes(32), secrets.token_bytes(16)
            state_path = Path(stage) / 'story.sealed'
            vaultcrypto.write_sealed(str(state_path), key, file_id, json.dumps(book).encode())
            encrypted = state_path.read_bytes()
            restored = json.loads(vaultcrypto.open_with_file_key(key, encrypted))
            assert restored == book
            report['checks'].append('Encrypted story round-trip')
            audio_source = 'Mira Vale said, “Wait for me.” Jonas said, “I will.”'
            audio_analysis = request('story_analyze', source=audio_source, title='Voice performance', previous=book, writer=WRITER)
            audio_book = audio_analysis['book']
            assignments = {c['id']: 'vivian' if 'mira' in c['name'].lower() else 'aiden'
                           for c in audio_book['characters']}
            slot = {"id": "live-audio", "path": str(Path(stage) / 'audio.sealed'),
                    "key": key.hex(), "file_id": file_id.hex()}
            audio_result = request('audiobook', book=audio_book, chapter_id=audio_analysis['chapter_id'],
                narrator='ryan', voices=assignments, rate=160, model_id='qwen3-tts-1.7b-custom-6bit',
                language='english', delivery='Warm natural storytelling, with gentle pauses.', vault_slots=[slot])
            assert audio_result['sealed'] and audio_result['sizes'][0] > 0
            sealed = Path(slot['path']).read_bytes()
            assert not sealed.startswith(b'RIFF'), 'Audio reached disk unencrypted'
            wav_data = vaultcrypto.open_with_file_key(key, sealed)
            with wave.open(io.BytesIO(wav_data), 'rb') as wav:
                assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000)
                assert wav.getnframes() > 24000, 'Audio must contain more than one second'
                report['audio_seconds'] = wav.getnframes() / 24000
            (output / 'fixture-neural-narration.wav').write_bytes(wav_data)
            report['checks'].append('Real neural character speech, encrypted WAV output and successful decryption')
            report['book'] = book
            report['status'] = 'passed'
            print('Live pipeline checks passed; listen to the fixture WAV to assess delivery.', flush=True)
        except Exception as exc:
            report['status'] = 'failed'; report['error'] = str(exc)
            message = str(exc).replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
            print(f'::error title=Live story/audio pipeline failure::{message}', flush=True)
            raise
        finally:
            (output / 'report.json').write_text(json.dumps(report, indent=2))
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
            reader.join(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('smoke-results'))
    args = parser.parse_args()
    run(args.output)
