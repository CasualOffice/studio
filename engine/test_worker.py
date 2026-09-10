"""Tests for the engine's pure logic.

Run with the app's own interpreter:

    ~/Library/Application\\ Support/com.melp.modelstudio/runtime/venv/bin/python3 \\
        -m unittest discover -s engine -v

These cover the parts that decide *what gets sent to a model* and *what gets
written to disk* — the places where a silent mistake changes a result rather
than raising. Loading actual weights is out of scope here.
"""

import importlib.util
import io
import os
import secrets
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


worker = _load("worker")
vc = _load("vaultcrypto")


class DropNamed(unittest.TestCase):
    """Which optional argument an error message is actually complaining about.

    This got it wrong once in a way that was hard to see: MLX-Gen's errors end
    with the list of parameters the route *does* accept, so a substring search
    matched an accepted name and dropped the wrong argument. That silently
    removed the image from an edit and surfaced much later as a crash inside
    MLX with no obvious connection.
    """

    def test_picks_the_rejected_name_not_an_accepted_one(self):
        msg = ("'image_path' is not a parameter of 'image-to-image' on "
               "'AbstractFramework/flux.2-klein-4b-4bit'. Accepted: canvas_policy, "
               "guidance, height, image_paths, image_strength, negative_prompt, "
               "num_inference_steps, prompt, scheduler, width.")
        opt = {"width": 512, "height": 512, "image_paths": ["a"], "image_path": "a"}
        self.assertEqual(worker._drop_named(opt, msg), "image_path")
        self.assertIn("image_paths", opt, "the plural form must survive")
        self.assertIn("width", opt, "an accepted parameter must not be dropped")

    def test_matches_cli_spelling_with_hyphens(self):
        msg = "--negative-prompt is not supported for FLUX.2 Klein distilled weights"
        opt = {"negative_prompt": "x", "width": 512}
        self.assertEqual(worker._drop_named(opt, msg), "negative_prompt")

    def test_matches_a_value_constraint(self):
        msg = "guidance > 1.0 is only supported for FLUX.2 Klein base models."
        opt = {"guidance": 3.5, "width": 512}
        self.assertEqual(worker._drop_named(opt, msg), "guidance")

    def test_reaches_into_nested_model_kwargs(self):
        msg = "Flux2Klein.__init__() got an unexpected keyword argument 'release_text_encoder'"
        opt = {"model_kwargs": {"release_text_encoder": True}}
        self.assertEqual(worker._drop_named(opt, msg), "model_kwargs.release_text_encoder")
        self.assertNotIn("model_kwargs", opt, "an emptied dict should be removed entirely")

    def test_returns_none_when_nothing_matches(self):
        # A real failure must surface rather than be retried forever.
        self.assertIsNone(worker._drop_named({"width": 512}, "out of memory"))


class KeepsIntent(unittest.TestCase):
    """A rewrite that loses the request is worse than no rewrite.

    A 2B model handed an image will sometimes describe the picture instead of
    following the instruction, turning "make it blue" into "remove the
    background". Performing a *different* edit silently is the worst outcome.
    """

    def test_rejects_a_rewrite_that_dropped_the_request(self):
        self.assertFalse(worker._keeps_intent("make it blue", "remove the background."))

    def test_accepts_a_faithful_rewrite(self):
        self.assertTrue(worker._keeps_intent(
            "make it blue", "change the teapot to blue, keeping the cloth unchanged"))

    def test_tolerates_inflection(self):
        self.assertTrue(worker._keeps_intent("make it red", "turn the jacket reddish"))

    def test_ignores_filler_words(self):
        # "make"/"it"/"the" carry no intent, so matching only those is not enough.
        self.assertFalse(worker._keeps_intent("add a hat", "make the image better"))


class CleanAssist(unittest.TestCase):
    def test_strips_labels_and_quotes(self):
        self.assertEqual(worker._clean_assist('Prompt: "a red cube"', "fb"), "a red cube")
        self.assertEqual(worker._clean_assist("You write: a cat", "fb"), "a cat")

    def test_falls_back_when_empty(self):
        self.assertEqual(worker._clean_assist("   ", "original"), "original")

    def test_drops_trailing_commentary(self):
        self.assertEqual(
            worker._clean_assist("a red cube\n\nNote: I added detail.", "fb"), "a red cube")


class SplitGenKwargs(unittest.TestCase):
    def test_required_and_optional_are_separated(self):
        req = {"prompt": "x", "steps": 4, "guidance": 3.5, "width": 512}
        required, optional = worker._split_gen_kwargs(req)
        self.assertEqual(required, {"prompt": "x", "num_inference_steps": 4})
        # Route-specific settings must be droppable, never fatal.
        self.assertIn("guidance", optional)
        self.assertIn("width", optional)

    def test_absent_values_are_omitted(self):
        required, optional = worker._split_gen_kwargs({"prompt": "x", "steps": 2})
        self.assertNotIn("guidance", optional)
        self.assertEqual(required["num_inference_steps"], 2)


class VaultFormat(unittest.TestCase):
    """The engine seals its own output; the host must be able to open it."""

    def test_round_trips_across_the_chunk_boundary(self):
        key, fid = secrets.token_bytes(32), secrets.token_bytes(16)
        for n in (0, 1, vc.CHUNK_SIZE - 1, vc.CHUNK_SIZE, vc.CHUNK_SIZE + 1):
            data = bytes((i % 251) for i in range(n))
            self.assertEqual(vc.open_with_file_key(key, vc.seal_with_file_key(key, fid, data)),
                             data, f"failed at {n} bytes")

    def test_tampering_is_detected(self):
        key, fid = secrets.token_bytes(32), secrets.token_bytes(16)
        sealed = bytearray(vc.seal_with_file_key(key, fid, b"secret"))
        sealed[-1] ^= 1
        with self.assertRaises(Exception):
            vc.open_with_file_key(key, bytes(sealed))

    def test_wrong_key_is_rejected(self):
        fid = secrets.token_bytes(16)
        sealed = vc.seal_with_file_key(secrets.token_bytes(32), fid, b"secret")
        with self.assertRaises(Exception):
            vc.open_with_file_key(secrets.token_bytes(32), sealed)

    def test_write_sealed_leaves_nothing_behind_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "blob")
            with self.assertRaises(ValueError):
                vc.write_sealed(dest, b"too short", secrets.token_bytes(16), b"x")
            self.assertEqual(os.listdir(d), [], "a failed write must not leave a partial file")


class Downscale(unittest.TestCase):
    """Cost in the vision model scales with pixels, and does so brutally:
    384px took 1.5s where 2048px took 58s. Sources must be shrunk first."""

    def test_large_images_are_reduced(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            big = os.path.join(d, "big.png")
            Image.new("RGB", (2400, 1800)).save(big)
            out = worker._downscale_for_assist([big])
            self.assertEqual(max(Image.open(out[0]).size), worker.ASSIST_MAX_EDGE)

    def test_small_images_are_left_alone(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            small = os.path.join(d, "small.png")
            Image.new("RGB", (320, 240)).save(small)
            self.assertEqual(worker._downscale_for_assist([small]), [small])


class Staging(unittest.TestCase):
    """One input must produce exactly one path.

    Staging returned the original *and* its converted copy, because the same
    list was used both to feed the model and to clean up afterwards. A single
    HEIC or WebP source therefore reached the model as two images.
    """

    def _sealed(self, tmp, fmt, ext):
        import secrets
        from PIL import Image
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except Exception:
            pass
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (10, 120, 200)).save(buf, format=fmt)
        key, fid = secrets.token_bytes(32), secrets.token_bytes(16)
        path = os.path.join(tmp, f"blob-{ext}")
        vc.write_sealed(path, key, fid, buf.getvalue())
        return {"id": f"i{ext}", "file_id": fid.hex(), "key": key.hex(),
                "path": path, "ext": ext}

    def test_one_path_per_input_whatever_the_format(self):
        for fmt, ext in (("PNG", "png"), ("WEBP", "webp"), ("JPEG", "jpg"), ("BMP", "bmp")):
            with tempfile.TemporaryDirectory() as tmp:
                entry = self._sealed(tmp, fmt, ext)
                staged = worker._stage_vault_inputs([entry])
                self.assertEqual(len(staged), 1, f"{ext} produced {len(staged)} paths")
                worker._discard_staged(staged)

    def test_two_inputs_produce_two_paths_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self._sealed(tmp, "WEBP", "webp")
            b = self._sealed(tmp, "PNG", "png")
            staged = worker._stage_vault_inputs([a, b])
            self.assertEqual(len(staged), 2)
            worker._discard_staged(staged)

    def test_cleanup_removes_the_converted_copy_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = self._sealed(tmp, "WEBP", "webp")
            staged = worker._stage_vault_inputs([entry])
            worker._discard_staged(staged)
            leftovers = [f for f in os.listdir(worker._stage_dir())
                         if f.startswith("iwebp")]
            self.assertEqual(leftovers, [], "conversion left a file behind")




class EditInstruction(unittest.TestCase):
    """The user's wording must survive verbatim.

    Letting a 2B model restate an instruction after showing it a picture meant
    it folded the picture's details into the request, and sometimes replaced
    the request outright. The sentence is now assembled from the user's own
    words; the model only supplies a subject noun.
    """

    def test_pronoun_is_resolved_to_the_subject(self):
        out = worker._compose_edit_instruction("make it blue", "a beige ceramic teapot")
        self.assertIn("blue", out)
        self.assertIn("teapot", out)
        self.assertNotIn(" it ", f" {out} ")
        # Word boundaries: substitution must not run words together or leave
        # double spaces, both of which broke an earlier index-based version.
        self.assertNotIn("  ", out)
        self.assertTrue(out.startswith("make a beige ceramic teapot blue"), out)

    def test_does_not_match_inside_another_word(self):
        out = worker._compose_edit_instruction("make the white parts warmer", "a jacket")
        self.assertIn("white", out, "a pronoun search matched inside 'white'")

    def test_users_own_words_are_preserved(self):
        for request in ("add steam coming out", "remove the handle",
                        "make the lid gold"):
            out = worker._compose_edit_instruction(request, "a teapot")
            first = request.split()[0]
            self.assertTrue(out.lower().startswith(first),
                            f"{out!r} did not begin with {request!r}")

    def test_intent_cannot_be_replaced(self):
        # Whatever the subject, the request has to still be in there.
        out = worker._compose_edit_instruction("make it blue", "a car in a field at night")
        self.assertIn("blue", out, "the actual request was lost")

    def test_says_what_to_leave_alone(self):
        out = worker._compose_edit_instruction("make it red", "a jacket")
        self.assertIn("unchanged", out)

    def test_works_without_a_subject(self):
        out = worker._compose_edit_instruction("brighten the sky", "")
        self.assertTrue(out.lower().startswith("brighten the sky"))


class TrimToSentence(unittest.TestCase):
    """A token limit lands mid-clause; a dangling fragment is worse than a
    shorter finished sentence."""

    def test_completes_at_the_last_full_stop(self):
        text = "a tabby cat on a sill. warm rim light throug"
        self.assertEqual(worker._trim_to_sentence(text), "a tabby cat on a sill.")

    def test_falls_back_to_the_last_clause(self):
        out = worker._trim_to_sentence("a bustling market at dusk, neon lights, vibrant crow")
        self.assertTrue(out.endswith("."))
        self.assertNotIn("crow", out)

    def test_leaves_a_complete_sentence_alone(self):
        text = "a red cube on a table."
        self.assertEqual(worker._trim_to_sentence(text), text)

    def test_enforces_the_word_cap(self):
        long = " ".join(["word"] * 200)
        self.assertLessEqual(len(worker._trim_to_sentence(long).split()), 56)




class Repetition(unittest.TestCase):
    """Small models loop. The repeats crowd out the actual subject."""

    def test_cuts_where_the_loop_starts(self):
        looped = ("a sleek black cat, sleek and smooth, in a sleek black room, "
                  "sleek black walls, sleek black floor, sleek black ceiling")
        out = worker._collapse_repetition(looped)
        self.assertIn("cat", out)
        self.assertLess(out.lower().count("sleek"), looped.lower().count("sleek"))

    def test_leaves_a_varied_prompt_alone(self):
        good = ("a vivid orange sunset over a mountain range, vibrant red sky, "
                "deep blue mountains, warm light.")
        self.assertEqual(worker._collapse_repetition(good).rstrip("."), good.rstrip("."))

    def test_always_ends_with_punctuation(self):
        self.assertTrue(worker._collapse_repetition("a cat, a dog").endswith("."))


if __name__ == "__main__":
    unittest.main(verbosity=2)
