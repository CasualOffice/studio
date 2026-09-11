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




class Enhancement(unittest.TestCase):
    """Deepen the request; never replace it.

    The instruction has to keep working as an instruction: the user's words
    lead, and what the picture shows is appended as context to preserve.
    """

    FACTS = {
        "SUBJECT": "a beige ceramic teapot",
        "SURFACE": "smooth glazed stoneware",
        "LIGHT": "soft daylight from the left",
        "SETTING": "crumpled linen cloth, pale wall",
    }

    def test_request_leads_and_survives(self):
        out = worker._enrich_edit_instruction("make it blue", self.FACTS)
        self.assertTrue(out.lower().startswith("make a beige ceramic teapot blue"), out)
        self.assertIn("blue", out)

    def test_image_facts_are_added_as_context(self):
        out = worker._enrich_edit_instruction("make it blue", self.FACTS)
        self.assertIn("glazed stoneware", out)
        self.assertIn("daylight", out)
        self.assertIn("linen", out)

    def test_degrades_to_the_plain_instruction(self):
        out = worker._enrich_edit_instruction("make it blue", {})
        self.assertIn("make it blue", out)
        self.assertIn("unchanged", out)

    def test_partial_facts_are_fine(self):
        out = worker._enrich_edit_instruction("add steam", {"SUBJECT": "a teapot"})
        self.assertIn("add steam", out)
        self.assertIn("teapot", out)

    def test_scene_parsing_tolerates_mess(self):
        facts = worker._parse_scene(
            "Here you go:\n"
            "- SUBJECT: a red car\n"
            "surface: glossy paint\n"
            "LIGHT: unknown\n"
            "SETTING: a wet street\n"
        )
        self.assertEqual(facts.get("SUBJECT"), "a red car")
        self.assertEqual(facts.get("SURFACE"), "glossy paint")
        self.assertNotIn("LIGHT", facts, "'unknown' should be dropped, not used")



class NoContradiction(unittest.TestCase):
    """Never ask the editor to preserve what the user asked to change.

    Told "put it on a dark wooden table", appending "leaving the linen cloth
    unchanged" instructs the model to keep the exact thing being replaced.
    """

    FACTS = {
        "SUBJECT": "a beige ceramic teapot",
        "SURFACE": "smooth glazed stoneware",
        "LIGHT": "soft daylight from the left",
        "SETTING": "crumpled linen cloth, pale wall",
    }

    def test_setting_is_not_preserved_when_it_is_the_target(self):
        for request in ("put it on a dark wooden table",
                        "change the background to a forest",
                        "move it to the floor"):
            out = worker._enrich_edit_instruction(request, self.FACTS)
            self.assertNotIn("linen", out, f"contradicted itself for {request!r}: {out}")

    def test_lighting_is_not_preserved_when_it_is_the_target(self):
        out = worker._enrich_edit_instruction("make the lighting darker", self.FACTS)
        self.assertNotIn("daylight", out, out)

    def test_surface_is_not_preserved_when_it_is_the_target(self):
        out = worker._enrich_edit_instruction("give it a rough matte finish", self.FACTS)
        self.assertNotIn("glazed", out, out)

    def test_unrelated_request_still_gets_the_context(self):
        out = worker._enrich_edit_instruction("make it blue", self.FACTS)
        self.assertIn("glazed stoneware", out)
        self.assertIn("linen", out)





class SignificantWords(unittest.TestCase):
    """Short words are usually the whole request."""

    def test_three_letter_subjects_count(self):
        for word in ("cat", "red", "sun", "car", "sky", "dog"):
            self.assertIn(word, worker._significant(f"a {word} in the picture"),
                          f"{word!r} was discarded")

    def test_filler_is_still_ignored(self):
        got = worker._significant("please make it more beautiful")
        self.assertNotIn("make", got)
        self.assertNotIn("more", got)



class Sib:
    def __init__(self, name):
        self.rfilename = name


class Info:
    def __init__(self, files, tags=()):
        self.siblings = [Sib(f) for f in files]
        self.tags = list(tags)


class AdapterDetection(unittest.TestCase):
    """A LoRA pasted into the model box has to be recognised as a LoRA.

    Downloading one as a model produces a folder that can never generate, and
    "unsupported architecture" sends the user off looking for a different
    model when what they pasted was perfectly good.
    """

    def test_single_lora_file_is_an_adapter(self):
        info = Info(["lora.safetensors"],
                    ["base_model:finetune:black-forest-labs/FLUX.1-dev"])
        self.assertTrue(worker._looks_like_adapter(info))

    def test_named_lora_among_several_files_is_an_adapter(self):
        info = Info(["pytorch_lora_weights.safetensors", "README.md"])
        self.assertTrue(worker._looks_like_adapter(info))

    def test_a_real_model_is_not_an_adapter(self):
        # A model repository declares its pipeline; an adapter does not.
        info = Info(["model_index.json", "transformer/diffusion_pytorch_model.safetensors",
                     "vae/diffusion_pytorch_model.safetensors"])
        self.assertFalse(worker._looks_like_adapter(info))

    def test_repo_with_no_weights_is_not_an_adapter(self):
        self.assertFalse(worker._looks_like_adapter(Info(["README.md"])))


class MfluxBackendDetection(unittest.TestCase):
    """FLUX.1 is a separate lineage the unified router cannot place."""

    def setUp(self):
        # These test the evidence rules, not Hugging Face. Blocking the lookup
        # keeps them fast and keeps them passing without a network.
        #
        # The library is not installed everywhere the suite runs -- CI has only
        # what the tests touch, and hugging_face_hub pulls a great deal it does
        # not. When it is absent there is nothing to block: the detector treats
        # a failed import exactly like a failed lookup.
        try:
            import huggingface_hub
        except ImportError:
            return

        real = huggingface_hub.hf_hub_download

        def refuse(*a, **k):
            raise OSError("offline")

        huggingface_hub.hf_hub_download = refuse
        self.addCleanup(setattr, huggingface_hub, "hf_hub_download", real)

    def test_weight_filename_names_the_variant(self):
        cases = {
            "flux1-schnell.safetensors": "schnell",
            "flux1-dev.safetensors": "dev",
            "flux1-kontext-dev.safetensors": "dev_kontext",
            "flux1-fill-dev.safetensors": "dev_fill",
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(
                    worker._detect_mflux_backend("owner/x", [], [filename]),
                    expected,
                )

    def test_specific_variants_win_over_the_plain_ones(self):
        # Kontext repos also ship files whose names contain "dev".
        self.assertEqual(
            worker._detect_mflux_backend(
                "black-forest-labs/FLUX.1-Kontext-dev", [],
                ["flux1-kontext-dev.safetensors", "ae.safetensors"]),
            "dev_kontext",
        )

    def test_a_prequantized_mflux_package_is_recognised(self):
        # These ship no model_index.json and no telltale weight filename; the
        # component layout is the evidence, and the name only picks the
        # variant once that layout has confirmed the architecture.
        files = ["transformer/x.safetensors", "vae/x.safetensors",
                 "text_encoder/x.safetensors", "text_encoder_2/x.safetensors",
                 "tokenizer/tokenizer.json"]
        self.assertEqual(
            worker._detect_mflux_backend("dhairyashil/FLUX.1-schnell-mflux-4bit",
                                         [], files),
            "schnell")
        self.assertEqual(
            worker._detect_mflux_backend("akx/FLUX.1-Kontext-dev-mflux-4bit",
                                         [], files),
            "dev_kontext")

    def test_the_flux_layout_alone_is_not_enough(self):
        # Same shape, nothing saying FLUX: claiming it would load weights
        # through an architecture they were not trained for.
        files = ["transformer/x.safetensors", "vae/x.safetensors",
                 "text_encoder/x.safetensors", "text_encoder_2/x.safetensors"]
        self.assertIsNone(
            worker._detect_mflux_backend("someone/mystery-model", [], files))

    def test_flux2_is_left_to_its_own_router(self):
        # FLUX.2 is a different architecture with a router of its own.
        files = ["transformer/x.safetensors", "vae/x.safetensors",
                 "text_encoder/x.safetensors", "text_encoder_2/x.safetensors"]
        for repo in ("AbstractFramework/flux.2-klein-4b-4bit",
                     "owner/FLUX2-klein-9b"):
            with self.subTest(repo=repo):
                self.assertIsNone(worker._detect_mflux_backend(repo, [], files))

    def test_not_flux_is_not_claimed(self):
        # Acceptance is not identification: guessing here would load weights
        # through an architecture they were not trained for.
        self.assertIsNone(worker._detect_mflux_backend(
            "runwayml/stable-diffusion-v1-5", ["text-to-image"],
            ["v1-5-pruned.safetensors", "model_index.json"]))


class SuggestedAlternatives(unittest.TestCase):
    """A route that names the remedy must not have it taken away.

    Wan VACE answers "does not take image_path; pass reference_image_paths
    instead". Dropping the parameter it just asked for left video generation
    running as though no picture had been supplied, so a photo produced a clip
    with nothing to do with it.
    """

    def test_the_suggested_parameter_survives(self):
        opt = {"reference_image_paths": ["/p.png"], "image_path": "/p.png"}
        dropped = worker._drop_named(
            opt, "Wan VACE does not take image_path; pass reference_image_paths instead.")
        self.assertEqual(dropped, "image_path")
        self.assertIn("reference_image_paths", opt)

    def test_other_phrasings_of_a_suggestion(self):
        for message in (
            "image_path is unsupported; use reference_image_paths instead.",
            "Do not pass image_path -- supply 'reference_image_paths'.",
            "image_path rejected, try reference_image_paths",
        ):
            with self.subTest(message=message):
                opt = {"reference_image_paths": ["/p.png"], "image_path": "/p.png"}
                self.assertEqual(worker._drop_named(opt, message), "image_path")
                self.assertIn("reference_image_paths", opt)

    def test_a_plain_rejection_still_drops(self):
        # Nothing suggested: the ordinary path must keep working.
        opt = {"guidance": 3.5, "width": 512}
        self.assertEqual(
            worker._drop_named(opt, "'guidance' is not a parameter of this route."),
            "guidance")



class Clarification(unittest.TestCase):
    """Making a request precise is not the same as writing a scene.

    The old enhancer was told it was "directing a photograph", so it invented
    one: "make it look better" came back as a painting in a dark room with a
    large window, none of which the user had asked for. With a T5 encoder
    every invented noun is something the picture actually contains.
    """

    def test_a_request_that_names_nothing_is_refused(self):
        self.assertIsNone(worker._clarified("make it look better", "UNCLEAR"))
        self.assertIsNone(worker._clarified("something nice", "  unclear  "))

    def test_an_inflected_verb_still_counts_as_kept(self):
        # "rises" and "rising" share three letters, and a four-letter prefix
        # decided the word had been dropped -- so a correct rewrite of any
        # prompt containing a verb was thrown away.
        for original, rewrite in (
            ("steam rises from the teapot", "steam rising from a glazed teapot"),
            ("a man runs", "a man running along a wet road"),
            ("she moves closer", "she moving closer, one hand out"),
            ("leaves fall", "leaves falling through cold air"),
            ("he carries a bag", "he carrying a canvas bag"),
        ):
            with self.subTest(original=original):
                self.assertIsNotNone(worker._clarified(original, rewrite))

    def test_a_dropped_subject_is_rejected(self):
        # "an old bicycle against a brick wall" came back without the wall.
        self.assertIsNone(worker._clarified(
            "an old bicycle against a brick wall",
            "an old bicycle with a rusted steel frame"))

    def test_everything_named_is_kept(self):
        out = worker._clarified(
            "a cat on a chair",
            "a tabby cat with dense grey-brown fur, curled on a worn oak chair")
        self.assertIsNotNone(out)
        self.assertIn("cat", out)
        self.assertIn("chair", out)

    def test_a_whole_invented_paragraph_is_rejected(self):
        # Far longer than the request means it stopped clarifying.
        bloat = ("a cat, a book, a window, a rug, a fireplace, a lamp, a vase, "
                 "a painting, a bookshelf, a rocking chair, a cushion, a blanket, "
                 "a teapot, a clock, a mirror, a plant, a basket, a candle")
        self.assertIsNone(worker._clarified("a cat", bloat))

    def test_a_preamble_is_skipped(self):
        out = worker._clarified(
            "a red car",
            "Sure! Here is the rewrite:\na red car with sun-faded paint")
        self.assertEqual(out, "a red car with sun-faded paint.")

    def test_a_short_request_may_still_expand(self):
        # One significant word legitimately expands; a ratio alone rejected it.
        out = worker._clarified(
            "a teapot",
            "a glazed stoneware teapot with a chipped spout and crazed white glaze")
        self.assertIsNotNone(out)


class ShotListParsing(unittest.TestCase):
    """Getting the panels back out of whatever the writer decided to say.

    Models fence JSON in backticks and preface it with a sentence however
    firmly they are told not to, so the array is located rather than assumed.
    """

    GOOD = ('[{"shot":"wide","subject":"Mira","action":"stands in the hallway",'
            '"setting":"a flat"},'
            '{"shot":"close-up","subject":"a tap","action":"water running",'
            '"setting":"the kitchen"}]')

    def test_a_bare_array_parses(self):
        panels = worker._parse_shotlist(self.GOOD, 2)
        self.assertEqual(len(panels), 2)
        self.assertEqual(panels[0]["shot"], "wide")
        self.assertEqual(panels[1]["setting"], "the kitchen")

    def test_a_fenced_array_parses(self):
        panels = worker._parse_shotlist(f"```json\n{self.GOOD}\n```", 2)
        self.assertEqual(len(panels), 2)

    def test_a_preface_is_ignored(self):
        panels = worker._parse_shotlist(
            f"Sure! Here are the panels:\n{self.GOOD}\nLet me know!", 2)
        self.assertEqual(len(panels), 2)

    def test_extra_panels_are_trimmed_to_what_was_asked(self):
        panels = worker._parse_shotlist(self.GOOD, 1)
        self.assertEqual(len(panels), 1)

    def test_who_is_in_frame_is_carried_through(self):
        panels = worker._parse_shotlist(
            '[{"shot":"close-up","subject":"a tap","action":"water running",'
            '"setting":"the kitchen","character_in_frame":false}]', 1)
        self.assertFalse(panels[0]["character_in_frame"])

    def test_a_missing_in_frame_flag_keeps_the_character(self):
        # A board is mostly about its character; a missing flag must not
        # quietly write them out of the panel.
        panels = worker._parse_shotlist(
            '[{"shot":"wide","subject":"Mira","action":"stands",'
            '"setting":"a hallway"}]', 1)
        self.assertTrue(panels[0]["character_in_frame"])

    def test_a_caption_is_kept_and_tidied(self):
        panels = worker._parse_shotlist(
            '[{"shot":"wide","subject":"Mira","action":"stands","setting":"a hall",'
            '"caption":"  The door   was\\n open.  "}]', 1)
        self.assertEqual(panels[0]["caption"], "The door was open.")

    def test_a_missing_caption_is_empty_not_absent(self):
        # The composer indexes captions positionally against panels, so a
        # missing one has to still occupy its place.
        panels = worker._parse_shotlist(
            '[{"shot":"wide","subject":"x","action":"y","setting":"z"}]', 1)
        self.assertEqual(panels[0]["caption"], "")

    def test_a_runaway_caption_is_cut(self):
        panels = worker._parse_shotlist(
            '[{"shot":"wide","subject":"x","action":"y","setting":"z",'
            f'"caption":"{"word " * 80}"}}]', 1)
        self.assertLessEqual(len(panels[0]["caption"]), 120)

    def test_dialogue_is_normalised(self):
        panels = worker._parse_shotlist(
            '[{"shot":"medium","subject":"Josephine","action":"kneels",'
            '"setting":"a door","dialogue":[{"speaker":"Josephine",'
            '"text":"  Louise,   open the door!  "}]}]', 1)
        d = panels[0]["dialogue"]
        self.assertEqual(len(d), 1)
        self.assertEqual(d[0]["text"], "Louise, open the door!")

    def test_bare_strings_become_unattributed_lines(self):
        panels = worker._parse_shotlist(
            '[{"shot":"wide","subject":"x","action":"y","setting":"z",'
            '"dialogue":["Go away."]}]', 1)
        self.assertEqual(panels[0]["dialogue"], [{"speaker": "", "text": "Go away."}])

    def test_a_panel_with_no_dialogue_gets_an_empty_list(self):
        # The composer indexes dialogue positionally, so the slot must exist.
        panels = worker._parse_shotlist(
            '[{"shot":"wide","subject":"x","action":"y","setting":"z"}]', 1)
        self.assertEqual(panels[0]["dialogue"], [])

    def test_a_crowd_of_speakers_is_capped(self):
        # Balloons sit on top of the panel; three of them leave no panel.
        many = ",".join(f'{{"speaker":"S{i}","text":"line {i}"}}' for i in range(5))
        panels = worker._parse_shotlist(
            f'[{{"shot":"wide","subject":"x","action":"y","setting":"z",'
            f'"dialogue":[{many}]}}]', 1)
        self.assertLessEqual(len(panels[0]["dialogue"]), 2)


class SceneNumbering(unittest.TestCase):
    """Which scene a panel belongs to decides where pages break, so a missing
    or malformed number must not silently become a page boundary."""

    def test_a_scene_number_is_kept(self):
        panels = worker._parse_shotlist(
            '[{"shot":"wide","subject":"x","action":"y","setting":"z",'
            '"scene":3,"scene_title":"The platform"}]', 1)
        self.assertEqual(panels[0]["scene"], 3)
        self.assertEqual(panels[0]["scene_title"], "The platform")

    def test_a_missing_scene_defaults_to_the_first(self):
        panels = worker._parse_shotlist(
            '[{"shot":"wide","subject":"x","action":"y","setting":"z"}]', 1)
        self.assertEqual(panels[0]["scene"], 1)

    def test_a_nonsense_scene_number_does_not_break_pagination(self):
        panels = worker._parse_shotlist(
            '[{"shot":"wide","subject":"x","action":"y","setting":"z","scene":0}]', 1)
        self.assertGreaterEqual(panels[0]["scene"], 1)


class TierLayout(unittest.TestCase):
    """A page is read as horizontal bands, and the bands set the pace.

    A wide shot takes its whole tier -- that is what makes it establishing,
    and a large panel is what slows a reader down. Tighter shots share one,
    which speeds the page up.
    """

    def test_a_wide_takes_its_own_tier(self):
        self.assertEqual(worker._tiers(["wide", "wide"]), [[0], [1]])

    def test_tighter_shots_share_a_tier(self):
        self.assertEqual(worker._tiers(["medium", "close-up"]), [[0, 1]])

    def test_a_tier_holds_at_most_three(self):
        tiers = worker._tiers(["medium"] * 5)
        self.assertTrue(all(len(t) <= 3 for t in tiers), tiers)

    def test_a_wide_interrupts_a_group(self):
        self.assertEqual(
            worker._tiers(["medium", "wide", "medium", "close-up"]),
            [[0], [1], [2, 3]])

    def test_no_panel_is_stranded_alone_at_the_end(self):
        # Three then one reads as a mistake; two and two reads as a pair.
        self.assertEqual(worker._tiers(["medium"] * 4), [[0, 1], [2, 3]])

    def test_every_panel_appears_exactly_once(self):
        shots = ["wide", "medium", "close-up", "medium", "wide", "close-up", "medium"]
        flat = [i for t in worker._tiers(shots) for i in t]
        self.assertEqual(sorted(flat), list(range(len(shots))))


class Pagination(unittest.TestCase):
    """Pages break where scenes do, because a scene starting halfway down a
    page reads as a jump rather than a change of place."""

    def test_one_scene_of_readable_length_is_one_page(self):
        self.assertEqual(worker._paginate([1, 1, 1, 1]), [[0, 1, 2, 3]])

    def test_a_new_scene_starts_a_new_page(self):
        self.assertEqual(worker._paginate([1, 1, 1, 2, 2, 2]),
                         [[0, 1, 2], [3, 4, 5]])

    def test_a_long_scene_splits_evenly(self):
        # Greedy chunking would leave the last page holding one panel, which
        # reads as the scene trailing off.
        pages = worker._paginate([1] * 10)
        self.assertTrue(all(len(p) >= worker.PAGE_MIN for p in pages), pages)
        self.assertEqual(sum(len(p) for p in pages), 10)

    def test_a_lone_panel_joins_the_page_before_it(self):
        # A single panel alone on a page is an accident, not a splash.
        pages = worker._paginate([1, 1, 1, 2])
        self.assertNotIn(1, [len(p) for p in pages])

    def test_every_panel_lands_on_exactly_one_page(self):
        scenes = [1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4]
        flat = [i for page in worker._paginate(scenes) for i in page]
        self.assertEqual(flat, list(range(len(scenes))))

    def test_no_page_exceeds_the_maximum(self):
        pages = worker._paginate([1] * 17)
        self.assertTrue(all(len(p) <= worker.PAGE_MAX for p in pages), pages)


class ModuleIntegrity(unittest.TestCase):
    """Names the module uses must exist.

    A constant was once deleted by a block rewrite whose replacement ran to
    the next `def`, and the function referencing it only fails when it runs --
    which needs a model, so no test reached it and the break shipped. This
    reads the module rather than running it.
    """

    def _tree(self):
        import ast

        return ast.parse(open(os.path.join(HERE, "worker.py")).read())

    def test_no_module_constant_is_referenced_without_being_defined(self):
        import ast
        import builtins

        tree = self._tree()
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                defined |= {t.id for t in node.targets if isinstance(t, ast.Name)}
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                defined.add(node.target.id)
            elif isinstance(node, ast.Tuple) and isinstance(
                    getattr(node, "ctx", None), ast.Store):
                defined |= {e.id for e in node.elts if isinstance(e, ast.Name)}
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                defined |= {(a.asname or a.name).split(".")[0] for a in node.names}

        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                and n.id.isupper() and len(n.id) > 3}
        missing = sorted(used - defined - set(dir(builtins)))
        self.assertEqual(missing, [], f"referenced but never defined: {missing}")

    def test_every_dispatched_op_exists(self):
        import ast

        tree = self._tree()
        functions = {n.name for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef)}
        dispatched = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "OPS" for t in node.targets):
                for v in node.value.values:
                    if isinstance(v, ast.Name):
                        dispatched.add(v.id)
        self.assertTrue(dispatched, "the op table was not found")
        self.assertEqual(sorted(dispatched - functions), [])

if __name__ == "__main__":
    unittest.main(verbosity=2)
