"""Tests for the engine's pure logic.

Run with the app's own interpreter:

    ~/Library/Application\\ Support/com.melp.modelstudio/runtime/venv/bin/python3 \\
        -m unittest discover -s engine -v

These cover the parts that decide *what gets sent to a model* and *what gets
written to disk* — the places where a silent mistake changes a result rather
than raising. Loading actual weights is out of scope here.
"""

import importlib.util
import json
import io
import os
import secrets
import struct
import sys
import tempfile
import unittest
import uuid

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
            with Image.open(out[0]) as reduced:
                self.assertEqual(max(reduced.size), worker.ASSIST_MAX_EDGE)

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
        return {"id": str(uuid.uuid4()), "file_id": fid.hex(), "key": key.hex(),
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
                         if f.startswith(entry["id"])]
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

    def test_empty_quality_labels_are_removed_before_rewriting(self):
        cleaned, removed = worker._clean_prompt_request(
            "masterpiece, 8k, a red bicycle, highly detailed")
        self.assertEqual(cleaned, "a red bicycle")
        self.assertEqual(set(removed), {"masterpiece", "8k", "highly detailed"})

    def test_quality_cleanup_keeps_descriptive_content(self):
        cleaned, _ = worker._clean_prompt_request(
            "a weathered red bicycle against a brick wall")
        self.assertEqual(cleaned, "a weathered red bicycle against a brick wall")


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

    def test_source_quotes_and_story_characters_are_preserved(self):
        raw = ('[{"shot":"wide","subject":"Mira","action":"enters",'
               '"setting":"home","character_in_frame":true,'
               '"characters":["Mira","Invented"],'
               '"source":"Mira comes home"}]')
        panels = worker._parse_shotlist(raw, 1, "Mira comes home late.")
        self.assertEqual(panels[0]["source"], "Mira comes home")
        self.assertEqual(panels[0]["characters"], ["Mira"])

    def test_coverage_reports_unanchored_passages(self):
        story = "Mira opens the door. The photographs face the wall. She freezes."
        panels = [{"source": "Mira opens the door"}, {"source": "She freezes"}]
        coverage = worker._story_coverage(story, panels)
        self.assertEqual(coverage["covered"], 2)
        self.assertEqual(coverage["total"], 3)
        self.assertEqual(coverage["missing"], ["The photographs face the wall."])

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

    def test_never_more_pages_than_panels(self):
        """The bound the Rust host reserves page slots against.

        `compose_board` reserves one vault slot per panel and refuses to
        compose if pagination wants more pages than it reserved. It used to
        reserve `ceil(panels / 3)`, on the reasoning that three panels is the
        smallest readable page -- but PAGE_MIN only applies when splitting one
        long scene, and a scene boundary ends a page wherever it falls. Three
        two-panel scenes make three pages out of six panels, so the reservation
        came up short and the board was refused on its final step with "ask for
        that many again", which names no control the interface has.

        Exhaustive over every scene composition up to 12 panels, because the
        shapes that break it are ordinary ones: prose that changes location
        often produces short scenes, and nothing renumbers them.
        """
        def compositions(n):
            """Every way to cut n panels into consecutive scene runs."""
            if n == 0:
                yield []
                return
            for first in range(1, n + 1):
                for rest in compositions(n - first):
                    yield [first] + rest

        for n in range(1, 13):
            for runs in compositions(n):
                scenes = []
                for scene, length in enumerate(runs, start=1):
                    scenes.extend([scene] * length)
                pages = worker._paginate(scenes)
                self.assertLessEqual(
                    len(pages), len(scenes),
                    f"{len(pages)} pages from {len(scenes)} panels, scenes={scenes}",
                )
                # And still a partition: nothing dropped, nothing duplicated.
                self.assertEqual(
                    [i for page in pages for i in page], list(range(len(scenes))),
                    f"pagination is not a partition for scenes={scenes}",
                )


class TileGeometry(unittest.TestCase):
    """Tiles must cover the output canvas exactly.

    Placement used int(x * factor) while each piece was sized by the model's
    own rounding, so the far edge could be left a pixel short. Nothing covers
    that column, the blend divides it by ~0, and a black line appears down the
    side of the finished picture. This checks the grid arithmetic directly,
    across sizes and scales, without needing a model.
    """

    def _cells(self, in_w, in_h, factor, tile_src, overlap_src):
        xs = worker._tile_spans(in_w, min(tile_src, in_w), overlap_src)
        ys = worker._tile_spans(in_h, min(tile_src, in_h), overlap_src)
        out_w, out_h = round(in_w * factor), round(in_h * factor)
        cells = []
        for (y0, y1) in ys:
            for (x0, x1) in xs:
                px0, px1 = round(x0 * factor), round(x1 * factor)
                py0, py1 = round(y0 * factor), round(y1 * factor)
                px1, py1 = min(px1, out_w), min(py1, out_h)
                if px1 > px0 and py1 > py0:
                    cells.append((px0, py0, px1, py1))
        return out_w, out_h, cells

    def test_every_output_pixel_is_covered(self):
        # Deliberately plain Python. The engine's own runtime has numpy, but
        # the interpreter that runs these tests in CI does not, and a test that
        # only runs on one machine is not a test.
        cases = [
            (512, 512, 2.0), (1024, 1024, 2.0), (1000, 667, 2.0),
            (513, 397, 2.0), (640, 480, 4.0), (777, 1013, 3.0),
            (256, 256, 2.0), (1920, 1080, 2.0),
        ]
        for in_w, in_h, factor in cases:
            tile_src = max(64, int(744 / factor))
            out_w, out_h, cells = self._cells(
                in_w, in_h, factor, tile_src, int(64 / factor))
            # Rows and columns are covered independently, so checking each
            # axis is equivalent to checking the grid and is far cheaper than
            # materialising one.
            cols = [False] * out_w
            rows = [False] * out_h
            for (px0, py0, px1, py1) in cells:
                for x in range(px0, px1):
                    cols[x] = True
                for y in range(py0, py1):
                    rows[y] = True
            missing = cols.count(False) + rows.count(False)
            self.assertEqual(
                missing, 0,
                f"{in_w}x{in_h} @{factor}x leaves {cols.count(False)} columns "
                f"and {rows.count(False)} rows uncovered -- those become a "
                f"black seam")

    def test_neighbouring_cells_overlap_so_the_blend_has_room(self):
        out_w, out_h, cells = self._cells(1000, 667, 2.0, 372, 32)
        self.assertGreater(len(cells), 1, "expected more than one tile")
        xs = sorted({(c[0], c[2]) for c in cells})
        for (a0, a1), (b0, b1) in zip(xs, xs[1:]):
            self.assertLess(b0, a1, "columns must overlap to blend")


class TiledUpscaleCancels(unittest.TestCase):
    """Cancel must stop a tiled upscale.

    Each tile is one opaque call into the model, so the loop is the only place
    it can stop. Without a check there, Cancel marked the job cancelled, the
    interface stopped showing it, and the engine kept working through every
    remaining tile -- which is indistinguishable from a hang.
    """

    def test_the_tile_loop_checks_for_cancellation(self):
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_upscale_tiled")
        body = ast.get_source_segment(src, fn) or ""
        self.assertIn("is_cancelled", body,
                      "_upscale_tiled must check cancellation between tiles")
        # and it must actually raise, not merely look
        self.assertIn("raise Cancelled()", body)

    def test_every_loop_over_model_calls_can_be_stopped(self):
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        tree = ast.parse(src)
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            if not any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(fn)):
                continue
            body = ast.get_source_segment(src, fn) or ""
            if not any(k in body for k in ("generate_image", "generate_video")):
                continue
            # Either an explicit check, or the progress handler that raises.
            self.assertTrue(
                "is_cancelled" in body or "_make_progress_handler" in body,
                f"{fn.name} loops over model calls with no way to stop it")


class VideoCanvasFloor(unittest.TestCase):
    """A canvas too small for the model must be refused, not drawn on.

    Wan returns coloured noise below roughly 640x368 rather than a worse
    picture, and the output is a perfectly valid MP4 every time -- right
    dimensions, right frame count, sealed without complaint. Nothing
    downstream can tell the difference, so the size has to be caught here.
    """

    def test_the_sizes_that_produced_noise_are_refused(self):
        for w, h in ((320, 192), (384, 224), (224, 384), (320, 320), (480, 272)):
            with self.assertRaises(ValueError, msg=f"{w}x{h} was allowed"):
                worker._check_video_size("t", w, h)

    def test_the_sizes_that_produced_pictures_are_allowed(self):
        for w, h in ((640, 368), (704, 384), (832, 480), (1280, 704)):
            worker._check_video_size("t", w, h)

    def test_the_floor_is_where_it_was_measured(self):
        # Between 480x272, which was noise, and 640x368, which was not.
        self.assertGreater(worker._VIDEO_MIN_PIXELS, 480 * 272)
        self.assertLessEqual(worker._VIDEO_MIN_PIXELS, 640 * 368)

    def test_the_message_says_what_to_do(self):
        try:
            worker._check_video_size("t", 320, 192)
        except ValueError as e:
            text = str(e)
            self.assertIn("640", text, "should name a size that works")
            self.assertIn("noise", text, "should say what goes wrong")
            self.assertIn("time", text, "should say the cost is time, not memory")


class MemoryHeadroom(unittest.TestCase):
    """The guard must not refuse work that would have succeeded.

    Free-plus-inactive badly understates what macOS can reclaim: a measured
    9.72 GiB image-to-video run completed on this machine with 5.2 GiB
    reported free. A guard that compared need against that figure would block
    almost every real run, which is a worse failure than the one it prevents.
    """

    def test_it_reads_the_machine_without_blowing_up(self):
        free = worker._free_ram_gib()
        self.assertTrue(free == -1.0 or 0.0 <= free < 1024.0,
                        f"implausible free memory: {free}")

    def test_a_run_that_fits_is_not_refused(self):
        # The measured case: 9.72 GiB needed, 5.2 GiB reported free, succeeded.
        self.assertLess(worker._MIN_FREE_GIB, 5.2,
                        "the threshold would have refused a run that worked")

    def test_the_threshold_is_where_aborts_actually_happened(self):
        # Every observed engine abort had free memory near zero.
        self.assertGreater(worker._MIN_FREE_GIB, 0.5)
        self.assertLessEqual(worker._MIN_FREE_GIB, 2.0)

    def test_a_need_of_zero_never_refuses_on_its_own(self):
        # Callers that do not know their peak must not be blocked by that.
        try:
            worker._check_headroom("t", 0.0)
        except ValueError as e:
            # Only acceptable if the machine really is out of memory now.
            self.assertLess(worker._free_ram_gib(), worker._MIN_FREE_GIB, str(e))


class EnhancerFixesSpelling(unittest.TestCase):
    """The thing anyone means by "enhancer", which it could not do.

    The retention check counted a *corrected* word as a *lost* word, so every
    spelling fix was rejected and the typo came back untouched. Measured: all
    three of three misspelled requests had their correct rewrite thrown away.
    """

    TYPOS = [
        ("a pictuer of a hous at nite",
         "a house at night, warm light in the windows, the road wet and black"),
        ("a wonan in a red dres",
         "a woman in a red satin dress, standing in a doorway"),
        ("a teapot on a tabel",
         "a beige ceramic teapot on a scrubbed pine table"),
        ("me and him was walking in forrest",
         "two people walking through a pine forest, mist between the trunks"),
    ]

    # Both words real, so a string comparison cannot tell a correction from a
    # substitution -- and guessing draws the wrong picture.
    SWAPS = [
        ("a cat on a chair", "a hat on a chair"),
        ("a bird on a wire", "a bard on a wire"),
        ("a dog in the snow", "a log in the snow"),
        ("a horse in a field", "a house in a field"),
    ]

    def test_a_corrected_spelling_is_not_a_lost_word(self):
        for typed, rewrite in self.TYPOS:
            out, why = worker._clarified_with_reason(typed, rewrite)
            self.assertIsNotNone(out, f"{typed!r} rejected as {why}")

    def test_a_different_word_is_still_a_lost_word(self):
        # Two earlier attempts at this were worse than the bug. Plain edit
        # distance let "a cat on a chair" come back as "a hat on a chair" and
        # called it a spelling fix.
        for typed, rewrite in self.SWAPS:
            out, _ = worker._clarified_with_reason(typed, rewrite)
            self.assertIsNone(out, f"{typed!r} -> {rewrite!r} was accepted")

    def test_a_real_word_is_never_a_misspelling_of_another(self):
        self.assertFalse(worker._misspelling_of("horse", "house"))
        self.assertFalse(worker._misspelling_of("setting", "sitting"))
        self.assertTrue(worker._misspelling_of("hous", "house"))
        self.assertTrue(worker._misspelling_of("tabel", "table"))
        self.assertTrue(worker._misspelling_of("nite", "night"))

    def test_without_a_dictionary_it_stays_strict(self):
        # No word list means no guessing: exact and prefix matching only, which
        # is where this started and is never wrong, only sometimes strict.
        saved = worker._DICTIONARY
        try:
            worker._DICTIONARY = set()
            self.assertFalse(worker._misspelling_of("tabel", "table"))
            self.assertFalse(worker._misspelling_of("horse", "house"))
        finally:
            worker._DICTIONARY = saved

    def test_the_corrections_are_reported(self):
        pairs = worker._corrections("a hous at nite", "a house at night")
        self.assertIn(["hous", "house"], pairs)
        self.assertIn(["nite", "night"], pairs)

    def test_a_disagreed_word_is_not_appended_as_nonsense(self):
        # Appending the dropped word produced "...sitting on a worn oak chair,
        # setting." -- dangling off the end, meaning nothing, on every rewrite
        # that corrected a homophone.
        self.assertIsNone(worker._repair_dropped(
            "a cat setting on a chair",
            "a tabby cat with dense fur, sitting on a worn oak chair"))

    def test_a_paraphrase_is_still_repaired(self):
        fixed = worker._repair_dropped(
            "a rainy street", "a wet street, slick asphalt under grey light")
        self.assertIsNotNone(fixed)
        self.assertIn("rainy", fixed)


class AdapterMustHoldAnAdapter(unittest.TestCase):
    """A "LoRA" with no adapter weights in it is refused, loudly.

    Nothing checked. `lora_info` listed every safetensors file in a repository
    with its size and offered them all, so a VAE repository was
    indistinguishable from an adapter repository: it downloaded, it appeared
    installed, it could be selected and given a strength, and then it changed
    nothing -- while the log said "adapters: ... @ 1.0".
    """

    def test_a_vae_is_named_for_what_it_is(self):
        keys = ["decoder.conv_in.weight", "decoder.conv_out.bias",
                "encoder.conv_in.weight"]
        self.assertIn("VAE", worker._describe_weights(keys))

    def test_a_full_model_is_named_for_what_it_is(self):
        keys = [f"transformer.blocks.{i}.attn.weight" for i in range(500)]
        self.assertIn("full model", worker._describe_weights(keys))

    def test_an_unresolvable_handle_is_not_refused(self):
        # A repo that is not on this machine is the loader's business, not
        # this check's: refusing it would block adapters that work.
        worker._check_is_adapter("t", "nobody/nothing:absent.safetensors")


class PanelRangeStaysARange(unittest.TestCase):
    def test_the_bottom_never_passes_the_top(self):
        # Only the top was capped at sixty, so past about 8,400 words the
        # bottom overtook it. The writer was told a chapter "usually lands
        # between 714 and 60 panels", and no count can sit inside that, so
        # every division of a long story was reported as suspect.
        for words in (0, 1, 300, 3000, 8400, 9000, 100_000):
            low, high = worker._panel_bounds(words)
            self.assertLessEqual(low, high, f"inverted at {words} words")
            self.assertGreaterEqual(low, 2)
            self.assertLessEqual(high, worker._MAX_PANELS)


class DivisionSurvivesAnImperfectWriter(unittest.TestCase):
    """A quote retyped is still a quote, and one odd field is not fatal."""

    STORY = ("Mira didn\u2019t look back. She pulled the door shut, and the "
             "hallway swallowed the sound. Outside, the rain had already "
             "started.")

    def test_a_retyped_quote_is_recognised(self):
        # Straight apostrophe for a curly one, and a dropped comma. Both
        # erased the quote, and the board then told the user those sentences
        # of their story had been left out -- while drawing them.
        panels = worker._parse_shotlist(json.dumps([
            {"shot": "medium", "subject": "Mira",
             "source": "Mira didn't look back.", "scene": 1},
            {"shot": "wide", "subject": "hallway",
             "source": "She pulled the door shut and the hallway swallowed "
                       "the sound.", "scene": 1},
        ]), 2, self.STORY)
        self.assertTrue(all(p["source"] for p in panels),
                        "a real quote was thrown away")
        cover = worker._story_coverage(self.STORY, panels)
        # Only the third sentence has no panel; the two that do are covered.
        self.assertEqual(cover["missing"],
                         ["Outside, the rain had already started."])

    def test_an_invented_quote_is_still_refused(self):
        panels = worker._parse_shotlist(json.dumps([
            {"shot": "wide", "subject": "x", "scene": 1,
             "source": "A dragon circled the tower and screamed."},
        ]), 1, self.STORY)
        self.assertEqual(panels[0]["source"], "")

    def test_a_scene_named_in_words_is_read(self):
        self.assertEqual(worker._as_scene("two"), 2)
        self.assertEqual(worker._as_scene("Scene 3"), 3)
        self.assertEqual(worker._as_scene(""), 1)
        self.assertEqual(worker._as_scene(None), 1)
        self.assertEqual(worker._as_scene(-4), 1)

    def test_one_unreadable_panel_does_not_lose_the_others(self):
        # This raised ValueError out of the whole parse, so a single odd
        # field cost the user every panel and the minutes spent making them.
        panels = worker._parse_shotlist(json.dumps([
            {"shot": "wide", "subject": "a", "scene": 1},
            {"shot": "wide", "subject": "b", "scene": "somewhere"},
            {"shot": "wide", "subject": "c", "scene": 2},
        ]), 3, "")
        self.assertEqual(len(panels), 3)


class EnhancerAcceptsGoodRewrites(unittest.TestCase):
    """The validators must not throw away work the model got right.

    Measured against eight ordinary requests, the expansion caps rejected five
    good rewrites -- "a red car with sun-faded paint and a dented wing, water
    running down the windows" was refused for being too long. A three word
    request becoming twenty-five words is the reason someone pressed the
    button, not evidence the model overreached. Retention stays strict; the
    caps do not.
    """

    def test_a_short_request_may_expand_a_long_way(self):
        out, why = worker._clarified_with_reason(
            "a bowl of ramen",
            "a bowl of ramen with clear broth, pale yellow noodles, a soft white "
            "egg floating in the centre, and dark strips of nori resting on the rim.")
        self.assertEqual(why, "ok", f"rejected as {why}")
        self.assertIsNotNone(out)

    def test_detail_added_to_a_named_subject_is_kept(self):
        out, why = worker._clarified_with_reason(
            "a red car in the rain at night",
            "a red car with sun-faded paint and a dented wing, sitting in a "
            "puddled parking lot at night, rain sliding down the windows.")
        self.assertEqual(why, "ok", f"rejected as {why}")

    def test_what_the_user_wrote_still_has_to_survive(self):
        # The loosened caps must not weaken the guarantee that matters.
        out, why = worker._clarified_with_reason(
            "an old bicycle against a brick wall",
            "an ornate bronze sundial on a lawn, with a stone birdbath beside it.")
        self.assertIsNone(out, "a rewrite about something else was accepted")

    def test_every_outcome_has_a_name(self):
        # The interface reports what happened rather than inferring it from
        # whether the text came back unchanged.
        for why in ("dropped", "lost_intent", "too_long", "rejected", "empty"):
            self.assertIn(why, worker._WHY_REJECTED, f"{why} has no explanation")


class CleanPromptRequest(unittest.TestCase):
    """The cleaner must remove folklore, not meaning.

    It removed "beautiful", "professional" and "perfect", which modify the
    subject: "a professional kitchen" is a kind of kitchen and "a beautiful
    ruin" is a judgement the picture has to carry. Worse, the removal happened
    before the intent guard, so the guard never saw the words and the reply
    said nothing had been taken out. Video prompts go through the same path.
    """

    def test_folklore_goes(self):
        for text, gone in (
            ("a harbour at dawn, 8k, masterpiece", {"8k", "masterpiece"}),
            ("a fox, trending on artstation", {"trending on artstation"}),
            ("a street, ultra detailed, unreal engine",
             {"ultra detailed", "unreal engine"}),
        ):
            cleaned, removed = worker._clean_prompt_request(text)
            self.assertEqual(set(removed), gone, text)
            for g in gone:
                self.assertNotIn(g, cleaned.lower())

    def test_words_that_modify_the_subject_stay(self):
        for text in (
            "a professional kitchen",
            "a beautiful ruin at dusk",
            "perfect symmetry in a stairwell",
            "a stunning view over the bay",
            "an amazing race through the market",
            "a gorgeous tiled floor",
        ):
            cleaned, removed = worker._clean_prompt_request(text)
            self.assertEqual(cleaned, text, f"{text!r} was altered")
            self.assertEqual(removed, [], f"{text!r} reported removals")

    def test_a_prompt_of_pure_folklore_is_left_for_the_caller(self):
        cleaned, removed = worker._clean_prompt_request("masterpiece, 8k, hdr")
        self.assertEqual(cleaned, "")
        self.assertTrue(removed)


class RepairDroppedWords(unittest.TestCase):
    """A paraphrase is repaired; a deletion is not.

    Rejecting a rewrite because it said "wet" instead of "rainy" hands the
    user back their own prompt unchanged, and they reasonably conclude the
    feature does nothing. Pasting a whole missing clause back on, though,
    papers over the failure the retention check exists to catch.
    """

    def test_one_paraphrased_word_is_put_back(self):
        out = worker._repair_dropped(
            "a woman on a rainy street",
            "a woman in a navy coat walking on a wet city street.")
        self.assertIsNotNone(out)
        self.assertIn("rainy", out)
        self.assertIn("navy coat", out, "the description must survive the repair")
        self.assertEqual(worker._dropped_words("a woman on a rainy street", out), [])

    def test_a_dropped_clause_is_still_rejected(self):
        # The rewrite lost the wall entirely. Three words is not a paraphrase.
        out = worker._repair_dropped(
            "an old bicycle against a brick wall",
            "an old bicycle with a rusted steel frame and cracked tires.")
        self.assertIsNone(out, "a deleted clause must not be pasted back on")

    def test_nothing_missing_is_not_a_repair(self):
        self.assertIsNone(worker._repair_dropped(
            "a tabby cat on a chair",
            "a tabby cat curled on a worn oak chair."))

    def test_dropped_words_understands_inflection(self):
        self.assertEqual(
            worker._dropped_words("the sun rises", "the sun rising over water."), [])
        self.assertEqual(
            worker._dropped_words("a rainy street", "rain falling on the street."), [])


class ClarifyReasons(unittest.TestCase):
    """A rejected rewrite is not a vague request.

    `_clarified` collapsed six different outcomes into None, and op_assist
    reported all of them as "this does not say what to draw yet". So a request
    that named a subject perfectly well was blamed for the model's bad answer.
    """

    def test_unclear_is_reported_as_unclear(self):
        _, why = worker._clarified_with_reason("make it better", "UNCLEAR")
        self.assertEqual(why, "unclear")

    def test_a_dropped_noun_is_blamed_on_the_rewrite(self):
        # The request names a wall; the rewrite loses it.
        out, why = worker._clarified_with_reason(
            "an old bicycle against a brick wall",
            "a rusted old bicycle leaning on its kickstand.")
        self.assertIsNone(out)
        self.assertNotEqual(why, "unclear",
                            "a dropped noun must not be called a vague request")

    def test_a_good_rewrite_is_accepted(self):
        out, why = worker._clarified_with_reason(
            "a tabby cat on a chair",
            "a tabby cat curled on a worn oak chair.")
        self.assertEqual(why, "ok")
        self.assertIsNotNone(out)
        self.assertIn("cat", out)
        self.assertIn("chair", out)

    def test_the_old_name_still_returns_just_the_text(self):
        self.assertIsNone(worker._clarified("make it better", "UNCLEAR"))
        self.assertIsInstance(
            worker._clarified("a tabby cat on a chair",
                              "a tabby cat curled on a worn oak chair."), str)


class AssistOwnsOnlyWhatItMade(unittest.TestCase):
    """op_assist must never delete a file the caller owns.

    It decrypts vault inputs into temporary files and deletes them on the way
    out, which is right. When it also learned to accept a plaintext `images`
    path, that path briefly joined the same list -- and the downscaler returns
    its input unchanged when the picture is already small, so a caller's own
    file could end up in the delete list. This checks the source survives.
    """

    def test_a_plaintext_source_survives(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "source.png")
            Image.new("RGB", (64, 64), (10, 20, 30)).save(src)

            # Exercise the ownership split without loading a vision model.
            direct = [src]
            raw_staged = []
            staged = worker._downscale_for_assist(direct + raw_staged)
            derived = [p for p in staged
                       if p not in direct and p not in raw_staged]
            worker._discard_staged(derived)
            worker._discard_staged(raw_staged)

            self.assertTrue(os.path.exists(src),
                            "op_assist deleted the caller's own image")

    def test_a_large_source_survives_and_its_copy_does_not(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "big.png")
            Image.new("RGB", (worker.ASSIST_MAX_EDGE * 2,
                              worker.ASSIST_MAX_EDGE * 2), (5, 5, 5)).save(src)
            direct = [src]
            staged = worker._downscale_for_assist(direct)
            derived = [p for p in staged if p not in direct]
            self.assertEqual(len(derived), 1, "expected a downscaled copy")
            worker._discard_staged(derived)
            self.assertTrue(os.path.exists(src), "deleted the caller's image")
            self.assertFalse(os.path.exists(derived[0]),
                             "left its own temporary behind")


class UpscaleFits(unittest.TestCase):
    """The upscale size guard.

    Going over the memory budget here does not raise -- MLX reports the Metal
    failure from a command-buffer completion handler, which reaches
    std::terminate and aborts the engine. So the arithmetic that decides
    whether to tile is the only thing standing between a large picture and a
    dead process, and it is worth testing directly.
    """

    def test_matches_the_measurements(self):
        # Measured on a 16 GB M4, SeedVR2 3B q4. See docs/measurements.md.
        for w, h, measured in ((512, 512, 6.43), (768, 768, 10.44),
                               (1024, 1024, 16.57)):
            got = worker._upscale_peak_gib(w, h)
            self.assertLess(abs(got - measured), 0.3,
                            f"{w}x{h}: predicted {got:.2f}, measured {measured}")

    def test_one_pass_is_refused_when_it_would_abort(self):
        budget = 12.0
        self.assertGreater(worker._upscale_peak_gib(1024, 1024), budget)
        self.assertLess(worker._upscale_peak_gib(768, 768), budget)

    def test_largest_edge_keeps_proportions_and_fits(self):
        for w, h in ((1024, 1024), (1600, 900), (640, 480)):
            edge = worker._largest_upscale_edge(w, h, 12.0)
            scale = edge / max(w, h)
            self.assertLessEqual(worker._upscale_peak_gib(int(w * scale),
                                                          int(h * scale)),
                                 12.0 + 0.05)

    def test_tile_spans_cover_everything_and_overlap(self):
        for total, tile, overlap in ((1000, 372, 32), (512, 372, 32),
                                     (300, 372, 32), (2048, 400, 64)):
            spans = worker._tile_spans(total, min(tile, total), overlap)
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], total, f"{total}/{tile} leaves a gap")
            for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
                self.assertLess(b0, a1, "neighbouring tiles must overlap")
            covered = set()
            for a, b in spans:
                covered |= set(range(a, b))
            self.assertEqual(len(covered), total, "every pixel must be covered")


class VideoLoadArguments(unittest.TestCase):
    """`op_video` must pass image_count exactly once.

    It used to set it inside plan_kw and also pass it positionally as 0, which
    is a TypeError raised before any work starts -- so animating a still never
    reached a model on a first-frame route, whatever model was selected. The
    failure needs real weights to reproduce, so it is checked statically.
    """

    def test_image_count_is_not_passed_twice(self):
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        tree = ast.parse(src)
        checked = 0
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "_load_model"):
                continue
            explicit = [k.arg for k in node.keywords if k.arg]
            has_star = any(k.arg is None for k in node.keywords)
            if has_star and "image_count" in explicit:
                segment = ast.get_source_segment(src, node) or ""
                self.assertIn("pop(", segment,
                              f"line {node.lineno}: image_count is passed "
                              "explicitly next to **kwargs that may also "
                              "carry it; pop it from the dict first")
                checked += 1
        self.assertGreater(checked, 0, "expected op_video's call to be covered")


class PanelFidelity(unittest.TestCase):
    """What reaches the picture has to be what the story said.

    The division is the writer's paraphrase of the prose; the brief is the
    writer's paraphrase of the division. Neither was checked against anything,
    and `panelPrompt` *replaces* the action and setting with the brief -- so a
    brief that dropped the moment was drawn as something the story does not
    contain, and nothing anywhere noticed.
    """

    ACTION = "pours the tea"
    SUBJECT = "Mira"

    def test_a_brief_that_keeps_the_moment_is_accepted(self):
        for brief in (
            "Mira pouring tea into a white cup, steam rising. Grey window light.",
            "Mira pours tea at the stove, steam rising in grey light.",
        ):
            self.assertTrue(worker._carries(self.ACTION, brief), brief)
            self.assertTrue(worker._carries(self.SUBJECT, brief), brief)

    def test_a_brief_that_lost_the_moment_is_caught(self):
        # Every one of these passed `_keeps_intent`, which is why it is not
        # the check for this: it asks whether *any* word survived.
        for brief in (
            "A narrow kitchen, dark green walls, bare oak boards, one window.",
            "Warm hallway light falling across a narrow flat.",
            "A kitchen.",
        ):
            self.assertTrue(worker._keeps_intent(
                f"medium shot. {self.SUBJECT}. {self.ACTION}. a narrow kitchen", brief))
            self.assertFalse(worker._carries(self.ACTION, brief), brief)

    def test_carries_is_not_defeated_by_inflection(self):
        self.assertTrue(worker._carries("rises slowly", "steam rising slowly"))
        self.assertTrue(worker._carries("carries a lamp", "carrying a lamp"))
        self.assertTrue(worker._carries("", "anything at all"))

    def test_carries_needs_every_content_word(self):
        self.assertFalse(worker._carries("pours the tea", "pours the coffee"))
        self.assertFalse(worker._carries("opens the window", "opens the door"))


class SourceSnapping(unittest.TestCase):
    """A panel's anchor is the user's own sentence, or it is nothing.

    The writer is told to quote the prose and often paraphrases it instead.
    That used to be thrown away -- the anchor was blanked, the rail said no
    quote was returned, and coverage counted the sentence as missing even
    though a panel was about it. The sentence is right there in what was
    pasted, so the paraphrase is matched back onto the prose and the prose
    wins.
    """

    STORY = ("Marta opened the kitchen door. The rain had not stopped since "
             "Tuesday. She poured the tea and sat down without drinking it. "
             "She ran.")

    def _anchor(self, source):
        return worker._clean_panel(
            {"shot": "medium", "subject": "Marta", "action": "a",
             "setting": "b", "source": source, "scene": 1}, self.STORY)["source"]

    def test_every_anchor_is_verbatim_from_the_story(self):
        for source in ("She poured the tea and sat down without drinking it.",
                       "Marta pours tea and sits",
                       "Marta opens the door of the kitchen",
                       "rain since Tuesday",
                       "She ran.",
                       "Marta boarded a train to Leeds",
                       "Marta", "the", ""):
            got = self._anchor(source)
            if got:
                self.assertIn(got, self.STORY,
                              f"{source!r} produced an anchor that is not in the story")

    def test_a_paraphrase_snaps_to_the_sentence_it_came_from(self):
        self.assertEqual(self._anchor("Marta pours tea and sits"),
                         "She poured the tea and sat down without drinking it.")
        self.assertEqual(self._anchor("Marta opens the door of the kitchen"),
                         "Marta opened the kitchen door.")
        self.assertEqual(self._anchor("rain since Tuesday"),
                         "The rain had not stopped since Tuesday.")

    def test_an_invented_beat_gets_no_anchor(self):
        # Fabricating an anchor is worse than having none: scoring by
        # character similarity used to pin "Marta boarded a train to Leeds"
        # onto "Marta opened the kitchen door." because the strings look alike.
        for invented in ("Marta boarded a train to Leeds",
                         "A dragon circled the tower",
                         "He signed the lease on Thursday morning"):
            self.assertEqual(self._anchor(invented), "", invented)

    def test_a_fragment_too_short_to_anchor_is_dropped(self):
        # A bare name is a substring of the prose, so it passes as a quote --
        # and then credits a whole sentence in the coverage figure.
        self.assertEqual(self._anchor("Marta"), "")
        self.assertEqual(self._anchor("the"), "")

    def test_a_whole_short_sentence_is_kept_however_short(self):
        self.assertEqual(self._anchor("She ran."), "She ran.")

    def test_snapping_makes_coverage_tell_the_truth(self):
        paraphrased = [{"source": s} for s in (
            "Marta opens the door of the kitchen",
            "rain since Tuesday",
            "Marta pours tea and sits")]
        snapped = [{"source": worker._snap_source(p["source"], self.STORY)}
                   for p in paraphrased]
        before = worker._story_coverage(self.STORY, paraphrased)["percent"]
        after = worker._story_coverage(self.STORY, snapped)["percent"]
        self.assertGreater(after, before)
        # Three panels over four sentences: "She ran." really is unanchored, so
        # 75% is the honest figure and 100% would be the old lie. Snapping is
        # for telling the truth about the division, not for flattering it.
        self.assertEqual(after, 75)
        self.assertIn("She ran.",
                      worker._story_coverage(self.STORY, snapped)["missing"])


class ShotlistTruncation(unittest.TestCase):
    """An overshooting division must be reported, not quietly shortened.

    `_parse_shotlist` sliced to the requested count before anything counted, so
    a story the writer divided into 26 beats became the first 20 with nothing
    said -- and `out_of_range`, written for exactly this, compared the
    already-truncated length against the range it had been truncated into and
    could never fire.
    """

    def _raw(self, n):
        beats = [{"shot": "wide", "subject": f"beat {i}", "action": "a",
                  "setting": "b", "source": "", "scene": 1} for i in range(n)]
        return "```json\n" + json.dumps(beats) + "\n```"

    def test_beats_are_counted_before_they_are_bounded(self):
        beats = worker._shotlist_beats(self._raw(26))
        self.assertEqual(len(beats), 26)

    def test_the_range_check_sees_the_real_count(self):
        low, high = 12, 20
        returned = len(worker._shotlist_beats(self._raw(26)))
        kept = worker._clean_shotlist(worker._shotlist_beats(self._raw(26))[:high], "")
        self.assertEqual(len(kept), high)
        # The bug: the old check asked this of `kept`, which is inside the
        # range by construction.
        self.assertFalse(not (low <= len(kept) <= high))
        self.assertTrue(not (low <= returned <= high))

    def test_the_one_shot_form_still_bounds(self):
        self.assertEqual(len(worker._parse_shotlist(self._raw(26), 20)), 20)
        self.assertEqual(len(worker._parse_shotlist(self._raw(5), 20)), 5)

    def test_a_division_inside_the_bound_is_not_reported_as_truncated(self):
        beats = worker._shotlist_beats(self._raw(14))
        kept = worker._clean_shotlist(beats[:20], "")
        self.assertEqual(len(beats), len(kept))

    def test_an_unusable_reply_still_raises(self):
        with self.assertRaises(ValueError):
            worker._shotlist_beats("no array here")
        with self.assertRaises(ValueError):
            worker._shotlist_beats("[]")
        with self.assertRaises(ValueError):
            worker._clean_shotlist(["not a dict", 7], "")


class CaptionWrapping(unittest.TestCase):
    """Text that cannot be broken at a space still has to stay in its panel.

    `_wrap` accepted any word that did not fit as long as the line was empty,
    which is right for a long word and wrong for one with no spaces in it at
    all: CJK prose, a URL, a hashtag, a pasted identifier. One unbroken run was
    painted through the neighbouring panel and off the page.
    """

    def setUp(self):
        from PIL import Image, ImageDraw
        self.draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        self.font = worker._caption_font(19)
        self.width = 330

    def _lines(self, text):
        return worker._wrap(self.draw, text, self.font, self.width)

    def test_nothing_is_drawn_wider_than_the_box(self):
        cases = {
            "url": "See https://example.com/a/very/long/path/that/never/breaks/anywhere",
            "cjk": "雨は火曜日から止んで"
                   "いないと台所は濡れた"
                   "ウールの匂いがした",
            "long word": "Pneumonoultramicroscopicsilicovolcanoconiosis",
            "hashtag": "#" + "a" * 120,
            "normal": "She opened the door.",
        }
        for label, text in cases.items():
            for line in self._lines(text):
                self.assertLessEqual(
                    self.draw.textlength(line, font=self.font), self.width,
                    f"{label}: {line!r} overflows the caption box")

    def test_no_text_is_lost_to_the_break(self):
        for text in ("a" * 200, "word " * 30, "https://example.com/" + "x" * 90):
            self.assertEqual("".join(self._lines(text)).replace(" ", ""),
                             text.replace(" ", ""))

    def test_ordinary_wrapping_is_unchanged(self):
        self.assertEqual(self._lines("She opened the door."),
                         ["She opened the door."])
        self.assertEqual(self._lines(""), [])

    def test_a_balloon_is_never_wider_than_its_cell(self):
        from PIL import Image, ImageDraw
        draw = ImageDraw.Draw(Image.new("RGB", (900, 600), (255, 255, 255)))
        used = worker._bubble(draw, "x" * 200, "Marta", self.font,
                              worker._caption_font(13), 200, 10, 380, 26, True)
        self.assertGreater(used, 0)


class CastMentions(unittest.TestCase):
    """Who the story is about decides who is drawn in every frame.

    `_mentions` sets the cast tiers, which pick the lead, which fills the shared
    lineup reference and every panel's cast brief. It seeded its score with a
    raw substring count, so a short name collected other words' letters.
    """

    PROSE = ("Marta opened the door and turned on the tap. "
             "The photographs were scattered. Ed had never been mentioned here.")

    def test_a_name_not_in_the_prose_scores_nothing(self):
        # "Ned" used to score 3 from opeNED, turNED, mentioNED.
        for absent in ("Ned", "Art", "Ora", "Hoto"):
            self.assertEqual(
                worker._mentions(self.PROSE, absent), 0,
                f"{absent!r} is not in the prose and must not be counted")

    def test_a_substring_of_other_words_does_not_outrank_the_protagonist(self):
        # "Ed" used to score 5 against Marta's 1, which handed the lineup sheet
        # and every panel brief to a character the story does not have.
        self.assertLessEqual(worker._mentions(self.PROSE, "Ed"),
                             worker._mentions(self.PROSE, "Marta"))

    def test_a_name_that_is_present_is_still_counted(self):
        story = "Louise wept. Later Louise slept, and Mrs. Mallard was still."
        self.assertEqual(worker._mentions(story, "Louise"), 2)
        self.assertEqual(worker._mentions(story, "Mrs. Mallard"), 1)
        # A surname alone still finds the same person.
        self.assertEqual(worker._mentions(story, "Mallard"), 1)

    def test_possessives_and_punctuation_still_count(self):
        story = "Marta's coat was wet. “Marta,” he said."
        self.assertEqual(worker._mentions(story, "Marta"), 2)


class SourceAnchoring(unittest.TestCase):
    """A panel's source passage has to have come out of the manuscript.

    The check was 80% content-word overlap against a *set*, so it asked only
    whether the words existed somewhere, in any order at any distance.
    """

    PROSE = ("Anna opened the door slowly. A grey cat slept on the mat by the "
             "fire. The kettle had boiled dry.")

    def test_a_verbatim_passage_is_accepted(self):
        self.assertTrue(worker._quotes_story("Anna opened the door slowly", self.PROSE))
        self.assertTrue(worker._quotes_story("A grey cat slept on the mat", self.PROSE))

    def test_a_recombination_of_the_story_words_is_refused(self):
        # Every content word is present and in order -- but Anna opened a door
        # and it was the cat that slept. Ordered overlap cannot catch this;
        # only adjacency can.
        self.assertFalse(worker._quotes_story("Anna slept on the mat", self.PROSE))

    def test_scrambled_story_words_are_refused(self):
        self.assertFalse(worker._quotes_story("the door opened Anna slowly", self.PROSE))

    def test_an_invented_passage_is_refused(self):
        self.assertFalse(worker._quotes_story("A dragon circled the tower", self.PROSE))

    def test_an_empty_passage_is_refused(self):
        self.assertFalse(worker._quotes_story("", self.PROSE))
        self.assertFalse(worker._quotes_story("the and of", self.PROSE))


class StoryCoverage(unittest.TestCase):
    """The only guarantee against a board that loses the middle of the story.

    It has to be wrong in neither direction, and it was wrong in both: a quote
    was reusable, so one source was credited against every sentence it
    resembled; and a sentence with no content words could never be anchored,
    so a fully quoted board still reported a shortfall.
    """

    FOUR = ("She ran into the rain. She ran into the street. "
            "She ran into the house. She ran into the room.")

    def test_one_quote_cannot_cover_four_similar_sentences(self):
        got = worker._story_coverage(self.FOUR, [{"source": "She ran into the rain."}])
        # Reported 100% with nothing missing while three sentences were lost.
        self.assertEqual(got["covered"], 1)
        self.assertEqual(got["total"], 4)
        self.assertEqual(got["percent"], 25)
        self.assertEqual(got["missing_total"], 3)

    def test_a_fully_quoted_story_is_fully_covered(self):
        sources = [{"source": s.strip() + "."} for s in self.FOUR.split(".") if s.strip()]
        got = worker._story_coverage(self.FOUR, sources)
        self.assertEqual(got["percent"], 100)
        self.assertEqual(got["missing"], [])

    def test_one_passage_may_span_several_sentences(self):
        # A passage longer than a sentence legitimately covers several, so it
        # is not spent by anchoring one of them.
        got = worker._story_coverage(self.FOUR, [{"source": self.FOUR}])
        self.assertEqual(got["percent"], 100)

    def test_a_sentence_with_no_content_words_is_not_counted_as_lost(self):
        prose = "Marta opened the door. It is. She ran into the rain."
        sources = [{"source": s.strip() + "."} for s in prose.split(".") if s.strip()]
        got = worker._story_coverage(prose, sources)
        # "It is." cannot be anchored by any quote, so it is not prose that can
        # go missing. This used to report 67% for a verbatim board.
        self.assertEqual(got["percent"], 100)
        self.assertEqual(got["total"], 2)
        self.assertNotIn("It is.", got["missing"])

    def test_the_real_missing_count_is_reported_alongside_the_truncated_list(self):
        prose = " ".join(f"Sentence number {n} said something." for n in range(30))
        got = worker._story_coverage(prose, [])
        self.assertEqual(len(got["missing"]), 20)
        self.assertEqual(got["missing_total"], 30)

    def test_an_empty_story_does_not_divide_by_zero(self):
        self.assertEqual(worker._story_coverage("", [])["percent"], 100)


class PageGeometry(unittest.TestCase):
    """The compositor's arithmetic, asserted without a golden image.

    Every one of these failures shipped, because a function whose output is an
    image feels like it needs a reference file to check. It does not: where an
    element lands, and what scale a panel is drawn at, are numbers.
    """

    def _style(self, width, size=19):
        return {"width": width, "margin": 34, "gutter": 18,
                "font": worker._caption_font(size),
                "small": worker._caption_font(13),
                "line_h": int(size * 1.4)}

    def _panels(self, count, px=768):
        from PIL import Image
        return [Image.new("RGB", (px, px), (120, 140 + i, 160)) for i in range(count)]

    def _meta(self, shots, captions=None, dialogue=None):
        n = len(shots)
        return {"shots": list(shots),
                "captions": list(captions or [""] * n),
                "dialogue": list(dialogue or [[]] * n),
                "scenes": [1] * n}

    def test_fit_never_enlarges_the_drawing(self):
        """A cell bigger than the panel must not invent detail.

        `scale = max(...)` alone upscaled every full-width establishing panel
        1.49x, making the panel the layout sets largest the softest on the page.
        """
        panel = self._panels(1)[0]
        for cell in ((1144, 663), (2000, 2000), (768, 445), (375, 375)):
            got = worker._fit(panel, *cell)
            self.assertEqual(got.size, cell, f"_fit must return exactly {cell}")
        # Enlarging would have to resample; the pixels must survive untouched.
        wide = worker._fit(panel, 1144, 663)
        self.assertEqual(wide.getpixel((1144 // 2, 663 // 2)),
                         panel.getpixel((384, 384)))

    def test_page_width_is_derived_from_the_panels(self):
        """So the widest cell matches the panel and needs no upscale."""
        for px in (512, 768, 1024):
            panels = self._panels(1, px=px)
            page = worker._render_page(panels, [0], self._meta(["wide"]),
                                       self._style(px + 68))
            self.assertEqual(page.width, px + 68)

    def test_a_lone_tight_shot_is_not_given_the_shallow_cell(self):
        """0.58 is for establishing shots, not for whoever is left over.

        `_tiers` strands a tight shot alone whenever it has no tight neighbour,
        and the shallow cell centre-crops 42% of its height away -- on a face,
        the top of the head and the chin.
        """
        style = self._style(836)
        wide = worker._render_page(self._panels(1), [0],
                                   self._meta(["wide"]), style)
        for shot in ("close-up", "medium"):
            tight = worker._render_page(self._panels(1), [0],
                                        self._meta([shot]), style)
            self.assertGreater(
                tight.height, wide.height,
                f"a lone {shot} must get a full-height cell, not the "
                "establishing shot's shallow one")
        inner = style["width"] - style["margin"] * 2
        self.assertEqual(wide.height, style["margin"] * 2 + round(inner * 0.58))

    def test_no_lone_non_wide_panel_ever_takes_the_shallow_cell(self):
        """Exhaustive over the shot layouts `_tiers` can produce."""
        import itertools
        shallow = 0
        for n in range(1, 7):
            for combo in itertools.product(("wide", "medium", "close-up"), repeat=n):
                for tier in worker._tiers(list(combo)):
                    if len(tier) == 1 and combo[tier[0]] != "wide":
                        shallow += 1
        # 562 of 1092 layouts used to strand a tight shot in a shallow cell.
        # They still get their own tier -- they just get a square one now,
        # which is what the two assertions above pin.
        self.assertGreater(shallow, 0, "expected lone tight shots to exist")

    def test_a_caption_is_never_painted_over_by_a_balloon(self):
        """The balloon used to start at a hardcoded 78px.

        A caption wrapping to three lines -- about thirteen words, an ordinary
        length for narration -- ran past it, and both became unreadable.
        """
        from PIL import Image, ImageDraw
        style = self._style(836)
        probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        captions = [
            "She opened the door.",
            "The rain had not stopped since Tuesday and the kitchen smelled "
            "of wet wool and old coffee grounds.",
            "Marta had spent eleven years in that kitchen and had never once, "
            "in all that time, thought of it as anything that belonged to her.",
        ]
        inner = style["width"] - style["margin"] * 2
        box_w = min(inner - 20, 430)
        for cap in captions:
            lines = len(worker._wrap(probe, cap, style["font"], box_w - 18))
            caption_bottom = 10 + lines * style["line_h"] + 12
            balloon_top = 10 + lines * style["line_h"] + 12 + 8
            self.assertGreater(
                balloon_top, caption_bottom,
                f"a {lines}-line caption is overlapped by the first balloon")

    def test_caption_box_reports_the_height_it_drew(self):
        from PIL import Image, ImageDraw
        draw = ImageDraw.Draw(Image.new("RGB", (600, 400), (255, 255, 255)))
        font = worker._caption_font(19)
        one = worker._caption_box(draw, "Short.", font, (10, 10, 430, 0), 26)
        many = worker._caption_box(
            draw, "The rain had not stopped since Tuesday and the kitchen "
                  "smelled of wet wool and old coffee grounds.",
            font, (10, 200, 430, 0), 26)
        self.assertEqual(one, 1 * 26 + 12)
        self.assertGreater(many, one)

    def test_all_pages_of_one_board_share_a_height(self):
        """A board whose pages are different shapes is not a comic.

        Page height was whatever its own tiers happened to sum to, so a
        two-panel page holding a wide shot came out three times the height of
        one holding two tight shots -- 1299 against 443 on the same board.
        """
        style = self._style(836)
        scenes = [1, 1, 2, 2, 3, 3]
        shots = ["wide", "close-up", "medium", "close-up", "wide", "medium"]
        meta = self._meta(shots)
        meta["scenes"] = scenes
        pages = worker._paginate(scenes)
        self.assertGreater(len(pages), 1, "expected a multi-page board")

        natural = [worker._page_plan([shots[i] for i in idx], style)[1]
                   for idx in pages]
        self.assertGreater(len(set(natural)), 1,
                           "expected these pages to differ before normalising")

        style["page_height"] = max(natural)
        heights = {worker._render_page(self._panels(6), idx, meta, style).height
                   for idx in pages}
        self.assertEqual(len(heights), 1,
                         f"pages came out at {sorted(heights)}")
        self.assertEqual(heights.pop(), max(natural))

    def test_a_short_page_is_padded_rather_than_stretched(self):
        """Filling the sheet by scaling panels up is the defect, not the fix."""
        style = self._style(836)
        shots = ["medium", "close-up"]
        plan, natural = worker._page_plan(shots, style)
        style["page_height"] = natural + 400
        page = worker._render_page(self._panels(2), [0, 1], self._meta(shots), style)
        self.assertEqual(page.height, natural + 400)
        # The cells keep their own size; only paper was added.
        self.assertEqual([cell_h for _t, _w, cell_h in plan],
                         [cell_h for _t, _w, cell_h
                          in worker._page_plan(shots, style)[0]])
        # And the spare paper is split, not all dumped at the bottom.
        top = page.getpixel((style["margin"] // 2, 4))
        self.assertEqual(top, worker.PAGE_GROUND)

    def test_every_drawn_element_stays_inside_the_page(self):
        """A caption or balloon outside its cell is a bug no assertion caught."""
        style = self._style(836)
        meta = self._meta(
            ["wide", "close-up", "medium"],
            captions=["A very long narration line that has to wrap at least "
                      "three times to be dangerous here."] * 3,
            dialogue=[[{"text": "Is that you?", "speaker": "Marta"},
                       {"text": "I have been waiting.", "speaker": "Jon"}]] * 3)
        page = worker._render_page(self._panels(3), [0, 1, 2], meta, style)
        self.assertEqual(page.width, style["width"])
        self.assertGreater(page.height, style["margin"] * 2)


class AdapterLoadArguments(unittest.TestCase):
    """A chosen style adapter has to reach the loader, not just the router.

    `_load_model` used to take `loras` and spend it on two things: setting
    `has_lora` for the router, and lengthening the model cache key. Neither
    attaches an adapter. `lora_paths`/`lora_scales` were built in exactly one
    place -- the mflux route -- so on every unified-route model (Qwen-Image,
    Z-Image, FLUX.2, Wan) picking an adapter changed nothing about the picture,
    while the log announced "adapters: owner/repo:file @ 1.0" and the altered
    cache key forced a multi-gigabyte reload to produce the identical image.

    Nothing caught it because the existing coverage tests adapter *detection*.
    This tests what is handed to the loader, which is where the adapter is
    either applied or lost.
    """

    def _stub_mlxgen(self, captured):
        import types

        mod = types.ModuleType("mlxgen")

        class Runtime:
            def cache_key(self, **_kw):
                return "stub-cache-key"

        def resolve_generation_runtime(**kw):
            captured["resolve"] = kw
            return Runtime()

        def load_generation_model(**kw):
            captured["load"] = kw
            return object()

        mod.resolve_generation_runtime = resolve_generation_runtime
        mod.load_generation_model = load_generation_model
        return mod

    def _load(self, **extra):
        import contextlib
        from unittest import mock

        captured = {}
        # The real policy call reaches into MLX; the routing decision under
        # test does not depend on it.
        previous = worker._POLICY_APPLIED
        worker._POLICY_APPLIED = True
        self.addCleanup(setattr, worker, "_POLICY_APPLIED", previous)
        self.addCleanup(worker.CACHE.unload)
        # `_downloads_allowed` imports mflux, which imports mlx a second time
        # into this freshly exec'd copy of the module -- and nanobind aborts the
        # interpreter on the duplicate type registration. The download gate is
        # not what is under test.
        with mock.patch.dict(sys.modules, {"mlxgen": self._stub_mlxgen(captured)}), \
             mock.patch.object(worker, "_downloads_allowed",
                               lambda *_a, **_k: contextlib.nullcontext()), \
             mock.patch.object(worker, "emit", lambda *_a, **_k: None), \
             mock.patch.object(worker, "log", lambda *_a, **_k: None):
            worker._load_model("req-adapters", "Qwen/Qwen-Image", 8, None, 0,
                               family="qwen_image", **extra)
        return captured

    def test_adapters_are_handed_to_the_loader(self):
        captured = self._load(
            loras=[{"path": "owner/repo:style.safetensors", "scale": 0.8}],
        )
        self.assertEqual(captured["load"].get("lora_paths"),
                         ["owner/repo:style.safetensors"])
        self.assertEqual(captured["load"].get("lora_scales"), [0.8])
        # Still announced to the router: that is what picks a route able to
        # take an adapter at all. It is not a substitute for the adapter.
        self.assertIs(captured["resolve"].get("has_lora"), True)

    def test_several_adapters_keep_their_order_and_strengths(self):
        captured = self._load(loras=[
            {"path": "a/one:x.safetensors", "scale": 1.0},
            {"path": "b/two:y.safetensors", "scale": 0.35},
        ])
        self.assertEqual(captured["load"]["lora_paths"],
                         ["a/one:x.safetensors", "b/two:y.safetensors"])
        self.assertEqual(captured["load"]["lora_scales"], [1.0, 0.35])

    def test_a_missing_strength_defaults_to_full(self):
        captured = self._load(loras=[{"path": "a/one:x.safetensors"}])
        self.assertEqual(captured["load"]["lora_scales"], [1.0])

    def test_no_adapter_keyword_when_none_were_chosen(self):
        captured = self._load()
        self.assertNotIn("lora_paths", captured["load"])
        self.assertNotIn("lora_scales", captured["load"])
        self.assertNotIn("has_lora", captured["resolve"])


class ModuleIntegrity(unittest.TestCase):
    """Names the module uses must exist.

    A constant was once deleted by a block rewrite whose replacement ran to
    the next `def`, and the function referencing it only fails when it runs --
    which needs a model, so no test reached it and the break shipped. This
    reads the module rather than running it.
    """

    def _tree(self):
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            return ast.parse(handle.read())

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



class TruncatedReplySalvage(unittest.TestCase):
    """A reply cut off mid-panel must not cost every panel in front of it.

    Found by running a real 310-word story through the real writer. It returned
    fifteen whole panels and was cut off inside the sixteenth; `rfind("]")`
    then landed on the closing bracket of the last panel's own `dialogue`
    array, so the slice was a list with half an object on the end, the parse
    raised, and all fifteen were discarded after minutes of waiting.
    """

    def _truncated(self):
        whole = [{"shot": "wide", "subject": f"beat {i}", "action": "a",
                  "setting": "kitchen", "scene": 1,
                  "dialogue": [{"speaker": "Mira", "text": "hello"}]}
                 for i in range(5)]
        body = json.dumps(whole)[:-1]  # drop the closing "]"
        return body + ', {"shot": "medium", "subject": "half", "caption": "'

    def test_complete_panels_are_salvaged(self):
        beats = worker._shotlist_beats(self._truncated())
        self.assertEqual(len(beats), 5)
        self.assertEqual(beats[0]["subject"], "beat 0")

    def test_the_half_written_panel_is_dropped(self):
        for beat in worker._shotlist_beats(self._truncated()):
            self.assertNotEqual(beat.get("subject"), "half")

    def test_nested_objects_do_not_end_a_panel_early(self):
        beats = worker._shotlist_beats(self._truncated())
        self.assertEqual(beats[0]["dialogue"][0]["speaker"], "Mira")

    def test_a_reply_with_nothing_complete_still_raises(self):
        with self.assertRaises(ValueError):
            worker._shotlist_beats('[{"shot": "wide", "subject": "half')

    def test_a_whole_reply_is_untouched(self):
        good = json.dumps([{"shot": "wide", "subject": "a"}] * 3)
        self.assertEqual(len(worker._shotlist_beats(good)), 3)


class NarrationVoice(unittest.TestCase):
    """Dialogue is not narration.

    Almost every third-person story contains "I", because its characters talk
    to each other. Asking the author of one to name its narrator is a question
    about somebody who does not exist -- caught by running a third-person
    fixture with two lines of dialogue in it.
    """

    def test_dialogue_does_not_make_a_story_first_person(self):
        for story in (
            'Jonas did not look up. "I did tell you. In writing."',
            '"I am going where the work is," he said. She put the letter down.',
            'She asked if he had told her. "I did," he said. "In writing."',
            'He said, “I will write properly.” Mira did not wave.',
        ):
            self.assertFalse(worker._is_first_person(story), story)

    def test_a_first_person_narrator_is_still_found(self):
        for story in (
            "I opened the door and she was already there.",
            "My sister was on the platform before me.",
            'I watched the rain. "You could have told me," she said.',
        ):
            self.assertTrue(worker._is_first_person(story), story)


class PlaceIdentity(unittest.TestCase):
    """One room per place, folded onto the places the story actually has.

    The writer names the same room differently from panel to panel. A real run
    gave "kitchen", "table in kitchen" and "kitchen table" for one kitchen:
    three rooms drawn, three chances to contradict each other, and two and a
    half minutes of drawing spent disagreeing with itself.
    """

    KNOWN = [{"name": "the kitchen", "description": "Narrow and green"},
             {"name": "the station", "description": "Iron roof"},
             {"name": "the platform", "description": ""}]

    def _key(self, setting, scene=1):
        return worker._place_key({"setting": setting, "scene": scene}, self.KNOWN)

    def test_the_writers_variants_fold_onto_one_room(self):
        for setting in ("kitchen", "the kitchen", "Kitchen", "her kitchen",
                        "table in kitchen", "kitchen table"):
            self.assertEqual(self._key(setting), "kitchen", setting)

    def test_a_revisited_place_is_the_same_room_across_scenes(self):
        self.assertEqual(self._key("kitchen", scene=1), self._key("kitchen", scene=6))

    def test_different_places_stay_different_inside_one_scene(self):
        # A writer that files the whole story under scene 1 still changes room.
        self.assertNotEqual(self._key("kitchen", scene=1),
                            self._key("station platform", scene=1))

    def test_a_place_the_story_does_not_name_keeps_its_own_words(self):
        self.assertEqual(self._key("damp streets"), "damp streets")

    def test_the_story_words_for_a_place_are_found(self):
        self.assertEqual(
            worker._place_described("kitchen", self.KNOWN), "Narrow and green")
        self.assertEqual(worker._place_described("attic", self.KNOWN), "")


class AdapterIsReallyAnAdapter(unittest.TestCase):
    """A file that holds no adapter weights must be refused, by name.

    Reported from use: a VAE repository is one .safetensors file with no
    pipeline declared, which is indistinguishable from an adapter repository in
    the listing -- so it installed, appeared as an adapter, could be selected
    and given a strength, and failed at draw time as mflux's "did not match any
    known adapter keys". That names the file but not what is wrong with it.
    """

    def _write(self, path, keys):
        """A safetensors file: 8-byte header length, then the JSON header."""
        header = {k: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
                  for k in keys}
        blob = json.dumps(header).encode()
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)))
            fh.write(blob)
            fh.write(b"\0\0\0\0")
        return path

    def test_a_vae_is_named_as_a_vae(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self._write(os.path.join(tmp, "Flux_HDR_VAE.safetensors"),
                            ["decoder.conv_in.weight", "decoder.conv_in.bias",
                             "encoder.conv_out.weight"])
            with self.assertRaises(ValueError) as caught:
                worker._check_is_adapter("r", f)
            message = str(caught.exception)
            self.assertIn("Flux_HDR_VAE.safetensors", message)
            self.assertIn("not a LoRA", message)
            self.assertIn("VAE", message)

    def test_a_real_adapter_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self._write(os.path.join(tmp, "style.safetensors"),
                            ["transformer.blocks.0.lora_A.weight",
                             "transformer.blocks.0.lora_B.weight"])
            worker._check_is_adapter("r", f)  # must not raise

    def test_a_text_encoder_is_named_as_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self._write(os.path.join(tmp, "te.safetensors"),
                            ["text_model.encoder.layers.0.weight"])
            with self.assertRaises(ValueError) as caught:
                worker._check_is_adapter("r", f)
            self.assertIn("text encoder", str(caught.exception))

    def test_a_plain_path_resolves(self):
        # The mflux route hands over a resolved path, not a repo handle. It
        # resolved neither, so the check silently declined to have an opinion.
        with tempfile.TemporaryDirectory() as tmp:
            f = self._write(os.path.join(tmp, "x.safetensors"), ["decoder.a"])
            self.assertEqual(worker._adapter_local_file(f), f)

    def test_an_unresolvable_handle_is_left_alone(self):
        # Better to let the loader speak than to refuse something that may be
        # perfectly good but simply is not on this machine yet.
        worker._check_is_adapter("r", "nobody/nothing:absent.safetensors")

    def test_a_vae_repository_is_not_offered_as_an_adapter(self):
        class Sibling:
            def __init__(self, name):
                self.rfilename = name

        class Repo:
            def __init__(self, ident, files, tags=()):
                self.id, self.tags = ident, list(tags)
                self.siblings = [Sibling(f) for f in files]

        # One .safetensors used to be evidence enough on its own.
        self.assertFalse(worker._looks_like_adapter(
            Repo("kpsss34/Flux-VAE-HDR", ["Flux_HDR_VAE.safetensors"])))
        self.assertFalse(worker._looks_like_adapter(
            Repo("some/clip-text-encoder", ["model.safetensors"])))
        self.assertTrue(worker._looks_like_adapter(
            Repo("ostris/cereal-lora", ["cereal.safetensors"], ["lora"])))
        self.assertTrue(worker._looks_like_adapter(
            Repo("someone/style", ["style_lora.safetensors"])))


class TruncatedCastSalvage(unittest.TestCase):
    """A cast reply cut off mid-entry must not cost the whole read.

    The division already salvaged a truncated reply; the cast reader parsed
    with `_json_block` and raised. A story with a large cast overruns the token
    cap, `rfind("}")` lands on the last complete entry instead of the end of
    the object, and the read failed with a raw JSON parser message after a
    couple of minutes of the writer -- having had every complete person in
    hand.
    """

    STORY = ("Mira opened the door. Jon was on the step. Ana had gone to the "
             "harbour and Ruth had stayed in the kitchen. Mira said nothing. "
             "Jon waited. Ana came back. Ruth poured the tea.")

    def _people(self):
        return [{"name": n, "description": "a coat"}
                for n in ("Mira", "Jon", "Ana", "Ruth")]

    def _cut_in_people(self):
        body = '{"people": ' + json.dumps(self._people())[:-1]
        return body + ', {"name": "Half", "description": "a scarf that'

    def _cut_in_places(self):
        places = [{"name": "the kitchen", "description": "wet wool"},
                  {"name": "the harbour", "description": "grey water"}]
        return ('{"people": ' + json.dumps(self._people())
                + ', "places": ' + json.dumps(places)[:-1]
                + ', {"name": "the sta')

    def test_complete_people_are_salvaged(self):
        found = worker._salvage_cast(self._cut_in_people())
        self.assertEqual([p["name"] for p in found["people"]],
                         ["Mira", "Jon", "Ana", "Ruth"])

    def test_the_half_written_entry_is_dropped(self):
        found = worker._salvage_cast(self._cut_in_people())
        self.assertNotIn("Half", [p["name"] for p in found["people"]])

    def test_a_cut_inside_places_still_yields_the_people(self):
        found = worker._salvage_cast(self._cut_in_places())
        self.assertEqual(len(found["people"]), 4)
        self.assertEqual([p["name"] for p in found["places"]],
                         ["the kitchen", "the harbour"])

    def test_places_are_not_collected_into_the_cast(self):
        """The two arrays run one after the other in the same object.

        An unbounded scan from "people" would read straight through the close
        bracket and file every place as a person.
        """
        found = worker._salvage_cast(self._cut_in_places())
        self.assertEqual([p["name"] for p in found["people"]],
                         ["Mira", "Jon", "Ana", "Ruth"])

    def test_a_whole_reply_still_goes_through_the_parser(self):
        whole = json.dumps({"people": self._people(), "places": []})
        self.assertEqual(len(worker._json_block(whole, "{", "}")["people"]), 4)

    def test_op_cast_keeps_what_arrived(self):
        """End to end, with the writer replaced by a reply that was cut off."""
        reply = self._cut_in_people()
        saved = (worker._write, worker.emit, worker.log)
        worker._write = lambda *a, **k: reply
        worker.emit = lambda payload: None
        worker.log = lambda *a, **k: None
        try:
            out = worker.op_cast("t", {"story": self.STORY})
        finally:
            worker._write, worker.emit, worker.log = saved
        self.assertEqual([p["name"] for p in out["people"]],
                         ["Mira", "Jon", "Ana", "Ruth"])

    def test_a_reply_with_nothing_complete_still_says_so(self):
        saved = (worker._write, worker.emit, worker.log)
        worker._write = lambda *a, **k: '{"people": [{"name": "Mi'
        worker.emit = lambda payload: None
        worker.log = lambda *a, **k: None
        try:
            with self.assertRaises(ValueError) as caught:
                worker.op_cast("t", {"story": self.STORY})
        finally:
            worker._write, worker.emit, worker.log = saved
        self.assertIn("could not read the cast", str(caught.exception))


class ResourcePolicyReachesEveryRoute(unittest.TestCase):
    """The upscale route ran with no MLX cache ceiling at all.

    Both loaders set `_POLICY_APPLIED` inline, and SeedVR2 is constructed
    directly without passing through either -- so the one operation heavy
    enough to abort the engine was the one operation with no bound on it.
    """

    def _source(self, name):
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.get_source_segment(src, fn) or ""

    def test_it_is_applied_once_per_process(self):
        calls = []
        saved = (worker.apply_resource_policy, worker.log, worker._POLICY_APPLIED)
        worker.apply_resource_policy = lambda: calls.append(1) or {}
        worker.log = lambda *a, **k: None
        worker._POLICY_APPLIED = False
        try:
            worker._apply_policy_once("t")
            worker._apply_policy_once("t")
        finally:
            (worker.apply_resource_policy, worker.log,
             worker._POLICY_APPLIED) = saved
        self.assertEqual(len(calls), 1)

    def test_every_route_that_allocates_applies_it(self):
        for name in ("_load_model", "_load_mflux_model", "op_upscale"):
            self.assertIn("_apply_policy_once(", self._source(name),
                          f"{name} allocates weights without bounding the "
                          "process first")

    def test_the_flag_is_set_in_exactly_one_place(self):
        """Otherwise the next route to be added copies the block again."""
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        setters = set()
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "_POLICY_APPLIED"
                        for t in node.targets):
                    setters.add(fn.name)
        self.assertEqual(setters, {"_apply_policy_once"})

    def test_the_policy_is_applied_before_the_requested_cache_limit(self):
        """The policy's own ceiling is a quarter of the budget.

        Applied afterwards it would *raise* a limit the request had lowered on
        purpose, which is the opposite of what `low_ram` asks for.
        """
        body = self._source("op_upscale")
        self.assertLess(body.index("_apply_policy_once("),
                        body.index("set_cache_limit("))


class TiledUpscaleIsBounded(unittest.TestCase):
    """The tiled path had a bound on each tile and none on the run.

    A 3024x4032 import at 5x came to about 690 pieces at roughly fifteen
    seconds each, and two persistent float32 canvases the size of the finished
    picture -- about 10 GiB -- which `_upscale_peak_gib` knows nothing about
    because it was fitted to one pass of the model. So the tile size was chosen
    from a budget that had already been spent, on the 16 GB machine this path
    exists to protect.
    """

    BUDGET = 12.0

    def _count(self, in_w, in_h, factor, budget=None):
        _src, _out, xs, ys = worker._tile_plan(in_w, in_h, factor,
                                               budget or self.BUDGET)
        return len(xs) * len(ys)

    def test_the_accumulators_are_counted_in_the_budget(self):
        for in_w, in_h, factor in ((1024, 1024, 2.0), (1000, 667, 2.0),
                                   (1600, 1200, 2.0), (768, 768, 3.0)):
            _src, tile_out, xs, ys = worker._tile_plan(in_w, in_h, factor,
                                                       self.BUDGET)
            if not xs:
                continue
            out_w, out_h = round(in_w * factor), round(in_h * factor)
            peak = (worker._upscale_peak_gib(tile_out, tile_out)
                    + worker._upscale_canvas_gib(out_w, out_h))
            self.assertLessEqual(
                peak, self.BUDGET * worker._UPSCALE_SAFETY + 0.01,
                f"{in_w}x{in_h} @{factor}x plans a tile that does not fit "
                f"beside its own canvas")

    def test_ignoring_the_canvas_used_to_overshoot(self):
        """Otherwise the check above proves nothing."""
        import math

        room = self.BUDGET * worker._UPSCALE_SAFETY - worker._UPSCALE_BASE_GIB
        old_tile = int(math.sqrt(room / worker._UPSCALE_GIB_PER_MPX * 1e6))
        peak = (worker._upscale_peak_gib(old_tile, old_tile)
                + worker._upscale_canvas_gib(3200, 2400))
        self.assertGreater(peak, self.BUDGET * worker._UPSCALE_SAFETY)

    def test_a_huge_enlargement_is_refused_before_the_loop(self):
        with self.assertRaises(ValueError) as caught:
            worker._refuse_tiled_upscale(3024, 4032, 5.0, self.BUDGET,
                                         self._count(3024, 4032, 5.0))
        message = str(caught.exception)
        self.assertIn("15120x20160", message)
        self.assertIn("Nothing was started", message)

    def test_the_refusal_names_the_tile_count(self):
        tiles = self._count(4032, 3024, 2.0)
        self.assertGreater(tiles, worker._max_upscale_tiles(),
                           "expected a photo at 2x to be over the cap")
        with self.assertRaises(ValueError) as caught:
            worker._refuse_tiled_upscale(4032, 3024, 2.0, self.BUDGET, tiles)
        self.assertIn(f"{tiles} pieces", str(caught.exception))

    def test_the_refusal_names_a_size_that_actually_works(self):
        for in_w, in_h, factor in ((3024, 4032, 5.0), (4032, 3024, 2.0)):
            workable = worker._largest_tiled_factor(in_w, in_h, factor,
                                                    self.BUDGET)
            self.assertGreater(workable, 1.0)
            with self.assertRaises(ValueError) as caught:
                worker._refuse_tiled_upscale(in_w, in_h, factor, self.BUDGET,
                                             self._count(in_w, in_h, factor))
            self.assertIn(f"{round(in_w * workable)}x{round(in_h * workable)}",
                          str(caught.exception))
            # And it has to be true, not merely printed.
            tiles = self._count(in_w, in_h, workable)
            self.assertTrue(tiles)
            self.assertLessEqual(tiles, worker._max_upscale_tiles())

    def test_the_sizes_that_worked_before_still_tile(self):
        for in_w, in_h, factor in ((512, 512, 2.0), (1024, 1024, 2.0),
                                   (1000, 667, 2.0), (768, 1024, 2.0)):
            tiles = self._count(in_w, in_h, factor)
            self.assertTrue(tiles, f"{in_w}x{in_h} @{factor}x tiles no more")
            self.assertLessEqual(tiles, worker._max_upscale_tiles())

    def test_the_cap_can_be_raised_deliberately(self):
        os.environ["MODELSTUDIO_MAX_UPSCALE_TILES"] = "800"
        try:
            self.assertEqual(worker._max_upscale_tiles(), 800)
        finally:
            del os.environ["MODELSTUDIO_MAX_UPSCALE_TILES"]
        self.assertEqual(worker._max_upscale_tiles(), 64)

    def test_the_refusal_happens_before_the_first_tile(self):
        """A refusal after the loop starts is a cancelled job, not a refusal."""
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "_upscale_tiled")
        body = ast.get_source_segment(src, fn) or ""
        self.assertIn("_refuse_tiled_upscale", body)
        self.assertLess(body.index("_refuse_tiled_upscale"),
                        body.index("for (y0, y1) in ys"))


class EditRewriteRejectionIsReported(unittest.TestCase):
    """A rejected edit rewrite said "Nothing to add."

    `_clarified_with_reason` returns the reason and `_clarify_edit` dropped it,
    so the edit path returned the untouched request with no `outcome` at all.
    The interface's fallback chain reads that as "your words were already
    fine" -- reported after a rewrite had been written, checked and thrown
    away.
    """

    def _clarify(self, reply, prompt="make the jacket red"):
        saved = (worker._write, worker.log)
        worker._write = lambda *a, **k: reply
        worker.log = lambda *a, **k: None
        try:
            return worker._clarify_edit("t", prompt,
                                        {"SUBJECT": "a green field jacket"})
        finally:
            worker._write, worker.log = saved

    def test_a_rejected_rewrite_comes_back_with_its_reason(self):
        line, why, attempt = self._clarify("a bicycle on a wet street at dawn")
        self.assertEqual(line, "make the jacket red", "the request is kept")
        self.assertEqual(why, "lost_intent")
        self.assertIn("bicycle", attempt, "what it wrote has to survive too")

    def test_an_accepted_rewrite_says_so(self):
        good = "change the green field jacket to red, leave everything else unchanged."
        line, why, _attempt = self._clarify(good)
        self.assertEqual(line, good)
        self.assertEqual(why, "ok")

    def test_composing_directly_is_not_a_rejection(self):
        """With no observations there is no rewrite to reject."""
        line, why, _attempt = worker._clarify_edit("t", "make the jacket red", {})
        self.assertEqual(why, "composed")
        self.assertIn("jacket", line)

    def test_op_assist_carries_the_reason_into_its_reply(self):
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "op_assist")
        for node in ast.walk(fn):
            if (isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and getattr(node.value.func, "id", "") == "_clarify_edit"):
                target = node.targets[0]
                self.assertIsInstance(
                    target, ast.Tuple,
                    "op_assist takes the rewrite and drops the reason again")
                self.assertIn("why", [e.id for e in target.elts])
                break
        else:
            self.fail("op_assist no longer calls _clarify_edit")
        body = ast.get_source_segment(src, fn) or ""
        edit_branch = body[:body.index("Writing the prompt")]
        self.assertIn("rewrite_rejected", edit_branch)
        self.assertIn("rejected_because", edit_branch)


class CaptionScripts(unittest.TestCase):
    """Captions that are not Latin were lettered as rows of empty boxes.

    `_CAPTION_FONTS` holds four Latin faces, and a face with no glyph for a
    character draws .notdef -- a box the wrap measures as though it were a
    letter, so nothing downstream could notice. Every Japanese, Chinese,
    Korean, Arabic, Hebrew and Devanagari caption came out unreadable.
    """

    SAMPLES = {
        "Japanese": "ねこが日本語で話した",
        "Chinese": "中文字体测试",
        "Korean": "한국어입니다",
        "Arabic": "مرحبا بالعالم",
        "Hebrew": "שלום עולם",
        "Devanagari": "नमस्ते दुनिया",
    }

    def _glyph(self, font, ch):
        mask = font.getmask(ch, mode="L")
        return mask.size, bytes(mask)

    def _boxes(self, font, text):
        """Characters this face draws as .notdef.

        The reference is a private-use character no font on any Mac maps, so
        its glyph *is* .notdef; anything that renders identically to it is a
        box rather than a letter.
        """
        notdef = self._glyph(font, "\ue0ff")
        return [ch for ch in text
                if not ch.isspace() and self._glyph(font, ch) == notdef]

    def test_the_latin_faces_really_cannot_letter_these(self):
        """Otherwise the test below proves nothing."""
        latin = worker._caption_font(19)
        for name, text in self.SAMPLES.items():
            self.assertTrue(self._boxes(latin, text),
                            f"{name} needs no fallback after all")

    def test_every_script_gets_a_face_that_can(self):
        latin = worker._caption_font(19)
        for name, text in self.SAMPLES.items():
            font = worker._lettering(text, 19, latin)
            self.assertEqual(
                self._boxes(font, text), [],
                f"{name} is still lettered as .notdef boxes")

    def test_a_caption_in_two_scripts_finds_one_face_for_both(self):
        latin = worker._caption_font(19)
        mixed = "日本語 and 한국어 together"
        font = worker._lettering(mixed, 19, latin)
        self.assertEqual(self._boxes(font, mixed), [])

    def test_latin_text_keeps_the_boards_own_face(self):
        """The common path must not change at all."""
        latin = worker._caption_font(19)
        self.assertIs(worker._lettering("She opened the door.", 19, latin),
                      latin)
        self.assertIs(worker._lettering("", 19, latin), latin)

    def test_every_listed_face_is_on_this_machine(self):
        """A list of files that are not there is a list of boxes."""
        for script, paths in worker._SCRIPT_FONTS.items():
            self.assertTrue(any(os.path.exists(p) for p in paths),
                            f"no face for {script} exists here")

    def test_both_renderers_choose_the_face_per_piece_of_lettering(self):
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        tree = ast.parse(src)
        for name in ("_render_page", "_render_strip"):
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertIn("_lettering", ast.get_source_segment(src, fn) or "",
                          f"{name} letters everything in one face")

    def test_a_japanese_board_composes(self):
        from PIL import Image

        panels = [Image.new("RGB", (768, 768), (120, 140, 160))]
        style = {"width": 836, "margin": 34, "gutter": 18, "size": 19,
                 "font": worker._caption_font(19),
                 "small": worker._caption_font(13),
                 "line_h": 26}
        meta = {"shots": ["wide"], "captions": ["ねこが日本語で話した"],
                "dialogue": [[{"text": "おはよう", "speaker": "ミラ"}]],
                "scenes": [1]}
        page = worker._render_page(panels, [0], meta, style)
        self.assertEqual(page.width, 836)


class CellTextStaysInItsCell(unittest.TestCase):
    """Lettering was bounded in width and not in height.

    In a three-panel tier the cell is about 369px wide while the caption box is
    up to 430 and wraps to as many lines as the words need, and each balloon
    was placed under the last one with nothing comparing the total to the cell.
    The caption and balloons covered the whole picture, ran into the tier below,
    and the second line of dialogue was drawn off the page and lost.
    """

    CAPTION = ("Marta had spent eleven years in that kitchen and had never "
               "once, in all that time, thought of it as anything that "
               "belonged to her, or to anyone she had ever known.")
    DIALOGUE = [{"text": "Is that you at the door again?", "speaker": "Marta"},
                {"text": "I have been waiting out here since the rain "
                         "started, and the bus does not come after nine.",
                 "speaker": "Jon"}]

    def _style(self, width, size=19):
        return {"width": width, "margin": 34, "gutter": 18, "size": size,
                "font": worker._caption_font(size),
                "small": worker._caption_font(13),
                "line_h": int(size * 1.4)}

    def _probe(self):
        from PIL import Image, ImageDraw
        return ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def _block_height(self, probe, style, cap, dialogue, cell_w, cell_h):
        """Where the lettering ends, measured the way it is drawn."""
        size, cap_h, bubbles = worker._cell_text_plan(
            probe, cap, dialogue, style, cell_w, cell_h)
        base = worker._text_size(style)
        body = style["font"] if size == base else worker._caption_font(size)
        line_h = max(1, int(size * 1.4))
        bottom = 10 + cap_h + 8 if cap else 16
        for line in dialogue[:bubbles]:
            bottom += worker._bubble_height(
                probe, line["text"], line["speaker"],
                worker._lettering(line["text"], size, body),
                worker._bubble_width(cell_w), line_h)
        return bottom, bubbles

    def test_nothing_is_drawn_below_its_cell(self):
        probe = self._probe()
        import itertools
        captions = ["", "Short.", self.CAPTION]
        dialogues = [[], self.DIALOGUE[:1], self.DIALOGUE]
        for width in (700, 836, 1000, 1211, 1240, 1400):
            style = self._style(width)
            for shots in (["medium"] * 3, ["medium"] * 2, ["wide"]):
                for _tier, cell_w, cell_h in worker._page_plan(shots, style)[0]:
                    for cap, dialogue in itertools.product(captions, dialogues):
                        bottom, _n = self._block_height(
                            probe, style, cap, dialogue, cell_w, cell_h)
                        self.assertLessEqual(
                            bottom, cell_h,
                            f"a {cell_w}x{cell_h} cell on a {width}px page "
                            f"letters {bottom}px of text")

    def test_the_unbounded_arithmetic_overflowed_the_cell(self):
        """The numbers this replaces, so the test above is not vacuous."""
        probe = self._probe()
        reach = {}
        for width, cell in ((1211, (369, 369)), (836, (244, 244))):
            style = self._style(width)
            _tier, cell_w, cell_h = worker._page_plan(["medium"] * 3, style)[0][0]
            self.assertEqual((cell_w, cell_h), cell,
                             "expected the three-panel cell this was reported on")
            line_h = style["line_h"]
            lines = len(worker._wrap(probe, self.CAPTION, style["font"],
                                     worker._caption_width(cell_w) - 18))
            bottom = 10 + lines * line_h + 12 + 8
            for line in self.DIALOGUE:
                bottom += worker._bubble_height(
                    probe, line["text"], line["speaker"], style["font"],
                    worker._bubble_width(cell_w), line_h)
            self.assertGreater(bottom, cell_h,
                               f"a {width}px page's three-panel cell used to "
                               "letter past its own bottom edge; the case has "
                               "stopped being one")
            reach[width] = bottom - cell_h
        self.assertGreater(reach[836], 18,
                           "the narrow page used to letter past the gutter and "
                           "into the tier below")

    def test_the_lettering_keeps_inside_the_cell_sideways_too(self):
        for cell_w in (198, 244, 369, 562, 1143):
            self.assertLessEqual(10 + worker._caption_width(cell_w), cell_w)
            self.assertLessEqual(worker._bubble_width(cell_w), cell_w)

    def test_a_roomy_cell_keeps_the_boards_own_size_and_every_balloon(self):
        """The common page must be drawn exactly as it was."""
        probe = self._probe()
        style = self._style(836)
        _tier, cell_w, cell_h = worker._page_plan(["wide"], style)[0][0]
        size, cap_h, bubbles = worker._cell_text_plan(
            probe, "The rain had not stopped since Tuesday.", self.DIALOGUE,
            style, cell_w, cell_h)
        self.assertEqual(size, 19)
        self.assertEqual(bubbles, 2)
        self.assertGreater(cap_h, 0)

    def test_a_crowded_cell_letters_smaller_before_it_drops_anything(self):
        probe = self._probe()
        style = self._style(1211)
        _tier, cell_w, cell_h = worker._page_plan(["medium"] * 3, style)[0][0]
        size, _cap_h, bubbles = worker._cell_text_plan(
            probe, self.CAPTION, self.DIALOGUE, style, cell_w, cell_h)
        self.assertLess(size, 19, "a smaller face is the first thing to try")
        self.assertEqual(bubbles, 2, "both lines of dialogue still fit")

    def test_the_gutter_between_tiers_is_left_as_paper(self):
        """Rendered, not measured: what spilled was drawn, not computed."""
        from PIL import Image

        style = self._style(1211)
        shots = ["medium"] * 6
        panels = [Image.new("RGB", (768, 768), (120, 140 + i, 160))
                  for i in range(6)]
        meta = {"shots": shots,
                "captions": [self.CAPTION] * 6,
                "dialogue": [list(self.DIALOGUE)] * 6,
                "scenes": [1] * 6}
        page = worker._render_page(panels, list(range(6)), meta, style)
        plan, _natural = worker._page_plan(shots, style)
        y = style["margin"]
        for _tier, _cell_w, cell_h in plan[:-1]:
            for row in range(y + cell_h + 4, y + cell_h + style["gutter"]):
                colours = {page.getpixel((x, row)) for x in range(page.width)}
                self.assertEqual(
                    colours, {worker.PAGE_GROUND},
                    f"something was drawn in the gutter at row {row}")
            y += cell_h + style["gutter"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
