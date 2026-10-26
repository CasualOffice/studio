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
import time
import unittest
import unittest.mock
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

    def test_two_to_a_tier_on_an_ordinary_page(self):
        # Three across a portrait page's band made cells four times taller
        # than wide.
        self.assertEqual(worker._tiers(["medium"] * 5), [[0, 1], [2, 3], [4]])

    def test_a_crowded_page_takes_three_without_stranding_one(self):
        tiers = worker._tiers(["medium"] * 10)
        self.assertTrue(all(len(t) <= 3 for t in tiers), tiers)
        self.assertLessEqual(len(tiers), 4)

    def test_a_wide_interrupts_a_group(self):
        self.assertEqual(
            worker._tiers(["medium", "wide", "medium", "close-up"]),
            [[0], [1], [2, 3]])

    def test_never_more_than_four_bands(self):
        self.assertEqual(len(worker._tiers(["wide"] * 6)), 4)

    def test_every_panel_appears_exactly_once(self):
        shots = ["wide", "medium", "close-up", "medium", "wide", "close-up", "medium"]
        flat = [i for t in worker._tiers(shots) for i in t]
        self.assertEqual(sorted(flat), list(range(len(shots))))


class PageCells(unittest.TestCase):
    """Cells fill a fixed page, and the board's own layout agrees with them."""

    def test_cells_fill_the_page_without_overlapping(self):
        import itertools
        for n in range(1, 7):
            for shots in itertools.product(("wide", "medium", "close-up"), repeat=n):
                for rtl in (False, True):
                    cells = worker._page_cells(list(shots), rtl)
                    for x, y, w, h in cells:
                        self.assertTrue(0 <= x and x + w <= 1.000001 and 0 <= y and y + h <= 1.000001)
                    for a, b in itertools.combinations(cells, 2):
                        apart = a[0] + a[2] <= b[0] + 1e-6 or b[0] + b[2] <= a[0] + 1e-6 \
                            or a[1] + a[3] <= b[1] + 1e-6 or b[1] + b[3] <= a[1] + 1e-6
                        self.assertTrue(apart, (shots, a, b))
                    self.assertAlmostEqual(max(y + h for _x, y, _w, h in cells), 1, places=5)

    def test_manga_mirrors_each_tier(self):
        ltr = worker._page_cells(["medium", "close-up"])
        rtl = worker._page_cells(["medium", "close-up"], rtl=True)
        self.assertLess(ltr[0][0], ltr[1][0])
        self.assertGreater(rtl[0][0], rtl[1][0])

    def test_the_board_and_the_compositor_lay_a_page_out_alike(self):
        """layout.ts reads the same file and must produce the same cells."""
        with open(os.path.join(HERE, "fixtures", "layout-cases.json"), encoding="utf-8") as handle:
            cases = json.load(handle)
        for case in cases:
            got = worker._page_cells(case["shots"], case["rtl"])
            for a, b in zip(got, case["cells"]):
                for u, v in zip(a, b):
                    self.assertAlmostEqual(u, v, places=4, msg=case)


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


class EnhancerMayRephrase(unittest.TestCase):
    """Losing a word is reported; losing the subject is still refused.

    Retention used to reject any rewrite that dropped a word, which made
    rephrasing impossible: "me and him was walking in forrest" cannot become
    good English without losing "him" and "was". It is shown for approval now,
    with the changes named, so vetoing a good rewrite on behalf of someone who
    can see it is the wrong call.
    """

    def test_a_modifier_may_be_rephrased_away(self):
        out, why = worker._clarified_with_reason(
            "a red car in the rain at night",
            "A red sedan sits in rain at night, wet asphalt reflecting a lamp.")
        self.assertIsNotNone(out)
        self.assertEqual(why, "rephrased")
        self.assertIn("car", worker._dropped_words(
            "a red car in the rain at night",
            "A red sedan sits in rain at night, wet asphalt reflecting a lamp."))

    def test_the_subject_may_not_be(self):
        # "a cat on a chair" keeps the chair, so any test asking whether *some*
        # of the request survived passes it. The head word is checked by name.
        for typed, rewrite in [
            ("a cat on a chair", "A hat rests on a chair."),
            ("a dog in the snow", "A log lies in the snow."),
        ]:
            out, why = worker._clarified_with_reason(typed, rewrite)
            self.assertIsNone(out, f"{typed!r} -> {rewrite!r} was accepted")
            self.assertEqual(why, "dropped")

    def test_losing_half_the_request_is_refused(self):
        out, _ = worker._clarified_with_reason(
            "an old bicycle against a brick wall", "A bicycle leans in a yard.")
        self.assertIsNone(out, "the wall was dropped and it passed")

    def test_instruction_words_are_not_things_to_keep(self):
        # "my dog but make him look like a king" is four content words and five
        # of these; counting them as lost refused every rendering of it.
        for w in ("but", "look", "like", "make", "turn", "into", "something"):
            self.assertIn(w, worker._NOT_A_SUBJECT)

    def test_a_placeholder_is_not_a_word_to_keep(self):
        # "somthing scary in a old hosue" names a house. Treating the
        # placeholder as a thing the rewrite had to repeat refused every
        # rendering of it.
        lost = worker._dropped_words(
            "somthing scary in a old hosue",
            "A derelict Victorian house looms in fog, boarded windows, door ajar.")
        self.assertNotIn("somthing", lost)
        self.assertNotIn("hosue", lost, "the spelling fix read as a loss")

    def test_a_mood_word_rendered_as_detail_goes_to_the_adjudicator(self):
        # "scary" becoming fog, boarding and an open door is the translation
        # being asked for, and no string comparison can see it. The literal
        # layer refuses, and op_assist asks the writer whether it is still the
        # same request -- which is the only thing that can answer.
        out, why = worker._clarified_with_reason(
            "somthing scary in a old hosue",
            "A derelict Victorian house looms in fog, boarded windows, door ajar.")
        self.assertIsNone(out)
        self.assertEqual(why, "dropped")
        self.assertIn("SAME", worker.SAME_REQUEST_SYSTEM)
        self.assertIn("DIFFERENT", worker.SAME_REQUEST_SYSTEM)

    def test_a_misspelling_no_longer_reads_as_a_different_request(self):
        # `_keeps_intent` matched four-character prefixes, and "hosue" and
        # "house" share three -- so correcting the spelling looked like
        # changing the subject.
        self.assertTrue(worker._keeps_intent(
            "a hosue at nite", "A house at night, one window lit."))

    def test_a_list_of_objects_is_still_refused(self):
        # A count cannot separate a described room from a list of its contents;
        # an eighty-word direction legitimately names several things. Shape can:
        # a list is almost entirely article-noun pairs.
        out, why = worker._clarified_with_reason(
            "a cat on a chair",
            "a cat, a chair, a book, a window, a rug, a fireplace, a lamp, "
            "a vase, a painting, a bookshelf, a cushion, a blanket, a clock.")
        self.assertIsNone(out)
        self.assertEqual(why, "rejected")

    def test_a_long_direction_is_not(self):
        out, why = worker._clarified_with_reason(
            "a cat on a chair",
            "A tabby cat with dense grey-brown fur lies curled on a worn oak "
            "chair, one paw tucked under its chin, tail hanging over the seat "
            "edge. Late afternoon sun comes low through a window to the left, "
            "warming the chair's arm and throwing a long soft shadow across "
            "the floorboards. Oil on canvas, visible brushwork.")
        self.assertIsNotNone(out, f"a real direction was refused as {why}")

    def test_the_instruction_asks_for_direction_not_paraphrase(self):
        system = worker.SCENE_DIRECTION_SYSTEM
        for field in ("SUBJECT", "COMPOSITION", "LIGHT", "STYLE"):
            self.assertIn(field, system)
        self.assertIn("rephrase freely", system)
        self.assertIn("UNCLEAR", system)
        # The rule that made it timid. It said "Never introduce a thing the
        # request does not already contain", which is why a four-word request
        # came back as thirteen words with no light and no framing.
        self.assertNotIn("Never introduce a thing", system)


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


class ComposeFollowsTheBoardsLayout(unittest.TestCase):
    """The compositor sets pages where the board broke them, each panel in the
    cell it was drawn for, read in the board's direction."""

    def _sealed(self, tmp, size, colour):
        import io
        import secrets
        import uuid

        from PIL import Image

        import vaultcrypto as vc

        fid, key = secrets.token_bytes(16), secrets.token_bytes(32)
        sid = str(uuid.uuid4())
        path = os.path.join(tmp, sid + ".bin")
        buf = io.BytesIO()
        Image.new("RGB", size, colour).save(buf, format="PNG")
        vc.write_sealed(path, key, fid, buf.getvalue())
        return {"id": sid, "file_id": fid.hex(), "key": key.hex(), "path": path, "ext": "png"}

    def _slot(self, tmp):
        import secrets
        import uuid
        fid, key = secrets.token_bytes(16), secrets.token_bytes(32)
        sid = str(uuid.uuid4())
        return {"id": sid, "file_id": fid.hex(), "key": key.hex(), "path": os.path.join(tmp, sid + ".out")}

    def test_pages_cells_and_direction_come_from_the_board(self):
        import io
        import tempfile

        from PIL import Image

        import vaultcrypto as vc

        tmp = tempfile.mkdtemp()
        shots = ["wide", "medium", "close-up", "medium"]
        rtl = worker._page_cells(shots[:3], rtl=True) + worker._page_cells(shots[3:], rtl=True)
        colours = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
        inputs = [self._sealed(tmp, (768, 768), c) for c in colours]
        slots = [self._slot(tmp) for _ in range(4)]
        result = worker.op_compose_board("t", {
            "vault_inputs": inputs, "vault_slots": slots, "layout": "page",
            "captions": [""] * 4, "shots": shots, "scenes": [1, 1, 1, 1],
            "dialogue": [[{"speaker": "Mira", "text": "Gone.", "side": "right"}], [], [], []],
            "pages": [0, 0, 0, 1], "cells": [list(c) for c in rtl], "direction": "rtl"})
        self.assertEqual(result["pages"], 2)
        self.assertEqual(result["overflow"], [])
        first = Image.open(io.BytesIO(vc.open_with_file_key(bytes.fromhex(slots[0]["key"]),
                                                           open(slots[0]["path"], "rb").read()))).convert("RGB")
        style = {"width": first.width, "margin": 34}
        # Read right to left: the second panel sits to the right of the third.
        second = worker._cell_px(rtl[1], style)
        third = worker._cell_px(rtl[2], style)
        self.assertGreater(second[0], third[0])
        centre = lambda c: (c[0] + c[2] // 2, c[1] + c[3] // 2)  # noqa: E731
        self.assertEqual(first.getpixel(centre(second)), colours[1])
        self.assertEqual(first.getpixel(centre(third)), colours[2])
        # Every page of the board is the same shape.
        self.assertEqual(len({tuple(d) for d in result["dimensions"]}), 1)


class ComicBindsIntoOneFile(unittest.TestCase):
    """Pages become a PDF, sealed, with the reading order inside the file.

    Exporting a comic as separate PNGs leaves its order in the file names, and
    the first person to sort them differently loses it.
    """

    def _sealed_page(self, tmp, colour):
        import io
        import secrets
        import uuid

        from PIL import Image

        import vaultcrypto as vc

        fid, key = secrets.token_bytes(16), secrets.token_bytes(32)
        sid = str(uuid.uuid4())
        path = os.path.join(tmp, sid + ".bin")
        buf = io.BytesIO()
        Image.new("RGBA", (620, 876), colour).save(buf, format="PNG")
        vc.write_sealed(path, key, fid, buf.getvalue())
        return {"id": sid, "file_id": fid.hex(), "key": key.hex(),
                "path": path, "ext": "png"}

    def test_three_pages_make_a_three_page_pdf(self):
        import secrets
        import tempfile
        import uuid

        import vaultcrypto as vc

        tmp = tempfile.mkdtemp()
        pages = [self._sealed_page(tmp, c) for c in
                 ((255, 255, 255, 255), (240, 240, 240, 255), (250, 250, 250, 255))]
        fid, key = secrets.token_bytes(16), secrets.token_bytes(32)
        slot = {"id": str(uuid.uuid4()), "file_id": fid.hex(), "key": key.hex(),
                "path": os.path.join(tmp, "out.bin")}
        result = worker.op_bind_comic(
            "t", {"vault_slots": [slot], "vault_inputs": pages, "title": "Mira"})

        self.assertEqual(result["pages"], 3)
        self.assertTrue(result["sealed"])
        pdf = vc.open_with_file_key(key, open(slot["path"], "rb").read())
        self.assertTrue(pdf.startswith(b"%PDF-"), "not a PDF")
        self.assertGreater(len(pdf), 1000)

    def test_binding_nothing_is_refused(self):
        with self.assertRaises(ValueError):
            worker.op_bind_comic("t", {"vault_slots": [{"id": "x"}],
                                       "vault_inputs": []})

    def test_a_slot_is_required(self):
        # Without one there is nowhere sealed to put it, and the alternative --
        # writing the PDF straight to disk -- is plaintext leaving the vault by
        # a path that is not the sanctioned one.
        with self.assertRaises(ValueError):
            worker.op_bind_comic("t", {"vault_slots": [], "vault_inputs": []})

    def test_it_leaves_no_decrypted_pages_behind(self):
        import secrets
        import tempfile
        import uuid

        tmp = tempfile.mkdtemp()
        pages = [self._sealed_page(tmp, (255, 255, 255, 255))]
        fid, key = secrets.token_bytes(16), secrets.token_bytes(32)
        slot = {"id": str(uuid.uuid4()), "file_id": fid.hex(), "key": key.hex(),
                "path": os.path.join(tmp, "out2.bin")}
        worker.op_bind_comic("t", {"vault_slots": [slot], "vault_inputs": pages,
                                   "title": "Mira"})
        left = [f for f in os.listdir(worker._stage_dir())]
        self.assertEqual(left, [], f"plaintext left in the staging directory: {left}")


class ChunkedTransferFallsBackToPlainHttp(unittest.TestCase):
    """A Xet failure is retried over plain HTTP, in process as well as out.

    Measured: a 9 GB fetch died at 6.9 GB with a 403 from the CDN, and the same
    fetch completed with the chunked transport disabled. The subprocess routes
    already recovered by setting the variable in a child's environment; the
    in-process routes -- which fetch the prompt writer and the picture reader --
    had no recovery, because the variable is read into a module constant at
    import, long before any download runs.

    Tested without huggingface_hub installed, because CI does not have it. The
    first version of these tests imported it and passed here and failed there.
    """

    TRANSPORT = RuntimeError(
        "Task error: File reconstruction error: CAS Client Error: Request "
        "error: HTTP status client error (403 Forbidden)"
    )

    def test_a_transport_failure_is_recognised(self):
        for message in (
            "CAS Client Error: 403",
            "File reconstruction error",
            "xet_get failed",
            "error decoding response body",
        ):
            self.assertTrue(worker._is_transfer_failure(RuntimeError(message)),
                            f"{message!r} not recognised")

    def test_a_request_failure_is_not(self):
        # Retrying a gated repository or a missing file just fails twice and
        # takes twice as long to say so.
        for message in (
            "401 Unauthorized: access to this repo is gated",
            "404 Not Found",
            "No space left on device",
        ):
            self.assertFalse(worker._is_transfer_failure(RuntimeError(message)),
                             f"{message!r} would be retried pointlessly")

    def test_it_retries_once_and_succeeds(self):
        calls = []
        disabled = []

        def fetch():
            calls.append(1)
            if len(calls) == 1:
                raise self.TRANSPORT
            return "fetched"

        original = worker._fetch_with_xet_disabled
        worker._fetch_with_xet_disabled = lambda f: (disabled.append(1), f())[1]
        try:
            self.assertEqual(worker._retry_without_xet(fetch, "t"), "fetched")
        finally:
            worker._fetch_with_xet_disabled = original
        self.assertEqual(len(calls), 2, "did not retry")
        self.assertEqual(len(disabled), 1, "retry did not disable the transport")

    def test_an_unrelated_failure_is_not_retried(self):
        calls = []

        def fetch():
            calls.append(1)
            raise RuntimeError("401 Unauthorized: access to this repo is gated")

        with self.assertRaises(RuntimeError):
            worker._retry_without_xet(fetch, "t")
        self.assertEqual(len(calls), 1)

    def test_a_working_fetch_is_not_touched(self):
        self.assertEqual(worker._retry_without_xet(lambda: "ok", "t"), "ok")


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
        used = worker._bubble(draw, "x" * 200, self.font, (200, 10, 380, 500), 10,
                              worker._bubble_width(380), 26, "left")
        self.assertGreater(used, 0)


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

    def _meta(self, shots, captions=None, dialogue=None, cells=None):
        n = len(shots)
        return {"shots": list(shots),
                "captions": list(captions or [""] * n),
                "dialogue": list(dialogue or [[]] * n),
                "scenes": [1] * n, "cells": cells or [None] * n}

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

    def test_page_width_is_the_style_width(self):
        for px in (512, 768, 1024):
            page, _short = worker._render_page(self._panels(1, px=px), [0], self._meta(["wide"]),
                                               self._style(px + 68))
            self.assertEqual(page.width, px + 68)

    def test_a_panel_drawn_for_its_cell_is_not_cropped(self):
        """Every panel used to be drawn square, and a wide cell cut 42% of
        its height away -- on a face, the top of the head and the chin. Drawn
        at its cell's shape, a panel lands whole: its corners are on the page."""
        from PIL import Image
        style = self._style(836)
        shots = ["wide", "medium", "close-up"]
        cells = worker._page_cells(shots)
        panels = []
        for cell in cells:
            _x, _y, w, h = worker._cell_px(cell, style)
            im = Image.new("RGB", (w, h), (90, 90, 90))
            for corner, colour in (((0, 0), (255, 0, 0)), ((w - 1, h - 1), (0, 0, 255))):
                for dx in range(-8, 9):
                    for dy in range(-8, 9):
                        px, py = min(w - 1, max(0, corner[0] + dx)), min(h - 1, max(0, corner[1] + dy))
                        im.putpixel((px, py), colour)
            panels.append(im)
        page, _short = worker._render_page(panels, [0, 1, 2], self._meta(shots, cells=cells), style)
        for cell in cells:
            x, y, w, h = worker._cell_px(cell, style)
            self.assertEqual(page.getpixel((x + 5, y + 5)), (255, 0, 0))
            self.assertEqual(page.getpixel((x + w - 6, y + h - 6)), (0, 0, 255))

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

    def test_all_pages_of_one_board_share_a_shape(self):
        """A board whose pages are different shapes is not a comic.

        Page height was whatever its own tiers happened to sum to, so a
        two-panel page holding a wide shot came out three times the height of
        one holding two tight shots -- 1299 against 443 on the same board.
        Every page is now one fixed shape, and its tiers fill it."""
        style = self._style(836)
        sizes = {worker._render_page(self._panels(len(shots)), list(range(len(shots))), self._meta(shots), style)[0].size
                 for shots in (["wide", "close-up"], ["medium", "close-up"], ["wide"] * 4, ["medium"] * 6)}
        self.assertEqual(len(sizes), 1, sizes)

    def test_every_drawn_element_stays_inside_the_page(self):
        """A caption or balloon outside its cell is a bug no assertion caught."""
        style = self._style(836)
        meta = self._meta(
            ["wide", "close-up", "medium"],
            captions=["A very long narration line that has to wrap at least "
                      "three times to be dangerous here."] * 3,
            dialogue=[[{"text": "Is that you?", "side": "left"},
                       {"text": "I have been waiting.", "side": "right"}]] * 3)
        page, _short = worker._render_page(self._panels(3), [0, 1, 2], meta, style)
        self.assertEqual(page.width, style["width"])
        self.assertGreater(page.height, style["margin"] * 2)


class BalloonPlacement(unittest.TestCase):
    """A balloon sits on its speaker's side and its tail points at them."""

    def _draw(self):
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (900, 700), (90, 90, 90))
        return im, ImageDraw.Draw(im)

    def _extent(self, im, cell):
        """The columns the balloon's white paint covers inside the cell."""
        x, y, w, h = cell
        cols = [cx for cx in range(x, x + w) if any(im.getpixel((cx, cy)) == (252, 252, 250)
                                                       for cy in range(y, y + h))]
        return min(cols), max(cols)

    def test_left_and_right_speakers_get_balloons_on_their_own_side(self):
        cell = (50, 50, 800, 600)
        font = worker._caption_font(19)
        for side, check in (("left", lambda lo, hi: hi < 50 + 800 * 0.7),
                            ("right", lambda lo, hi: lo > 50 + 800 * 0.3)):
            im, draw = self._draw()
            worker._bubble(draw, "Is that you at the door again?", font, cell, 60,
                           worker._bubble_width(800), 26, side)
            lo, hi = self._extent(im, cell)
            self.assertTrue(check(lo, hi), (side, lo, hi))

    def test_the_tail_points_at_the_speaker_and_an_offscreen_line_has_none(self):
        cell = (50, 50, 800, 600)
        font = worker._caption_font(19)
        heights = {}
        for side in ("left", "right", "off"):
            im, draw = self._draw()
            used = worker._bubble(draw, "Last train!", font, cell, 60, worker._bubble_width(800), 26, side)
            body_bottom = 60 + used - 18
            # Below the balloon body, only a tail is painted.
            tail = [cx for cx in range(50, 850) if im.getpixel((cx, body_bottom + 8)) == (252, 252, 250)]
            heights[side] = (sum(tail) / len(tail)) if tail else None
        self.assertLess(heights["left"], 50 + 400)
        self.assertGreater(heights["right"], 50 + 400)
        self.assertIsNone(heights["off"])

    def test_no_speaker_name_is_lettered_under_a_balloon(self):
        import inspect
        self.assertNotIn("speaker", inspect.signature(worker._bubble).parameters)
        self.assertNotIn("speaker", inspect.signature(worker._bubble_height).parameters)

    def test_words_cover_at_most_about_a_third_of_an_ordinary_panel(self):
        from PIL import Image, ImageDraw
        probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        style = {"width": 836, "margin": 34, "size": 19, "font": worker._caption_font(19),
                 "small": worker._caption_font(13), "line_h": 26}
        _size, cap_h, n, _whole = worker._cell_text_plan(
            probe, "The rain had not stopped.", [{"text": "Is that you?"}, {"text": "It is."}],
            style, 380, 560)
        used = 10 + cap_h + 8 + sum(worker._bubble_height(probe, t, style["font"], worker._bubble_width(380), 26)
                                    for t in ("Is that you?", "It is."))
        self.assertEqual(n, 2)
        self.assertLessEqual(used, 560 * worker._TEXT_COVER)


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


class OneHelperIsResidentAtATime(unittest.TestCase):
    """Reading a picture and then writing about it held both models at once.

    `_helpers_resident` exists because there are two of them, and every guard
    that made room for "the assistant" was written when there was one. The
    guard between the two helpers themselves was never written: the generator
    loader, the video route and the idle path all free both, and the one
    transition that always happens -- reader, then writer, back to back on
    every assisted edit -- freed neither. 2B and 4B resident together, which is
    the one-resident rule broken by the only pair guaranteed to run in
    sequence.
    """

    def _source(self, name):
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.get_source_segment(src, fn) or ""

    def test_the_reader_gives_up_the_writer(self):
        self.assertIn("_unload_writer()", self._source("_load_assistant"))

    def test_the_writer_gives_up_the_reader(self):
        self.assertIn("_release_assistant_only()", self._source("_load_writer"))

    def test_the_writer_does_not_unload_itself_on_the_way_in(self):
        """`_unload_assistant` drops both, which is wrong in this direction.

        There the reader is being replaced *by* the writer, so releasing the
        writer on the way to loading it is at best wasted work.
        """
        self.assertNotIn("_unload_assistant()", self._source("_load_writer"))

    def test_releasing_the_reader_alone_leaves_the_writer(self):
        saved = (dict(worker._ASSIST), dict(worker._WRITER))
        try:
            worker._ASSIST.update(key="reader", model=object())
            worker._WRITER.update(key="writer", model=object())
            worker._release_assistant_only()
            self.assertIsNone(worker._ASSIST["model"])
            self.assertIsNotNone(worker._WRITER["model"],
                                 "the writer is what this makes room for")
        finally:
            worker._ASSIST.clear(); worker._ASSIST.update(saved[0])
            worker._WRITER.clear(); worker._WRITER.update(saved[1])

    def test_the_idle_path_still_releases_both(self):
        saved = (dict(worker._ASSIST), dict(worker._WRITER))
        try:
            worker._ASSIST.update(key="reader", model=object())
            worker._WRITER.update(key="writer", model=object())
            worker._unload_assistant()
            self.assertIsNone(worker._ASSIST["model"])
            self.assertIsNone(worker._WRITER["model"])
        finally:
            worker._ASSIST.clear(); worker._ASSIST.update(saved[0])
            worker._WRITER.clear(); worker._WRITER.update(saved[1])


class CancelIsNotAFailedCheck(unittest.TestCase):
    """Pressing Cancel was swallowed and the job carried on.

    Both helpers sit behind a bare `except Exception`, which is right for a
    missing writer and wrong for `Cancelled`: caught there it became an
    ordinary negative result -- a DIFFERENT verdict, or a directly composed
    instruction -- and the request went on to build a reply about a rewrite the
    user had already stopped waiting for.
    """

    def _source(self, name):
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.get_source_segment(src, fn) or ""

    def test_cancellation_is_re_raised_before_the_general_handler(self):
        for name in ("_same_request", "_clarify_edit"):
            src = self._source(name)
            with self.subTest(fn=name):
                self.assertIn("except Cancelled:", src)
                self.assertLess(
                    src.index("except Cancelled:"), src.index("except Exception"),
                    f"{name} catches Exception before Cancelled, so Cancelled "
                    "never reaches its own handler")

    def test_a_cancelled_write_propagates_out_of_the_check(self):
        def boom(*_a, **_k):
            raise worker.Cancelled("stopped")

        saved = worker._write
        worker._write = boom
        try:
            with self.assertRaises(worker.Cancelled):
                worker._same_request("t", "a cat", "a tabby cat", None)
            with self.assertRaises(worker.Cancelled):
                worker._clarify_edit("t", "make it red", {"subject": "a jacket"})
        finally:
            worker._write = saved

    def test_an_ordinary_failure_is_still_absorbed(self):
        """A missing writer must not become a failed request."""
        def boom(*_a, **_k):
            raise RuntimeError("no writer installed")

        saved = worker._write
        worker._write = boom
        try:
            self.assertFalse(worker._same_request("t", "a cat", "a dog", None))
            line, why, _attempt = worker._clarify_edit(
                "t", "make it red", {"subject": "a jacket"})
            self.assertEqual(why, "composed")
            self.assertIn("red", line)
        finally:
            worker._write = saved


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
        # The two helper loaders were missing, which is how the 3.2 GiB writer
        # came to be the one load in the engine with no ceiling on it -- on the
        # route reached first, from a text box, while the machine is already
        # holding whatever the user was working on.
        for name in ("_load_model", "_load_mflux_model", "op_upscale",
                     "_load_writer", "_load_assistant"):
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

    def test_generation_applies_the_policy_before_its_cache_cap(self):
        """Generate and edit set the low-RAM cap, then the first load reset it.

        `_load_model` applies the policy on first use, which put the cache
        back to a quarter of the budget right after low-RAM mode had capped
        it at 1 GiB.
        """
        events = []
        saved = (worker.apply_resource_policy, worker.set_cache_limit,
                 worker.log, worker._POLICY_APPLIED, worker._load_model)

        class Stop(Exception):
            pass

        def load(*_a, **_k):
            events.append("load")
            raise Stop()

        worker.apply_resource_policy = lambda: events.append("policy") or {}
        worker.set_cache_limit = lambda gb: events.append(("cap", gb)) or {}
        worker.log = lambda *a, **k: None
        worker._POLICY_APPLIED = False
        worker._load_model = load
        try:
            with self.assertRaises(Stop):
                worker._run_generation(
                    "t", {"model": "m", "low_ram": True, "seed": 1},
                    "text-to-image", image_count=0)
        finally:
            (worker.apply_resource_policy, worker.set_cache_limit, worker.log,
             worker._POLICY_APPLIED, worker._load_model) = saved
        self.assertEqual(events[:2], ["policy", ("cap", 1.0)])

    def test_video_applies_the_policy_before_it_checks_headroom(self):
        body = self._source("op_video")
        self.assertLess(body.index("_apply_policy_once("),
                        body.index("_check_headroom("))


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
        source = {n.name: ast.get_source_segment(src, n) or "" for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)}
        self.assertIn("_lettering", source["_letter_cell"])
        for name in ("_render_page", "_render_strip"):
            self.assertIn("_letter_cell", source[name], f"{name} letters everything in one face")

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
        page, _short = worker._render_page(panels, [0], meta, style)
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
        size, cap_h, bubbles, _whole = worker._cell_text_plan(
            probe, cap, dialogue, style, cell_w, cell_h)
        base = worker._text_size(style)
        body = style["font"] if size == base else worker._caption_font(size)
        line_h = max(1, int(size * 1.4))
        bottom = 10 + cap_h + 8 if cap else 16
        for line in dialogue[:bubbles]:
            bottom += worker._bubble_height(
                probe, line["text"],
                worker._lettering(line["text"], size, body),
                worker._bubble_width(cell_w), line_h)
        return bottom, bubbles

    def _cells(self, shots, style):
        return [worker._cell_px(c, style)[2:] for c in worker._page_cells(shots)]

    def test_nothing_is_drawn_below_its_cell(self):
        probe = self._probe()
        import itertools
        captions = ["", "Short.", self.CAPTION]
        dialogues = [[], self.DIALOGUE[:1], self.DIALOGUE]
        for width in (700, 836, 1000, 1211, 1240, 1400):
            style = self._style(width)
            for shots in (["medium"] * 8, ["medium"] * 3, ["medium"] * 2, ["wide"]):
                for cell_w, cell_h in self._cells(shots, style):
                    for cap, dialogue in itertools.product(captions, dialogues):
                        bottom, _n = self._block_height(
                            probe, style, cap, dialogue, cell_w, cell_h)
                        self.assertLessEqual(
                            bottom, cell_h,
                            f"a {cell_w}x{cell_h} cell on a {width}px page "
                            f"letters {bottom}px of text")

    def test_a_small_cell_is_still_measured(self):
        """The case the measuring exists for: a small cell and long words."""
        probe = self._probe()
        style = self._style(836)
        bottom, n = self._block_height(probe, style, self.CAPTION, self.DIALOGUE, 246, 240)
        self.assertLessEqual(bottom, 240)
        self.assertLess(n, 2)

    def test_the_lettering_keeps_inside_the_cell_sideways_too(self):
        for cell_w in (198, 244, 369, 562, 1143):
            self.assertLessEqual(10 + worker._caption_width(cell_w), cell_w)
            self.assertLessEqual(worker._bubble_width(cell_w), cell_w)

    def test_a_roomy_cell_keeps_the_boards_own_size_and_every_balloon(self):
        """The common page must be drawn exactly as it was."""
        probe = self._probe()
        style = self._style(836)
        cell_w, cell_h = self._cells(["wide"], style)[0]
        size, cap_h, bubbles, _whole = worker._cell_text_plan(
            probe, "The rain had not stopped since Tuesday.", self.DIALOGUE,
            style, cell_w, cell_h)
        self.assertEqual(size, 19)
        self.assertEqual(bubbles, 2)
        self.assertGreater(cap_h, 0)

    def test_a_crowded_cell_letters_smaller_before_it_drops_anything(self):
        probe = self._probe()
        style = self._style(1211)
        cell_w, cell_h = 369, 420
        size, _cap_h, bubbles, _whole = worker._cell_text_plan(
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
        page, _short = worker._render_page(panels, list(range(6)), meta, style)
        cells = [worker._cell_px(c, style) for c in worker._page_cells(shots)]
        tops = sorted({y for _x, y, _w, _h in cells})
        for upper, lower in zip(tops, tops[1:]):
            bottom = max(y + h for _x, y, _w, h in cells if y == upper)
            for row in range(bottom + 4, lower - 1):
                colours = {page.getpixel((x, row)) for x in range(page.width)}
                self.assertEqual(
                    colours, {worker.PAGE_GROUND},
                    f"something was drawn in the gutter at row {row}")



class FixtureFX01(unittest.TestCase):
    """The fixture is a contract, so it is checked before anything uses it.

    An answer key that has drifted from its own source is worse than no answer
    key: every test built on it still passes, and all of them are now asserting
    the wrong thing. These verify the fixture against itself -- the spans quote
    the source exactly, the beat graph is closed, the entities referenced exist
    -- so a later edit to either half cannot quietly separate them.
    """

    @classmethod
    def setUpClass(cls):
        import json

        path = os.path.join(HERE, "fixtures", "fx01.json")
        with open(path, encoding="utf-8") as handle:
            cls.fx = json.load(handle)

    def test_every_span_quotes_the_source_exactly(self):
        source = self.fx["source"]
        for group in ("beats", "places", "speech"):
            for item in self.fx[group]:
                if "span" not in item:
                    continue
                start, end = item["span"]
                with self.subTest(item=item.get("id") or item.get("text")):
                    self.assertGreaterEqual(start, 0)
                    self.assertLessEqual(end, len(source))
                    self.assertLess(start, end, "an empty span anchors nothing")
                    self.assertTrue(source[start:end].strip())

    def test_the_quoted_text_is_unique_in_the_source(self):
        """Otherwise an offset and a search disagree about which one is meant."""
        source = self.fx["source"]
        for beat in self.fx["beats"]:
            start, end = beat["span"]
            with self.subTest(beat=beat["id"]):
                self.assertEqual(source.count(source[start:end]), 1)

    def test_the_beats_cover_the_story_in_order(self):
        spans = [b["span"] for b in self.fx["beats"]]
        self.assertEqual(spans, sorted(spans),
                         "beats are listed out of source order")
        for (_a, end), (start, _b) in zip(spans, spans[1:]):
            self.assertLessEqual(end, start, "two beats claim the same text")

    def test_every_actor_and_dependency_resolves(self):
        ids = {e["id"] for e in self.fx["entities"]}
        beats = {b["id"] for b in self.fx["beats"]}
        for beat in self.fx["beats"]:
            with self.subTest(beat=beat["id"]):
                self.assertTrue(set(beat["actors"]) <= ids,
                                f"{beat['actors']} names someone not in the cast")
                self.assertTrue(set(beat.get("depends_on", [])) <= beats)
                self.assertNotIn(beat["id"], beat.get("depends_on", []))
        for move in self.fx["state_transitions"]:
            self.assertIn(move["at_beat"], beats)

    def test_the_reveal_runs_to_the_end_of_the_source(self):
        """FX-05 tests chunking; this is why the case matters.

        "The train had gone" is not a separate beat, it is what the empty
        platform means -- and it is the last sentence, so any division that
        stops early loses the point of the scene rather than a detail.
        """
        last = self.fx["beats"][-1]
        self.assertTrue(last["essential"])
        self.assertEqual(last["span"][1], len(self.fx["source"]))

    def test_no_run_of_words_is_left_unassigned(self):
        """Every clause belongs to a beat, a place, a line, or to context.

        Character by character rather than sentence by sentence, because the
        interesting gaps are inside sentences: the setting clause that opens
        the second one, and the speech embedded in the fourth. Punctuation and
        spacing between spans are not claimed by anything and do not need to
        be -- what must not survive is a run of words nothing accounts for.
        """
        source = self.fx["source"]
        covered = [False] * len(source)
        for group in ("beats", "places", "speech", "context_only"):
            for item in self.fx[group]:
                start, end = item["span"]
                for i in range(start, end):
                    covered[i] = True
        gap, orphans = [], []
        for i, ch in enumerate(source):
            if covered[i] or not ch.isalnum():
                if gap:
                    orphans.append("".join(gap))
                    gap = []
            else:
                gap.append(ch)
        if gap:
            orphans.append("".join(gap))
        self.assertEqual(orphans, [], f"words no part of the key claims: {orphans}")

    def test_the_narrator_is_one_entity_with_the_named_one(self):
        narrators = [e for e in self.fx["entities"] if e.get("narrator")]
        self.assertEqual(len(narrators), 1)
        self.assertIn("Mi", narrators[0]["aliases"])
        self.assertIn("I", narrators[0]["aliases"])

    def test_the_key_says_what_must_not_be_claimed(self):
        """A key of only what should be present passes on invented output."""
        self.assertGreaterEqual(len(self.fx["must_not"]), 4)
        for item in self.fx["must_not"]:
            self.assertTrue(item["claim"] and item["why"])


class CorrectEnglishIsNotATypo(unittest.TestCase):
    """The user's own plurals were shown back to them as spelling mistakes.

    `/usr/share/dict/words` holds base forms and nothing else -- `cars`,
    `trees`, `houses`, `hands`, `holds`, `stands` and `wears` are all absent --
    and the typo check's first question is whether the typed word is a word. It
    is not in the list, so it became a candidate to be corrected: "trees in the
    mist" rewritten as "tree trunks in the mist" reported trees -> tree as a
    spelling the enhancer had fixed, and "Mira wears red" against a rewrite
    mentioning a warm hallway reported wears -> warm.

    Most plural nouns and third-person verbs in a real prompt were eligible.
    """

    def setUp(self):
        if not worker._dictionary():
            self.skipTest("no system word list on this machine")

    def test_the_word_list_really_is_missing_these(self):
        """Otherwise the tests below pass for the wrong reason."""
        missing = [w for w in ("cars", "trees", "hands", "holds", "wears")
                   if w not in worker._dictionary()]
        self.assertEqual(missing, ["cars", "trees", "hands", "holds", "wears"],
                         "this word list carries inflections, so the defect "
                         "these tests describe cannot occur on it")

    def test_an_inflection_counts_as_a_word(self):
        for w in ("cars", "trees", "hands", "holds", "wears", "running",
                  "walked", "houses", "rises"):
            with self.subTest(word=w):
                self.assertTrue(worker._is_word(w), f"{w} is English")

    def test_a_plural_is_not_offered_as_a_correction(self):
        for original, improved in (
                ("trees in the mist", "tree trunks in the mist."),
                ("hands holding a cup", "hand holds a cup."),
                ("cars on a street", "a car on a wet street."),
                ("Mira wears red", "A warm hallway with Mira in it.")):
            with self.subTest(original=original):
                self.assertEqual(worker._corrections(original, improved), [],
                                 "correct English reported as a typo")

    def test_a_real_typo_is_still_caught(self):
        self.assertEqual(worker._corrections("a hous on a hill",
                                             "a house on a hill."),
                         [["hous", "house"]])
        self.assertEqual(worker._corrections("a tabel", "a table."),
                         [["tabel", "table"]])

    def test_a_real_word_is_still_not_a_misspelling_of_another(self):
        """The guard this whole mechanism exists for."""
        self.assertEqual(worker._corrections("a cat on a chair",
                                             "a hat on a chair."), [])
        self.assertFalse(worker._misspelling_of("horse", "house"))

    def test_a_misspelled_plural_is_still_a_typo(self):
        # "housse" is not an inflection of anything, so the leniency above
        # must not reach it.
        self.assertFalse(worker._is_word("housse"))


class EveryAnswerHasTheSameShape(unittest.TestCase):
    """A reply used to be whichever dict the returning branch happened to write.

    There were five of them with five different key sets, and the interface
    reconstructed what it had not been told: `outcome` was absent on two paths,
    so it fell through a chain of guesses ending at the card for a successful
    rewrite. The user was shown their own untouched prompt, folklore and all,
    headed "Suggested prompt", above a button that replaced it with itself.

    Guessing is the wrong repair. One constructor, every field every time, and
    a stamp saying which contract it was written against.
    """

    def _assist_source(self):
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        return src, next(n for n in ast.walk(ast.parse(src))
                         if isinstance(n, ast.FunctionDef) and n.name == "op_assist")

    def test_no_branch_builds_a_reply_of_its_own(self):
        import ast

        _src, fn = self._assist_source()
        literals = [n.lineno for n in ast.walk(fn)
                    if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)]
        self.assertEqual(literals, [],
                         f"dict-literal replies at lines {literals}; a branch "
                         "that writes its own shape can omit a field")

    def test_every_field_is_present_whatever_happened(self):
        shapes = [
            worker._assist_reply("a", "a"),
            worker._assist_reply("a", "a", unclear=True, outcome="unclear"),
            worker._assist_reply("a", "a", outcome="rewrite_rejected",
                                 rejected_because="dropped", attempt="b"),
            worker._assist_reply("b", "a", changed=True, added=["b"]),
        ]
        keys = {frozenset(shape) for shape in shapes}
        self.assertEqual(len(keys), 1, "the replies do not agree on their keys")
        for shape in shapes:
            self.assertEqual(shape["version"], worker.ASSIST_REPLY_VERSION)
            for field in ("outcome", "changed", "saw_image", "unclear",
                          "description", "note", "attempt", "rejected_because",
                          "added", "dropped", "corrected", "removed"):
                self.assertIn(field, shape)

    def test_a_mistyped_field_is_an_error_not_a_silent_absence(self):
        """Otherwise the interface simply never sees it, which is harder to find."""
        with self.assertRaises(ValueError):
            worker._assist_reply("a", "a", sawimage=True)

    def test_the_caller_cannot_be_overridden_on_prompt_or_original(self):
        reply = worker._assist_reply("rewritten", "typed")
        self.assertEqual(reply["prompt"], "rewritten")
        self.assertEqual(reply["original"], "typed")


class RemovedAndDroppedAreDifferentFacts(unittest.TestCase):
    """The same word was reported twice, as two contradictory things.

    `removed` is what the cleaner stripped on the way in -- quality folklore
    the user typed and the enhancer does not pass on. `dropped` is what the
    rewrite no longer says. They are different facts and the interface draws
    them as the same red chip under different labels.

    `dropped` was diffed against the prompt as typed rather than the cleaned
    one that was actually judged, so every stripped word came back a second
    time as a word the rewrite had lost. "a masterpiece 8k cat on a chair"
    reported `removed: [masterpiece, 8k]` and `dropped: [masterpiece]`: the
    user was told the same word had been taken out on purpose and lost by
    accident.
    """

    def test_the_cleaner_and_the_rewrite_do_not_report_the_same_word(self):
        typed = "a masterpiece 8k cat on a chair"
        cleaned, removed = worker._clean_prompt_request(typed)
        self.assertIn("masterpiece", removed)
        # The rewrite is judged against the cleaned prompt, so the diff that
        # names what it lost has to be against the same text.
        dropped = worker._dropped_words(cleaned, "a tabby cat on an oak chair.")
        self.assertNotIn("masterpiece", dropped)
        self.assertFalse(set(dropped) & set(removed),
                         "a word cannot be both deliberately removed and lost")

    def test_the_reply_diffs_against_the_prompt_that_was_judged(self):
        """Asserted at the call site, because that is where it was wrong.

        The first attempt at this fixed the comment and the test and not the
        code: `_dropped_words` was still being called with `original_prompt` in
        `op_assist` while a test called it with the cleaned prompt directly and
        passed. A test of a pure function cannot see which argument production
        hands it.
        """
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "op_assist")
        calls = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == "_dropped_words"]
        self.assertTrue(calls, "op_assist no longer reports dropped words")
        for call in calls:
            first = call.args[0]
            self.assertIsInstance(first, ast.Name)
            self.assertEqual(
                first.id, "user_prompt",
                "dropped words are diffed against the prompt that was judged, "
                "not the one that was typed -- otherwise every word the "
                "cleaner stripped is reported a second time as a loss")

    def test_a_word_the_rewrite_really_loses_is_still_named(self):
        cleaned, _removed = worker._clean_prompt_request(
            "a masterpiece cat on a chair")
        self.assertIn("chair", worker._dropped_words(cleaned, "a tabby cat."))


class ALongerWordIsADifferentWord(unittest.TestCase):
    """A rewrite could swap the subject for a word that starts the same way.

    `_dropped_words` compared stems with an unbounded prefix match, which is
    not a stem match -- it is "starts with the same letters", and a longer word
    starting with the same letters is a different thing rather than an
    inflection. So every one of these was accepted as having kept the subject:

        a cat on a chair  ->  a catalogue on a chair
        a red car         ->  a red cardigan
        a man on a boat   ->  a mango on a boat

    which is the failure `_misspelling_of` was written to stop, reopened from
    the other side: cat -> hat was caught and cat -> catalogue was not.
    """

    SWAPS = (
        ("a cat on a chair", "a catalogue on a chair.", "cat"),
        ("a red car", "a red cardigan.", "car"),
        ("a man on a boat", "a mango on a boat.", "man"),
        ("a bell over the door", "a bellows over the door.", "bell"),
    )

    def test_a_longer_word_does_not_stand_in_for_the_subject(self):
        for original, line, lost in self.SWAPS:
            with self.subTest(original=original):
                self.assertIn(lost, worker._dropped_words(original, line))

    def test_the_swap_reaches_the_check_that_gates_a_rewrite(self):
        """`_carries` is the all-words check, and it must see the loss too.

        `_keeps_intent` is not the check to assert this on. It is an `any` over
        content words -- one survivor passes it, which is what it is for -- so
        "a catalogue on a chair" passes on "chair" alone and correctly. Its
        prefix match is left unbounded on purpose: "red" -> "reddish" grows by
        four letters legitimately while "man" -> "mango" grows by two and does
        not, so no length threshold tells them apart. Stems do, which is why
        the bound belongs where it is.
        """
        for original, line, _lost in self.SWAPS:
            with self.subTest(original=original):
                self.assertFalse(
                    worker._carries(original, line),
                    f"{line!r} was accepted as carrying {original!r}")

    def test_an_inflection_is_still_the_same_word(self):
        """The bound is one letter, which is what an inflection needs."""
        for original, line in (("the sun rises", "the sun rising over the roofs."),
                               ("trees in the mist", "a tree line in the mist."),
                               ("a man runs", "a man running down the street.")):
            with self.subTest(original=original):
                self.assertEqual(worker._dropped_words(original, line), [])


class TextThatDoesNotFitIsReported(unittest.TestCase):
    """A composed page used to lose speech without saying so.

    Two things cut text and neither told anyone. The parser kept the first two
    spoken lines and the first ninety characters of each, before the user had
    read any of it. Then the composer, measuring a cell that genuinely could
    not hold what was left, returned a smaller balloon count and drew that --
    so a three-hander arrived as a two-hander, was composed, looked finished,
    and was bound into a PDF with one speaker missing.

    The parser no longer cuts. The composer still does, because a 244px cell
    really cannot hold three balloons, but it now reports every panel it did it
    to. These tests hold that reporting in place.
    """

    THREE = [{"speaker": "Marta", "text": "Is that you at the door again?"},
             {"speaker": "Jon", "text": "I have been waiting out here since "
                                        "the rain started, and the bus does "
                                        "not come after nine."},
             {"speaker": "Marta", "text": "Then you had better come in before "
                                          "you catch your death of it."}]

    def _style(self, width, size=19):
        return {"width": width, "margin": 34, "gutter": 18, "size": size,
                "font": worker._caption_font(size),
                "small": worker._caption_font(13),
                "line_h": int(size * 1.4)}

    def _panels(self, n, px=768):
        from PIL import Image
        return [Image.new("RGB", (px, px), (120, 140 + i, 160))
                for i in range(n)]

    def _meta(self, shots, dialogue, captions=None):
        n = len(shots)
        return {"shots": list(shots),
                "captions": list(captions or [""] * n),
                "dialogue": list(dialogue),
                "scenes": [1] * n}

    def test_a_cell_that_cannot_hold_its_dialogue_says_which_panel(self):
        style = self._style(836)
        shots = ["medium"] * 8
        meta = self._meta(shots, [list(self.THREE)] * 8)
        _page, short = worker._render_page(self._panels(8), list(range(8)),
                                           meta, style)
        self.assertTrue(short, "a crowded page's cell cannot letter three balloons")
        for record in short:
            self.assertEqual(record["spoken"], 3)
            self.assertLess(record["lettered"], 3)
            self.assertIn(record["panel"], range(8))

    def test_a_page_that_letters_everything_reports_nothing(self):
        """The ordinary page, which must not start warning about itself."""
        style = self._style(836)
        meta = self._meta(["wide"], [self.THREE[:1]],
                          ["The rain had not stopped since Tuesday."])
        _page, short = worker._render_page(self._panels(1), [0], meta, style)
        self.assertEqual(short, [])

    def test_the_report_names_a_clipped_caption_separately(self):
        """A caption this long reaches a cell only by being typed there.

        The writer's own narration is bounded at 120 characters when it is
        parsed. A caption the user edited is not bounded anywhere, which is the
        case that matters: it is approved text, and it was being clipped at the
        bottom of the box with nothing returned to say so.
        """
        caption = ("Marta had spent eleven years in that kitchen and had "
                   "never once thought of it as hers. " * 20).strip()
        style = self._style(836)
        shots = ["medium"] * 8
        meta = self._meta(shots, [[]] * 8, [caption] * 8)
        _page, short = worker._render_page(self._panels(8), list(range(8)),
                                           meta, style)
        self.assertTrue(short, "this caption does not fit a crowded page's cell")
        self.assertTrue(all(r["caption_clipped"] for r in short))
        self.assertTrue(all(r["spoken"] == 0 for r in short))

    def test_the_scroll_measures_its_balloons_and_reports_them_too(self):
        """The strip never measured at all -- it capped at two and drew.

        Uncapped and unmeasured, the third balloon is lettered past the bottom
        of its own panel and into the gap, where it reads as belonging to the
        panel after it.
        """
        style = self._style(836)
        style["page_height"] = 0
        long_talk = [{"speaker": f"S{i}",
                      "text": "I have been waiting out here since the rain "
                              "started and the bus does not come after nine."}
                     for i in range(8)]
        meta = self._meta(["medium"], [long_talk])
        _page, short = worker._render_strip(self._panels(1), meta, style)
        self.assertTrue(short, "eight balloons do not fit one scroll panel")
        self.assertEqual(short[0]["spoken"], 8)
        self.assertLess(short[0]["lettered"], 8)

    def test_the_scroll_keeps_the_lines_it_can_hold(self):
        style = self._style(836)
        meta = self._meta(["medium"], [self.THREE[:1]])
        _page, short = worker._render_strip(self._panels(1), meta, style)
        self.assertEqual(short, [])

    def test_nothing_the_scroll_letters_runs_past_its_panel(self):
        """Measured the way it is drawn, not asserted about the report."""
        from PIL import Image, ImageDraw

        style = self._style(836)
        probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        base = worker._text_size(style)
        panels = self._panels(1)
        width, margin = style["width"], style["margin"]
        im_w = width - margin * 2
        im_h = im_w  # the panel keeps its own (square) shape
        long_talk = [{"speaker": f"S{i}", "text": "The bus does not come "
                                                  "after nine o'clock."}
                     for i in range(9)]
        meta = self._meta(["medium"], [long_talk])
        _page, short = worker._render_strip(panels, meta, style)
        drawn = short[0]["lettered"] if short else len(long_talk)
        used = 16
        for line in long_talk[:drawn]:
            used += worker._bubble_height(
                probe, line["text"],
                worker._lettering(line["text"], base, style["font"]),
                worker._bubble_width(im_w), style["line_h"])
        self.assertLessEqual(used, im_h,
                             "the scroll lettered past the bottom of its panel")


class StagingIsNotLeftBehind(unittest.TestCase):
    """Briefly-decrypted vault images must not outlive the worker that made them.

    `atexit` alone did not cover it: SIGTERM's default disposition terminates
    without unwinding, and SIGTERM is what the host sends when the app quits.
    The scavenger then declined to remove the leftovers for a day, while
    SECURITY.md promised they were "scavenged at the next engine start".
    """

    def _stage(self, tmp, pid, age_seconds=0):
        d = os.path.join(tmp, f"msvault-{pid}-abc" if pid else "msvault-legacy")
        os.makedirs(d)
        open(os.path.join(d, "secret.png"), "wb").write(b"decrypted")
        if age_seconds:
            old = time.time() - age_seconds
            os.utime(d, (old, old))
        return d

    def test_a_directory_whose_worker_is_gone_goes_at_once(self):
        # A dead pid: nothing owns these pictures any more.
        dead = 999999
        self.assertFalse(worker._process_alive(dead))
        with tempfile.TemporaryDirectory() as tmp:
            d = self._stage(tmp, dead)
            with unittest.mock.patch.object(tempfile, "gettempdir", lambda: tmp):
                worker._scavenge_stale_stage_dirs()
            self.assertFalse(os.path.exists(d),
                             "a fresh directory from a dead worker must still go")

    def test_a_live_workers_directory_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._stage(tmp, os.getpid(), age_seconds=48 * 60 * 60)
            with unittest.mock.patch.object(tempfile, "gettempdir", lambda: tmp):
                worker._scavenge_stale_stage_dirs()
            self.assertTrue(os.path.exists(d),
                            "a running worker's staging must survive, however old")

    def test_a_name_without_a_pid_still_falls_back_to_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh = self._stage(tmp, None)
            with unittest.mock.patch.object(tempfile, "gettempdir", lambda: tmp):
                worker._scavenge_stale_stage_dirs()
            self.assertTrue(os.path.exists(fresh))
            old = time.time() - 48 * 60 * 60
            os.utime(fresh, (old, old))
            with unittest.mock.patch.object(tempfile, "gettempdir", lambda: tmp):
                worker._scavenge_stale_stage_dirs()
            self.assertFalse(os.path.exists(fresh))

    def test_the_pid_is_readable_from_the_name(self):
        self.assertEqual(worker._stage_owner_pid("msvault-4321-xyz"), 4321)
        self.assertIsNone(worker._stage_owner_pid("msvault-legacy"))
        self.assertIsNone(worker._stage_owner_pid("something-else"))

    def test_a_terminated_worker_removes_its_own_staging(self):
        """The end to end case: SIGTERM, and nothing decrypted is left behind."""
        import subprocess
        import sys as _sys
        import textwrap

        engine_dir = os.path.dirname(os.path.abspath(worker.__file__))
        with tempfile.TemporaryDirectory() as tmp:
            script = textwrap.dedent(f"""
                import os, sys, time
                sys.path.insert(0, {engine_dir!r})
                import worker
                d = worker._stage_dir()
                with open(os.path.join(d, "secret.png"), "wb") as fh:
                    fh.write(b"decrypted")
                # Marked, because importing the engine can print before this.
                # It goes to stderr on purpose: worker.py sets
                # `sys.stdout = sys.stderr` so a stray print can never corrupt
                # the JSON protocol on fd 1.
                print("STAGED " + d, flush=True)
                time.sleep(30)
            """)
            proc = subprocess.Popen(
                [_sys.executable, "-c", script],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                env={**os.environ, "TMPDIR": tmp})
            self.addCleanup(proc.kill)
            staged = ""
            deadline = time.time() + 60
            while time.time() < deadline:
                line = proc.stderr.readline()
                if not line:
                    break
                if line.startswith("STAGED "):
                    staged = line[len("STAGED "):].strip()
                    break
            self.assertTrue(staged and os.path.isdir(staged),
                            f"the child should have staged; got {staged!r}")
            self.assertTrue(os.path.exists(os.path.join(staged, "secret.png")))

            proc.terminate()
            proc.wait(timeout=20)
            self.assertFalse(
                os.path.exists(staged),
                "SIGTERM must remove the decrypted staging, not leave it for a day")



def _shaping_available() -> bool:
    """Whether the two packages that do the reshaping are installed.

    CI installs pillow, pillow-heif and cryptography and nothing else, so these
    are absent there. `_shape_for_display` is documented to return the text
    unchanged when they are missing -- deliberately, so a caption is drawn
    badly rather than not at all -- and asserting it reorders anyway made CI
    red for a behaviour that is correct.
    """
    try:
        import arabic_reshaper  # noqa: F401
        import bidi  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_shaping_available(),
                     "arabic_reshaper and python-bidi are not installed")
class RightToLeftCaptions(unittest.TestCase):
    """Arabic and Hebrew as they are read, not as they are stored.

    Finding a font with the glyphs was only half of it. Pillow's wheel is built
    without raqm -- `PIL.features.check("raqm")` is False, and so are harfbuzz
    and fribidi -- so it draws right-to-left text in isolated forms and in
    logical order: letters that do not join, running the wrong way. That is
    arguably worse than the empty boxes it replaced, because it looks like text
    somebody rendered rather than text somebody dropped.
    """

    ARABIC = "الفتاة تقف عند النافذة وتنظر الى المطر"
    HEBREW = "הילדה עומדת ליד החלון ומביטה בגשם"
    ENGLISH = "The rain had not stopped since Tuesday."

    def _draw(self):
        from PIL import Image, ImageDraw
        return ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def test_english_is_returned_exactly_as_it_came(self):
        # The whole path must be a no-op for left-to-right text, or every
        # existing caption changes the day this ships.
        self.assertEqual(worker._shape_for_display(self.ENGLISH), self.ENGLISH)
        self.assertEqual(worker._shape_for_display(""), "")
        self.assertFalse(worker._has_rtl(self.ENGLISH))

    def test_arabic_is_joined_and_reordered(self):
        self.assertTrue(worker._has_rtl(self.ARABIC))
        shaped = worker._shape_for_display(self.ARABIC)
        self.assertNotEqual(shaped, self.ARABIC)
        # Presentation forms are what "joined" looks like in Unicode.
        self.assertTrue(any(0xFE70 <= ord(c) <= 0xFEFF for c in shaped))

    def test_hebrew_is_reordered(self):
        self.assertTrue(worker._has_rtl(self.HEBREW))
        self.assertNotEqual(worker._shape_for_display(self.HEBREW), self.HEBREW)

    def test_wrapping_happens_in_logical_order(self):
        # Shaping the whole caption and then breaking it puts words on the
        # wrong lines: the break has to fall where the sentence breaks.
        draw, font = self._draw(), worker._caption_font(19)
        lines = worker._wrap(draw, self.ARABIC, font, 330)
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertLessEqual(draw.textlength(line, font=font), 330)

    def test_a_missing_package_degrades_rather_than_crashes(self):
        import builtins
        real = builtins.__import__

        def blocked(name, *a, **k):
            if name.split(".")[0] in ("arabic_reshaper", "bidi"):
                raise ModuleNotFoundError(name)
            return real(name, *a, **k)

        builtins.__import__ = blocked
        try:
            self.assertEqual(worker._shape_for_display(self.ARABIC), self.ARABIC)
        finally:
            builtins.__import__ = real

    def test_the_packages_are_pinned(self):
        lock = open(os.path.join(HERE, "requirements.lock"), encoding="utf-8").read()
        self.assertIn("arabic-reshaper==", lock)
        self.assertIn("python-bidi==", lock)

if __name__ == "__main__":
    unittest.main(verbosity=2)
