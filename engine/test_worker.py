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

    def test_every_speaker_survives_the_parser(self):
        """Balloons sit on top of the panel, so a cell has a real limit.

        It is not this function's limit to enforce. Five lines used to become
        two here, before the user had seen any of them and before anything knew
        how large the cell would be -- so the cut was made by the part of the
        pipeline least able to judge it and reported by nothing. The composer
        measures the cell it actually has and says what would not fit.
        """
        many = ",".join(f'{{"speaker":"S{i}","text":"line {i}"}}' for i in range(5))
        panels = worker._parse_shotlist(
            f'[{{"shot":"wide","subject":"x","action":"y","setting":"z",'
            f'"dialogue":[{many}]}}]', 1)
        self.assertEqual(len(panels[0]["dialogue"]), 5)
        self.assertEqual([d["speaker"] for d in panels[0]["dialogue"]],
                         [f"S{i}" for i in range(5)])

    def test_a_long_line_is_not_shortened_on_the_way_in(self):
        """Ninety characters was where a spoken line used to be cut.

        Mid-word, with no ellipsis and no record, so the panel arrived holding
        a sentence that stopped. Whether the line is too long for its bubble is
        a question about the bubble.
        """
        line = "I have been waiting out here since the rain started, and the " \
               "bus does not come after nine, and I am not going home."
        self.assertGreater(len(line), 90)
        panels = worker._parse_shotlist(
            '[{"shot":"wide","subject":"x","action":"y","setting":"z",'
            f'"dialogue":[{{"speaker":"Jon","text":"{line}"}}]}}]', 1)
        self.assertEqual(panels[0]["dialogue"][0]["text"], line)


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


class PanelsHaveAPurpose(unittest.TestCase):
    """A panel now says what it is for, not only how far away the camera is.

    `shot` is distance. It does not say what the panel is doing, and the two
    are different questions: a close-up of a face registering bad news and a
    close-up of a dripping tap want opposite treatment -- one needs the
    character held exactly, the other must not have her anywhere near it. The
    pipeline had only distance.
    """

    def panel(self, **over):
        base = {"shot": "medium", "subject": "x", "action": "y",
                "setting": "z", "scene": 1}
        return json.dumps([{**base, **over}])

    def test_the_writer_is_believed(self):
        for stated in worker.PURPOSES:
            got = worker._parse_shotlist(self.panel(purpose=stated), 1)
            self.assertEqual(got[0]["purpose"], stated)

    def test_a_nonsense_purpose_is_inferred_instead(self):
        got = worker._parse_shotlist(self.panel(purpose="banana"), 1)
        self.assertIn(got[0]["purpose"], worker.PURPOSES)

    def test_spoken_words_make_it_a_dialogue_panel(self):
        got = worker._parse_shotlist(
            self.panel(dialogue=[{"speaker": "Mira", "text": "Who is there?"}]), 1)
        self.assertEqual(got[0]["purpose"], "dialogue")

    def test_a_stated_detail_is_not_given_the_character(self):
        got = worker._parse_shotlist(
            self.panel(shot="close-up", subject="a dripping tap",
                       purpose="detail"), 1)
        self.assertFalse(got[0]["character_in_frame"])

    def test_an_inferred_purpose_never_writes_anyone_out(self):
        # A wide shot whose subject is "Mira" is not an establishing shot
        # merely because the characters array was left empty. Inference is good
        # enough to label a panel and not good enough to decide who is in it.
        got = worker._parse_shotlist(
            self.panel(shot="wide", subject="Mira", action="stands"), 1)
        self.assertTrue(got[0]["character_in_frame"])


class ShotMixIsReported(unittest.TestCase):
    """The writer is told to vary the shots. Nothing checked whether it had.

    Story coverage gets a full report with a missing-passages list. Shot
    distribution got none, so twelve close-ups in a row went unnoticed until
    forty minutes of drawing had been spent on them.
    """

    def panels(self, shots):
        return worker._parse_shotlist(json.dumps(
            [{"shot": s, "subject": f"s{i}", "action": "a", "setting": "b",
              "scene": 1} for i, s in enumerate(shots)]), len(shots))

    def test_a_flat_page_is_named(self):
        mix = worker._shot_mix(self.panels(["close-up"] * 9))
        self.assertEqual(mix["uniform"], "close-up")
        self.assertEqual(mix["counts"]["close-up"], 9)

    def test_a_varied_page_is_not(self):
        mix = worker._shot_mix(
            self.panels(["wide", "medium", "close-up", "medium", "wide", "close-up"]))
        self.assertIsNone(mix["uniform"])

    def test_a_short_run_is_not_called_flat(self):
        # Three of one size is a short scene, not a flat page.
        self.assertIsNone(worker._shot_mix(self.panels(["wide"] * 3))["uniform"])

    def test_it_counts_what_the_panels_are_for(self):
        mix = worker._shot_mix(self.panels(["wide", "medium", "close-up", "wide"]))
        self.assertEqual(sum(mix["purposes"].values()), 4)

    def test_no_panels_is_not_an_error(self):
        mix = worker._shot_mix([])
        self.assertEqual(mix["total"], 0)
        self.assertIsNone(mix["uniform"])


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
        # Every one of these keeps *some* word of the panel, which is why an
        # any-word check was never the check for this.
        for brief in (
            "A narrow kitchen, dark green walls, bare oak boards, one window.",
            "Warm hallway light falling across a narrow flat.",
            "A kitchen.",
        ):
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
            page, _short = worker._render_page(panels, [0], self._meta(["wide"]),
                                       self._style(px + 68))
            self.assertEqual(page.width, px + 68)

    def test_a_lone_tight_shot_is_not_given_the_shallow_cell(self):
        """0.58 is for establishing shots, not for whoever is left over.

        `_tiers` strands a tight shot alone whenever it has no tight neighbour,
        and the shallow cell centre-crops 42% of its height away -- on a face,
        the top of the head and the chin.
        """
        style = self._style(836)
        wide, _short = worker._render_page(self._panels(1), [0],
                                   self._meta(["wide"]), style)
        for shot in ("close-up", "medium"):
            tight, _short = worker._render_page(self._panels(1), [0],
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
        heights = {worker._render_page(self._panels(6), idx, meta, style)[0].height
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
        page, _short = worker._render_page(self._panels(2), [0, 1],
                                          self._meta(shots), style)
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
        page, _short = worker._render_page(self._panels(3), [0, 1, 2], meta, style)
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

    The SAME check sits behind a bare `except Exception`, which is right for a
    missing writer and wrong for `Cancelled`: caught there it became an
    ordinary negative result -- a DIFFERENT verdict -- and the request went on
    to build a reply about a rewrite the user had already stopped waiting for.
    """

    def _source(self, name):
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.get_source_segment(src, fn) or ""

    def test_cancellation_is_re_raised_before_the_general_handler(self):
        for name in ("_same_request",):
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
        size, cap_h, bubbles, _whole = worker._cell_text_plan(
            probe, "The rain had not stopped since Tuesday.", self.DIALOGUE,
            style, cell_w, cell_h)
        self.assertEqual(size, 19)
        self.assertEqual(bubbles, 2)
        self.assertGreater(cap_h, 0)

    def test_a_crowded_cell_letters_smaller_before_it_drops_anything(self):
        probe = self._probe()
        style = self._style(1211)
        _tier, cell_w, cell_h = worker._page_plan(["medium"] * 3, style)[0][0]
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
        plan, _natural = worker._page_plan(shots, style)
        y = style["margin"]
        for _tier, _cell_w, cell_h in plan[:-1]:
            for row in range(y + cell_h + 4, y + cell_h + style["gutter"]):
                colours = {page.getpixel((x, row)) for x in range(page.width)}
                self.assertEqual(
                    colours, {worker.PAGE_GROUND},
                    f"something was drawn in the gutter at row {row}")
            y += cell_h + style["gutter"]



class TheExpectedRangeIsACheckNotACap(unittest.TestCase):
    """The range was enforced, and enforcing it cut the end off the story.

    `_panel_bounds` says what it is for in its own docstring: "Not a target. It
    exists so a writer that returns three panels for three thousand words, or
    ninety for three hundred, is recognised as having failed rather than
    believed." The division was then sliced to the top of that range, which is
    not recognising anything -- it is deleting the evidence and reporting the
    shortened length.

    Measured on FX-01 with the installed 4B writer. 74 words derives a range of
    2-4, and the key requires five essential beats, so a correct board was
    already impossible. The writer returned eight sensible beats; four were
    kept. The slice takes the front, and the back of a scene is where its
    reveal is: the unlocking and the empty platform both went, and the board
    scored 67% with the ending missing. Unsliced, the same request on the same
    writer returns seven or eight panels and anchors every essential beat.
    """

    def test_the_writers_division_is_not_sliced_to_the_range(self):
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "op_shotlist")
        calls = [n for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "_clean_shotlist"]
        self.assertTrue(calls)
        for call in calls:
            sliced = call.args[0]
            self.assertIsInstance(sliced, ast.Subscript, "expected beats[:n]")
            upper = sliced.slice.upper
            self.assertIsInstance(upper, ast.Name)
            self.assertEqual(
                upper.id, "_MAX_PANELS",
                "the division is bounded by what a board holds, not by the "
                "expected range -- slicing to `high` cut the last beats off "
                "the story, and the last beats are where the reveal is")

    def test_a_short_scene_can_still_have_more_beats_than_the_range(self):
        """The case the bound made impossible.

        FX-01 is 74 words and needs five essential beats. Nothing may stop a
        division of it from having five panels.
        """
        low, high = worker._panel_bounds(74)
        self.assertLess(high, 5, "this test describes a story the range "
                                 "cannot accommodate; pick another length")
        beats = [{"shot": "medium", "subject": "Mira", "action": f"beat {i}",
                  "setting": "a waiting room", "caption": ""}
                 for i in range(8)]
        kept = worker._clean_shotlist(beats[:worker._MAX_PANELS], "")
        self.assertEqual(len(kept), 8,
                         f"a division of 8 was cut to the range {low}-{high}")

    def test_a_board_is_still_bounded_by_what_it_holds(self):
        """Sixty panels is a real limit; the range was not."""
        beats = [{"shot": "medium", "subject": "x", "action": f"a{i}",
                  "setting": "y", "caption": ""}
                 for i in range(worker._MAX_PANELS + 20)]
        kept = worker._clean_shotlist(beats[:worker._MAX_PANELS], "")
        self.assertEqual(len(kept), worker._MAX_PANELS)

    def test_the_range_is_still_computed_and_still_reported(self):
        """Removing the cap must not remove the warning.

        A division far outside what this much prose should produce is worth
        seeing before forty minutes of drawing. It is now said rather than
        silently applied.
        """
        import ast

        with open(os.path.join(HERE, "worker.py"), encoding="utf-8") as handle:
            src = handle.read()
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "op_shotlist")
        body = ast.get_source_segment(src, fn) or ""
        self.assertIn("_panel_bounds(", body)
        self.assertIn("out_of_range", body)
        self.assertIn("truncated_from", body)


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


class WhatTheCastReadMustNotDoToFX01(unittest.TestCase):
    """The fixture, pointed at the deterministic parts of the pipeline.

    Nothing here loads a model. What it checks is the code that runs either
    side of one: how a story is cut into units, how a panel's quote is matched
    back onto it, and what the coverage figure means -- measured against a
    human-authored key rather than against the pipeline's own last output.
    """

    @classmethod
    def setUpClass(cls):
        import json

        with open(os.path.join(HERE, "fixtures", "fx01.json"),
                  encoding="utf-8") as handle:
            cls.fx = json.load(handle)
        cls.story = cls.fx["source"]

    def test_the_story_divides_into_sentences_that_rebuild_it(self):
        units = worker._story_units(self.story)
        self.assertGreater(len(units), 4)
        for unit in units:
            self.assertIn(unit.strip(), self.story)

    def _quotes(self, *groups):
        out = []
        for group in groups:
            for item in group:
                start, end = item["span"]
                out.append({"source": self.story[start:end]})
        return out

    def test_the_beats_and_the_context_account_for_the_whole_story(self):
        """Every sentence is assigned to something, or the key is incomplete.

        Source accounting has four dispositions and this fixture uses two of
        them: a sentence carries a beat, or it is context that no panel has to
        depict. Anything left over is a sentence the answer key has no opinion
        about, which makes every coverage assertion below ambiguous.
        """
        panels = self._quotes(self.fx["beats"], self.fx["context_only"])
        coverage = worker._story_coverage(self.story, panels)
        self.assertEqual(coverage["missing_total"], 0,
                         f"unaccounted for: {coverage['missing']}")

    def test_the_essential_beats_alone_leave_only_context_behind(self):
        """What a board can legitimately not draw, and nothing more."""
        panels = self._quotes(
            [b for b in self.fx["beats"] if b["essential"]])
        missing = worker._story_coverage(self.story, panels)["missing"]
        context = [self.story[c["span"][0]:c["span"][1]] for c in
                   self.fx["context_only"]]
        for unit in missing:
            self.assertTrue(
                any(unit.strip() in c for c in context),
                f"{unit!r} is not a beat and is not declared context")

    def test_dropping_the_reveal_is_reported_as_missing(self):
        """The case the coverage figure exists for."""
        keep = [b for b in self.fx["beats"] if b["id"] != "b5"]
        panels = [{"source": self.story[b["span"][0]:b["span"][1]]}
                  for b in keep]
        coverage = worker._story_coverage(self.story, panels)
        self.assertGreater(coverage["missing_total"], 0)
        self.assertLess(coverage["percent"], 100)

    def test_a_paraphrase_is_snapped_back_to_the_author_s_words(self):
        """The anchor stores what the user wrote, not what the writer said."""
        snapped = worker._snap_source(
            "Mira hands Jonas the brass key from her coat", self.story)
        self.assertTrue(snapped)
        self.assertIn(snapped.strip().rstrip("."), self.story)

    def test_anchoring_everything_does_not_prove_the_panels_are_right(self):
        """BT-024, as a test rather than as a paragraph.

        Every sentence is quoted, so the coverage figure is perfect -- and the
        panel says Jonas gave Mira the key. Source accounting verifies
        provenance, not depicted meaning, and a board can score 100% while
        reversing the one action the scene is about.
        """
        units = worker._story_units(self.story)
        panels = [{"source": u, "action": "Jonas hands Mira the brass key"}
                  for u in units]
        coverage = worker._story_coverage(self.story, panels)
        self.assertEqual(coverage["percent"], 100)
        self.assertEqual(coverage["missing_total"], 0)
        # And the key says so.
        self.assertTrue(
            any("Mira gave it to Jonas" in q["a"]
                for q in self.fx["comprehension"]),
            "the fixture must carry the fact this board gets backwards")

    def test_the_speaker_of_the_only_line_is_not_in_the_cast_of_the_beat(self):
        """Speech is not presence: the guard is heard through a locked door."""
        line = next(s for s in self.fx["speech"])
        self.assertFalse(line["visible"])
        beat = next(b for b in self.fx["beats"] if b.get("speaker") == "guard")
        self.assertEqual(beat["actors"], ["guard"])
        self.assertNotIn("mira", beat["actors"])

    def test_the_narrator_is_detected_so_a_name_can_be_asked_for(self):
        self.assertTrue(worker._is_first_person(self.story))


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


class FakeWriter:
    """Stands in for `_write` and the picture reader, recording every call.

    `replies` are handed out in order to prompt-writing calls; the SAME check
    is answered by `verdict`. Nothing is loaded and nothing is inferred, so
    these tests can say exactly what op_assist does with each kind of answer
    a real writer gives -- clean, wrapped in reasoning, broken, or wrong.
    """

    def __init__(self, replies, verdict="SAME", regions=None):
        self.replies = list(replies)
        self.verdict = verdict
        self.regions = regions
        self.calls = []
        self.judged = []

    def write(self, _req_id, system, user, max_tokens=160, temperature=0.3,
              repo=None, label="", fraction=None, preview=None):
        if system == worker.SAME_REQUEST_SYSTEM:
            self.judged.append(user)
            return self.verdict
        self.calls.append({"system": system, "user": user, "temperature": temperature,
                           "max_tokens": max_tokens, "repo": repo, "preview": preview})
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def read(self, _req_id, _repo, _images, question):
        self.question = question
        return self.regions or ""


def _assist(fake, prompt, mode="generate", goal="enhance", target=None,
            picture=False, reader=True):
    req = {"prompt": prompt, "mode": mode, "goal": goal,
           "writer": "mlx-community/Qwen3.5-9B-MLX-4bit",
           "target": target or {"id": "flux2-klein-4b-4bit", "family": "flux2",
                                "name": "FLUX.2 Klein 4B"}}
    if not reader:
        req["assistant"] = None
    patches = {
        "_write": fake.write,
        "_read_picture": fake.read,
        "_stage_vault_inputs": lambda _i: ["/tmp/staged.png"] if picture else [],
        "_downscale_for_assist": lambda paths: paths,
        "_discard_staged": lambda _paths: None,
        "emit": lambda *_a, **_k: None,
        "log": lambda *_a, **_k: None,
    }
    with unittest.mock.patch.multiple(worker, **patches):
        return worker.op_assist("t", req)


def _reply(**fields):
    return json.dumps(fields)


class AssistWritesForTheTargetModel(unittest.TestCase):
    """The writer is told which model, which mode and which goal."""

    def test_the_target_model_reaches_the_system_prompt(self):
        for target, expect in (
            ({"id": "flux2-klein-9b-4bit", "family": "flux2"}, "Qwen3 language model"),
            ({"id": "qwen-image-2512-8bit", "family": "qwen"}, "renders written text"),
            ({"id": "z-image-turbo-4bit", "family": "z-image"}, "photographic"),
            ({"id": "flux1-dev-4bit", "family": "flux"}, "T5 encoder"),
        ):
            with self.subTest(target=target["id"]):
                fake = FakeWriter([_reply(prompt="a cat on a chair, in soft light")])
                _assist(fake, "a cat on a chair", target=target)
                self.assertIn(expect, fake.calls[0]["system"])

    def test_the_best_writer_is_used_not_a_hardcoded_one(self):
        fake = FakeWriter([_reply(faithful="A cat on a chair.",
                                  prompt="A cat sits on a chair.")])
        _assist(fake, "a cat on a chair")
        self.assertEqual(fake.calls[0]["repo"], "mlx-community/Qwen3.5-9B-MLX-4bit")

    def test_no_copyable_examples_reach_any_writer(self):
        """The tabby on a worn oak chair was copied into unrelated requests."""
        import prompt_assist as pa

        for goal in pa.GOALS:
            for mode in pa.MODES:
                for profile in pa.PROFILES:
                    text = pa.system_prompt(goal, mode, profile).lower()
                    for copied in ("tabby", "oak chair", "brick house",
                                   "teapot", "wet street", "oil on canvas"):
                        self.assertNotIn(copied, text, (goal, mode, profile))

    def test_the_reply_says_what_it_was_written_for(self):
        fake = FakeWriter([_reply(prompt="A cat sits on a chair.")])
        out = _assist(fake, "a cat on a chair", goal="tighten")
        self.assertEqual(out["version"], 2)
        self.assertEqual(out["goal"], "tighten")
        self.assertEqual(out["mode"], "generate")
        self.assertEqual(out["target_label"], "FLUX.2 Klein 4B")


class AssistReadsWhateverTheWriterSends(unittest.TestCase):
    """Think tags, broken JSON and prose are all still answers."""

    GOOD = _reply(
        subject="a cat", quoted_text=[], constraints=[],
        faithful="A cat sitting on a chair.",
        prompt="A grey cat sits upright on a wooden kitchen chair, lit by "
               "soft morning light from a window, seen from eye level.",
        additions=["grey", "wooden kitchen", "lit by soft morning light from a window"],
        styles=["watercolour illustration with loose washes"])

    def test_three_variants_from_one_call(self):
        fake = FakeWriter([self.GOOD])
        out = _assist(fake, "a cat on a chair")
        self.assertEqual(len(fake.calls), 1, "one write, not a chain")
        self.assertEqual([v["kind"] for v in out["variants"]],
                         ["faithful", "enriched", "stylistic"])
        self.assertEqual(out["outcome"], "proposal")
        self.assertIn("morning light", out["prompt"], "the enriched one leads")
        stylistic = out["variants"][2]
        self.assertTrue(stylistic["prompt"].endswith(
            "Watercolour illustration with loose washes."))
        self.assertIn("Watercolour illustration with loose washes",
                      stylistic["additions"])

    def test_additions_are_only_phrases_the_prompt_really_contains(self):
        reply = _reply(prompt="A cat sits on a chair by a window.",
                       additions=["by a window", "a velvet cushion"])
        out = _assist(FakeWriter([reply]), "a cat on a chair")
        self.assertEqual(out["added"], ["by a window"],
                         "a chip must remove something that is there")

    def test_think_tags_are_stripped_before_parsing(self):
        fake = FakeWriter(["<think>\nThe user wants a cat.\n</think>\n" + self.GOOD])
        out = _assist(fake, "a cat on a chair")
        self.assertEqual(out["outcome"], "proposal")
        self.assertNotIn("think", out["prompt"].lower())

    def test_an_unclosed_think_block_is_not_the_prompt(self):
        fake = FakeWriter(["<think>I should describe", self.GOOD])
        out = _assist(fake, "a cat on a chair")
        self.assertEqual(len(fake.calls), 2, "the empty answer was retried")
        self.assertNotIn("should describe", out["prompt"])

    def test_json_cut_off_mid_answer_is_still_read(self):
        truncated = ('{"subject": "a cat", "faithful": "A cat on a chair.", '
                     '"prompt": "A grey cat sits on a wooden chair in soft light')
        out = _assist(FakeWriter([truncated]), "a cat on a chair")
        self.assertEqual(out["outcome"], "proposal")
        self.assertIn("soft light", out["prompt"])

    def test_json_with_a_trailing_comma_is_still_read(self):
        out = _assist(FakeWriter(['{"prompt": "A cat on a chair by a window.",}']),
                      "a cat on a chair")
        self.assertIn("window", out["prompt"])

    def test_nothing_usable_twice_is_said_plainly(self):
        fake = FakeWriter(["{{{", ""])
        out = _assist(fake, "a cat on a chair")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(out["outcome"], "rewrite_rejected")
        self.assertEqual(out["prompt"], "a cat on a chair", "the words are kept")
        self.assertIn("nothing usable", out["note"])

    def test_a_prose_answer_keeps_every_line(self):
        """Only the first line used to be kept, so a second-line detail vanished."""
        prose = "A cat sits on a chair.\nThe chair is red and has no cushion."
        out = _assist(FakeWriter([prose]), "a cat on a chair", goal="tighten")
        self.assertIn("no cushion", out["prompt"])

    def test_unclear_is_reported_as_unclear(self):
        out = _assist(FakeWriter(['{"unclear": true}']), "make it better")
        self.assertEqual(out["outcome"], "unclear")
        self.assertTrue(out["unclear"])

    def test_cancel_propagates(self):
        fake = FakeWriter([worker.Cancelled("stopped")])
        with self.assertRaises(worker.Cancelled):
            _assist(fake, "a cat on a chair")
        self.assertEqual(fake.judged, [], "no verdict after a cancel")

    def test_the_draft_shows_prose_not_json(self):
        fake = FakeWriter([self.GOOD])
        _assist(fake, "a cat on a chair")
        preview = fake.calls[0]["preview"]
        self.assertEqual(preview('{"subject": "a cat", "prompt": "A grey cat si'),
                         "A grey cat si")


class AssistHoldsWhatTheRequestFixes(unittest.TestCase):
    """Counts, negations, quoted text and pace survive or are named as lost."""

    REQUEST = "2 red cars with no sunroof, plain white background"

    def test_a_dropped_negation_is_retried_cooler_and_never_waved_through(self):
        lost = _reply(prompt="Two red cars with a glass sunroof on a plain white background.")
        kept = _reply(prompt="Two red cars, neither with a sunroof, on a plain "
                             "white background.")
        fake = FakeWriter([lost, kept], verdict="SAME")
        out = _assist(fake, self.REQUEST)
        self.assertEqual(len(fake.calls), 2)
        self.assertLess(fake.calls[1]["temperature"], fake.calls[0]["temperature"])
        self.assertEqual(fake.judged, [],
                         "a lost negation is never put to the writer's own judgement")
        self.assertEqual(out["dropped"], [])
        self.assertIn("neither with a sunroof", out["prompt"])

    def test_a_loss_that_survives_the_retry_is_named(self):
        lost = _reply(prompt="Three red cars on a white background.")
        out = _assist(FakeWriter([lost, lost]), self.REQUEST)
        self.assertEqual(out["outcome"], "proposal")
        self.assertIn("2 red cars", out["dropped"])
        self.assertIn("no sunroof", out["dropped"])
        self.assertIn("no longer says", out["note"])
        self.assertIn("2 red cars", out["variants"][0]["missing"])

    def test_the_constraints_are_handed_to_the_writer(self):
        fake = FakeWriter([_reply(prompt="Two red cars without a sunroof on a "
                                         "plain white background.")])
        _assist(fake, self.REQUEST)
        self.assertIn("Must keep: 2 red cars; no sunroof", fake.calls[0]["user"])

    def test_quoted_text_with_4k_survives_the_cleaner(self):
        request = 'A shop sign reading exactly "BEST QUALITY 4K", 8k, masterpiece'
        fake = FakeWriter([_reply(
            quoted_text=["BEST QUALITY 4K"],
            prompt='A painted shop sign that reads "Best Quality 4k" in bold '
                   'white capitals above a door.')])
        out = _assist(fake, request, goal="tighten")
        self.assertIn('"BEST QUALITY 4K"', fake.calls[0]["user"],
                      "the writer was never shown the lettering")
        self.assertEqual(sorted(out["removed"]), ["8k", "masterpiece"])
        self.assertIn('"BEST QUALITY 4K"', out["prompt"],
                      "lettering whose case drifted is put back exactly")
        self.assertEqual(out["dropped"], [])

    def test_quoted_text_the_writer_lost_is_flagged(self):
        request = 'A sign reading "OPEN LATE"'
        lost = _reply(prompt="A neon sign glowing above a doorway at night.")
        out = _assist(FakeWriter([lost, lost]), request)
        self.assertIn('"OPEN LATE"', out["dropped"])

    def test_a_sprint_stays_a_sprint(self):
        request = "A runner sprints quickly across the track. Locked camera."
        slowed = _reply(prompt="A runner moves in a slow, grounded drift across "
                               "the track; the camera stays locked.")
        kept = _reply(prompt="A runner sprints quickly across the red track, "
                             "arms pumping, while the locked camera holds the lane.")
        fake = FakeWriter([slowed, kept])
        out = _assist(fake, request, mode="video",
                      target={"id": "wan22-ti2v-5b-8bit", "family": "wan"})
        system = fake.calls[0]["system"]
        self.assertNotIn("Never 'suddenly', 'quickly'", system)
        self.assertIn("Keep the pace they asked for", system)
        self.assertIn("sprints quickly", out["prompt"])
        self.assertEqual(out["dropped"], [])

    def test_a_soft_miss_is_judged_once_and_accepted(self):
        """The subject reworded is the case a good paraphrase trips."""
        reply = _reply(subject="dog", prompt="A golden retriever wearing a "
                                             "velvet robe and a gold crown.")
        fake = FakeWriter([reply], verdict="SAME")
        out = _assist(fake, "my dog but make him look like a king", goal="tighten")
        self.assertEqual(len(fake.judged), 1)
        self.assertEqual(len(fake.calls), 1, "no retry once it is judged the same")
        self.assertEqual(out["dropped"], [])
        self.assertIn("dog", out["variants"][0]["reworded"])

    def test_tighten_is_one_variant_and_a_smaller_budget(self):
        fake = FakeWriter([_reply(prompt="A cat on a chair.")])
        _assist(fake, "a cat on a chiar", goal="tighten")
        tight = fake.calls[0]["max_tokens"]
        fake2 = FakeWriter([_reply(prompt="A cat on a chair.")])
        _assist(fake2, "a cat on a chiar", goal="enhance")
        self.assertLess(tight, fake2.calls[0]["max_tokens"])
        self.assertIn("Add nothing", fake.calls[0]["system"])


class AssistEditsLookBeforeTheyResolve(unittest.TestCase):
    """An edit names the region it means, or says it could not look."""

    REGIONS = ("1. man in a jacket | centre, foreground | red, black\n"
               "2. brick wall | background | brown\n"
               "3. bicycle | left, background | blue\n")

    def test_it_resolves_to_the_most_plausible_region(self):
        fake = FakeWriter([_reply(prompt="Make the man's jacket in the centre "
                                         "foreground blue, keeping the wall unchanged.")],
                          regions=self.REGIONS)
        out = _assist(fake, "make it blue", mode="edit", picture=True,
                      target={"id": "flux1-kontext-4bit", "family": "flux"})
        self.assertTrue(out["saw_image"])
        self.assertIn("'it' most likely means: man in a jacket (centre, "
                      "foreground, red, black)", fake.calls[0]["user"])
        self.assertTrue(out["target_region"].startswith("man in a jacket"))
        self.assertEqual(len(out["regions"]), 3)
        self.assertIn("Kontext", fake.calls[0]["system"])
        self.assertIn("left", fake.question, "the reader is asked for positions")

    def test_a_named_thing_beats_the_prominent_one(self):
        fake = FakeWriter([_reply(prompt="Make the bicycle on the left red.")],
                          regions=self.REGIONS)
        out = _assist(fake, "make the bike on the left red", mode="edit",
                      picture=True)
        # "bike" is not "bicycle", so the place settles it -- but only among
        # things it could mean; with nothing named, "it" is not asked about.
        self.assertNotIn("'it' most likely means", fake.calls[0]["user"])
        out2 = _assist(FakeWriter([_reply(prompt="Make the bicycle red.")],
                                  regions=self.REGIONS),
                       "make the bicycle red", mode="edit", picture=True)
        self.assertTrue(out2["target_region"].startswith("bicycle"))
        self.assertEqual(out["outcome"], "proposal")

    def test_without_a_reader_it_says_it_did_not_look(self):
        fake = FakeWriter([_reply(prompt="Make the jacket red, keeping everything "
                                         "else unchanged.")])
        out = _assist(fake, "make the jacket red", mode="edit", picture=True,
                      reader=False)
        self.assertFalse(out["saw_image"])
        self.assertIn("not looked at", out["note"])
        self.assertIn("You cannot see the picture", fake.calls[0]["system"])
        self.assertNotIn("The picture contains", fake.calls[0]["user"])

    def test_an_edit_is_written_as_an_instruction(self):
        fake = FakeWriter([_reply(prompt="Make the jacket red.")])
        _assist(fake, "make the jacket red", mode="edit", reader=False)
        self.assertIn("imperative", fake.calls[0]["system"])
        self.assertIn("not a description of a new picture", fake.calls[0]["system"])


class ALongerWordIsADifferentWord(unittest.TestCase):
    """A brief could swap the subject for a word that starts the same way.

    A stem compared by unbounded prefix is not a stem match -- it is "starts
    with the same letters", and a longer word starting with the same letters
    is a different thing rather than an inflection. So every one of these was
    once accepted as having kept the subject:

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

    def test_the_swap_reaches_the_check_that_gates_a_rewrite(self):
        """`_carries` is the all-words check, and it must see the loss."""
        for original, line, _lost in self.SWAPS:
            with self.subTest(original=original):
                self.assertFalse(
                    worker._carries(original, line),
                    f"{line!r} was accepted as carrying {original!r}")



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
        shots = ["medium"] * 3
        meta = self._meta(shots, [list(self.THREE)] * 3)
        _page, short = worker._render_page(self._panels(3), [0, 1, 2],
                                           meta, style)
        self.assertTrue(short, "a 244px cell cannot letter three balloons")
        for record in short:
            self.assertEqual(record["spoken"], 3)
            self.assertLess(record["lettered"], 3)
            self.assertIn(record["panel"], (0, 1, 2))

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
                   "never once thought of it as hers. " * 8).strip()
        style = self._style(836)
        shots = ["medium"] * 3
        meta = self._meta(shots, [[]] * 3, [caption] * 3)
        _page, short = worker._render_page(self._panels(3), [0, 1, 2],
                                           meta, style)
        self.assertTrue(short, "this caption does not fit a 244px cell")
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
        im_h = round(im_w * 0.78)
        long_talk = [{"speaker": f"S{i}", "text": "The bus does not come "
                                                  "after nine o'clock."}
                     for i in range(9)]
        meta = self._meta(["medium"], [long_talk])
        _page, short = worker._render_strip(panels, meta, style)
        drawn = short[0]["lettered"] if short else len(long_talk)
        used = 16
        for line in long_talk[:drawn]:
            used += worker._bubble_height(
                probe, line["text"], line["speaker"],
                worker._lettering(line["text"], base, style["font"]),
                min(im_w - 40, 420), style["line_h"])
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
