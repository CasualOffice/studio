"""Prompt help: what the writer is asked, and how its answer is checked.

No model or platform dependencies, so every rule here can be tested without
loading anything. `worker.op_assist` does the loading and the staging, and
calls in here for everything else.

The previous enhancer had one system prompt for every image model, three
concrete few-shot examples the writer copied into unrelated requests, and a
stack of word-matching validators that could not see counts, negation or which
colour belonged to which object -- while rejecting good paraphrases for not
repeating a word. What replaced it:

* The writer is told which model the prompt is for, whether it is a new
  picture, an edit or a clip, and what the person wants from it: a fuller
  description (enhance), the same request said clearly (tighten), or
  stylistic alternatives (style).
* It answers in JSON: the subject, any quoted text, the constraints it has to
  keep, the prompt, and the phrases it added. One call gives every variant.
* What must survive is extracted from the request deterministically -- quoted
  text, counts, negations, colour-object bindings, positions, pace -- and each
  variant is checked against it. A model asked to list its own constraints
  can simply leave one out, so its list is only ever added to ours.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

GOALS = ("enhance", "tighten", "style")
MODES = ("generate", "edit", "video")

# --------------------------------------------------------------------------
# Who the prompt is for
# --------------------------------------------------------------------------

# One entry per kind of model the studio drives, not per checkpoint. `words`
# is the length the full description aims for; a faithful rewrite is never
# held to it. `guide` is what the model's authors and its text encoder say it
# reads well, in a sentence or two the writer can follow.
PROFILES: dict[str, dict[str, Any]] = {
    "flux2": {
        "label": "FLUX.2", "words": (60, 120),
        "guide": "FLUX.2 reads prompts with a Qwen3 language model, so it "
                 "follows plain, grammatical sentences written the way you "
                 "would brief a person. Put the subject first, then the "
                 "setting, then the light, then camera and framing, then the "
                 "medium. No tag lists and no weights.",
    },
    "qwen_image": {
        "label": "Qwen-Image", "words": (80, 200),
        "guide": "Qwen-Image handles long, detailed prompts and renders "
                 "written text accurately. Text that must appear goes in "
                 "double quotes exactly as given, with its lettering style, "
                 "size, colour and place in the picture said in words.",
    },
    "z_image": {
        "label": "Z-Image", "words": (30, 70),
        "guide": "Z-Image does best with a short, photographic description: "
                 "subject, setting, light, then lens and framing the way a "
                 "photographer would say them.",
    },
    "flux1": {
        "label": "FLUX.1", "words": (40, 90),
        "guide": "FLUX.1 reads prompts with a T5 encoder and follows natural "
                 "sentences, not keyword lists. Subject first.",
    },
    "generic": {
        "label": "the image model", "words": (40, 100),
        "guide": "Write natural sentences, subject first.",
    },
    "kontext": {
        "label": "FLUX.1 Kontext", "words": (8, 50), "edit": True,
        "guide": "Kontext follows short imperative instructions. Name the "
                 "thing to change by what it looks like and where it is, say "
                 "what it becomes, then say what stays the same -- for a "
                 "person, their face, pose and identity.",
    },
    "qwen_edit": {
        "label": "Qwen-Image-Edit", "words": (8, 70), "edit": True,
        "guide": "Qwen-Image-Edit follows imperative instructions and can "
                 "rewrite lettering: new text goes in double quotes exactly "
                 "as it must read.",
    },
    "flux2_edit": {
        "label": "FLUX.2", "words": (8, 60), "edit": True,
        "guide": "FLUX.2 edits from one plain imperative sentence: what to "
                 "change, where it is, what it becomes, and what to keep.",
    },
    "edit": {
        "label": "the image editor", "words": (8, 50), "edit": True,
        "guide": "Write one imperative instruction: what to change, where it "
                 "is, what it becomes, and what to keep.",
    },
    "wan": {
        "label": "Wan", "words": (40, 100), "video": True,
        "guide": "Wan follows a description of one continuous shot in the "
                 "present tense: who or what is in it, what they do and how "
                 "they move, then what the camera does.",
    },
    "ltx": {
        "label": "LTX-Video", "words": (40, 110), "video": True,
        "guide": "LTX-Video likes one flowing paragraph in the present "
                 "tense: the action first, then movements and gestures, then "
                 "appearance, setting, camera movement and light.",
    },
    "video": {
        "label": "the video model", "words": (40, 100), "video": True,
        "guide": "Describe one continuous shot in the present tense: the "
                 "subject and how it moves, then what the camera does.",
    },
}


def profile_for(mode: str, model_id: str | None = "", family: str | None = "") -> str:
    """The profile key for the model the prompt is going to.

    The mode wins over the model: an edit is an edit whichever checkpoint
    runs it, and an image model handed a video request is a caller's
    mistake, not a reason to write a still description.
    """
    mid = (model_id or "").lower()
    fam = (family or "").lower()
    if mode == "video":
        if fam == "ltx" or "ltx" in mid:
            return "ltx"
        if fam == "wan" or "wan" in mid:
            return "wan"
        return "video"
    if mode == "edit":
        if "kontext" in mid:
            return "kontext"
        if fam == "qwen" or "qwen-image" in mid:
            return "qwen_edit"
        if fam == "flux2" or "flux2" in mid:
            return "flux2_edit"
        return "edit"
    if fam == "flux2" or "flux2" in mid:
        return "flux2"
    if fam == "qwen" or "qwen-image" in mid:
        return "qwen_image"
    if fam == "z-image" or "z-image" in mid:
        return "z_image"
    if fam == "flux" or "flux1" in mid:
        return "flux1"
    return "generic"


# --------------------------------------------------------------------------
# Words
# --------------------------------------------------------------------------

# British and American spellings of the words this module compares most.
_SPELLINGS = {"color": "colour", "colors": "colours", "colored": "coloured",
              "gray": "grey", "colorless": "colourless"}


def _tokens(text: str) -> list[str]:
    """Lowercase words, hyphenated compounds split, spellings unified."""
    words = re.findall(r"[a-z0-9]+(?:['’][a-z]+)?", text.lower())
    return [_SPELLINGS.get(w, w) for w in words]


def _forms(word: str) -> set[str]:
    """The shapes an inflected word could have come from.

    A set rather than a single stem because English does not agree with any
    one suffix rule: "trees" loses an s, "boxes" loses es, "running" loses
    ing and a doubled n. Two words are the same if any of their forms meet.
    """
    w = word.lower()
    out = {w}
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        out.add(w[:-1])
    if len(w) > 4 and w.endswith("es"):
        out.add(w[:-2])
    if len(w) > 4 and w.endswith("ies"):
        out.add(w[:-3] + "y")
    for suffix in ("ing", "ed", "ly"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            base = w[: -len(suffix)]
            out |= {base, base + "e"}
            if len(base) > 2 and base[-1] == base[-2]:
                out.add(base[:-1])
    return out


def _edits(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    row = list(range(len(a) + 1))
    for j, cb in enumerate(b, start=1):
        prev, row[0] = row[0], j
        for i, ca in enumerate(a, start=1):
            prev, row[i] = row[i], min(row[i] + 1, row[i - 1] + 1, prev + (ca != cb))
    return row[-1]


def same_word(a: str, b: str) -> bool:
    return bool(_forms(a) & _forms(b))


def near_word(a: str, b: str) -> bool:
    """The same word, or one typed a letter or two wrong.

    Only for deciding whether the writer's subject came from the request: a
    request for a "hous" is a request for a house. The first letter has to
    agree, which is what keeps "cat" from becoming "hat".
    """
    if same_word(a, b):
        return True
    if len(a) < 4 or len(b) < 4 or a[0] != b[0]:
        return False
    limit = 2 if max(len(a), len(b)) >= 7 else 1
    return _edits(a, b) <= limit


_FILLER = {
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "at", "with", "for",
    "is", "are", "be", "it", "its", "this", "that", "my", "me", "i", "some",
    "very", "please", "make", "into", "as", "by", "from", "his", "her",
    "their", "but", "look", "like", "just", "picture", "image", "photo",
}


def _content(text: str) -> list[str]:
    return [w for w in _tokens(text) if w not in _FILLER and len(w) > 1]


# --------------------------------------------------------------------------
# Quoted text, and the cleaner that must never touch it
# --------------------------------------------------------------------------

# Straight double quotes, curly double quotes, and single quotes that are not
# apostrophes: "the cat's toy" must not open a quotation.
_QUOTE_RE = re.compile(
    r'"([^"\n]{1,200})"'
    r"|“([^”\n]{1,200})”"
    r"|(?<![\w])'([^'\n]{1,200})'(?![\w])"
)


def quoted(text: str) -> list[str]:
    """Every quoted string in the request, verbatim and in order."""
    out = []
    for m in _QUOTE_RE.finditer(text):
        inner = next(g for g in m.groups() if g is not None)
        if inner.strip() and inner not in out:
            out.append(inner)
    return out


def _outside_quotes(text: str) -> list[tuple[bool, str]]:
    """The text as (is_quoted, piece) runs, quotes included in their piece."""
    out, at = [], 0
    for m in _QUOTE_RE.finditer(text):
        if m.start() > at:
            out.append((False, text[at:m.start()]))
        out.append((True, m.group(0)))
        at = m.end()
    if at < len(text):
        out.append((False, text[at:]))
    return out


# Words that promise quality without describing anything. They survive from
# Stable Diffusion prompting, where they acted as style tokens, and none of
# the models here was trained to read them that way.
#
# Deliberately narrow. "beautiful", "professional" and "perfect" modify the
# subject -- "a professional kitchen" is a kind of kitchen -- so only terms
# that name a rendering target, an award or a resolution belong here. And
# never inside quotes: a sign that reads "BEST QUALITY 4K" is lettering the
# person wants drawn, and stripping it was how a request for that sign came
# back as "letters between two blank spaces".
FOLKLORE = (
    "masterpiece", "best quality", "high quality", "highly detailed",
    "ultra detailed", "ultra realistic", "8k", "4k", "uhd", "hdr",
    "award winning", "award-winning", "trending on artstation",
    "artstation", "unreal engine", "octane render",
)


def clean_request(text: str) -> tuple[str, list[str]]:
    """Remove prompt folklore outside quotes. Quoted text is never touched."""
    removed: list[str] = []
    pieces: list[str] = []
    for is_quote, piece in _outside_quotes(text):
        if is_quote:
            pieces.append(piece)
            continue
        for phrase in sorted(FOLKLORE, key=len, reverse=True):
            pattern = re.compile(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])",
                                 re.IGNORECASE)
            if pattern.search(piece):
                if phrase not in removed:
                    removed.append(phrase)
                piece = pattern.sub("", piece)
        pieces.append(piece)
    cleaned = "".join(pieces)
    if removed:
        cleaned = re.sub(r"\s*[,;]\s*(?=[,;]|$)", "", cleaned)
        cleaned = re.sub(r"^\s*[,;]\s*", "", cleaned)
        cleaned = re.sub(r"\s+([,;.])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,;-"), removed


# --------------------------------------------------------------------------
# What must survive
# --------------------------------------------------------------------------

_NUMBERS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "dozen": 12, "twenty": 20,
}
COLOURS = {
    "red", "orange", "yellow", "green", "blue", "purple", "violet", "pink",
    "black", "white", "grey", "brown", "gold", "golden", "silver", "beige",
    "teal", "cyan", "magenta", "turquoise", "navy", "maroon", "crimson",
    "scarlet", "ivory", "cream",
}
# Words that sit between a number or a colour and its noun without being it.
_MODIFIERS = {
    "small", "big", "large", "tiny", "huge", "little", "old", "young", "new",
    "tall", "short", "long", "identical", "matching", "bright", "dark",
    "light", "pale", "deep", "wooden", "metal", "glass", "plastic", "shiny",
    "matte", "round", "square", "fluffy", "furry", "striped", "spotted",
    "same", "different", "separate", "more", "other", "of",
}
_NEGATORS = {"no", "without", "not", "never", "nor", "neither", "zero", "none",
             "dont", "don't", "doesn't", "lacking", "free", "absent"}
_SKIP_AFTER_NEGATION = {"a", "an", "any", "the", "one", "more", "other",
                        "visible", "obvious"}
_PACE = {
    "quickly", "quick", "fast", "rapidly", "rapid", "slowly", "slow",
    "sprint", "sprints", "sprinting", "dash", "dashes", "rush", "rushes",
    "race", "races", "racing", "sudden", "suddenly", "gradually", "gently",
    "explode", "explodes", "exploding", "speeding", "crawl", "crawls",
    "timelapse", "hurry", "hurries", "frantic", "frantically", "leisurely",
}
_CAMERA = (
    "locked camera", "static camera", "camera locked", "locked-off",
    "handheld", "tracking shot", "dolly", "pan", "pans", "zoom", "zooms",
    "orbit", "orbits", "crane", "tilt", "tilts", "push in", "pull back",
    "slow motion", "time-lapse",
)
_RELATION_RE = re.compile(
    r"\b(?:(?:to|on|at) the (?:left|right)|(?:left|right) of|in front of|"
    r"on top of|above|below|behind|beneath|underneath|between)\b",
    re.IGNORECASE)
_RELATION_KEY = {"in front of": "front", "on top of": "top"}


def _constraint(kind: str, text: str, hard: bool, **match: Any) -> dict[str, Any]:
    return {"kind": kind, "text": text, "hard": hard, **match}


def _clause_constraints(words: list[str],
                        seen_bindings: set[tuple[str, str]]) -> list[dict[str, Any]]:
    """Counts, negations and colour bindings within one clause."""
    out: list[dict[str, Any]] = []
    for i, w in enumerate(words):
        value = int(w) if w.isdigit() and 1 < int(w) < 1000 else _NUMBERS.get(w)
        if value is None:
            continue
        for j in range(i + 1, min(len(words), i + 5)):
            noun = words[j]
            if noun in COLOURS or noun in _MODIFIERS or noun in _FILLER:
                continue
            out.append(_constraint("count", " ".join(words[i:j + 1]), True,
                                   number=value, noun=noun))
            break

    i = 0
    while i < len(words):
        if words[i] not in _NEGATORS or words[i] in ("free", "absent"):
            i += 1
            continue
        j = i + 1
        while j < len(words):
            while j < len(words) and (words[j] in _SKIP_AFTER_NEGATION
                                      or words[j] in COLOURS
                                      or words[j] in _MODIFIERS):
                j += 1
            if j >= len(words) or words[j] in _FILLER or words[j] in _NEGATORS:
                break
            out.append(_constraint("negation", f"{words[i]} {words[j]}", True,
                                   noun=words[j]))
            # "no shading or colour": one negation, two things it covers.
            if j + 2 < len(words) and words[j + 1] in ("or", "nor", "and"):
                j += 2
                continue
            break
        i = max(j, i + 1)

    for i, w in enumerate(words):
        if w not in COLOURS:
            continue
        # A colour binds forward to its noun ("a red cube"), skipping a
        # material or a size in between. A colour after a verb -- "wears
        # red", "painted red" -- belongs to what came before, which the
        # owner pattern in `extract_constraints` handles.
        if i and words[i - 1] in ("wears", "wearing", "in", "painted", "is",
                                  "are", "dressed", "coloured", "turn", "turns"):
            continue
        for j in range(i + 1, min(len(words), i + 3)):
            noun = words[j]
            if noun in _MODIFIERS:
                continue
            if noun in COLOURS or noun in _FILLER or noun in _NEGATORS:
                break
            if (w, noun) not in seen_bindings:
                seen_bindings.add((w, noun))
                out.append(_constraint("colour", f"{w} {noun}", True,
                                       colour=w, noun=noun))
            break
    return out


def extract_constraints(request: str) -> list[dict[str, Any]]:
    """What the request fixes, read from its words rather than from the writer.

    Hard constraints are the ones a rewrite is known to lose and a model
    judging its own rewrite is known to wave through: quoted text, counts,
    negations, colours bound to objects, positions and pace. A missed hard
    constraint is never excused by the writer saying the rewrite is the same.
    """
    out: list[dict[str, Any]] = []
    for text in quoted(request):
        out.append(_constraint("quoted", f'"{text}"', True, exact=text))

    # Everything else is read outside the quotes, so a sign reading "NO
    # PARKING" is lettering, not a negation. And a clause at a time, so the
    # colour that ends one clause is not bound to the name that starts the
    # next: "Mira wears red; Jonas wears blue" has no red Jonas in it.
    plain = " ".join(p for q, p in _outside_quotes(request) if not q)
    seen_bindings: set[tuple[str, str]] = set()
    for clause in re.split(r"[.;:!?,()]", plain):
        out += _clause_constraints(_tokens(clause), seen_bindings)

    # "Mira wears red": the colour belongs to the person named before it.
    for m in re.finditer(
            r"\b([A-Z][a-z]+)\s+(?:wears|is wearing|wearing|is dressed in|"
            r"dressed in|in)\s+(?:a |an )?([a-z]+)\b", request):
        name, colour = m.group(1), m.group(2).lower()
        colour = _SPELLINGS.get(colour, colour)
        if colour in COLOURS and (colour, name.lower()) not in seen_bindings:
            seen_bindings.add((colour, name.lower()))
            out.append(_constraint("colour", f"{name} in {colour}", True,
                                   colour=colour, owner=name.lower()))

    for m in _RELATION_RE.finditer(plain):
        phrase = m.group(0).lower()
        key = _RELATION_KEY.get(phrase) or next(
            (k for k in ("left", "right") if k in phrase), phrase)
        out.append(_constraint("relation", phrase, True, word=key))

    words = _tokens(plain)
    for w in words:
        if w in _PACE and not any(c["kind"] == "pace" and c["word"] == w for c in out):
            out.append(_constraint("pace", w, True, word=w))

    low = " ".join(words)
    for phrase in _CAMERA:
        key = " ".join(_tokens(phrase))
        if re.search(r"\b" + re.escape(key) + r"\b", low):
            if not any(c["kind"] == "camera" and key in c["text"] for c in out):
                out.append(_constraint("camera", key, False, words=key.split()))
    return out


def writer_constraints(listed: list[str], request: str,
                       known: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The writer's own list, kept only where the request actually says it.

    Taken as an addition to the extracted constraints, never a replacement: a
    model asked to name what it must keep can simply leave a thing out. And a
    listed constraint the request does not contain is the writer's invention,
    which is the opposite of something to lock.
    """
    asked = _content(request)
    out = []
    for text in listed[:8]:
        words = _content(text)
        if not words or len(words) > 6:
            continue
        if not all(any(near_word(w, a) for a in asked) for w in words):
            continue
        if any(set(words) <= set(_tokens(k["text"])) for k in known + out):
            continue
        out.append(_constraint("listed", text, False, words=words))
    return out


def _positions(words: list[str], target: str) -> list[int]:
    return [i for i, w in enumerate(words) if same_word(w, target)]


def satisfied(c: dict[str, Any], text: str) -> bool:
    """Whether one constraint is still true of a candidate prompt."""
    kind = c["kind"]
    if kind == "quoted":
        return c["exact"] in text
    words = _tokens(text)
    if kind == "count":
        for i, w in enumerate(words):
            value = int(w) if w.isdigit() else _NUMBERS.get(w)
            if value == c["number"] and any(
                    same_word(x, c["noun"]) for x in words[i + 1:i + 6]):
                return True
        return False
    if kind == "negation":
        noun = c["noun"]
        for i in _positions(words, noun):
            if any(x in _NEGATORS for x in words[max(0, i - 4):i]):
                return True
            # "sunroof-free", "shadow-free"
            if i + 1 < len(words) and words[i + 1] in ("free", "less"):
                return True
        return any(w in (noun + "less", noun + "free") for w in words)
    if kind == "colour":
        colour = c["colour"]
        if "owner" in c:
            for i in _positions(words, c["owner"]):
                window = words[i + 1:i + 10]
                if any(same_word(x, colour) for x in window):
                    return True
            return False
        for i in _positions(words, c["noun"]):
            # Before the noun, with no other colour in between ...
            for x in reversed(words[max(0, i - 4):i]):
                if same_word(x, colour):
                    return True
                if x in COLOURS:
                    break
            # ... or after it, joined: "a cube painted red", "a cube of red
            # glass", "a coat in deep red".
            after = words[i + 1:i + 4]
            for k, x in enumerate(after):
                if same_word(x, colour) and k > 0 and after[k - 1] in (
                        "in", "painted", "coloured", "dyed", "of", "deep",
                        "bright", "pale", "dark"):
                    return True
        return False
    if kind == "relation":
        return any(w == c["word"] for w in words) or c["word"] in " ".join(words)
    if kind == "pace":
        return any(same_word(w, c["word"]) for w in words)
    if kind in ("subject", "target"):
        # Any of its words: "a golden retriever" is still said by "the
        # retriever", and the colour of a target may be what is changing.
        return any(any(near_word(w, x) for x in words) for w in c["words"])
    if kind in ("camera", "listed"):
        return all(any(same_word(w, x) or near_word(w, x) for x in words)
                   for w in c["words"])
    return True


def _fix_quotes(text: str, locked: list[dict[str, Any]]) -> str:
    """Put quoted text back exactly as written when only its case drifted.

    The one repair made without asking: "Best Quality 4k" for a sign that
    must read "BEST QUALITY 4K" is the right lettering in the wrong case, and
    restoring the case cannot change what the picture is.
    """
    for c in locked:
        if c["kind"] != "quoted" or c["exact"] in text:
            continue
        at = text.lower().find(c["exact"].lower())
        if at >= 0:
            text = text[:at] + c["exact"] + text[at + len(c["exact"]):]
    return text


# --------------------------------------------------------------------------
# Asking the writer
# --------------------------------------------------------------------------

def _schema(goal: str) -> str:
    """The reply shape, written with placeholders the writer cannot copy.

    No concrete example on purpose. The last version carried three -- a tabby
    on a worn oak chair in oil on canvas, a brick house on a wet street -- and
    a small writer handed a short request returned the nearest example
    almost word for word. Placeholders say the shape and nothing else.
    """
    fields = [
        '"unclear": false',
        '"subject": "<the main thing, in a few words>"',
        '"quoted_text": ["<each piece of text that must appear, exactly>"]',
        '"constraints": ["<each count, colour, position, exclusion or pace '
        'the request fixes, in its words>"]',
    ]
    if goal == "enhance":
        fields += [
            '"faithful": "<the request said clearly, adding nothing>"',
            '"prompt": "<the full description>"',
            '"additions": ["<a short phrase copied from prompt>"]',
            '"styles": ["<one medium and finish that would suit it>"]',
        ]
    elif goal == "style":
        fields += [
            '"prompt": "<the picture described, naming no style or medium>"',
            '"additions": ["<a short phrase copied from prompt>"]',
            '"styles": ["<style one>", "<style two>", "<style three>"]',
        ]
    else:
        fields += ['"prompt": "<the request said clearly, adding nothing>"',
                   '"additions": []']
    return "{" + ", ".join(fields) + "}"


def system_prompt(goal: str, mode: str, profile: str, *, looked: bool = False,
                  has_picture: bool = False) -> str:
    p = PROFILES.get(profile, PROFILES["generic"])
    lo, hi = p["words"]
    lines = [
        f"You write prompts for {p['label']}. A person has typed a rough "
        "request, often in a hurry: misspelled, ungrammatical, half a "
        "sentence. Work out what they mean and write the prompt.",
        "",
        "Always:",
        "- Fix spelling and grammar.",
        "- Keep the subject and every detail they gave: how many, which colour "
        "belongs to which thing, where things are, anything they said must not "
        "appear, and the pace of any motion.",
        "- Copy any quoted text exactly, with its capitals and punctuation, "
        "inside double quotes.",
        "- Never write quality words such as masterpiece, best quality, 8k or "
        "highly detailed, unless they are inside quoted text.",
        "- If the request names nothing that could be shown -- 'make it "
        "better', 'something nice' -- reply {\"unclear\": true} and nothing "
        "else.",
        "",
    ]
    if mode == "edit":
        lines += [
            "This is an instruction for an image editor, not a description of a "
            "new picture. Write it as an imperative: name the thing to change "
            "and where it is, say what it becomes, and say what must stay as "
            "it is. Never describe the rest of the picture beyond naming what "
            "to keep.",
        ]
        if looked:
            lines.append(
                "You are told what the picture contains. Use it only to say "
                "which thing is meant and where it is. When the request says "
                "'it' or 'this', you are told the most likely target.")
        else:
            lines.append(
                "You cannot see the picture" + (
                    "" if has_picture else ", and none was given") +
                ". Do not describe anything in it the request does not "
                "mention, and do not guess what it contains.")
        lines.append("")
    elif mode == "video":
        lines += [
            "This is a prompt for a short video clip. Say how the subject "
            "moves and what the camera does, or that it stays still. Keep the "
            "pace they asked for: if they say quickly or sprints, it is fast. "
            "If they give no pace, choose one that suits the action.",
        ]
        if looked:
            lines.append(
                "The clip starts from a picture whose contents you are told. "
                "Say how what is already there moves; add nothing it does not "
                "contain.")
        lines.append("")

    if goal == "tighten":
        lines += [
            "Goal: tighten. Fix clarity only. Add nothing -- no new objects, "
            "colours, light, camera or style. 'prompt' is the request said "
            "clearly, about as long as it was. 'additions' is empty.",
        ]
    elif goal == "style":
        lines += [
            f"Goal: style alternatives. 'prompt' describes the picture clearly "
            f"in {lo} to {hi} words and names no style or medium. 'styles' "
            "gives three different looks for it, each a short phrase naming a "
            "medium and its finish, as different from one another as you can "
            "make them while still suiting the subject.",
        ]
    else:
        lines += [
            "Goal: enhance. 'faithful' is the request said clearly, adding "
            f"nothing. 'prompt' expands it into a full description of {lo} to "
            f"{hi} words. Every detail you add must suit what they asked for; "
            "do not add people, animals, text or objects that change what the "
            "picture is about. Choose no art style or medium unless the "
            "request implies one -- that belongs in 'styles', which gives one "
            "that would suit it.",
        ]
    lines += [
        "'additions' lists what you added to 'prompt', each a short phrase "
        "copied word for word from it.",
        "",
        f"About {p['label']}: {p['guide']}",
        "",
        "Reply with one JSON object on one line and nothing else:",
        _schema(goal),
    ]
    return "\n".join(lines)


def user_message(request: str, locked: list[dict[str, Any]],
                 regions: list[dict[str, str]] | None = None,
                 target: dict[str, str] | None = None,
                 mode: str = "generate") -> str:
    out = [f"Request: {request}"]
    keep = [c["text"] for c in locked if c["hard"]]
    if keep:
        out.append("Must keep: " + "; ".join(keep))
    if regions:
        intro = ("The clip starts from a picture containing:" if mode == "video"
                 else "The picture contains:")
        out.append(intro)
        out += [f"- {describe_region(r)}" for r in regions]
    if target:
        out.append(f"'it' most likely means: {describe_region(target)}")
    return "\n".join(out)


def token_budget(goal: str, profile: str, request: str) -> int:
    """A ceiling on the writer's reply, sized to what was asked for.

    A 9B writer manages about fifteen tokens a second here, so the ceiling is
    the latency: room for the fields asked for and no more. A tighten is a
    request-sized answer; an enhance carries a faithful line, the full
    description, its additions and one style.
    """
    lo, hi = PROFILES.get(profile, PROFILES["generic"])["words"]
    asked = len(request.split())
    overhead = 90 + 2 * asked  # subject, quoted text, constraints, keys
    if goal == "tighten":
        return int(overhead + 1.6 * (asked + 20))
    if goal == "style":
        return int(overhead + 1.4 * hi + 100)
    return int(overhead + 1.6 * (asked + 20) + 1.4 * hi + 90)


# --------------------------------------------------------------------------
# Reading the writer's answer
# --------------------------------------------------------------------------

def strip_think(text: str) -> str:
    """Remove a reasoning block, closed or not.

    Thinking is turned off for every writer that has it, and the template
    pre-fills an empty block. This is the belt to that braces: a writer that
    reasons anyway must not have its reasoning read as the prompt. An
    unclosed block means the answer never started.
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.replace("</think>", "").strip()


def _close_truncated(s: str) -> str:
    """Close a JSON object the writer was cut off in the middle of."""
    stack: list[str] = []
    in_str = esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    out = s + ('"' if in_str else "")
    out = re.sub(r",\s*$", "", out)
    if re.search(r":\s*$", out):
        out += "null"
    return out + "".join(reversed(stack))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [" ".join(str(v).split()) for v in value
            if isinstance(v, (str, int, float)) and str(v).strip()]


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return re.sub(r"^(?:prompt|faithful)\s*:\s*", "", text, flags=re.I)


def _normalise(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "unclear": obj.get("unclear") is True,
        "subject": _text(obj.get("subject")),
        "quoted_text": _string_list(obj.get("quoted_text")),
        "constraints": _string_list(obj.get("constraints")),
        "faithful": _text(obj.get("faithful")),
        "prompt": _text(obj.get("prompt")),
        "additions": _string_list(obj.get("additions")),
        "styles": _string_list(obj.get("styles")),
    }


def parse_reply(raw: str) -> tuple[dict[str, Any] | None, str]:
    """The writer's answer as fields, and how it had to be read.

    In order: clean JSON; JSON with a trailing comma or cut off before its
    closing braces; the fields picked out one by one; and finally a writer
    that ignored the format and answered in prose, whose whole answer -- not
    its first line, which is how a second-line constraint used to disappear --
    is taken as the prompt.
    """
    text = strip_think(raw)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M).strip()
    if not text:
        return None, "empty"
    start = text.find("{")
    if start >= 0:
        end = text.rfind("}")
        tries = [text[start:end + 1]] if end > start else []
        tries.append(text[start:])
        for candidate in tries:
            for fixed in (candidate,
                          re.sub(r",\s*([}\]])", r"\1", candidate),
                          _close_truncated(re.sub(r",\s*([}\]])", r"\1", candidate))):
                try:
                    obj = json.loads(fixed)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    return _normalise(obj), "json" if fixed is candidate else "repaired"
        salvaged = _salvage(text[start:])
        if salvaged:
            return salvaged, "salvaged"
        return None, "broken"
    if text.upper().startswith("UNCLEAR"):
        return _normalise({"unclear": True}), "prose"
    return _normalise({"prompt": text}), "prose"


def _salvage(text: str) -> dict[str, Any] | None:
    found: dict[str, Any] = {}
    for key in ("subject", "faithful", "prompt"):
        m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)' % key, text)
        if m:
            try:
                found[key] = json.loads('"' + m.group(1) + '"')
            except ValueError:
                found[key] = m.group(1)
    for key in ("quoted_text", "constraints", "additions", "styles"):
        m = re.search(r'"%s"\s*:\s*\[(.*?)\]' % key, text, flags=re.S)
        if m:
            found[key] = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    if not (found.get("prompt") or found.get("faithful")):
        return None
    return _normalise(found)


def preview(raw: str) -> str:
    """What to show while the reply is still arriving.

    The reply is JSON, and a person watching braces and key names learns
    nothing. This shows the prose field being written -- the full prompt once
    it starts, the faithful line before that.
    """
    text = strip_think(raw)
    for key in ("prompt", "faithful"):
        m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)' % key, text)
        if m:
            return m.group(1).replace('\\"', '"')[-400:]
    return "" if "{" in text else text[-400:]


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------

def with_style(text: str, style: str) -> tuple[str, str]:
    """The prompt with a style sentence after it, and that sentence's phrase.

    Appended rather than woven in so it stays a separable addition: the chip
    that turns it off removes exactly this phrase and nothing else.
    """
    phrase = style.strip().rstrip(".")
    if not phrase:
        return text, ""
    phrase = phrase[0].upper() + phrase[1:]
    return f"{text.rstrip().rstrip('.')}. {phrase}.", phrase


def _additions_in(text: str, listed: list[str]) -> list[str]:
    """The listed additions that really are in the text, verbatim.

    Each becomes a chip that removes it, which only works if the phrase is
    there to remove. A phrase the writer paraphrased in its own list is
    dropped from the chips rather than offered as a switch that does nothing.
    """
    low = text.lower()
    out: list[str] = []
    for phrase in listed[:12]:
        phrase = phrase.strip(" ,.;")
        if not phrase or len(phrase.split()) > 14 or phrase.lower() not in low:
            continue
        if any(phrase.lower() in o.lower() or o.lower() in phrase.lower() for o in out):
            continue
        out.append(phrase)
    return out


def build_variants(parsed: dict[str, Any], goal: str, mode: str) -> list[dict[str, Any]]:
    """Faithful, enriched and stylistic versions from one answer.

    One call yields all of them: the faithful line and the full description
    are both fields of the reply, and a stylistic version is the description
    with one of its styles added. Asking three times would triple the wait
    for the same words.
    """
    faithful = parsed.get("faithful") or ""
    prompt = parsed.get("prompt") or ""
    styles = parsed.get("styles") or []
    out: list[dict[str, Any]] = []

    def add(kind: str, label: str, text: str, additions: list[str]) -> None:
        text = " ".join(text.split())
        if not text or any(v["prompt"].lower() == text.lower() for v in out):
            return
        out.append({"kind": kind, "label": label, "prompt": text,
                    "additions": additions, "missing": [], "reworded": []})

    if goal == "tighten":
        add("faithful", "Clearer", prompt or faithful, [])
        return out
    if goal == "style":
        base = prompt or faithful
        if not styles:
            add("faithful", "As described", base,
                _additions_in(base, parsed.get("additions") or []))
        for style in styles[:3]:
            text, phrase = with_style(base, style)
            label = " ".join(phrase.split()[:3])
            add("stylistic", label, text, [phrase] if phrase else [])
        return out

    add("faithful", "Faithful", faithful, [])
    enriched_adds = _additions_in(prompt, parsed.get("additions") or [])
    add("enriched", "Enriched", prompt, enriched_adds)
    if styles and prompt and mode != "edit":
        text, phrase = with_style(prompt, styles[0])
        add("stylistic", "Stylistic", text, enriched_adds + ([phrase] if phrase else []))
    if not out and prompt:
        add("enriched", "Enriched", prompt, enriched_adds)
    return out


def check_variants(variants: list[dict[str, Any]],
                   checks: list[dict[str, Any]]) -> None:
    """Mark each variant with the constraints it no longer satisfies."""
    for v in variants:
        v["prompt"] = _fix_quotes(v["prompt"], checks)
        missing = [c for c in checks if not satisfied(c, v["prompt"])]
        v["missing"] = [c["text"] for c in missing]
        v["hard_missing"] = [c["text"] for c in missing if c["hard"]]


def checks_for(locked: list[dict[str, Any]], parsed: dict[str, Any],
               request: str, target: dict[str, str] | None = None
               ) -> list[dict[str, Any]]:
    """Everything a variant must still say: ours, the writer's, the subject."""
    checks = list(locked)
    # Quoted text the writer reported that our pattern missed -- quotes typed
    # some other way -- counts too, as long as the request really contains it.
    for text in parsed.get("quoted_text") or []:
        if text and text in request and not any(
                c["kind"] == "quoted" and c["exact"] == text for c in checks):
            checks.append(_constraint("quoted", f'"{text}"', True, exact=text))
    checks += writer_constraints(parsed.get("constraints") or [], request, checks)
    subject = _content(parsed.get("subject") or "")
    asked = _content(request)
    # The subject is checked only when it came from the request. A subject
    # the writer made up is not something to hold every variant to.
    if subject and asked and any(near_word(s, a) for s in subject for a in asked):
        nouns = [w for w in subject if w not in COLOURS and w not in _MODIFIERS]
        checks.append(_constraint("subject", parsed["subject"], False,
                                  words=nouns or subject))
    if target:
        thing = _content(target.get("thing", ""))
        if thing:
            nouns = [w for w in thing if w not in COLOURS and w not in _MODIFIERS]
            checks.append(_constraint("target", target["thing"], False,
                                      words=nouns or thing))
    return checks


def _score(variants: list[dict[str, Any]]) -> int:
    return sum(3 * len(v.get("hard_missing", [])) + len(v["missing"])
               for v in variants)


def propose(write: Callable[[float], str], same: Callable[[str], bool],
            request: str, goal: str, mode: str, locked: list[dict[str, Any]],
            target: dict[str, str] | None = None,
            log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    """Ask the writer, check what it wrote, and decide what to offer.

    At most two writes and one judgement, where the old path could make four
    calls for one rewrite. The order:

    1. One write at the goal's temperature.
    2. Each variant is checked against every constraint. If only soft ones
       fail -- the writer's own list, the subject, a camera word -- the writer
       is asked once whether the rewrite still asks for the same picture,
       because those checks are the ones a good paraphrase can trip.
    3. A hard miss, or a DIFFERENT, is one more write at a lower temperature.
    4. Whatever is best is offered with what it is missing named, rather than
       thrown away: the person can see the loss and decide.
    """
    temperatures = (0.3, 0.15) if goal == "tighten" else (0.6, 0.25)
    best: dict[str, Any] | None = None
    judged = False
    calls = 0
    for temperature in temperatures:
        raw = write(temperature)
        calls += 1
        parsed, how = parse_reply(raw)
        if parsed is None:
            log(f"writer reply unusable ({how}); {raw[:120]!r}")
            continue
        if parsed["unclear"] and not (parsed["prompt"] or parsed["faithful"]):
            return {"unclear": True, "variants": [], "parsed": parsed,
                    "calls": calls, "how": how}
        variants = build_variants(parsed, goal, mode)
        if not variants:
            log(f"writer reply had no prompt ({how})")
            continue
        checks = checks_for(locked, parsed, request, target)
        check_variants(variants, checks)
        attempt = {"unclear": False, "variants": variants, "parsed": parsed,
                   "calls": calls, "how": how, "checks": checks}
        if any(v["missing"] for v in variants) and not judged and not any(
                v["hard_missing"] for v in variants):
            judged = True
            failing = next(v for v in variants if v["missing"])
            if same(failing["prompt"]):
                log("soft checks failed; the writer reads it as the same request")
                for v in variants:
                    v["reworded"], v["missing"] = v["missing"], []
        if best is None or _score(variants) < _score(best["variants"]):
            best = attempt
        if _score(variants) == 0:
            break
        log(f"variants miss {[v['missing'] for v in variants]}; "
            + ("trying once more" if temperature != temperatures[-1] else "offering the best"))
    if best is None:
        return {"unclear": False, "variants": [], "parsed": None, "calls": calls,
                "how": "empty"}
    best["calls"] = calls
    return best


def primary(variants: list[dict[str, Any]], goal: str) -> dict[str, Any] | None:
    """The variant shown first: the richest one that lost nothing."""
    if not variants:
        return None
    order = {"tighten": ("faithful",), "style": ("stylistic", "faithful"),
             "enhance": ("enriched", "stylistic", "faithful")}[goal]
    for kind in order:
        for v in variants:
            if v["kind"] == kind and not v["missing"]:
                return v
    return min(variants, key=lambda v: (len(v.get("hard_missing", [])),
                                        len(v["missing"])))


# --------------------------------------------------------------------------
# Reading the picture, for an edit
# --------------------------------------------------------------------------

# Asked of the picture reader. Positions and colours rather than a caption:
# "make the left one blue" and "remove the person on the left" can only be
# settled by knowing what is where, and the old four labelled lines --
# subject, surface, light, setting -- named one thing and placed nothing.
READER_SYSTEM = (
    "List what is in this picture, the main subject first. One line per "
    "thing, at most six lines, each written as:\n"
    "thing | where | colours\n"
    "where is left, centre or right, and foreground or background. colours "
    "are its one or two main colours. Name each thing plainly in two to five "
    "words. No sentences and nothing else."
)

_WHERE_WORDS = ("left", "centre", "center", "right", "middle", "foreground",
                "background", "top", "bottom")


def parse_regions(text: str) -> list[dict[str, str]]:
    """The reader's lines as things with a place and colours.

    Tolerant: a small vision model drops columns, numbers its lines, or falls
    back to the old labelled shape. Whatever has a thing in it is kept.
    """
    out: list[dict[str, str]] = []
    for line in strip_think(text).splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not line:
            continue
        if "|" in line:
            parts = [p.strip(" .") for p in line.split("|")]
        elif ":" in line:
            key, _, value = line.partition(":")
            if key.strip().upper() not in ("SUBJECT", "OBJECT", "THING"):
                continue
            parts = [value.strip(" .")]
        else:
            continue
        thing = " ".join(parts[0].split()[:6])
        if not thing or thing.lower() in ("thing", "unknown", "none"):
            continue
        where = parts[1].lower().replace("center", "centre") if len(parts) > 1 else ""
        colours = parts[2].lower() if len(parts) > 2 else ""
        out.append({"thing": thing, "where": where, "colours": colours})
        if len(out) == 6:
            break
    return out


def describe_region(r: dict[str, str]) -> str:
    detail = ", ".join(x for x in (r.get("where", ""), r.get("colours", "")) if x)
    return f"{r['thing']} ({detail})" if detail else r["thing"]


_PRONOUN_RE = re.compile(r"\b(it|this|that|them|these|those|him|her|he|she|they)\b",
                         re.IGNORECASE)


def resolve_target(request: str, regions: list[dict[str, str]]) -> dict[str, str] | None:
    """The region a request most plausibly means.

    A request that names something the picture contains means that thing,
    and when two match ("the person on the left"), the one in the place it
    names. A request that says only "it" means the most prominent thing:
    listed first, in the centre, in front. The old version resolved every
    pronoun to the whole subject sentence, so "make it blue" became "make a
    man in a jacket blue".
    """
    if not regions:
        return None
    asked = [w for w in _content(request) if w not in COLOURS]
    places = [w for w in _tokens(request) if w in _WHERE_WORDS]

    def named(r: dict[str, str]) -> bool:
        return any(same_word(a, t) for a in asked for t in _content(r["thing"]))

    matches = [r for r in regions if named(r)]
    if matches:
        if places and len(matches) > 1:
            placed = [r for r in matches if any(p in r["where"] for p in places)]
            if placed:
                return placed[0]
        return matches[0]
    if not _PRONOUN_RE.search(request):
        return None

    def prominence(item: tuple[int, dict[str, str]]) -> int:
        i, r = item
        return (3 if i == 0 else 0) + (2 if "centre" in r["where"] else 0) \
            + (1 if "foreground" in r["where"] else 0)

    return max(enumerate(regions), key=prominence)[1]
