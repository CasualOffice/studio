"""A narration script: the story as one storyteller would tell it aloud.

Much of what people write is not written to be read aloud -- chat-style
lines, emoji, abbreviations, texting spellings -- and Hinglish written in the
Latin alphabet cannot be pronounced by any local voice: an M4 test of one
sentence through Kokoro's Hindi voice, transcribed back by Whisper, came out
22% wrong in Latin script ("main" as में, "ki" as काई, "peete hain" as
"पी थेन") and 7% wrong once the same words were written in Devanagari.

Most of that needs no language model. `normalize` and `segments` prepare any
text for the ear by rule, instantly and the same way every time: chat lines
are read as the name, a pause, then the message; texting shorthand is said in
full; emoji are dropped. Hindi voices then get their Hindi words spelled in
Devanagari by hinglish.py. A first listen is therefore never behind a model.

The story model is an opt-in on top: `suggest` asks it to retell a paragraph
at a time and keeps only a retelling that still uses the author's words, for
the person to accept or ignore as a diff. The chapter itself is never changed.
"""
from __future__ import annotations

import re

import hinglish
import story


SCRIPT_SYSTEM = """You prepare a story for one storyteller to read aloud, the way an audiobook
narrator tells it. Keep the author's own words: every Hindi word stays that
Hindi word and every English word stays that English word -- never translate,
never swap a word for a synonym, never change tum/aap/tu or usne/unhone.
Keep every event and every line of dialogue, in order. Change only what a
listener cannot follow: a chat line "Name: text" becomes "Name ne likha, ..."
(or "kaha" if spoken); spell out abbreviations and texting spellings (msg as
message, u as you, 5 min as paanch minute); describe an emoji's feeling in a few
words only if it matters ("gusse mein"), otherwise drop it. Do not add events,
opinions or a summary. Return only the narration text, as paragraphs.
"""


class Stop(Exception):
    """Raised by a writer or progress callback: keep what is written so far."""


def _words(text):
    return re.findall(r"\w+", text.casefold())


def _ok(text, source):
    """Whether a reply is a plausible retelling of the passage.

    Length alone let a fluent paraphrase through: the story model swapped
    words for synonyms and translated Hinglish while staying the same size.
    Most of the author's words must survive, and most of the reply's words
    must be the author's."""
    if not text.strip() or not 0.4 <= len(text) / max(1, len(source)) <= 3.0:
        return False
    told, original = _words(text), _words(source)
    if not original:
        return True
    kept = set(told)
    known = set(original)
    return (sum(w in kept for w in original) / len(original) >= 0.6
            and sum(w in known for w in told) / max(1, len(told)) >= 0.5)


def _clean(text):
    text = re.sub(r"^```\w*\s*|\s*```$", "", text.strip())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def suggest(source, write, progress=lambda *_: None, words=350):
    """The story model's retelling of each paragraph, as suggestions.

    Each suggestion names the exact source text it would replace, so the
    person can accept it as a diff; a paragraph the writer leaves unchanged,
    or cannot retell acceptably after one retry, has none. Stop ends early
    and keeps the suggestions made so far. Returns (suggestions, stopped)."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("there is no text to prepare")
    groups = [g for paragraph in re.split(r"\n\s*\n", source) if paragraph.strip()
              for g in story.passages(paragraph, max_words=words, max_units=40)]
    found, stopped = [], False
    for i, group in enumerate(groups):
        text = "".join(u["text"] for u in group).strip()
        if len(_words(text)) < 3:
            continue
        try:
            progress(i, len(groups))
            for attempt in range(2):
                reply = _clean(write(SCRIPT_SYSTEM, ("Passage:\n" + text) if not attempt else
                                     ("Passage:\n" + text + "\nYour previous version did not follow the rules: "
                                      "keep every event and the author's words, do not translate.")))
                if _ok(reply, text):
                    if " ".join(reply.split()) != " ".join(text.split()):
                        found.append({"index": i, "original": text, "suggestion": reply})
                    break
        except Stop:
            stopped = True
            break
    return found, stopped


def apply(source, suggestions):
    """The source with the given suggestions accepted."""
    for s in suggestions:
        source = source.replace(s["original"], s["suggestion"], 1)
    return source


def script(source, write, devanagari=False, progress=lambda *_: None, words=350):
    """The chapter with every acceptable suggestion applied; optionally with
    its Hindi words spelled in Devanagari by hinglish.devanagari -- a word
    list, not the model, which on 4B translated and invented words when asked
    to transliterate. Returns (script, stopped)."""
    found, stopped = suggest(source, write, progress, words)
    parts = [p.strip() for p in re.split(r"\n\s*\n", apply(source, found)) if p.strip()]
    if devanagari:
        parts = [hinglish.devanagari(p) for p in parts]
    return "\n\n".join(parts), stopped


# Emoji and pictographs: a voice reads them by name ("annoyed face"), which
# no storyteller would say.
EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\uFE0F\u200D]+")


def speakable(text):
    """Text as a voice should receive it: no emoji, no stray symbols."""
    text = EMOJI.sub("", text)
    text = re.sub(r"[*_#~`|<>\[\]{}]+", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text)


# Texting shorthand, said in full. Only whole lowercase words: "U" and "K"
# are initials as often as not.
SHORTHAND = {
    "msg": "message", "msgs": "messages", "u": "you", "ur": "your", "r": "are", "pls": "please",
    "plz": "please", "thx": "thanks", "tnx": "thanks", "ty": "thank you", "btw": "by the way",
    "idk": "I don't know", "omg": "oh my god", "b4": "before", "gr8": "great", "tmrw": "tomorrow",
    "tmr": "tomorrow", "k": "okay", "kk": "okay", "brb": "be right back", "asap": "as soon as possible",
    "gn": "good night", "gm": "good morning", "bday": "birthday", "pic": "picture", "pics": "pictures",
    "ppl": "people", "coz": "because", "cuz": "because", "bcoz": "because", "bcz": "because",
    "abt": "about", "wht": "what", "wat": "what", "rn": "right now", "nvm": "never mind",
    "lol": "", "lmao": "", "convo": "conversation",
}
UNITS = {"min": "minute", "mins": "minute", "hr": "hour", "hrs": "hour", "sec": "second",
         "secs": "second", "km": "kilometre", "kms": "kilometre", "kg": "kilo", "kgs": "kilo"}


def normalize(text):
    """Text as a storyteller would say it, by rule rather than by model.

    Shorthand is said in full, "5 min" as "5 minutes", stretched words
    ("sooo", "hmmm") and stacked punctuation ("??!!") are evened out, and
    emoji and markup are dropped. Hindi words are left for hinglish.py, which
    spells them for the voice that reads them."""
    text = speakable(text)
    text = re.sub(r"\b(\d+)\s*(" + "|".join(UNITS) + r")\b",
                  lambda m: f"{m[1]} {UNITS[m[2]]}{'' if m[1] == '1' else 's'}", text)
    text = re.sub(r"(?<![\w'])[a-z][a-z0-9]*(?![\w'])", lambda m: SHORTHAND.get(m[0], m[0]), text)
    text = re.sub(r"([A-Za-z])\1{2,}", r"\1", text)
    text = re.sub(r"([!?])[!?]+", r"\1", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r" +([,.!?])", r"\1", text)
    # What a dropped word ("lol") or emoji leaves behind: "hour, ." or ", ,".
    text = re.sub(r"[,;:]+(\s*[.!?])", r"\1", text)
    text = re.sub(r",(\s*,)+", ",", text)
    return re.sub(r"[ \t]{2,}", " ", text)


# A line of a chat transcript, optionally time-stamped: "[10:42] Arjun: hi".
CHAT = re.compile(r"^\s*(?:\[[^\]\n]{1,20}\]\s*|\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?\s*[-–]?\s*)?"
                  r"([^\W\d][\w .'-]{0,30}?)\s*:\s+(\S.*)$")
# A scene break: "***", "* * *", "---", "~~~", "#".
SCENE_BREAK = re.compile(r"^\s*(?:(?:\*\s*){3,}|-{3,}|~{3,}|#|§)\s*$")
HEADING = re.compile(r"^\s*(?:#{1,6}\s+\S.*|(?:chapter|part|prologue|epilogue|interlude)\b.{0,60}"
                     r"|[IVXLC]+\.?|\d{1,3}\.?)\s*$", re.IGNORECASE)
# Pauses a storyteller leaves, in seconds, before a sentence: after another
# sentence, after a speaker's name in a chat line, after a paragraph, and at
# a heading or scene break. Butted together, clips ran one sentence into the
# next with no breath between them.
PAUSES = {"sentence": 0.12, "speaker": 0.3, "paragraph": 0.5, "scene": 0.9}
# Without headings, a marker every this many words (two minutes or so), so a
# long chapter still has places to jump to.
MARKER_WORDS = 300


def _heading(block):
    line = block.strip()
    return "\n" not in line and len(line) <= 80 and bool(HEADING.match(line))


def _opening(text, words=6):
    tokens = text.split()
    return " ".join(tokens[:words]) + ("…" if len(tokens) > words else "")


def _sentences(text):
    """Sentence texts, stripped, keeping quoted dialogue whole.

    A unit that starts in lower case continues the one before it: the story
    model's units cut '"Jonas?" she called.' at the quote, and read as two
    clips the tag came out as a separate, flat little sentence."""
    out = []
    for u in story.units(text):
        piece = u["text"].strip()
        if not any(c.isalnum() for c in piece):
            continue
        if out and piece[0].islower():
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def segments(script_text):
    """Text as narration segments: one storyteller, a sentence at a time.

    A sentence is the unit of streaming, of the transcript and of re-reading,
    so each segment is one sentence and says which pause comes before it and
    which marker it falls under. Headings become markers by their own words;
    a text without them gets a marker every few hundred words, named by the
    paragraph it starts at, instead of "Passage 1", "Passage 2"."""
    result, pause, label, since, paragraph = [], "scene", None, 0, 0
    for block in re.split(r"\n\s*\n", script_text):
        if not block.strip():
            continue
        if SCENE_BREAK.match(block):
            pause, since = "scene", MARKER_WORDS
            continue
        if _heading(block):
            heading = normalize(block).strip()
            if any(c.isalnum() for c in heading):
                label, since = heading, 0
                result.append({"speaker_id": None, "text": heading, "mood": "", "scene": label,
                               "pause": "scene", "paragraph": paragraph})
                paragraph, pause = paragraph + 1, "paragraph"
            continue
        text = normalize(block)
        if since >= MARKER_WORDS or label is None:
            label, since = _opening(text), 0
        since += len(text.split())
        for n, line in enumerate(block.strip("\n").split("\n")):
            # Matched before normalizing, which drops a time stamp's brackets.
            chat = CHAT.match(line)
            if chat:
                # The name, a breath, then the message: "Arjun: kahan ho" read
                # straight through sounds like one sentence about Arjun.
                result.append({"speaker_id": None, "text": normalize(chat[1]).strip() + ".", "mood": "",
                               "scene": label, "pause": "paragraph" if n else pause, "paragraph": paragraph})
                pause, line = "speaker", chat[2]
            for sentence in _sentences(normalize(line)):
                result.append({"speaker_id": None, "text": sentence, "mood": "", "scene": label,
                               "pause": pause, "paragraph": paragraph})
                pause = "sentence"
        paragraph, pause = paragraph + 1, "paragraph"
    return result


def refine(segments):
    """Character-voiced segments split into sentences, keeping who speaks.

    The analysed chapter is cut at quotes, so one narration segment can be a
    paragraph or three, and its scenes are titled "Passage 1", "Passage 2" by
    the quick voice analysis. Sentences are what stream, show in the
    transcript and can be re-read on their own; a passage number is replaced
    by the words the passage opens with."""
    result, scene, names, paragraph = [], object(), {}, 0
    for segment in segments:
        text = normalize(segment["text"])
        raw, new_scene = segment.get("scene"), segment.get("scene") != scene
        scene = raw
        if raw not in names and any(c.isalnum() for c in text):
            names[raw] = _opening(text) if re.fullmatch(r"Passage \d+", raw or "") else raw
        for b, block in enumerate(re.split(r"\n\s*\n", text)):
            paragraph += bool(b)
            for n, sentence in enumerate(_sentences(block)):
                pause = "scene" if new_scene else "paragraph" if b and not n else "sentence"
                result.append({**segment, "text": sentence, "scene": names.get(raw, raw),
                               "pause": pause, "paragraph": paragraph})
                new_scene = False
    return result
