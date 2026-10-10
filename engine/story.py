"""Shared, versioned story analysis. No model or platform dependencies.

Offsets are Python Unicode code-point offsets into chapter.source. Renderers
use the stored text, not JavaScript string slicing. Model replies reference
numbered source units; the model never retypes the author's narration.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re
import uuid

SCENE_SYSTEM = """You are a story editor and director. Return only the requested JSON.
Read this scene for causality, dramatic turns and emotion, not equal coverage.
Use the supplied numbered source units, each exactly once, in order. Group
adjacent units into beats. Importance is 1 (context), 2 (quiet action), 3
(meaningful change), 4 (turn), 5 (climax/reveal). Preserve the ending.
Every visible.character_id and dialogue.speaker_id must EXACTLY match an id in
the character bible or in this reply's characters array. For a new character,
declare a unique short local id (c1, c2, c3), then reuse that exact ID in beats.
Never use an undeclared ID or the schema placeholder as a character reference.
Canonical characters have stable IDs. Reuse them for aliases; never create a
second person for a narrator already named. New characters must have a name
or noun phrase literally in this scene; never invent appearance or aliases.
Keep a speaking character offscreen when only heard. Attribute every quoted
utterance; use null speaker_id when uncertain rather than guessing. Dialogue
text must be an exact substring of the source. Do not paraphrase it.
Beats describe visible action, location, tension, mood, and each visible
character's expression and body language. Do not invent events. Distinguish
what happens from how the frame conveys its emotional meaning.
Character visual describes lasting identity only (face, hair, build, distinguishing
features), never clothing, accessories, injury or wetness. Those belong in each
visible actor's appearance: the COMPLETE current outfit and temporary condition,
not a delta. Carry it forward until the source changes it; a changed dress REPLACES
the old dress, never layers over it. Do not invent unspecified details.
location_id is a stable descriptive key for the actual room/place; reuse on return.
location_state describes current layout, important objects, lighting and weather.
Carry unchanged facts forward, but replace facts changed by the story.
"""


def units(source):
    """Sentence units, keeping multi-sentence quoted dialogue intact."""
    result, start, quote = [], 0, None
    pairs = {'“': '”', '「': '」', '"': '"'}
    for i, ch in enumerate(source):
        closed_quote = False
        if quote:
            if ch == quote and (i == 0 or source[i - 1] != "\\"):
                quote = None
                closed_quote = True
        elif ch in pairs:
            quote = pairs[ch]
        if not quote and (((ch in ".!?。！？" or (closed_quote and i > 0 and source[i - 1] in ".!?。！？")) and (i + 1 == len(source) or source[i + 1].isspace()))
                          or (ch == "\n" and (i + 1 == len(source) or source[i + 1] != "\n"))):
            if source[start:i + 1].strip():
                result.append({"id": len(result), "start": start, "end": i + 1, "text": source[start:i + 1]})
                start = i + 1
    if start < len(source):
        if not source[start:].strip() and result:
            result[-1]["end"] = len(source)
            result[-1]["text"] += source[start:]
        else:
            result.append({"id": len(result), "start": start, "end": len(source), "text": source[start:]})
    return result


def passages(source, max_words=450, max_units=8):
    """Bound calls at source-unit boundaries; explicit scene breaks end a call."""
    groups, group, words = [], [], 0
    for unit in units(source):
        if len(unit["text"]) > 16000:
            raise ValueError("A single source passage is over 16,000 characters; split it at paragraph boundaries before analysis")
        n = len(unit["text"].split())
        if group and (len(group) >= max_units or words + n > max_words
                      or re.search(r'\n\s*(?:\*\s*\*\s*\*|---+)\s*\n', unit["text"])):
            groups.append(group)
            group, words = [], 0
        # A single enormous sentence still needs bounded input.
        if n > max_words and not re.search(r'[“"「]', unit["text"]):
            if group:
                groups.append(group)
                group, words = [], 0
            pieces = list(re.finditer(r'\S+\s*', unit["text"]))
            start = unit["start"]
            for at in range(0, len(pieces), max_words):
                end = unit["start"] + pieces[min(at + max_words, len(pieces)) - 1].end()
                groups.append([{"start": start, "end": end, "text": source[start:end]}])
                start = end
            continue
        group.append(unit)
        words += n
        if "\n\n" in unit["text"]:
            groups.append(group)
            group, words = [], 0
    if group:
        groups.append(group)
    for group in groups:
        for i, unit in enumerate(group):
            unit["id"] = i
    return groups


def instruction(group, characters, continuity=None):
    schema = {
        "title": "scene title", "summary": "causal/emotional meaning",
        "characters": [{"id": "c1", "name": "literal source name",
                        "aliases": [], "visual": "lasting face/hair/build only, no outfit",
                        "voice": "voice direction"}],
        "beats": [{"units": [0], "importance": 3, "action": "visible action",
                   "setting": "location", "location_id": "stable room/place key",
                   "location_state": "current layout, objects, light and weather", "mood": "emotional tone", "tension": 3,
                   "visible": [{"character_id": "c1", "expression": "expression",
                                "body_language": "pose", "appearance": "complete current outfit and temporary condition"}],
                   "dialogue": [{"speaker_id": None, "text": "exact quoted words",
                                 "offscreen": True}]}],
    }
    return ("For unnamed first-person narration only, canonical name Narrator is allowed. "
            "Never use Narrator when a known character already narrates. "
            "Do not group multiple people by shared third-person pronouns.\nCharacter bible:\n" + json.dumps(characters, ensure_ascii=False)
            + "\nPrior continuity state (apply only unchanged facts):\n" + json.dumps(continuity or {}, ensure_ascii=False)
            + "\nReturn this schema with one beat per meaningful moment:\n"
            + json.dumps(schema, ensure_ascii=False)
            + "\nNumbered source units:\n" + json.dumps(group, ensure_ascii=False))


def _text(value, field, empty=True):
    if not isinstance(value, str) or len(value) > 4000 or (not empty and not value.strip()):
        raise ValueError(f"{field} must be a {'nonempty ' if not empty else ''}string")
    return value.strip()


def _array(value, field):
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def parse_scene(raw, group, bible, scene_id):
    """Strict JSON and source/identity validation. No partial JSON salvage."""
    raw = raw.strip()
    if raw.startswith('```json') and raw.endswith('```'):
        raw = raw[7:-3].strip()
    reply = json.loads(raw)
    if not isinstance(reply, dict):
        raise ValueError("scene must be an object")
    title = _text(reply.get("title"), "title", False)
    summary = _text(reply.get("summary"), "summary", False)
    characters = copy.deepcopy(bible)
    lookup = {c["id"]: c["id"] for c in characters}
    source = ''.join(u["text"] for u in group)
    local_ids = set()
    narrative = re.sub(r'[“"]([^”"]+)[”"]|「([^」]+)」', "", source)
    first_person = bool(re.search(r'\b(?:I|my|me)\b', narrative, re.I))
    narrator_roles = set()
    proposed = _array(reply.get("characters"), "characters")
    # Read named participants first so a later narration-role declaration can
    # bind to a named first-person speaker regardless of JSON array order.
    proposed = sorted(proposed, key=lambda person: str(person.get("name", "")).strip().casefold() == "narrator")
    for person in proposed:
        name = _text(person.get("name"), "name", False)
        if name.casefold() == "narrator":
            name = "Narrator"
        local_id = _text(person.get("id"), "character id", False)
        if local_id in local_ids:
            raise ValueError("each local character ID must be unique")
        local_ids.add(local_id)
        aliases = [_text(a, "alias", False) for a in _array(person.get("aliases", []), "aliases")]
        aliases = [a for a in aliases if a.casefold() not in {"he", "she", "they", "him", "her", "his", "their", "we", "you"}]
        literal_name = bool(re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', source, re.I))
        if name == "Narrator" and not first_person and not literal_name:
            # Narration in a third-person story is a voice role, not a person.
            # It does not belong in the character bible or visible cast.
            narrator_roles.update({local_id, "narrator"})
            lookup[local_id] = None
            continue
        introduced = literal_name and bool(re.search(
            r"\b(?:my name is|i am|i['’]m)\s+" + re.escape(name) + r'(?!\w)', narrative, re.I))
        if introduced:
            aliases += [word for word in ["I", "my", "me"] if re.search(r'\b' + word + r'\b', narrative, re.I)]
        identity = {name.casefold(), *(a.casefold() for a in aliases)}
        matches = [c for c in characters if c["id"] == local_id or identity.intersection(
            {c["name"].casefold(), *(a.casefold() for a in c["aliases"])})]
        if len(matches) > 1:
            raise ValueError(f"ambiguous character identity: {name}")
        existing = matches[0] if matches else None
        # Only literal aliases can be added to the persistent bible.
        aliases = [a for a in aliases if re.search(r'(?<!\w)' + re.escape(a) + r'(?!\w)', source, re.I)]
        anonymous_narrator = name == "Narrator" and first_person
        if anonymous_narrator:
            narrators = [c for c in characters if c["name"].casefold() == "narrator"
                         or {a.casefold() for a in c["aliases"]}.intersection({"i", "my", "me"})]
            if len(narrators) > 1:
                raise ValueError("the first-person narrator is ambiguous; use its exact character ID")
            if narrators:
                existing = narrators[0]
            aliases += [word for word in ["I", "my", "me"] if re.search(r'\b' + word + r'\b', narrative, re.I)]
        if not existing and not anonymous_narrator and not re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', source, re.I):
            raise ValueError(f"character {name} is not in this scene")
        if name.casefold() in {"i", "me", "my", "he", "she", "they", "him", "her", "we", "you"}:
            raise ValueError("use a canonical name, not a pronoun")
        visual = _text(person.get("visual", ""), "visual")
        voice = _text(person.get("voice", ""), "voice")
        if existing:
            if existing["name"].casefold() == "narrator" and name != "Narrator" and literal_name:
                existing["aliases"] = list(dict.fromkeys([*existing["aliases"], existing["name"]]))
                existing["name"] = name
            existing["aliases"] = list(dict.fromkeys([*existing["aliases"], *aliases, name]))
            # First chapter's identity stays stable; later facts are additive.
            if visual and visual not in existing["visual"]:
                existing["visual"] = ' '.join(filter(None, [existing["visual"], visual]))
            if not existing["voice"]:
                existing["voice"] = voice
        else:
            existing = {"id": str(uuid.uuid4()), "name": name, "aliases": aliases,
                        "visual": visual, "voice": voice}
            characters.append(existing)
        lookup[local_id] = existing["id"]
    # The model may use a canonical name or alias instead of the requested ID.
    # Resolve only exact, unique bindings to validated characters; no fuzzy
    # matching or character invention. Generated UUIDs also remain valid.
    names = {}
    for character in characters:
        lookup.setdefault(character["id"], character["id"])
        for name in [character["name"], *character["aliases"]]:
            names.setdefault(name.strip().casefold(), set()).add(character["id"])

    def resolve(reference, field):
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"{field} must name a declared character ID")
        reference = reference.strip()
        if reference in lookup:
            return lookup[reference]
        if reference.casefold() in narrator_roles:
            return None
        matches = names.get(reference.casefold(), set())
        if len(matches) == 1:
            return next(iter(matches))
        if len(matches) > 1:
            raise ValueError(f"{field} is ambiguous: {reference!r}; use its exact character ID")
        valid = ', '.join(f"{local}={next((c['name'] for c in characters if c['id'] == cid), 'narration voice')}"
                          for local, cid in lookup.items())
        raise ValueError(f"{field} {reference!r} is not declared. Valid IDs: {valid or '(none)'}. "
                         "Declare new people in characters before referencing them in beats.")

    beats, consumed = [], []
    for i, beat in enumerate(_array(reply.get("beats"), "beats")):
        ids = _array(beat.get("units"), "units")
        if not ids or any(type(n) is not int or n < 0 or n >= len(group) for n in ids):
            raise ValueError("beat units must be valid source-unit numbers")
        consumed.extend(ids)
        start, end = group[ids[0]]["start"], group[ids[-1]]["end"]
        excerpt = ''.join(group[n]["text"] for n in ids)
        importance, tension = beat.get("importance"), beat.get("tension")
        if type(importance) is not int or not 1 <= importance <= 5:
            raise ValueError("importance must be 1..5")
        if type(tension) is not int or not 1 <= tension <= 5:
            raise ValueError("tension must be 1..5")
        visible = []
        for actor in _array(beat.get("visible"), "visible"):
            cid = resolve(actor.get("character_id"), "visible character")
            if cid is None:
                continue
            if cid in [a["character_id"] for a in visible]:
                raise ValueError("duplicate visible character")
            visible.append({"character_id": cid,
                            "expression": _text(actor.get("expression"), "expression"),
                            "body_language": _text(actor.get("body_language"), "body language"),
                            "appearance": _text(actor.get("appearance", ""), "appearance")})
        # Quoted speech is source data, not something the analyst must copy.
        # The analyst supplies optional attribution only. Missing, paraphrased
        # or invented lines never replace/drop the original quotation.
        proposed_lines = {}
        for line in _array(beat.get("dialogue") or [], "dialogue"):
            if not isinstance(line, dict) or not isinstance(line.get("text"), str):
                continue
            text = line["text"].strip().strip('“”"「」').strip()
            if text:
                proposed_lines.setdefault(text, []).append(line)
        dialogue = []
        for quoted in re.finditer(r'[“"]([^”"]+)[”"]|「([^」]+)」', excerpt):
            group_no = 1 if quoted.group(1) is not None else 2
            original = quoted.group(group_no)
            text = original.strip()
            if not text:
                continue
            at = quoted.start(group_no) + len(original) - len(original.lstrip())
            candidates = proposed_lines.get(text, [])
            line = candidates.pop(0) if candidates else None
            cid, offscreen, review = None, True, line is None
            if line is not None:
                try:
                    cid = resolve(line.get("speaker_id"), "dialogue speaker") if line.get("speaker_id") is not None else None
                except ValueError:
                    review = True
                if type(line.get("offscreen")) is bool:
                    offscreen = line["offscreen"]
                else:
                    review = True
                review = review or cid is None
            dialogue.append({"speaker_id": cid, "text": text, "offscreen": offscreen,
                             "start": start + at, "end": start + at + len(text),
                             "attribution_review": review})
        beats.append({"id": f"{scene_id}-b{i + 1}", "start": start, "end": end,
                      "source": excerpt, "importance": importance, "tension": tension,
                      "action": _text(beat.get("action"), "action", False),
                      "setting": _text(beat.get("setting"), "setting"),
                      "location_id": _text(beat.get("location_id", ""), "location id"),
                      "location_state": _text(beat.get("location_state", ""), "location state"),
                      "mood": _text(beat.get("mood"), "mood"),
                      "visible": visible, "dialogue": dialogue})
    if consumed != list(range(len(group))):
        raise ValueError("beats must cover every source unit exactly once, in order")
    return {"id": scene_id, "title": title, "summary": summary,
            "start": group[0]["start"], "end": group[-1]["end"], "beats": beats}, characters


def _remember_state(state, beat):
    for actor in beat["visible"]:
        if actor.get("appearance"):
            state["characters"][actor["character_id"]] = actor["appearance"]
    key = beat.get("location_id") or beat["setting"].casefold()
    if beat.get("location_state"):
        state["locations"][key] = beat["location_state"]


def analyze(source, write, previous=None, title="Chapter", progress=lambda *_: None,
            passage_words=450, passage_units=8):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("paste a chapter first")
    if len(source) > 200000:
        raise ValueError("a chapter must be under 200,000 characters; add chapters separately")
    book = copy.deepcopy(previous) if previous else {
        "version": 1, "id": str(uuid.uuid4()), "characters": [], "chapters": []}
    validate_book(book)
    chapter_id = str(uuid.uuid4())
    groups = passages(source, max_words=passage_words, max_units=passage_units)
    scenes = []
    continuity = {"characters": {}, "locations": {}}
    for old_chapter in book["chapters"]:
        for old_scene in old_chapter["scenes"]:
            for old_beat in old_scene["beats"]:
                _remember_state(continuity, old_beat)
    for i, group in enumerate(groups):
        progress(i, len(groups))
        prompt = instruction(group, book["characters"], continuity)
        for attempt in range(2):
            raw = write(SCENE_SYSTEM, prompt)
            try:
                scene, characters = parse_scene(raw, group, book["characters"], f"{chapter_id}-s{i + 1}")
                break
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                if attempt:
                    raise ValueError(f"Scene {i + 1} could not be validated: {exc}") from exc
                prompt += (f"\nYour previous reply failed validation: {exc}."
                           "\nPrevious reply to repair:\n" + raw[:20000]
                           + "\nReturn corrected complete JSON, preserving the full source coverage. "
                           "Every visible character and speaker must refer to an ID declared in "
                           "the character bible or your characters array. Use null for uncertain speakers.")
        scenes.append(scene)
        for beat in scene["beats"]:
            _remember_state(continuity, beat)
        book["characters"] = characters
    chapter = {"id": chapter_id, "title": title.strip() or "Chapter",
               "source": source, "source_hash": hashlib.sha256(source.encode()).hexdigest(),
               "scenes": scenes}
    book["chapters"].append(chapter)
    validate_book(book)
    return book


def validate_book(book):
    if not isinstance(book, dict) or book.get("version") != 1:
        raise ValueError("unsupported story model version")
    _text(book.get("id"), "book id", False)
    characters = _array(book.get("characters"), "characters")
    known = set()
    for c in characters:
        cid = _text(c.get("id"), "character id", False)
        if cid in known:
            raise ValueError("duplicate character id")
        known.add(cid)
        _text(c.get("name"), "name", False)
        _array(c.get("aliases"), "aliases")
        _text(c.get("visual"), "visual")
        _text(c.get("voice"), "voice")
    for chapter in _array(book.get("chapters"), "chapters"):
        source = chapter.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("chapter source is missing")
        if chapter.get("source_hash") != hashlib.sha256(source.encode()).hexdigest():
            raise ValueError("chapter source changed after analysis")
        cursor = 0
        for scene in _array(chapter.get("scenes"), "scenes"):
            for beat in _array(scene.get("beats"), "beats"):
                start, end = beat.get("start"), beat.get("end")
                if type(start) is not int or type(end) is not int or start != cursor or not start < end <= len(source):
                    raise ValueError("beat spans must partition the chapter")
                if beat.get("source") != source[start:end]:
                    raise ValueError("beat source is not exact")
                cursor = end
                last = start
                for line in _array(beat.get("dialogue"), "dialogue"):
                    a, b = line.get("start"), line.get("end")
                    if type(a) is not int or type(b) is not int or not last <= a < b <= end or source[a:b] != line.get("text"):
                        raise ValueError("dialogue spans must match source and not overlap")
                    last = b
                    if line.get("speaker_id") is not None and line["speaker_id"] not in known:
                        raise ValueError("unknown speaker")
                for actor in _array(beat.get("visible"), "visible"):
                    if actor.get("character_id") not in known:
                        raise ValueError("unknown visible character")
                if type(beat.get("importance")) is not int or not 1 <= beat["importance"] <= 5:
                    raise ValueError("invalid beat importance")
        if cursor != len(source):
            raise ValueError("chapter analysis is incomplete")
    return book


def narration_segments(book, chapter_id):
    validate_book(book)
    chapter = next((c for c in book["chapters"] if c["id"] == chapter_id), None)
    if not chapter:
        raise ValueError("choose an analyzed chapter")
    source, cursor, segments = chapter["source"], 0, []
    for scene in chapter["scenes"]:
        for beat in scene["beats"]:
            for line in beat["dialogue"]:
                if line["start"] > cursor:
                    segments.append({"speaker_id": None, "text": source[cursor:line["start"]]})
                segments.append({"speaker_id": line["speaker_id"], "text": line["text"], "mood": beat["mood"]})
                cursor = line["end"]
    if cursor < len(source):
        segments.append({"speaker_id": None, "text": source[cursor:]})
    return segments


def _chunks(text, words=36):
    tokens = list(re.finditer(r'\S+\s*', text))
    return [text[tokens[i].start():tokens[min(i + words, len(tokens)) - 1].end()].strip()
            for i in range(0, len(tokens), words)]


def direct(book, chapter_id):
    """Deterministic director: more frames at turns, never rewrite lettering."""
    validate_book(book)
    chapter = next((c for c in book["chapters"] if c["id"] == chapter_id), None)
    if not chapter:
        raise ValueError("choose an analyzed chapter")
    people = {c["id"]: c for c in book["characters"]}
    panels = []
    for scene_no, scene in enumerate(chapter["scenes"], 1):
        for beat_no, beat in enumerate(scene["beats"]):
            narration, cursor = [], 0
            for line in beat["dialogue"]:
                a, b = line["start"] - beat["start"], line["end"] - beat["start"]
                narration.append(beat["source"][cursor:a])
                cursor = b
            narration.append(beat["source"][cursor:])
            narrative = re.sub(r'[“"]\s*[”"]|「\s*」', '', ''.join(narration))
            captions = _chunks(narrative)
            dialogue = [{"speaker": people[d["speaker_id"]]["name"] if d["speaker_id"] else "Unknown speaker",
                         "speaker_id": d["speaker_id"], "text": text, "offscreen": d["offscreen"],
                         "attribution_review": d.get("attribution_review", False)}
                        for d in beat["dialogue"] for text in _chunks(d["text"], 28)]
            frames = max(len(captions), len(dialogue), 3 if beat["importance"] == 5 else 2 if beat["importance"] == 4 else 1)
            actors = [{**a, "name": people[a["character_id"]]["name"],
                       "visual": people[a["character_id"]]["visual"]} for a in beat["visible"]]
            for frame in range(frames):
                reaction = frame > 0 and frame == frames - 1 and bool(actors) and beat["importance"] >= 4
                panels.append({"beat_id": beat["id"], "chapter_id": chapter_id,
                    "shot": "close-up" if reaction else "wide" if beat_no == 0 and frame == 0 else "medium",
                    "purpose": "reaction" if reaction else "establishing" if beat_no == 0 else "action",
                    "subject": ', '.join(a["name"] for a in actors), "action": beat["action"],
                    "setting": beat["setting"], "character_in_frame": bool(actors),
                    "characters": [a["name"] for a in actors], "cast": actors,
                    "mood": beat["mood"], "tension": beat["tension"], "importance": beat["importance"],
                    "source": beat["source"] if frame == 0 else "",
                    "caption": captions[frame] if frame < len(captions) else "",
                    "dialogue": [dialogue[frame]] if frame < len(dialogue) else [],
                    "scene": scene_no, "scene_title": scene["title"],
                    "place": beat["setting"], "place_key": beat.get("location_id") or beat["setting"].casefold(),
                    "location_state": beat.get("location_state", ""),
                    "description": ""})
    return panels
