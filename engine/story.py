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

VOICE_SYSTEM = """You cast an audiobook. Return only the requested JSON, compact on one line.
Quoted speech in the passage is numbered like [3]“...”. For every number give the
speaker's character id, or null when the text does not make the speaker clear.
Use ids from the character bible. Declare only people new to the bible in
characters, with a short local id (c1, c2) and the name exactly as the source
gives it, then use that id. Never use a pronoun, a group, or a phrase describing
someone as a name. For unnamed first-person narration the name Narrator is
allowed. mood is the passage's emotional tone in a few words. voice is a short
direction for how a new person sounds.
"""


def voice_instruction(group, characters):
    """The passage with its quotes numbered, for speaker attribution only."""
    text, cursor, numbered = ''.join(u["text"] for u in group), 0, []
    for n, (at, quoted) in enumerate(quotes(text), 1):
        open_at = at - 1 if at and text[at - 1] in '“"「' else at
        numbered.append(text[cursor:open_at] + f"[{n}]")
        cursor = open_at
    numbered.append(text[cursor:])
    bible = [{k: c[k] for k in ("id", "name", "aliases", "voice")} for c in characters]
    schema = {"mood": "emotional tone", "characters": [{"id": "c1", "name": "literal source name",
              "aliases": [], "voice": "voice direction"}], "speakers": [{"quote": 1, "speaker_id": "c1"}]}
    return ("Character bible:\n" + json.dumps(bible, ensure_ascii=False)
            + "\nReturn this schema:\n" + json.dumps(schema, ensure_ascii=False)
            + "\nPassage:\n" + ''.join(numbered))


def voice_scene_reply(raw, group, number, bible=()):
    """A voice reply restated as a scene reply, so `parse_scene` applies every
    identity and source rule to it unchanged. One beat per passage: the
    audiobook needs who speaks and the mood, not shots or appearances."""
    reply = _loads(raw)
    if not isinstance(reply, dict):
        raise ValueError("reply must be an object")
    _tidy_reply(reply, group, bible)
    said = {}
    for line in _array(reply.get("speakers", []), "speakers"):
        if isinstance(line, dict) and type(line.get("quote")) is int:
            who = line.get("speaker_id")
            said[line["quote"]] = who if isinstance(who, str) and who.strip() else None
    text = ''.join(u["text"] for u in group)
    characters = [{**c, "visual": ""} for c in _array(reply.get("characters", []), "characters")
                  if isinstance(c, dict)]
    mood = reply.get("mood") if isinstance(reply.get("mood"), str) else ""
    return json.dumps({
        "title": f"Passage {number}", "summary": "Voice casting", "characters": characters,
        "beats": [{"units": [u["id"] for u in group], "importance": 2, "tension": 2,
                   "action": "Narrated passage", "setting": "", "mood": mood[:200], "visible": [],
                   "dialogue": [{"speaker_id": said.get(n), "text": quoted, "offscreen": False}
                                for n, (_, quoted) in enumerate(quotes(text), 1)]}]})


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
        # Paragraph breaks do not end a call. Dialogue puts nearly every line
        # in its own paragraph, so a 300-word exchange cost eleven writer
        # calls, each re-sending the whole instruction to analyse one line.
        group.append(unit)
        words += n
    if group:
        groups.append(group)
    for group in groups:
        for i, unit in enumerate(group):
            unit["id"] = i
    return groups


# The Board's analysis in as few written tokens as the drawing needs. The old
# reply restated every quote, and every visible person's expression, pose and
# full outfit in every beat: about 22 output tokens per word of story, which on
# a laptop-sized writer is an hour for a long chapter before a panel is drawn.
# Here quotes are numbered and attributed by number, outfits and rooms are
# stated once per passage, and the reply is restated into the scene format so
# `parse_scene` still applies every identity and source rule.
COMPACT_SYSTEM = """You are a story editor and director. Return only the requested JSON,
compact on one line with no indentation.
Read the numbered source units for causality, dramatic turns and emotion. Group
adjacent units into beats; use every unit exactly once, in order. importance is
1 (context), 2 (quiet action), 3 (meaningful change), 4 (turn), 5 (climax/reveal).
Quoted speech is numbered like [3]“...”. In speakers give, for each number, the
speaker's id, or null when the text does not make the speaker clear.
Use ids from the character bible. Declare in characters only people new to the
bible (short local id c1, c2, with the name exactly as the source gives it), or
a known person with a new literal alias. Never use a pronoun, a group, or a
phrase describing someone as a name. For unnamed first-person narration the
name Narrator is allowed; never when a known character already narrates.
visual is lasting identity only (face, hair, build), never clothing.
cast lists who is visible in a beat; keep someone only heard out of it. feel is
their expression and pose in a few words.
looks gives, once for this passage, each visible person's COMPLETE current outfit
and condition; a change REPLACES the old outfit. Omit people whose look has not
changed from the continuity state. places gives each place's layout, objects,
light and weather once; place is a short stable name, reused when the story
returns there. Do not invent events, appearance or details the source lacks.
Never copy quotation marks into any field; describe speech, do not quote it.
"""


def compact_instruction(group, characters, continuity=None):
    """Units with their quotes numbered, and only the continuity that matters."""
    text = ''.join(u["text"] for u in group)
    marks = {}
    for n, (at, _) in enumerate(quotes(text), 1):
        marks[at - 1 if at and text[at - 1] in '“"「' else at] = n
    numbered, offset = [], 0
    for u in group:
        body = u["text"]
        for at in sorted((a - offset for a in marks if offset <= a < offset + len(body)), reverse=True):
            body = body[:at] + f"[{marks[at + offset]}]" + body[at:]
        numbered.append({"id": u["id"], "text": body})
        offset += len(u["text"])
    continuity = continuity or {}
    known = {c["id"] for c in characters}
    state = {"looks": {k: v for k, v in continuity.get("characters", {}).items() if k in known},
             "places": dict(list(continuity.get("locations", {}).items())[-6:])}
    bible = [{**c, "visual": c["visual"][:VISUAL_LIMIT]} for c in characters]
    schema = {"title": "scene title",
              "characters": [{"id": "c1", "name": "literal source name", "aliases": [],
                              "visual": "face, hair, build", "voice": "voice direction"}],
              "looks": [{"id": "c1", "appearance": "complete current outfit and condition"}],
              "places": [{"place": "kitchen", "state": "layout, objects, light, weather"}],
              "beats": [{"units": [0, 1], "importance": 3, "action": "visible action", "place": "kitchen",
                         "mood": "emotional tone", "cast": [{"id": "c1", "feel": "expression and pose"}]}],
              "speakers": [{"quote": 1, "id": "c1"}]}
    return ("Character bible:\n" + json.dumps(bible, ensure_ascii=False)
            + "\nContinuity state:\n" + json.dumps(state, ensure_ascii=False)
            + "\nReturn this schema:\n" + json.dumps(schema, ensure_ascii=False)
            + "\nNumbered source units:\n" + json.dumps(numbered, ensure_ascii=False))


def _place_key(place):
    return re.sub(r'[^\w]+', '-', place.casefold()).strip('-')


def _loads(raw):
    """Parse a writer's JSON, repairing only mechanical slips.

    Small writers copy the story's curly quotes into a field and then close
    the field with one -- `"...too.’”, "place"` -- which made the whole
    passage unreadable. Only when parsing fails: a curly closing quote just
    before the failure, followed by a delimiter, gets the straight quote it
    was meant to be followed by; trailing commas are dropped. Nothing is
    guessed beyond that."""
    raw = raw.strip()
    if raw.startswith('```'):
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw)
    for _ in range(8):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            fixed = re.sub(r',(\s*[}\]])', r'\1', raw)
            if fixed == raw:
                window = raw[max(0, exc.pos - 200):exc.pos + 1]
                slip = None
                for m in re.finditer(r'[”’」』](?=\s*[,}\]])', window):
                    slip = max(0, exc.pos - 200) + m.end()
                if slip is not None:
                    fixed = raw[:slip] + '"' + raw[slip:]
                else:
                    # A straight quote left unescaped inside a field ends the
                    # string early: the last one before the failure that is not
                    # followed by a delimiter is the stray, so escape it.
                    stray = None
                    for m in re.finditer(r'(?<!\\)"(?!\s*[,:}\]])', raw[:exc.pos]):
                        stray = m.start()
                    if stray is None:
                        raise
                    fixed = raw[:stray] + '\\"' + raw[stray + 1:]
            raw = fixed
    return json.loads(raw)


def _tidy_reply(reply, group, bible):
    """Repairs a small writer's slips that need no judgement, in place.

    A declared person whose name is nowhere in the passage, or beats that skip
    or repeat a unit, used to reject the whole passage -- and after a retry
    leave it unanalysed, losing everyone in it. The person is dropped (lines
    they were given stay unattributed) and the units are regrouped in order;
    no source text is touched either way."""
    text = ''.join(u["text"] for u in group)
    known = {c["id"] for c in bible}
    keep, dropped = [], set()
    for person in reply.get("characters") or []:
        if not isinstance(person, dict):
            continue
        names = [person.get("name"), *(person.get("aliases") or [])]
        literal = any(isinstance(n, str) and n.strip() and re.search(
            r'(?<!\w)' + re.escape(n.strip()) + r'(?!\w)', text, re.I) for n in names)
        narrator = isinstance(person.get("name"), str) and re.fullmatch(
            r'(?:the\s+)?narrator', person["name"].strip(), re.I)
        if literal or narrator or person.get("id") in known:
            keep.append(person)
        else:
            dropped.add(person.get("id"))
    reply["characters"] = keep
    beats = [b for b in reply.get("beats") or [] if isinstance(b, dict)]
    for beat in beats:
        beat["cast"] = [a for a in beat.get("cast") or [] if isinstance(a, dict) and a.get("id") not in dropped]
    for line in reply.get("speakers") or []:
        if isinstance(line, dict) and line.get("id", line.get("speaker_id")) in dropped:
            line["id"] = line["speaker_id"] = None
    # Walk the units in order. A unit starts a new run when it is claimed by
    # a beat not used yet; otherwise (unclaimed, repeated, or claimed out of
    # order) it joins the current run. Every beat is a contiguous run, in order.
    claim = {}
    for b, beat in enumerate(beats):
        for n in beat.get("units") or []:
            if type(n) is int and 0 <= n < len(group):
                claim.setdefault(n, b)
    if beats:
        runs, used = [], set()
        for n in range(len(group)):
            b = claim.get(n)
            if not runs or (b is not None and b not in used):
                b = b if b is not None and b not in used else next(
                    (i for i in range(len(beats)) if i not in used), None)
                if b is None:
                    runs[-1][1].append(n)
                    continue
                used.add(b)
                runs.append((b, [n]))
            else:
                runs[-1][1].append(n)
        reply["beats"] = [{**beats[b], "units": ids} for b, ids in runs]
    return reply


def compact_scene_reply(raw, group, continuity=None, bible=()):
    """A compact reply restated as a scene reply. A reply already in the scene
    format (a larger writer may keep it) is passed through unchanged."""
    reply = _loads(raw)
    if not isinstance(reply, dict):
        raise ValueError("scene must be an object")
    beats = _array(reply.get("beats"), "beats")
    if "summary" in reply or any(isinstance(b, dict) and ("visible" in b or "setting" in b) for b in beats):
        return json.dumps(reply)
    beats = _tidy_reply(reply, group, bible)["beats"]
    continuity = continuity or {}
    looks = {l.get("id"): l.get("appearance") for l in _array(reply.get("looks", []), "looks")
             if isinstance(l, dict) and isinstance(l.get("appearance"), str)}
    places = {l.get("place"): l.get("state") for l in _array(reply.get("places", []), "places")
              if isinstance(l, dict) and isinstance(l.get("state"), str)}
    said = {}
    for line in _array(reply.get("speakers", []), "speakers"):
        if isinstance(line, dict) and type(line.get("quote")) is int:
            who = line.get("id", line.get("speaker_id"))
            said[line["quote"]] = who if isinstance(who, str) and who.strip() else None
    text = ''.join(u["text"] for u in group)
    spoken = [(at, quoted, said.get(n)) for n, (at, quoted) in enumerate(quotes(text), 1)]
    starts = [sum(len(g["text"]) for g in group[:i]) for i in range(len(group) + 1)]
    out = []
    for beat in beats:
        if not isinstance(beat, dict):
            raise ValueError("each beat must be an object")
        ids = _array(beat.get("units"), "units")
        if not ids or any(type(n) is not int or n < 0 or n >= len(group) for n in ids):
            raise ValueError("beat units must be valid source-unit numbers")
        lo, hi = starts[ids[0]], starts[ids[-1] + 1]
        place = beat.get("place") if isinstance(beat.get("place"), str) else ""
        cast = [a for a in _array(beat.get("cast", []), "cast") if isinstance(a, dict)]
        present = {a.get("id") for a in cast}
        visible = [{"character_id": a.get("id"), "expression": str(a.get("feel") or "")[:300],
                    "body_language": "",
                    "appearance": looks.get(a.get("id")) or continuity.get("characters", {}).get(a.get("id"), "")}
                   for a in cast]
        importance = beat.get("importance")
        out.append({"units": ids, "importance": importance, "tension": importance,
                    "action": beat.get("action"), "setting": place, "location_id": _place_key(place),
                    "location_state": places.get(place) or continuity.get("locations", {}).get(_place_key(place), ""),
                    "mood": beat.get("mood") if isinstance(beat.get("mood"), str) else "",
                    "visible": visible,
                    "dialogue": [{"speaker_id": who, "text": quoted, "offscreen": who not in present}
                                 for at, quoted, who in spoken if lo <= at < hi]})
    title = reply.get("title") if isinstance(reply.get("title"), str) and reply["title"].strip() else "Scene"
    return json.dumps({"title": title, "summary": title, "characters": reply.get("characters", []),
                       "beats": out})


def _text(value, field, empty=True):
    if not isinstance(value, str) or len(value) > 4000 or (not empty and not value.strip()):
        raise ValueError(f"{field} must be a {'nonempty ' if not empty else ''}string")
    return value.strip()


# Curly, straight and Japanese quotes. A quote never crosses a line break, so
# a paragraph left open by the continued-speech convention cannot swallow the
# narration that follows it.
# An opening quote that ends its paragraph unclosed, with the next paragraph
# reopening, is continued speech and still a complete utterance.
QUOTE = re.compile(r'[“"]([^”"\n]+)(?:[”"]|(?=\n[ \t]*\n\s*[“"]))|「([^」\n]+)」')


def quotes(text):
    """(offset, utterance) for each complete quoted utterance, outer space trimmed."""
    result = []
    for m in QUOTE.finditer(text):
        group = 1 if m.group(1) is not None else 2
        inner = m.group(group)
        if inner.strip():
            result.append((m.start(group) + len(inner) - len(inner.lstrip()), inner.strip()))
    return result


VISUAL_LIMIT = 600


NO_VISUAL = re.compile(r"(?:none|n/?a|unknown|unspecified|not (?:described|specified|given|stated|mentioned)|none (?:given|defined|described|stated|mentioned))\.?", re.I)


def _merge_visual(known, new):
    """Add only facts not already known; identity text stays bounded.

    Every scene re-describes the cast. Appending each description grew the
    text without limit until the book failed validation and every panel
    prompt carried the whole history."""
    merged = known.strip()
    for fact in re.split(r'(?<=[.;])\s+|\s*;\s*', new.strip()):
        fact = fact.strip().rstrip('.;').strip()
        if not fact or _plain(fact) in _plain(merged):
            continue
        candidate = f"{merged}; {fact}" if merged else fact
        if len(candidate) > VISUAL_LIMIT:
            break
        merged = candidate
    return merged[:VISUAL_LIMIT]


def _plain(text):
    return re.sub(r'[\W_]+', ' ', text).strip().casefold()


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
    local_ids, dropped = set(), set()
    narrative = QUOTE.sub("", source)
    # English, romanized Hindi or Devanagari: a Hinglish "maine socha" is as
    # first-person as "I thought", and its narrator a person in the scene.
    first_person = bool(FIRST_PERSON.search(narrative) or re.search(r'\b(?:we|us|our)\b', narrative, re.I))
    for person in _array(reply.get("characters"), "characters"):
        name = _text(person.get("name"), "name", False)
        local_id = _text(person.get("id"), "character id", False)
        if local_id in local_ids:
            raise ValueError("each local character ID must be unique")
        local_ids.add(local_id)
        if re.fullmatch(r'(?:the\s+)?narrator', name, re.I):
            name = "Narrator"
            known = [c for c in characters if is_narrator(c)]
            if len(known) == 1:
                # Models re-declare the narrator per passage; bind to the known one.
                lookup[local_id] = known[0]["id"]
                continue
            if len(known) > 1:
                raise ValueError("ambiguous narrator; use the narrator's exact character ID")
            if not first_person:
                # A third-person narrating voice is not a person in the scene.
                dropped.add(local_id)
                continue
        aliases = [_text(a, "alias", False) for a in _array(person.get("aliases", []), "aliases")]
        aliases = [a for a in aliases if a.casefold() not in {"he", "she", "they", "him", "her", "his", "their", "we", "you"}]
        identity = {name.casefold(), *(a.casefold() for a in aliases)}
        matches = [c for c in characters if c["id"] == local_id or identity.intersection(
            {c["name"].casefold(), *(a.casefold() for a in c["aliases"])})]
        if len(matches) > 1:
            raise ValueError(f"ambiguous character identity: {name}")
        existing = matches[0] if matches else None
        # Only literal aliases can be added to the persistent bible.
        aliases = [a for a in aliases if re.search(r'(?<!\w)' + re.escape(a) + r'(?!\w)', source, re.I)]
        anonymous_narrator = name == "Narrator"
        if not existing and not anonymous_narrator and not re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', source, re.I):
            raise ValueError(f"character {name} is not in this scene")
        if name.casefold() in {"i", "me", "my", "he", "she", "they", "him", "her", "we", "you"}:
            raise ValueError("use a canonical name, not a pronoun")
        visual = _text(person.get("visual", ""), "visual")
        # "none", "not described": an answer about the description, not one.
        # Kept, it reached every panel prompt as the character's look.
        if NO_VISUAL.fullmatch(visual.strip()):
            visual = ""
        voice = _text(person.get("voice", ""), "voice")
        if existing:
            existing["aliases"] = list(dict.fromkeys([*existing["aliases"], *aliases, name]))
            # First chapter's identity stays stable; later facts are additive.
            existing["visual"] = _merge_visual(existing["visual"], visual)
            if not existing["voice"]:
                existing["voice"] = voice
        else:
            existing = {"id": str(uuid.uuid4()), "name": name, "aliases": aliases,
                        "visual": _merge_visual("", visual), "voice": voice}
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
        if reference in dropped:
            return None
        if reference in lookup:
            return lookup[reference]
        matches = names.get(reference.casefold(), set())
        if len(matches) == 1:
            return next(iter(matches))
        if len(matches) > 1:
            raise ValueError(f"{field} is ambiguous: {reference!r}; use its exact character ID")
        if re.fullmatch(r'(?:the\s+)?narrator', reference, re.I):
            return None
        valid = ', '.join(f"{local}={next(c['name'] for c in characters if c['id'] == cid)}"
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
        # The source, not the reply, decides what was said and where: every
        # quoted utterance becomes a line with its exact span. Model lines only
        # attribute speakers, so a retyped, trimmed, merged or skipped quote
        # cannot drop or alter the author's words.
        dialogue = [{"speaker_id": None, "text": text, "offscreen": True,
                     "start": start + a, "end": start + a + len(text)} for a, text in quotes(excerpt)]
        cursor = 0
        for line in _array(beat.get("dialogue"), "dialogue"):
            if not isinstance(line, dict) or not isinstance(line.get("text"), str):
                continue
            said = _plain(line["text"])
            at = next((n for n in range(cursor, len(dialogue)) if said and (
                said == _plain(dialogue[n]["text"]) or said in _plain(dialogue[n]["text"])
                or _plain(dialogue[n]["text"]) in said)), None)
            if at is None:
                continue
            # One reply line may join an utterance the narration interrupts.
            last = at
            while last + 1 < len(dialogue) and _plain(dialogue[last + 1]["text"]) in said:
                last += 1
            for n in range(at, last + 1):
                if line.get("speaker_id") is not None:
                    dialogue[n]["speaker_id"] = resolve(line["speaker_id"], "dialogue speaker")
                if type(line.get("offscreen")) is bool:
                    dialogue[n]["offscreen"] = line["offscreen"]
            cursor = last + 1
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
    _attribute_by_tags(beats, source, group[0]["start"], characters)
    return {"id": scene_id, "title": title, "summary": summary,
            "start": group[0]["start"], "end": group[-1]["end"], "beats": beats}, characters


SPEECH_VERBS = (r'said|says|asked|asks|replied|replies|answered|called|calls|shouted|yelled|cried|'
                r'whispered|muttered|murmured|added|snapped|laughed|sighed|told|continued|insisted|warned|began')


# The narrator in their own narration: English, romanized Hindi, Devanagari.
# "main" alone is also an English adjective, so it counts only beside others.
_EDGE_L, _EDGE_R = r"(?<![\w\u0900-\u097F])", r"(?![\w\u0900-\u097F])"
FIRST_PERSON = re.compile(_EDGE_L + r"(?:I|[Mm]e|[Mm]y|[Mm]yself|(?i:main|maine|mujhe|mujhse|mujhko|mera|meri|mere)"
                          r"|मैं|मैंने|मुझे|मुझसे|मुझको|मेरा|मेरी|मेरे)" + _EDGE_R)


def is_narrator(character):
    return character["name"] == "Narrator" or bool(set(character["aliases"]).intersection({"I", "me", "my"}))


def _ensure_narrator(characters, source):
    """A story told in the first person gets a Narrator: in a comic they are a
    person to draw, and in the prose their own lines need someone to belong to.
    First person means the narration, not the dialogue, says I, main or मैं --
    at least three times, so one stray word is not a narrator -- or tags a
    line of its own ('"...," I said')."""
    if any(is_narrator(c) for c in characters):
        return characters
    told = len(FIRST_PERSON.findall(QUOTE.sub(" ", source)))
    tagged = re.search(r'[”"」]\s*,?\s*I\s+(?:' + SPEECH_VERBS + r')\b', source)
    if told < 3 and not (told and tagged):
        return characters
    return [*characters, {"id": str(uuid.uuid4()), "name": "Narrator", "aliases": [], "visual": "", "voice": ""}]


def ensure_narrator(book):
    """The book with a Narrator if any chapter is told in the first person --
    for books analysed before the narrator was a character."""
    source = "\n\n".join(c["source"] for c in book["chapters"])
    characters = _ensure_narrator(book["characters"], source)
    return book if characters is book["characters"] else {**book, "characters": characters}


def _attribute_by_tags(beats, text, base, characters):
    """Give an unattributed line the speaker its speech tag names.

    The writer leaves many lines unattributed even when the prose says who
    spoke -- `"...," I said`, `Agnes said, "..."` -- and the audiobook then
    reads them in the narrator's voice. Only an explicit tag next to the quote
    counts: a character's name or alias, or "I" for a first-person narrator,
    beside a speech verb. A bare "she said" is left alone; guessing it is how
    lines end up in the wrong mouth. The second half of an utterance the tag
    interrupts takes the speaker of the first."""
    names = []
    for c in characters:
        for name in [c["name"], *c["aliases"]]:
            if name.casefold() in {"i", "me", "my"}:
                continue
            if name.strip():
                names.append((name.strip(), c["id"]))
    narrator = [c["id"] for c in characters if is_narrator(c)]
    lines = [d for b in beats for d in b["dialogue"]]

    def who(tag):
        found = set()
        for name, cid in names:
            word = r'(?<!\w)' + re.escape(name) + r'(?!\w)'
            if re.search(word + r'\s+(?:' + SPEECH_VERBS + r')\b', tag) or \
                    re.search(r'\b(?:' + SPEECH_VERBS + r')\s+' + word, tag):
                found.add(cid)
        if len(narrator) == 1 and re.search(r'\bI\s+(?:' + SPEECH_VERBS + r')\b|\b(?:' + SPEECH_VERBS + r')\s+I\b', tag):
            found.add(narrator[0])
        return found.pop() if len(found) == 1 else None

    tagged = set()
    for n, line in enumerate(lines):
        a, b = line["start"] - base, line["end"] - base
        # The tag runs from the quote to the sentence end or the next quote.
        after = re.split(r'[“"「]|(?<=[.!?])\s', text[b + 1:b + 120], maxsplit=1)[0]
        # `"Closed." She said, "..."`: a full stop then a capital starts a new
        # sentence, whose tag is for the next quote, not this one.
        if line["text"].rstrip().endswith(".") and re.match(r'\s*[A-Z]', after) \
                and not re.match(r'\s*I\b', after):
            after = ""
        before = re.split(r'[”"」]|[.!?]\s', text[max(0, a - 120):max(0, a - 1)])[-1]
        # A tag that names the speaker settles it, even over the writer: a
        # small writer gave '"I'm not late," I said' to the woman being
        # answered. Pronoun tags below only fill gaps.
        named = who(after) or who(before)
        if named:
            line["speaker_id"] = named
            tagged.add(n)
            continue
        if line["speaker_id"] is not None:
            continue
        if re.search(
                r'\b(?:he|she)\s+(?:' + SPEECH_VERBS + r')\b|\b(?:' + SPEECH_VERBS + r')\s+(?:he|she)\b',
                after or before, re.I):
            # "she said": the person the narration last named, if any. Quoted
            # words are skipped -- a name inside speech is not who is speaking.
            narration = QUOTE.sub(" ", text[max(0, a - 400):max(0, a - 1)])
            last = None
            for name, cid in names:
                if cid in narrator:
                    continue
                for m in re.finditer(r'(?<!\w)' + re.escape(name) + r'(?!\w)', narration):
                    if last is None or m.end() > last[0]:
                        last = (m.end(), cid)
            line["speaker_id"] = last[1] if last else None
    for n, line in enumerate(lines):
        # A line's own tag stands. A line that continues the previous
        # speaker past their tag ('"...," I said. "..."') takes that speaker
        # even over the writer; otherwise this only fills gaps.
        if n in tagged:
            continue
        for other in (lines[n - 1] if n else None, lines[n + 1] if n + 1 < len(lines) else None):
            if other is None or other["speaker_id"] is None:
                continue
            if line["speaker_id"] is not None and other is not (lines[n - 1] if n else None):
                continue
            gap = text[min(line["end"], other["end"]) - base + 1:max(line["start"], other["start"]) - base - 1]
            # Only a tag between them: `"...," she said, "..."` or `"...," I said. "..."`.
            # The tag must belong to the first quote: it ended on a comma, or the
            # tag starts lowercase (`"Hi!" she said. "Bye."`). `"Closed." She
            # said, "Go home."` is a tag for the second quote only.
            first = lines[n - 1] if other is lines[n - 1] else line
            tag = len(gap) < 50 and re.search(r'\b(?:' + SPEECH_VERBS + r')\b', gap) \
                and not re.search(r'[.!?]\s+\S', gap.strip()) \
                and (first["text"].rstrip().endswith((",", "，", "、")) or re.match(r'\s*(?:[a-z]|I\b)', gap))
            # Or the first quote was left open at a paragraph break: speech
            # continued into the next paragraph, which reopens without a tag.
            left_open = text[first["end"] - base:first["end"] - base + 1] not in ('”', '"', '」')
            if tag or (gap.strip() == "" and "\n" in gap and left_open):
                line["speaker_id"] = other["speaker_id"]
                break


def _remember_state(state, beat):
    for actor in beat["visible"]:
        if actor.get("appearance"):
            state["characters"][actor["character_id"]] = actor["appearance"]
    key = beat.get("location_id") or beat["setting"].casefold()
    if beat.get("location_state"):
        state["locations"][key] = beat["location_state"]


def _unanalysed_scene(group, scene_id, number):
    """Exact source and quotes for a passage the writer could not analyse.

    One stubborn passage used to discard every scene already analysed. The
    passage keeps its words, its quoted speech (unattributed) and the board
    still gets a frame for it; nothing about who or where is guessed."""
    beats = []
    for i, unit in enumerate(group):
        beats.append({"id": f"{scene_id}-b{i + 1}", "start": unit["start"], "end": unit["end"],
                      "source": unit["text"], "importance": 2, "tension": 2,
                      "action": unit["text"].strip()[:300], "setting": "", "location_id": "",
                      "location_state": "", "mood": "", "visible": [],
                      "dialogue": [{"speaker_id": None, "text": text, "offscreen": True,
                                    "start": unit["start"] + a, "end": unit["start"] + a + len(text)}
                                   for a, text in quotes(unit["text"])]})
    return {"id": scene_id, "title": f"Passage {number} (not analysed)", "summary": "",
            "start": group[0]["start"], "end": group[-1]["end"], "beats": beats, "unanalysed": True}


class Stop(Exception):
    """Raised by a writer or progress callback to end analysis early.

    Stopping used to throw away every finished passage, so an hour of
    analysis was all or nothing. The finished passages are kept and the rest
    are left unanalysed, for `resume` to pick up."""


def _analyze_passage(group, number, scene_id, characters, continuity, write, mode, warn, may_refuse):
    voice = mode == "voice"
    prompt = (voice_instruction(group, characters) if voice
              else compact_instruction(group, characters, continuity))
    for attempt in range(2):
        raw = write(VOICE_SYSTEM if voice else COMPACT_SYSTEM, prompt)
        try:
            reply = (voice_scene_reply(raw, group, number, characters) if voice
                     else compact_scene_reply(raw, group, continuity, characters))
            return parse_scene(reply, group, characters, scene_id)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            if attempt:
                # A writer that cannot produce any valid scene is broken or
                # unsuitable; refuse rather than return a chapter of stubs.
                if may_refuse:
                    raise ValueError(f"Scene {number} could not be validated: {exc}") from exc
                warn(f"Scene {number} could not be validated ({exc}); kept its text unanalysed")
                return _unanalysed_scene(group, scene_id, number), characters
            prompt += (f"\nYour previous reply failed validation: {exc}."
                       "\nPrevious reply to repair:\n" + raw[:20000]
                       + "\nReturn corrected complete JSON, preserving the full source coverage. "
                       "Every visible character and speaker must refer to an ID declared in "
                       "the character bible or your characters array. Use null for uncertain speakers.")


def _continuity(chapters, scenes=()):
    state = {"characters": {}, "locations": {}}
    for scene in [s for c in chapters for s in c["scenes"]] + list(scenes):
        for beat in scene["beats"]:
            _remember_state(state, beat)
    return state


def analyze(source, write, previous=None, title="Chapter", progress=lambda *_: None,
            passage_words=450, passage_units=8, warn=lambda _: None, mode="full"):
    """`mode="voice"` asks only who speaks each quote and the mood: enough for
    the audiobook at a fraction of the writing, but nothing to draw from."""
    if mode not in ("full", "voice"):
        raise ValueError("unknown analysis mode")
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
    continuity = _continuity(book["chapters"])
    for i, group in enumerate(groups):
        try:
            progress(i, len(groups))
            refuse = not any(not s.get("unanalysed") for s in scenes) and i >= min(2, len(groups) - 1)
            scene, book["characters"] = _analyze_passage(
                group, i + 1, f"{chapter_id}-s{i + 1}", book["characters"], continuity,
                write, mode, warn, refuse)
        except Stop:
            scenes.extend(_unanalysed_scene(g, f"{chapter_id}-s{j + 1}", j + 1)
                          for j, g in enumerate(groups[i:], i))
            break
        scenes.append(scene)
        for beat in scene["beats"]:
            _remember_state(continuity, beat)
    # Once more over the whole chapter: a tag's name can sit in the passage
    # before, and unanalysed passages have no attribution of their own.
    book["characters"] = _ensure_narrator(book["characters"], source)
    _attribute_by_tags([b for s in scenes for b in s["beats"]], source, 0, book["characters"])
    chapter = {"id": chapter_id, "title": title.strip() or "Chapter",
               "source": source, "source_hash": hashlib.sha256(source.encode()).hexdigest(),
               "scenes": scenes, **({"analysis": "voice"} if mode == "voice" else {})}
    book["chapters"].append(chapter)
    validate_book(book)
    return book


def resume(book, chapter_id, write, progress=lambda *_: None, passage_words=450,
           passage_units=8, warn=lambda _: None):
    """Analyse only the passages a stopped or failed run left unanalysed.

    Each is re-divided and analysed with the bible and continuity of
    everything before it; finished passages are never sent again."""
    book = copy.deepcopy(book)
    validate_book(book)
    chapter = next((c for c in book["chapters"] if c["id"] == chapter_id), None)
    if not chapter:
        raise ValueError("choose an analyzed chapter")
    mode = chapter.get("analysis") or "full"
    before = book["chapters"][:book["chapters"].index(chapter)]
    plan = []
    for k, scene in enumerate(chapter["scenes"]):
        if scene.get("unanalysed"):
            groups = passages(chapter["source"][scene["start"]:scene["end"]],
                              max_words=passage_words, max_units=passage_units)
            for group in groups:
                for unit in group:
                    unit["start"] += scene["start"]
                    unit["end"] += scene["start"]
            plan.append((k, groups))
    total, done, stopped, rebuilt = sum(len(g) for _, g in plan), 0, False, []
    replaced = dict((k, []) for k, _ in plan)
    for k, groups in plan:
        for group in groups:
            scene_id = f"{chapter_id}-s{uuid.uuid4().hex[:8]}"
            if not stopped:
                try:
                    progress(done, total)
                    earlier = [done_scene for j, old in enumerate(chapter["scenes"][:k])
                               for done_scene in replaced.get(j, [old])]
                    scene, book["characters"] = _analyze_passage(
                        group, k + 1, scene_id, book["characters"],
                        _continuity(before, earlier + replaced[k]), write, mode, warn, False)
                except Stop:
                    stopped = True
            if stopped:
                scene = _unanalysed_scene(group, scene_id, k + 1)
            replaced[k].append(scene)
            done += 1
    for k, scene in enumerate(chapter["scenes"]):
        rebuilt.extend(replaced.get(k, [scene]))
    chapter["scenes"] = rebuilt
    book["characters"] = _ensure_narrator(book["characters"], chapter["source"])
    _attribute_by_tags([b for s in rebuilt for b in s["beats"]], chapter["source"], 0, book["characters"])
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
        # Books saved before visuals were bounded may hold longer text; it is
        # trimmed where it is used, so it must not make a saved book unreadable.
        if not isinstance(c.get("visual"), str):
            raise ValueError("visual must be a string")
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


def _with_tags(book, chapter):
    """The chapter with speech tags applied, for books analysed before them."""
    chapter = copy.deepcopy(chapter)
    _attribute_by_tags([b for s in chapter["scenes"] for b in s["beats"]], chapter["source"], 0, book["characters"])
    return chapter


def narration_segments(book, chapter_id):
    validate_book(book)
    chapter = next((c for c in book["chapters"] if c["id"] == chapter_id), None)
    if not chapter:
        raise ValueError("choose an analyzed chapter")
    chapter = _with_tags(book, chapter)
    # Narration is split at beats so it carries the beat's mood like dialogue
    # does; it is most of a chapter and was always read without one.
    # Each segment names its scene, so narration can mark where scenes begin.
    source, segments = chapter["source"], []
    for scene in chapter["scenes"]:
        tag = {"scene": scene["title"]}
        for beat in scene["beats"]:
            cursor = beat["start"]
            for line in beat["dialogue"]:
                if line["start"] > cursor:
                    segments.append({"speaker_id": None, "text": source[cursor:line["start"]], "mood": beat["mood"], **tag})
                segments.append({"speaker_id": line["speaker_id"], "text": line["text"], "mood": beat["mood"], **tag})
                cursor = line["end"]
            if beat["end"] > cursor:
                segments.append({"speaker_id": None, "text": source[cursor:beat["end"]], "mood": beat["mood"], **tag})
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
    chapter = _with_tags(book, chapter)
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
            dialogue = [{"speaker": people[d["speaker_id"]]["name"] if d["speaker_id"] else "",
                         "speaker_id": d["speaker_id"], "text": text, "offscreen": d["offscreen"]}
                        for d in beat["dialogue"] for text in _chunks(d["text"], 28)]
            frames = max(len(captions), len(dialogue), 3 if beat["importance"] == 5 else 2 if beat["importance"] == 4 else 1)
            actors = [{**a, "name": people[a["character_id"]]["name"],
                       "visual": people[a["character_id"]]["visual"][:VISUAL_LIMIT]} for a in beat["visible"]]
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
