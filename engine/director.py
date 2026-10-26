"""The director: from an analysed chapter to a shot list worth drawing.

Analysis says what happens. Direction decides how a reader lives through it:
which instants get a panel, where the camera stands, how close it is, who it
favours, where people stand, how large each panel is and what words go in it.
The old planner skipped all of that -- it cut a beat's prose into 36-word
pieces and gave every piece the beat's one summary, so a page was the same
picture redrawn with a new seed.

Structure comes from rules, taken from comics practice:
  * Cohn's visual narrative grammar: Establisher, Initial, Prolongation,
    Peak, Release -- one Peak per scene, and it gets the largest panel.
  * McCloud's transitions: action, subject, scene and aspect-to-aspect (the
    quiet panels that make time pass) chosen from what the beat is.
  * Coverage: establish a new place wide, cover dialogue as a two-shot then
    over-the-shoulder shot/reverse with silent reactions, keep everyone on
    one side of the 180-degree line.
  * Variety: never the same shot, angle and subject twice running; no more
    than four close shots in ten.
  * Budget: about one panel per 55 words of prose, never more than the pages
    asked for hold. A comic adapts a story; it does not draw every sentence.

Each panel shows its own instant, in the order the prose tells it, and comes
with a short visual brief -- who, doing what, where, how they look -- built
from the analysis, never the raw sentence. Shot, layout and lettering are
decided here, by rules. A writer model, when one is installed, only rewrites
the briefs the rules could not make drawable, in batches.
"""
from __future__ import annotations

import json
import re

import story

SHOTS = ("EWS", "WS", "MS", "MCU", "CU", "ECU", "INSERT")
ANGLES = ("eye", "high", "low", "bird", "worm", "dutch")
FRAMINGS = ("establishing", "single", "two", "group", "ots", "pov", "insert")
ROLES = ("E", "I", "L", "P", "R")
FORMATS = ("manga", "webtoon", "comic", "picture")

# How close each shot is, for "push in" and for the close-shot cap.
CLOSENESS = {"EWS": 0, "WS": 1, "MS": 2, "MCU": 3, "CU": 4, "ECU": 5, "INSERT": 5}
CLOSE = {"CU", "ECU", "INSERT"}
SLOTS = ("left third", "right third", "centre", "far left", "far right")
# Manga reads right to left, so the first person in a scene -- usually the
# first to speak -- stands on the right, where the reader's eye starts.
SLOTS_RTL = ("right third", "left third", "centre", "far right", "far left")

# The page budget. Measured adaptations run near 90 prose words a panel;
# 55 keeps dialogue scenes readable without drawing every sentence. Panels a
# page: about five in a Western comic, three or four in manga.
WORDS_PER_PANEL = 55
PANELS_PER_PAGE = {"comic": 5, "manga": 4, "webtoon": 6, "picture": 1}
DEFAULT_PAGES = 4
# Words a panel may letter, captions and balloons together (Western target
# 25, ceiling 35; manga ceiling 25), and the most one balloon holds.
PANEL_WORDS = {"comic": 35, "manga": 25, "webtoon": 30, "picture": 10 ** 6}
BALLOON_WORDS = 25

WAIT = re.compile(r"\b(wait(?:ed|ing)?|hours?|minutes?|later|long time|silence|silent|still|"
                  r"clock|night fell|morning came|days? passed|lingered|stayed)\b", re.I)
VULNERABLE = re.compile(r"\b(sad|grief|alone|lonely|afraid|fear|small|lost|tears?|weep|cry)\b", re.I)
VIOLENT = re.compile(r"\b(fight|strike|hit|slam|crash|run|chase|fall|scream|attack|shot|blood|storm)\b", re.I)
THOUGHT = re.compile(r"\bI\b[^.!?]*\b(thought|knew|wondered|felt|remembered|realised|realized|decided|"
                     r"had (?:already )?decided|hoped|feared)\b"
                     r"|(?i:\b(?:maine socha|mujhe laga|mujhe pata (?:tha|thi)|mujhe yaad aaya|maine tay kiya)\b)"
                     r"|मैंने सोचा|मुझे लगा|मुझे पता था|मुझे याद आया")
PERSON = re.compile(r"\b(he|she|they|I|we|him|her|them|me|us|his|hers|their|my)\b", re.I)
FIRST_PERSON = re.compile(story.FIRST_PERSON.pattern + r"|\b(?:we|us|our)\b"
                          r"|(?<![\w\u0900-\u097F])(?:(?i:hum|humne|hamne|hamara|hamari|hamare)|हम|हमने)(?![\w\u0900-\u097F])")
# A first-person sentence about looking is the one place a point-of-view shot
# helps; everywhere else the narrator is a person in the story, drawn.
LOOKING = re.compile(r"\b(?:I|we)\s+(?:\w+\s+)?(saw|looked|watched|noticed|read|stared|glanced|spotted|caught sight)\b"
                     r"|\b(?:maine|main|humne|hamne)\s+(?:\w+\s+){0,3}(?:dekha|dekhi|dekhe|dekh|ghoor|padha|padhi|notice)"
                     r"|(?:मैंने|मैं|हमने)\s+(?:\S+\s+){0,3}(?:देखा|देखी|देखे|देख|घूर|पढ़ा|पढ़ी)", re.I)
# What a sentence is about decides the shot, not where it falls in the beat.
PLACE = re.compile(r"\b(wind|sky|sea|rain|snow|fog|storm|sun|moon|light|dark|room|kitchen|hall|street|"
                   r"harbou?r|pier|shore|beach|window|walls?|house|town|city|forest|field|road|smell(?:ed|s)?|"
                   r"air|gulls?|birds?|waves?|breakwater|buoy|bell)\b", re.I)
DETAIL = re.compile(r"\b(hands?|fingers?|eyes?|cup|mug|glass|kettle|key|letter|note|net|knife|gun|ring|"
                    r"phone|book|bag|suitcase|coin|door ?handle|button|photograph|clock)\b", re.I)
FEELING = re.compile(r"\b(looked|stared|glanced|smiled|frowned|laughed|wept|cried|tears?|blushed|"
                     r"flinched|sighed|gasped|trembl\w*|nodded|shook (?:his|her|my) head)\b", re.I)
# A sentence that tells rather than shows: a name, a thought, a memory. Its
# panel is briefed from the beat's analysed action instead of its words.
ABSTRACT = re.compile(r"\b(?:name(?:d|s)?|called me|thought|knew|wondered|remembered|realis|realiz|believed|"
                      r"hoped|meant|seemed|decided|always|never|once|used to|years? ago)\b", re.I)


def _named(sentence, people):
    """Characters the sentence names (name or alias), in order of mention."""
    found = []
    for c in people:
        for n in [c["name"], *c["aliases"]]:
            m = re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", sentence) if len(n) > 2 else None
            if m:
                found.append((m.start(), c))
                break
    return [c for _, c in sorted(found, key=lambda x: x[0])]


PLACE_NOUNS = (r"kitchen|bedroom|bathroom|hall|hallway|parlou?r|living room|study|office|shop|cafe|café|bar|pub|"
               r"inn|street|alley|road|forest|woods|beach|shore|pier|harbou?r|dock|house|cottage|church|school|"
               r"classroom|station|platform|train|carriage|car|boat|ship|deck|garden|yard|field|cave|castle|"
               r"tower|library|hospital|ward|attic|cellar|basement|barn|market|square|bridge|roof|stairs|lighthouse")
# A place the prose moves into: entered ("into the hall", "walked back to her
# house") or opening the sentence as where we now are ("Her kitchen smelled
# of tar"). "She thought of her house" or "I left my car" mention a place
# without going there, and "The harbour lay grey" names where we already are.
PLACE_NAMED = re.compile(
    r"\b(?:into|inside|onto)\s+(?:the|her|his|my|their|our)\s+(" + PLACE_NOUNS + r")\b"
    r"|\b(?:went|walked|ran|stepped|came|drove|hurried|climbed|returned|headed|moved|crossed|got)\s+"
    r"(?:back\s+)?(?:in|into|to|inside|onto|across|up to|down to)\s+(?:the|her|his|my|their|our)\s+(" + PLACE_NOUNS + r")\b"
    r"|^(?:Her|His|My|Their|Our)\s+(" + PLACE_NOUNS + r")\b", re.I)


def _named_place(sentences, current):
    """The place a passage moves to, when its prose names one the analysis
    did not: a small analyst kept a whole kitchen scene on the pier."""
    for sentence in sentences:
        m = PLACE_NAMED.search(sentence.strip())
        place = m and next(g for g in m.groups() if g)
        if place and place.casefold() not in (current or "").casefold():
            return place.casefold()
    return None


def _human(place):
    return re.sub(r"[_-]+", " ", place or "").strip()


# A bare speech tag left when the quote is taken out of its sentence:
# "she said.", "Agnes said, and laughed.". Only ever applied to a sentence
# that held a quote -- "Rain began." and "Mira laughed." are moments, and
# matching them against the speech verbs on their own silently dropped them.
TAG_ONLY = re.compile(r"^(?:and\s+)?(?:\w+\s+){0,2}(?:" + story.SPEECH_VERBS + r")(?:\s+\w+){0,2}[.!?]?$", re.I)


def _clean(text):
    return re.sub(r"\s+", " ", text).strip(" ,;:—–-\n")


CLOSING_QUOTE = ('”', '"', '」')


def _narration(text, after_quote=False):
    """One unit's narration with its quotes taken out, or "" when what is
    left is only a speech tag. Only a unit holding a quote, or straight after
    one ('“Run!” she cried.' is two units), can be a tag."""
    quoted = after_quote or bool(story.QUOTE.search(text))
    bare = _clean(story.QUOTE.sub(" ", text))
    if not re.search(r"\w", bare) or (quoted and (len(bare.split()) < 2 or TAG_ONLY.match(bare))):
        return ""
    return bare


def _sentences(text):
    """Sentences of narration, quotes removed; empty pieces and bare speech
    tags ("she said.") dropped -- a tag is not a moment, and a writer given
    one invents something to draw. A sentence with no quote is never a tag."""
    out, after = [], False
    for unit in story.units(text):
        bare = _narration(unit["text"], after)
        if bare:
            out.append(bare)
        after = unit["text"].rstrip().endswith(CLOSING_QUOTE)
    return out


def _items(beat):
    """The beat as it is told, in order: lines spoken, and sentences of
    narration. A spoken unit keeps whatever narration it carries beyond its
    tag ("she said, and laughed"), which is that line's own action.

    The old planner took dialogue and narration as two lists and gave each
    line the next unused sentence as its moment, so a line was drawn over an
    unrelated sentence from further down the beat."""
    items, after = [], False
    for unit in story.units(beat["source"]):
        lo, hi = beat["start"] + unit["start"], beat["start"] + unit["end"]
        said = [d for d in beat["dialogue"] if lo <= d["start"] < hi]
        bare = _narration(unit["text"], after)
        after = unit["text"].rstrip().endswith(CLOSING_QUOTE)
        if said:
            items.append({"kind": "say", "lines": said, "rest": bare})
        elif bare and items and items[-1]["kind"] == "say" and not items[-1]["rest"] \
                and re.match(r"[a-z]", bare):
            # '“Run!” she cried, and ran.' -- the narration after a quote,
            # lower-case, is that line's own action.
            items[-1]["rest"] = bare
        elif bare:
            items.append({"kind": "tell", "text": bare})
    return items


def _acting(actor):
    """The analysed expression and pose, so a dialogue panel is acted rather
    than left for the writer to invent ("kneels, trembling")."""
    acting = ", ".join(x for x in (actor.get("expression"), actor.get("body_language")) if x)
    return f", {acting}" if acting else ""


def _who(actor):
    return {"character_id": actor["character_id"], "name": actor["name"], "visual": actor.get("visual", ""),
            "expression": actor.get("expression", ""), "body_language": actor.get("body_language", ""),
            "appearance": actor.get("appearance", "")}


class _Blocking:
    """Where each person stands in each place, kept for the whole chapter.

    People keep their side of the frame and face each other across the
    180-degree line, so a conversation does not flip from panel to panel."""

    def __init__(self, rtl=False):
        self.places = {}
        self.slots = SLOTS_RTL if rtl else SLOTS

    def place(self, key, actors):
        spots = self.places.setdefault(key, {})
        for a in actors:
            if a["character_id"] not in spots:
                taken = set(spots.values())
                spots[a["character_id"]] = next((s for s in self.slots if s not in taken), "centre")
        return {a["name"]: spots[a["character_id"]] for a in actors}

    @staticmethod
    def facing(position):
        return "right" if "left" in position else "left" if "right" in position else "the camera"


def _panel(beat, scene_no, title, chapter_id, **kw):
    cast = kw.pop("cast", [])
    positions = kw.pop("positions", {})
    p = {
        "beat_id": beat["id"], "chapter_id": chapter_id, "scene": scene_no, "scene_title": title,
        "shot": "MS", "angle": "eye", "framing": "single", "camera_side": "A", "focus": "",
        "role": "I", "transition": "action", "weight": 2, "moment": "", "purpose": "action",
        "cast": cast, "characters": [a["name"] for a in cast], "character_in_frame": bool(cast),
        "positions": {n: positions[n] for n in (a["name"] for a in cast) if n in positions},
        "mood": beat["mood"], "tension": beat["tension"], "importance": beat["importance"],
        "setting": beat["setting"], "place": beat["setting"],
        "place_key": beat.get("location_id") or beat["setting"].casefold(),
        "location_state": beat.get("location_state", ""),
        "caption": "", "dialogue": [], "source": "", "description": "",
        # What this panel puts in front of the reader, as source sentences:
        # the one it draws and any it letters. Sentences merged into it to
        # meet the page budget, but neither drawn nor lettered, are `folded`.
        "covers": [], "folded": [], "action_hint": beat["action"], "priority": 0,
    }
    p.update(kw)
    if p["moment"] and not p["covers"] and kw.get("drawn", True):
        p["covers"] = [p["moment"]]
    p.pop("drawn", None)
    p["facing"] = {n: _Blocking.facing(pos) for n, pos in p["positions"].items()}
    p["subject"] = p["focus"] or ", ".join(p["characters"])
    p["action"] = p["moment"]
    return p


def _line(d, people, visible):
    return {"speaker": people[d["speaker_id"]]["name"] if d["speaker_id"] in people else "",
            "speaker_id": d["speaker_id"], "text": d["text"], "offscreen": d["speaker_id"] not in visible}


def budget(words, fmt="manga", pages=DEFAULT_PAGES):
    """How many panels a chapter of `words` gets: about one per 55 words,
    at least three, never more than `pages` pages of this format hold."""
    if fmt not in FORMATS:
        raise ValueError("unknown book format")
    cap = max(1, int(pages or DEFAULT_PAGES)) * PANELS_PER_PAGE[fmt]
    return max(1, min(cap, max(3, round(words / WORDS_PER_PANEL))))


def _section_of(starts, source, at):
    lead = len(source[at:]) - len(source[at:].lstrip())
    return sum(1 for s in starts if s <= at + lead)


def plan(book, chapter_id, fmt="manga", pages=DEFAULT_PAGES, limit=None):
    """The rule-based shot list for a chapter. Deterministic and instant.

    `limit` overrides the page budget's panel count; tests use it to see
    every moment the rules would draw before any is merged."""
    if fmt not in FORMATS:
        raise ValueError("unknown book format")
    story.validate_book(book)
    chapter = next((c for c in book["chapters"] if c["id"] == chapter_id), None)
    if not chapter:
        raise ValueError("choose an analyzed chapter")
    chapter = story._with_tags(book, chapter)
    people = {c["id"]: c for c in book["characters"]}
    narrator = next((c for c in book["characters"] if story.is_narrator(c)), None)
    blocking, panels = _Blocking(rtl=fmt == "manga"), []
    last_place = None
    picture = fmt == "picture"

    # Scenes are the story's own, found in its text; an analysis cut into
    # eight-sentence passages put a scene break, and so a page break and a
    # Peak, wherever the count ran out.
    starts = story.scene_starts(chapter["source"])
    beats = [b for s in chapter["scenes"] for b in s["beats"]]
    sections, titles = {}, {}
    for scene in chapter["scenes"]:
        for beat in scene["beats"]:
            n = _section_of(starts, chapter["source"], beat["start"]) + 1
            sections[beat["id"]] = n
            titles.setdefault(n, scene["title"])
    peaks = {}
    for beat in beats:
        n = sections[beat["id"]]
        if n not in peaks or (beat["importance"], beat["tension"]) > (peaks[n]["importance"], peaks[n]["tension"]):
            peaks[n] = beat

    moved, moved_from = None, None
    for beat in beats:
        scene_no = sections[beat["id"]]
        is_peak = peaks.get(scene_no) is beat
        # A move lasts until the analysis itself names another place:
        # while it keeps repeating the old one, it has simply not noticed.
        if moved and beat["setting"].casefold() != moved_from:
            moved = None
        named = _named_place(_sentences(beat["source"]), beat["setting"])
        if named:
            moved, moved_from = named, beat["setting"].casefold()
        if moved and moved not in beat["setting"].casefold():
            beat = {**beat, "setting": moved, "location_id": moved, "location_state": ""}
        actors = [{**a, "name": people[a["character_id"]]["name"],
                   "visual": people[a["character_id"]]["visual"][:story.VISUAL_LIMIT]} for a in beat["visible"]]
        visible = {a["character_id"] for a in actors}
        key = beat.get("location_id") or beat["setting"].casefold()
        setting_seen = _human(beat["setting"]).casefold()
        positions = blocking.place(key, actors)
        items = _items(beat)
        first_of_beat = len(panels)
        importance = beat["importance"]
        mk = lambda **kw: _panel(beat, scene_no, titles[scene_no], chapter_id, positions=positions, **kw)  # noqa: E731

        # A new place is shown before anything happens in it.
        if (key, setting_seen) != last_place and (key or setting_seen):
            # Only a sentence about the place itself opens on it; one about
            # a person acting belongs to that person's panel.
            first = items[0] if items and items[0]["kind"] == "tell" else None
            opening = first["text"] if first and PLACE.search(first["text"]) and not PERSON.search(first["text"]) \
                and not _named(first["text"], book["characters"]) else ""
            if opening:
                items.pop(0)
            new_scene = not panels or panels[-1]["scene"] != scene_no
            panels.append(mk(shot="EWS" if last_place is None else "WS", angle="high", framing="establishing",
                             role="E", transition="scene", weight=3, purpose="establishing",
                             moment=opening or f"The {_human(beat['setting']) or 'place'}, seen whole",
                             drawn=bool(opening), caption=opening if picture else "",
                             priority=(60 if not panels else 40) if new_scene else 25))
            last_place = (key, setting_seen)

        waiting = not any(i["kind"] == "say" for i in items) and WAIT.search(beat["source"]) and importance <= 3
        talk = 0
        n_told = 0
        for item in items:
            if item["kind"] == "say":
                lines = [_line(d, people, visible) for d in item["lines"]]
                speaker = next((a for a in actors if a["character_id"] == lines[0]["speaker_id"]), None)
                listeners = [a for a in actors if a is not speaker]
                rest = item["rest"]
                # The same speaker's next line joins this panel rather than
                # taking a new one, two lines at most, nothing told between.
                prev = panels[-1] if len(panels) > first_of_beat else None
                if prev and prev.get("talk") and not rest and len(prev["dialogue"]) < 2 \
                        and prev["dialogue"][-1]["speaker_id"] == lines[0]["speaker_id"]:
                    prev["dialogue"].extend(lines)
                    continue
                if talk == 0 and len(actors) >= 2:
                    p = mk(shot="MS", framing="two", cast=[_who(a) for a in actors], role="I", transition="subject",
                           moment=rest or f"{', '.join(a['name'] for a in actors)} face each other")
                elif speaker and listeners and not lines[0]["offscreen"]:
                    p = mk(shot="MCU", framing="ots", focus=speaker["name"],
                           cast=[_who(speaker), _who(listeners[0])], role="L", transition="subject",
                           moment=rest or f"{speaker['name']} speaks{_acting(speaker)}, seen over {listeners[0]['name']}'s shoulder")
                elif speaker:
                    p = mk(shot="MCU", focus=speaker["name"], cast=[_who(speaker)], role="L",
                           transition="subject", moment=rest or f"{speaker['name']} speaks{_acting(speaker)}")
                else:
                    # Heard, not seen: the listener carries the line.
                    target = listeners[0] if listeners else (actors[0] if actors else None)
                    p = mk(shot="MCU" if target else "INSERT", framing="single" if target else "insert",
                           focus=target["name"] if target else "", cast=[_who(target)] if target else [],
                           role="L", transition="subject",
                           moment=rest or (f"{target['name']} listens{_acting(target)}" if target else "A voice from off panel"))
                if not rest:
                    p["covers"] = []
                p["dialogue"], p["talk"] = lines, True
                p["priority"] = importance * 15 + (8 if speaker else 2)
                if picture and rest:
                    p["caption"] = rest
                panels.append(p)
                talk += 1
                continue
            sentence = item["text"]
            if waiting:
                # Time passing is shown, not told: the same place from small
                # aspects, with the person only glimpsed.
                odd = n_told % 2
                panels.append(mk(shot="INSERT" if not odd else "WS", angle="eye" if odd else "high",
                                 framing="insert" if not odd else "single",
                                 cast=[] if not odd else [_who(a) for a in actors[:1]],
                                 focus=actors[0]["name"] if actors and odd else "",
                                 role="L", transition="aspect", purpose="detail", weight=1,
                                 moment=sentence, caption=sentence if picture or n_told == 0 else "",
                                 priority=importance * 6 + (8 if n_told < 2 else 0)))
                n_told += 1
                continue
            by_id = {a["character_id"]: a for a in actors}
            named = _named(sentence, book["characters"])
            # Someone the sentence names is in the panel, even if the
            # analysis forgot to list them as visible.
            subjects = [by_id.get(c["id"]) or {"character_id": c["id"], "name": c["name"],
                        "visual": c["visual"][:story.VISUAL_LIMIT], "expression": "", "body_language": ""}
                        for c in named]
            # The narrator is a character: a first-person sentence draws
            # them, unless it is about what they see.
            first_person = bool(FIRST_PERSON.search(sentence))
            narrator_pov = first_person and bool(LOOKING.search(sentence)) and not subjects
            if first_person and narrator and not narrator_pov and all(x["character_id"] != narrator["id"] for x in subjects):
                subjects = [by_id.get(narrator["id"]) or {"character_id": narrator["id"], "name": narrator["name"],
                            "visual": narrator["visual"][:story.VISUAL_LIMIT], "expression": "", "body_language": ""},
                            *subjects]
            if not subjects and not narrator_pov and not (PERSON.search(sentence) and not first_person):
                shot, frame, cast = ("WS" if PLACE.search(sentence) else "INSERT"), "insert", []
            # A detail shot is for a sentence about the object or the
            # hands; a long sentence that only mentions one is a scene.
            elif DETAIL.search(sentence) and not FEELING.search(sentence) and (
                    len(sentence.split()) <= 10 or re.search(r"\b(hands?|fingers?)\b", sentence, re.I)):
                shot, frame, cast = "INSERT", "insert", subjects[:1] or [_who(a) for a in actors[:1]]
            elif FEELING.search(sentence):
                shot, frame, cast = "MCU" if n_told == 0 else "CU", "single", subjects[:1] or [_who(a) for a in actors[:1]]
            elif narrator_pov:
                shot, frame, cast = "MS", "pov", [_who(a) for a in actors if not narrator or a["character_id"] != narrator["id"]]
            else:
                cast = subjects or [_who(a) for a in actors]
                shot, frame = ("MS" if len(cast) > 1 else "WS" if n_told == 0 else "MS"), ("two" if len(cast) == 2 else "group" if len(cast) > 2 else "single")
            cast = [_who(c) if "character_id" in c else c for c in cast]
            focus = cast[0]["name"] if cast and frame != "pov" else ""
            positions.update(blocking.place(key, cast))
            panels.append(mk(shot=shot, framing=frame, focus=focus, cast=cast, role="I" if n_told == 0 else "L",
                             purpose="detail" if shot == "INSERT" else "action", moment=sentence,
                             caption=sentence if picture else "",
                             priority=importance * 15 + (10 if cast else 0) + (4 if n_told == 0 else 0)))
            n_told += 1

        if len(panels) == first_of_beat:
            # A beat with nothing left to tell still happens on the page.
            panels.append(mk(cast=[_who(a) for a in actors], moment=beat["action"], drawn=False,
                             priority=importance * 10))

        # The Peak: one per scene, the largest panel, camera by feeling.
        if is_peak and importance >= 3 and len(panels) > first_of_beat:
            mine = [p for p in panels[first_of_beat:] if p["role"] != "E"] or panels[first_of_beat:]
            top = next((p for p in reversed(mine) if p["character_in_frame"]), mine[-1])
            feeling = f"{beat['mood']} {beat['action']}"
            top.update(role="P", weight=3, transition="moment", priority=100)
            if top["shot"] == "INSERT":
                feeling = ""  # an object's close-up has no power angle
            if VIOLENT.search(feeling):
                top.update(angle="dutch" if beat["tension"] >= 5 else "low")
            elif VULNERABLE.search(feeling):
                top.update(angle="high", shot="CU" if top["character_in_frame"] else top["shot"])
            elif feeling:
                top.update(angle="low" if beat["tension"] >= 4 else "eye")
        if panels[first_of_beat:]:
            panels[first_of_beat]["source"] = beat["source"]

    words = len(chapter["source"].split())
    allowed = limit or budget(words, fmt, pages)
    if picture:
        panels = _spreads(panels, chapter, allowed, starts)
    else:
        panels = _fit_budget(panels, allowed)
        _letter(panels, fmt)
    for p in panels:
        p["description"] = _brief(p, narrator)
        p.pop("talk", None)
    return rules(panels)


def _fit_budget(panels, limit):
    """Merge the least important panels into their neighbours until the
    chapter fits its pages. Nothing spoken is lost: a merged panel's lines
    move into the panel that absorbs it, and its sentences are carried there
    -- lettered as a caption when there is room (`_letter`), reported as
    folded when there is not."""
    panels = list(panels)
    while len(panels) > limit and len(panels) > 1:
        def neighbours(i):
            return [j for j in (i - 1, i + 1) if 0 <= j < len(panels)]
        within = [i for i in range(len(panels))
                  if any(panels[j]["scene"] == panels[i]["scene"] for j in neighbours(i))]
        pool = within or list(range(len(panels)))
        # The least important panel goes first; of equals, the later one, so
        # a scene keeps its opening.
        i = min(pool, key=lambda k: (panels[k]["priority"], -k))
        options = [j for j in neighbours(i) if panels[j]["scene"] == panels[i]["scene"]] or neighbours(i)

        def fit(j):
            q, p = panels[j], panels[i]
            speakers = {d["speaker_id"] for d in p["dialogue"]}
            here = {a["character_id"] for a in q["cast"]}
            return ((q["beat_id"] == p["beat_id"]) * 2 + (bool(speakers) and speakers <= here) * 2
                    - len(q["dialogue"]) * 0.5 + (j < i) * 0.1)
        j = max(options, key=fit)
        panels[j] = _absorb(panels[j], panels[i], before=i < j)
        del panels[i]
    return panels


def _absorb(keep, gone, before):
    """`keep` with `gone` merged into it, in reading order."""
    keep = dict(keep)
    here = {a["character_id"] for a in keep["cast"]}
    lines = [{**d, "offscreen": d.get("offscreen") or d["speaker_id"] not in here} for d in gone["dialogue"]]
    carried = [s for s in [*gone["covers"], *gone["folded"]] if s]
    if before:
        keep["dialogue"] = lines + keep["dialogue"]
        keep["folded"] = carried + keep["folded"]
        keep["source"] = gone["source"] + keep["source"]
    else:
        keep["dialogue"] = keep["dialogue"] + lines
        keep["folded"] = keep["folded"] + carried
        keep["source"] = keep["source"] + gone["source"]
    if gone.get("caption") and gone["caption"] not in keep["caption"]:
        keep["caption"] = " ".join(x for x in ([gone["caption"], keep["caption"]] if before
                                               else [keep["caption"], gone["caption"]]) if x)
    keep["priority"] = max(keep["priority"], gone["priority"])
    return keep


def _words(text):
    return len(re.findall(r"\w+", text or ""))


def _balloons(text, limit=BALLOON_WORDS):
    """One spoken line as balloons of at most `limit` words, broken between
    sentences where it can be, so a long speech is never cut short."""
    pieces = re.findall(r"[^.!?…]+[.!?…]+[”’\"]?\s*|[^.!?…]+$", text) or [text]
    out, current = [], ""
    for piece in pieces:
        if current and _words(current + piece) > limit:
            out.append(current.strip())
            current = ""
        while _words(piece) > limit:
            words = piece.split()
            out.append(" ".join(words[:limit]))
            piece = " ".join(words[limit:])
        current += piece
    if current.strip():
        out.append(current.strip())
    return out or [text]


def _letter(panels, fmt):
    """Words on the page, within comics' budgets.

    Balloons hold at most 25 words; a longer line is split into several, all
    of it kept. First-person thought is the narrator's voice, a caption.
    Sentences merged into a panel are lettered as narration while the panel
    has words to spare, in reading order; the rest are left folded, and the
    coverage report names them."""
    room = PANEL_WORDS[fmt]
    for p in panels:
        lines = []
        for d in p["dialogue"]:
            lines.extend({**d, "text": t} for t in _balloons(d["text"]))
        p["dialogue"] = lines
        used = sum(_words(d["text"]) for d in lines)
        caption = []
        if not p["caption"] and p["moment"] in p["covers"] and THOUGHT.search(p["moment"]):
            caption.append(p["moment"])
            used += _words(p["moment"])
        folded = []
        for sentence in p["folded"]:
            if used + _words(sentence) <= room and _words(sentence) <= BALLOON_WORDS:
                caption.append(sentence)
                used += _words(sentence)
            else:
                folded.append(sentence)
        if caption:
            p["caption"] = " ".join(x for x in [p["caption"], *caption] if x)
            p["covers"] = p["covers"] + [s for s in caption if s not in p["covers"]]
        p["folded"] = folded


def _spreads(panels, chapter, limit, starts):
    """A picture book shows one moment per scene, as a spread, with the
    author's whole text for that scene beside it; the opening keeps its
    establishing spread when the scene has no stronger image. More scenes
    than the pages hold share a spread, the text of both kept whole."""
    out = []
    source = chapter["source"]
    bounds = [0, *starts, len(source)]
    for scene_no in sorted({p["scene"] for p in panels}):
        mine = [p for p in panels if p["scene"] == scene_no]
        key = next((p for p in mine if p["role"] == "P"), None) \
            or next((p for p in mine if p["character_in_frame"]), mine[0])
        text = source[bounds[scene_no - 1]:bounds[scene_no]] if scene_no < len(bounds) else ""
        spread = dict(key, weight=3, dialogue=[], caption=text.strip(), source=text, folded=[],
                      covers=[s for p in mine for s in p["covers"]] + [_line_text(d) for p in mine for d in p["dialogue"]])
        if spread["shot"] in ("ECU", "INSERT"):
            spread["shot"] = "MS"
        out.append(spread)
    while len(out) > max(1, limit):
        i = min(range(len(out) - 1), key=lambda k: len(out[k]["caption"]) + len(out[k + 1]["caption"]))
        a, b = out[i], out[i + 1]
        out[i] = dict(a, caption=f"{a['caption']}\n\n{b['caption']}", source=a["source"] + b["source"],
                      covers=a["covers"] + b["covers"])
        del out[i + 1]
    return out


def _line_text(d):
    return d["text"]


# ---- the brief: what is in frame, said for a drawer -------------------------

def _third_person(sentence, name):
    """A first-person sentence told about the narrator by name, so "I placed
    it in his hand" is drawn as Mira placing it rather than as nobody. The
    first name is enough, and reads less like a form."""
    if not name:
        return sentence
    name = name.split()[0]
    out = re.sub(r"\bI\b|(?<![\w’'])[Mm]e\b|\b[Mm]yself\b", name, sentence)
    out = re.sub(r"\b[Mm]y\b", f"{name}’s", out)
    out = re.sub(r"\b[Ww]e\b", f"{name} and the others", out)
    out = re.sub(r"\b[Oo]ur\b", "their", out)
    return re.sub(r"(?<![\w’'])us\b", "them", out)


def _brief(p, narrator):
    """The visual brief: who, doing what, where, how they look -- from the
    analysis and the panel's own sentence, never the sentence as it stands.

    Drawn from the raw sentence, a panel of "My name is Mira Vale, though
    Jonas calls me Mi" asked the drawer for a name. A sentence that tells
    rather than shows (a name, a thought, a memory) is briefed from the
    beat's analysed action instead, and marked weak so a writer, when there
    is one, can do better."""
    name = narrator["name"] if narrator else ""
    moment = p["moment"] if p["moment"] in p["covers"] or p["moment"] in p["folded"] else ""
    weak = not moment or bool(ABSTRACT.search(moment) or THOUGHT.search(moment)
                               or story.SELF_NAMED.search(moment)) or _words(moment) < 3
    place = _human(p["setting"])
    if p["framing"] == "establishing" or (not p["cast"] and p["framing"] == "insert"):
        what = _third_person(moment, name) if moment else f"the {place or 'place'}"
        parts = [what, p["location_state"], "nobody in view" if p["framing"] == "establishing" else ""]
        p["brief_weak"] = False
    else:
        in_frame = {a["character_id"] for a in p["cast"]}
        speaker = next((d["speaker"] for d in p["dialogue"] if d.get("speaker_id") in in_frame), "")
        if p["dialogue"] and speaker and (weak or SAID.match(moment)):
            # The line's own action, said of its speaker: "Agnes said, and
            # laughed" is drawn as Agnes laughing, not as the word "said".
            rest = SAID.sub("", moment).strip(" ,") if moment and SAID.match(moment) else ""
            others = [n for n in p["characters"] if n != speaker]
            what = (f"{speaker} {rest[4:]}" if rest.startswith("and ") else f"{speaker}, {rest}" if rest
                    else f"{speaker} speaks" + (f" to {', '.join(others)}" if others else ""))
        elif p["dialogue"] and weak:
            what = f"{', '.join(p['characters'])} listening" if p["characters"] else p["action_hint"]
        else:
            what = p["action_hint"] if weak else moment
        what = _third_person(what, name)
        if place and place.casefold() not in what.casefold():
            what = f"{what.rstrip('.')}, at the {place}"
        acting = "; ".join(f"{a['name']}: {', '.join(x for x in (a['expression'], a['body_language']) if x)}"
                           for a in p["cast"] if a.get("expression") or a.get("body_language"))
        parts = [what, acting]
        p["brief_weak"] = weak and not p["dialogue"]
    brief = ". ".join(_sentence_case(_clean(x).rstrip(".")) for x in parts if x and _clean(x)) + "."
    words = brief.split()
    return " ".join(words[:45]).rstrip(",;:") + ("" if len(words) <= 45 else "…")


# The speech tag at the head of a line's own narration.
SAID = re.compile(r"^(?:\w+\s+){0,2}(?:" + story.SPEECH_VERBS + r")\b[,\s]*", re.I)


def _sentence_case(text):
    return text[:1].upper() + text[1:]


def script(book, chapter_id):
    """Everything a reader of the chapter takes in, in order: each sentence
    of narration and each line spoken. What `coverage` counts against."""
    chapter = next(c for c in book["chapters"] if c["id"] == chapter_id)
    out = []
    for scene in chapter["scenes"]:
        for beat in scene["beats"]:
            for item in _items(beat):
                if item["kind"] == "say":
                    out.extend(d["text"] for d in item["lines"])
                    if item["rest"]:
                        out.append(item["rest"])
                else:
                    out.append(item["text"])
    return out


def coverage(lines, panels, cap=20):
    """How much of the chapter reaches the page, honestly: a sentence counts
    when a panel draws it or letters it, and a spoken line when a balloon
    carries it. A sentence merged into a panel to meet the budget and shown
    nowhere is missing, and so is anything no panel carries at all."""
    def plain(text):
        return re.sub(r"\W+", " ", text or "").strip().casefold()
    shown = set()
    lettered = " "
    for p in panels:
        shown.update(plain(s) for s in p.get("covers", []))
        # Balloons are joined back up: a long line is lettered as several.
        lettered += " ".join([plain(p.get("caption", "")), *(plain(d["text"]) for d in p.get("dialogue", []))]) + " | "
    missing = [s for s in lines if plain(s) not in shown and f" {plain(s)} " not in lettered]
    total = len(lines)
    covered = total - len(missing)
    return {"covered": covered, "total": total, "percent": round(100 * covered / total) if total else 100,
            "missing": missing[:cap], "missing_total": len(missing), "sentences": lines}


def rules(panels):
    """Hard rules, applied after any planner -- the rules or a writer model."""
    out = []
    for p in panels:
        p = dict(p)
        # Establishing a place and pulling back at the end are calm by nature.
        if p["framing"] == "establishing" and p["angle"] in ("dutch", "worm", "low"):
            p["angle"] = "bird" if p["role"] == "R" else "high"
        if p["shot"] not in SHOTS:
            p["shot"] = "MS"
        if p["angle"] not in ANGLES:
            p["angle"] = "eye"
        if p["framing"] not in FRAMINGS:
            p["framing"] = "single"
        prev = out[-1] if out else None
        # Never the same shot twice running, unless it is a deliberate wait.
        if prev and (prev["shot"], prev["angle"], prev["focus"]) == (p["shot"], p["angle"], p["focus"]) \
                and p["transition"] != "aspect":
            if p["shot"] in ("MS", "MCU", "CU") and p["character_in_frame"]:
                p["shot"] = {"MS": "MCU", "MCU": "CU", "CU": "MS"}[p["shot"]]
            else:
                p["angle"] = "high" if p["angle"] == "eye" else "eye"
        out.append(p)
    # A close shot holds one person: listing everyone visible drew a two-
    # person medium shot whatever the shot was called.
    for p in out:
        if p["shot"] in ("MCU", "CU", "ECU") and p["framing"] not in ("ots", "two", "group") and len(p["cast"]) > 1:
            keep = [a for a in p["cast"] if a["name"] == p["focus"]] or p["cast"][:1]
            p["cast"], p["characters"] = keep, [a["name"] for a in keep]
            p["positions"] = {k: v for k, v in p["positions"].items() if k in p["characters"]}
            p["facing"] = {k: v for k, v in p["facing"].items() if k in p["characters"]}
            p["focus"] = keep[0]["name"]
            ids = {a["character_id"] for a in keep}
            p["dialogue"] = [{**d, "offscreen": d.get("offscreen") or d.get("speaker_id") not in ids}
                             for d in p["dialogue"]]
    # No more than four close shots in any ten.
    for i, p in enumerate(out):
        window = out[max(0, i - 9):i + 1]
        if p["shot"] in CLOSE and sum(q["shot"] in CLOSE for q in window) > 4 and p["role"] != "P":
            p["shot"] = "MCU" if p["character_in_frame"] else "MS"
    for p in out:
        p["subject"] = p["focus"] or ", ".join(p["characters"])
        p["action"] = p["moment"]
        p["purpose"] = {"E": "establishing", "R": p.get("purpose", "reaction")}.get(p["role"], p["purpose"])
    return out


# ---- the writer's briefs -------------------------------------------------------

DIRECTOR_SYSTEM = """You write visual briefs for comic panels. Return only compact JSON on one line.
Each panel gives the sentence it shows, who is in frame, the place and the
analysed action. For every panel write visual: what is in frame, in at most
30 words -- who, doing what, pose, expression, what is around them, light.
Say it as something a camera can see: a thought, a name or a memory becomes
the person's face, pose or the object it is about. Use the names given; in
first-person text "I", "me" and "my" are the narrator named in the panel. Do
not invent events, people or objects the panel lacks. Do not name the shot.
"""

# Panels refined per writer call: enough to share the instruction, few enough
# that a small writer keeps the ids straight.
REFINE_BATCH = 10


def refine_prompt(rows_in, cast):
    rows = [{"id": i, "text": p["moment"][:240], "action": p["action_hint"][:160],
             "in frame": p["characters"], "place": _human(p["setting"]), "mood": p["mood"]}
            for i, p in rows_in]
    people = [{"name": c["name"], "look": c["visual"][:160]} for c in cast]
    schema = {"panels": [{"id": 0, "visual": "what is in frame"}]}
    return ("Cast:\n" + json.dumps(people, ensure_ascii=False) + "\nReturn this schema:\n"
            + json.dumps(schema) + "\nPanels:\n" + json.dumps(rows, ensure_ascii=False))


def apply_refinement(panels, ids, raw):
    """Take the writer's briefs for the panels it was asked about. The shot,
    angle, framing and layout stay the rules' own."""
    try:
        reply = story._loads(raw)
    except ValueError:
        return panels
    out = [dict(p) for p in panels]
    for row in reply.get("panels", []) if isinstance(reply, dict) else []:
        if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] not in ids:
            continue
        visual = row.get("visual")
        if isinstance(visual, str) and visual.strip():
            out[row["id"]]["description"] = " ".join(visual.split())[:300]
            out[row["id"]]["brief_weak"] = False
    return out


class Stop(Exception):
    """Raised by a writer or progress callback: keep what is refined so far."""


def direct(book, chapter_id, fmt="manga", write=None, progress=lambda *_: None, pages=DEFAULT_PAGES, limit=None):
    """The shot list, with the briefs the rules could not make drawable
    rewritten by a writer when one is given, a batch at a time.

    A batch whose refinement fails keeps the rules' briefs; Stop keeps the
    batches done so far and the rules' briefs for the rest."""
    panels = plan(book, chapter_id, fmt, pages, limit)
    if write is None:
        return panels
    weak = [i for i, p in enumerate(panels) if p.get("brief_weak")]
    batches = [weak[k:k + REFINE_BATCH] for k in range(0, len(weak), REFINE_BATCH)]
    for n, batch in enumerate(batches):
        try:
            progress(n, len(batches))
            raw = write(DIRECTOR_SYSTEM, refine_prompt([(i, panels[i]) for i in batch], book["characters"]))
            panels = apply_refinement(panels, set(batch), raw)
        except Stop:
            break
        except Exception:  # a writer failure leaves the rules' briefs, never no panels
            continue
    return panels
