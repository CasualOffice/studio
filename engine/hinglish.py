"""Hinglish written in the Latin alphabet, spelled for a Hindi voice.

Kokoro's Hindi voices read Devanagari through espeak-ng's Hindi rules and
read Latin words as English. Romanized Hindi ("main kal office gaya tha")
therefore comes out as English nonsense: an M4 test, transcribed back by
Whisper, was 22% wrong in Latin script and 7% wrong once the Hindi words were
in Devanagari with the English words left in Latin.

This module makes that conversion without a language model, word by word:

- a hand-checked table for the few hundred words that carry most Hinglish
  text, including texting spellings ("nhi", "kr") and the words whose most
  common romanization in the lexicon is wrong ("me" as मि);
- the Dakshina lexicon (hinglish.tsv) for everything else Hindi;
- an English lexicon (misaki's, installed with Kokoro) so that English words
  stay English;
- context for words that are both ("main", "to", "is", "us", "par"): they
  follow the language of the words around them.

A sentence with no Hindi in it is left as it is.
"""
from __future__ import annotations

import functools
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
CLAUSE = re.compile(r"[,;:\"\u201c\u201d()\u2014-]")
SENTENCE = re.compile(r"[^.!?।\n]+[.!?।]*\s*|\n+")

# Spellings checked by hand. These win over the lexicon, whose most frequent
# romanization is sometimes a name or a loanword (tab as टेब, us as यूएस).
COMMON = {
    "main": "मैं", "mai": "मैं", "mein": "में", "me": "में", "mujhe": "मुझे", "mujhse": "मुझसे",
    "maine": "मैंने", "mera": "मेरा", "meri": "मेरी", "mere": "मेरे", "hum": "हम", "ham": "हम",
    "humne": "हमने", "hamara": "हमारा", "hamari": "हमारी", "hamare": "हमारे", "tum": "तुम",
    "tumne": "तुमने", "tumhe": "तुम्हें", "tumhein": "तुम्हें", "tumko": "तुमको", "tumhara": "तुम्हारा",
    "tumhari": "तुम्हारी", "tumhare": "तुम्हारे", "tu": "तू", "tujhe": "तुझे", "tera": "तेरा",
    "teri": "तेरी", "tere": "तेरे", "aap": "आप", "aapne": "आपने", "aapko": "आपको", "aapka": "आपका",
    "aapki": "आपकी", "aapke": "आपके", "wo": "वो", "woh": "वो", "vo": "वो", "voh": "वो", "ye": "ये",
    "yeh": "ये", "yah": "यह", "is": "इस", "us": "उस", "in": "इन", "un": "उन", "use": "उसे",
    "usne": "उसने", "usko": "उसको", "uska": "उसका", "uski": "उसकी", "uske": "उसके", "unhe": "उन्हें",
    "unhone": "उन्होंने", "unka": "उनका", "unki": "उनकी", "unke": "उनके", "iska": "इसका",
    "iski": "इसकी", "iske": "इसके", "apna": "अपना", "apni": "अपनी", "apne": "अपने",
    "hai": "है", "h": "है", "hain": "हैं", "ho": "हो", "hu": "हूँ", "hoon": "हूँ", "hun": "हूँ",
    "tha": "था", "thi": "थी", "the": "थे", "thin": "थीं", "hua": "हुआ", "hui": "हुई", "hue": "हुए",
    "ka": "का", "ki": "की", "ke": "के", "ko": "को", "ne": "ने", "se": "से", "par": "पर", "pe": "पे",
    "tak": "तक", "liye": "लिए", "lie": "लिए", "saath": "साथ", "sath": "साथ", "baad": "बाद",
    "pehle": "पहले", "pahle": "पहले", "aur": "और", "or": "और", "ya": "या", "lekin": "लेकिन",
    "magar": "मगर", "par_": "पर", "kyunki": "क्योंकि", "kyuki": "क्योंकि", "isliye": "इसलिए",
    "to": "तो", "toh": "तो", "bhi": "भी", "hi": "ही", "na": "ना", "nahi": "नहीं", "nahin": "नहीं",
    "nhi": "नहीं", "nai": "नहीं", "mat": "मत", "kya": "क्या", "kyu": "क्यों", "kyun": "क्यों",
    "kyon": "क्यों", "kaise": "कैसे", "kaisa": "कैसा", "kaisi": "कैसी", "kab": "कब", "kahan": "कहाँ",
    "kaha": "कहा", "kaun": "कौन", "kitna": "कितना", "kitni": "कितनी", "kitne": "कितने", "jab": "जब",
    "tab": "तब", "ab": "अब", "abhi": "अभी", "kabhi": "कभी", "phir": "फिर", "fir": "फिर",
    "aaj": "आज", "kal": "कल", "parso": "परसों", "yahan": "यहाँ", "yaha": "यहाँ", "wahan": "वहाँ",
    "waha": "वहाँ", "idhar": "इधर", "udhar": "उधर", "upar": "ऊपर", "neeche": "नीचे", "niche": "नीचे",
    "andar": "अंदर", "bahar": "बाहर", "paas": "पास", "door": "दूर", "dur": "दूर", "jo": "जो",
    "koi": "कोई", "kuch": "कुछ", "kuchh": "कुछ", "sab": "सब", "sabko": "सबको", "bahut": "बहुत",
    "bohot": "बहुत", "bahot": "बहुत", "bht": "बहुत", "sirf": "सिर्फ", "bas": "बस", "ek": "एक",
    "do": "दो", "teen": "तीन", "char": "चार", "paanch": "पाँच", "haan": "हाँ", "han": "हाँ",
    "ha": "हाँ", "nahi_": "नहीं", "accha": "अच्छा", "acha": "अच्छा", "achha": "अच्छा",
    "acchi": "अच्छी", "achhi": "अच्छी", "theek": "ठीक", "thik": "ठीक", "sahi": "सही",
    "galat": "गलत", "yaar": "यार", "yr": "यार", "bhai": "भाई", "arre": "अरे", "are": "अरे",
    "arey": "अरे", "re": "रे", "kar": "कर", "kr": "कर", "karo": "करो", "karna": "करना",
    "krna": "करना", "karte": "करते", "karti": "करती", "karta": "करता", "kiya": "किया",
    "kiye": "किए", "ki_": "की", "raha": "रहा", "rahi": "रही", "rahe": "रहे", "rha": "रहा",
    "rhi": "रही", "rhe": "रहे", "gaya": "गया", "gayi": "गई", "gai": "गई", "gaye": "गए",
    "gye": "गए", "laga": "लगा", "lagi": "लगी", "lage": "लगे", "diya": "दिया", "di": "दी",
    "liya": "लिया", "li": "ली", "le": "ले", "de": "दे", "dekh": "देख", "dekha": "देखा",
    "dekho": "देखो", "sun": "सुन", "suno": "सुनो", "suna": "सुना", "bol": "बोल", "bola": "बोला",
    "boli": "बोली", "bolo": "बोलो", "socha": "सोचा", "socho": "सोचो", "chal": "चल", "chalo": "चलो",
    "chala": "चला", "chali": "चली", "aa": "आ", "aaya": "आया", "aayi": "आई", "aai": "आई",
    "aao": "आओ", "ja": "जा", "jao": "जाओ", "jana": "जाना", "pata": "पता", "chahiye": "चाहिए",
    "sakta": "सकता", "sakti": "सकती", "sakte": "सकते", "wala": "वाला", "wali": "वाली",
    "wale": "वाले", "din": "दिन", "raat": "रात", "subah": "सुबह", "shaam": "शाम", "ghar": "घर",
    "dil": "दिल", "pyaar": "प्यार", "pyar": "प्यार", "dost": "दोस्त", "log": "लोग", "baat": "बात",
    "baatein": "बातें", "chai": "चाय", "pani": "पानी", "khana": "खाना", "man": "मन", "mann": "मन",
    "jaan": "जान", "sach": "सच", "sach_": "सच", "jaldi": "जल्दी", "dheere": "धीरे", "zara": "ज़रा",
    "jara": "ज़रा", "shayad": "शायद", "bilkul": "बिलकुल", "matlab": "मतलब", "waise": "वैसे",
    "aise": "ऐसे", "jaise": "जैसे", "phasi": "फँसी", "phasa": "फँसा", "karun": "करूँ", "karu": "करूँ", "jaun": "जाऊँ",
    "jau": "जाऊँ", "bolun": "बोलूँ", "dekhun": "देखूँ", "kahun": "कहूँ", "pal": "पल", "band": "बंद",
    "bus": "बस", "hona": "होना", "hoga": "होगा", "hogi": "होगी", "honge": "होंगे",
}
COMMON = {k.rstrip("_"): v for k, v in COMMON.items()}

# Words English shares with romanized Hindi. They take the language of the
# words around them in their clause; everything else in COMMON is Hindi
# wherever it appears. In Hinglish the first group is nearly always Hindi, so
# it stays Hindi unless English words surround it; the second is Hindi only
# beside a Hindi word.
LEANS_HINDI = {"main", "me", "ho", "hi", "na", "ya", "ye", "par", "mat", "ab", "jab", "tab", "kar",
               "pe", "tu", "bas", "aa", "ja", "le", "de", "li", "di", "re", "han", "ha", "to"}
LEANS_ENGLISH = {"is", "us", "in", "use", "do", "are", "or", "man", "log", "sun", "door", "bus",
                 "band", "pal", "din", "teen", "char"}
# "the" is the English article far more often than थे, which only ever
# follows the verb it completes ("wo aaye the"). It was in LEANS_HINDI, so
# "The boss ne kaha" opened with थे and "Use the door, yaar" became
# "उसे थे दूर, यार": with no English word beside it, every shared word in
# the clause fell back to Hindi together.
ARTICLE = "the"
AMBIGUOUS = LEANS_HINDI | LEANS_ENGLISH | {ARTICLE}

# Enough English to keep the commonest words English without the lexicon.
ENGLISH_CORE = set("""a an and the of to in on at for from with by as is are was were be been am it this that these
those i you he she we they me him her us them my your his our their what which who whom whose when where why how
not no yes but if or so then than too very can could will would shall should may might must do does did done have
has had get got go went gone come came say said see saw know knew think thought want like just only also about
into over after before up down out off again there here all any some more most other such one two three okay ok
please sorry thanks thank hello hi bye office boss meeting cancel cancelled late time phone call message msg""".split())


@functools.lru_cache(maxsize=1)
def lexicon():
    """Romanization -> Devanagari from the Dakshina lexicon."""
    table = {}
    with open(os.path.join(HERE, "hinglish.tsv"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            roman, _, deva = line.rstrip("\n").partition("\t")
            if deva:
                table[roman] = deva
    return table


@functools.lru_cache(maxsize=1)
def english():
    """English words: misaki's lexicon when Kokoro is installed, else a core list."""
    words = set(ENGLISH_CORE)
    try:
        import importlib.resources as resources
        with resources.files("misaki").joinpath("data/us_gold.json").open(encoding="utf-8") as f:
            words |= {w.lower() for w in json.load(f) if w.isalpha()}
    except Exception:
        pass
    return words


def _kind(word):
    """'h' Hindi, 'e' English, 'a' either, 'u' unknown -- for one lowercased word."""
    if word in AMBIGUOUS:
        return "a"
    if word in COMMON:
        return "h"
    if "'" in word or word in english():
        return "e"
    if word in lexicon():
        return "h"
    return "u"


# "ki" after saying, thinking or knowing is the conjunction कि, not की.
KI_AFTER = set("""bola boli bole kaha socha sochi soche laga lagta lagti lage pata hai tha thi maloom
bataya batayi poocha pucha chahta chahti chahte dekha suna yaad mana likha""".split())

CONSONANTS = [("chh", "छ"), ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"), ("th", "थ"),
              ("dh", "ध"), ("ph", "फ"), ("bh", "भ"), ("sh", "श"), ("k", "क"), ("g", "ग"),
              ("c", "क"), ("j", "ज"), ("t", "त"), ("d", "द"), ("n", "न"), ("p", "प"),
              ("f", "फ़"), ("b", "ब"), ("m", "म"), ("y", "य"), ("r", "र"), ("l", "ल"),
              ("v", "व"), ("w", "व"), ("s", "स"), ("h", "ह"), ("z", "ज़"), ("q", "क"), ("x", "क्स")]
VOWELS = [("aa", "आ", "ा"), ("ee", "ई", "ी"), ("ii", "ई", "ी"), ("oo", "ऊ", "ू"), ("uu", "ऊ", "ू"),
          ("ai", "ऐ", "ै"), ("au", "औ", "ौ"), ("a", "अ", ""), ("e", "ए", "े"), ("i", "इ", "ि"),
          ("o", "ओ", "ो"), ("u", "उ", "ु")]


def spell(word):
    """A phonetic Devanagari spelling for a romanized Hindi word no table knows,
    such as an inflection ("bolegi") or a name."""
    word, out, i, after_consonant = word.lower(), [], 0, False
    while i < len(word):
        vowel = next((v for v in VOWELS if word.startswith(v[0], i)), None)
        if vowel:
            roman, alone, sign = vowel
            last = i + len(roman) == len(word)
            if roman == "a" and last and after_consonant:
                sign = "ा"           # bola, raha: a final a is long
            elif roman == "i" and last and after_consonant:
                sign = "ी"
            out.append(sign if after_consonant else alone)
            i, after_consonant = i + len(roman), False
            continue
        roman, letter = next((c for c in CONSONANTS if word.startswith(c[0], i)), (word[i], ""))
        nxt = word[i + len(roman):i + len(roman) + 1]
        nasal = nxt not in "aeiouy" if nxt else word[i - 1] in "iou"
        if roman == "n" and out and not after_consonant and nasal:
            out.append("ं")          # bandh, hain, nahin: n before a consonant, or ending -in -on -un, is nasal
            i += 1
            continue
        if after_consonant:
            out.append("्")
        out.append(letter)
        i, after_consonant = i + len(roman), bool(letter)
    if out and out[-1] == "्":
        out.pop()
    return "".join(out)


def _hindi(word):
    return COMMON.get(word) or lexicon().get(word) or spell(word)


def sentence(text):
    """One sentence with its Hindi words in Devanagari, or unchanged if it has none."""
    words = list(WORD.finditer(text))
    lower = [m.group().lower() for m in words]
    kinds = [_kind(w) for w in lower]
    hindi, eng = kinds.count("h"), kinds.count("e")
    if not hindi or hindi < 0.3 * (hindi + eng):
        return text
    hindi_sentence = hindi >= eng
    clause, n = [], 0
    for i, m in enumerate(words):
        n += bool(i and CLAUSE.search(text[words[i - 1].end():m.start()]))
        clause.append(n)

    def known(step, i):
        j = i + step
        while 0 <= j < len(kinds) and clause[j] == clause[i] and kinds[j] not in "he":
            j += step
        return kinds[j] if 0 <= j < len(kinds) and clause[j] == clause[i] else None

    # Settle what can be settled first, so the shared words that follow have
    # something to lean on. A word no table knows is Hindi among Hindi words,
    # else left alone; "the" is थे only straight after a Hindi word and
    # never before an English one.
    for i, word in enumerate(lower):
        if kinds[i] == "u":
            kinds[i] = "h" if hindi_sentence and len(word) > 1 and "'" not in word else "e"
    for i, word in enumerate(lower):
        if word == ARTICLE:
            left = kinds[i - 1] if i and clause[i - 1] == clause[i] else None
            right = kinds[i + 1] if i + 1 < len(kinds) and clause[i + 1] == clause[i] else None
            hindi_before = left == "h" or (left == "a" and lower[i - 1] in LEANS_HINDI)
            kinds[i] = "h" if hindi_before and right != "e" else "e"

    chosen = []
    for i, (word, kind) in enumerate(zip(lower, kinds)):
        if kind == "a":
            near = {known(-1, i), known(1, i)} - {None}
            if word in LEANS_HINDI:
                kind = "e" if known(-1, i) == known(1, i) == "e" else "h"
            else:
                left, right = known(-1, i), known(1, i)
                kind = "h" if left == "h" or (left is None and (right == "h" or not near and hindi_sentence)) else "e"
        if kind != "h":
            chosen.append(None)
        elif word == "ki" and i and lower[i - 1] in KI_AFTER:
            chosen.append("कि")
        else:
            chosen.append(_hindi(word))
    out, at = [], 0
    for m, deva in zip(words, chosen):
        out.append(text[at:m.start()])
        out.append(deva or m.group())
        at = m.end()
    out.append(text[at:])
    return "".join(out)


def devanagari(text):
    """Text for a Hindi voice: romanized Hindi in Devanagari, English left in Latin.

    Hindi sentences end with a danda so the voice pauses as it would after a
    full stop."""
    if not WORD.search(text):
        return text
    parts = []
    for piece in SENTENCE.findall(text):
        told = sentence(piece)
        if told != piece and DEVANAGARI.search(told):
            told = re.sub(r"\.(\s*)$", "।\\1", told)
        parts.append(told)
    return "".join(parts)
