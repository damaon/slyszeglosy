"""Polish spelling to phones with lexical stress. Rules only, no data files.

Pipeline for one phrase (the words between two punctuation marks):

1. Each word: lexicon respelling if listed, then letters -> phones
   (digraphs, softening i, ki/gi, au/eu).
2. Nasal vowels: ę/ą before a stop or affricate become vowel + homorganic
   nasal (ręka -> rɛŋka, ząb -> zɔmp); before l/ł and nasals they lose their
   nasality; word-final ę is oral (idę -> idɛ).
3. n before a velar is ŋ (bank -> baŋk).
4. Voicing, across word boundaries inside the phrase: an obstruent cluster
   takes the voicing of its last obstruent (prośba -> prɔʑba, jest bardzo ->
   jɛzd bardzɔ). w and rz do not voice what precedes them and are devoiced
   after a voiceless obstruent of the same word (twój -> tfuj, krzak -> kʂak).
   A word-final cluster is voiceless before a vowel or sonorant (Warsaw
   norm: brat Ali -> brat ali) and at the end of the phrase (Bóg -> buk).
5. Stress: penultimate syllable, with listed and rule-based exceptions
   (-yka nouns, -liśmy/-byśmy, conditional -łbym); secondary stress on the
   first syllable of words with four or more syllables; monosyllabic
   function words unstressed.
"""

from dataclasses import dataclass
import re
import unicodedata

from .lexicon import ANTEPENULTIMATE_IKA, PROCLITICS, STEMS, UNSTRESSED, WORDS
from .symbols import OBSTRUENTS, VOICED_PARTNER, VOICELESS, VOICELESS_PARTNER, VOWEL_SET

BOUNDARY, STRESS = "|", "ˈ"
LETTERS = frozenset("aąbcćdeęfghijklłmnńoóprqsśtuvwxyzźż")
VOWEL_LETTERS = frozenset("aąeęioóuy")

SINGLE = {"a": "a", "ą": "ɔ̃", "b": "b", "c": "ts", "ć": "tɕ", "d": "d", "e": "ɛ", "ę": "ɛ̃", "f": "f", "g": "ɡ",
          "h": "x", "i": "i", "j": "j", "k": "k", "l": "l", "ł": "w", "m": "m", "n": "n", "ń": "ɲ", "o": "ɔ",
          "ó": "u", "p": "p", "q": "k", "r": "r", "s": "s", "ś": "ɕ", "t": "t", "u": "u", "v": "v", "w": "v",
          "y": "ɨ", "z": "z", "ź": "ʑ", "ż": "ʐ"}
DIGRAPHS = {"dź": "dʑ", "dż": "dʐ", "dz": "dz", "cz": "tʂ", "sz": "ʂ", "rz": "ʐ", "ch": "x"}
SOFT = {"c": "tɕ", "s": "ɕ", "z": "ʑ", "n": "ɲ"}          # ci si zi ni (+ dzi, handled first)
PALATAL = {"k": "c", "g": "ɟ"}                            # ki gi
# Letters before a/e + u that make the u a syllable of its own (na-uka, nie-uk).
SYLLABIC_U_PREFIXES = frozenset(("na", "za", "po", "do", "prze", "pra", "nie", "przy", "wy"))

STOPS_AFFRICATES = frozenset(("p", "b", "t", "d", "k", "ɡ", "c", "ɟ", "ts", "dz", "tʂ", "dʐ", "tɕ", "dʑ"))
HOMORGANIC = {**dict.fromkeys(("p", "b"), "m"), **dict.fromkeys(("t", "d", "ts", "dz", "tʂ", "dʐ"), "n"),
              **dict.fromkeys(("tɕ", "dʑ"), "ɲ"), **dict.fromkeys(("k", "ɡ", "c", "ɟ"), "ŋ")}
ORAL = {"ɛ̃": "ɛ", "ɔ̃": "ɔ"}
NASALS = frozenset(("m", "n", "ɲ", "ŋ"))
VELARS = frozenset(("k", "ɡ", "c", "ɟ"))
TRANSPARENT = frozenset(("w", "rz"))     # spelling origins that do not pass voicing leftwards

_CONDITIONAL = re.compile(r"[łl][aoyi]?by[mś]?$")
_PAST_PLURAL = re.compile(r"(li|ły)(śmy|ście)$")


@dataclass
class Segment:
    phone: str
    origin: str = ""          # "w" or "rz" where the spelling decides the voicing behaviour
    word: int = 0
    stressed: bool = False    # explicit ˈ in a respelling


def fold(word):
    """Lowercase; letters outside the Polish alphabet lose their diacritics
    (é -> e, ü -> u); anything that is still not a letter is dropped, except
    the respelling marks."""
    out = []
    for char in word.lower():
        if char in LETTERS or char in (BOUNDARY, STRESS):
            out.append(char)
            continue
        base = "".join(c for c in unicodedata.normalize("NFD", char) if not unicodedata.combining(c))
        out.extend(c for c in base if c in LETTERS)
    return "".join(out)


def respell(word):
    """Lexicon respelling of a lowercase word, else the word with stem fixes."""
    if word in WORDS:
        return WORDS[word]
    for stem, replacement in STEMS:
        if stem in word:
            return word.replace(stem, replacement, 1)
    return word


def spell(word):
    """Letters (with optional ``|`` and ``ˈ`` marks) -> segments."""
    out, pending_stress, letters_before = [], False, ""

    def emit(phone, origin=""):
        nonlocal pending_stress
        segment = Segment(phone, origin)
        if pending_stress and phone in VOWEL_SET:
            segment.stressed, pending_stress = True, False
        out.append(segment)

    for morpheme in word.split(BOUNDARY):
        s = morpheme.replace(STRESS, "\0")    # keep positions; \0 marks the stress
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "\0":
                pending_stress = True
                i += 1
                continue
            rest = s[i:].replace("\0", "")
            after = lambda k: rest[k] if k < len(rest) else ""
            if rest.startswith("dzi"):
                emit("dʑ")
                i += _advance(s, i, 3 if after(3) in VOWEL_LETTERS else 2)
                continue
            if rest[:2] in DIGRAPHS:
                emit(DIGRAPHS[rest[:2]], "rz" if rest[:2] == "rz" else "")
                i += _advance(s, i, 2)
                continue
            if ch in SOFT and after(1) == "i":
                emit(SOFT[ch])
                i += _advance(s, i, 2 if after(2) in VOWEL_LETTERS else 1)
                continue
            if ch in PALATAL and after(1) == "i":
                emit(PALATAL[ch])
                i += _advance(s, i, 2 if after(2) in VOWEL_LETTERS else 1)
                continue
            if ch in "ae" and after(1) == "u" and _diphthong(letters_before + s[:i + 1].replace("\0", ""), rest):
                emit(SINGLE[ch])
                emit("w")
                i += _advance(s, i, 2)
                continue
            if ch == "i" and out and out[-1].phone not in VOWEL_SET and out[-1].phone != "j" and after(1) in VOWEL_LETTERS:
                emit("j")                     # pies -> pjɛs, Anglii -> aŋɡlji
            elif ch == "x":
                emit("k")
                emit("s")
            elif ch in SINGLE:
                emit(SINGLE[ch], "w" if ch in "wv" else "")
            i += 1
        letters_before += morpheme.replace(STRESS, "")
    return out


def _advance(s, i, letters):
    """Index step that covers ``letters`` letters of ``s`` from ``i``,
    stepping over stress marks in between."""
    step = 0
    while letters:
        if s[i + step] != "\0":
            letters -= 1
        step += 1
    return step


def _diphthong(before, rest):
    """Is ``au``/``eu`` at the start of ``rest`` one syllable (auto, Europa)?"""
    if before in SYLLABIC_U_PREFIXES:
        return False
    if rest[0] == "e" and (rest == "eum" or rest[2:4] == "sz"):     # muzeum, Mateusz
        return False
    return True


def nasal_vowels(word):
    out = []
    for index, segment in enumerate(word):
        following = word[index + 1].phone if index + 1 < len(word) else None
        if segment.phone not in ORAL:
            out.append(segment)
            continue
        if following is None:
            if segment.phone == "ɛ̃":
                segment.phone = "ɛ"
            out.append(segment)
        elif following in STOPS_AFFRICATES:
            segment.phone = ORAL[segment.phone]
            out += [segment, Segment(HOMORGANIC[following], word=segment.word)]
        elif following in ("l", "w") or following in NASALS:
            segment.phone = ORAL[segment.phone]
            out.append(segment)
        else:
            out.append(segment)
    return out


def velar_nasal(word):
    for index, segment in enumerate(word[:-1]):
        if segment.phone == "n" and word[index + 1].phone in VELARS:
            segment.phone = "ŋ"
    return word


def _set_voicing(segment, voiced):
    if voiced and segment.phone in VOICED_PARTNER:
        segment.phone = VOICED_PARTNER[segment.phone]
    elif not voiced and segment.phone in VOICELESS_PARTNER:
        segment.phone = VOICELESS_PARTNER[segment.phone]


def voicing(phrase):
    """Voicing assimilation over a flat list of segments of one phrase."""
    count = len(phrase)
    word_final = [i == count - 1 or phrase[i + 1].word != phrase[i].word for i in range(count)]
    start = 0
    while start < count:
        if phrase[start].phone not in OBSTRUENTS:
            start += 1
            continue
        end = start
        while end + 1 < count and phrase[end + 1].phone in OBSTRUENTS:
            end += 1
        run = range(start, end + 1)
        if word_final[end]:
            for i in run:
                _set_voicing(phrase[i], False)
        else:
            trigger = end
            while trigger >= start and phrase[trigger].origin in TRANSPARENT:
                trigger -= 1
            if trigger >= start:
                voiced = phrase[trigger].phone not in VOICELESS
                if word_final[trigger]:
                    voiced = False          # what follows is w/rz of the next word: final devoicing
                for i in run:
                    if i <= trigger or phrase[i].word == phrase[trigger].word:
                        _set_voicing(phrase[i], voiced)
        start = end + 1
    return phrase


def stress_position(key, vowels):
    """Index (from the start) of the stressed vowel of a word with ``vowels``
    vowels; ``key`` is the lowercase spelling of the word."""
    from_end = 2
    if vowels >= 3 and _ika(key):
        from_end = 3
    elif vowels >= 4 and key.endswith(("byśmy", "byście")):
        from_end = 4
    elif vowels >= 3 and (_PAST_PLURAL.search(key) or _CONDITIONAL.search(key)):
        from_end = 3
    return max(vowels - from_end, 0)


def _ika(key):
    for ending in ("a", "i", "ę", "ą"):
        if key.endswith(ending) and key[:-1] in ANTEPENULTIMATE_IKA:
            return True
    return key.endswith("ce") and key[:-2] + "k" in ANTEPENULTIMATE_IKA


def stress(word, key):
    """``[(phone, level)]`` for one word's segments."""
    vowels = [i for i, s in enumerate(word) if s.phone in VOWEL_SET]
    levels = [0] * len(word)
    if not vowels:
        return list(zip((s.phone for s in word), levels))
    explicit = [i for i in vowels if word[i].stressed]
    if explicit:
        levels[explicit[0]] = 1
    elif len(vowels) == 1:
        if key not in UNSTRESSED:
            levels[vowels[0]] = 1
    else:
        position = stress_position(key, len(vowels))
        levels[vowels[position]] = 1
        if len(vowels) >= 4 and position >= 2:
            levels[vowels[0]] = 2
    return [(s.phone, level) for s, level in zip(word, levels)]


def phrase_phones(words):
    """Words of one phrase (strings) -> list of words as ``[(phone, stress)]``.

    A single-consonant preposition (w, z) is joined to the next word."""
    keys = [fold(w) for w in words]
    keys = [k for k in keys if k.replace(BOUNDARY, "").replace(STRESS, "")]
    groups = []          # (host key for stress rules, [segments])
    carry = []
    for key in keys:
        plain = key.replace(BOUNDARY, "").replace(STRESS, "")
        segments = spell(respell(key) if key == plain else key)
        if plain in PROCLITICS and key == plain:
            carry += segments
            continue
        groups.append((plain, carry + segments))
        carry = []
    if carry:
        groups.append(("w", carry))
    flat = []
    for number, (_, segments) in enumerate(groups):
        for segment in segments:
            segment.word = number
        segments[:] = velar_nasal(nasal_vowels(segments))
        flat += segments
    voicing(flat)
    return [stress(segments, key) for key, segments in groups]
