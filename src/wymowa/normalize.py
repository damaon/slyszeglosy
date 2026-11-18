"""Written Polish to speakable words: numbers, dates, times, money, units,
abbreviations, acronyms. Output: words and punctuation only, no digits.

Scope (MVP): the forms below. Numbers elsewhere are read as nominative
cardinals; the case a preposition would require ("z 5 osobami") is not
inflected yet. Training scripts spell numbers out, so this matters for
inference only.
"""

import re
import unicodedata

from .lexicon import WORDS
from .numbers import cardinal, digits, ordinal, plural

QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
                        "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
                        "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")"})
UPPER = "A-ZĄĆĘŁŃÓŚŹŻ"
LOWER = "a-ząćęłńóśźż"
MONTHS = ("stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca", "sierpnia", "września",
          "października", "listopada", "grudnia")
LETTER_NAMES = {"a": "a", "ą": "ą", "b": "be", "c": "ce", "ć": "cie", "d": "de", "e": "e", "ę": "ę", "f": "ef",
                "g": "gie", "h": "ha", "i": "i", "j": "jot", "k": "ka", "l": "el", "ł": "eł", "m": "em", "n": "en",
                "ń": "eń", "o": "o", "ó": "u", "p": "pe", "q": "ku", "r": "er", "s": "es", "ś": "eś", "t": "te",
                "u": "u", "v": "fau", "w": "wu", "x": "iks", "y": "igrek", "z": "zet", "ź": "ziet", "ż": "żet"}
# Acronyms read as words, not letter by letter.
ACRONYM_WORDS = {"nato": "nato", "zus": "zus", "nik": "nik", "gus": "gus", "pesel": "pesel", "ufo": "ufo",
                 "lot": "lot", "pis": "pis", "krus": "krus", "nask": "nask", "pap": "pap", "pan": "pan",
                 "ok": "oˈkej", "covid": "kowid", "unesco": "unesko", "nasa": "nasa", "fifa": "fifa", "uefa": "uefa",
                 "it": "aj|ˈti", "ai": "a|ˈi", "usa": "u|es|ˈa"}

# Noun forms after a number: (1, 2-4, 5+, fraction).
UNITS = {
    "km/h": ("kilometr na godzinę", "kilometry na godzinę", "kilometrów na godzinę", "kilometra na godzinę"),
    "km": ("kilometr", "kilometry", "kilometrów", "kilometra"), "kg": ("kilogram", "kilogramy", "kilogramów", "kilograma"),
    "cm": ("centymetr", "centymetry", "centymetrów", "centymetra"), "mm": ("milimetr", "milimetry", "milimetrów", "milimetra"),
    "ml": ("mililitr", "mililitry", "mililitrów", "mililitra"), "min": ("minuta", "minuty", "minut", "minuty"),
    "m": ("metr", "metry", "metrów", "metra"), "g": ("gram", "gramy", "gramów", "grama"), "l": ("litr", "litry", "litrów", "litra"),
    "h": ("godzina", "godziny", "godzin", "godziny"), "s": ("sekunda", "sekundy", "sekund", "sekundy"),
    "°C": ("stopień Celsjusza", "stopnie Celsjusza", "stopni Celsjusza", "stopnia Celsjusza"),
    "°": ("stopień", "stopnie", "stopni", "stopnia"),
    "gr": ("grosz", "grosze", "groszy", "grosza"),
    "tys.": ("tysiąc", "tysiące", "tysięcy", "tysiąca"), "mln": ("milion", "miliony", "milionów", "miliona"),
    "mld": ("miliard", "miliardy", "miliardów", "miliarda"),
}
CURRENCIES = {
    "zł": (("złoty", "złote", "złotych", "złotego"), ("grosz", "grosze", "groszy")),
    "pln": (("złoty", "złote", "złotych", "złotego"), ("grosz", "grosze", "groszy")),
    "€": (("euro",) * 4, ("cent", "centy", "centów")), "eur": (("euro",) * 4, ("cent", "centy", "centów")),
    "euro": (("euro",) * 4, ("cent", "centy", "centów")),
    "$": (("dolar", "dolary", "dolarów", "dolara"), ("cent", "centy", "centów")),
    "usd": (("dolar", "dolary", "dolarów", "dolara"), ("cent", "centy", "centów")),
}
ABBREVIATIONS = (
    ("m.in.", "między innymi"), ("np.", "na przykład"), ("itd.", "i tak dalej"), ("itp.", "i tym podobne"),
    ("tj.", "to jest"), ("tzn.", "to znaczy"), ("tzw.", "tak zwany"), ("ok.", "około"), ("ul.", "ulica"),
    ("al.", "aleja"), ("pl.", "plac"), ("os.", "osiedle"), ("nr", "numer"), ("godz.", "godzina"), ("dr", "doktor"),
    ("prof.", "profesor"), ("mgr", "magister"), ("inż.", "inżynier"), ("hab.", "habilitowany"), ("św.", "święty"),
    ("ks.", "ksiądz"), ("ds.", "do spraw"), ("wg", "według"), ("ws.", "w sprawie"), ("cdn.", "ciąg dalszy nastąpi"),
    ("pt.", "pod tytułem"), ("jw.", "jak wyżej"), ("zob.", "zobacz"), ("por.", "porównaj"), ("wyd.", "wydanie"),
    ("tel.", "telefon"), ("str.", "strona"), ("woj.", "województwo"), ("pow.", "powiat"), ("gm.", "gmina"),
    ("im.", "imienia"), ("gen.", "generał"), ("płk", "pułkownik"), ("kpt.", "kapitan"), ("mjr", "major"),
    ("pkt", "punkt"), ("art.", "artykuł"), ("ust.", "ustęp"), ("poz.", "pozycja"), ("zł", "złotych"),
    ("tys.", "tysięcy"), ("mln", "milionów"), ("mld", "miliardów"), ("proc.", "procent"),
)
# Abbreviations that are also ordinary words before a full stop ("Dałem im."):
# expanded only before a capitalised name, or before a lowercase word.
NEEDS_NAME = frozenset(("im.", "ul.", "al.", "pl.", "os.", "gen.", "ks.", "św."))
NEEDS_WORD = frozenset(("ok.", "por.", "str.", "pow."))
# Only these may also end a sentence; a title (prof., dr) never does.
SENTENCE_FINAL = frozenset(("itd.", "itp.", "cdn.", "tys.", "proc.", "jw.", "np."))
# "im." is "imienia" only after an institution or a capitalised word.
INSTITUTIONS = frozenset("""szkoła szkoły szkole szkołę liceum gimnazjum uniwersytet uniwersytetu akademia akademii
instytut instytutu fundacja fundacji park parku stadion stadionu szpital szpitala teatr teatru plac placu
ulica ulicy biblioteka biblioteki muzeum centrum dom domu""".split())
# Street words agree with a preceding preposition: (no preposition, genitive, locative).
STREETS = {"ul.": ("ulica", "ulicy", "ulicy"), "al.": ("aleja", "alei", "alei"), "pl.": ("plac", "placu", "placu"),
           "os.": ("osiedle", "osiedla", "osiedlu")}
LOCATIVE_PREPOSITIONS = frozenset(("na", "przy", "w", "we", "o", "po"))
GENITIVE_PREPOSITIONS = frozenset(("z", "ze", "do", "od", "spod", "koło", "obok", "u", "wzdłuż", "naprzeciwko"))
SYMBOLS = {"&": " i ", "+": " plus ", "=": " równa się ", "@": " małpa ", "%": " procent ", "/": " ", "\\": " ",
           "*": " ", "#": " ", "_": " ", "~": " ", "^": " ", "<": " ", ">": " ", "°": " stopni "}
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_NUMBER = r"\d+(?:[,.]\d+)?"
SPACES = re.compile("[" + "".join(map(chr, (0x00A0, 0x2009, 0x202F))) + "]")   # no-break, thin spaces


def _sentence_end(text, end):
    """Did an abbreviation's full stop at ``end`` also end the sentence?"""
    return bool(re.match(rf"\s*$|\s+[{UPPER}\"(]", text[end:]))


def _decimal(value):
    """``"3,05"`` -> ``"trzy przecinek zero pięć"``."""
    whole, _, fraction = value.replace(".", ",").partition(",")
    words = cardinal(int(whole))
    if fraction:
        zeros = len(fraction) - len(fraction.lstrip("0"))
        rest = fraction.lstrip("0")
        words += " przecinek " + " ".join(["zero"] * zeros + ([cardinal(int(rest))] if rest else []))
    return words


def _amount(value, forms):
    """Number + agreeing noun; a fraction takes the genitive singular."""
    if "," in value or "." in value:
        return f"{_decimal(value)} {forms[3]}"
    n = int(value)
    return f"{cardinal(n)} {plural(n, forms)}"


def _roman(token):
    total, previous = 0, 0
    for char in reversed(token):
        value = ROMAN[char]
        total = total - value if value < previous else total + value
        previous = max(previous, value)
    return total if _to_roman(total) == token else None


def _to_roman(n):
    out = ""
    for value, symbol in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
                          (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out, n = out + symbol, n - value
    return out


def dates(text):
    def numeric(m):
        day, month, year = int(m["d"]), int(m["m"]), int(m["y"])
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return m[0]
        tail = "." if m["r"] and _sentence_end(text_ref[0], m.end()) else ""
        return f"{ordinal(day, 'm_gen')} {MONTHS[month - 1]} {ordinal(year, 'm_gen')} roku{tail}"

    def written(m):
        words = f"{ordinal(int(m['d']), 'm_gen')} {m['month']}"
        if m["y"]:
            words += f" {ordinal(int(m['y']), 'm_gen')} roku"
            if m["r"] and _sentence_end(text_ref[0], m.end()):
                words += "."
        return words

    text_ref = [text]
    text = re.sub(r"\b(?P<d>\d{1,2})\.(?P<m>\d{1,2})\.(?P<y>\d{4})(?P<r>\s*r\.)?", numeric, text)
    text_ref[0] = text
    months = "|".join(MONTHS)
    text = re.sub(rf"\b(?P<d>\d{{1,2}})\s+(?P<month>{months})(?:\s+(?P<y>\d{{3,4}})(?:\s*(?P<r>r\.|roku\b))?)?", written, text)
    return text


def years(text):
    def year(m):
        locative = bool(m["w"])
        words = ordinal(int(m["y"]), "m_loc" if locative else "m_gen")
        tail = "." if m["r"] == "r." and _sentence_end(text, m.end()) else ""
        return f"{m['w'] or ''}{words} roku{tail}"
    return re.sub(r"\b(?P<w>[wW]\s+)?(?P<y>\d{3,4})\s*(?P<r>r\.|roku\b)", year, text)


def times(text):
    cases = {"o": "f_loc", "po": "f_loc", "od": "f_gen", "do": "f_gen", "około": "f_gen", "przed": "f_inst",
             "między": "f_inst", "za": "f_inst", "na": "f_acc"}
    hour_word = {"f_loc": "godzinie", "f_gen": "godziny", "f_inst": "godziną", "f_acc": "godzinę", "f_nom": "godzina"}

    def time(m):
        hour, minute = int(m["h"]), m["min"]
        preposition = (m["p"] or "").lower()
        if hour > 24 or int(minute) > 59 or (m["sep"] == "." and not (preposition or m["g"])):
            return m[0]
        form = cases.get(preposition, "f_nom")
        words = ordinal(hour, form)
        if minute != "00":
            words += " " + ("zero " + cardinal(int(minute[1])) if minute[0] == "0" else cardinal(int(minute)))
        lead = (m["p"] + " ") if m["p"] else ""
        if m["g"]:
            lead += hour_word[form] + " "
        return lead + words

    pattern = (r"\b(?:(?P<p>[oO]|[oO]d|[dD]o|[pP]o|[pP]rzed|[oO]koło|[mM]iędzy|[zZ]a|[nN]a)\s+)?"
               r"(?:(?P<g>godz\.|godzinie|godziny|godziną|godzina)\s*)?(?P<h>\d{1,2})(?P<sep>[:.])(?P<min>\d{2})\b")
    return re.sub(pattern, time, text)


def money(text):
    def amount(value, currency):
        main, sub = CURRENCIES[currency.lower()]
        whole, _, fraction = value.replace(".", ",").partition(",")
        n = int(whole)
        words = f"{cardinal(n)} {plural(n, main)}"
        if fraction:
            cents = int(fraction.ljust(2, "0")[:2])
            if cents:
                words += f" {cardinal(cents)} {plural(cents, sub)}"
        return words

    symbols = r"zł|PLN|€|EUR|euro|\$|USD"
    text = re.sub(rf"(?P<c>\$|€)\s?(?P<v>\d+(?:[,.]\d{{1,2}})?)\b", lambda m: amount(m["v"], m["c"]), text)
    return re.sub(rf"(?P<v>\d+(?:[,.]\d{{1,2}})?)\s?(?P<c>{symbols})(?![{LOWER}])", lambda m: amount(m["v"], m["c"]), text)


def percents(text):
    return re.sub(rf"(?P<v>{_NUMBER})\s?%", lambda m: _decimal(m["v"]) + " procent", text)


def units(text):
    names = sorted(UNITS, key=len, reverse=True)
    alternatives = "|".join(re.escape(n) for n in names)

    def unit(m):
        words = _amount(m["v"], UNITS[m["u"]])
        if m["u"].endswith(".") and _sentence_end(text, m.end()):
            words += "."
        return words
    return re.sub(rf"(?P<v>{_NUMBER})\s?(?P<u>{alternatives})(?![{LOWER}A-Z])", unit, text)


def romans(text):
    def roman(m):
        value = _roman(m["n"])
        if value is None:
            return m[0]
        after = m["after"] or ""
        if after:
            locative = bool(m["w"])
            if after.startswith("wiek") and after != "wiek":
                form = "m_loc" if locative else "m_gen"
                return f"{m['w'] or ''}{ordinal(value, form)} {after}"
            form = "m_loc" if locative else "m_nom"
            tail = "." if after == "w." and _sentence_end(m.string, m.end()) else ""
            return f"{m['w'] or ''}{ordinal(value, form)} {'wieku' if locative else 'wiek'}{tail}"
        return f"{m['name']} {ordinal(value)}"
    text = re.sub(r"\b(?P<w>[wW]\s+)?(?P<n>[IVXLC]+)\s+(?P<after>w\.|wieku|wiekiem|wiek)(?![a-ząćęłńóśźż])", roman, text)
    # "Jan Paweł II", "Zygmunt I": name and number on one line (a verse may start with the word "I")
    return re.sub(rf"(?P<name>\b[{UPPER}][{LOWER}]{{2,}})[ \t]+(?P<n>[IVX]+)\b(?P<after>)(?P<w>)", roman, text)


PHONE_CUE = re.compile(r"(?:\b(?:tel|telefon\w*|numer\w*|nr|zadzwoń|dzwoń|dzwonić|komórk\w*|sms\w*|infolini\w*|faks\w*|fax"
                       r"|kontakt\w*|whatsapp)\b\.?:?\s*(?:\S+\s+){0,3})$", re.IGNORECASE)
PHONE = re.compile(r"(?<![\d,.])(?<!\d[ -])(?P<cc>\+ ?48[ -]?)?(?P<body>\(?\d{2,3}\)?(?:[ -]\d{2,3}){2,3}|\d{9})(?![ -]?\d|[,.]\d)")


def phones(text):
    """Telephone numbers are read as they are grouped, a pause between groups: "600 123 456" ->
    "sześćset, sto dwadzieścia trzy, czterysta pięćdziesiąt sześć"; a group with a leading zero is
    read digit by digit. A number counts as a phone when it has nine digits and +48, hyphens or
    brackets, or a cue word (tel., numer, zadzwoń...) just before it; "600 123 456 zł" stays an amount."""
    def phone(m):
        body = m["body"]
        groups = re.findall(r"\d+", body)
        if sum(map(len, groups)) != 9:
            return m[0]
        cued = bool(PHONE_CUE.search(text[max(0, m.start() - 60):m.start()]))
        if not (m["cc"] or "-" in body or "(" in body or cued):
            return m[0]
        if len(groups) == 1:
            groups = [body[:3], body[3:6], body[6:]]
        words = [digits(g) if g.startswith("0") else cardinal(int(g)) for g in groups]
        lead = "plus czterdzieści osiem, " if m["cc"] else ""
        return lead + ", ".join(words)
    return PHONE.sub(phone, text)


def groups(text):
    text = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b(?!\.\d)", lambda m: m[0].replace(".", ""), text)   # 10.000
    return re.sub(r"(?<=\d) (?=\d{3}\b)", "", text)                                          # 1 000 000


def plain_numbers(text):
    text = re.sub(r"(?<![\d,])(\d+)\.(\d+)\b", r"\1,\2", text)                                # 3.5 -> 3,5
    text = re.sub(r"(?<=\d)-(?=[a-ząćęłńóśźż])", " ", text)                                  # 5-latek
    text = re.sub(r"(?:(?<=\s)|^)-(?=\d)", "minus ", text)

    def number(m):
        value = m[0]
        if "," in value:
            return _decimal(value)
        if len(value) > 12 or (len(value) > 1 and value.startswith("0")):
            return digits(value)
        return cardinal(int(value))
    return re.sub(r"\d+(?:,\d+)?", number, text)


def abbreviations(text):
    for short, long in ABBREVIATIONS:
        pattern = rf"(?<![\w.])(?i:{re.escape(short)})" + ("" if short.endswith(".") else r"\.?") + r"(?![\w])"

        def replace(m, short=short, long=long):
            following = text_now[0][m.end():]
            before = re.search(r"(\S+)\s*$", text_now[0][:m.start()])
            previous = before[1] if before else ""
            if before and re.search(r"(^|[.!?…]\s*)$", text_now[0][:before.start()]):
                previous = previous.lower()     # capitalised only because it starts the sentence
            if short in NEEDS_NAME and not re.match(rf"\s+[{UPPER}]", following):
                return m[0]
            if short in NEEDS_WORD and not re.match(rf"\s+[{LOWER}]", following):
                return m[0]
            if short == "im." and not (previous.lower() in INSTITUTIONS or previous[:1].isupper()):
                return m[0]
            if short in STREETS:
                nominative, genitive, locative = STREETS[short]
                word = previous.lower()
                long = locative if word in LOCATIVE_PREPOSITIONS else genitive if word in GENITIVE_PREPOSITIONS else nominative
            final = short in SENTENCE_FINAL and _sentence_end(text_now[0], m.end())
            return long + ("." if final else "")
        text_now = [text]
        text = re.sub(pattern, replace, text)
    return text


def acronyms(text):
    def spell(letters):
        names = [LETTER_NAMES.get(c, c) for c in letters.lower()]
        return "|".join(names[:-1] + ["ˈ" + names[-1]])

    def acronym(m):
        word, suffix = m["a"], m["s"]
        key = word.lower()
        if key in ACRONYM_WORDS and not suffix:
            return ACRONYM_WORDS[key]
        if len(word) >= 5 and re.search(r"[AEIOUYĄĘÓ]", word):
            return key + (suffix or "")
        if suffix:
            return "|".join(LETTER_NAMES.get(c, c) for c in key) + suffix
        return spell(word)
    return re.sub(rf"(?<![\w|ˈ])(?P<a>[{UPPER}]{{2,}})(?:-(?P<s>[{LOWER}]+))?(?![\w])", acronym, text)


def hyphens(text):
    def joined(m):
        key = m[0].lower()
        return WORDS[key] if key in WORDS else m[0].replace("-", " ")
    return re.sub(rf"[\w|ˈ]+(?:-[\w|ˈ]+)+", joined, text)


def normalize(text):
    text = unicodedata.normalize("NFKC", text).translate(QUOTES)
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"-{2,}|(?<=\s)-(?=\s)", "—", text)
    text = SPACES.sub(" ", text).replace("|", " ").replace("ˈ", "")    # the respelling marks are ours
    text = re.sub(r"(?<=\w)'(?=\w)", "", text)                  # O'Neill, McDonald's
    for step in (phones, groups, dates, years, times, money, percents, units, romans, plain_numbers, abbreviations, acronyms):
        text = step(text)
    text = hyphens(text)
    for symbol, words in SYMBOLS.items():
        text = text.replace(symbol, words)
    return re.sub(r"\s+", " ", text).strip()
