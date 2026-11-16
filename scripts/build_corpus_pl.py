"""Script list for the Polish recordings (phase 3).

Sources, all free of third-party licence obligations:
* Mozilla Common Voice Polish sentences (CC0): sentence-collector.txt and
  selected-europarl-v7-pl.txt from common-voice/server/data/pl.
* Sentences written for this project (corpus/own_sentences.txt).
* Templates for numbers, dates, times, money and addresses. They are written
  with digits and spelled out by wymowa.normalize, so training sees exactly the
  forms the normaliser produces at inference.

Every script passes the front end without unknown phones and with nothing
left for the normaliser to change. Splits are by sentence; the bake-off
sentences are kept out entirely (they are the "new sentences" test).

    python scripts/build_corpus_pl.py --collector .../sentence-collector.txt --europarl .../selected-europarl-v7-pl.txt \
        --own corpus/own_sentences.txt --exclude scripts/bakeoff/sentences.jsonl --output data/corpus/scripts.jsonl
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wymowa import normalize, pieces                 # noqa: E402
from wymowa.lexicon import WORDS                     # noqa: E402
from wymowa.symbols import PHONES                    # noqa: E402

UPPER, LOWER = "A-ZĄĆĘŁŃÓŚŹŻ", "a-ząćęłńóśźż"
SHAPE = re.compile(rf"^[{UPPER}][{UPPER}{LOWER} ,;:'\"\-—–…!?.()]+[.?!]$")
FOREIGN = re.compile(r"[qxvQXV]|th|sh|ee|ph|[äöüéèàçñ]", re.IGNORECASE)
ACRONYM = re.compile(rf"\b[{UPPER}]{{2,}}\b")
ARCHAIC = re.compile(r"\b(?:aboż|alboż|azaliż|zaiste|acz|wżdy|jeno|poco|czemuż|iżby|onegdaj|atoli|lubo|"
                     r"waćpan\w*|waszmość\w*|mospan\w*|imć|jegomość\w*|pono)\b", re.IGNORECASE)
SPLITS = {"validation": 300, "test": 300}

FIRST = ("Anna Maria Katarzyna Małgorzata Agnieszka Barbara Ewa Krystyna Elżbieta Zofia Joanna Magdalena Monika Natalia "
         "Julia Zuzanna Hanna Alicja Weronika Oliwia Piotr Krzysztof Andrzej Tomasz Paweł Michał Marcin Jakub Adam "
         "Łukasz Mateusz Kacper Szymon Filip Wojciech Grzegorz Jan Stanisław Bartosz Antoni").split()
LAST = ("Nowak Kowalski Wiśniewski Wójcik Kowalczyk Kamiński Lewandowski Zieliński Szymański Woźniak Dąbrowski "
        "Kozłowski Jankowski Mazur Kwiatkowski Krawczyk Piotrowski Grabowski Nowakowski Pawłowski").split()
CITIES = ("Warszawy Krakowa Gdańska Wrocławia Poznania Łodzi Szczecina Lublina Katowic Białegostoku Rzeszowa Torunia "
          "Olsztyna Opola Kielc").split()
STREETS = "Długiej Polnej Leśnej Słonecznej Krótkiej Szkolnej Ogrodowej Lipowej Brzozowej Kwiatowej Mickiewicza Kościuszki".split()
DAYS = ("w poniedziałek", "we wtorek", "w środę", "w czwartek", "w piątek", "w sobotę", "w niedzielę")
MONTHS = ("stycznia lutego marca kwietnia maja czerwca lipca sierpnia września października listopada grudnia").split()
# Masculine and neuter nouns only: with them "dwa", "trzy", "cztery" agree as the normaliser writes them.
ITEMS = (("jabłko", "jabłka", "jabłek"), ("bilet", "bilety", "biletów"), ("talerz", "talerze", "talerzy"),
         ("krzesło", "krzesła", "krzeseł"), ("długopis", "długopisy", "długopisów"), ("zeszyt", "zeszyty", "zeszytów"))
# Numbers whose noun is genitive plural ("pięć osób", "dwanaście lat"), safe for nouns of any gender.
MANY = [n for n in range(5, 100) if n % 10 in (0, 1, 5, 6, 7, 8, 9) or 11 <= n % 100 <= 19]


def plural(n, forms):
    return forms[0] if n == 1 else forms[1] if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else forms[2]


def templates(rng, count):
    """Template sentences written with digits; normalize() spells them out."""
    name = lambda: rng.choice(FIRST)

    def full():
        first, last = rng.choice(FIRST), rng.choice(LAST)
        if first.endswith("a"):                    # women: Kowalska, Zielińska
            last = re.sub(r"([sc]k)i$", r"\1a", last)
        return f"{first} {last}"
    hour = lambda: f"{rng.randint(6, 22)}:{rng.choice(['00', '05', '10', '15', '20', '30', '40', '45', '50'])}"

    def items():
        n = rng.randint(2, 40)
        return f"{n} {plural(n, rng.choice(ITEMS))}"
    price = lambda: f"{rng.randint(2, 499)},{rng.choice(['00', '50', '99', '20', '75', '49'])} zł"
    patterns = [
        lambda: f"Pociąg do {rng.choice(CITIES)} odjeżdża o {hour()} z peronu {rng.randint(1, 9)}.",
        lambda: f"Spotkanie z zarządem przesunęliśmy na {hour()}.",
        lambda: f"Czy możemy się spotkać {rng.choice(DAYS)} o {hour()}?",
        lambda: f"Bilet kosztuje {price()}, a karta miejska {price()}.",
        lambda: f"Za zakupy zapłaciłam {price()}.",
        lambda: f"Ile kosztuje ten sweter, {price()}?",
        lambda: f"Urodziła się {rng.randint(1, 28)} {rng.choice(MONTHS)} {rng.randint(1950, 2024)} r.",
        lambda: f"W {rng.randint(1990, 2026)} r. przeprowadziliśmy się do {rng.choice(CITIES)}.",
        lambda: f"Termin mija {rng.randint(1, 28)}.{rng.randint(1, 12):02d}.{rng.randint(2025, 2030)} r.",
        lambda: f"Mieszkam przy ul. {rng.choice(STREETS)} {rng.randint(1, 120)}.",
        lambda: f"{full()} mieszka na ul. {rng.choice(STREETS)} {rng.randint(1, 80)} w {rng.choice(['Warszawie', 'Krakowie', 'Gdańsku', 'Poznaniu', 'Lublinie'])}.",
        lambda: f"Do {rng.choice(CITIES)} zostało jeszcze {rng.randint(3, 480)} km.",
        lambda: f"Na zajęcia zapisało się {rng.choice(MANY)} osób.",
        lambda: f"Kupiliśmy {items()} i {items()}.",
        lambda: f"Na termometrze jest {-rng.randint(5, 20)} stopni.",
        lambda: f"Rabat wynosi {rng.choice([5, 10, 15, 20, 25, 30, 40, 50])}%, więc warto poczekać.",
        lambda: f"Wynik to {rng.randint(1, 99)},{rng.randint(1, 9)} punktu na {rng.choice([100, 50, 20])}.",
        lambda: f"Ma {rng.choice([n for n in MANY if n >= 18])} lat i wciąż codziennie biega.",
        lambda: f"{name()}, zadzwoń do mnie przed {hour()}, dobrze?",
        lambda: f"Konferencja trwa od {hour()} do {hour()}.",
        lambda: f"Miasto liczy około {rng.randint(10, 990)} tys. mieszkańców.",
        lambda: f"Rozdział {rng.randint(2, 30)} zaczyna się na stronie {rng.randint(20, 400)}.",
        lambda: f"Paczka waży {rng.randint(1, 30)} kg i kosztuje {price()} za wysyłkę.",
        lambda: f"Jan Paweł II odwiedził Polskę w {rng.choice([1979, 1983, 1987, 1991, 1997, 1999, 2002])} r.",
        lambda: "W XXI wieku praca zdalna stała się codziennością.",
    ]
    out = set()
    while len(out) < count:
        out.add(rng.choice(patterns)())
    return sorted(out)


def clean(line, add_stop=False):
    line = re.sub(r"\s+", " ", line.strip().strip('"').strip())
    line = line.replace("„", '"').replace("”", '"').replace("’", "'")
    if add_stop and line and line[-1].isalpha():
        line += "."
    return line


def usable(text):
    """Why ``text`` cannot be a script, or None."""
    if not SHAPE.match(text):
        return "shape"
    if not 25 <= len(text) <= 180 or not 4 <= len(text.split()) <= 28:
        return "length"
    if re.search(r"\d", text) or ACRONYM.search(text) or ARCHAIC.search(text):
        return "digits_acronym_archaic"
    words = re.findall(rf"[{UPPER}{LOWER}]+", text)
    if any(FOREIGN.search(w) and w.lower() not in WORDS for w in words):
        return "foreign"
    if text.count('"') % 2 or text.count("(") != text.count(")") or "..." in text or "…" in text:
        return "punctuation"
    return None


def front_end(text):
    """Phone count, or None when the front end cannot read it plainly."""
    spoken = normalize(text)
    words_in = re.findall(rf"[{UPPER}{LOWER}]+", text.lower())
    words_out = re.findall(rf"[{UPPER}{LOWER}]+", spoken.lower())
    if words_in != words_out:
        return None
    count = 0
    for punctuation, value in pieces(text):
        if not punctuation:
            for word in value:
                for phone, _ in word:
                    if phone not in PHONES:
                        return None
                    count += 1
    return count or None


def phone_counts(texts):
    counts, pairs = collections.Counter(), collections.Counter()
    for text in texts:
        for punctuation, value in pieces(text):
            if punctuation:
                continue
            for word in value:
                phones = [p for p, _ in word]
                counts.update(phones)
                pairs.update(zip(phones, phones[1:]))
    return counts, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--europarl", type=Path, required=True)
    parser.add_argument("--own", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-characters", type=int, default=560000, help="About 20 h for two voices at 15 chars/s")
    parser.add_argument("--europarl-share", type=float, default=0.25)
    parser.add_argument("--templates", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    seen = set()
    for path in args.exclude:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(json.loads(line)["text"].lower())
    rejected = collections.Counter()

    def pool(lines, origin, url, add_stop=False):
        out = []
        for line in lines:
            text = clean(line, add_stop)
            reason = usable(text)
            if reason:
                rejected[f"{origin}:{reason}"] += 1
                continue
            if text.lower() in seen:
                rejected[f"{origin}:duplicate"] += 1
                continue
            seen.add(text.lower())
            out.append({"text": text, "origin": origin, "source_url": url})
        rng.shuffle(out)
        return out

    base = "https://github.com/common-voice/common-voice/tree/main/server/data/pl"
    collector = pool(args.collector.read_text(encoding="utf-8").splitlines(), "common_voice_sentence_collector", base)
    europarl = pool(rng.sample(args.europarl.read_text(encoding="utf-8").splitlines(), 40000), "common_voice_europarl", base, True)
    own = pool(args.own.read_text(encoding="utf-8").splitlines(), "slyszeglosy_own", "corpus/own_sentences.txt")
    spelled = []
    for text in templates(rng, args.templates * 2):
        spoken = normalize(text)
        if spoken.lower() not in seen and front_end(spoken):
            seen.add(spoken.lower())
            spelled.append({"text": spoken, "written": text, "origin": "slyszeglosy_templates", "source_url": "scripts/build_corpus_pl.py"})
    spelled = spelled[:args.templates]
    print(json.dumps({"pool": {"collector": len(collector), "europarl": len(europarl), "own": len(own), "templates": len(spelled)}}))

    chosen = []
    budget = {"own": sum(len(r["text"]) for r in own), "templates": sum(len(r["text"]) for r in spelled)}
    remaining = args.target_characters - budget["own"] - budget["templates"]
    shares = {"europarl": args.europarl_share * remaining, "collector": (1 - args.europarl_share) * remaining}
    chosen += own + spelled
    for name, rows in (("collector", collector), ("europarl", europarl)):
        # A quarter questions and a tenth exclamations where the source has them; longer sentences first.
        quota = {"?": 0.25 * shares[name], "!": 0.10 * shares[name]}
        quota["."] = shares[name] - quota["?"] - quota["!"]
        used = collections.Counter()
        for row in sorted(rows, key=lambda r: len(r["text"]) < 40):
            end = row["text"][-1]
            if used[end] + len(row["text"]) > quota[end]:
                continue
            if not front_end(row["text"]):
                rejected[f"{row['origin']}:front_end"] += 1
                continue
            used[end] += len(row["text"])
            chosen.append(row)
        if used["?"] < quota["?"] or used["!"] < quota["!"]:      # the source ran out of questions or exclamations
            spare = shares[name] - sum(used.values())
            for row in rows:
                if spare <= 0:
                    break
                if row in chosen or row["text"][-1] != "." or not front_end(row["text"]):
                    continue
                chosen.append(row)
                spare -= len(row["text"])

    # Splits by sentence: validation and test from collector and own sentences only.
    held = [r for r in chosen if r["origin"] in ("common_voice_sentence_collector", "slyszeglosy_own")]
    rng.shuffle(held)
    split_of = {}
    start = 0
    for split, size in SPLITS.items():
        for row in held[start:start + size]:
            split_of[id(row)] = split
        start += size
    rows = []
    for row in chosen:
        text = row["text"]
        identifier = hashlib.sha256(text.encode()).hexdigest()[:16]
        rows.append({"id": identifier, "text": text, "written": row.get("written", text), "split": split_of.get(id(row), "train"),
                     "category": {"?": "question", "!": "exclamation"}.get(text[-1], "statement"), "origin": row["origin"],
                     "source_id": identifier, "source_url": row["source_url"],
                     "source_license": "cc0-1.0" if row["origin"].startswith("common_voice") else "cc-by-nc-4.0",
                     "word_count": len(text.split()), "character_count": len(text)})
    rows.sort(key=lambda r: (r["character_count"], r["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts, pairs = phone_counts(r["text"] for r in rows)
    total_chars = sum(r["character_count"] for r in rows)
    report = {"scripts": len(rows), "characters": total_chars, "estimated_hours_two_voices": round(2 * total_chars / 15 / 3600, 1),
              "by_origin": collections.Counter(r["origin"] for r in rows), "by_split": collections.Counter(r["split"] for r in rows),
              "by_category": collections.Counter(r["category"] for r in rows),
              "rarest_phones": counts.most_common()[:-11:-1], "diphones": len(pairs),
              "rejected": dict(rejected.most_common())}
    args.output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
