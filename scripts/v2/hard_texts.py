"""V2 phase E texts: about 3000 sentences with numbers, dates, hours, amounts, units,
abbreviations, acronyms, Roman numerals, phone numbers and postal codes, plus
questions and exclamations (MC Speech is mostly statements from books).

Only forms that wymowa.normalize reads correctly are used (checked by hand). Each
row keeps the written form ("written") and the spoken form the front end makes of it
("text"); the teacher reads the spoken form, so the audio always matches the phonemes.

Splits: train, 100 validation, 150 test_hard (held out from training; phase H).

    PYTHONPATH=src python scripts/v2/hard_texts.py --v1-corpus data/corpus/scripts.jsonl --output ~/sg-data/v2/hard/texts.jsonl
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_corpus_pl import (CITIES, DAYS, FIRST, ITEMS, LAST, MANY, MONTHS, STREETS, front_end,  # noqa: E402
                             plural, templates)
from wymowa import normalize                                                                    # noqa: E402

ACRONYMS = ("NFZ", "PKP", "ZUS", "PZU", "GUS", "NBP", "PKO", "UE", "USA", "ONZ", "PAN", "SOR", "RTV", "AGD", "PIT", "VAT")
LOCATIVE = ("Warszawie", "Krakowie", "Gdańsku", "Poznaniu", "Lublinie", "Wrocławiu", "Łodzi", "Toruniu")
KINGS = ("Bolesław III Krzywousty", "Kazimierz III Wielki", "Zygmunt II August", "Henryk VIII", "Ludwik XIV",
         "Jan III Sobieski", "Władysław II Jagiełło", "Karol V", "Piotr I Wielki", "Elżbieta II")
CENTURIES = ("XV", "XVI", "XVII", "XVIII", "XIX", "XX", "XXI")


def new_patterns(rng):
    hour = lambda: f"{rng.randint(5, 23)}:{rng.choice(['00', '05', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55'])}"
    date = lambda: f"{rng.randint(1, 28)} {rng.choice(MONTHS)}"
    full_date = lambda: f"{rng.randint(1, 28)}.{rng.randint(1, 12):02d}.{rng.randint(1990, 2035)} r."
    price = lambda: f"{rng.randint(1, 2999)},{rng.choice(['00', '10', '25', '50', '75', '90', '99'])} zł"
    person = lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}"

    def items():
        n = rng.randint(2, 60)
        return f"{n} {plural(n, rng.choice(ITEMS))}"
    phone = lambda: f"{rng.randint(500, 899)}-{rng.randint(100, 999)}-{rng.randint(100, 999)}"
    postal = lambda: f"{rng.randint(10, 99)}-{rng.randint(100, 999)}"
    return [
        # hours and dates
        lambda: f"Spotkanie zaczyna się o {hour()} i potrwa do {hour()}.",
        lambda: f"Przyjdź o godz. {hour()}, bo potem wychodzę.",
        lambda: f"Czy autobus o {hour()} jeździ także {rng.choice(DAYS)}?",
        lambda: f"Sklep jest czynny od {hour()} do {hour()}.",
        lambda: f"Wizyta u lekarza jest {date()} o {hour()}.",
        lambda: f"Urodziny {rng.choice(FIRST)} wypadają {date()}.",
        lambda: f"Umowa obowiązuje od {full_date()}",
        lambda: f"Faktura z dnia {full_date()} jest już opłacona.",
        lambda: f"Wyjeżdżamy {date()}, a wracamy {date()}.",
        lambda: f"Do {date()} trzeba złożyć wniosek, prawda?",
        lambda: f"W {rng.randint(1918, 2026)} r. wszystko się zmieniło.",
        lambda: f"Dom zbudowano w {rng.randint(1890, 2020)} roku.",
        # money and amounts
        lambda: f"Rachunek za prąd wyniósł {price()}.",
        lambda: f"Czy ta kurtka naprawdę kosztuje {price()}?",
        lambda: f"Kurs euro wynosi dziś {rng.randint(4, 4)},{rng.randint(10, 60)} zł.",
        lambda: f"Bilet do Berlina kosztuje {rng.randint(10, 99)} €, a do Londynu ${rng.randint(20, 150)}.",
        lambda: f"Budżet miasta to {rng.randint(2, 950)} mln zł.",
        lambda: f"Na koncie zostało mi tylko {price()}!",
        lambda: f"Zwrot podatku PIT wyniesie {price()}.",
        lambda: f"Do ceny trzeba doliczyć {rng.choice([5, 8, 23])}% VAT.",
        lambda: f"Inflacja spadła do {rng.randint(1, 15)},{rng.randint(1, 9)}%.",
        lambda: f"Cena wzrosła o {rng.choice([3, 5, 7, 10, 12, 15, 20, 25, 30])}% w ciągu roku.",
        # counts and units
        lambda: f"W magazynie zostało {items()} i {items()}.",
        lambda: f"Na koncert przyszło {rng.randint(2, 99) * 100} osób.",
        lambda: f"Na autostradzie można jechać {rng.choice([110, 120, 130, 140])} km/h.",
        lambda: f"Rano było {rng.randint(1, 30)}°C, a wieczorem tylko {rng.randint(1, 15)}°C.",
        lambda: f"Do przepisu potrzeba {rng.randint(2, 900)} g mąki i {rng.randint(2, 6)} jajek.",
        lambda: f"Plecak waży {rng.randint(2, 25)} kg, a namiot {rng.randint(2, 9)} kg.",
        lambda: f"Z {rng.choice(CITIES)} do morza jest około {rng.randint(40, 700)} km.",
        lambda: f"Pokój ma {rng.randint(8, 60)} m² i duże okno.",
        lambda: f"Rzeka ma tu {rng.randint(20, 900)} m szerokości.",
        lambda: f"Miasto liczy {rng.randint(10, 999)} tys. mieszkańców.",
        # abbreviations and acronyms
        lambda: f"Prof. {person()} poprowadzi dziś wykład.",
        lambda: f"Dr {person()} przyjmuje od {hour()}.",
        lambda: f"Na spotkaniu był m.in. {person()}.",
        lambda: f"Kup warzywa, owoce, pieczywo itd.",
        lambda: f"Wrócę późno, tzn. po {hour()}.",
        lambda: f"Biuro mieści się przy ul. {rng.choice(STREETS)} {rng.randint(1, 150)}.",
        lambda: f"Mieszkanie nr {rng.randint(1, 120)} jest na trzecim piętrze.",
        lambda: f"Formularz z {rng.choice(ACRONYMS)} trzeba wysłać do {date()}.",
        lambda: f"Dzwoniłem do {rng.choice(ACRONYMS)} i do {rng.choice(ACRONYMS)}, ale nikt nie odebrał.",
        lambda: f"Czy {rng.choice(ACRONYMS)} odpowiedział już na pismo?",
        lambda: f"Pociąg PKP do {rng.choice(CITIES)} ma {rng.randint(5, 90)} minut opóźnienia.",
        # Roman numerals
        lambda: f"{rng.choice(KINGS)} to postać, o której uczymy się w szkole.",
        lambda: f"Ten kościół zbudowano w {rng.choice(CENTURIES)} wieku.",
        lambda: f"W {rng.choice(CENTURIES)} w. miasto bardzo się rozrosło.",
        # phone numbers and postal codes
        lambda: f"Zadzwoń pod numer {phone()}.",
        lambda: f"Mój nowy numer to {phone()}, zapisz go!",
        lambda: f"Kod pocztowy to {postal()} {rng.choice(['Warszawa', 'Kraków', 'Gdańsk', 'Poznań', 'Lublin'])}.",
        lambda: f"Paczkę wyślij na ul. {rng.choice(STREETS)} {rng.randint(1, 90)}, {postal()} {rng.choice(['Łódź', 'Toruń', 'Opole'])}.",
        # questions and exclamations with numbers
        lambda: f"Naprawdę zapłaciłeś {price()}?",
        lambda: f"Jak to, już {hour()}?",
        lambda: f"Ile osób przyjdzie {date()}?",
        lambda: f"Wygraliśmy {rng.randint(2, 9)} do {rng.randint(0, 1)}!",
        lambda: f"Brawo, {rng.randint(90, 100)} punktów na 100!",
        lambda: f"{rng.choice(FIRST)}, masz {rng.choice([n for n in MANY if n >= 20])} lat, a zachowujesz się jak dziecko!",
    ]


def spoken_ok(written):
    """The spoken form, or None when the front end does not read the written form plainly."""
    spoken = normalize(written)
    if re.search(r"\d|[$€%°²]", spoken) or re.search(r"\b\w+\.\s+[a-ząćęłńóśźż]", spoken) or not front_end(spoken):
        return None
    return spoken


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--templates", type=int, default=900)
    parser.add_argument("--new", type=int, default=1500)
    parser.add_argument("--intonation", type=int, default=650, help="Common Voice questions and exclamations (CC0)")
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows, seen = [], set()

    def add(written, spoken, category, origin, url, license_="own"):
        if spoken and spoken.lower() not in seen:
            seen.add(spoken.lower())
            rows.append({"id": "h_" + hashlib.sha1(spoken.encode()).hexdigest()[:14], "text": spoken, "written": written,
                         "category": category, "origin": origin, "source_url": url, "source_license": license_})

    for written in templates(rng, args.templates * 3):
        if sum(r["origin"] == "slyszeglosy_templates" for r in rows) >= args.templates:
            break
        add(written, spoken_ok(written), "numbers", "slyszeglosy_templates", "scripts/build_corpus_pl.py")
    patterns = new_patterns(rng)
    attempts = 0
    while sum(r["origin"] == "slyszeglosy_v2_templates" for r in rows) < args.new and attempts < args.new * 20:
        attempts += 1
        written = rng.choice(patterns)()
        add(written, spoken_ok(written), "hard", "slyszeglosy_v2_templates", "scripts/v2/hard_texts.py")
    v1 = [json.loads(line) for line in args.v1_corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    intonation = [r for r in v1 if r["category"] in ("question", "exclamation") and r["origin"].startswith("common_voice")]
    rng.shuffle(intonation)
    for r in intonation[:args.intonation]:
        add(r["text"], r["text"], r["category"], r["origin"], r["source_url"], r["source_license"])
    rng.shuffle(rows)
    hard = [r for r in rows if not r["origin"].startswith("common_voice")]
    held = {r["id"]: "test_hard" for r in hard[:150]}
    validation = [r["id"] for r in rows if r["id"] not in held][:100]
    held.update({i: "validation" for i in validation})
    for r in rows:
        r["split"] = held.get(r["id"], "train")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts = {}
    for r in rows:
        counts[f"{r['origin']}/{r['split']}"] = counts.get(f"{r['origin']}/{r['split']}", 0) + 1
    print(json.dumps({"rows": len(rows), "characters": sum(len(r["text"]) for r in rows), **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
