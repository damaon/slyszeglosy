"""V2 phase B: how the front end reads the MC Speech transcripts.

Reports texts the front end cannot pronounce, texts with digits, abbreviations,
Roman numerals and foreign letters (q, v, x), the words most often missing from
the lexicon-free rules, and phone / phone-pair coverage compared with the V1 corpus.

    PYTHONPATH=src python scripts/v2/text_report.py --records ~/sg-data/v2/mc/generated-mc.jsonl \
        --v1 data/corpus/scripts.jsonl --output ~/sg-data/v2/text-report.json
"""

import argparse
import collections
import json
import re
from pathlib import Path

from tiny_voca.text import PHONE_SET, NothingToPronounce, phonemize

ABBREVIATION = re.compile(r"\b[a-ząćęłńóśźż]{1,4}\.(?=\s+[a-ząćęłńóśźż0-9])")
ROMAN = re.compile(r"\b[IVXLC]{2,}\b")
FOREIGN = re.compile(r"[qvxQVX]")


def coverage(texts):
    phones, pairs, failed = collections.Counter(), collections.Counter(), []
    for text in texts:
        try:
            phonemized = phonemize(text)
        except (NothingToPronounce, ValueError) as error:
            failed.append({"text": text, "error": str(error)[:120]})
            continue
        if phonemized.unknown:
            failed.append({"text": text, "error": f"unknown phones {phonemized.unknown}"})
        symbols = [s for s in phonemized.symbols if s in PHONE_SET]
        phones.update(symbols)
        pairs.update(zip(symbols, symbols[1:]))
    return phones, pairs, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    texts = sorted({json.loads(line)["text"] for line in args.records.read_text(encoding="utf-8").splitlines()})
    v1 = sorted({json.loads(line)["text"] for line in args.v1.read_text(encoding="utf-8").splitlines()})
    phones, pairs, failed = coverage(texts)
    v1_phones, v1_pairs, _ = coverage(v1)
    digits = [t for t in texts if re.search(r"\d", t)]
    abbreviations = collections.Counter(m.group(0) for t in texts for m in ABBREVIATION.finditer(t))
    romans = collections.Counter(m.group(0) for t in texts for m in ROMAN.finditer(t))
    foreign = [t for t in texts if FOREIGN.search(t)]
    questions = sum(t.rstrip(" \"'”»)").endswith("?") for t in texts)
    exclamations = sum(t.rstrip(" \"'”»)").endswith("!") for t in texts)
    missing_pairs = {f"{a} {b}": n for (a, b), n in v1_pairs.most_common() if pairs[(a, b)] < 3 and n >= 20}
    report = {
        "texts": len(texts), "unpronounceable": failed[:50], "unpronounceable_count": len(failed),
        "with_digits": len(digits), "digit_examples": digits[:20],
        "abbreviations": dict(abbreviations.most_common(40)), "roman_numerals": dict(romans.most_common(20)),
        "foreign_letters": len(foreign), "foreign_examples": foreign[:20],
        "questions": questions, "exclamations": exclamations,
        "phones_missing_vs_v1": sorted(set(v1_phones) - set(phones)),
        "rare_phones": {p: n for p, n in sorted(phones.items(), key=lambda x: x[1])[:12]},
        "phone_pairs": len(pairs), "phone_pairs_v1": len(v1_pairs),
        "pairs_common_in_v1_rare_here": dict(list(missing_pairs.items())[:40]),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, (list, dict)) or k in
                      ("phones_missing_vs_v1", "rare_phones")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
