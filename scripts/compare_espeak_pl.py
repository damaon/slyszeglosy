"""Where wymowa and eSpeak NG (voice pl) disagree, word by word. Reference only.

eSpeak is not a gold standard for Polish: it misses final devoicing after
nasals (ząb -> zɔmb), reads dz as dʑ (dzwon) and voices ć before a vowel
(pięć -> pjɛɲdʑ). The report groups the differences so each kind can be
checked by ear and either fixed in wymowa or written down as eSpeak's.

    python -I scripts/compare_espeak_pl.py --words-from tests/test_wymowa.py --text corpus.txt --output report.json
"""

import argparse
import collections
import difflib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from espeak_pl import ipa                       # noqa: E402
from wymowa.g2p import phrase_phones            # noqa: E402
from wymowa.normalize import normalize          # noqa: E402

# eSpeak's Polish symbols -> ours. Palatalised labials before i are allophones we do not mark.
# A palatalised labial is our consonant + j (pies: pʲɛs = pjɛs); before i the j is implied.
ESPEAK_TO_WYMOWA = {"ʃ": "ʂ", "ʒ": "ʐ", "tʃ": "tʂ", "dʒ": "dʐ", "kʲ": "c", "ɡʲ": "ɟ", "pʲ": "p j", "bʲ": "b j", "mʲ": "m j",
                    "vʲ": "v j", "fʲ": "f j", "ɲʲ": "ɲ", "ç": "x", "ɔː": "ɔ", "aː": "a", "ɛː": "ɛ", "iː": "i", "uː": "u",
                    "ɨː": "ɨ", "e": "ɛ", "o": "ɔ", "g": "ɡ"}
WORD = re.compile(r"[a-ząćęłńóśźż]+", re.IGNORECASE)


def espeak_phones(word):
    out = []
    for unit in ipa(word).replace(" ", "_").split("_"):
        unit = unit.replace("ˈ", "").replace("ˌ", "").strip()
        if unit:
            out += ESPEAK_TO_WYMOWA.get(unit, unit).split()
    return [p for i, p in enumerate(out) if not (p == "j" and i + 1 < len(out) and out[i + 1] == "i")]


def wymowa_phones(word):
    return [phone for item in phrase_phones([word]) for phone, _ in item]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", type=Path, nargs="*", default=[], help="Plain-text files; their words are compared")
    parser.add_argument("--words-from", type=Path, nargs="*", default=[], help="Files whose quoted lowercase words are taken")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    words = collections.Counter()
    for path in args.text:
        words.update(w.lower() for w in WORD.findall(normalize(path.read_text(encoding="utf-8"))))
    for path in args.words_from:
        words.update(re.findall(r'"([a-ząćęłńóśźż]+)":', path.read_text(encoding="utf-8")))
    chosen = [w for w, _ in words.most_common(args.limit)]
    same, phones_total, phones_same = 0, 0, 0
    kinds, examples = collections.Counter(), collections.defaultdict(list)
    for word in chosen:
        ours, theirs = wymowa_phones(word), espeak_phones(word)
        matcher = difflib.SequenceMatcher(a=ours, b=theirs, autojunk=False)
        phones_total += max(len(ours), len(theirs))
        phones_same += sum(block.size for block in matcher.get_matching_blocks())
        if ours == theirs:
            same += 1
            continue
        for tag, a0, a1, b0, b1 in matcher.get_opcodes():
            if tag != "equal":
                kind = f"{' '.join(ours[a0:a1]) or '∅'} -> {' '.join(theirs[b0:b1]) or '∅'}"
                kinds[kind] += 1
                if len(examples[kind]) < 5:
                    examples[kind].append(f"{word}: {' '.join(ours)} | {' '.join(theirs)}")
    report = {"words": len(chosen), "identical_words": same, "word_agreement": round(same / max(len(chosen), 1), 4),
              "phone_agreement": round(phones_same / max(phones_total, 1), 4),
              "differences": [{"wymowa -> espeak": kind, "count": count, "examples": examples[kind]}
                              for kind, count in kinds.most_common()]}
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("words", "identical_words", "word_agreement", "phone_agreement")}))
    for item in report["differences"][:25]:
        print(f"{item['count']:5d}  {item['wymowa -> espeak']:20s}  {item['examples'][0]}")


if __name__ == "__main__":
    main()
