"""Which recogniser label best matches each of our phones, measured on audio.

The forced aligner (wav2vec2-lv-60-espeak-cv-ft) was fine-tuned on Common
Voice with eSpeak transcriptions made in 2021. Its Polish labels are not the
ones today's eSpeak prints (it hears ɨ as "y", ɛ as "e", dʑ as "dʒ"), so the
label map in tiny_voca.align is chosen from data, not from eSpeak's docs:

1. Align every recording with the current map.
2. For each phone occurrence, score every plain IPA label of the recogniser
   by its mean log-probability over the phone's span.
3. Map each phone to the label with the best mean score; repeat from 1.

Labels may coincide (ɕ and ʂ can both be "ʃ"): alignment only needs the
order of the phones, and CTC keeps repeated labels apart with a blank.

    PYTHONPATH=src python scripts/calibrate_aligner_labels.py data/bakeoff/*/manifest.jsonl --output aligner_labels.json
"""

import argparse
import collections
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf

from tiny_voca import align
from tiny_voca.align import PhoneRecognizer, token_spans
from tiny_voca.text import PHONE_SET, PHONES, UNK, phonemize

PLAIN = re.compile(r"^[^\d\[\]^.\"<>:!?A-Z]+$")      # no tone digits, no ASCII-encoded labels, no specials


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recognizer = PhoneRecognizer("cuda", local_files_only=False)
    vocabulary = {label: index for label, index in recognizer.vocabulary.items() if PLAIN.match(label) and index != recognizer.blank}
    candidates = np.array(sorted(vocabulary.values()))
    names = {index: label for label, index in recognizer.vocabulary.items()}
    clips = []
    for manifest in args.manifests:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("audio"):
                wave, rate = sf.read(manifest.parent / row["audio"], dtype="float32")
                symbols = [s for s in phonemize(row["text"]).symbols if s in PHONE_SET or s == UNK]
                clips.append((symbols, recognizer.log_probs(wave, rate)))
    print(f"{len(clips)} recordings")
    mapping = dict(align.ALIGNER_LABEL_FALLBACK)
    history = []
    for iteration in range(args.iterations):
        totals = collections.defaultdict(lambda: np.zeros(len(candidates)))
        current = collections.defaultdict(list)
        counts = collections.Counter()
        failed = 0
        for symbols, log_probs in clips:
            labels = [recognizer.vocabulary[mapping.get(s, s)] for s in symbols]
            try:
                spans = token_spans(log_probs, labels, recognizer.blank)
            except ValueError:
                failed += 1
                continue
            for symbol, (a, b), label in zip(symbols, spans, labels):
                totals[symbol] += log_probs[a:b + 1][:, candidates].mean(0)
                current[symbol].append(float(np.exp(log_probs[a:b + 1, label]).mean()))
                counts[symbol] += 1
        table, changed = {}, 0
        for symbol in PHONES:
            if not counts[symbol]:
                continue
            means = totals[symbol] / counts[symbol]
            order = np.argsort(-means)[:3]
            best = names[int(candidates[order[0]])]
            table[symbol] = {"n": counts[symbol], "label": mapping.get(symbol, symbol),
                             "probability": round(float(np.mean(current[symbol])), 3), "best": best,
                             "top3": [names[int(candidates[i])] for i in order]}
            if best != mapping.get(symbol, symbol):
                changed += 1
        weighted = sum(t["probability"] * t["n"] for t in table.values()) / sum(t["n"] for t in table.values())
        history.append({"iteration": iteration, "mean_phone_probability": round(weighted, 4), "failed": failed, "changed": changed})
        print(json.dumps(history[-1]))
        for symbol, row in table.items():
            if row["best"] == symbol:
                mapping.pop(symbol, None)
            else:
                mapping[symbol] = row["best"]
        if not changed:
            break
    unseen = [p for p in PHONES if p not in table]
    result = {"recordings": len(clips), "history": history, "phones": table, "unseen_phones": unseen,
              "label_map": {k: v for k, v in sorted(mapping.items()) if k in PHONE_SET}}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for symbol, row in sorted(table.items(), key=lambda item: item[1]["probability"]):
        print(f"{symbol:4s} n={row['n']:4d} label={row['label']:4s} p={row['probability']:.2f} best={row['best']:4s} top3={row['top3']}")
    print("unseen:", unseen)
    print("label map:", json.dumps(result["label_map"], ensure_ascii=False))


if __name__ == "__main__":
    main()
