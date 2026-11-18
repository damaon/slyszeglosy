"""Test sets for comparing ASR models as TTS validators (ground truth known by construction).

    PYTHONPATH=src python scripts/asrtest/make_sets.py --manifest ~/sg-data/gen/manifest-test.jsonl \
        --gen ~/sg-data/gen --checkpoint weights/inference.pt --output ~/sg-data/asrtest --count 150

Sets (the reference text is always the original sentence):
  teacher     the teacher's recording (OmniVoice), as screened in V1
  ok          our model reading the sentence correctly
  wrong_word  our model reading the sentence with one word swapped for another
  mispron     our model reading the sentence with one phone changed in one word (sz->s, rz->z, ę->e, ...)
An ASR is a good validator when it scores ok/teacher low and wrong_word/mispron high.
"""

import argparse
import json
import random
from pathlib import Path

import soundfile as sf
import torch

from tiny_voca.infer import _voice, limit_peak, load_model
from tiny_voca.text import PHONES, SYMBOL_TO_ID, Phonemized, phonemize

# Near phones a listener would hear as a different word, not as an accent.
SWAP = {"ʂ": "s", "ʐ": "z", "tʂ": "ts", "dʐ": "dz", "ɕ": "ʂ", "ʑ": "ʐ", "tɕ": "tʂ", "dʑ": "dʐ", "ɨ": "i", "i": "ɨ",
        "ɛ̃": "ɛ", "ɔ̃": "ɔ", "r": "l", "w": "l", "x": "k", "ɲ": "n", "v": "f", "b": "p", "d": "t", "ɡ": "k"}
PHONE_SET = frozenset(PHONES)


def words_of(phones):
    """Index ranges of the words in a phonemized sentence (between <sp>, marks and BOS/EOS)."""
    ranges, start = [], None
    for i, symbol in enumerate(phones.symbols):
        is_phone = symbol in PHONE_SET
        if is_phone and start is None:
            start = i
        if not is_phone and start is not None:
            ranges.append((start, i))
            start = None
    return ranges


def mispronounce(phones, rng):
    candidates = []
    for start, end in words_of(phones):
        if end - start >= 4:
            for i in range(start, end):
                if phones.symbols[i] in SWAP:
                    candidates.append(i)
    if not candidates:
        return None, None
    i = rng.choice(candidates)
    symbols = list(phones.symbols)
    before = symbols[i]
    symbols[i] = SWAP[before]
    changed = Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), phones.stress, phones.unknown)
    return changed, f"{before}->{symbols[i]} at {i}"


def swap_word(text, pool, rng):
    words = text.rstrip(".!?…").split(" ")
    choices = [i for i, w in enumerate(words) if len(w.strip(",;:")) >= 4 and w.strip(",;:").isalpha()]
    if not choices:
        return None, None
    i = rng.choice(choices)
    old = words[i].strip(",;:")
    new = rng.choice([w for w in pool if w.lower() != old.lower() and abs(len(w) - len(old)) <= 3] or pool)
    words[i] = words[i].replace(old, new)
    return " ".join(words) + text[len(text.rstrip(".!?…")):], f"{old}->{new}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gen", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_text = {}
    for row in rows:
        by_text.setdefault(row["text_id"], []).append(row)
    text_ids = sorted(by_text)
    rng.shuffle(text_ids)
    pool = sorted({w.strip(",.;:!?…\"'()") for row in rows for w in row["text"].split() if len(w.strip(",.;:!?…\"'()")) >= 4
                   and w.strip(",.;:!?…\"'()").isalpha() and w.strip(",.;:!?…\"'()").islower()})
    model = load_model(args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    rate = model.config.sample_rate
    out_rows = []
    for n, text_id in enumerate(text_ids[:args.count]):
        voice = "F" if n % 2 == 0 else "M"
        record = next((r for r in by_text[text_id] if r["voice_id"] == voice), by_text[text_id][0])
        text = record["text"]
        out_rows.append({"id": f"teacher_{record['id']}", "set": "teacher", "ref": text, "audio": str(args.gen / record["audio"]), "detail": ""})
        phones = phonemize(text)
        variants = [("ok", phones, "")]
        changed, detail = mispronounce(phones, rng)
        if changed is not None:
            variants.append(("mispron", changed, detail))
        swapped, swap_detail = swap_word(text, pool, rng)
        if swapped is not None:
            variants.append(("wrong_word", phonemize(swapped), swap_detail))
        for name, ph, detail in variants:
            wave, _ = _voice(model, ph, voice, 1.0, 0.0, 0, 1.0)
            path = args.output / name / f"{text_id}_{voice}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, limit_peak(wave).numpy(), rate)
            out_rows.append({"id": f"{name}_{text_id}_{voice}", "set": name, "ref": text, "audio": str(path), "detail": detail})
    (args.output / "sets.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    counts = {}
    for r in out_rows:
        counts[r["set"]] = counts.get(r["set"], 0) + 1
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
