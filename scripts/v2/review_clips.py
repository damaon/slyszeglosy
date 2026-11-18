"""Random clips joined into one file for listening (phase A/D review), with their texts.

    python scripts/v2/review_clips.py --root ~/sg-data/v2/mc --count 30 --output data/v2-review/mc_30_losowych
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pattern", default="generated-*.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Path without extension: .wav and .txt are written")
    args = parser.parse_args()
    rows = [json.loads(line) for path in sorted(args.root.glob(args.pattern))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("audio")]
    pick = random.Random(args.seed).sample(rows, min(args.count, len(rows)))
    gap = np.zeros(int(0.7 * 24000), dtype=np.float32)
    parts, lines = [], []
    for n, row in enumerate(pick, 1):
        wave, rate = sf.read(args.root / row["audio"], dtype="float32")
        parts += [wave, gap]
        lines.append(f"{n:2d}. {row['text']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output.with_suffix(".wav"), np.concatenate(parts), rate)
    args.output.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(pick)} clips, {sum(len(p) for p in parts) / rate:.0f} s")


if __name__ == "__main__":
    main()
