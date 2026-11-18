"""V2 phase D texts: the 100 V1 bake-off sentences (scripts/bakeoff/sentences.jsonl) plus 50
held-out hard cases (split test_hard of scripts/v2/hard_texts.py), in the form generate_voxcpm.py reads.

    PYTHONPATH=src python scripts/v2/bakeoff_texts.py --hard ~/sg-data/v2/hard/texts.jsonl --output ~/sg-data/v2/bakeoff/texts.jsonl
"""

import argparse
import json
from pathlib import Path

from wymowa import normalize

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for line in (HERE / "bakeoff" / "sentences.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows.append({"id": f"b_{r['id']}", "text": normalize(r["text"]), "written": r["text"], "split": "bakeoff",
                     "category": r["category"], "origin": "slyszeglosy_bakeoff", "source_url": "scripts/bakeoff/sentences.jsonl",
                     "source_license": "own"})
    hard = [json.loads(line) for line in args.hard.read_text(encoding="utf-8").splitlines()]
    rows += [{**r, "id": "b_" + r["id"], "split": "bakeoff"} for r in hard if r["split"] == "test_hard"][:50]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(len(rows), "texts")


if __name__ == "__main__":
    main()
