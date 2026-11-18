"""CER per system from the Parakeet transcripts of the voice-change eval set.

    python scripts/asrtest/run_asr.py --engine parakeet --sets data/voicemorph/eval/sets.jsonl   # ~/.venvs/parakeet
    PYTHONPATH=src:scripts python scripts/voicemorph/cer.py --work data/voicemorph
"""

import argparse
import json
import statistics
from pathlib import Path

import jiwer

from screen_audio_pl import CER_LIMIT, plain

parser = argparse.ArgumentParser()
parser.add_argument("--work", type=Path, default=Path("data/voicemorph"))
d = parser.parse_args().work / "eval"
rows = {r["id"]: r for r in map(json.loads, (d / "sets.jsonl").read_text(encoding="utf-8").splitlines())}
out = {}
for line in (d / "parakeet.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    row = rows[r["id"]]
    out.setdefault(row["system"], []).append(jiwer.cer(plain(row["ref"]), plain(r["transcript"]) or "<pusto>"))
report = {s: {"CER": round(100 * statistics.mean(v), 2), "over_5%": sum(c > CER_LIMIT for c in v)} for s, v in out.items()}
for s, v in report.items():
    print(f"{s:10s} CER {v['CER']:.2f}%  >5%: {v['over_5%']}")
(d / "cer.json").write_text(json.dumps(report, indent=1))
