"""Score the ASR engines on the test sets: false rejections on good audio, detections on bad audio.

    PYTHONPATH=src:scripts python scripts/asrtest/score.py --dir ~/sg-data/asrtest
"""

import argparse
import json
import re
import statistics
from pathlib import Path

import jiwer

from screen_audio_pl import CER_LIMIT, CER_LIMIT_DIGITS, plain

ENGINES = ("whisper", "parakeet", "qwen")
GOOD, BAD = ("teacher", "ok"), ("wrong_word", "mispron")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    rows = {r["id"]: r for r in map(json.loads, (args.dir / "sets.jsonl").read_text(encoding="utf-8").splitlines())}
    results = {}
    for engine in ENGINES:
        path = args.dir / f"{engine}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            ref = plain(rows[r["id"]]["ref"])
            hyp = plain(r["transcript"])
            cer = jiwer.cer(ref, hyp or "<pusto>")
            limit = CER_LIMIT_DIGITS if re.search(r"\d", r["transcript"]) else CER_LIMIT
            results.setdefault(r["id"], {})[engine] = {"cer": cer, "reject": cer > limit, "exact": ref == hyp, "hyp": r["transcript"]}
    engines = [e for e in ENGINES if all(e in v for v in results.values())]
    combos = {e: (lambda v, e=e: v[e]["reject"]) for e in engines}
    if {"parakeet", "qwen"} <= set(engines):
        combos["parakeet OR qwen"] = lambda v: v["parakeet"]["reject"] or v["qwen"]["reject"]
        combos["parakeet AND qwen"] = lambda v: v["parakeet"]["reject"] and v["qwen"]["reject"]
    if set(ENGINES) <= set(engines):
        combos["any of 3"] = lambda v: any(v[e]["reject"] for e in ENGINES)
    report = {}
    for name, rejects in combos.items():
        line = {}
        for s in GOOD + BAD:
            ids = [i for i in results if rows[i]["set"] == s]
            rate = sum(rejects(results[i]) for i in ids) / len(ids)
            line[s] = round(100 * rate, 1)
        report[name] = line
    cers = {e: {s: round(100 * statistics.mean(results[i][e]["cer"] for i in results if rows[i]["set"] == s), 2)
                for s in GOOD + BAD} for e in engines}
    corrected = {e: round(100 * statistics.mean(results[i][e]["exact"] for i in results if rows[i]["set"] == "mispron"), 1)
                 for e in engines}
    examples = []
    for i, v in results.items():
        if rows[i]["set"] == "mispron" and len(examples) < 12 and len({v[e]["reject"] for e in engines}) > 1:
            examples.append({"ref": rows[i]["ref"], "change": rows[i]["detail"], **{e: v[e]["hyp"] for e in engines}})
    print(json.dumps({"reject_percent (good sets: lower is better; bad sets: higher is better)": report,
                      "mean_cer_percent": cers, "mispron_transcribed_as_original_percent": corrected,
                      "disagreements_on_mispron": examples}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
