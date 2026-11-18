"""V2 screening with two ASR engines: reject a clip when BOTH Whisper
large-v3-turbo and Parakeet-TDT-0.6B-v3 hear something other than the text (word rule:
any difference after wymowa normalisation, spaces ignored). Plus the V1 audio rules
(duration 0.6-35 s, 6-25 normalised characters per second, not mostly silence).

    PYTHONPATH=src:scripts python scripts/v2/screen.py sets --root R          # R/asr/sets.jsonl: clips still to screen
    ~/.venvs/sg/bin/python       scripts/asrtest/run_asr.py --engine whisper  --sets R/asr/sets.jsonl
    ~/.venvs/parakeet/bin/python scripts/asrtest/run_asr.py --engine parakeet --sets R/asr/sets.jsonl
    PYTHONPATH=src:scripts python scripts/v2/screen.py decide --root R [--finalize]

R holds generated-*.jsonl records (scripts/v2/mc_extract.py, scripts/v2/generate_voxcpm.py).
Results go to R/screened.jsonl (latest line per clip wins); --finalize writes
R/manifest-<split>.jsonl in the form tiny_voca.prepare reads, and R/screening-summary.json.
"""

import argparse
import collections
import json
import re
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf

from screen_audio_pl import finalize, load, plain


def compact(text):
    return plain(text).replace(" ", "")


def audio_reasons(path, reference):
    wave, rate = sf.read(path, dtype="float32")
    if not wave.size or not np.isfinite(wave).all() or np.abs(wave).max() < 1e-5:
        return ["invalid_or_silent_audio"]
    reasons, seconds = [], len(wave) / rate
    if not 0.6 <= seconds <= 35:
        reasons.append("duration")
    if not 6 <= len(reference) / seconds <= 25:
        reasons.append("speaking_rate")
    if float(np.mean(np.abs(wave) < 1e-4)) > 0.8:
        reasons.append("excessive_silence")
    return reasons


def todo(root):
    generated, screened = load(root, "generated-*.jsonl"), load(root, "screened.jsonl")
    return generated, screened, [r for cid, r in sorted(generated.items())
                                 if r.get("audio") and (cid not in screened or screened[cid]["attempt"] != r["attempt"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("sets", "decide"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    asr = args.root / "asr"
    asr.mkdir(exist_ok=True)
    generated, screened, rows = todo(args.root)
    if args.stage == "sets":
        (asr / "sets.jsonl").write_text("".join(json.dumps({"id": r["id"], "attempt": r["attempt"], "audio": str(args.root / r["audio"])})
                                                + "\n" for r in rows), encoding="utf-8")
        print(json.dumps({"generated": len(generated), "to_screen": len(rows)}))
        return
    sets = {r["id"]: r for r in map(json.loads, (asr / "sets.jsonl").read_text(encoding="utf-8").splitlines())} if rows else {}
    heard = {}
    for engine in ("whisper", "parakeet"):
        path = asr / f"{engine}.jsonl"
        if rows:
            heard[engine] = {r["id"]: r["transcript"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    counts = collections.Counter()
    with (args.root / "screened.jsonl").open("a", encoding="utf-8") as out:
        for record in rows:
            if record["id"] not in sets or sets[record["id"]]["attempt"] != record["attempt"] or \
                    any(record["id"] not in h for h in heard.values()):
                counts["not_transcribed"] += 1
                continue
            reference = plain(record["text"])
            reasons = audio_reasons(args.root / record["audio"], reference)
            transcripts = {e: heard[e][record["id"]] for e in heard}
            cers = {e: round(float(jiwer.cer(reference, plain(t) or "<pusto>")), 4) for e, t in transcripts.items()}
            differs = {e: compact(t) != compact(record["text"]) for e, t in transcripts.items()}
            if not reasons and all(differs.values()):
                reasons.append("asr_mismatch")
            result = {"id": record["id"], "attempt": record["attempt"], "status": "rejected" if reasons else "accepted",
                      "reasons": reasons, "cer": cers["parakeet"], "cer_whisper": cers["whisper"],
                      "transcript": transcripts["parakeet"], "transcript_whisper": transcripts["whisper"],
                      "differs": [e for e, d in differs.items() if d]}
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            screened[record["id"]] = result
            counts[result["status"]] += 1
            counts.update(f"reason:{r}" for r in reasons)
            counts.update(f"differs:{e}" for e in result["differs"])
    print(json.dumps(dict(counts)))
    if args.finalize:
        finalize(args.root, generated, screened)


if __name__ == "__main__":
    main()
