"""Screen generated recordings with Whisper; write the prepare.py manifests.

For each latest attempt of a clip in <root>/generated-*.jsonl that has not
been screened yet:
* reject silent, non-finite, too short (<0.6 s) or too long (>35 s) audio,
  more than 80% near-silence, or an implausible rate (outside 6-25 normalised
  characters per second);
* transcribe with Whisper large-v3-turbo (language pl) and reject above 5%
  character error rate. Both sides go through wymowa.normalize. When Whisper
  writes digits, case endings of numbers can differ ("15 stopni" vs
  "piętnastu"), so the limit is then 12%.

Results go to <root>/screened.jsonl (appended; the latest line per clip wins).
With --finalize, <root>/manifest-<split>.jsonl lists the accepted clips in the
form tiny_voca.prepare reads (audio relative to <root>).

Run in the main environment (scripts/setup_env.sh):
    PYTHONPATH=src python scripts/screen_audio_pl.py --root data/gen [--finalize]
"""

import argparse
import collections
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf

from wymowa.normalize import normalize

CER_LIMIT, CER_LIMIT_DIGITS = 0.05, 0.12


def plain(text):
    text = normalize(text).replace("|", "").replace("ˈ", "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def load(root, pattern):
    latest = {}
    for path in sorted(root.glob(pattern)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record["id"] not in latest or record.get("attempt", 1) >= latest[record["id"]].get("attempt", 1):
                    latest[record["id"]] = record
    return latest


def finalize(root, generated, screened):
    counts = collections.Counter()
    by_split = collections.defaultdict(list)
    for cid, record in sorted(generated.items()):
        screen = screened.get(cid)
        if screen and screen["status"] == "accepted" and screen["attempt"] == record["attempt"]:
            by_split[record["split"]].append({**record, "normalized_cer": screen["cer"], "asr_transcript": screen["transcript"]})
            counts[record["split"]] += 1
    for split, rows in by_split.items():
        (root / f"manifest-{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    hours = {s: round(sum(r["duration_seconds"] for r in rows) / 3600, 2) for s, rows in by_split.items()}
    rejected = collections.Counter(reason for cid, s in screened.items() if s["status"] != "accepted"
                                   and generated.get(cid, {}).get("attempt") == s["attempt"] for reason in s["reasons"])
    summary = {"accepted": dict(counts), "hours": hours, "total": len(generated),
               "rejected_final": sum(1 for cid, s in screened.items() if s["status"] != "accepted"), "reasons": dict(rejected)}
    (root / "screening-summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--whisper", default="large-v3-turbo")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    generated = load(args.root, "generated-*.jsonl")
    screened = load(args.root, "screened.jsonl")
    todo = [r for cid, r in sorted(generated.items()) if cid not in screened or screened[cid]["attempt"] != r["attempt"]]
    print(json.dumps({"generated": len(generated), "to_screen": len(todo)}), flush=True)
    if todo:
        import torch  # noqa: F401  loads the CUDA libraries (cuBLAS, cuDNN) that ctranslate2 needs
        from faster_whisper import BatchedInferencePipeline, WhisperModel
        import jiwer
        from scipy.signal import resample_poly
        asr = BatchedInferencePipeline(WhisperModel(args.whisper, device="cuda", compute_type="float16"))
        with (args.root / "screened.jsonl").open("a", encoding="utf-8") as out:
            for index, record in enumerate(todo):
                reasons, transcript, cer = [], "", 1.0
                wave = None
                if record.get("audio"):
                    wave, rate = sf.read(args.root / record["audio"], dtype="float32")
                if wave is None or not wave.size or not np.isfinite(wave).all() or np.abs(wave).max() < 1e-5:
                    reasons.append("invalid_or_silent_audio")
                else:
                    seconds = len(wave) / rate
                    reference = plain(record["text"])
                    if not 0.6 <= seconds <= 35:
                        reasons.append("duration")
                    if not 6 <= len(reference) / seconds <= 25:
                        reasons.append("speaking_rate")
                    if float(np.mean(np.abs(wave) < 1e-4)) > 0.8:
                        reasons.append("excessive_silence")
                    if not reasons:
                        segments, _ = asr.transcribe(resample_poly(wave, 2, 3), language="pl", batch_size=args.batch,
                                                     beam_size=1, without_timestamps=True)
                        transcript = " ".join(s.text.strip() for s in segments).strip()
                        cer = jiwer.cer(reference, plain(transcript) or "<pusto>")
                        limit = CER_LIMIT_DIGITS if re.search(r"\d", transcript) else CER_LIMIT
                        if cer > limit:
                            reasons.append("asr_mismatch")
                result = {"id": record["id"], "attempt": record["attempt"], "status": "rejected" if reasons else "accepted",
                          "reasons": reasons, "cer": round(float(cer), 4), "transcript": transcript}
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                screened[record["id"]] = result
                if index % 500 == 0:
                    out.flush()
                    print(json.dumps({"screened": index + 1, "of": len(todo)}), flush=True)
    if args.finalize:
        finalize(args.root, generated, screened)
    else:
        status = collections.Counter(s["status"] for cid, s in screened.items() if generated.get(cid, {}).get("attempt") == s["attempt"])
        print(json.dumps(dict(status)))


if __name__ == "__main__":
    main()
