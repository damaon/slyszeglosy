"""Speaker similarity to the real MC Speech voice (WavLM-SV cosine to a centroid of real clips).

The centroid comes from --ref-count random MC train clips; the ceiling is the mean similarity of
--ref-count OTHER real clips. Each --root (generated-*.jsonl records) is scored on its clips.

    PYTHONPATH=src python scripts/v2/similarity.py --mc ~/sg-data/v2/mc --roots ~/sg-data/v2/bakeoff/A ~/sg-data/v2/bakeoff/B
"""

import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np
import torch


def records(root):
    latest = {}
    for path in sorted(root.glob("generated-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("audio") and (r["id"] not in latest or r["attempt"] >= latest[r["id"]]["attempt"]):
                latest[r["id"]] = r
    return list(latest.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--ref-count", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from transformers import AutoFeatureExtractor, WavLMForXVector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()

    def embed(path):
        wave, _ = librosa.load(path, sr=16000)
        inputs = fe(wave, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            return torch.nn.functional.normalize(sv(**inputs).embeddings[0], dim=-1).cpu().numpy()

    mc = [r for r in records(args.mc) if r["split"] == "train" and r["duration_seconds"] >= 2]
    sample = random.Random(0).sample(mc, 2 * args.ref_count)
    centroid = np.mean([embed(args.mc / r["audio"]) for r in sample[:args.ref_count]], 0)
    centroid /= np.linalg.norm(centroid)
    report = {"real_mc_ceiling": round(float(np.mean([embed(args.mc / r["audio"]) @ centroid for r in sample[args.ref_count:]])), 4)}
    per_clip = {}
    for root in args.roots:
        scores = {r["id"]: float(embed(root / r["audio"]) @ centroid) for r in records(root)}
        per_clip[str(root)] = scores
        values = np.array(list(scores.values()))
        report[str(root)] = {"clips": len(values), "mean": round(float(values.mean()), 4),
                             "p10": round(float(np.percentile(values, 10)), 4), "min": round(float(values.min()), 4)}
    print(json.dumps(report, indent=1))
    if args.output:
        args.output.write_text(json.dumps({"summary": report, "per_clip": per_clip}, indent=1))


if __name__ == "__main__":
    main()
