"""Choose one designed reference voice per gender from the scored candidates.

Rule: pitch in a plausible adult range, then lowest CER, then the best fit
to our phonemes. Writes chosen.json next to the candidates; edit it by hand
to override the pick after listening.

    python scripts/bakeoff/pick_voices.py data/bakeoff/voices/candidates.scores.jsonl
"""

import json
from pathlib import Path
import sys

PITCH = {"F": (150, 280), "M": (80, 160)}


def main():
    path = Path(sys.argv[1])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = {}
    for voice, (low, high) in PITCH.items():
        pool = [r for r in rows if r["voice"] == voice and r.get("f0_median_hz") and low <= r["f0_median_hz"] <= high]
        pool = pool or [r for r in rows if r["voice"] == voice]
        best = min(pool, key=lambda r: (round(r["cer"], 2), -(r.get("phone_probability") or 0)))
        chosen[voice] = {"audio": best["audio"], "text": best["text"], "candidate": best["id"], "instruct": best["instruct"],
                         "cer": best["cer"], "phone_probability": best.get("phone_probability"), "f0_median_hz": best["f0_median_hz"]}
        print(voice, json.dumps(chosen[voice], ensure_ascii=False))
    (path.parent / "chosen.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
