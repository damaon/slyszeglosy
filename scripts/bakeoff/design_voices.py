"""Candidate reference voices, designed by OmniVoice from attributes only.

No recording of a real person is involved: each candidate is generated from an
attribute description ("female, young adult, moderate pitch") reading one
fixed Polish text. The chosen pair becomes the reference that both teachers
clone, so the bake-off compares teachers, not voices.

Run in ~/.venvs/omni:
    python scripts/bakeoff/design_voices.py --out data/bakeoff/voices
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

REFERENCE_TEXT = ("Dzień dobry. Dzisiaj opowiem krótko o tym, jak wygląda zwyczajny poranek w naszym mieście: "
                  "ludzie śpieszą się do pracy, a w kawiarniach pachnie świeżą kawą.")
CANDIDATES = {
    "F": ["female, young adult, moderate pitch", "female, middle-aged, moderate pitch", "female, young adult, low pitch"],
    "M": ["male, middle-aged, low pitch", "male, young adult, moderate pitch", "male, middle-aged, moderate pitch"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    from omnivoice import OmniVoice
    args.out.mkdir(parents=True, exist_ok=True)
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    rows = []
    for voice, instructs in CANDIDATES.items():
        for number, instruct in enumerate(instructs):
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                start = time.time()
                audio = np.asarray(model.generate(text=REFERENCE_TEXT, instruct=instruct)[0], dtype=np.float32)
                name = f"{voice}{number}_s{seed}"
                peak = float(np.abs(audio).max()) or 1.0
                sf.write(args.out / f"{name}.wav", audio * (0.95 / peak), 24000, subtype="PCM_16")
                rows.append({"id": name, "voice": voice, "instruct": instruct, "seed": seed, "text": REFERENCE_TEXT,
                             "audio": f"{name}.wav", "seconds": round(len(audio) / 24000, 3),
                             "generation_seconds": round(time.time() - start, 3)})
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (args.out / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
