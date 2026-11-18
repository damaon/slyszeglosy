"""V2 phases D and E: the MC Speech voice reads the hard-case texts with VoxCPM2 (openbmb/VoxCPM2,
Apache-2.0), cloned from one MC Speech clip and its transcript (the same clip as prompt and
reference, as the VoxCPM README recommends for similarity).

The teacher reads the spoken form ("text", from wymowa.normalize), so its audio always
matches the phonemes the front end makes. The audio is resampled to 24 kHz, trimmed and
levelled like the MC Speech clips (scripts/v2/mc_extract.py), so both sources sound alike
to the model. Records have the fields of scripts/generate_audio_pl.py; resumable; pass 2
(--retry) regenerates clips that scripts/v2/screen.py rejected, once, with another seed.

    ~/.venvs/voxcpm/bin/python scripts/v2/generate_voxcpm.py --texts T.jsonl --reference R.json --root OUT [--limit N]
    ~/.venvs/voxcpm/bin/python scripts/v2/generate_voxcpm.py ... --retry
R.json: {"audio": "/path/clip.flac", "text": "exact transcript"}
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))
from mc_extract import RATE, level                     # noqa: E402
from tiny_voca.thermal import Thermostat               # noqa: E402

MODEL = "openbmb/VoxCPM2"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def tts_text(spoken):
    """Letter names of acronyms come joined ("pe|ka|ˈpe"); the teacher gets plain words."""
    return " ".join(spoken.replace("|", " ").replace("ˈ", "").split())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--splits", nargs="+", help="Only texts of these splits")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-temp", type=int, default=0)
    args = parser.parse_args()
    thermostat = Thermostat(args.max_temp, log=lambda m: print(json.dumps({"thermal": m}, ensure_ascii=False), flush=True))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    rows = [r for r in load_jsonl(args.texts) if not args.splits or r["split"] in args.splits][:args.limit]
    args.root.mkdir(parents=True, exist_ok=True)
    generated = {}
    for record in load_jsonl(args.root / "generated-voxcpm.jsonl"):
        if record["id"] not in generated or record["attempt"] >= generated[record["id"]]["attempt"]:
            generated[record["id"]] = record
    if args.retry:
        screened = {r["id"]: r for r in load_jsonl(args.root / "screened.jsonl")}
        wanted = {cid for cid, r in screened.items() if r["status"] != "accepted" and generated.get(cid, {}).get("attempt", 1) < 2}
        jobs, attempt = [r for r in rows if r["id"] in wanted], 2
    else:
        jobs, attempt = [r for r in rows if r["id"] not in generated], 1
    print(json.dumps({"attempt": attempt, "jobs": len(jobs)}), flush=True)
    if not jobs:
        return
    from voxcpm import VoxCPM
    import voxcpm
    version = getattr(voxcpm, "__version__", None)
    if version is None:
        import importlib.metadata
        version = importlib.metadata.version("voxcpm")
    model = VoxCPM.from_pretrained(MODEL, load_denoiser=False)
    rate = model.tts_model.sample_rate
    audio_dir = args.root / "audio"
    audio_dir.mkdir(exist_ok=True)
    started, speech_seconds = time.time(), 0.0
    with (args.root / "generated-voxcpm.jsonl").open("a", encoding="utf-8") as log:
        for index, row in enumerate(jobs):
            started += thermostat()
            seed = args.seed + index + attempt * 1_000_003
            torch.manual_seed(seed)
            np.random.seed(seed % 2**32)
            begin = time.time()
            wave = model.generate(text=tts_text(row["text"]), prompt_wav_path=reference["audio"], prompt_text=reference["text"],
                                  reference_wav_path=reference["audio"], cfg_value=args.cfg, inference_timesteps=args.steps,
                                  normalize=False, denoise=False, retry_badcase=True)
            seconds = time.time() - begin
            wave = np.asarray(wave, dtype=np.float32).reshape(-1)
            trimmed, stats = level(wave, rate) if wave.size and np.isfinite(wave).all() else (None, "invalid")
            path = audio_dir / f"{row['id']}.flac"
            if trimmed is not None:
                sf.write(path, trimmed, RATE, subtype="PCM_16")
            record = {"id": row["id"], "text_id": row["id"], "voice_id": "MC", "text": row["text"], "written": row.get("written"),
                      "split": row["split"], "category": row["category"], "origin": row["origin"],
                      "source_id": row["id"], "source_license": row.get("source_license"), "source_url": row.get("source_url"),
                      "teacher_model": f"{MODEL} (voxcpm {version})", "reference_audio": Path(reference["audio"]).name,
                      "audio": f"audio/{row['id']}.flac" if trimmed is not None else None, "generation_seed": seed,
                      "attempt": attempt, "generation_seconds": round(seconds, 3), "cfg": args.cfg, "steps": args.steps,
                      **(stats if trimmed is not None else {"duration_seconds": 0.0, "error": stats})}
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            speech_seconds += record["duration_seconds"]
            if index % 25 == 0:
                log.flush()
                elapsed = time.time() - started
                print(json.dumps({"done": index + 1, "of": len(jobs), "audio_hours": round(speech_seconds / 3600, 3),
                                  "rtf": round(elapsed / max(speech_seconds, 1e-6), 3),
                                  "eta_minutes": round(elapsed / (index + 1) * (len(jobs) - index - 1) / 60)}), flush=True)


if __name__ == "__main__":
    main()
