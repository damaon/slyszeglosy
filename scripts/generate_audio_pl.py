"""Generate the Polish recordings with the teacher (OmniVoice), two voices.

Same recipe as the teacher bake-off (scripts/bakeoff/): OmniVoice 0.2.1 clones the
two reference voices in voices/, language "pl", default decoding (32 steps).
Batches of texts with similar length go through the model together.

Pass 1 generates every (script, voice) pair. scripts/screen_audio_pl.py then
scores the clips with Whisper; pass 2 (--retry) regenerates the rejected ones
once with another seed, as tiny-voca did. Resumable: <root>/generated-<shard>.jsonl
records every clip, and a rerun skips what is there. Several processes can
share the GPU with --shard k --shards n.

Run in the teacher environment (scripts/bakeoff/setup.sh):
    python scripts/generate_audio_pl.py --scripts data/corpus/scripts.jsonl --voices voices/voices.json --root data/gen
    python scripts/generate_audio_pl.py ... --retry          # after screening
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

MODEL = "k2-fsa/OmniVoice"
RATE = 24000


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scripts", type=Path, required=True)
    parser.add_argument("--voices", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--retry", action="store_true", help="Regenerate clips that screening rejected (attempt 2)")
    parser.add_argument("--limit", type=int, help="Only the first N scripts (smoke test)")
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--shard", type=int, default=0, help="This process takes every --shards-th job, from this one")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-temp", type=int, default=0, help="Pause between batches while the GPU is at or above this °C (0: off)")
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tiny_voca.thermal import Thermostat
    thermostat = Thermostat(args.max_temp, log=lambda message: print(json.dumps({"thermal": message}, ensure_ascii=False), flush=True))
    from omnivoice import OmniVoice
    voices = json.loads(args.voices.read_text(encoding="utf-8"))
    rows = load_jsonl(args.scripts)[:args.limit]
    args.root.mkdir(parents=True, exist_ok=True)
    log_path = args.root / f"generated-{args.shard}.jsonl"
    generated = {}
    for path in sorted(args.root.glob("generated-*.jsonl")):
        for record in load_jsonl(path):
            if record["id"] not in generated or record["attempt"] >= generated[record["id"]]["attempt"]:
                generated[record["id"]] = record    # the last attempt of a clip wins
    if args.retry:
        screened = {r["id"]: r for r in load_jsonl(args.root / "screened.jsonl")}
        wanted = {cid for cid, r in screened.items() if r["status"] != "accepted" and generated.get(cid, {}).get("attempt", 1) < 2}
        attempt = 2
    else:
        wanted, attempt = None, 1
    jobs = [{"id": f"{row['id']}_{voice}", "voice": voice, "row": row} for row in rows for voice in voices]
    jobs = [j for j in jobs if (j["id"] in wanted if wanted is not None else j["id"] not in generated)]
    jobs.sort(key=lambda j: (len(j["row"]["text"]), j["id"]))
    jobs = jobs[args.shard::args.shards]
    print(json.dumps({"attempt": attempt, "jobs": len(jobs)}), flush=True)
    if not jobs:
        return
    model = OmniVoice.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.float16)
    prompts = {v: model.create_voice_clone_prompt(ref_audio=str(args.voices.parent / r["audio"]), ref_text=r["text"])
               for v, r in voices.items()}
    audio_dir = args.root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    started, done_seconds = time.time(), 0.0
    with log_path.open("a", encoding="utf-8") as log:
        for start in range(0, len(jobs), args.batch):
            started += thermostat()                 # time spent cooling is not generation time
            batch = jobs[start:start + args.batch]
            seed = args.seed + start + attempt * 1_000_003
            torch.manual_seed(seed)
            begin = time.time()
            waves = model.generate(text=[j["row"]["text"] for j in batch], language="pl",
                                   voice_clone_prompt=[prompts[j["voice"]] for j in batch])
            seconds = time.time() - begin
            for job, wave in zip(batch, waves):
                wave = np.asarray(wave, dtype=np.float32)
                peak = float(np.abs(wave).max()) if wave.size else 0.0
                path = audio_dir / f"{job['id']}.flac"
                if peak > 0 and np.isfinite(wave).all():
                    sf.write(path, wave * (0.95 / peak), RATE, subtype="PCM_16")
                row = job["row"]
                record = {"id": job["id"], "text_id": row["id"], "voice_id": job["voice"], "text": row["text"],
                          "split": row["split"], "category": row["category"], "origin": row["origin"],
                          "source_id": row["source_id"], "source_license": row["source_license"],
                          "source_url": row["source_url"], "teacher_model": f"{MODEL} 0.2.1",
                          "reference_audio": voices[job["voice"]]["audio"], "audio": f"audio/{job['id']}.flac" if peak > 0 else None,
                          "duration_seconds": round(len(wave) / RATE, 3), "generation_seed": seed, "attempt": attempt,
                          "batch_generation_seconds": round(seconds, 3)}
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                done_seconds += record["duration_seconds"]
            log.flush()
            if (start // args.batch) % 20 == 0:
                elapsed = time.time() - started
                print(json.dumps({"done": start + len(batch), "of": len(jobs), "audio_hours": round(done_seconds / 3600, 2),
                                  "rtf": round(elapsed / max(done_seconds, 1e-6), 3),
                                  "eta_minutes": round(elapsed / (start + len(batch)) * (len(jobs) - start - len(batch)) / 60)}),
                      flush=True)


if __name__ == "__main__":
    main()
