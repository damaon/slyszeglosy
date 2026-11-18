"""V2 phase A: MC Speech parquet shards -> 24 kHz mono FLAC clips and generation-style records.

Each clip is resampled to 24 kHz, trimmed to the speech plus 0.15 s on both ends, and
scaled so the speech is at -20 dBFS RMS (peak kept at or below 0.95). The records use the
same fields as scripts/generate_audio_pl.py, so scripts/v2/screen.py can treat them like
generated audio. Splits: 300 validation and 300 test sentences at random (seed 0), the rest
train; identical texts always share a split.

    PYTHONPATH=src python scripts/v2/mc_extract.py --shards ~/sg-data/mc/data --root ~/sg-data/v2/mc
"""

import argparse
import concurrent.futures
import io
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 24000
TARGET_DB, PEAK, PAD = -20.0, 0.95, 0.15
SOURCE_URL = "https://huggingface.co/datasets/czyzi0/the-mc-speech-dataset"


def frame_db(wave, size=1024, hop=256):
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(wave, (0, size)), size)[::hop]
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


def level(wave, rate):
    """Mono float audio at any rate -> (24 kHz audio trimmed to the speech plus PAD, speech at TARGET_DB,
    peak <= PEAK; stats), or (None, reason). Shared by the MC Speech clips and the teacher's clips."""
    import librosa
    raw_peak = float(np.abs(wave).max()) if wave.size else 0.0
    clipped = float(np.mean(np.abs(wave) > 0.999)) if wave.size else 0.0
    if rate != RATE:
        wave = librosa.resample(wave, orig_sr=rate, target_sr=RATE, res_type="soxr_hq")
    db = frame_db(wave)
    if not db.size or db.max() < -90:
        return None, "silent"
    speech = db > db.max() - 40
    first, last = np.argmax(speech), len(speech) - 1 - np.argmax(speech[::-1])
    start = max(0, first * 256 - int(PAD * RATE))
    end = min(len(wave), (last + 1) * 256 + 1024 + int(PAD * RATE))
    trimmed = wave[start:end]
    tdb = frame_db(trimmed)
    active = tdb > tdb.max() - 40
    speech_rms_db = 10 * np.log10(np.mean(10 ** (tdb[active] / 10)))
    noise_db = float(np.percentile(tdb, 5))
    gain = 10 ** ((TARGET_DB - speech_rms_db) / 20)
    peak = float(np.abs(trimmed).max()) * gain
    if peak > PEAK:
        gain *= PEAK / peak
    trimmed = (trimmed * gain).astype(np.float32)
    return trimmed, {"duration_seconds": round(len(trimmed) / RATE, 3), "raw_peak": round(raw_peak, 4),
                     "clipped_fraction": round(clipped, 6), "snr_db": round(float(speech_rms_db - noise_db), 1),
                     "trimmed_seconds": round((len(wave) - len(trimmed)) / RATE, 3), "gain_db": round(float(20 * np.log10(gain)), 1)}


def process(job):
    key, data, out = job
    wave, rate = sf.read(io.BytesIO(data), dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(1)
    trimmed, stats = level(wave, rate)
    if trimmed is None:
        return {"key": key, "error": stats}
    sf.write(out, trimmed, RATE, subtype="PCM_16")
    return {"key": key, **stats}


def category(text):
    end = text.rstrip(" \"'”»)")[-1:]
    return {"?": "question", "!": "exclamation"}.get(end, "statement")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--validation", type=int, default=300)
    parser.add_argument("--test", type=int, default=300)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    audio = args.root / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard in sorted(args.shards.glob("train-*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=256, columns=["id", "transcript"]):
            rows.extend(batch.to_pylist())
    texts = sorted({r["transcript"].strip() for r in rows})
    random.Random(0).shuffle(texts)
    split_of = {t: "validation" for t in texts[:args.validation]}
    split_of.update({t: "test" for t in texts[args.validation:args.validation + args.test]})
    print(json.dumps({"clips": len(rows), "distinct_texts": len(texts)}), flush=True)

    log = args.root / "generated-mc.jsonl"
    done = {json.loads(line)["source_id"] for line in log.read_text(encoding="utf-8").splitlines()} if log.exists() else set()

    def batches():
        """Lists of jobs, a few hundred clips at a time (the executor would read every shard into memory at once)."""
        for shard in sorted(args.shards.glob("train-*.parquet")):
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=384):
                jobs = [(row["id"], row["audio"]["bytes"], audio / f"mc_{row['id'][:16]}.flac")
                        for row in batch.to_pylist() if row["id"] not in done]
                if jobs:
                    yield jobs

    def results(pool):
        for jobs in batches():
            yield from pool.map(process, jobs, chunksize=8)

    texts_by_id = {r["id"]: r["transcript"].strip() for r in rows}
    errors = 0
    with log.open("a", encoding="utf-8") as stream, concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for index, result in enumerate(results(pool)):
            if "error" in result:
                errors += 1
                continue
            key = result.pop("key")
            text = texts_by_id[key]
            record = {"id": f"mc_{key[:16]}", "text_id": f"mc_{key[:16]}", "voice_id": "MC", "text": text,
                      "split": split_of.get(text, "train"), "category": category(text), "origin": "mc_speech",
                      "source_id": key, "source_license": "cc0-1.0", "source_url": SOURCE_URL, "teacher_model": None,
                      "audio": f"audio/mc_{key[:16]}.flac", "attempt": 1, **result}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 2000 == 0:
                stream.flush()
                print(json.dumps({"done": index + 1}), flush=True)
    print(json.dumps({"errors": errors}))


if __name__ == "__main__":
    main()
