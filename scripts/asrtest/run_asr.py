"""Transcribe the ASR test sets with one engine; writes <output>/<engine>.jsonl.

    python scripts/asrtest/run_asr.py --engine parakeet --sets ~/sg-data/asrtest/sets.jsonl   # ~/.venvs/asr
    python scripts/asrtest/run_asr.py --engine qwen     --sets ~/sg-data/asrtest/sets.jsonl   # ~/.venvs/asr
    python scripts/asrtest/run_asr.py --engine whisper  --sets ~/sg-data/asrtest/sets.jsonl   # ~/.venvs/sg (V1 settings)
"""

import argparse
import json
import time
from pathlib import Path

import librosa


def load16k(path):
    wave, _ = librosa.load(path, sr=16000, mono=True)
    return wave


def parakeet(paths, batch):
    import onnx_asr
    import onnxruntime
    from huggingface_hub import snapshot_download
    if hasattr(onnxruntime, "preload_dlls"):
        onnxruntime.preload_dlls()  # CUDA 13 / cuDNN from pip (~/.venvs/parakeet, scripts/asrtest/setup_parakeet_gpu.sh)
    # A real directory, not the HF cache: onnxruntime refuses external weights that resolve outside the model folder.
    local = Path.home() / "sg-data" / "models" / "parakeet-tdt-0.6b-v3-onnx"
    snapshot_download("istupakov/parakeet-tdt-0.6b-v3-onnx", local_dir=local)
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", local,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    out = []
    for i in range(0, len(paths), batch):
        waves = [load16k(p) for p in paths[i:i + batch]]
        out.extend(model.recognize(waves, sample_rate=16000))
    return out


def qwen(paths, batch):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    model_id = "Qwen/Qwen3-ASR-1.7B-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.bfloat16)
    out = []
    for i in range(0, len(paths), batch):
        waves = [load16k(p) for p in paths[i:i + batch]]
        inputs = processor.apply_transcription_request(audio=waves, language="Polish").to(model.device, model.dtype)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        out.extend(processor.decode(ids[:, inputs["input_ids"].shape[1]:], return_format="transcription_only"))
    return out


def whisper(paths, batch):
    import torch  # noqa: F401  (loads the CUDA libraries faster-whisper needs)
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    asr = BatchedInferencePipeline(WhisperModel("large-v3-turbo", device="cuda", compute_type="float16"))
    out = []
    for p in paths:
        segments, _ = asr.transcribe(load16k(p), language="pl", batch_size=batch, beam_size=1, without_timestamps=True)
        out.append(" ".join(s.text.strip() for s in segments).strip())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("parakeet", "qwen", "whisper"), required=True)
    parser.add_argument("--sets", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", help="Output file name (default: the engine)")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.sets.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = time.monotonic()
    texts = {"parakeet": parakeet, "qwen": qwen, "whisper": whisper}[args.engine]([r["audio"] for r in rows], args.batch)
    seconds = time.monotonic() - start
    out = args.sets.parent / f"{args.name or args.engine}.jsonl"
    out.write_text("".join(json.dumps({"id": r["id"], "transcript": t}, ensure_ascii=False) + "\n" for r, t in zip(rows, texts)),
                   encoding="utf-8")
    print(json.dumps({"engine": args.engine, "clips": len(rows), "seconds": round(seconds, 1)}))


if __name__ == "__main__":
    main()
