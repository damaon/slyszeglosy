"""Read the bake-off sentences with one teacher, cloning each reference voice.

    ~/.venvs/omni/bin/python scripts/bakeoff/generate.py --engine omnivoice  --voices data/bakeoff/voices/chosen.json
    ~/.venvs/cbx/bin/python  scripts/bakeoff/generate.py --engine chatterbox --voices data/bakeoff/voices/chosen.json

Output: <out>/<engine>/<voice>/<id>.wav (24 kHz, 16-bit, peak 0.95) and
<out>/<engine>/manifest.jsonl with generation time per clip. Resumable: clips
that exist are skipped. Seeds are fixed per sentence and voice.
"""

import argparse
import json
from pathlib import Path
import time
import zlib

import numpy as np
import soundfile as sf
import torch

RATE = 24000


class OmniVoiceEngine:
    name = "omnivoice"

    def __init__(self, voices):
        from omnivoice import OmniVoice
        self.model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
        self.prompts = {v: self.model.create_voice_clone_prompt(ref_audio=r["audio"], ref_text=r["text"]) for v, r in voices.items()}

    def speak(self, text, voice):
        return np.asarray(self.model.generate(text=text, voice_clone_prompt=self.prompts[voice])[0], dtype=np.float32), RATE


class ChatterboxEngine:
    name = "chatterbox"

    def __init__(self, voices, checkpoint="v3"):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        self.model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model=checkpoint)
        self.voices, self.current = voices, None

    def speak(self, text, voice):
        if voice != self.current:
            self.model.prepare_conditionals(self.voices[voice]["audio"], exaggeration=0.5)
            self.current = voice
        wave = self.model.generate(text, language_id="pl")
        return wave.squeeze(0).cpu().numpy().astype(np.float32), self.model.sr


def resample(wave, rate):
    if rate == RATE:
        return wave
    import torchaudio.functional as AF
    return AF.resample(torch.from_numpy(wave), rate, RATE).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=("omnivoice", "chatterbox"), required=True)
    parser.add_argument("--voices", type=Path, required=True, help='JSON {"F": {"audio": path, "text": ...}, "M": ...}')
    parser.add_argument("--sentences", type=Path, default=Path(__file__).with_name("sentences.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/bakeoff"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    voices = json.loads(args.voices.read_text(encoding="utf-8"))
    for reference in voices.values():
        reference["audio"] = str((args.voices.parent / reference["audio"]).resolve())
    rows = [json.loads(line) for line in args.sentences.read_text(encoding="utf-8").splitlines() if line.strip()][:args.limit]
    engine = OmniVoiceEngine(voices) if args.engine == "omnivoice" else ChatterboxEngine(voices)
    root = args.out / engine.name
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    done = {json.loads(line)["clip"] for line in manifest.read_text(encoding="utf-8").splitlines()} if manifest.exists() else set()
    with manifest.open("a", encoding="utf-8") as stream:
        for voice in voices:
            (root / voice).mkdir(parents=True, exist_ok=True)
            for row in rows:
                clip = f"{voice}/{row['id']}"
                if clip in done:
                    continue
                torch.manual_seed(zlib.crc32(clip.encode()))
                torch.cuda.synchronize()
                start = time.time()
                wave, rate = engine.speak(row["text"], voice)
                torch.cuda.synchronize()
                seconds = time.time() - start
                wave = resample(wave, rate)
                peak = float(np.abs(wave).max()) if wave.size else 0.0
                if peak > 0:
                    sf.write(root / f"{clip}.wav", wave * (0.95 / peak), RATE, subtype="PCM_16")
                record = {"clip": clip, "engine": engine.name, "voice": voice, "id": row["id"], "category": row["category"],
                          "text": row["text"], "audio": f"{clip}.wav" if peak > 0 else None,
                          "audio_seconds": round(len(wave) / RATE, 3), "generation_seconds": round(seconds, 3)}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{clip}  {record['audio_seconds']:.1f}s audio in {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
