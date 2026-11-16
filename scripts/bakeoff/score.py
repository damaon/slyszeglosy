"""Score teacher recordings: intelligibility, fit to our phonemes, speed, pitch.

For every row of each manifest (JSONL with "text" and "audio" relative to the
manifest's folder) it writes <manifest>.scores.jsonl with:

* cer, wer: Whisper large-v3-turbo (language pl) against the text. Both sides
  go through wymowa.normalize, so "21" and "dwadzieścia jeden" agree.
* phone_probability, weak_phone_fraction, weak_words: the forced aligner's
  confidence in OUR phoneme sequence (wav2vec2-lv-60-espeak-cv-ft, the model
  used for annotation). Words with weak phones are where the teacher and
  wymowa disagree: a teacher error or a gap in wymowa's rules.
* f0_median_hz, chars_per_second, seconds.

Run in ~/.venvs/sg:
    PYTHONPATH=src python scripts/bakeoff/score.py data/bakeoff/*/manifest.jsonl
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf

from tiny_voca.align import PhoneRecognizer, token_spans
from tiny_voca.text import PHONE_SET, PUNCTUATION, SP, UNK, phonemize
from wymowa.g2p import BOUNDARY, STRESS, fold
from wymowa.lexicon import PROCLITICS
from wymowa.normalize import normalize

WEAK = 0.1          # the annotation gate's weak-phone threshold (align.quality)


def plain(text):
    """Lowercase words of normalised text, for CER/WER."""
    text = normalize(text).replace(BOUNDARY, "").replace(STRESS, "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def word_labels(text):
    """Orthographic word for each phonemised word, in order (proclitics joined)."""
    words, carry = [], ""
    for token in re.findall(r"[^\s" + re.escape("".join(PUNCTUATION)) + r"]+", normalize(text)):
        key = fold(token).replace(BOUNDARY, "").replace(STRESS, "")
        if not key:
            continue
        if key in PROCLITICS:
            carry += token + " "
            continue
        words.append(carry + token)
        carry = ""
    return words + ([carry.strip()] if carry else [])


def fit(recognizer, text, wave, rate):
    phones = phonemize(text)
    labels, owner, word = [], [], 0
    for symbol in phones.symbols:
        if symbol == SP or symbol in PUNCTUATION:
            if owner and owner[-1] == word:
                word += 1
            continue
        if symbol in PHONE_SET or symbol == UNK:
            labels.append(recognizer.label(symbol))
            owner.append(word)
    log_probs = recognizer.log_probs(wave, rate)
    spans = token_spans(log_probs, labels, recognizer.blank)
    probability = np.array([float(np.exp(log_probs[a:b + 1, label]).mean()) for (a, b), label in zip(spans, labels)])
    names = word_labels(text)
    weak = sorted({names[owner[i]] if owner[i] < len(names) else f"#{owner[i]}" for i in np.flatnonzero(probability < WEAK)})
    return {"phone_probability": round(float(probability.mean()), 4),
            "weak_phone_fraction": round(float((probability < WEAK).mean()), 4), "weak_words": weak}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--whisper", default="large-v3-turbo")
    args = parser.parse_args()
    import jiwer
    import librosa
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly
    asr = WhisperModel(args.whisper, device="cuda", compute_type="float16")
    recognizer = PhoneRecognizer("cuda", local_files_only=False)
    for manifest in args.manifests:
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        out = []
        for row in rows:
            if not row.get("audio"):
                out.append({**row, "failed": True})
                continue
            wave, rate = sf.read(manifest.parent / row["audio"], dtype="float32")
            segments, _ = asr.transcribe(resample_poly(wave, 2, 3) if rate == 24000 else wave, language="pl", beam_size=5,
                                         condition_on_previous_text=False, vad_filter=False)
            transcript = " ".join(s.text.strip() for s in segments).strip()
            reference, hypothesis = plain(row["text"]), plain(transcript) or "<pusto>"
            f0, voiced, _ = librosa.pyin(wave, fmin=60, fmax=500, sr=rate, frame_length=1024, hop_length=256)
            score = {**row, "transcript": transcript, "cer": round(jiwer.cer(reference, hypothesis), 4),
                     "wer": round(jiwer.wer(reference, hypothesis), 4),
                     "f0_median_hz": round(float(np.nanmedian(f0[voiced])), 1) if voiced.any() else None,
                     "seconds": round(len(wave) / rate, 3),
                     "chars_per_second": round(len(reference) / max(len(wave) / rate, 1e-6), 2)}
            try:
                score.update(fit(recognizer, row["text"], wave, rate))
            except ValueError as error:
                score.update(alignment_error=str(error))
            out.append(score)
            print(f"{manifest.parent.name}/{row.get('clip', row.get('id'))}  cer={score['cer']:.3f}  "
                  f"fit={score.get('phone_probability')}  weak={score.get('weak_words')}", flush=True)
        target = manifest.with_name(manifest.stem + ".scores.jsonl")
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")


if __name__ == "__main__":
    main()
