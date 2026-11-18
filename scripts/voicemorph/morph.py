"""Change the voice of a trained checkpoint without training: measure the target voice,
then fold the difference into four linear layers (pitch head, duration head, the two
filter heads of the source-filter synthesis). Guide: docs/zmiana-glosu.md.

    M="python scripts/voicemorph/morph.py --checkpoint weights/slyszeglosy.pt --work data/voicemorph"
    $M data --folder moje-nagrania      # or: --parquet <MC Speech shard>
    $M fit                              # parallel synthesis, statistics, edited checkpoints
    $M render                           # held-out sentences with every system
    python scripts/asrtest/run_asr.py --engine parakeet --sets data/voicemorph/eval/sets.jsonl   # optional, ~/.venvs/parakeet
    python scripts/voicemorph/cer.py --work data/voicemorph                                      # optional
    $M score                            # speaker similarity, MCD, pitch, tempo
    $M samples                          # side-by-side listening files

Why it works at all: the model's sound is made by a fixed source-filter synthesizer.
Pitch is one output of a linear head (log F0), the vocal tract filter is a cepstrum from
another linear head. A pitch change, a frequency warp (vocal tract length) and a spectral
map are linear maps on those outputs, so they can be multiplied into the weights:
W' = T W, b' = T b + t. Nothing is trained: the maps come from statistics and one
closed-form least-squares solve on time-aligned parallel frames.

The edited layers are shared by both voices; use the result with the source voice only.
"""

import argparse
import io
import json
import math
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from tiny_voca import infer

SR, HOP = 24000, 256
PERIOD = 1000 * HOP / SR
NC, NA, NN = 80, 16, 32           # causal / anticausal / noise cepstrum sizes (config)
FIT_DIM = 40                      # coefficients the spectral maps may change
DENSE = np.linspace(0, math.pi, 2049)
ARGS = None                       # parsed command line


# ---------- analysis ----------

def cos_basis(n, grid=DENSE):
    return np.cos(np.outer(grid, np.arange(n)))


AN_GRID = np.linspace(0, math.pi, 513)                 # CheapTrick, fft 1024
AN_PINV = np.linalg.pinv(cos_basis(NC, AN_GRID))


def analyse(wave):
    """pyworld F0 and the log-amplitude envelope as cosine coefficients
    (same convention as the model: log|H(w)| = sum_n c_n cos(n w)), one row per model frame."""
    import pyworld
    x = np.ascontiguousarray(wave, dtype=np.float64)
    f0, t = pyworld.dio(x, SR, f0_floor=60, f0_ceil=500, frame_period=PERIOD)
    f0 = pyworld.stonemask(x, f0, t, SR)
    sp = pyworld.cheaptrick(x, f0, t, SR, f0_floor=71, fft_size=1024)
    ceps = (0.5 * np.log(sp + 1e-12)) @ AN_PINV.T
    return f0, ceps


def norm_feat(ceps):
    f = ceps[:, 1:25]
    return f - f.mean(0)


def dtw_pairs(a, b):
    _, path = librosa.sequence.dtw(X=norm_feat(a).T, Y=norm_feat(b).T, metric="euclidean")
    return path[::-1]


def trimmed_seconds(wave):
    trimmed, _ = librosa.effects.trim(wave, top_db=35)
    return len(trimmed) / SR


# ---------- model ----------

def load(path):
    infer.match_frontend(path)
    return infer.load_model(path, ARGS.device)


def model_cepstra(h, model):
    """Harmonic head output [frames, 98] -> total log-amplitude cepstrum [frames, 80] of the filter."""
    vf = model.vocoder
    c = h[:, :NC] * vf.causal_lifter.cpu().numpy()
    a = h[:, NC:NC + NA] * vf.anti_lifter.cpu().numpy()
    c[:, 0] += h[:, NC + NA:].mean(1)
    c[:, 1:NA + 1] += a
    return c


def synth_with_filter(model, text, voice):
    """One utterance; returns (wave, harmonic head output per frame)."""
    captured = []
    hook = model.vocoder.harmonic.register_forward_hook(lambda m, i, o: captured.append(o[0].float().cpu().numpy()))
    try:
        wave, _, _ = infer.speak(model, text, voice)
    finally:
        hook.remove()
    return infer.limit_peak(wave).numpy(), captured[0]


# ---------- linear maps on log-amplitude cepstra, folded into the weights ----------

def warp_freq(w, alpha):
    """All-pass (bilinear) frequency warp; alpha > 0 moves frequencies up."""
    return w + 2 * np.arctan(alpha * np.sin(w) / (1 - alpha * np.cos(w)))


def warp_matrix(alpha, n):
    """Coefficients of E(w) -> coefficients of E(warp^-1(w)): formants move by the warp."""
    warped = np.cos(np.outer(warp_freq(DENSE, -alpha), np.arange(n)))
    return np.linalg.pinv(cos_basis(n)) @ warped


def fold_harmonic(state, T, t, model):
    """Total log-amplitude m = c + P a becomes T m + t; the anticausal part a (glottal pulse) stays,
    so c' = T c + (T - I) P a + t. Folded into the harmonic head (lifters included)."""
    vf = model.vocoder
    Lc, La = vf.causal_lifter.cpu().double(), vf.anti_lifter.cpu().double()
    W, b = state["vocoder.harmonic.weight"].double().clone(), state["vocoder.harmonic.bias"].double().clone()
    T, t = torch.from_numpy(T).double(), torch.from_numpy(t).double()
    P = torch.zeros(NC, NA, dtype=torch.float64)
    P[1:NA + 1] = torch.eye(NA, dtype=torch.float64)
    mix = (T - torch.eye(NC, dtype=torch.float64)) @ P
    W[:NC], b[:NC] = ((T @ (Lc[:, None] * W[:NC]) + mix @ (La[:, None] * W[NC:NC + NA])) / Lc[:, None],
                      (T @ (Lc * b[:NC]) + mix @ (La * b[NC:NC + NA]) + t) / Lc)
    state["vocoder.harmonic.weight"], state["vocoder.harmonic.bias"] = W.float(), b.float()


def fold_noise(state, T, model):
    L = model.vocoder.noise_lifter.cpu().double()
    W, b = state["vocoder.noise.weight"].double().clone(), state["vocoder.noise.bias"].double().clone()
    T = torch.from_numpy(T).double()
    W[:NN] = (T @ (L[:, None] * W[:NN])) / L[:, None]
    b[:NN] = (T @ (L * b[:NN])) / L
    state["vocoder.noise.weight"], state["vocoder.noise.bias"] = W.float(), b.float()


def fold_pitch(state, k, d):
    """log(F0/200) -> k * log(F0/200) + d for the oscillator; the decoder still sees the old
    value (prosody_projection compensates), so the filter it predicts is unchanged."""
    W, b = state["prosody_head.weight"].clone(), state["prosody_head.bias"].clone()
    W[0], b[0] = k * W[0], k * b[0] + d
    Wp, bp = state["prosody_projection.weight"].clone(), state["prosody_projection.bias"].clone()
    bp = bp - Wp[:, 0] * d / k
    Wp[:, 0] = Wp[:, 0] / k
    state.update({"prosody_head.weight": W, "prosody_head.bias": b,
                  "prosody_projection.weight": Wp, "prosody_projection.bias": bp})


def fold_tempo(state, ratio):
    b = state["duration.out.bias"].clone()
    b[0] += math.log(ratio)          # log1p(frames): about frames * ratio
    state["duration.out.bias"] = b


# ---------- stages ----------

def read_meta():
    return json.loads((ARGS.work / "target.json").read_text(encoding="utf-8"))


def split(meta, name):
    return [row for row in meta if row["split"] == name]


def stage_data():
    """Target clips -> <work>/target/*.wav (24 kHz mono) and target.json with fit / eval / ref splits."""
    out = ARGS.work / "target"
    out.mkdir(parents=True, exist_ok=True)
    clips = []                                     # (name, waveform, sample rate, text)
    if ARGS.parquet:
        import pyarrow.parquet as pq
        for row in pq.read_table(ARGS.parquet).to_pylist()[:ARGS.limit]:
            wave, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            clips.append((row["id"], wave, sr, row["transcript"]))
    else:
        for line in (ARGS.folder / "metadata.csv").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            name, text = line.split("|", 1)
            wave, sr = sf.read(ARGS.folder / name, dtype="float32")
            clips.append((Path(name).stem, wave, sr, text.strip()))
        clips = clips[:ARGS.limit]
    if len(clips) < 60:
        sys.exit(f"{len(clips)} clips: at least 60 are needed (better 300+)")
    order = list(range(len(clips)))
    random.Random(0).shuffle(order)
    n_ref, n_eval = min(60, len(clips) // 6), min(100, len(clips) // 5)
    names = {i: "ref" if rank < n_ref else "eval" if rank < n_ref + n_eval else "fit" for rank, i in enumerate(order)}
    meta = []
    for i, (name, wave, sr, text) in enumerate(clips):
        if wave.ndim > 1:
            wave = wave.mean(1)
        if sr != SR:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=SR)
        path = out / f"{i:04d}.wav"
        sf.write(path, wave, SR)
        meta.append({"i": i, "id": name, "text": text, "audio": str(path), "split": names[i]})
    (ARGS.work / "target.json").write_text(json.dumps(meta, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(meta)} clips: fit {len(split(meta, 'fit'))}, eval {n_eval}, ref {n_ref}")


def stage_fit():
    meta = read_meta()
    model = load(ARGS.checkpoint)
    src_dir = ARGS.work / "fit-source"
    src_dir.mkdir(exist_ok=True)
    X, Y, E = [], [], []                  # aligned voiced frames: model cepstra, target analysis; analysis-minus-model offsets
    lf_s, lf_t, dur_s, dur_t = [], [], [], []
    for row in split(meta, "fit"):
        try:
            wave_s, h = synth_with_filter(model, row["text"], ARGS.voice)
        except Exception as error:     # too long for one call, nothing to pronounce
            print("skip", row["i"], error, file=sys.stderr)
            continue
        sf.write(src_dir / f"{row['i']:04d}.wav", wave_s, SR)
        wave_t, _ = sf.read(row["audio"], dtype="float32")
        f0_s, an_s = analyse(wave_s)
        f0_t, an_t = analyse(wave_t)
        m_s = model_cepstra(h, model)
        n = min(len(m_s), len(an_s))
        f0_s, an_s, m_s = f0_s[:n], an_s[:n], m_s[:n]
        voiced = f0_s > 0
        E.append(an_s[voiced] - m_s[voiced])
        lf_s.append(np.log(f0_s[voiced] / 200))
        lf_t.append(np.log(f0_t[f0_t > 0] / 200))
        dur_s.append(trimmed_seconds(wave_s))
        dur_t.append(trimmed_seconds(wave_t))
        for i, j in dtw_pairs(an_s, an_t):
            if f0_s[i] > 0 and f0_t[j] > 0:
                X.append(m_s[i])
                Y.append(an_t[j])
    X, Y, e = np.array(X), np.array(Y), np.concatenate(E).mean(0)
    Y = Y - e                              # target analysis expressed in the model's cepstrum domain
    lf_s, lf_t = np.concatenate(lf_s), np.concatenate(lf_t)
    k = float(lf_t.std() / lf_s.std())
    d = float(lf_t.mean() - k * lf_s.mean())
    tempo = float(sum(dur_t) / sum(dur_s))
    print(f"pairs {len(X)}; F0 source {200*math.exp(lf_s.mean()):.0f} Hz sd {lf_s.std():.3f}, "
          f"target {200*math.exp(lf_t.mean()):.0f} Hz sd {lf_t.std():.3f} -> k {k:.3f} d {d:.3f}; tempo x{tempo:.3f}")

    def err(T, t):
        return float(((Y - X @ T.T - t)[:, 1:FIT_DIM] ** 2).sum(1).mean())

    # "warp": one warp factor (vocal tract length) + mean spectrum difference, gain excluded.
    best = None
    for alpha in np.round(np.arange(-0.20, 0.201, 0.01), 2):
        T = warp_matrix(alpha, NC)
        delta = (Y - X @ T.T).mean(0)
        delta[0], delta[FIT_DIM:] = 0.0, 0.0
        if best is None or err(T, delta) < best[1]:
            best = (float(alpha), err(T, delta), T, delta)
    alpha, _, T_warp, t_warp = best

    # "affine<dim>": least squares on coefficients 1..dim-1 (gain row and column kept).
    def affine(dim):
        sl = slice(1, dim)
        xs, ys = X[:, sl], Y[:, sl]
        mx, my = xs.mean(0), ys.mean(0)
        xc, yc = xs - mx, ys - my
        lam = 1e-3 * np.trace(xc.T @ xc) / xc.shape[1]
        A = np.linalg.solve(xc.T @ xc + lam * np.eye(xc.shape[1]), xc.T @ yc).T
        T, t = np.eye(NC), np.zeros(NC)
        T[sl, sl], t[sl] = A, my - A @ mx
        return T, t

    # "diag": each coefficient's mean and spread matched (no cross terms, no shrinkage).
    def diag(dim):
        sl = slice(1, dim)
        s = Y[:, sl].std(0) / X[:, sl].std(0)
        T, t = np.eye(NC), np.zeros(NC)
        T[sl, sl], t[sl] = np.diag(s), Y[:, sl].mean(0) - s * X[:, sl].mean(0)
        return T, t

    spectral = {"warp": (T_warp, t_warp), "affine40": affine(40), "affine20": affine(20), "diag": diag(FIT_DIM)}
    report = {"pairs": len(X), "k": k, "d": d, "tempo": tempo, "alpha": alpha,
              "err_none": err(np.eye(NC), np.zeros(NC)), **{f"err_{n}": err(*m) for n, m in spectral.items()}}
    print(" ".join(f"{key} {value:.4f}" for key, value in report.items() if key.startswith("err")), f"(warp alpha {alpha:+.2f})")

    base = torch.load(ARGS.checkpoint, map_location="cpu", weights_only=True)
    for name in ("pitch", *spectral):
        state = dict(base)
        state["model"] = {key: value.clone() for key, value in base["model"].items()}
        fold_pitch(state["model"], k, d)
        fold_tempo(state["model"], tempo)
        if name in spectral:
            fold_harmonic(state["model"], *spectral[name], model)
        if name == "warp":
            fold_noise(state["model"], warp_matrix(alpha, NN), model)
        torch.save(state, ARGS.work / f"inference-{name}.pt")
    (ARGS.work / "fit.json").write_text(json.dumps(report, indent=1))
    print("saved", ", ".join(f"inference-{n}.pt" for n in ("pitch", *spectral)))


SYSTEMS = ("pitch", "warp", "affine40", "affine20", "diag")


def stage_render():
    meta = read_meta()
    other = [v for v in ("F", "M") if v != ARGS.voice][0]
    systems = {f"base_{ARGS.voice}": (ARGS.checkpoint, ARGS.voice), f"base_{other}": (ARGS.checkpoint, other),
               **{n: (ARGS.work / f"inference-{n}.pt", ARGS.voice) for n in SYSTEMS}}
    rows = []
    for name, (path, voice) in systems.items():
        model = load(path)
        out = ARGS.work / "eval" / name
        out.mkdir(parents=True, exist_ok=True)
        for row in split(meta, "eval"):
            wave, _ = infer.speak_text(model, row["text"], voice)
            p = out / f"{row['i']:04d}.wav"
            sf.write(p, wave.numpy(), SR)
            rows.append({"id": f"{name}/{row['i']}", "system": name, "i": row["i"], "ref": row["text"], "audio": str(p)})
        del model
        torch.cuda.empty_cache()
    for row in split(meta, "eval"):
        rows.append({"id": f"target/{row['i']}", "system": "target", "i": row["i"], "ref": row["text"], "audio": row["audio"]})
    (ARGS.work / "eval" / "sets.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(len(rows), "clips")


def stage_score():
    from transformers import AutoFeatureExtractor, WavLMForXVector
    meta = {row["i"]: row for row in read_meta()}
    rows = [json.loads(line) for line in (ARGS.work / "eval" / "sets.jsonl").read_text(encoding="utf-8").splitlines()]
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(ARGS.device).eval()

    def embed(path):
        wave, _ = librosa.load(path, sr=16000)
        inputs = fe(wave, sampling_rate=16000, return_tensors="pt").to(ARGS.device)
        with torch.no_grad():
            e = sv(**inputs).embeddings[0]
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    centroid = np.mean([embed(row["audio"]) for row in meta.values() if row["split"] == "ref"], 0)
    centroid /= np.linalg.norm(centroid)
    target_an = {i: analyse(sf.read(row["audio"], dtype="float32")[0]) for i, row in meta.items() if row["split"] == "eval"}
    cer_path = ARGS.work / "eval" / "cer.json"
    cer = json.loads(cer_path.read_text()) if cer_path.exists() else {}
    report = {}
    for system in dict.fromkeys(r["system"] for r in rows):
        sims, mcds, lf, dur = [], [], [], []
        for r in (r for r in rows if r["system"] == system):
            wave, _ = sf.read(r["audio"], dtype="float32")
            sims.append(float(embed(r["audio"]) @ centroid))
            f0, an = analyse(wave)
            lf.append(np.log(f0[f0 > 0]))
            dur.append(trimmed_seconds(wave))
            path = dtw_pairs(an, target_an[r["i"]][1])
            diff = (an[path[:, 0], 1:25] - target_an[r["i"]][1][path[:, 1], 1:25]) / 2      # real-cepstrum convention
            mcds.append(float((10 / math.log(10)) * np.sqrt(2 * (diff ** 2).sum(1)).mean()))
        lf = np.concatenate(lf)
        report[system] = {"SIM": round(float(np.mean(sims)), 4), "MCD_dB": round(float(np.mean(mcds)), 2),
                          "F0_Hz": round(float(math.exp(lf.mean())), 1), "F0_sd": round(float(lf.std()), 3),
                          "seconds": round(float(np.sum(dur)), 1), **({"CER_%": cer[system]["CER"]} if system in cer else {})}
        print(f"{system:10s}", report[system], flush=True)
    (ARGS.work / "eval" / "score.json").write_text(json.dumps(report, indent=1))


def stage_samples():
    """<work>/samples/<system>.wav: the same four held-out sentences, one file per system."""
    meta = split(read_meta(), "eval")[:4]
    out = ARGS.work / "samples"
    out.mkdir(exist_ok=True)
    gap = np.zeros(int(0.6 * SR), dtype=np.float32)
    for system in ("target", f"base_{ARGS.voice}", *SYSTEMS):
        parts = []
        for row in meta:
            path = row["audio"] if system == "target" else ARGS.work / "eval" / system / f"{row['i']:04d}.wav"
            wave, _ = sf.read(path, dtype="float32")
            parts += [wave / max(1e-6, float(np.abs(wave).max())) * 0.8, gap]
        sf.write(out / f"{system}.wav", np.concatenate(parts), SR)
    (out / "zdania.txt").write_text("\n".join(row["text"] for row in meta) + "\n", encoding="utf-8")
    print("samples in", out)


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("data", "fit", "render", "score", "samples"))
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/slyszeglosy.pt"), help="Inference checkpoint to edit")
    parser.add_argument("--work", type=Path, default=Path("data/voicemorph"), help="Working folder (clips, results, edited checkpoints)")
    parser.add_argument("--voice", choices=("F", "M"), default="M", help="Voice of the checkpoint to move towards the target")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    target = parser.add_argument_group("data stage")
    target.add_argument("--folder", type=Path, help="Folder with metadata.csv (file.wav|text per line) and the recordings")
    target.add_argument("--parquet", type=Path, help="A Hugging Face parquet shard with audio + transcript columns (MC Speech)")
    target.add_argument("--limit", type=int, default=460, help="Use at most this many clips")
    ARGS = parser.parse_args()
    if ARGS.stage == "data" and not (ARGS.folder or ARGS.parquet):
        parser.error("data needs --folder or --parquet")
    ARGS.work.mkdir(parents=True, exist_ok=True)
    {"data": stage_data, "fit": stage_fit, "render": stage_render, "score": stage_score, "samples": stage_samples}[ARGS.stage]()


if __name__ == "__main__":
    main()
