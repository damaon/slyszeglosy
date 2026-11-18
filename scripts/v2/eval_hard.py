"""V2 phase H: the held-out hard cases (split test_hard of scripts/v2/hard_texts.py), read from
their WRITTEN form (digits, abbreviations) through the whole front end, by each checkpoint/voice.

    PYTHONPATH=src python scripts/v2/eval_hard.py render --texts T.jsonl --out DIR --system v2:CKPT:MC --system v1F:CKPT:F
    ~/.venvs/parakeet/bin/python scripts/asrtest/run_asr.py --engine parakeet --sets DIR/sets.jsonl
    ~/.venvs/sg/bin/python       scripts/asrtest/run_asr.py --engine whisper  --sets DIR/sets.jsonl
    PYTHONPATH=src:scripts python scripts/v2/eval_hard.py score --out DIR [--teacher HARD_ROOT]

CER is against the spoken form (wymowa.normalize), both sides through plain() as in screening.
--teacher adds the VoxCPM2 recordings of the same texts (their screening transcripts) as a reference.
"""

import argparse
import json
import statistics
from pathlib import Path

import soundfile as sf


def render(args):
    import torch
    from tiny_voca import infer
    rows = [json.loads(line) for line in args.texts.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == "test_hard"]
    sets = []
    for spec in args.system:
        name, path, voice = spec.split(":")
        infer.match_frontend(Path(path))
        model = infer.load_model(Path(path), "cuda" if torch.cuda.is_available() else "cpu")
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        for r in rows:
            wave, _ = infer.speak_text(model, r["written"], voice)
            p = out / f"{r['id']}.wav"
            sf.write(p, wave.numpy(), model.config.sample_rate)
            sets.append({"id": f"{name}/{r['id']}", "system": name, "text_id": r["id"], "ref": r["text"], "written": r["written"],
                         "audio": str(p)})
        del model
    (args.out / "sets.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in sets), encoding="utf-8")
    print(len(sets), "clips")


def score(args):
    import jiwer
    from screen_audio_pl import plain
    sets = {s["id"]: s for s in map(json.loads, (args.out / "sets.jsonl").read_text(encoding="utf-8").splitlines())}
    heard = {e: {r["id"]: r["transcript"] for r in map(json.loads, (args.out / f"{e}.jsonl").read_text(encoding="utf-8").splitlines())}
             for e in ("parakeet", "whisper")}
    cer = {}
    for sid, s in sets.items():
        for engine, h in heard.items():
            cer.setdefault(s["system"], {}).setdefault(engine, []).append(jiwer.cer(plain(s["ref"]), plain(h[sid]) or "<pusto>"))
    if args.teacher:
        ids = {s["text_id"] for s in sets.values()}
        for line in (args.teacher / "screened.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["id"] in ids:
                cer.setdefault("teacher", {}).setdefault("parakeet", {})[r["id"]] = r["cer"]
                cer["teacher"].setdefault("whisper", {})[r["id"]] = r["cer_whisper"]
        cer["teacher"] = {e: list(v.values()) for e, v in cer.get("teacher", {}).items()}
    report = {system: {f"CER_{e}_%": round(100 * statistics.mean(v), 2) for e, v in engines.items()} | {"clips": len(engines["parakeet"])}
              for system, engines in cer.items()}
    print(json.dumps(report, indent=1))
    (args.out / "report.json").write_text(json.dumps(report, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("render", "score"))
    parser.add_argument("--texts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--system", action="append", default=[], help="name:checkpoint:voice")
    parser.add_argument("--teacher", type=Path)
    args = parser.parse_args()
    (render if args.stage == "render" else score)(args)


if __name__ == "__main__":
    main()
