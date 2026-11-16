"""Bake-off summary (report.md, report.json) and a listening page (index.html).

    python scripts/bakeoff/report.py data/bakeoff
"""

import collections
import html
import json
from pathlib import Path
import statistics
import sys


def load(root):
    clips = []
    for path in sorted(root.glob("*/manifest.scores.jsonl")):
        clips += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return clips


def mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def summarize(clips):
    groups = collections.defaultdict(list)
    for clip in clips:
        groups[(clip["engine"], clip["voice"])].append(clip)
    summary = []
    for (engine, voice), rows in sorted(groups.items()):
        ok = [r for r in rows if not r.get("failed")]
        f0 = [r["f0_median_hz"] for r in ok if r.get("f0_median_hz")]
        summary.append({
            "engine": engine, "voice": voice, "clips": len(rows), "failed": len(rows) - len(ok),
            "cer_mean": mean(r["cer"] for r in ok), "cer_median": round(statistics.median(r["cer"] for r in ok), 4),
            "over_5pct_cer": sum(r["cer"] > 0.05 for r in ok), "wer_mean": mean(r["wer"] for r in ok),
            "phone_probability": mean(r.get("phone_probability") for r in ok),
            "weak_phone_fraction": mean(r.get("weak_phone_fraction") for r in ok),
            "alignment_errors": sum("alignment_error" in r for r in ok),
            "rtf": round(sum(r["generation_seconds"] for r in ok) / max(sum(r["audio_seconds"] for r in ok), 1e-6), 3),
            "f0_median_hz": round(statistics.median(f0), 1) if f0 else None,
            "f0_spread_hz": round(statistics.pstdev(f0), 1) if len(f0) > 1 else None,
            "chars_per_second": mean(r["chars_per_second"] for r in ok),
        })
    return summary


def by_category(clips):
    table = collections.defaultdict(lambda: collections.defaultdict(list))
    for clip in clips:
        if not clip.get("failed"):
            table[clip["category"]][clip["engine"]].append(clip["cer"])
    return {category: {engine: mean(values) for engine, values in engines.items()} for category, engines in sorted(table.items())}


def weak_words(clips):
    counts = collections.defaultdict(collections.Counter)
    for clip in clips:
        for word in clip.get("weak_words", []):
            counts[clip["engine"]][word.lower()] += 1
    return {engine: counter.most_common(25) for engine, counter in counts.items()}


def markdown(summary, categories, weak, worst):
    lines = ["# Bake-off nauczycieli TTS", "", "| Silnik | Głos | Klipy | CER śr. | CER med. | >5% CER | WER śr. | "
             "Dopasowanie fonemów | Słabe fonemy | RTF | F0 med. | F0 rozrzut | zn./s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['engine']} | {s['voice']} | {s['clips']} | {s['cer_mean']:.3f} | {s['cer_median']:.3f} | "
                     f"{s['over_5pct_cer']} | {s['wer_mean']:.3f} | {s['phone_probability']} | {s['weak_phone_fraction']} | "
                     f"{s['rtf']} | {s['f0_median_hz']} | {s['f0_spread_hz']} | {s['chars_per_second']} |")
    engines = sorted({s["engine"] for s in summary})
    lines += ["", "## CER według kategorii", "", "| Kategoria | " + " | ".join(engines) + " |",
              "|---|" + "---:|" * len(engines)]
    for category, values in categories.items():
        lines.append(f"| {category} | " + " | ".join(f"{values.get(e, float('nan')):.3f}" for e in engines) + " |")
    lines += ["", "## Najczęstsze słowa ze słabymi fonemami", ""]
    for engine, items in weak.items():
        lines.append(f"- **{engine}**: " + ", ".join(f"{w} ({n})" for w, n in items))
    lines += ["", "## Najgorsze klipy (CER)", ""]
    for clip in worst:
        lines.append(f"- {clip['engine']} {clip['clip']} CER {clip['cer']:.3f}: „{clip['text']}” → „{clip['transcript']}”")
    return "\n".join(lines) + "\n"


def page(root, clips, summary):
    engines = sorted({c["engine"] for c in clips})
    voices = sorted({c["voice"] for c in clips})
    cell = {(c["engine"], c["voice"], c["id"]): c for c in clips}
    sentences = {}
    for clip in clips:
        sentences.setdefault(clip["id"], clip)
    chosen = root / "voices" / "chosen.json"
    references = json.loads(chosen.read_text(encoding="utf-8")) if chosen.exists() else {}
    css = """:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e2dc;--bad:#b3261e;--ok:#1e6b3a;--chip:#f0eee8}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#a3a29b;--line:#33322e;--bad:#ff8a80;--ok:#7fd49b;--chip:#26251f}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:22px;margin:0 0 4px}p{color:var(--muted);margin:0 0 16px}table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid var(--line);padding:6px 8px;vertical-align:top;text-align:left}th{font-weight:600;position:sticky;top:0;background:var(--bg)}
audio{width:210px;height:32px}.cer{font-size:12px;padding:1px 6px;border-radius:9px;background:var(--chip)}
.bad{color:var(--bad)}.ok{color:var(--ok)}.cat{color:var(--muted);font-size:12px}.wrap{overflow-x:auto}
.blind .label{visibility:hidden}"""
    out = ["<!doctype html><html lang='pl'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
           f"<title>Bake-off nauczycieli</title><style>{css}</style><body>",
           "<h1>Bake-off nauczycieli TTS</h1><p>Ten sam głos referencyjny klonowany przez oba modele. "
           "<label><input type='checkbox' onchange=\"document.body.classList.toggle('blind',this.checked)\"> ukryj nazwy modeli i CER (odsłuch na ślepo)</label></p>"]
    if references:
        out.append("<p>Referencje: " + " · ".join(
            f"{v}: <audio controls preload='none' src='voices/{html.escape(r['audio'])}'></audio> <span class='cat'>{html.escape(r['instruct'])}</span>"
            for v, r in references.items()) + "</p>")
    out.append("<div class='wrap'><table><tr><th>Zdanie</th>" + "".join(
        f"<th><span class='label'>{e}</span> {v}</th>" for v in voices for e in engines) + "</tr>")
    for sid, first in sentences.items():
        row = [f"<tr><td>{html.escape(first['text'])}<div class='cat'>{sid} · {first['category']}</div></td>"]
        for voice in voices:
            for engine in engines:
                clip = cell.get((engine, voice, sid))
                if not clip or clip.get("failed"):
                    row.append("<td>—</td>")
                    continue
                grade = "bad" if clip["cer"] > 0.05 else "ok"
                row.append(f"<td><audio controls preload='none' src='{engine}/{html.escape(clip['audio'])}'></audio>"
                           f"<div class='label'><span class='cer {grade}'>CER {clip['cer']:.2f}</span> "
                           f"<span class='cat'>{html.escape(clip['transcript'])}</span></div></td>")
        out.append("".join(row) + "</tr>")
    out.append("</table></div></body></html>")
    (root / "index.html").write_text("\n".join(out), encoding="utf-8")


def main():
    root = Path(sys.argv[1])
    clips = load(root)
    summary, categories, weak = summarize(clips), by_category(clips), weak_words(clips)
    worst = sorted((c for c in clips if not c.get("failed")), key=lambda c: -c["cer"])[:15]
    (root / "report.json").write_text(json.dumps({"summary": summary, "categories": categories, "weak_words": weak},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    (root / "report.md").write_text(markdown(summary, categories, weak, worst), encoding="utf-8")
    page(root, clips, summary)
    print((root / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
