#!/usr/bin/env bash
# Build the files the demo serves into public/assets (they are not stored in git).
#
#   bash build_assets.sh /path/to/v1.onnx /path/to/v2.onnx
#
# Copies the two ONNX models (exported with `python -m tiny_voca.export`) as
# model.onnx (V1) and model-v2.onnx (V2), and writes config.json / config-v2.json
# from the metadata next to each. Zips the phonemizer for Pyodide: the wymowa
# package and tiny_voca/text.py as slyszeglosy_text.py, so the browser makes
# exactly the tokens the models were trained on. Both models must use the same
# symbols, since they share the one phonemizer.
set -euo pipefail
cd "$(dirname "$0")"
V1="${1:?path to the V1 model.onnx (its .json metadata must sit next to it)}"
V2="${2:?path to the V2 model.onnx (its .json metadata must sit next to it)}"
mkdir -p public/assets
cp "$V1" public/assets/model.onnx
cp "$V2" public/assets/model-v2.onnx
python3 - "${V1%.onnx}.json" public/assets/config.json "${V2%.onnx}.json" public/assets/config-v2.json <<'PY'
import json, sys
metas = []
for source, target in zip(sys.argv[1::2], sys.argv[2::2]):
    meta = json.load(open(source, encoding="utf-8"))
    meta["model"] = {"max_tokens": meta["max_tokens"], "max_frames": meta["max_frames"]}
    open(target, "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
    metas.append(meta)
if any(meta["symbols"] != metas[0]["symbols"] for meta in metas):
    sys.exit("the models use different symbols: one phonemizer cannot serve both")
PY
python3 - <<'PY'
import pathlib, zipfile
src = pathlib.Path("../src")
with zipfile.ZipFile("public/assets/phonemizer.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((src / "wymowa").glob("*.py")):
        archive.write(path, f"wymowa/{path.name}")
    archive.write(src / "tiny_voca" / "text.py", "slyszeglosy_text.py")
PY
ls -l public/assets
