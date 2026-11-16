#!/usr/bin/env bash
# Build the files the demo serves into public/assets (they are not stored in git).
#
#   bash build_assets.sh /path/to/model.onnx
#
# Copies the ONNX model (exported with `python -m tiny_voca.export`) and writes
# its config.json from the metadata next to it, then zips the phonemizer for
# Pyodide: the wymowa package and tiny_voca/text.py as slyszeglosy_text.py, so
# the browser makes exactly the tokens the model was trained on.
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:?path to model.onnx (its .json metadata must sit next to it)}"
mkdir -p public/assets
cp "$MODEL" public/assets/model.onnx
python3 - "${MODEL%.onnx}.json" public/assets/config.json <<'PY'
import json, sys
meta = json.load(open(sys.argv[1], encoding="utf-8"))
meta["model"] = {"max_tokens": meta["max_tokens"], "max_frames": meta["max_frames"]}
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
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
