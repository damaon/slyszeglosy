#!/usr/bin/env bash
# VoxCPM2 (openbmb/VoxCPM2, Apache-2.0): teacher for the V2 hard cases (numbers, dates, abbreviations).
# Own environment: it brings its own PyTorch build.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
V=~/.venvs/voxcpm
uv venv -q --allow-existing --python 3.11 $V
uv pip install -q --python $V/bin/python voxcpm soundfile librosa
$V/bin/python - <<'P'
import torch, voxcpm
print("voxcpm", getattr(voxcpm, "__version__", "?"), "torch", torch.__version__, "cuda", torch.cuda.is_available())
P
