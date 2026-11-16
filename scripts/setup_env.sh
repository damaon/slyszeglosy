#!/usr/bin/env bash
# Python environment for data preparation, training and tests (Linux with an NVIDIA GPU).
# Usage: bash scripts/setup_env.sh [venv directory, default ~/.venvs/sg]
set -euo pipefail
VENV="${1:-$HOME/.venvs/sg}"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || pip3 install --user -q uv
uv venv -q --allow-existing --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -q "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cu124
uv pip install -q "numpy<3" "soundfile<1" "librosa>=0.10.2,<1" pyarrow "huggingface-hub<2" scipy onnx onnxruntime
# eSpeak NG as a shared library, used ONLY by scripts/compare_espeak_pl.py (GPL, never shipped).
uv pip install -q espeakng-loader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
