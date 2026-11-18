#!/usr/bin/env bash
# Parakeet TDT v3 on the GPU: onnxruntime-gpu is built for CUDA 13, so this environment
# takes the CUDA 13 and cuDNN libraries from pip (no system CUDA, no sudo). Kept apart from
# the PyTorch environments, which bring their own CUDA 12 libraries.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
V=~/.venvs/parakeet
uv venv -q --allow-existing --python 3.11 $V
uv pip install -q --python $V/bin/python "onnxruntime-gpu[cuda,cudnn]" "onnx-asr[hub]" librosa soundfile
$V/bin/python - <<'P'
import onnxruntime as ort
ort.preload_dlls()            # load the CUDA/cuDNN libraries that came with the pip packages
print("onnxruntime", ort.__version__, ort.get_available_providers())
P
