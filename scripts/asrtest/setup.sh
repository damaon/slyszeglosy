#!/usr/bin/env bash
# Environment for the ASR comparison: Qwen3-ASR (transformers>=5.13) and Parakeet TDT v3 (onnx-asr).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
V=~/.venvs/asr
uv venv -q --allow-existing --python 3.11 $V
uv pip install -q --python $V/bin/python "torch==2.8.*" "torchaudio==2.8.*" --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python $V/bin/python "transformers>=5.13.0" accelerate soundfile librosa "onnx-asr[gpu,hub]" jiwer
$V/bin/python -c "import torch, transformers, onnx_asr, onnxruntime as o; print('asr ok', torch.__version__, transformers.__version__, o.get_available_providers())"
