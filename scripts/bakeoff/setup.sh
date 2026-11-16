#!/usr/bin/env bash
# Environments for the teacher bake-off (phase 2). Their dependencies conflict,
# so each teacher gets its own venv:
#   ~/.venvs/omni  OmniVoice 0.2.1 (torch 2.8 cu128)       code Apache-2.0, weights CC-BY-NC
#   ~/.venvs/cbx   chatterbox-tts master  (torch 2.6)      code and weights MIT
#   ~/.venvs/sg    scoring: faster-whisper, jiwer, wav2vec2 aligner (transformers<5)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || pip3 install --user -q uv

uv venv -q --allow-existing --python 3.11 ~/.venvs/omni
VIRTUAL_ENV=~/.venvs/omni uv pip install -q --python ~/.venvs/omni/bin/python \
    "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=~/.venvs/omni uv pip install -q --python ~/.venvs/omni/bin/python "omnivoice==0.2.1" soundfile
~/.venvs/omni/bin/python -c "import omnivoice, torch; print('omni ok', torch.__version__, torch.cuda.is_available())"

uv venv -q --allow-existing --python 3.11 ~/.venvs/cbx
# PyPI 0.1.7 ships only the v2 checkpoint; the multilingual v3 checkpoint needs master (pinned).
CHATTERBOX_COMMIT=ab55923c27757c90cfcc407baa57dc25a7d25217
VIRTUAL_ENV=~/.venvs/cbx uv pip install -q --python ~/.venvs/cbx/bin/python \
    "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox@$CHATTERBOX_COMMIT" soundfile
~/.venvs/cbx/bin/python -c "import chatterbox, torch; print('cbx ok', torch.__version__, torch.cuda.is_available())"

VIRTUAL_ENV=~/.venvs/sg uv pip install -q --python ~/.venvs/sg/bin/python \
    "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu124
VIRTUAL_ENV=~/.venvs/sg uv pip install -q --python ~/.venvs/sg/bin/python \
    "faster-whisper>=1.0,<2" "jiwer>=3,<5" "transformers>=4.40,<5"
~/.venvs/sg/bin/python -c "import faster_whisper, jiwer, transformers; print('sg ok', transformers.__version__)"
