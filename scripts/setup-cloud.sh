#!/usr/bin/env bash
# Cloud Linux development setup. Windows uses setup.ps1/install-neural.ps1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.cache/pip}"
python3 -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12"'
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
if [[ "${1:-}" == "--neural" ]]; then
  python3 -m venv /workspace/.neural
  /workspace/.neural/bin/python -m pip install -r requirements.txt
  /workspace/.neural/bin/python -m pip install 'torch==2.7.1+cpu' 'torchvision==0.22.1+cpu' --index-url https://download.pytorch.org/whl/cpu
  /workspace/.neural/bin/python -m pip install -r requirements-neural.txt
  /workspace/.neural/bin/python -m pip check
  /workspace/.neural/bin/python -c 'from src.models.weights import teacher_weights; path, _ = teacher_weights({"allow_download": True}); print("Verified EfficientAD teacher:", path)'
fi
