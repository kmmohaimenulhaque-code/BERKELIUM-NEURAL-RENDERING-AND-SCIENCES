#!/usr/bin/env bash
# Install vLLM (ROCm build) into its OWN venv, .venv-vllm, so its pinned torch cannot overwrite the
# training venv's ROCm torch. Source: vLLM ROCm wheel index (wheels.vllm.ai/rocm), Python 3.12.
# If this fails on your ROCm version, use the Docker route in scripts/serve_vllm.sh (VLLM_MODE=docker).
set -euo pipefail
VENV="${VLLM_VENV:-.venv-vllm}"
python3.12 -m venv "$VENV"
"$VENV/bin/pip" install -U pip
"$VENV/bin/pip" install vllm --extra-index-url https://wheels.vllm.ai/rocm/
"$VENV/bin/python" - <<'PY'
import torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "hip", torch.version.hip,
      "gpu", torch.cuda.is_available() and torch.cuda.get_device_name(0))
PY
echo "ok: $VENV/bin/vllm"
