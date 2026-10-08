#!/usr/bin/env bash
# Berkelium MI300X setup. Never touches the ROCm torch install.
# Usage (repo root):  bash scripts/setup_mi300x.sh
set -euo pipefail

echo "== 0. Preconditions =="
[ -f pyproject.toml ] || { echo "run from the repo root"; exit 1; }
[ -n "${VIRTUAL_ENV:-}" ] || { echo "activate your training venv first (source .venv/bin/activate)"; exit 1; }
python - <<'PY'
import sys, torch
assert sys.version_info >= (3, 11), f"Python >= 3.11 required, have {sys.version}"
hip = getattr(torch.version, "hip", None)
assert hip, f"torch {torch.__version__} is not a ROCm build - fix this before continuing"
assert torch.cuda.is_available(), "GPU not visible to torch"
p = torch.cuda.get_device_properties(0)
print(f"OK torch {torch.__version__} | HIP {hip} | {p.name} | {getattr(p, 'gcnArchName', '?')} | {p.total_memory/2**30:.0f} GB")
PY

echo "== 1. System packages (Gmsh/CalculiX run as external processes) =="
if command -v apt-get >/dev/null; then
  SUDO=$([ "$(id -u)" -eq 0 ] && echo "" || echo "sudo")
  $SUDO apt-get update -y
  $SUDO apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    libglu1-mesa libxcursor1 libxinerama1 libxft2 libxrender1 libgl1 \
    calculix-ccx
fi

echo "== 2. Python deps (all extras; torch is in none of them) =="
python -m pip install -U pip wheel setuptools
python -m pip install -e ".[occt,mesh,api,dev,physics,model,train]"
# Gmsh CLI (separate process; never imported by Berkelium)
command -v gmsh >/dev/null || python -m pip install "gmsh>=4.15"

echo "== 3. Re-check torch was not replaced =="
python -c "import torch; assert torch.version.hip, 'torch was replaced by a non-ROCm build'; print('torch still ROCm:', torch.__version__)"

echo "== 4. vLLM in its OWN venv (.venv-vllm) - its wheel pins its own torch =="
if [ ! -x .venv-vllm/bin/vllm ]; then
  bash scripts/install_vllm_rocm.sh || echo "WARN: vLLM wheel failed; use Docker: VLLM_MODE=docker VLLM_IMAGE=vllm/vllm-openai-rocm:<tag>"
fi

echo "== 5. Verify =="
mkdir -p out/logs out/eval
python scripts/env_manifest.py > out/env_manifest_mi300x.json
python - <<'PY'
import importlib
for m in ["numpy","scipy","sympy","pydantic","skfem","meshio","OCP","manifold3d","trimesh",
          "fastapi","httpx","transformers","peft","accelerate","datasets","yaml","pytest"]:
    try: mod = importlib.import_module(m); print(f"  {m:13s} {getattr(mod,'__version__','ok')}")
    except Exception as e: print(f"  {m:13s} MISSING ({e})")
PY
gmsh --version 2>/dev/null | tail -1 | sed 's/^/  gmsh /' || echo "  gmsh MISSING"
command -v ccx >/dev/null && echo "  ccx ok" || echo "  ccx missing (optional)"
pytest -q -p no:warnings
echo "== DONE. Manifest: out/env_manifest_mi300x.json. Next: docs/MI300X_ENDGAME_RUNBOOK.md step 1 =="
