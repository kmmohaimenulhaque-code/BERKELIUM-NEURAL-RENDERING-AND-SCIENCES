# Implementation status

Branch `core/m1-foundation`. Everything below is implemented and tested on Linux (Python 3.12, CPU) unless marked.

| Area | Module | Status |
|---|---|---|
| Units, expression language (AST, Pratt parser, dimension-checked eval, NumPy/closure field compile, no eval) | `berkelium.units`, `berkelium.expr` | done, tested |
| Canonical schemas (DesignProposal, DesignRecord, all layers), JSON Schema export, RFC 8785 hashing, migrations | `berkelium.schema` | done, tested |
| Geometry IR (primitives, profiles, extrude/revolve, booleans, transforms, patterns, implicit fields, tags, deps, expression args) | `berkelium.geometry.ir` | done, tested |
| OCCT exact-BREP backend (deterministic STEP; lossy-flagged STL/GLB) | `geometry.occt` | done, tested (OCP 8.0.1, 7.x imports supported) |
| Manifold mesh backend + level sets; capability planner (field → STEP refused) | `geometry.manifold_backend`, `geometry.backend` | done, tested |
| Validation L0–L4, L5 (envelope; other checks `not_evaluated`), L6 gear, L7 always `not_evaluated` | `berkelium.validation`, `manufacturing` | done, tested |
| CEM protocol, registry (entry points `berkelium.cems`) | `berkelium.cem.protocol` | done |
| Spur gear pair CEM (sourced formulas, rack-simulated involute + trochoid fillet, derive, checks, mesh-interference) | `cem.library.gear` | done, oracle-tested |
| Creative procedural designs (lantern BREP, gyroid orb field) | `berkelium.designs` | done, tested |
| Deterministic pipeline → content-hashed DesignRecord, artifact store, job stages | `berkelium.pipeline` | done, tested |
| Dataset factory (plan / repair / procedural; verified; family splits; manifests; rejected log) | `berkelium.datasets` | done, tested |
| Model gateway (OpenAI-compatible vLLM/Fireworks, replay), orchestrator + JSON-Patch repair loop, evaluation harness | `berkelium.ai` | done, tested offline (fake server) |
| SFT rendering + length audit | `berkelium.training.data` | done, tested |
| MI300X smoke test, LoRA SFT (PEFT + transformers.Trainer, shared tokenisation/masking in `training/common.py`), vLLM install/serve scripts | `training/`, `scripts/` | CPU dry-run passed end-to-end with Qwen3-0.6B (smoke + 2-step SFT + adapter save); **not yet run on MI300X** |
| Endpoint robustness: transport errors → `GatewayError`, loopback bypasses proxies, `berkelium model-check --wait`, eval preflight + abort after 3 consecutive endpoint failures (no fake 0 % baseline) | `berkelium.ai`, `berkelium.cli` | done, tested |
| Core API v1 (schemas, CEMs, designs, JSON-Patch revisions, validation, artifacts, intents) + CLI | `berkelium.api`, `berkelium.cli` | done, tested |
| Zero-shot Qwen3-32B baseline | `berkelium eval` | **blocked**: needs a Qwen3-32B endpoint (MI300X vLLM or Fireworks) |
| Studio integration | — | not started |

## Run order on the MI300X

Two environments: `.venv` (Berkelium + training, keeps your ROCm torch) and `.venv-vllm` (vLLM's own pinned torch).
vLLM and training cannot share the GPU: vLLM reserves 90 % of VRAM until it exits.

```bash
# 0. setup (training venv)
source .venv/bin/activate
pip install -e ".[occt,mesh,api,dev,train]"
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"  # must still be +rocm
pytest
berkelium dataset --out out/ds --seed 0 --n-valid 400 --n-boundary 100 --n-invalid 200 --n-procedural 30
berkelium sft-render --dataset out/ds --out out/sft
scripts/install_vllm_rocm.sh            # once; or VLLM_MODE=docker VLLM_IMAGE=vllm/vllm-openai-rocm:<tag>

# 1. zero-shot baseline
mkdir -p out/logs
scripts/serve_vllm.sh > out/logs/vllm-base.log 2>&1 &
export BERKELIUM_MODEL_URL=http://127.0.0.1:8000/v1 BERKELIUM_MODEL=Qwen/Qwen3-32B
berkelium model-check --wait 1800      # first run downloads ~65 GB, then loads
berkelium eval --dataset out/ds --label base --out out/eval/base.json
pkill -f "vllm serve"; sleep 20

# 2. smoke test, then 3. training (refuse to start if < 120 GB VRAM free)
python training/smoke_test.py --config training/configs/qwen3_32b_lora.yaml --data out/sft/train.chat.jsonl --steps 10 --out training/runs/smoke.json
python training/sft_lora.py --config training/configs/qwen3_32b_lora.yaml

# 4. post-training eval
scripts/serve_vllm.sh training/runs/qwen3-32b-lora > out/logs/vllm-lora.log 2>&1 &
export BERKELIUM_MODEL=berkelium
berkelium model-check --wait 1800
berkelium eval --dataset out/ds --label lora --out out/eval/lora.json
pkill -f "vllm serve"
```

Exit code 3 from `model-check`/`eval` means the endpoint is not ready or was lost; no report is written.
A valid eval report has `transport_error_rate: 0`.
