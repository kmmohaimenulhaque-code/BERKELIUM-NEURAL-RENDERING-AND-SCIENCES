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
| MI300X smoke test, LoRA SFT (PEFT+TRL), vLLM serve script | `training/`, `scripts/` | written, **not executed** (no GPU here) |
| Core API v1 (schemas, CEMs, designs, JSON-Patch revisions, validation, artifacts, intents) + CLI | `berkelium.api`, `berkelium.cli` | done, tested |
| Zero-shot Qwen3-32B baseline | `berkelium eval` | **blocked**: needs a Qwen3-32B endpoint (MI300X vLLM or Fireworks) |
| Studio integration | — | not started |

## Run order on the MI300X
```bash
pip install -e ".[occt,mesh,api,dev]" && pytest
berkelium dataset --out out/ds --seed 0 --n-valid 400 --n-boundary 100 --n-invalid 200 --n-procedural 30
berkelium sft-render --dataset out/ds --out out/sft
scripts/serve_vllm.sh &                      # base model
export BERKELIUM_MODEL_URL=http://localhost:8000/v1 BERKELIUM_MODEL=Qwen/Qwen3-32B
berkelium eval --dataset out/ds --label base --out out/eval/base.json
python training/smoke_test.py --config training/configs/qwen3_32b_lora.yaml --data out/sft/train.chat.jsonl --steps 10
python training/sft_lora.py --config training/configs/qwen3_32b_lora.yaml        # only after smoke passes
scripts/serve_vllm.sh training/runs/qwen3-32b-lora & ; export BERKELIUM_MODEL=berkelium
berkelium eval --dataset out/ds --label lora --out out/eval/lora.json
```
