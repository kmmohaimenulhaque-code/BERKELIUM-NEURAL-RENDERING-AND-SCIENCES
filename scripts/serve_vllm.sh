#!/usr/bin/env bash
# Serve Qwen3-32B (+ optional Berkelium LoRA) on MI300X with vLLM-ROCm, OpenAI-compatible API on :8000.
# Then: export BERKELIUM_MODEL_URL=http://localhost:8000/v1 BERKELIUM_MODEL=Qwen/Qwen3-32B  (or =berkelium)
set -euo pipefail
ADAPTER="${1:-}"
ARGS=(Qwen/Qwen3-32B --dtype bfloat16 --max-model-len 16384 --port 8000 --seed 0)
if [[ -n "$ADAPTER" ]]; then ARGS+=(--enable-lora --max-lora-rank 16 --lora-modules "berkelium=$ADAPTER"); fi
exec vllm serve "${ARGS[@]}"
