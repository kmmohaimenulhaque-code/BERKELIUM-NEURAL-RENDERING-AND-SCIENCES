#!/usr/bin/env bash
# Serve Qwen3-32B (+ optional Berkelium LoRA adapter dir) on the MI300X, OpenAI-compatible API on :8000.
#   scripts/serve_vllm.sh                                   # base model, served as Qwen/Qwen3-32B
#   scripts/serve_vllm.sh training/runs/qwen3-32b-lora      # base + adapter, adapter served as "berkelium"
# Wait for readiness with:  berkelium model-check --wait 1800
# Env: VLLM_MODE=auto|venv|path|docker  VLLM_VENV=.venv-vllm  VLLM_IMAGE=vllm/vllm-openai-rocm:<tag>
#      MODEL=Qwen/Qwen3-32B  PORT=8000  GPU_UTIL=0.90  MAX_LEN=16384
# vLLM reserves GPU_UTIL of VRAM until it exits: stop it (pkill -f "vllm serve") before training.
set -euo pipefail
ADAPTER="${1:-}"
MODEL="${MODEL:-Qwen/Qwen3-32B}"; PORT="${PORT:-8000}"; GPU_UTIL="${GPU_UTIL:-0.90}"; MAX_LEN="${MAX_LEN:-16384}"
MODE="${VLLM_MODE:-auto}"; VENV="${VLLM_VENV:-.venv-vllm}"
ARGS=("$MODEL" --dtype bfloat16 --max-model-len "$MAX_LEN" --port "$PORT" --seed 0
      --gpu-memory-utilization "$GPU_UTIL" --host 127.0.0.1)
if [[ -n "$ADAPTER" ]]; then
  [[ -f "$ADAPTER/adapter_config.json" ]] || { echo "no adapter_config.json in $ADAPTER" >&2; exit 2; }
  ARGS+=(--enable-lora --max-lora-rank 16 --lora-modules "berkelium=$ADAPTER")
fi
if [[ "$MODE" == auto ]]; then
  if [[ -x "$VENV/bin/vllm" ]]; then MODE=venv
  elif command -v vllm >/dev/null; then MODE=path
  elif command -v docker >/dev/null; then MODE=docker
  else
    echo "vLLM not found. Run scripts/install_vllm_rocm.sh (separate venv), or install Docker and use" >&2
    echo "VLLM_MODE=docker VLLM_IMAGE=vllm/vllm-openai-rocm:<tag>. Do NOT pip install vllm into the" >&2
    echo "training venv: its wheel pins its own torch and can replace your ROCm torch." >&2
    exit 127
  fi
fi
case "$MODE" in
  venv) exec "$VENV/bin/vllm" serve "${ARGS[@]}" ;;
  path) exec vllm serve "${ARGS[@]}" ;;
  docker)
    IMAGE="${VLLM_IMAGE:?set VLLM_IMAGE, e.g. vllm/vllm-openai-rocm:<tag> from hub.docker.com/r/vllm/vllm-openai-rocm/tags}"
    MOUNTS=(-v "$HOME/.cache/huggingface:/root/.cache/huggingface")
    if [[ -n "$ADAPTER" ]]; then
      ABS="$(cd "$ADAPTER" && pwd)"; MOUNTS+=(-v "$ABS:/adapter:ro")
      ARGS=("${ARGS[@]/berkelium=$ADAPTER/berkelium=/adapter}")
    fi
    ARGS=("${ARGS[@]/127.0.0.1/0.0.0.0}")
    exec docker run --rm --network host --ipc host --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined "${MOUNTS[@]}" -e HF_TOKEN --entrypoint vllm "$IMAGE" serve "${ARGS[@]}" ;;
  *) echo "unknown VLLM_MODE=$MODE" >&2; exit 2 ;;
esac
