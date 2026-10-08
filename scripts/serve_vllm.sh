#!/usr/bin/env bash
# Serve Qwen3-32B + ANY NUMBER of Berkelium LoRA adapters from ONE vLLM server (OpenAI API on :8000).
#
#   scripts/serve_vllm.sh                                        # base only, served as Qwen/Qwen3-32B
#   scripts/serve_vllm.sh m1=Mhaquehaque/berkelium-qwen3-32b-lora agent-v2=training/runs/qwen3-32b-lora-agent-v2
#   scripts/serve_vllm.sh training/runs/qwen3-32b-lora           # legacy: one adapter, served as $ADAPTER_NAME (berkelium)
#
# Each adapter is NAME=SOURCE; SOURCE is a local adapter dir or a Hugging Face repo id (downloaded once to
# models/adapters/NAME, root files only). --max-lora-rank is computed from the adapters' own adapter_config.json
# (AGENT-V2 / V2-DPO are r=32: the old hard-coded 16 made vLLM refuse them and exit).
# Writes the server PID to $VLLM_PIDFILE (default out/logs/vllm.pid) so `berkelium model-check` can fail fast.
# Env: VLLM_MODE=auto|venv|path|docker  VLLM_VENV=.venv-vllm  VLLM_IMAGE=...  MODEL=Qwen/Qwen3-32B  PORT=8000
#      GPU_UTIL=0.90  MAX_LEN=16384  VLLM_EAGER=1 (skip graph capture: faster start, slower decode)  PYTHON=python3
# Stop it with: pkill -f "vllm serve"   (Ctrl+C on a later command does NOT stop a backgrounded server)
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-32B}"; PORT="${PORT:-8000}"; GPU_UTIL="${GPU_UTIL:-0.90}"; MAX_LEN="${MAX_LEN:-16384}"
MODE="${VLLM_MODE:-auto}"; VENV="${VLLM_VENV:-.venv-vllm}"; PY="${PYTHON:-python3}"
PIDFILE="${VLLM_PIDFILE:-out/logs/vllm.pid}"; mkdir -p "$(dirname "$PIDFILE")"

NAMES=(); PATHS=()
for spec in "$@"; do
  if [[ "$spec" == *=* ]]; then name="${spec%%=*}"; src="${spec#*=}"; else name="${ADAPTER_NAME:-berkelium}"; src="$spec"; fi
  if [[ ! -f "$src/adapter_config.json" ]]; then
    if [[ ! -e "$src" && "$src" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then   # owner/repo, not a path
      dest="models/adapters/$name"
      echo "downloading adapter $name from Hugging Face repo $src -> $dest" >&2
      "$PY" - "$src" "$dest" <<'PYDL'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["adapter_config.json", "adapter_model.safetensors", "tokenizer*", "chat_template.jinja", "run_manifest.json"])
PYDL
      src="$dest"
    fi
    [[ -f "$src/adapter_config.json" ]] || { echo "adapter $name: no adapter_config.json in $src" >&2; exit 2; }
  fi
  NAMES+=("$name"); PATHS+=("$src")
done

ARGS=("$MODEL" --dtype bfloat16 --max-model-len "$MAX_LEN" --port "$PORT" --seed 0
      --gpu-memory-utilization "$GPU_UTIL" --host 127.0.0.1)
[[ "${VLLM_EAGER:-0}" == 1 ]] && ARGS+=(--enforce-eager)
if (( ${#NAMES[@]} )); then
  RANK=$("$PY" - "${PATHS[@]}" <<'PYR'
import json, sys
r = max(json.load(open(f"{p}/adapter_config.json"))["r"] for p in sys.argv[1:])
print(next(x for x in (8, 16, 32, 64, 128, 256, 512) if x >= r))
PYR
)
  MODS=(); for i in "${!NAMES[@]}"; do MODS+=("${NAMES[$i]}=${PATHS[$i]}"); done
  ARGS+=(--enable-lora --max-lora-rank "$RANK" --max-loras "${#NAMES[@]}" --lora-modules "${MODS[@]}")
  echo "serving $MODEL + adapters: ${NAMES[*]} (max-lora-rank $RANK)" >&2
fi

if [[ "$MODE" == auto ]]; then
  if [[ -x "$VENV/bin/vllm" ]]; then MODE=venv
  elif command -v vllm >/dev/null; then MODE=path
  elif command -v docker >/dev/null; then MODE=docker
  else
    echo "vLLM not found. Either use the no-server evaluator: python scripts/eval_generations.py (backend local)," >&2
    echo "or run scripts/install_vllm_rocm.sh, or Docker: VLLM_MODE=docker VLLM_IMAGE=vllm/vllm-openai-rocm:<tag>." >&2
    exit 127
  fi
fi
echo $$ > "$PIDFILE"     # exec keeps this PID: it becomes the server's PID
case "$MODE" in
  venv) exec "$VENV/bin/vllm" serve "${ARGS[@]}" ;;
  path) exec vllm serve "${ARGS[@]}" ;;
  docker)
    IMAGE="${VLLM_IMAGE:?set VLLM_IMAGE, e.g. vllm/vllm-openai-rocm:<tag> from hub.docker.com/r/vllm/vllm-openai-rocm/tags}"
    MOUNTS=(-v "$HOME/.cache/huggingface:/root/.cache/huggingface")
    for i in "${!NAMES[@]}"; do
      ABS="$(cd "${PATHS[$i]}" && pwd)"; MOUNTS+=(-v "$ABS:/adapters/${NAMES[$i]}:ro")
      ARGS=("${ARGS[@]/${NAMES[$i]}=${PATHS[$i]}/${NAMES[$i]}=/adapters/${NAMES[$i]}}")
    done
    ARGS=("${ARGS[@]/127.0.0.1/0.0.0.0}")
    exec docker run --rm --network host --ipc host --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined "${MOUNTS[@]}" -e HF_TOKEN --entrypoint vllm "$IMAGE" serve "${ARGS[@]}" ;;
  *) echo "unknown VLLM_MODE=$MODE" >&2; exit 2 ;;
esac
