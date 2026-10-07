# MI300X runbook: Qwen3-32B base vs M1 LoRA vs new agent LoRA on ClaimEnv (ADR-022)

Run from the repo root with `.venv` active. vLLM and training cannot share the GPU (vLLM holds ~90 % VRAM).

## 0. Update + dependencies
```bash
git fetch origin && git checkout physics/p1-numerical && git pull
pip install -e ".[physics,train,api,dev]"        # torch is NOT in any extra: your ROCm torch stays
python -c "import torch;print(torch.__version__, torch.version.hip)"   # must still show +rocm
pytest -q tests/test_agent_env.py tests/test_agent_llm.py tests/test_evidence.py tests/test_laws.py tests/test_discovery_memory.py
# optional (only for gmsh-based physics tests, which skip without it): apt-get install -y gmsh libglu1-mesa
```
## 1. Build the verified training data (CPU, seconds)
```bash
python scripts/agent_build_training.py --out out/agent     # -> train/val chat files, pairs.jsonl, manifest.json
```
## 2. Baselines: base Qwen3-32B and the M1 LoRA (ONE vLLM server serves both names)
```bash
mkdir -p out/logs
scripts/serve_vllm.sh training/runs/qwen3-32b-lora > out/logs/vllm-m1.log 2>&1 &
export BERKELIUM_MODEL_URL=http://127.0.0.1:8000/v1
BERKELIUM_MODEL=Qwen/Qwen3-32B berkelium model-check --wait 1800
BERKELIUM_MODEL=Qwen/Qwen3-32B python scripts/agent_llm_eval.py --split heldout --label base --out out/agent/eval_base.json
BERKELIUM_MODEL=berkelium      python scripts/agent_llm_eval.py --split heldout --label m1_lora --out out/agent/eval_m1.json
pkill -f "vllm serve"; sleep 20
```
## 3. Train the NEW adapter (smoke first)
```bash
python training/smoke_test.py --config training/configs/qwen3_32b_lora_agent.yaml --data out/agent/train.chat.jsonl --steps 10 --out training/runs/smoke_agent.json
python training/sft_lora.py --config training/configs/qwen3_32b_lora_agent.yaml    # -> training/runs/qwen3-32b-lora-agent-v1
```
## 4. Evaluate the new adapter
```bash
ADAPTER_NAME=agent scripts/serve_vllm.sh training/runs/qwen3-32b-lora-agent-v1 > out/logs/vllm-agent.log 2>&1 &
BERKELIUM_MODEL=agent berkelium model-check --wait 1800
BERKELIUM_MODEL=agent python scripts/agent_llm_eval.py --split heldout --label agent_v1 --out out/agent/eval_agent.json
pkill -f "vllm serve"
```
## 5. Compare
```bash
python - <<'PY'
import json
for n in ("base","m1","agent"):
    try: r=json.load(open(f"out/agent/eval_{n}.json"))["llm"]
    except FileNotFoundError: continue
    print(n, {k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items() if k!="tasks"})
PY
```
Exit code 3 from model-check / eval = endpoint not ready or lost (no report written). A valid report has `invalid_action_rate`
small and the reference rows (`reference_cheapest_sufficient`, `reference_overconfident`) show the scale: ~+0.5 / -1.0.

Preference data (`out/agent/pairs.jsonl`, TRL conversational DPO format) is built but there is NO DPO script yet: do SFT first.
