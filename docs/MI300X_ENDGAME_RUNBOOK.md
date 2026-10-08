# MI300X endgame runbook — model generations BASE, M1, AGENT-V1, AGENT-V2, AGENT-V2-DPO (ADR-022..024)

Every adapter is a NEW directory; nothing overwrites an earlier generation. vLLM and training never share the GPU.
Run from the repo root, `.venv` active, on `physics/p1-numerical`.

## 0. Audit (never reinstall the ROCm stack)
```bash
git pull && pip install -e ".[physics,train,api,dev]"          # torch is in no extra: ROCm torch is kept
python scripts/env_manifest.py > out/env_manifest_mi300x.json   # GPU, gfx arch, ROCm/HIP, versions, git head
pytest -q                                                        # must be green before any training
```
## 1. Data (CPU, < 2 min) — every file has a manifest with hashes and split policy
```bash
python scripts/agent_build_training.py --out out/agent                         # AGENT-V1: ClaimEnv
python scripts/model_build_training.py --out out/agent_v2 --merge out/agent    # AGENT-V2: + ModelEnv (seed 1)
```
## 2. Baselines on BOTH environments (one vLLM server serves base + M1 names)
```bash
scripts/serve_vllm.sh training/runs/qwen3-32b-lora > out/logs/vllm-m1.log 2>&1 &
export BERKELIUM_MODEL_URL=http://127.0.0.1:8000/v1
for M in Qwen/Qwen3-32B berkelium; do L=$([ $M = berkelium ] && echo m1 || echo base)
  BERKELIUM_MODEL=$M berkelium model-check --wait 1800
  BERKELIUM_MODEL=$M python scripts/agent_llm_eval.py --split heldout --label $L --out out/eval/claim_$L.json
  BERKELIUM_MODEL=$M python scripts/model_llm_eval.py --split heldout --label $L --out out/eval/model_$L.json
  BERKELIUM_MODEL=$M python scripts/model_llm_eval.py --split train   --label $L --out out/eval/model_${L}_seen.json
done; pkill -f "vllm serve"; sleep 20
```
## 3. AGENT-V1 (ClaimEnv SFT) and AGENT-V2 (ClaimEnv + ModelEnv SFT) — smoke test first, always
```bash
python training/smoke_test.py --config training/configs/qwen3_32b_lora_agent_v2.yaml --data out/agent_v2/train.chat.jsonl --steps 10 --out training/runs/smoke_v2.json
python training/sft_lora.py --config training/configs/qwen3_32b_lora_agent.yaml       # -> ...-agent-v1
python training/sft_lora.py --config training/configs/qwen3_32b_lora_agent_v2.yaml    # -> ...-agent-v2
```
## 4. AGENT-V2-DPO (own DPO implementation, frozen precomputed reference = V2 SFT adapter)
```bash
python training/dpo_lora.py --config training/configs/qwen3_32b_dpo_agent_v2.yaml     # -> ...-agent-v2-dpo
```
## 5. Evaluate each generation (repeat per adapter; ADAPTER_NAME is the served name)
```bash
for G in agent-v1 agent-v2 agent-v2-dpo; do
  ADAPTER_NAME=$G scripts/serve_vllm.sh training/runs/qwen3-32b-lora-$G > out/logs/vllm-$G.log 2>&1 &
  BERKELIUM_MODEL=$G berkelium model-check --wait 1800
  BERKELIUM_MODEL=$G python scripts/agent_llm_eval.py --split heldout --label $G --out out/eval/claim_$G.json
  BERKELIUM_MODEL=$G python scripts/model_llm_eval.py --split heldout --label $G --out out/eval/model_$G.json
  BERKELIUM_MODEL=$G python scripts/model_llm_eval.py --split train   --label $G --out out/eval/model_${G}_seen.json
  pkill -f "vllm serve"; sleep 20
done
```
## 6. Close the loop: failures -> curriculum -> AGENT-V3
```bash
python scripts/failure_analysis.py out/eval/model_agent-v2-dpo_seen.trajectories.jsonl --out out/curriculum_v3.json
python scripts/model_build_training.py --out out/agent_v3 --merge out/agent --seed 2 --curriculum out/curriculum_v3.json
# copy qwen3_32b_lora_agent_v2.yaml -> _v3.yaml (data: out/agent_v3, output_dir: ...-agent-v3), then steps 3-5.
```
Curriculum is built ONLY from TRAIN-domain failures (the `_seen` eval); held-out domains are never trained on, so the
held-out numbers remain a transfer measurement. The benchmark (seed 0) is never used for training (seeds >= 1).

## 7. Commit the evidence
Commit `out/eval/*.json`, every `training/runs/*/run_manifest.json`, `out/*/manifest.json`, `out/env_manifest_mi300x.json`
(not the adapters). Primary metrics: `grounded_correct_rate`, `wrong_rate` (must not rise), `ungrounded_rate`, `mean_return`;
`status_only_accuracy` / `answer_only_accuracy` are reported to expose answer-matching that is not grounded.
