# MI300X endgame runbook — model generations BASE, M1, AGENT-V1, AGENT-V2, AGENT-V2-DPO (ADR-022..025)

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
## 2. Evaluate ALL generations — one command, no server (ADR-025)
```bash
pkill -f "vllm serve"; sleep 15                    # a server left over from an earlier Ctrl+C still holds the GPU
python scripts/eval_generations.py --limit 2       # smoke: ~10 min incl. model load; proves everything works
python scripts/eval_generations.py                 # full: BASE, M1, AGENT-V1, AGENT-V2, AGENT-V2-DPO
cat out/eval/summary.md
```
- Loads Qwen3-32B ONCE in-process (transformers + PEFT, the stack the adapters were trained with) and switches the
  four adapters in place; adapters are fetched from their Hugging Face repos (root files only).
- Prints progress every 15 s (episodes done, calls/s, ETA). Every finished episode is on disk immediately:
  **Ctrl+C is safe — rerun the same command to resume.**
- Per generation: 120 ClaimEnv held-out + 28 ModelEnv held-out-domain + 40 ModelEnv seen-domain tasks. Trained
  adapters finish in minutes; BASE/M1 are slowest on ModelEnv (they may use all 40 steps per task).
- Options: `--backend vllm` (starts ONE server with all adapters, waits visibly, fails fast, always stops it),
  `--endpoint URL` (existing server), `--generations agent-v2,agent-v2-dpo`, `--suites claim`,
  `--adapter agent-v2=training/runs/qwen3-32b-lora-agent-v2` (local copy), `--suites planning` (M1's own task,
  gear-only prompt as trained; slow, BASE and M1 only), `--batch 8` if memory is tight.

### If something looks stuck
| Symptom | Cause | Fix |
|---|---|---|
| `model-check --wait` prints nothing for minutes (old runbook) | silent polling while vLLM downloads/loads, or after it crashed | use step 2 above; `model-check` now prints every 30 s and aborts if the server died |
| vLLM log: `LoRA rank 32 is greater than max_lora_rank 16` | old serve script hard-coded rank 16; AGENT-V2/DPO are r=32 | fixed: rank is read from the adapters |
| `no adapter_config.json in training/runs/...` | adapters were trained on another machine/session | fixed: HF repo ids are downloaded automatically |
| eval runs for hours with no output (old runbook) | one request at a time, up to 4096 tokens each, temperature 0.7 | fixed: batched, greedy, 256-token cap, progress + ETA |
| `GPU busy: only N GB free` | leftover vLLM from an earlier Ctrl+C | `pkill -f "vllm serve"; sleep 15` |

## 3. Training AGENT-V1 / AGENT-V2 (DONE 2026-10-08 — adapters on Hugging Face; rerun only for a new generation)
```bash
python training/smoke_test.py --config training/configs/qwen3_32b_lora_agent_v2.yaml --data out/agent_v2/train.chat.jsonl --steps 10 --out training/runs/smoke_v2.json
python training/sft_lora.py --config training/configs/qwen3_32b_lora_agent.yaml       # -> ...-agent-v1
python training/sft_lora.py --config training/configs/qwen3_32b_lora_agent_v2.yaml    # -> ...-agent-v2
```
## 4. AGENT-V2-DPO (DONE 2026-10-08 — own DPO, frozen precomputed reference = V2 SFT adapter)
```bash
python training/dpo_lora.py --config training/configs/qwen3_32b_dpo_agent_v2.yaml     # -> ...-agent-v2-dpo
```
## 5. Close the loop: failures -> curriculum -> AGENT-V3
```bash
python scripts/failure_analysis.py out/eval/model_seen__agent-v2-dpo.trajectories.jsonl --out out/curriculum_v3.json
python scripts/model_build_training.py --out out/agent_v3 --merge out/agent --seed 2 --curriculum out/curriculum_v3.json
# copy qwen3_32b_lora_agent_v2.yaml -> _v3.yaml (data: out/agent_v3, output_dir: ...-agent-v3), train (step 3),
# then: python scripts/eval_generations.py --generations agent-v2-dpo,agent-v3 --adapter agent-v3=training/runs/qwen3-32b-lora-agent-v3
```
Curriculum is built ONLY from TRAIN-domain failures (the `_seen` eval); held-out domains are never trained on, so the
held-out numbers remain a transfer measurement. The benchmark (seed 0) is never used for training (seeds >= 1).

## 6. Commit the evidence
Commit `out/eval/*.json`, `out/eval/summary.md`, `out/eval/eval_manifest.json`, every `training/runs/*/run_manifest.json`, `out/*/manifest.json`, `out/env_manifest_mi300x.json`
(not the adapters). Primary metrics: `grounded_correct_rate`, `wrong_rate` (must not rise), `ungrounded_rate`, `mean_return`;
`status_only_accuracy` / `answer_only_accuracy` are reported to expose answer-matching that is not grounded.
