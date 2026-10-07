"""MI300X smoke test (run BEFORE any long training): environment, model load, tokenizer, forward,
backward, optimizer step, N steps on real Berkelium examples, peak memory. Writes a JSON report.

    python training/smoke_test.py --config training/configs/qwen3_32b_lora.yaml \
        --data out/sft/train.chat.jsonl --steps 10 --out training/runs/smoke.json
Use --model Qwen/Qwen3-0.6B for a CPU/small-GPU dry run of the same code path."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--out", default="training/runs/smoke.json")
    a = ap.parse_args()
    import torch
    import yaml
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = yaml.safe_load(Path(a.config).read_text())
    model_id = a.model or cfg["model"]
    max_len = a.max_length or cfg["max_length"]
    rep: dict = {"python": platform.python_version(), "torch": torch.__version__,
                 "hip": getattr(torch.version, "hip", None), "cuda_available": torch.cuda.is_available(),
                 "model": model_id, "steps": a.steps, "stages": {}}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        p = torch.cuda.get_device_properties(0)
        rep["device"] = {"name": p.name, "total_gb": round(p.total_memory / 2**30, 1), "count": torch.cuda.device_count()}
        torch.cuda.reset_peak_memory_stats()

    def stage(name, fn):
        t = time.perf_counter()
        out = fn()
        rep["stages"][name] = {"seconds": round(time.perf_counter() - t, 2),
                               "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else None}
        return out

    tok = stage("tokenizer", lambda: AutoTokenizer.from_pretrained(model_id, revision=cfg.get("revision")))
    model = stage("load", lambda: AutoModelForCausalLM.from_pretrained(
        model_id, revision=cfg.get("revision"), torch_dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "sdpa")).to(dev))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    tr, tot = model.get_nb_trainable_parameters()
    rep["trainable_params"], rep["total_params"] = tr, tot
    rows = [json.loads(x) for x in Path(a.data).read_text().splitlines() if x.strip()][: max(a.steps, 1)]
    batches = []
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"][:-1], tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False)
        full = tok.apply_chat_template(r["messages"], tokenize=True, enable_thinking=False)
        full = full[:max_len]
        labels = [-100] * min(len(prompt), len(full)) + full[len(prompt):]
        batches.append((torch.tensor([full], device=dev), torch.tensor([labels], device=dev)))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    ids, lab = batches[0]
    loss = stage("forward", lambda: model(input_ids=ids, labels=lab).loss)
    stage("backward", lambda: loss.backward())
    stage("optimizer_step", lambda: (opt.step(), opt.zero_grad()))
    losses = [float(loss)]

    def steps():
        for i in range(1, a.steps):
            x, y = batches[i % len(batches)]
            lo = model(input_ids=x, labels=y).loss
            lo.backward()
            opt.step()
            opt.zero_grad()
            losses.append(float(lo))
    stage("train_steps", steps)
    rep["losses"] = losses
    rep["tokens_per_example"] = [int(b[0].shape[1]) for b in batches]
    rep["ok"] = all(x == x for x in losses)  # no NaN
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: rep[k] for k in ("ok", "losses", "stages")}, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
