"""MI300X smoke test (run BEFORE any long training). Same code path as sft_lora.py (training/common.py):
environment, free-VRAM guard, tokenizer, masking, model load, LoRA, forward, backward, optimizer step,
N steps on real Berkelium examples, peak memory. Writes a JSON report.

    python training/smoke_test.py --config training/configs/qwen3_32b_lora.yaml \\
        --data out/sft/train.chat.jsonl --steps 10 --out training/runs/smoke.json
Use --model Qwen/Qwen3-0.6B for a CPU/small-GPU dry run of the same code path."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import encode, load_model, lora_config, read_rows, require_free_gpu  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--min-free-gb", type=float, default=120.0)
    ap.add_argument("--out", default="training/runs/smoke.json")
    a = ap.parse_args()
    import torch
    import yaml
    from peft import get_peft_model
    from transformers import AutoTokenizer

    cfg = yaml.safe_load(Path(a.config).read_text())
    model_id = a.model or cfg["model"]
    max_len = a.max_length or cfg["max_length"]
    rep: dict = {"python": platform.python_version(), "torch": torch.__version__,
                 "hip": getattr(torch.version, "hip", None), "model": model_id, "steps": a.steps, "stages": {}}
    rep["gpu"] = require_free_gpu(a.min_free_gb if a.model is None else 0.0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def stage(name, fn):
        t = time.perf_counter()
        out = fn()
        rep["stages"][name] = {"seconds": round(time.perf_counter() - t, 2),
                               "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == "cuda" else None}
        return out

    tok = stage("tokenizer", lambda: AutoTokenizer.from_pretrained(model_id, revision=cfg.get("revision")))
    rows = read_rows(a.data)[: max(a.steps, 1)]
    enc = stage("encode", lambda: [encode(tok, r["messages"], max_len) for r in rows])
    rep["tokens_per_example"] = [len(e["input_ids"]) for e in enc]
    rep["target_tokens_per_example"] = [e["n_target"] for e in enc]
    rep["truncated"] = sum(e["truncated"] for e in enc)
    if any(e["n_target"] == 0 for e in enc):
        raise SystemExit("an example has no target tokens after masking/truncation")
    model = stage("load", lambda: load_model(cfg, model_id))
    model = get_peft_model(model, lora_config(cfg))
    model.enable_input_require_grads()
    rep["trainable_params"], rep["total_params"] = model.get_nb_trainable_parameters()
    batches = [(torch.tensor([e["input_ids"]], device=dev), torch.tensor([e["labels"]], device=dev)) for e in enc]
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
    rep["ok"] = all(x == x and x != float("inf") for x in losses)  # no NaN/inf
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: rep[k] for k in ("ok", "gpu", "losses", "stages")}, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
