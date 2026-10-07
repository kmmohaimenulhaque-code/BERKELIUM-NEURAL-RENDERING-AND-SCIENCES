"""Qwen3-32B bf16 LoRA SFT (PEFT + transformers.Trainer). Run only after training/smoke_test.py succeeds.

    python training/sft_lora.py --config training/configs/qwen3_32b_lora.yaml

Tokenisation/masking come from training/common.py — the exact path the smoke test verified — instead of
TRL's assistant_only_loss, which needs {% generation %} markers that Qwen3's chat template does not have.
Writes the adapter + run_manifest.json (config/dataset hashes, token stats, library versions, losses)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import encode, load_model, lora_config, read_rows, require_free_gpu  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="override (e.g. Qwen/Qwen3-0.6B for a dry run)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--min-free-gb", type=float, default=120.0)
    a = ap.parse_args()
    import peft
    import torch
    import transformers
    import yaml
    from datasets import Dataset
    from peft import get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    raw = Path(a.config).read_bytes()
    cfg = yaml.safe_load(raw)
    t, d = cfg["train"], cfg["data"]
    gpu = require_free_gpu(a.min_free_gb if a.model is None else 0.0)
    model_id = a.model or cfg["model"]
    tok = transformers.AutoTokenizer.from_pretrained(model_id, revision=cfg.get("revision"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def build(path):
        enc = [encode(tok, r["messages"], cfg["max_length"]) for r in read_rows(path)]
        bad = [i for i, e in enumerate(enc) if e["n_target"] == 0]
        if bad:
            raise SystemExit(f"{path}: rows {bad[:5]} have no target tokens after truncation")
        stats = {"rows": len(enc), "truncated": sum(e["truncated"] for e in enc),
                 "max_tokens": max(len(e["input_ids"]) for e in enc)}
        ds = Dataset.from_list([{k: e[k] for k in ("input_ids", "attention_mask", "labels")} for e in enc])
        return ds, stats

    train_ds, train_stats = build(d["train"])
    val_ds, val_stats = build(d["val"])
    model = get_peft_model(load_model(cfg, model_id), lora_config(cfg))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    import inspect
    targ = inspect.signature(TrainingArguments.__init__).parameters
    # transformers 5 folded warmup_ratio into warmup_steps (a float in [0,1) is a ratio); 4.x has both
    warm = {"warmup_ratio": t["warmup_ratio"]} if "warmup_ratio" in targ else {"warmup_steps": t["warmup_ratio"]}
    args = TrainingArguments(
        output_dir=cfg["output_dir"], num_train_epochs=t["epochs"], max_steps=a.max_steps,
        per_device_train_batch_size=t["per_device_batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=t["gradient_accumulation_steps"], learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler"], weight_decay=t["weight_decay"], **warm,
        logging_steps=t["logging_steps"], eval_strategy="steps", eval_steps=t["eval_steps"],
        save_strategy="steps", save_steps=t["save_steps"], save_total_limit=3,
        bf16=torch.cuda.is_available(), gradient_checkpointing=False,  # already enabled in load_model
        seed=t["seed"], data_seed=t["seed"], report_to="none", remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True))
    trainer.train()
    final_eval = trainer.evaluate()
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    Path(cfg["output_dir"], "run_manifest.json").write_text(json.dumps({
        "model": model_id, "config_sha256": hashlib.sha256(raw).hexdigest(),
        "train_sha256": sha(d["train"]), "val_sha256": sha(d["val"]),
        "train_stats": train_stats, "val_stats": val_stats, "gpu": gpu, "final_eval": final_eval,
        "log_history": trainer.state.log_history,
        "versions": {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                     "transformers": transformers.__version__, "peft": peft.__version__}}, indent=2, default=str))


if __name__ == "__main__":
    main()
