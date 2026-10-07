"""Qwen3-32B LoRA SFT with PEFT + TRL. Run only after training/smoke_test.py succeeds.

    python training/sft_lora.py --config training/configs/qwen3_32b_lora.yaml
Writes adapters + a run manifest (config hash, dataset hashes, library versions) to output_dir."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    import peft
    import torch
    import transformers
    import trl
    import yaml
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    raw = Path(a.config).read_bytes()
    cfg = yaml.safe_load(raw)
    t, lc, d = cfg["train"], cfg["lora"], cfg["data"]
    ds = load_dataset("json", data_files={"train": d["train"], "validation": d["val"]})
    kw = dict(output_dir=cfg["output_dir"], num_train_epochs=t["epochs"],
              per_device_train_batch_size=t["per_device_batch_size"],
              gradient_accumulation_steps=t["gradient_accumulation_steps"], learning_rate=t["learning_rate"],
              lr_scheduler_type=t["lr_scheduler"], warmup_ratio=t["warmup_ratio"], weight_decay=t["weight_decay"],
              logging_steps=t["logging_steps"], eval_strategy="steps", eval_steps=t["eval_steps"],
              save_steps=t["save_steps"], bf16=True, gradient_checkpointing=cfg["gradient_checkpointing"],
              seed=t["seed"], report_to="none")
    params = inspect.signature(SFTConfig.__init__).parameters
    kw["max_length" if "max_length" in params else "max_seq_length"] = cfg["max_length"]
    if "assistant_only_loss" in params:
        kw["assistant_only_loss"] = d["assistant_only_loss"]
    elif d["assistant_only_loss"]:
        raise SystemExit("installed TRL lacks assistant_only_loss; upgrade TRL rather than train on prompts")
    args = SFTConfig(**kw)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg.get("revision"), torch_dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("revision"))
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                         processing_class=tok,
                         peft_config=LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                                target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    Path(cfg["output_dir"], "run_manifest.json").write_text(json.dumps({
        "config_sha256": hashlib.sha256(raw).hexdigest(), "train_sha256": sha(d["train"]),
        "val_sha256": sha(d["val"]), "versions": {"torch": torch.__version__, "hip": torch.version.hip,
                                                  "transformers": transformers.__version__, "peft": peft.__version__,
                                                  "trl": trl.__version__}}, indent=2))


if __name__ == "__main__":
    main()
