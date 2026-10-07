"""Shared by smoke_test.py and sft_lora.py so the smoke test exercises the exact tokenisation, masking and
model-loading path that real training uses (a smoke test on a different path proves nothing).

Masking: loss only on the final assistant turn. The prompt is rendered with the SAME chat-template flags the
gateway uses at inference (enable_thinking=False), so train/inference prompts are byte-identical."""

from __future__ import annotations

import json
from pathlib import Path

IGNORE = -100


def read_rows(path: str | Path) -> list[dict]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    for r in rows:
        if not r.get("messages") or r["messages"][-1]["role"] != "assistant":
            raise ValueError(f"row {r.get('id')}: last message must be the assistant target")
    return rows


def _ids(x) -> list[int]:
    # transformers >= 4.5x may return a BatchEncoding/dict from apply_chat_template(tokenize=True)
    if isinstance(x, dict) or hasattr(x, "keys"):
        x = x["input_ids"]
    return list(x)


def encode(tok, messages: list[dict], max_len: int) -> dict:
    prompt = _ids(tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False))
    full = _ids(tok.apply_chat_template(messages, tokenize=True, enable_thinking=False))
    if full[:len(prompt)] != prompt:
        raise ValueError("chat template: prompt is not a prefix of prompt+answer; masking would be wrong")
    truncated = len(full) > max_len
    full = full[:max_len]
    labels = [IGNORE] * min(len(prompt), len(full)) + full[len(prompt):]
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels,
            "n_target": sum(t != IGNORE for t in labels), "truncated": truncated}


def require_free_gpu(min_free_gb: float) -> dict:
    """Refuse to start if another process (typically a still-running vLLM server, which reserves ~90 % of
    VRAM by default) holds the GPU. Returns device info."""
    import torch
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    free, total = torch.cuda.mem_get_info(0)
    info = {"device": torch.cuda.get_device_properties(0).name, "free_gb": round(free / 2**30, 1),
            "total_gb": round(total / 2**30, 1), "count": torch.cuda.device_count()}
    if free / 2**30 < min_free_gb:
        raise SystemExit(f"only {info['free_gb']} GB of {info['total_gb']} GB free; need >= {min_free_gb}. "
                         "Stop the vLLM server (pkill -f 'vllm serve') before training.")
    return info


def load_model(cfg: dict, model_id: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM
    kw = dict(revision=cfg.get("revision"), torch_dtype=torch.bfloat16,
              attn_implementation=cfg.get("attn_implementation", "sdpa"), low_cpu_mem_usage=True)
    if torch.cuda.is_available():
        kw["device_map"] = {"": 0}   # stream weights straight to the GPU; no 65 GB host-RAM copy
    model = AutoModelForCausalLM.from_pretrained(model_id or cfg["model"], **kw)
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model


def lora_config(cfg: dict):
    from peft import LoraConfig
    lc = cfg["lora"]
    return LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                      target_modules=lc["target_modules"], task_type="CAUSAL_LM")
