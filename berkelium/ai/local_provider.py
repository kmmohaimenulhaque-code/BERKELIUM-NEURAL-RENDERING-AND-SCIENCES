"""In-process model provider: transformers + PEFT, one base model, many named LoRA adapters (ADR-025).

No server, no ports, no vLLM install: it uses exactly the stack the Berkelium adapters were trained with
(training/common.py), so if training ran on a machine, evaluation runs there too.

  p = LocalHFProvider("Qwen/Qwen3-32B", {"agent-v2": "/path/or/downloaded/adapter"})
  p.use("agent-v2"); p.generate_many([messages, ...])      # batched greedy decoding
  p.use("base")                                            # adapters disabled -> the base model

Prompts are rendered with the model's chat template and ``enable_thinking=False`` — byte-identical to the
training prompt (common.encode) and to what the gateway asks vLLM for. Decoding is greedy. On out-of-memory the
batch is split in halves and retried, so a too-large batch degrades speed, not correctness.
"""

from __future__ import annotations

import time
from contextlib import nullcontext

from .gateway import GatewayError, Generation


class LocalHFProvider:
    name = "local_hf"

    def __init__(self, base: str = "Qwen/Qwen3-32B", adapters: dict[str, str] | None = None,
                 max_new_tokens: int = 512, revision: str | None = None, dtype: str = "bfloat16"):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.base_id, self.max_new_tokens = base, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(base, revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dkey = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        kw = {dkey: getattr(torch, dtype), "low_cpu_mem_usage": True, "attn_implementation": "sdpa"}
        if torch.cuda.is_available():
            kw["device_map"] = {"": 0}
        net = AutoModelForCausalLM.from_pretrained(base, revision=revision, **kw)
        self.adapters = dict(adapters or {})
        if self.adapters:
            from peft import PeftModel
            names = list(self.adapters)
            net = PeftModel.from_pretrained(net, self.adapters[names[0]], adapter_name=names[0])
            for n in names[1:]:
                net.load_adapter(self.adapters[n], adapter_name=n)
        net.eval()
        self.net = net
        self.model = "base"
        eos = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")]
        self.eos = sorted({e for e in eos if isinstance(e, int) and e >= 0})

    # ---- provider interface (same shape as gateway providers)
    def use(self, name: str) -> None:
        if name != "base" and name not in self.adapters:
            raise GatewayError(f"{self.name}: no adapter {name!r}; loaded {sorted(self.adapters)}")
        self.model = name

    def served_models(self) -> list[str]:
        return ["base", *self.adapters]

    def check(self, wait_s: float = 0.0, **_) -> list[str]:
        if self.model not in self.served_models():
            raise GatewayError(f"{self.name}: {self.model!r} not loaded")
        return self.served_models()

    def generate(self, messages, json_schema=None, decoding=None) -> Generation:
        t = time.perf_counter()
        mx = getattr(decoding, "max_tokens", None)
        text = self.generate_many([messages], max_new_tokens=mx)[0]
        return Generation(text, self.model, self.name, time.perf_counter() - t)

    def generate_many(self, batch: list[list[dict]], max_new_tokens: int | None = None) -> list[str]:
        if not batch:
            return []
        try:
            return self._gen(batch, max_new_tokens or self.max_new_tokens)
        except self.torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            self.torch.cuda.empty_cache()
            h = len(batch) // 2
            return self.generate_many(batch[:h], max_new_tokens) + self.generate_many(batch[h:], max_new_tokens)

    def _gen(self, batch, max_new_tokens: int) -> list[str]:
        torch = self.torch
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for m in batch]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        dev = next(self.net.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        if self.adapters and self.model == "base":
            ctx = self.net.disable_adapter()
        else:
            if self.adapters:
                self.net.set_adapter(self.model)
            ctx = nullcontext()
        with torch.inference_mode(), ctx:
            out = self.net.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                    top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id,
                                    eos_token_id=self.eos)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
