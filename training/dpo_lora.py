"""Direct Preference Optimization (Rafailov et al., 2023, "Direct Preference Optimization: Your Language Model is
Secretly a Reward Model", NeurIPS) on Berkelium's ENVIRONMENT-LABELLED pairs (chosen = grounded/verified action,
rejected = ungrounded / wrong action at the same observation). Written on transformers + PEFT directly (no TRL),
so the exact loss is visible and testable:

  L = -log sigmoid( beta * [ (log pi(c) - log pi(r)) - (log ref(c) - log ref(r)) ] )

log-probs are summed over the ASSISTANT tokens only (same masking as SFT, training/common.py). The reference is the
starting adapter (normally the SFT adapter), FROZEN by precomputing its log-probs once before any update.

  python training/dpo_lora.py --config training/configs/qwen3_32b_dpo_agent_v2.yaml
  python training/dpo_lora.py --config ... --model Qwen/Qwen3-0.6B --init-adapter none --max-steps 6   # CPU dry run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import encode, load_model, lora_config, require_free_gpu  # noqa: E402


def seq_logp(model, ids, labels):
    import torch
    out = model(input_ids=ids).logits[:, :-1].float()
    tgt = labels[:, 1:]
    lp = torch.log_softmax(out, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * (tgt != -100)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--init-adapter", default=None, help="SFT adapter dir, or 'none' for a fresh LoRA (ref = base)")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-free-gb", type=float, default=120.0)
    a = ap.parse_args()
    import torch
    import yaml
    from peft import PeftModel, get_peft_model
    from transformers import AutoTokenizer
    cfg = yaml.safe_load(Path(a.config).read_text())
    d = cfg["dpo"]
    model_id = a.model or cfg["model"]
    gpu = require_free_gpu(0.0 if a.model else a.min_free_gb)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(d.get("seed", 0))
    tok = AutoTokenizer.from_pretrained(model_id, revision=cfg.get("revision"))
    pairs_path = a.pairs or d["pairs"]
    pairs = [json.loads(x) for x in Path(pairs_path).read_text().splitlines() if x.strip()]
    enc = []
    for p in pairs:
        c = encode(tok, p["prompt"] + p["chosen"], cfg["max_length"])
        r = encode(tok, p["prompt"] + p["rejected"], cfg["max_length"])
        if c["n_target"] and r["n_target"] and not c["truncated"] and not r["truncated"]:
            enc.append((c, r))
    base = load_model(cfg, model_id)
    init = a.init_adapter or d.get("init_adapter", "none")
    model = get_peft_model(base, lora_config(cfg)) if init == "none" else PeftModel.from_pretrained(base, init, is_trainable=True)
    t = lambda e: (torch.tensor([e["input_ids"]], device=dev), torch.tensor([e["labels"]], device=dev))  # noqa: E731
    ref = []
    model.eval()
    with torch.no_grad():
        for c, r in enc:                      # frozen reference = the starting policy
            if init == "none":
                with model.disable_adapter():
                    ref.append((float(seq_logp(model, *t(c))), float(seq_logp(model, *t(r)))))
            else:
                ref.append((float(seq_logp(model, *t(c))), float(seq_logp(model, *t(r)))))
    model.train()
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=d["learning_rate"])
    beta, accum = d["beta"], d.get("gradient_accumulation_steps", 1)
    steps = a.max_steps or d["epochs"] * max(1, len(enc) // accum)
    log, i = [], 0
    for step in range(steps):
        tot, margin, acc = 0.0, 0.0, 0
        for _ in range(accum):
            k = i % len(enc)
            i += 1
            (c, r), (rc, rr) = enc[k], ref[k]
            pc, pr = seq_logp(model, *t(c)), seq_logp(model, *t(r))
            z = beta * ((pc - pr) - (rc - rr))
            loss = -torch.nn.functional.logsigmoid(z).mean() / accum
            loss.backward()
            tot += float(loss) * accum
            margin += float(z) / beta
            acc += int(float(z) > 0)
        opt.step()
        opt.zero_grad()
        log.append({"step": step, "loss": tot / accum, "reward_margin": margin / accum, "pref_acc": acc / accum})
        print(json.dumps(log[-1]))
    out = Path(a.out or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    import peft
    import transformers
    (out / "run_manifest.json").write_text(json.dumps({
        "method": "DPO (own implementation, frozen precomputed reference)", "model": model_id, "init_adapter": init,
        "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
        "pairs_sha256": hashlib.sha256(Path(pairs_path).read_bytes()).hexdigest(), "pairs_used": len(enc),
        "pairs_total": len(pairs), "beta": beta, "steps": steps, "gpu": gpu, "log": log,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}},
        indent=2))


if __name__ == "__main__":
    main()
