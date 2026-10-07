"""Render verified dataset rows to chat SFT examples; audit lengths. Torch-free and testable."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..schema.hashing import sha256_bytes


def compact(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def to_chat(row: dict) -> dict:
    """messages = system + user (as generated) + assistant (the verified target, compact JSON)."""
    if not row.get("verification", {}).get("passed"):
        raise ValueError(f"refusing to render unverified example {row.get('id')}")
    return {"id": row["id"], "task": row["task"], "family_id": row["family_id"],
            "messages": [*row["messages"], {"role": "assistant", "content": compact(row["target"])}]}


def render(in_files: Iterable[str | Path], out_file: str | Path, tasks: set[str] | None = None) -> dict:
    rows = []
    for f in in_files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if tasks is None or r["task"] in tasks:
                    rows.append(to_chat(r))
    data = "".join(compact(r) + "\n" for r in rows).encode()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_bytes(data)
    return {"rows": len(rows), "sha256": sha256_bytes(data), "file": str(out_file)}


def length_audit(chat_file: str | Path, tokenizer=None, max_len: int = 8192) -> dict:
    """Token lengths if a tokenizer (with apply_chat_template) is given, else a chars/3.5 estimate."""
    lens = []
    for line in Path(chat_file).read_text().splitlines():
        r = json.loads(line)
        if tokenizer is not None:
            ids = tokenizer.apply_chat_template(r["messages"], tokenize=True, enable_thinking=False)
            lens.append(len(ids))
        else:
            lens.append(int(sum(len(m["content"]) for m in r["messages"]) / 3.5))
    lens.sort()
    n = len(lens)
    return {"n": n, "estimated": tokenizer is None, "max": lens[-1] if n else 0,
            "p50": lens[n // 2] if n else 0, "p95": lens[int(n * 0.95)] if n else 0,
            "over_max_len": sum(x > max_len for x in lens), "max_len": max_len}
