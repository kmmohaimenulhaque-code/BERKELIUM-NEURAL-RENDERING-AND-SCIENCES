"""Canonical JSON (RFC 8785 / JCS) and sha256 content hashing."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import rfc8785
from pydantic import BaseModel

# Floats are quantised before hashing so that platform-level last-bit noise in geometry
# measurements cannot change a hash. 12 significant digits is far below every tolerance used.
SIGNIFICANT_DIGITS = 12


def _quantise(v: Any) -> Any:
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        if abs(v) > 2**53:
            raise ValueError("integers above 2^53 are not hashable canonically")
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("non-finite float in hashed content")
        if v == 0.0:
            return 0
        q = float(f"{v:.{SIGNIFICANT_DIGITS}g}")
        return int(q) if q.is_integer() and abs(q) < 2**53 else q
    if isinstance(v, list | tuple):
        return [_quantise(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _quantise(x) for k, x in v.items()}
    raise TypeError(f"cannot canonicalise {type(v).__name__}")


def canonical_bytes(obj: Any) -> bytes:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return rfc8785.dumps(_quantise(obj))


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
