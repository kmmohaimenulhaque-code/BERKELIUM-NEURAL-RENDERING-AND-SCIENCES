"""Engineering memory: what Berkelium has computed, verified, discovered or failed at — as COMPUTABLE records.

Record kinds (the minimum set E3 needs; each was required by an experiment, none is speculative):
  evidence   one evaluation of an evidence source at a point: inputs, estimate, error, provenance
  relation   a promoted (discovered or calibrated) relation: formula, validity box, model-form bound, the
             evidence ids it was fitted to and validated on, the gate it passed
  rejection  a candidate relation that FAILED promotion, with the reason (negative knowledge is kept)
  resolution a decided/undecided claim with its evidence chain

Properties: append-only JSONL; record id = sha256 of canonical (kind, key) so the same computation is never
stored or run twice (memoisation = memory); a record's provenance names its producer and version. Nothing in
memory is model-written: producers are deterministic Berkelium code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class Record:
    id: str
    kind: str
    key: dict
    value: dict
    provenance: dict


class EngineeringMemory:
    KINDS = ("evidence", "relation", "rejection", "resolution")

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._recs: dict[str, Record] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._recs[d["id"]] = Record(**d)

    @staticmethod
    def rid(kind: str, key: dict) -> str:
        return hashlib.sha256(_canon({"kind": kind, "key": key}).encode()).hexdigest()

    def put(self, kind: str, key: dict, value: dict, provenance: dict) -> Record:
        if kind not in self.KINDS:
            raise ValueError(f"unknown record kind {kind!r}")
        i = self.rid(kind, key)
        if i in self._recs:
            return self._recs[i]
        r = Record(i, kind, json.loads(_canon(key)), json.loads(_canon(value)), json.loads(_canon(provenance)))
        self._recs[i] = r
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(_canon(r.__dict__) + "\n")
        return r

    def get(self, kind: str, key: dict) -> Record | None:
        return self._recs.get(self.rid(kind, key))

    def query(self, kind: str, where: Callable[[Record], bool] = lambda r: True) -> list[Record]:
        return sorted((r for r in self._recs.values() if r.kind == kind and where(r)), key=lambda r: r.id)

    def memo(self, key: dict, compute: Callable[[], dict], provenance: dict) -> tuple[dict, bool]:
        """Return (value, was_cached). The same evidence is never computed twice."""
        hit = self.get("evidence", key)
        if hit is not None:
            return hit.value, True
        return self.put("evidence", key, compute(), provenance).value, False

    def __len__(self) -> int:
        return len(self._recs)
