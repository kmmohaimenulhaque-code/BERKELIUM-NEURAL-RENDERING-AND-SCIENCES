"""Export canonical JSON Schemas (consumed by Studio for TS type generation and by constrained decoding)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from ..geometry.ir import GeometryGraph
from .common import SCHEMA_VERSION
from .design import DesignProposal
from .evaluation import ValidationReport
from .record import DesignRecord

EXPORTED: dict[str, type[BaseModel]] = {
    "DesignProposal": DesignProposal,
    "DesignRecord": DesignRecord,
    "GeometryGraph": GeometryGraph,
    "ValidationReport": ValidationReport,
}


def json_schema(name: str) -> dict:
    model = EXPORTED[name]
    schema = model.model_json_schema(mode="validation")
    schema["$id"] = f"https://berkelium.dev/schema/{SCHEMA_VERSION}/{name}.json"
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def export_all(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in EXPORTED:
        p = out / f"{name}.schema.json"
        p.write_text(json.dumps(json_schema(name), indent=2, sort_keys=True) + "\n")
        paths.append(p)
    return paths
