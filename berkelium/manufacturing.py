"""Manufacturing profiles (architecture §6) and L5 validators.

Only checks with a deterministic, honest implementation return pass/fail (build envelope).
Min wall / overhang / trapped volume need distance-field or ray analyses that do not exist yet:
they return ``not_evaluated`` rather than a guess."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from ._base import Strict
from .schema.evaluation import Fidelity, ValidationResult

V = "manufacturing@0.1"


class ManufacturingProfile(Strict):
    process: Literal["cnc_milling", "hobbing", "fdm", "sla", "lpbf", "casting", "unspecified"] = "unspecified"
    envelope_mm: Annotated[list[float], Field(min_length=3, max_length=3)] | None = None
    min_wall_mm: float | None = None
    min_hole_mm: float | None = None
    max_overhang_deg: float | None = None
    tolerance_class: str | None = None


def validate_manufacturing(profile: ManufacturingProfile | None, target: str, bbox_min, bbox_max) -> list[ValidationResult]:
    if profile is None:
        return []
    out = []
    if profile.envelope_mm is not None:
        ext = sorted(b - a for a, b in zip(bbox_min, bbox_max, strict=True))
        env = sorted(profile.envelope_mm)
        ok = all(e <= v for e, v in zip(ext, env, strict=True))
        out.append(ValidationResult(validator=V, level=5, status="pass" if ok else "fail", target=target,
                                    message=f"bounding box {[round(e, 3) for e in ext]} mm vs envelope {env} mm "
                                            "(any axis-aligned orientation)",
                                    fidelity=Fidelity(kind="geometric", method="bbox_envelope")))
    for field, label in (("min_wall_mm", "minimum wall"), ("min_hole_mm", "minimum hole"),
                         ("max_overhang_deg", "overhang")):
        if getattr(profile, field) is not None:
            out.append(ValidationResult(validator=V, level=5, status="not_evaluated", target=target,
                                        message=f"{label} check not implemented (needs distance-field analysis)",
                                        fidelity=Fidelity(kind="geometric", method=field)))
    return out
