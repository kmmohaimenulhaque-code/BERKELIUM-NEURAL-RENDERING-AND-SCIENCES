"""QuantityModel lives outside the schema package so the physics schemas can use it without an import cycle
(schema.evaluation -> physics.schema -> schema.common). Re-exported by berkelium.schema.common."""

from __future__ import annotations

from pydantic import field_validator

from ._base import Strict
from .units import Quantity, UnitError, parse_unit


class QuantityModel(Strict):
    """A value with an explicit unit, e.g. {"value": 12, "unit": "mm"}."""

    value: float
    unit: str = "1"

    @field_validator("unit")
    @classmethod
    def _unit_parses(cls, v: str) -> str:
        try:
            parse_unit(v)
        except UnitError as e:
            raise ValueError(str(e)) from None
        return v

    def q(self) -> Quantity:
        return Quantity.of(self.value, self.unit)

    @classmethod
    def from_q(cls, q: Quantity, unit: str) -> QuantityModel:
        return cls(value=q.to(unit), unit=unit)
