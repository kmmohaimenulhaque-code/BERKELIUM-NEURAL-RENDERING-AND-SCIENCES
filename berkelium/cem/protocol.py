"""CEM protocol and registry (architecture §4, ADR-006).

A CEM is a versioned, pure, deterministic object: typed Requirements -> derive -> typed Parameters
-> check (analytic, pre-geometry) -> expand (Geometry IR, ports, relations) -> geometry validators
(post-realisation) -> sample (dataset generation). No global state, no I/O, no viewer side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel

from ..geometry.ir import GeometryGraph
from ..schema.design import Port, Relation
from ..schema.evaluation import Derivation, ValidationResult

SampleMode = Literal["valid", "boundary", "invalid"]


@dataclass(frozen=True)
class CEMMeta:
    name: str                       # e.g. "gear.spur_pair"
    version: str                    # semver "0.1.0"; references use name@major.minor
    domain: str
    summary: str
    kind: Literal["cem", "parametric_template"]  # parametric_template = no engineering derive (ADR-006)
    references: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def ref(self) -> str:
        major, minor, *_ = self.version.split(".")
        return f"{self.name}@{major}.{minor}"


@dataclass
class DeriveResult:
    parameters: BaseModel
    derivations: list[Derivation] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    diagnostics: list[ValidationResult] = field(default_factory=list)


@dataclass
class BodySpec:
    """One body produced by a CEM: its geometry graph output name plus placement in the component frame."""

    graph_output: str
    tags: list[str] = field(default_factory=list)


@dataclass
class ComponentGeometry:
    graph: GeometryGraph
    bodies: dict[str, BodySpec]          # body name -> spec (graph output)
    ports: list[Port] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)   # internal relations, e.g. gear mesh
    notes: list[str] = field(default_factory=list)


@dataclass
class GeometryContext:
    """What a CEM geometry validator may look at: realised bodies and the backend that made them."""

    backend: Any
    bodies: dict[str, Any]
    measurements: dict[str, Any]
    parameters: BaseModel


class CEM:
    meta: ClassVar[CEMMeta]
    Requirements: ClassVar[type[BaseModel]]
    Parameters: ClassVar[type[BaseModel]]

    def derive(self, req: BaseModel) -> DeriveResult:
        raise NotImplementedError

    def check(self, p: BaseModel) -> list[ValidationResult]:
        return []

    def expand(self, p: BaseModel) -> ComponentGeometry:
        raise NotImplementedError

    def geometry_validators(self) -> list[Callable[[GeometryContext], list[ValidationResult]]]:
        return []

    def analyses(self) -> dict[str, str]:
        """analysis name -> 'analytic' | 'solver_required' | 'unsupported'."""
        return {}

    def sample(self, rng: np.random.Generator, mode: SampleMode) -> BaseModel:
        raise NotImplementedError

    def quantities(self, p: BaseModel) -> dict[str, Any]:
        """Named engineering quantities (berkelium.units.Quantity) for requirement/constraint evaluation."""
        return {}


class RegistryError(KeyError):
    pass


class CEMRegistry:
    def __init__(self) -> None:
        self._by_ref: dict[str, CEM] = {}

    def register(self, cem: CEM) -> CEM:
        ref = cem.meta.ref
        if ref in self._by_ref:
            raise RegistryError(f"duplicate CEM {ref}")
        self._by_ref[ref] = cem
        return cem

    def get(self, ref: str) -> CEM:
        if ref not in self._by_ref:
            raise RegistryError(f"unknown CEM {ref!r}; available: {sorted(self._by_ref)}")
        return self._by_ref[ref]

    def refs(self) -> list[str]:
        return sorted(self._by_ref)

    def summary(self) -> list[dict]:
        """Compact registry summary for model context and the API."""
        out = []
        for ref in self.refs():
            c = self._by_ref[ref]
            out.append({"ref": ref, "kind": c.meta.kind, "domain": c.meta.domain, "summary": c.meta.summary,
                        "requirements_schema": c.Requirements.model_json_schema(),
                        "parameters_schema": c.Parameters.model_json_schema()})
        return out

    def load_entry_points(self, group: str = "berkelium.cems") -> None:
        for ep in entry_points(group=group):
            obj = ep.load()
            self.register(obj() if isinstance(obj, type) else obj)


_DEFAULT: CEMRegistry | None = None


def default_registry() -> CEMRegistry:
    global _DEFAULT
    if _DEFAULT is None:
        reg = CEMRegistry()
        from .library.gear.cem import SpurGearPairCEM
        reg.register(SpurGearPairCEM())
        from .library.beam.cem import CantileverBeamCEM
        reg.register(CantileverBeamCEM())
        reg.load_entry_points()
        _DEFAULT = reg
    return _DEFAULT
