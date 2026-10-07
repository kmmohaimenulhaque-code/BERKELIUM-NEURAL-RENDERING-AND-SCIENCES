"""Backend abstraction (architecture §5.2): capabilities, execution, measurement, export, planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .ir import GeometryGraph

RepKind = Literal["brep", "mesh", "field"]
ExportFormat = Literal["step", "stl", "glb", "brep"]


class BackendError(RuntimeError):
    """Base class for typed geometry backend errors."""

    code = "BACKEND_ERROR"

    def __init__(self, message: str, op_id: str | None = None):
        super().__init__(message if op_id is None else f"[{op_id}] {message}")
        self.op_id = op_id


class BackendUnavailable(BackendError):
    code = "BACKEND_UNAVAILABLE"


class UnsupportedOperation(BackendError):
    code = "UNSUPPORTED_OP"


class OperationFailed(BackendError):
    code = "OP_FAILED"


class InvalidGeometry(BackendError):
    code = "INVALID_GEOMETRY"


class UnsupportedConversion(BackendError):
    code = "UNSUPPORTED_CONVERSION"


@dataclass(frozen=True)
class BodyMeasurements:
    """Raw measurements of one body. ``tolerance_mm`` states the geometric resolution of the
    representation (0 for exact BREP up to kernel precision; chord/voxel size for meshes)."""

    backend: str
    representation: RepKind
    volume_mm3: float
    area_mm2: float
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    center_of_mass: tuple[float, float, float] | None
    n_solids: int
    n_shells: int
    closed: bool
    valid: bool
    genus: int | None
    tolerance_mm: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Realization:
    backend: str
    representation: RepKind
    bodies: dict[str, Any]                      # output name -> backend handle
    approximations: list[str] = field(default_factory=list)  # human-readable statements of lossy steps
    tolerance_mm: float = 0.0


class GeometryBackend(Protocol):
    name: str
    version: str
    representation: RepKind

    def capabilities(self) -> set[str]: ...
    def execute(self, graph: GeometryGraph) -> Realization: ...
    def measure(self, handle: Any) -> BodyMeasurements: ...
    def export(self, handle: Any, fmt: ExportFormat) -> tuple[bytes, bool, float]:
        """Return (bytes, lossy, tolerance_mm)."""
        ...


def check_capabilities(backend: GeometryBackend, graph: GeometryGraph) -> None:
    caps = backend.capabilities()
    for o in graph.ops:
        if o.op not in caps:  # type: ignore[attr-defined]
            raise UnsupportedOperation(f"{backend.name} cannot execute '{o.op}'", o.id)  # type: ignore[attr-defined]


def available_backends() -> dict[str, GeometryBackend]:
    out: dict[str, GeometryBackend] = {}
    try:
        from .occt import OCCTBackend
        out["occt"] = OCCTBackend()
    except BackendUnavailable:
        pass
    try:
        from .manifold_backend import ManifoldBackend
        out["manifold"] = ManifoldBackend()
    except BackendUnavailable:
        pass
    return out


def plan_backend(graph: GeometryGraph, backends: dict[str, GeometryBackend] | None = None,
                 prefer: str | None = None) -> GeometryBackend:
    """Choose a backend by representation requirement and capability.

    Rule: field ops need a field-capable mesh backend (level sets) and can never be made exact;
    exact ops prefer OCCT; ``any`` prefers OCCT (authoritative) and falls back to Manifold."""
    backends = backends if backends is not None else available_backends()
    need = graph.required_representation()
    order = {"field": ["manifold"], "exact": ["occt", "manifold"], "any": ["occt", "manifold"]}[need]
    if prefer:
        order = [prefer] + [b for b in order if b != prefer]
    errors = []
    for name in order:
        b = backends.get(name)
        if b is None:
            errors.append(f"{name}: not installed")
            continue
        try:
            check_capabilities(b, graph)
            return b
        except UnsupportedOperation as e:
            errors.append(f"{name}: {e}")
    raise UnsupportedOperation("no backend can execute this graph: " + "; ".join(errors))
