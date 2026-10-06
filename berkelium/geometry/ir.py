"""Geometry IR — a backend-independent DAG of typed operations (architecture §5.1).

* Lengths are millimetres, angles are degrees (``units`` is fixed to "mm").
* Any scalar argument may be a number **or** an expression string (berkelium.expr) that is
  resolved against design parameters before execution. This is what makes procedural,
  LLM-authored geometry *data* rather than code.
* Each op declares a representation requirement (``exact`` | ``field`` | ``any``) and carries
  semantic ``tags`` so validators and Studio can point at the geometry a result refers to.
* Profile ops produce planar 2-D loops in the XY plane; solid ops produce bodies.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, model_validator

from ..schema.common import Id, Strict

Scalar = float | str  # number or expression string
Vec2 = Annotated[list[Scalar], Field(min_length=2, max_length=2)]
Vec3 = Annotated[list[Scalar], Field(min_length=3, max_length=3)]
Representation = Literal["exact", "field", "any"]
OutputKind = Literal["profile", "solid"]


class GeometryIRError(ValueError):
    """Structural error in a geometry graph (dangling reference, cycle, type mismatch)."""


class OpBase(Strict):
    id: Id
    tags: list[str] = Field(default_factory=list, description="Semantic tags, e.g. tooth_flank")
    output_kind: ClassVar[OutputKind] = "solid"
    requires: ClassVar[Representation] = "any"

    def deps(self) -> list[str]:
        return []

    def input_kinds(self) -> dict[str, OutputKind]:
        """Map of dependency id -> required kind."""
        return {}


# ---------------------------------------------------------------- primitives
class Box(OpBase):
    op: Literal["box"] = "box"
    size: Vec3
    center: bool = False


class Cylinder(OpBase):
    op: Literal["cylinder"] = "cylinder"
    radius: Scalar
    height: Scalar
    center: bool = False


class Cone(OpBase):
    op: Literal["cone"] = "cone"
    radius1: Scalar
    radius2: Scalar
    height: Scalar


class Sphere(OpBase):
    op: Literal["sphere"] = "sphere"
    radius: Scalar


class Torus(OpBase):
    op: Literal["torus"] = "torus"
    major_radius: Scalar
    minor_radius: Scalar


# ---------------------------------------------------------------- profiles (2-D, XY plane)
class LineSeg(Strict):
    kind: Literal["line"] = "line"
    to: Vec2


class ArcSeg(Strict):
    """Circular arc from the current point through ``through`` to ``to``."""

    kind: Literal["arc"] = "arc"
    through: Vec2
    to: Vec2


class SplineSeg(Strict):
    """Interpolating cubic B-spline from the current point through ``points`` (last = end point).

    ``max_deviation_mm`` is the producer's stated bound on |spline - intended curve|; backends
    record their own approximation error separately."""

    kind: Literal["spline"] = "spline"
    points: Annotated[list[Vec2], Field(min_length=2)]
    max_deviation_mm: float | None = None


Segment = Annotated[LineSeg | ArcSeg | SplineSeg, Field(discriminator="kind")]


class Profile(OpBase):
    """Closed planar loop starting at ``start``; the last segment must return to ``start``."""

    op: Literal["profile"] = "profile"
    start: Vec2
    segments: Annotated[list[Segment], Field(min_length=1)]
    output_kind: ClassVar[OutputKind] = "profile"


class CircleProfile(OpBase):
    op: Literal["circle"] = "circle"
    radius: Scalar
    center: Vec2 = Field(default_factory=lambda: [0.0, 0.0])
    output_kind: ClassVar[OutputKind] = "profile"


# ---------------------------------------------------------------- sweeps
class Extrude(OpBase):
    """Extrude a profile along +Z by ``height`` (mm)."""

    op: Literal["extrude"] = "extrude"
    profile: Id
    height: Scalar

    def deps(self) -> list[str]:
        return [self.profile]

    def input_kinds(self) -> dict[str, OutputKind]:
        return {self.profile: "profile"}


class Revolve(OpBase):
    """Revolve an XY profile about the Y axis by ``angle`` degrees."""

    op: Literal["revolve"] = "revolve"
    profile: Id
    angle: Scalar = 360.0

    def deps(self) -> list[str]:
        return [self.profile]

    def input_kinds(self) -> dict[str, OutputKind]:
        return {self.profile: "profile"}


# ---------------------------------------------------------------- booleans
class Union(OpBase):
    op: Literal["union"] = "union"
    inputs: Annotated[list[Id], Field(min_length=2)]

    def deps(self) -> list[str]:
        return list(self.inputs)

    def input_kinds(self) -> dict[str, OutputKind]:
        return {i: "solid" for i in self.inputs}


class Difference(OpBase):
    op: Literal["difference"] = "difference"
    base: Id
    tools: Annotated[list[Id], Field(min_length=1)]

    def deps(self) -> list[str]:
        return [self.base, *self.tools]

    def input_kinds(self) -> dict[str, OutputKind]:
        return {i: "solid" for i in self.deps()}


class Intersection(OpBase):
    op: Literal["intersection"] = "intersection"
    inputs: Annotated[list[Id], Field(min_length=2)]

    def deps(self) -> list[str]:
        return list(self.inputs)

    def input_kinds(self) -> dict[str, OutputKind]:
        return {i: "solid" for i in self.inputs}


# ---------------------------------------------------------------- transforms
class Translate(OpBase):
    op: Literal["translate"] = "translate"
    input: Id
    offset: Vec3

    def deps(self) -> list[str]:
        return [self.input]

    def input_kinds(self) -> dict[str, OutputKind]:
        return {self.input: "solid"}


class Rotate(OpBase):
    op: Literal["rotate"] = "rotate"
    input: Id
    axis: Vec3 = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    angle: Scalar = 0.0
    origin: Vec3 = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    def deps(self) -> list[str]:
        return [self.input]

    def input_kinds(self) -> dict[str, OutputKind]:
        return {self.input: "solid"}


class Mirror(OpBase):
    op: Literal["mirror"] = "mirror"
    input: Id
    normal: Vec3
    origin: Vec3 = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    def deps(self) -> list[str]:
        return [self.input]

    def input_kinds(self) -> dict[str, OutputKind]:
        return {self.input: "solid"}


# ---------------------------------------------------------------- patterns
class PolarPattern(OpBase):
    """``count`` copies rotated about the Z axis through the origin over ``angle`` degrees,
    fused into one body. ``angle == 360`` spaces copies evenly without duplicating the first."""

    op: Literal["polar_pattern"] = "polar_pattern"
    input: Id
    count: Annotated[int, Field(ge=1, le=10000)]
    angle: Scalar = 360.0

    def deps(self) -> list[str]:
        return [self.input]

    def input_kinds(self) -> dict[str, OutputKind]:
        return {self.input: "solid"}


class LinearPattern(OpBase):
    op: Literal["linear_pattern"] = "linear_pattern"
    input: Id
    count: Annotated[int, Field(ge=1, le=10000)]
    step: Vec3

    def deps(self) -> list[str]:
        return [self.input]

    def input_kinds(self) -> dict[str, OutputKind]:
        return {self.input: "solid"}


Op = Annotated[
    Box | Cylinder | Cone | Sphere | Torus | Profile | CircleProfile | Extrude | Revolve
    | Union | Difference | Intersection | Translate | Rotate | Mirror | PolarPattern | LinearPattern,
    Field(discriminator="op"),
]


class GeometryGraph(Strict):
    """A DAG of operations. ``outputs`` names the bodies this graph delivers."""

    units: Literal["mm"] = "mm"
    ops: list[Op]
    outputs: dict[str, Id] = Field(description="output name -> op id (must be a solid)")

    @model_validator(mode="after")
    def _structure(self) -> GeometryGraph:
        self.check()
        return self

    # ---- structural helpers
    def by_id(self) -> dict[str, OpBase]:
        return {o.id: o for o in self.ops}

    def check(self) -> None:
        seen: dict[str, OpBase] = {}
        for o in self.ops:
            if o.id in seen:
                raise GeometryIRError(f"duplicate op id {o.id!r}")
            seen[o.id] = o
        for o in self.ops:
            for dep, kind in o.input_kinds().items():
                if dep not in seen:
                    raise GeometryIRError(f"op {o.id!r} references unknown op {dep!r}")
                if seen[dep].output_kind != kind:
                    raise GeometryIRError(f"op {o.id!r} needs a {kind} from {dep!r}, "
                                          f"got {seen[dep].output_kind}")
        for name, oid in self.outputs.items():
            if oid not in seen:
                raise GeometryIRError(f"output {name!r} references unknown op {oid!r}")
            if seen[oid].output_kind != "solid":
                raise GeometryIRError(f"output {name!r} must be a solid")
        self.topological_order()

    def topological_order(self) -> list[str]:
        """Deterministic topological order (Kahn's algorithm, ties broken by declaration order)."""
        ops = self.by_id()
        order_index = {o.id: i for i, o in enumerate(self.ops)}
        indeg = {i: 0 for i in ops}
        users: dict[str, list[str]] = {i: [] for i in ops}
        for o in self.ops:
            for d in set(o.deps()):
                indeg[o.id] += 1
                users[d].append(o.id)
        ready = sorted((i for i, n in indeg.items() if n == 0), key=order_index.__getitem__)
        out: list[str] = []
        while ready:
            cur = ready.pop(0)
            out.append(cur)
            for u in users[cur]:
                indeg[u] -= 1
                if indeg[u] == 0:
                    ready.append(u)
                    ready.sort(key=order_index.__getitem__)
        if len(out) != len(ops):
            raise GeometryIRError("geometry graph contains a cycle")
        return out

    def required_representation(self) -> Representation:
        reqs = {type(o).requires for o in self.ops}
        if "field" in reqs:
            return "field"
        if "exact" in reqs:
            return "exact"
        return "any"

    def expressions(self) -> list[str]:
        """All expression-string arguments used by the graph (for dependency analysis)."""
        found: list[str] = []

        def walk(v: Any) -> None:
            if isinstance(v, str):
                found.append(v)
            elif isinstance(v, list):
                for x in v:
                    walk(x)
            elif isinstance(v, dict):
                for x in v.values():
                    walk(x)

        for o in self.ops:
            d = o.model_dump(exclude={"id", "tags", "op", "profile", "input", "inputs", "base", "tools"})
            for k, v in d.items():
                if k == "kind":
                    continue
                if isinstance(v, list) and v and isinstance(v[0], dict):  # segments
                    for seg in v:
                        walk({kk: vv for kk, vv in seg.items() if kk != "kind"})
                else:
                    walk(v)
        return found

    def resolve(self, env: Mapping[str, Any]) -> GeometryGraph:
        """Return a copy with every expression argument evaluated to a float.

        Length-valued results are converted to mm. Dimensionless results are taken as-is; for
        ``angle`` arguments that means *degrees*, unless the expression contains an explicit
        angle-unit literal ([deg]/[rad]/[rev]), in which case the SI radian value is converted."""
        from ..expr import Num, evaluate, parse
        from ..expr.ast import Binary, Call, Unary
        from ..units import DIMENSIONLESS, Quantity, parse_unit

        def has_angle_unit(n: Any) -> bool:
            match n:
                case Num(unit=u):
                    return u in ("deg", "rad", "rev")
                case Unary(operand=o):
                    return has_angle_unit(o)
                case Binary(left=a, right=b):
                    return has_angle_unit(a) or has_angle_unit(b)
                case Call(args=args):
                    return any(has_angle_unit(a) for a in args)
            return False

        mm_factor, length_dim = parse_unit("mm")

        def conv(key: str, v: Any) -> Any:
            if isinstance(v, str):
                try:
                    node = parse(v)
                    val = evaluate(node, env)
                except Exception as e:  # ExprSyntaxError or ExprError
                    raise GeometryIRError(f"cannot resolve {key}={v!r}: {e}") from None
                if not isinstance(val, Quantity):
                    raise GeometryIRError(f"{key}={v!r} evaluated to a boolean")
                if val.dim == length_dim:
                    return val.si / mm_factor
                if val.dim == DIMENSIONLESS:
                    if key == "angle" and has_angle_unit(node):
                        return math.degrees(val.si)  # explicit [deg]/[rad] literal: SI radians -> deg
                    return float(val.si)
                raise GeometryIRError(f"{key}={v!r} has unsupported dimension for geometry")
            if isinstance(v, list):
                return [conv(key, x) for x in v]
            if isinstance(v, dict):
                return {k: conv(k, x) for k, x in v.items()}
            return v

        raw = self.model_dump()
        for op in raw["ops"]:
            for k in list(op.keys()):
                if k in ("id", "tags", "op", "profile", "input", "inputs", "base", "tools", "count"):
                    continue
                op[k] = conv(k, op[k])
        return GeometryGraph.model_validate(raw)

    def is_resolved(self) -> bool:
        return not self.expressions()
