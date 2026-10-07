"""Manifold mesh backend (manifold3d, Apache-2.0): guaranteed-manifold mesh booleans, previews,
and level-set meshing of implicit fields. Every curved primitive and profile is a declared
approximation; the chord tolerance used is recorded on the Realization and on measurements."""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from .backend import (
    BackendUnavailable,
    BodyMeasurements,
    OperationFailed,
    Realization,
    UnsupportedConversion,
    UnsupportedOperation,
    check_capabilities,
)
from .ir import (
    ArcSeg,
    Box,
    CircleProfile,
    Cone,
    Cylinder,
    Difference,
    Extrude,
    GeometryGraph,
    ImplicitField,
    Intersection,
    LinearPattern,
    LineSeg,
    Mirror,
    PolarPattern,
    Profile,
    Revolve,
    Rotate,
    Sphere,
    SplineSeg,
    Torus,
    Translate,
    Union,
)

try:
    import manifold3d as mf
except ImportError as e:  # pragma: no cover
    raise BackendUnavailable(f"manifold3d not installed: {e}") from None


def _segments(radius: float, tol: float) -> int:
    if radius <= tol:
        return 8
    return max(8, int(math.ceil(math.pi / math.acos(1.0 - tol / radius))))


def _rot_matrix(axis, angle_deg, origin) -> list[list[float]]:
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        raise OperationFailed("zero rotation axis")
    x, y, z = a / n
    t = math.radians(angle_deg)
    c, s, C = math.cos(t), math.sin(t), 1 - math.cos(t)
    R = np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                  [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                  [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])
    o = np.asarray(origin, dtype=float)
    tvec = o - R @ o
    return np.hstack([R, tvec[:, None]]).tolist()


class ManifoldBackend:
    name = "manifold"
    representation = "mesh"

    def __init__(self, chord_tolerance_mm: float = 0.01):
        try:
            self.version = version("manifold3d")
        except PackageNotFoundError:
            self.version = "unknown"
        self.tol = chord_tolerance_mm

    def capabilities(self) -> set[str]:
        return {"box", "cylinder", "cone", "sphere", "torus", "profile", "circle", "extrude", "revolve",
                "union", "difference", "intersection", "translate", "rotate", "mirror",
                "polar_pattern", "linear_pattern", "implicit"}

    def execute(self, graph: GeometryGraph) -> Realization:
        if not graph.is_resolved():
            raise OperationFailed("graph contains unresolved expressions; call graph.resolve(env) first")
        check_capabilities(self, graph)
        ops = graph.by_id()
        out: dict[str, Any] = {}
        notes: list[str] = []
        tol = self.tol
        for oid in graph.topological_order():
            o = ops[oid]
            try:
                out[oid] = self._exec(o, out, notes)
            except (UnsupportedOperation, OperationFailed):
                raise
            except Exception as e:
                raise OperationFailed(f"{type(e).__name__}: {e}", oid) from None
            if isinstance(o, ImplicitField):
                tol = max(tol, o.voxel)
            if isinstance(out[oid], mf.Manifold):
                st = out[oid].status()
                if str(st) not in ("Error.NoError", "NoError"):
                    raise OperationFailed(f"manifold status {st}", oid)
        bodies = {n: out[i] for n, i in graph.outputs.items()}
        for n, b in bodies.items():
            if b.is_empty():
                raise OperationFailed(f"output {n!r} is empty")
        return Realization(self.name, "mesh", bodies, approximations=notes, tolerance_mm=tol)

    def _exec(self, o, s, notes):
        f = float
        match o:
            case Box(size=sz, center=c):
                return mf.Manifold.cube([f(v) for v in sz], c)
            case Cylinder(radius=r, height=h, center=c):
                return mf.Manifold.cylinder(f(h), f(r), -1.0, _segments(f(r), self.tol), c)
            case Cone(radius1=r1, radius2=r2, height=h):
                return mf.Manifold.cylinder(f(h), f(r1), f(r2), _segments(max(f(r1), f(r2)), self.tol), False)
            case Sphere(radius=r):
                return mf.Manifold.sphere(f(r), _segments(f(r), self.tol))
            case Torus(major_radius=R, minor_radius=r):
                cs = mf.CrossSection.circle(f(r), _segments(f(r), self.tol)).translate((f(R), 0.0))
                return mf.Manifold.revolve(cs, _segments(f(R) + f(r), self.tol), 360.0)
            case Profile():
                return self._profile(o, notes)
            case CircleProfile(radius=r, center=c):
                return mf.CrossSection.circle(f(r), _segments(f(r), self.tol)).translate((f(c[0]), f(c[1])))
            case Extrude(profile=p, height=h):
                return mf.Manifold.extrude(s[p], f(h))
            case Revolve(profile=p, angle=a):
                bb = s[p].bounds()
                segs = _segments(max(abs(bb[0]), abs(bb[2]), 1e-6), self.tol)
                m = mf.Manifold.revolve(s[p], segs, f(a))
                # Manifold revolves about the profile Y axis and maps it to Z; rotate so the axis stays +Y
                # (matching OCCT: right-handed rotation about +Y).
                return m.transform(_rot_matrix((1, 0, 0), -90.0, (0, 0, 0)))
            case Union(inputs=ins):
                return mf.Manifold.batch_boolean([s[i] for i in ins], mf.OpType.Add)
            case Intersection(inputs=ins):
                acc = s[ins[0]]
                for i in ins[1:]:
                    acc = acc ^ s[i]
                return acc
            case Difference(base=b, tools=tools):
                acc = s[b]
                for t in tools:
                    acc = acc - s[t]
                return acc
            case Translate(input=i, offset=v):
                return s[i].translate([f(x) for x in v])
            case Rotate(input=i, axis=ax, angle=a, origin=org):
                return s[i].transform(_rot_matrix([f(x) for x in ax], f(a), [f(x) for x in org]))
            case Mirror(input=i, normal=n, origin=org):
                o3 = [f(x) for x in org]
                return s[i].translate([-x for x in o3]).mirror([f(x) for x in n]).translate(o3)
            case PolarPattern(input=i, count=n, angle=a):
                total = f(a)
                step = total / n if abs(total - 360.0) < 1e-9 else (total / (n - 1) if n > 1 else 0.0)
                copies = [s[i].transform(_rot_matrix((0, 0, 1), step * k, (0, 0, 0))) for k in range(n)]
                return copies[0] if n == 1 else mf.Manifold.batch_boolean(copies, mf.OpType.Add)
            case LinearPattern(input=i, count=n, step=v):
                copies = [s[i].translate([f(x) * k for x in v]) for k in range(n)]
                return copies[0] if n == 1 else mf.Manifold.batch_boolean(copies, mf.OpType.Add)
            case ImplicitField():
                return self._implicit(o, notes)
        raise UnsupportedOperation(f"manifold: '{o.op}' not implemented", o.id)

    def _profile(self, o: Profile, notes: list[str]):
        pts: list[tuple[float, float]] = [(float(o.start[0]), float(o.start[1]))]
        for seg in o.segments:
            cur = pts[-1]
            match seg:
                case LineSeg(to=to):
                    pts.append((float(to[0]), float(to[1])))
                case ArcSeg(through=th, to=to):
                    pts.extend(self._arc_points(cur, (float(th[0]), float(th[1])), (float(to[0]), float(to[1])), o.id))
                case SplineSeg(points=sp):
                    pts.extend((float(p[0]), float(p[1])) for p in sp)
                    notes.append(f"{o.id}: spline segment meshed as polyline through its {len(sp)} control points")
        if math.dist(pts[0], pts[-1]) > 1e-6:
            raise OperationFailed("profile is not closed (last point != start)", o.id)
        poly = np.asarray(pts[:-1], dtype=float)
        x, y = poly[:, 0], poly[:, 1]
        if 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 0:
            poly = poly[::-1]
        return mf.CrossSection([poly], mf.FillRule.Positive)

    def _arc_points(self, p1, p2, p3, op_id):
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(d) < 1e-12:
            raise OperationFailed("degenerate arc (collinear points)", op_id)
        ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
        uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
        r = math.hypot(x1 - ux, y1 - uy)
        a1, a2, a3 = (math.atan2(y - uy, x - ux) for x, y in (p1, p2, p3))

        def ccw(a, b):  # angle from a to b going counter-clockwise
            return (b - a) % (2 * math.pi)

        sweep = ccw(a1, a3)
        if ccw(a1, a2) > sweep:  # through-point not on the ccw path: go clockwise
            sweep -= 2 * math.pi
        n = max(2, int(math.ceil(abs(sweep) / (2 * math.pi) * _segments(r, self.tol))))
        return [(ux + r * math.cos(a1 + sweep * k / n), uy + r * math.sin(a1 + sweep * k / n)) for k in range(1, n + 1)]

    def _implicit(self, o: ImplicitField, notes: list[str]):
        from ..expr import parse
        from ..expr.vectorized import compile_scalar
        f = compile_scalar(parse(o.expr), ("x", "y", "z"))
        lo, hi = [float(v) for v in o.bounds_min], [float(v) for v in o.bounds_max]
        # Berkelium convention: f <= 0 inside. Manifold: positive inside.
        m = mf.Manifold.level_set(lambda x, y, z: -f(x, y, z), lo + hi, float(o.voxel), 0.0, -1.0)
        notes.append(f"{o.id}: implicit field meshed by marching tetrahedra at voxel {o.voxel} mm")
        return m

    def measure(self, m) -> BodyMeasurements:
        bb = m.bounding_box()
        parts = m.decompose()
        mesh = m.to_mesh()
        v = np.asarray(mesh.vert_properties)[:, :3]
        t = np.asarray(mesh.tri_verts)
        a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        cross = np.cross(b - a, c - a)
        vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
        com = (((a + b + c) * vol6[:, None]).sum(0) / (4 * vol6.sum())).tolist() if vol6.sum() else None
        return BodyMeasurements(
            backend=f"{self.name}@{self.version}", representation="mesh", volume_mm3=float(m.volume()),
            area_mm2=float(np.linalg.norm(cross, axis=1).sum() / 2), bbox_min=tuple(bb[:3]), bbox_max=tuple(bb[3:]),
            center_of_mass=tuple(com) if com else None, n_solids=len(parts), n_shells=len(parts),
            closed=True, valid=str(m.status()) in ("Error.NoError", "NoError"), genus=int(m.genus()),
            tolerance_mm=self.tol, extra={"n_triangles": int(m.num_tri()), "n_vertices": int(m.num_vert())},
        )

    def export(self, m, fmt: str) -> tuple[bytes, bool, float]:
        if fmt not in ("stl", "glb"):
            raise UnsupportedConversion(f"manifold (mesh) cannot export {fmt}: mesh -> exact BREP is not possible")
        from .meshio import mesh_bytes
        mesh = m.to_mesh()
        return mesh_bytes(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), fmt), True, self.tol
