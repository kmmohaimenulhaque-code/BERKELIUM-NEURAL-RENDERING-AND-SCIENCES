"""OCCT (Open CASCADE) exact-BREP backend via OCP bindings.

OCCT is LGPL-2.1 with the OCCT exception and is used as an unmodified, dynamically loaded
dependency (see docs/SOURCES_AND_LICENSES.md)."""

from __future__ import annotations

import contextlib
import math
import os
import re
import tempfile
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from .backend import (
    BackendUnavailable,
    BodyMeasurements,
    InvalidGeometry,
    OperationFailed,
    Realization,
    UnsupportedConversion,
    UnsupportedOperation,
    check_capabilities,
)
from .curves import spline_points
from .ir import (
    ArcSeg,
    Box,
    CircleProfile,
    Cone,
    Cylinder,
    Difference,
    Extrude,
    GeometryGraph,
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
    from OCP.Bnd import Bnd_Box
    from OCP.BRep import BRep_Tool
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_Transform,
    )
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCone,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakePrism,
        BRepPrimAPI_MakeRevol,
        BRepPrimAPI_MakeSphere,
        BRepPrimAPI_MakeTorus,
    )
    from OCP.BRepTools import BRepTools
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    try:  # OCP >= 8 moved NCollection instantiations into OCP.collections
        from OCP.collections import HArray1_gp_Pnt as TColgp_HArray1OfPnt
        from OCP.collections import (
            IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as TopTools_IndexedMapOfShape,
        )
    except ImportError:  # OCP 7.x
        from OCP.TColgp import TColgp_HArray1OfPnt
        from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
except ImportError as e:  # pragma: no cover
    raise BackendUnavailable(f"OCP (cadquery-ocp) not installed: {e}") from None


def _ocp_version() -> str:
    for dist in ("cadquery-ocp-novtk", "cadquery-ocp"):
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
    return "unknown"


def _count(shape, kind) -> int:
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return m.Size()


def _api(obj, name):
    """OCP 7 exposes static methods with an ``_s`` suffix; OCP 8 drops it for some classes."""
    return getattr(obj, name + "_s", None) or getattr(obj, name)


def _pt(v) -> gp_Pnt:
    return gp_Pnt(float(v[0]), float(v[1]), 0.0)


@contextlib.contextmanager
def _quiet_stdout():
    """OCCT's STEP writer prints transfer statistics on fd 1; keep library output off our stdout."""
    import sys
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


class OCCTBackend:
    name = "occt"
    representation = "brep"
    # STL/GLB from BREP are tessellations: lossy and labelled as such.
    mesh_linear_deflection_mm = 0.01
    mesh_angular_deflection_rad = 0.1

    def __init__(self) -> None:
        self.version = _ocp_version()

    def capabilities(self) -> set[str]:
        return {"box", "cylinder", "cone", "sphere", "torus", "profile", "circle", "extrude", "revolve",
                "union", "difference", "intersection", "translate", "rotate", "mirror",
                "polar_pattern", "linear_pattern"}

    # ------------------------------------------------------------------ execution
    def execute(self, graph: GeometryGraph) -> Realization:
        if not graph.is_resolved():
            raise OperationFailed("graph contains unresolved expressions; call graph.resolve(env) first")
        check_capabilities(self, graph)
        ops = graph.by_id()
        shapes: dict[str, Any] = {}
        for oid in graph.topological_order():
            o = ops[oid]
            try:
                shapes[oid] = self._exec(o, shapes)
            except (UnsupportedOperation, OperationFailed):
                raise
            except Exception as e:  # OCCT raises Standard_Failure subclasses
                raise OperationFailed(f"{type(e).__name__}: {e}", oid) from None
            if shapes[oid] is None or shapes[oid].IsNull():
                raise OperationFailed("operation produced a null shape", oid)
        return Realization(self.name, "brep", {n: shapes[i] for n, i in graph.outputs.items()})

    def _exec(self, o, s: dict[str, Any]):
        f = float
        match o:
            case Box(size=sz, center=c):
                x, y, z = (f(v) for v in sz)
                if min(x, y, z) <= 0:
                    raise OperationFailed("box size must be positive", o.id)
                p = gp_Pnt(-x / 2, -y / 2, -z / 2) if c else gp_Pnt(0, 0, 0)
                return BRepPrimAPI_MakeBox(p, x, y, z).Shape()
            case Cylinder(radius=r, height=h, center=c):
                ax = gp_Ax2(gp_Pnt(0, 0, -f(h) / 2 if c else 0), gp_Dir(0, 0, 1))
                return BRepPrimAPI_MakeCylinder(ax, f(r), f(h)).Shape()
            case Cone(radius1=r1, radius2=r2, height=h):
                return BRepPrimAPI_MakeCone(f(r1), f(r2), f(h)).Shape()
            case Sphere(radius=r):
                return BRepPrimAPI_MakeSphere(f(r)).Shape()
            case Torus(major_radius=R, minor_radius=r):
                return BRepPrimAPI_MakeTorus(f(R), f(r)).Shape()
            case Profile():
                return self._profile_face(o)
            case CircleProfile(radius=r, center=c):
                from OCP.gp import gp_Circ
                circ = gp_Circ(gp_Ax2(gp_Pnt(f(c[0]), f(c[1]), 0), gp_Dir(0, 0, 1)), f(r))
                wire = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(circ).Edge()).Wire()
                return BRepBuilderAPI_MakeFace(wire, True).Face()
            case Extrude(profile=p, height=h):
                return BRepPrimAPI_MakePrism(s[p], gp_Vec(0, 0, f(h))).Shape()
            case Revolve(profile=p, angle=a):
                ax = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0))
                return BRepPrimAPI_MakeRevol(s[p], ax, math.radians(f(a))).Shape()
            case Union(inputs=ins):
                return self._fold(BRepAlgoAPI_Fuse, [s[i] for i in ins], o.id)
            case Intersection(inputs=ins):
                return self._fold(BRepAlgoAPI_Common, [s[i] for i in ins], o.id)
            case Difference(base=b, tools=tools):
                return self._fold(BRepAlgoAPI_Cut, [s[b]] + [s[t] for t in tools], o.id)
            case Translate(input=i, offset=v):
                t = gp_Trsf()
                t.SetTranslation(gp_Vec(*(f(x) for x in v)))
                return BRepBuilderAPI_Transform(s[i], t, True).Shape()
            case Rotate(input=i, axis=ax, angle=a, origin=org):
                t = gp_Trsf()
                t.SetRotation(gp_Ax1(gp_Pnt(*(f(x) for x in org)), gp_Dir(*(f(x) for x in ax))), math.radians(f(a)))
                return BRepBuilderAPI_Transform(s[i], t, True).Shape()
            case Mirror(input=i, normal=n, origin=org):
                t = gp_Trsf()
                t.SetMirror(gp_Ax2(gp_Pnt(*(f(x) for x in org)), gp_Dir(*(f(x) for x in n))))
                return BRepBuilderAPI_Transform(s[i], t, True).Shape()
            case PolarPattern(input=i, count=n, angle=a):
                total = f(a)
                step = total / n if abs(total - 360.0) < 1e-9 else (total / (n - 1) if n > 1 else 0.0)
                copies = []
                for k in range(n):
                    t = gp_Trsf()
                    t.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.radians(step * k))
                    copies.append(BRepBuilderAPI_Transform(s[i], t, True).Shape())
                return copies[0] if n == 1 else self._fold(BRepAlgoAPI_Fuse, copies, o.id)
            case LinearPattern(input=i, count=n, step=v):
                copies = []
                for k in range(n):
                    t = gp_Trsf()
                    t.SetTranslation(gp_Vec(*(f(x) * k for x in v)))
                    copies.append(BRepBuilderAPI_Transform(s[i], t, True).Shape())
                return copies[0] if n == 1 else self._fold(BRepAlgoAPI_Fuse, copies, o.id)
        raise UnsupportedOperation(f"occt: '{o.op}' not implemented", o.id)

    @staticmethod
    def _fold(algo, shapes, op_id):
        acc = shapes[0]
        for nxt in shapes[1:]:
            op = algo(acc, nxt)
            op.Build()
            if not op.IsDone() or (hasattr(op, "HasErrors") and op.HasErrors()):
                raise OperationFailed(f"{algo.__name__} failed", op_id)
            acc = op.Shape()
        return acc

    def _profile_face(self, o: Profile):
        wire = BRepBuilderAPI_MakeWire()
        cur = [float(o.start[0]), float(o.start[1])]
        start = list(cur)
        for seg in o.segments:
            match seg:
                case LineSeg(to=to):
                    p1, p2 = _pt(cur), _pt(to)
                    if p1.Distance(p2) < 1e-9:
                        raise OperationFailed("zero-length line segment", o.id)
                    edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
                    cur = [float(to[0]), float(to[1])]
                case ArcSeg(through=th, to=to):
                    arc = GC_MakeArcOfCircle(_pt(cur), _pt(th), _pt(to))
                    if not arc.IsDone():
                        raise OperationFailed("degenerate arc", o.id)
                    edge = BRepBuilderAPI_MakeEdge(arc.Value()).Edge()
                    cur = [float(to[0]), float(to[1])]
                case SplineSeg(points=pts):
                    allp = spline_points([(cur[0], cur[1])] + [(float(p[0]), float(p[1])) for p in pts])
                    arr = TColgp_HArray1OfPnt(1, len(allp))
                    for k, p in enumerate(allp, 1):
                        arr.SetValue(k, _pt(p))
                    interp = GeomAPI_Interpolate(arr, False, 1e-9)
                    interp.Perform()
                    if not interp.IsDone():
                        raise OperationFailed("spline interpolation failed", o.id)
                    edge = BRepBuilderAPI_MakeEdge(interp.Curve()).Edge()
                    cur = list(allp[-1])
            wire.Add(edge)
            if not wire.IsDone():
                raise OperationFailed("profile segments are not connected", o.id)
        if math.dist(cur, start) > 1e-6:
            raise OperationFailed("profile is not closed (last point != start)", o.id)
        face = BRepBuilderAPI_MakeFace(wire.Wire(), True)
        if not face.IsDone():
            raise OperationFailed("cannot make planar face from profile", o.id)
        return face.Face()

    # ------------------------------------------------------------------ measurement
    def measure(self, shape) -> BodyMeasurements:
        vp, sp = GProp_GProps(), GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, vp)
        BRepGProp.SurfaceProperties_s(shape, sp)
        box = Bnd_Box()
        box.SetGap(0.0)
        BRepBndLib.AddOptimal_s(shape, box, False, False)
        lo, hi = box.CornerMin(), box.CornerMax()
        xmin, ymin, zmin, xmax, ymax, zmax = lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()
        com = vp.CentreOfMass()
        n_solids, n_shells = _count(shape, TopAbs_SOLID), _count(shape, TopAbs_SHELL)
        v, e, fcount = _count(shape, TopAbs_VERTEX), _count(shape, TopAbs_EDGE), _count(shape, TopAbs_FACE)
        closed = self._all_shells_closed(shape)
        valid = BRepCheck_Analyzer(shape).IsValid()
        # BREP Euler-Poincare for a single closed manifold solid with faces without inner loops:
        # V - E + F = 2 - 2g (+ ring-face corrections). Only reported as auxiliary info.
        return BodyMeasurements(
            backend=f"{self.name}@{self.version}", representation="brep", volume_mm3=vp.Mass(),
            area_mm2=sp.Mass(), bbox_min=(xmin, ymin, zmin), bbox_max=(xmax, ymax, zmax),
            center_of_mass=(com.X(), com.Y(), com.Z()), n_solids=n_solids, n_shells=n_shells, closed=closed,
            valid=valid, genus=None, tolerance_mm=0.0,
            extra={"n_vertices": v, "n_edges": e, "n_faces": fcount},
        )

    @staticmethod
    def _all_shells_closed(shape) -> bool:
        exp = TopExp_Explorer(shape, TopAbs_SHELL)
        any_shell = False
        while exp.More():
            any_shell = True
            shell = _api(TopoDS, 'Shell')(exp.Current())
            if not _api(BRep_Tool, 'IsClosed')(shell):
                return False
            exp.Next()
        return any_shell

    def intersection_volume(self, a, b) -> float:
        op = BRepAlgoAPI_Common(a, b)
        op.Build()
        if not op.IsDone():
            raise OperationFailed("common (intersection) failed")
        shp = op.Shape()
        if shp.IsNull():
            return 0.0
        vp = GProp_GProps()
        BRepGProp.VolumeProperties_s(shp, vp)
        return max(0.0, vp.Mass())

    # ------------------------------------------------------------------ tessellation / export
    def tessellate(self, shape, linear_deflection: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        lin = linear_deflection or self.mesh_linear_deflection_mm
        BRepMesh_IncrementalMesh(shape, lin, False, self.mesh_angular_deflection_rad, True)
        verts: list[tuple[float, float, float]] = []
        tris: list[tuple[int, int, int]] = []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        from OCP.TopAbs import TopAbs_REVERSED
        while exp.More():
            face = _api(TopoDS, 'Face')(exp.Current())
            loc = TopLoc_Location()
            tri = _api(BRep_Tool, 'Triangulation')(face, loc)
            if tri is not None:
                trsf = loc.Transformation()
                base = len(verts)
                for k in range(1, tri.NbNodes() + 1):
                    p = tri.Node(k).Transformed(trsf)
                    verts.append((p.X(), p.Y(), p.Z()))
                rev = face.Orientation() == TopAbs_REVERSED
                for k in range(1, tri.NbTriangles() + 1):
                    a, b, c = tri.Triangle(k).Get()
                    tris.append((base + a - 1, base + c - 1, base + b - 1) if rev else
                                (base + a - 1, base + b - 1, base + c - 1))
            exp.Next()
        if not tris:
            raise InvalidGeometry("tessellation produced no triangles")
        return np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int64)

    def export(self, shape, fmt: str) -> tuple[bytes, bool, float]:
        if fmt == "step":
            return self._step_bytes(shape), False, 0.0
        if fmt == "brep":
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "s.brep")
                BRepTools.Write_s(shape, p)
                return open(p, "rb").read(), False, 0.0
        if fmt in ("stl", "glb"):
            from .meshio import mesh_bytes
            v, t = self.tessellate(shape)
            return mesh_bytes(v, t, fmt), True, self.mesh_linear_deflection_mm
        raise UnsupportedConversion(f"occt cannot export {fmt}")

    @staticmethod
    def _step_bytes(shape) -> bytes:
        Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
        Interface_Static.SetCVal_s("write.step.unit", "MM")
        with tempfile.TemporaryDirectory() as d, _quiet_stdout():
            w = STEPControl_Writer()
            w.Transfer(shape, STEPControl_AsIs)
            p = os.path.join(d, "s.step")
            if w.Write(p) != IFSelect_ReturnStatus.IFSelect_RetDone:
                raise OperationFailed("STEP write failed")
            data = open(p, "rb").read()
        # The STEP header embeds the temp path, a wall-clock timestamp and a per-process product counter;
        # normalise them so identical geometry gives identical, content-addressable bytes.
        data = re.sub(rb"FILE_NAME\('[^']*','[^']*'", b"FILE_NAME('berkelium.step','1970-01-01T00:00:00'", data, count=1)
        data = re.sub(rb"Open CASCADE STEP translator [0-9.]+ [0-9]+", b"berkelium", data)
        counter = iter(range(1, 1 << 30))  # OCCT numbers assembly occurrences with a process-global counter
        return re.sub(rb"NEXT_ASSEMBLY_USAGE_OCCURRENCE\('[0-9]+'",
                      lambda _m: b"NEXT_ASSEMBLY_USAGE_OCCURRENCE('%d'" % next(counter), data)



