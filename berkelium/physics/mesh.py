"""Mesh layer (PHYSICS-2): geometry -> tetrahedral mesh with semantic regions, quality and hashes.

Generators
* ``structured_box`` — deterministic tensor-product tets of an axis-aligned box (verification problems;
  exact nested refinement). Provides tags xmin/xmax/ymin/ymax/zmin/zmax.
* ``gmsh`` — the general path: a STEP artifact from the OCCT backend is meshed by the Gmsh executable in a
  SEPARATE PROCESS (Gmsh is GPL-2.0-or-later; Berkelium never links it — ADR-016). Single-threaded and
  option-pinned for reproducibility.

Regions are resolved on the mesh by level-set selectors over facet centroid/normal (see schema.Region), so a
boundary condition is attached to a *meaning* ("the face at x = L"), never to a mesher's facet numbering.
All coordinates inside a MeshBundle are metres; selectors are evaluated in millimetres.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..expr.parser import parse
from ..expr.vectorized import evaluate_array
from .schema import MeshStats, Region

MM = 1e-3


class MeshError(RuntimeError):
    pass


@dataclass
class MeshBundle:
    mesh: object                      # skfem.MeshTet1 (metres)
    generator: str
    generator_version: str
    h_mm: float
    tags: dict[str, np.ndarray] = field(default_factory=dict)   # generator-provided facet sets
    sha256: str = ""

    def facets(self) -> np.ndarray:
        return self.mesh.boundary_facets()

    def facet_geometry(self, facets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Centroids (mm) and unit OUTWARD normals of boundary facets."""
        m = self.mesh
        tri = m.p[:, m.facets[:, facets]]                       # (3, 3 vertices, nf)
        c = tri.mean(axis=1)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], axis=0)
        n /= np.linalg.norm(n, axis=0)
        cell = m.f2t[0, facets]                                  # owning tet for boundary facets
        inward = m.p[:, m.t[:, cell]].mean(axis=1) - c
        n *= np.where((n * inward).sum(axis=0) > 0, -1.0, 1.0)
        return c / MM, n

    def resolve(self, region: Region) -> np.ndarray:
        if region.tag is not None:
            if region.tag not in self.tags:
                raise MeshError(f"region {region.id}: generator {self.generator} provides no tag {region.tag!r}; "
                                f"available {sorted(self.tags)}")
            sel = self.tags[region.tag]
        else:
            # Evaluated at the facet's VERTICES (which lie on the exact CAD surface), not its centroid
            # (which sags inside curved surfaces by ~h^2/8R): a facet belongs iff all 3 vertices satisfy
            # field <= 0. Normals come from the facet itself.
            bf = self.facets()
            _, n = self.facet_geometry(bf)
            node = parse(region.field)
            worst = None
            for k in range(3):
                v = self.mesh.p[:, self.mesh.facets[k, bf]] / MM
                env = {"x": v[0], "y": v[1], "z": v[2], "nx": n[0], "ny": n[1], "nz": n[2]}
                val = np.broadcast_to(evaluate_array(node, env), bf.shape)
                worst = val if worst is None else np.maximum(worst, val)
            sel = bf[worst <= 0.0]
        if sel.size == 0:
            raise MeshError(f"region {region.id} selects no boundary facets")
        return np.sort(sel)


def _hash(p: np.ndarray, t: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.round(p, 12), dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(t, dtype="<i8").tobytes())
    return h.hexdigest()


def volume(mesh) -> float:
    P = mesh.p[:, mesh.t]
    a, b, c, d = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    return float(np.abs(np.einsum("ij,ij->j", b - a, np.cross(c - a, d - a, axis=0))).sum() / 6.0)


def quality(mesh) -> np.ndarray:
    """Normalised tet shape quality q = 12 (3V)^(2/3) / sum(l^2); 1 = regular tet, -> 0 degenerate."""
    P = mesh.p[:, mesh.t]                                        # (3, 4, nt)
    a, b, c, d = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    V = np.abs(np.einsum("ij,ij->j", b - a, np.cross(c - a, d - a, axis=0))) / 6.0
    edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
    s = sum(((u - v) ** 2).sum(axis=0) for u, v in edges)
    return 12.0 * (3.0 * V) ** (2.0 / 3.0) / s


def stats(b: MeshBundle, regions: dict[str, np.ndarray], element_type: str) -> MeshStats:
    q = quality(b.mesh)
    return MeshStats(generator=f"{b.generator}@{b.generator_version}", element_type=element_type,
                     n_nodes=int(b.mesh.p.shape[1]), n_cells=int(b.mesh.t.shape[1]),
                     n_boundary_facets=int(b.facets().size), h_mm=b.h_mm, min_quality=float(q.min()),
                     mean_quality=float(q.mean()), n_poor=int((q < 0.1).sum()),
                     regions={k: int(v.size) for k, v in sorted(regions.items())}, sha256=b.sha256)


def structured_box(size_mm: tuple[float, float, float], divisions: tuple[int, int, int],
                   origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> MeshBundle:
    import skfem
    axes = [np.linspace(o, o + s, n + 1) * MM for o, s, n in zip(origin_mm, size_mm, divisions, strict=True)]
    m = skfem.MeshTet.init_tensor(*axes)
    bf = m.boundary_facets()
    c = m.p[:, m.facets[:, bf]].mean(axis=1)
    tags = {}
    for i, ax in enumerate("xyz"):
        lo, hi = axes[i][0], axes[i][-1]
        tol = 1e-9 * max(1.0, abs(hi - lo))
        tags[f"{ax}min"] = bf[np.abs(c[i] - lo) < tol]
        tags[f"{ax}max"] = bf[np.abs(c[i] - hi) < tol]
    h = float(np.prod([s / n for s, n in zip(size_mm, divisions, strict=True)]) ** (1 / 3))
    return MeshBundle(m, "structured_box", "0.1", h, tags, _hash(m.p, m.t))


def gmsh_version() -> str | None:
    exe = shutil.which("gmsh")
    if not exe:
        return None
    r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
    return (r.stdout.strip() or r.stderr.strip()).splitlines()[-1] if r.returncode == 0 else None


def gmsh_from_step(step: bytes, size_mm: float, curvature_points: int = 12, timeout_s: float = 600.0) -> MeshBundle:
    """Mesh a STEP solid with the Gmsh executable (out of process). STEP units are mm (OCCT backend writes
    MM); the mesh is converted to metres here, explicitly."""
    import meshio
    import skfem
    exe, ver = shutil.which("gmsh"), gmsh_version()
    if not exe or not ver:
        raise MeshError("gmsh executable not found (install gmsh; it runs as a separate process)")
    with tempfile.TemporaryDirectory() as d:
        Path(d, "part.step").write_bytes(step)
        Path(d, "mesh.geo").write_text(
            'SetFactory("OpenCASCADE");\n'
            'Geometry.OCCTargetUnit = "MM";\n'
            'Merge "part.step";\n'
            "General.NumThreads = 1;\n"
            "Mesh.Algorithm = 6;\nMesh.Algorithm3D = 1;\n"
            f"Mesh.MeshSizeMax = {size_mm!r};\nMesh.MeshSizeMin = {size_mm / 4!r};\n"
            f"Mesh.MeshSizeFromCurvature = {int(curvature_points)};\n"
            "Mesh.Optimize = 1;\nMesh.ElementOrder = 1;\nMesh.SaveAll = 1;\n"
            "Mesh 3;\n"
            'Save "mesh.msh";\n')
        r = subprocess.run([exe, "mesh.geo", "-0", "-nopopup", "-format", "msh41"], cwd=d, capture_output=True,
                           text=True, timeout=timeout_s)
        out = Path(d, "mesh.msh")
        if r.returncode != 0 or not out.exists():
            raise MeshError(f"gmsh failed ({r.returncode}): {(r.stderr or r.stdout)[-600:]}")
        mm = meshio.read(out)
    tets = [c.data for c in mm.cells if c.type == "tetra"]
    if not tets:
        raise MeshError("gmsh produced no tetrahedra (is the STEP a closed solid?)")
    t = np.vstack(tets)
    used = np.unique(t)
    remap = -np.ones(mm.points.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    p = mm.points[used].T * MM
    t = remap[t].T
    m = skfem.MeshTet(np.ascontiguousarray(p), np.ascontiguousarray(t))
    return MeshBundle(m, "gmsh", ver, float(size_mm), {}, _hash(m.p, m.t))
