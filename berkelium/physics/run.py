"""Run an AnalysisCase deterministically and return a SimulationResult (never a bare number).

For numerical physics: mesh at ``convergence.levels`` systematically refined levels, solve each, compute
every QoI on each level, estimate discretisation error per QoI (verification.study), and report the finest
value with its GCI band. Status is the worst over QoIs: failed > unsupported > non_converged > converged.
"""

from __future__ import annotations

import io
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..schema.hashing import sha256_of
from . import mesh as meshmod
from .fluids import OpenFOAMAdapter, pipe_flow
from .schema import (
    AnalysisCase,
    ConvergenceStudy,
    Estimate,
    FieldArtifact,
    LevelSolution,
    Material,
    SimProvenance,
    SimulationResult,
)
from .solvers.fem import SolveError, elastic, thermal
from .verification import study

ArtifactSink = Callable[[bytes, str], str]    # (data, ext) -> sha256

REFS = ["Celik et al. (2008) J. Fluids Eng. 130(7):078001 (GCI procedure)",
        "Roache (1994) J. Fluids Eng. 116:405-413 (GCI)",
        "scikit-fem: Gustafsson & McBain (2020) JOSS 5(52):2369"]


def _formal_order(case: AnalysisCase, qid: str, k: int) -> float:
    q = next(o for o in case.outputs if o.id == qid)
    if q.field in ("von_mises", "heat_flux_normal") and q.kind != "flux":
        return float(k)
    return float(k + 1)


def _vtu(b: meshmod.MeshBundle, point_data: dict, cell_data: dict) -> bytes:
    import meshio
    pts = b.mesh.p.T
    cells = [("tetra", b.mesh.t.T)]
    m = meshio.Mesh(pts, cells, point_data=point_data, cell_data={k: [v] for k, v in cell_data.items()})
    with tempfile.TemporaryDirectory() as d:
        f = Path(d, "f.vtu")
        meshio.write(f, m, binary=False)
        return f.read_bytes()


def run_case(case: AnalysisCase, materials: dict[str, Material], step: bytes | None = None,
             geometry_sha256: str | None = None, sink: ArtifactSink | None = None) -> SimulationResult:
    t0 = time.perf_counter()
    base = dict(case_id=case.id, physics=case.physics, target=case.target)
    if case.physics == "pipe_flow_reduced_order":
        return pipe_flow(case)
    if case.physics == "cfd":
        return OpenFOAMAdapter().run(case, step or b"")
    mat = materials.get(case.material or "")
    if mat is None:
        return SimulationResult(**base, fidelity="not_evaluated", status="failed",
                                message=f"unknown material {case.material!r}")
    if case.mesh is None:
        return SimulationResult(**base, fidelity="not_evaluated", status="failed", message="no mesh spec")
    ms, cv = case.mesh, case.convergence
    k = ms.element_order
    solver = {"structural_linear_static": elastic, "thermal_steady": thermal}[case.physics]
    levels, stats_l, warns, fields = [], [], [], []
    last = None
    size0 = ms.size.q().to("mm")
    for lv in range(cv.levels):
        try:
            if ms.generator == "structured_box":
                if ms.divisions is None or step is not None:
                    raise SolveError("structured_box needs 'divisions' and box extents (no STEP)")
                ratio = round(cv.refinement_ratio)
                if abs(ratio - cv.refinement_ratio) > 1e-9:
                    raise SolveError("structured_box needs an integer refinement ratio")
                ext = ms.box_mm or [d * size0 for d in ms.divisions]
                b = meshmod.structured_box(tuple(ext), tuple(d * ratio ** lv for d in ms.divisions))
            else:
                if step is None:
                    raise SolveError("gmsh path needs the component's STEP artifact")
                # size AND curvature resolution refine together, so the levels are systematic
                b = meshmod.gmsh_from_step(step, size0 / cv.refinement_ratio ** lv,
                                           curvature_points=round(12 * cv.refinement_ratio ** lv))
            regions = {r.id: b.resolve(r) for r in case.regions}
            out = solver(b, case, mat, regions, k)
        except (SolveError, meshmod.MeshError, ValueError) as e:
            return SimulationResult(**base, fidelity="numerical", status="failed",
                                    message=f"level {lv}: {e}", levels=levels, mesh=stats_l, warnings=warns,
                                    wall_seconds=round(time.perf_counter() - t0, 3))
        st = meshmod.stats(b, regions, f"tetra P{k}")
        if st.n_poor:
            warns.append(f"level {lv}: {st.n_poor} cells with shape quality < 0.1")
        if out.residual > case.solver.residual_tolerance:
            return SimulationResult(**base, fidelity="numerical", status="non_converged",
                                    message=f"level {lv}: linear residual {out.residual:.2e} > "
                                            f"{case.solver.residual_tolerance:.0e}", levels=levels, mesh=stats_l + [st])
        h_eff = float((meshmod.volume(b.mesh) / b.mesh.t.shape[1]) ** (1 / 3) / meshmod.MM) \
            if ms.generator == "gmsh" else b.h_mm
        levels.append(LevelSolution(level=lv, h_mm=h_eff, n_cells=st.n_cells, n_dofs=out.n_dofs,
                                    values=out.values, residual=out.residual, balance=out.balance,
                                    mesh_sha256=st.sha256))
        stats_l.append(st)
        warns += [w for w in out.warnings if w not in warns]
        last = (b, out)
    studies: list[ConvergenceStudy] = []
    est: list[Estimate] = []
    for q in case.outputs:
        vals = [lvl.values[q.id] for lvl in levels]
        hs = [lvl.h_mm for lvl in levels]
        same = [abs(lv.values[o.id]) for lv in levels for o in case.outputs if o.unit == q.unit]
        floor = 1e-6 * max(same)      # near-zero QoIs are judged against the case's same-unit scale
        s = study(q.id, vals, hs, cv.safety_factor, cv.max_relative_error, _formal_order(case, q.id, k), floor)
        studies.append(s)
        derr = None if s.gci_fine is None else s.gci_fine * max(abs(vals[-1]), floor)
        est.append(Estimate(id=q.id, value=vals[-1], unit=q.unit, fidelity="numerical",
                            status=s.status if s.status != "not_evaluated" else "non_converged",
                            discretization_error=derr, method=f"{case.physics} FE P{k}; {s.method}",
                            notes=[s.message]))
    rank = {"failed": 4, "unsupported": 3, "non_converged": 2, "not_evaluated": 1, "converged": 0}
    status = max((e.status for e in est), key=rank.get, default="converged")
    if last is not None and sink is not None:
        b, out = last
        data = _vtu(b, out.point_data, out.cell_data)
        fields.append(FieldArtifact(name=f"{case.id}.fields", sha256=sink(data, "vtu"), format="vtu",
                                    bytes=len(data), level=len(levels) - 1))
    mesher = stats_l[-1].generator if stats_l else "-"
    import skfem
    prov = SimProvenance(
        solver=f"berkelium.fem.{case.physics}", solver_version="0.1",
        mesher=mesher.split("@")[0], mesher_version=mesher.split("@")[-1],
        geometry_sha256=geometry_sha256, case_sha256=sha256_of(case.model_dump(mode="json")),
        material_sha256=sha256_of(mat.model_dump(mode="json")),
        settings={"skfem": getattr(skfem, "__version__", "unknown"), "numpy": np.__version__,
                  "element_order": k, "linear_solver": case.solver.linear_solver}, references=REFS)
    msg = {"converged": "all quantities of interest converged within the stated discretisation tolerance",
           "non_converged": "solved, but at least one quantity is not mesh-converged (see convergence)"}
    return SimulationResult(**base, fidelity="numerical", status=status, message=msg.get(status, status),
                            estimates=est, convergence=studies, levels=levels, mesh=stats_l, fields=fields,
                            assumptions=_assumptions(case), warnings=warns, provenance=prov,
                            wall_seconds=round(time.perf_counter() - t0, 3))


def _assumptions(case: AnalysisCase) -> list[str]:
    a = {"structural_linear_static": ["small strain, linear elastic isotropic material, static loading",
                                      "error bands are DISCRETISATION error only (model-form error not included)"],
         "thermal_steady": ["steady state, constant isotropic conductivity",
                            "error bands are DISCRETISATION error only (model-form error not included)"]}
    return a.get(case.physics, []) + list(case.assumptions)


_ = io  # reserved for in-memory writers
