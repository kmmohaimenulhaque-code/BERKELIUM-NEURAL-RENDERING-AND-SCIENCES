"""Deterministic execution pipeline (architecture §2, §9):

DesignProposal -> L0 schema -> normalisation -> per component: CEM derive/check/expand or procedural
graph resolution -> backend planning -> realisation -> measurement -> artifacts -> L1-L7 validation
-> DesignRecord (content-hashed, with provenance). Job stages are recorded separately from the record
so wall-clock timings never enter hashed content.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from . import __version__
from .cem.protocol import CEM, CEMRegistry, GeometryContext, default_registry
from .geometry.backend import BackendError, GeometryBackend, available_backends, plan_backend
from .manufacturing import validate_manufacturing
from .schema.common import Diagnostic, Provenance, QuantityModel
from .schema.design import Component, DesignProposal, Parameter
from .schema.evaluation import ArtifactRef, Evaluation, Fidelity, Measurement, ValidationResult
from .schema.hashing import sha256_bytes, sha256_of
from .schema.record import DesignRecord, RealizedComponent
from .units import Quantity, UnitError
from .validation import (
    constraint_results,
    geometric_results,
    parameter_results,
    requirement_results,
    schema_result,
    summarize,
)

GENERATOR = f"berkelium.pipeline@{__version__}"


# ------------------------------------------------------------------ job graph / events
@dataclass
class Stage:
    name: str
    status: str = "pending"
    seconds: float = 0.0
    detail: str = ""


@dataclass
class Job:
    stages: list[Stage] = field(default_factory=list)
    on_event: Callable[[Stage], None] | None = None

    def run(self, name: str, fn: Callable[[], Any]) -> Any:
        st = Stage(name, "running")
        self.stages.append(st)
        if self.on_event:
            self.on_event(st)
        t = time.perf_counter()
        try:
            out = fn()
            st.status = "done"
            return out
        except Exception as e:
            st.status, st.detail = "failed", f"{type(e).__name__}: {e}"
            raise
        finally:
            st.seconds = round(time.perf_counter() - t, 6)
            if self.on_event:
                self.on_event(st)


class ArtifactStore:
    """Content-addressed files: <root>/<sha[:2]>/<sha>.<ext>."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put(self, data: bytes, ext: str) -> tuple[str, str]:
        h = sha256_bytes(data)
        rel = f"{h[:2]}/{h}.{ext}"
        p = self.root / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return h, rel

    def path(self, sha: str, ext: str) -> Path:
        return self.root / sha[:2] / f"{sha}.{ext}"


@dataclass
class PipelineResult:
    record: DesignRecord | None
    job: Job
    schema_errors: list[dict] | None = None


# ------------------------------------------------------------------ normalisation
def normalize_inputs(model: type[BaseModel], raw: dict[str, Any]) -> BaseModel:
    """Convert {"value", "unit"} inputs to each field's declared unit, then validate the typed model."""
    out: dict[str, Any] = {}
    for k, v in raw.items():
        fld = model.model_fields.get(k)
        unit = (fld.json_schema_extra or {}).get("unit") if fld is not None and isinstance(
            fld.json_schema_extra, dict) else None
        if isinstance(v, QuantityModel):
            v = v.model_dump()
        if isinstance(v, dict) and set(v) <= {"value", "unit"} and "value" in v:
            q = Quantity.of(v["value"], v.get("unit", "1"))
            v = q.to(unit) if unit else q.to("1")
        out[k] = v
    return model.model_validate(out)


def _param_env(params: list[Parameter]) -> dict[str, Any]:
    env: dict[str, Any] = {}
    for p in params:
        v = p.value
        if isinstance(v, QuantityModel):
            env[p.id] = v.q()
        elif isinstance(v, bool):
            env[p.id] = v
        elif isinstance(v, int | float):
            unit = p.unit or ("deg" if p.kind == "angle" else "mm" if p.kind == "length" else "1")
            env[p.id] = Quantity.of(v, unit)
    return env


def _plain(model: BaseModel) -> tuple[dict, dict]:
    vals, units = {}, {}
    for k, f in type(model).model_fields.items():
        v = getattr(model, k)
        if v is None or isinstance(v, list):
            continue
        vals[k] = v
        if isinstance(f.json_schema_extra, dict) and "unit" in f.json_schema_extra:
            units[k] = f.json_schema_extra["unit"]
    return vals, units


# ------------------------------------------------------------------ main entry
def run(proposal: DesignProposal | dict | str, *, registry: CEMRegistry | None = None,
        realize: bool = True, artifacts: ArtifactStore | None = None, backend: str | None = None,
        backends: dict[str, GeometryBackend] | None = None,
        on_event: Callable[[Stage], None] | None = None) -> PipelineResult:
    job = Job(on_event=on_event)
    reg = registry or default_registry()

    def _parse() -> DesignProposal:
        if isinstance(proposal, DesignProposal):
            return proposal
        if isinstance(proposal, str):
            return DesignProposal.model_validate_json(proposal)
        return DesignProposal.model_validate(proposal)

    try:
        prop = job.run("schema", _parse)
    except ValidationError as e:
        return PipelineResult(None, job, schema_errors=e.errors(include_url=False))

    results: list[ValidationResult] = [schema_result(True, "DesignProposal validates against schema 0.1.0")]
    ev = Evaluation()
    realized: list[RealizedComponent] = []
    env: dict[str, Any] = _param_env(prop.specification.parameters)
    sources: dict[str, str] = {k: "specification.parameters" for k in env}
    results += parameter_results(prop.specification.parameters)
    bks = backends if backends is not None else (available_backends() if realize else {})

    for comp in prop.structure.components:
        try:
            if comp.kind == "cem":
                job.run(f"cem:{comp.id}", lambda c=comp: _run_cem(c, reg, realize, bks, backend, artifacts, ev,
                                                                    results, realized, env, sources, prop))
            else:
                job.run(f"procedural:{comp.id}", lambda c=comp: _run_procedural(
                    c, realize, bks, backend, artifacts, ev, results, realized, env, sources, prop))
        except _ComponentFailed:
            continue

    if prop.specification.analyses:
        job.run("analyses", lambda: _run_analyses(prop, realize, bks, backend, artifacts, ev, results, realized))
    results += job.run("requirements", lambda: requirement_results(prop.specification.requirements, env, sources))
    results += job.run("constraints", lambda: constraint_results(prop.specification.constraints, env))
    ev.validation = summarize(results)
    p_hash = sha256_of(prop)
    rec = DesignRecord(id=f"design-{p_hash[:16]}", intent=prop.intent, specification=prop.specification,
                       structure=prop.structure, realized=realized, evaluation=ev,
                       provenance=Provenance(author="system", generator=GENERATOR, inputs_hash=p_hash),
                       proposal_hash=p_hash)
    rec.content_hash = sha256_of(rec)
    return PipelineResult(rec, job)


class _ComponentFailed(Exception):
    pass


def _run_analyses(prop, realize, bks, backend, artifacts, ev, results, realized) -> None:
    """Realise each analysed component through the exact-BREP backend, hand its STEP to the physics layer,
    and record SimulationResults + L7 results. Nothing here is model-written."""
    from .physics import l7
    from .physics.run import run_case
    from .physics.schema import SimulationResult
    mats = {m.id: m for m in prop.specification.materials}
    graphs = {rc.component: rc.geometry for rc in realized if rc.geometry is not None}
    for c in prop.structure.components:
        if c.kind == "procedural" and c.geometry is not None:
            graphs.setdefault(c.id, c.geometry)
    sink = (lambda data, ext: artifacts.put(data, ext)[0]) if artifacts else None
    for case in prop.specification.analyses:
        comp, _, body = case.target.partition(".")
        step = sha = None
        needs_geo = case.physics in ("structural_linear_static", "thermal_steady") and \
            (case.mesh is None or case.mesh.generator == "gmsh")
        if needs_geo:
            if not realize or "occt" not in bks or comp not in graphs:
                why = "realize=False" if not realize else ("no OCCT backend" if "occt" not in bks
                                                           else f"no geometry for {comp!r}")
                r = SimulationResult(case_id=case.id, physics=case.physics, target=case.target,
                                     fidelity="not_evaluated", status="not_evaluated", message=f"not run: {why}")
                ev.simulation.append(r)
                results.extend(l7.case_results(r))
                continue
            be = bks["occt"]
            g = graphs[comp]
            g = g.resolve({}) if hasattr(g, "is_resolved") and not g.is_resolved() else g
            real = be.execute(g)
            shape = real.bodies[body] if body else next(iter(real.bodies.values()))
            step = be.export(shape, "step")[0]
            from .schema.hashing import sha256_bytes
            sha = sha256_bytes(step)
        r = run_case(case, mats, step=step, geometry_sha256=sha, sink=sink)
        ev.simulation.append(r)
        results.extend(l7.case_results(r))
    results.extend(l7.requirement_results(prop.specification.requirements, ev.simulation))


def _fail(results, level, target, msg, code, ev) -> None:
    results.append(ValidationResult(validator="pipeline@0.1", level=level, status="fail", target=target,
                                    message=msg, fidelity=Fidelity(kind="rule", method=code)))
    ev.diagnostics.append(Diagnostic(code=code, severity="error", message=msg, path=f"/structure/{target}"))
    raise _ComponentFailed


def _run_cem(comp: Component, reg, realize, bks, backend, artifacts, ev, results, realized, env, sources, prop):
    try:
        cem: CEM = reg.get(comp.cem)
    except KeyError as e:
        _fail(results, 0, comp.id, str(e), "UNKNOWN_CEM", ev)
    assumptions: list[str] = []
    try:
        if comp.requirements:
            req = normalize_inputs(cem.Requirements, comp.requirements)
            d = cem.derive(req)
            params = d.parameters
            ev.derivations.extend(x.model_copy(update={"id": f"{comp.id}_{x.id}", "target": comp.id})
                                  for x in d.derivations)
            assumptions = d.assumptions
            if comp.parameters:
                merged = params.model_dump() | normalize_inputs(cem.Parameters, params.model_dump()
                                                                | dict(comp.parameters)).model_dump()
                params = cem.Parameters.model_validate(merged)
        else:
            params = normalize_inputs(cem.Parameters, dict(comp.parameters))
    except (ValidationError, UnitError) as e:
        _fail(results, 1, comp.id, f"invalid CEM inputs: {e}".splitlines()[0] + f" ({len(getattr(e, 'errors', lambda: [])())} errors)",
              "INVALID_CEM_INPUT", ev)
    except ValueError as e:
        _fail(results, 6, comp.id, f"derivation failed: {e}", "DERIVE_FAILED", ev)
    results.append(ValidationResult(validator="param.core@0.1", level=1, status="pass", target=comp.id,
                                    message=f"{cem.meta.ref} parameters validate", fidelity=Fidelity(kind="rule")))
    checks = cem.check(params)
    for r in checks:
        r.target = r.target.replace("pair", comp.id, 1) if r.target.startswith("pair") else r.target
    results.extend(checks)
    for k, q in cem.quantities(params).items():
        env[f"{comp.id}.{k}"] = q
        sources[f"{comp.id}.{k}"] = f"cem:{cem.meta.ref}"
    vals, units = _plain(params)
    rc = RealizedComponent(component=comp.id, cem=cem.meta.ref, parameters=vals, parameter_units=units,
                           provenance=Provenance(author=f"cem:{cem.meta.ref}", generator=GENERATOR,
                                                 inputs_hash=sha256_of(params), references=list(cem.meta.references)))
    realized.append(rc)
    if assumptions:
        ev.diagnostics.extend(Diagnostic(code="ASSUMPTION", severity="info", message=a, path=f"/structure/{comp.id}")
                              for a in assumptions)
    if any(r.status == "fail" for r in checks):
        results.append(ValidationResult(validator="pipeline@0.1", level=3, status="not_evaluated", target=comp.id,
                                        message="geometry not realised: analytic checks failed",
                                        fidelity=Fidelity(kind="geometric")))
        return
    try:
        geo = cem.expand(params)
    except ValueError as e:
        _fail(results, 3, comp.id, f"expand failed: {e}", "EXPAND_FAILED", ev)
    rc.geometry = geo.graph
    analysed = any(a.target.partition(".")[0] == comp.id for a in prop.specification.analyses)
    for name, a in cem.analyses().items():
        if a == "solver_required" and not analysed:
            results.append(ValidationResult(validator="simulation@0.1", level=7, status="not_evaluated",
                                            target=comp.id, message=f"{name}: solver required; no analysis case requested",
                                            fidelity=Fidelity(kind="numerical")))
    if not realize:
        results.append(ValidationResult(validator="pipeline@0.1", level=3, status="not_evaluated", target=comp.id,
                                        message="geometry realisation skipped (realize=False)",
                                        fidelity=Fidelity(kind="geometric")))
        return
    bodies, meas, b = _realize(comp.id, geo.graph, bks, backend, artifacts, ev, results, env, sources, prop)
    ctx = GeometryContext(backend=b, bodies=bodies, measurements=meas, parameters=params)
    for v in cem.geometry_validators():
        for r in v(ctx):
            r.target = r.target.replace("pair", comp.id, 1) if r.target.startswith("pair") else r.target
            results.append(r)


def _run_procedural(comp, realize, bks, backend, artifacts, ev, results, realized, env, sources, prop):
    try:
        graph = comp.geometry.resolve(env)
    except ValueError as e:
        _fail(results, 2, comp.id, f"cannot resolve geometry expressions: {e}", "UNRESOLVED_GEOMETRY", ev)
    realized.append(RealizedComponent(component=comp.id, geometry=graph,
                                      provenance=Provenance(author="system", generator=GENERATOR,
                                                            inputs_hash=sha256_of(comp.geometry))))
    if not realize:
        results.append(ValidationResult(validator="pipeline@0.1", level=3, status="not_evaluated", target=comp.id,
                                        message="geometry realisation skipped (realize=False)",
                                        fidelity=Fidelity(kind="geometric")))
        return
    _realize(comp.id, graph, bks, backend, artifacts, ev, results, env, sources, prop,
             multi_body_ok="multi_body" in comp.tags)


def _realize(cid, graph, bks, backend, artifacts, ev, results, env, sources, prop, multi_body_ok=False):
    try:
        b = plan_backend(graph, bks, prefer=backend)
        real = b.execute(graph)
    except BackendError as e:
        _fail(results, 3, cid, f"geometry realisation failed: {e}", getattr(e, "code", "BACKEND_ERROR"), ev)
    for note in real.approximations:
        ev.diagnostics.append(Diagnostic(code="APPROXIMATION", severity="info", message=note, path=f"/structure/{cid}"))
    meas = {}
    for name, h in sorted(real.bodies.items()):
        m = b.measure(h)
        meas[name] = m
        t = f"{cid}.{name}"
        tol = QuantityModel(value=m.tolerance_mm, unit="mm")
        ev.measurements += [
            Measurement(id=f"{cid}_{name}_volume", target=t, name="volume", method="backend", backend=m.backend,
                        value=QuantityModel(value=m.volume_mm3, unit="mm^3"), tolerance=tol),
            Measurement(id=f"{cid}_{name}_area", target=t, name="surface_area", method="backend", backend=m.backend,
                        value=QuantityModel(value=m.area_mm2, unit="mm^2"), tolerance=tol),
            Measurement(id=f"{cid}_{name}_bbox", target=t, name="bbox_min_max_mm", method="backend",
                        backend=m.backend, vector=[*m.bbox_min, *m.bbox_max], tolerance=tol)]
        env[f"{t}.volume"] = Quantity.of(m.volume_mm3, "mm^3")
        sources[f"{t}.volume"] = f"{cid}_{name}_volume"
        for k, ax in enumerate("xyz"):
            env[f"{t}.extent_{ax}"] = Quantity.of(m.bbox_max[k] - m.bbox_min[k], "mm")
        results += geometric_results(t, m, None if multi_body_ok else 1)
        results += validate_manufacturing(prop.specification.manufacturing, t, m.bbox_min, m.bbox_max)
        if artifacts is not None:
            fmts = ("step", "glb") if b.representation == "brep" else ("glb", "stl")
            for fmt in fmts:
                data, lossy, tol_mm = b.export(h, fmt)
                sha, rel = artifacts.put(data, fmt)
                ev.artifacts.append(ArtifactRef(sha256=sha, format=fmt, target=t, backend=f"{b.name}@{b.version}",
                                                representation="brep" if fmt == "step" else "mesh", lossy=lossy,
                                                tolerance=QuantityModel(value=tol_mm, unit="mm") if lossy else None,
                                                bytes=len(data), path=rel))
    return real.bodies, meas, b
