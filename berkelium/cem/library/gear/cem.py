"""Spur gear pair CEM (external involute spur gears, 20 deg full-depth basic rack).

Scope and honesty: closed-form geometry checks (undercut, involute interference, fillet contact,
root clearance, pointed tips, contact ratio) plus a *preliminary* Lewis bending estimate.
AGMA/ISO rating, contact (pitting) stress, dynamics, lubrication and materials are NOT evaluated
and are reported as such. Sources: see formulas.py and docs/SOURCES_AND_LICENSES.md.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, model_validator

from ...._base import Strict
from ....geometry.ir import GeometryGraph
from ....schema.common import QuantityModel
from ....schema.design import Frame, Port, Relation
from ....schema.evaluation import Derivation, Fidelity, ValidationResult
from ....units import Quantity
from ...protocol import CEM, CEMMeta, ComponentGeometry, DeriveResult, GeometryContext
from . import formulas as F
from .profile import gear_outline_segments, generate_flank

ALPHA_DEG = 20.0
ALPHA = math.radians(ALPHA_DEG)
VALIDATOR = "cem.gear.spur_pair@0.1"
Process = Literal["cut_or_milled", "hobbed_or_shaped", "shaved_or_ground", "cast_iron_cast_profile"]
# Designer-overridable default candidate set for module selection. This is a list of options to search,
# not a claim of conformance to any preferred-number standard.
DEFAULT_CANDIDATE_MODULES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
SHIFT_RANGE = (-0.5, 1.0)


def U(unit: str) -> Any:
    return Field(json_schema_extra={"unit": unit})


class Requirements(Strict):
    """What a user (or model) can ask for. Numbers are in the unit named in each field's schema."""

    ratio: Annotated[float, Field(ge=1.0, le=10.0, description="gear teeth / pinion teeth")]
    ratio_tolerance: Annotated[float, Field(ge=0.0, le=0.2, description="relative")] = 0.02
    module: Annotated[float | None, Field(gt=0, le=50, json_schema_extra={"unit": "mm"})] = None
    center_distance: Annotated[float | None, Field(gt=0, json_schema_extra={"unit": "mm"})] = None
    face_width: Annotated[float | None, Field(gt=0, json_schema_extra={"unit": "mm"})] = None
    min_pinion_teeth: Annotated[int | None, Field(ge=6, le=200)] = None
    pinion_torque: Annotated[float | None, Field(gt=0, json_schema_extra={"unit": "N*mm"})] = None
    pinion_speed: Annotated[float | None, Field(gt=0, json_schema_extra={"unit": "rev/min"})] = None
    allowable_bending_stress: Annotated[float | None, Field(gt=0, json_schema_extra={"unit": "MPa"})] = None
    manufacturing: Process = "cut_or_milled"
    candidate_modules: list[float] = Field(default_factory=lambda: list(DEFAULT_CANDIDATE_MODULES))
    bore_diameter_pinion: Annotated[float | None, Field(gt=0, json_schema_extra={"unit": "mm"})] = None
    bore_diameter_gear: Annotated[float | None, Field(gt=0, json_schema_extra={"unit": "mm"})] = None


class Parameters(Strict):
    module: Annotated[float, Field(gt=0.05, le=50, json_schema_extra={"unit": "mm"})]
    z1: Annotated[int, Field(ge=6, le=400, description="pinion teeth")]
    z2: Annotated[int, Field(ge=6, le=400, description="gear teeth")]
    x1: Annotated[float, Field(ge=-1.0, le=1.5)] = 0.0
    x2: Annotated[float, Field(ge=-1.0, le=1.5)] = 0.0
    pressure_angle: Literal[20.0] = Field(20.0, json_schema_extra={"unit": "deg"})
    face_width: Annotated[float, Field(gt=0, json_schema_extra={"unit": "mm"})]
    backlash_allowance: Annotated[float, Field(ge=0, json_schema_extra={"unit": "mm"},
                                               description="circular thinning per gear at the reference circle")] = 0.0
    bore_diameter_pinion: Annotated[float | None, Field(gt=0, json_schema_extra={"unit": "mm"})] = None
    bore_diameter_gear: Annotated[float | None, Field(gt=0, json_schema_extra={"unit": "mm"})] = None
    pinion_torque: Annotated[float | None, Field(gt=0, json_schema_extra={"unit": "N*mm"})] = None
    pinion_speed: Annotated[float | None, Field(gt=0, json_schema_extra={"unit": "rev/min"})] = None
    allowable_bending_stress: Annotated[float | None, Field(gt=0, json_schema_extra={"unit": "MPa"})] = None
    manufacturing: Process = "cut_or_milled"

    @model_validator(mode="after")
    def _ordering(self) -> Parameters:
        if self.z2 < self.z1:
            raise ValueError("z1 is the pinion: require z2 >= z1")
        return self


def _r(status: str, target: str, msg: str, kind: str = "analytic", method: str | None = None,
       measured: tuple[float, str] | None = None, limit: tuple[float, str] | None = None,
       comparator: str | None = None, refs: tuple[str, ...] = (), assumptions: tuple[str, ...] = ()) -> ValidationResult:
    return ValidationResult(
        validator=VALIDATOR, level=6, status=status, target=target, message=msg,  # type: ignore[arg-type]
        measured=QuantityModel(value=measured[0], unit=measured[1]) if measured else None,
        limit=QuantityModel(value=limit[0], unit=limit[1]) if limit else None, comparator=comparator,
        fidelity=Fidelity(kind=kind, method=method, references=list(refs), assumptions=list(assumptions)))  # type: ignore[arg-type]


class SpurGearPairCEM(CEM):
    meta = CEMMeta(
        name="gear.spur_pair", version="0.1.0", domain="power_transmission",
        summary="External involute spur gear pair (20 deg full depth): sizing from ratio/module/centre distance "
                "or Lewis-based module selection; undercut, interference, contact-ratio and preliminary Lewis checks.",
        kind="cem", references=(F.REF_KHK, F.REF_SHI), tags=("gear", "spur", "transmission"))
    Requirements = Requirements
    Parameters = Parameters

    # ------------------------------------------------------------------ derive
    def derive(self, req: Requirements) -> DeriveResult:  # type: ignore[override]
        assumptions = [f"pressure angle {ALPHA_DEG} deg, full-depth basic rack (addendum 1.0 m, dedendum 1.25 m, "
                       f"tip radius 0.300 m) [SHI]"]
        z_undercut = math.ceil(F.min_teeth_no_undercut(ALPHA) - 1e-9)
        z1_min = req.min_pinion_teeth or z_undercut
        modules = [req.module] if req.module else sorted(req.candidate_modules)
        best = None
        for m in modules:
            cand = self._teeth_for(req, m, z1_min)
            if cand is None:
                continue
            z1, z2, x1, x2 = cand
            face = req.face_width or 4.0 * math.pi * m
            p = Parameters(module=m, z1=z1, z2=z2, x1=x1, x2=x2, face_width=face,
                           bore_diameter_pinion=req.bore_diameter_pinion, bore_diameter_gear=req.bore_diameter_gear,
                           pinion_torque=req.pinion_torque, pinion_speed=req.pinion_speed,
                           allowable_bending_stress=req.allowable_bending_stress, manufacturing=req.manufacturing)
            fails = [r for r in self.check(p) if r.status == "fail"]
            if not fails:
                best = p
                break
            if req.module:
                best = p  # the user fixed the module: return it with its failures visible
                break
        if best is None:
            raise ValueError("no candidate module satisfies the requirements and all checks")
        if req.face_width is None:
            assumptions.append("face width = 4 p, the middle of Shigley's 3p-5p guideline [SHI §14-1]")
        if best.x1 or best.x2:
            assumptions.append("total profile shift from the KHK inverse calculation, split equally between "
                               "pinion and gear unless the pinion needs more to avoid undercut (designer choice)")
        if not req.module:
            assumptions.append("module = smallest candidate passing all checks (incl. Lewis when loads given)")
        derivs = [Derivation(id="z_min_no_undercut", target="pair", formula="2/sin^2(alpha)",
                             value=QuantityModel(value=F.min_teeth_no_undercut(ALPHA), unit="1"),
                             references=[F.REF_KHK + " Eq. 3.9", F.REF_SHI + " Eq. 13-13"])]
        return DeriveResult(parameters=best, derivations=derivs, assumptions=assumptions)

    def _teeth_for(self, req: Requirements, m: float, z1_min: int) -> tuple[int, int, float, float] | None:
        choices = []
        for z1 in range(z1_min, z1_min + 60):
            z2 = round(z1 * req.ratio)
            if z2 > 400 or z2 < z1:
                continue
            err = abs(z2 / z1 - req.ratio) / req.ratio
            if err > req.ratio_tolerance:
                continue
            if req.center_distance is None:
                choices.append((0.0, err, z1, z2, 0.0, 0.0))
                if len(choices) > 3:
                    break
                continue
            try:
                xs = F.shift_sum_for_centre_distance(m, z1, z2, ALPHA, req.center_distance)
            except ValueError:
                continue
            if SHIFT_RANGE[0] <= xs <= SHIFT_RANGE[1]:
                # split: equal halves, but never below the pinion's no-undercut shift [KHK Eq. 3.10]
                x1 = round(max(xs / 2.0, F.min_shift_no_undercut(z1, ALPHA)), 12)
                x2 = xs - x1
                if x2 < F.min_shift_no_undercut(z2, ALPHA):
                    continue
                choices.append((abs(xs), err, z1, z2, x1, x2))
        if not choices:
            return None
        choices.sort(key=lambda c: (c[0] > 1e-12, c[1], c[0], c[2]))
        _, _, z1, z2, x1, x2 = choices[0]
        return z1, z2, x1, x2

    # ------------------------------------------------------------------ analytic checks
    def pair_state(self, p: Parameters) -> dict[str, float]:
        m, z1, z2, x1, x2 = p.module, p.z1, p.z2, p.x1, p.x2
        aw, it = F.working_pressure_angle(z1, z2, x1, x2, ALPHA)
        a_w = F.centre_distance_working(m, z1, z2, ALPHA, aw)
        return {"alpha_w": aw, "alpha_w_iterations": it, "a_w": a_w, "ratio": z2 / z1,
                "eps": F.transverse_contact_ratio(m, z1, z2, x1, x2, ALPHA),
                "ra1": F.tip_diameter(m, z1, x1) / 2, "ra2": F.tip_diameter(m, z2, x2) / 2,
                "rb1": F.base_diameter(m, z1, ALPHA) / 2, "rb2": F.base_diameter(m, z2, ALPHA) / 2}

    def check(self, p: Parameters) -> list[ValidationResult]:  # type: ignore[override]
        out: list[ValidationResult] = []
        m = p.module
        st = self.pair_state(p)
        refs_u = (F.REF_KHK + " Eq. 3.10",)
        undercut = False
        for name, z, x in (("pinion", p.z1, p.x1), ("gear", p.z2, p.x2)):
            xmin = F.min_shift_no_undercut(z, ALPHA)
            ok = x >= xmin - 1e-12
            undercut |= not ok
            out.append(_r("pass" if ok else "fail", f"pair.{name}", f"undercut: x={x:.4f}, x_min={xmin:.4f} for z={z}"
                          + ("" if ok else " (undercut; undercut gears are not generated by this CEM)"),
                          method="rack_cutter_undercut", measured=(x, "1"), limit=(xmin, "1"), comparator=">=",
                          refs=refs_u))
        for name, zt, zo, xt, xo in (("pinion_tip", p.z1, p.z2, p.x1, p.x2), ("gear_tip", p.z2, p.z1, p.x2, p.x1)):
            mg = F.involute_interference_margin(m, zt, zo, xt, xo, ALPHA)
            out.append(_r("pass" if mg >= 0 else "fail", f"pair.{name}",
                          f"involute interference margin {mg:.4f} mm along the line of action",
                          method="line_of_action_interference", measured=(mg, "mm"), limit=(0.0, "mm"),
                          comparator=">=", refs=("derived from line-of-action geometry; consistent with "
                                                 + F.REF_SHI + " Eqs. 13-11/13-12",)))
        eps = st["eps"]
        if eps < 1.0:
            s, msg = "fail", "contact ratio below 1: continuous action is lost"
        elif eps < 1.5:
            s, msg = "warn", "contact ratio below ~1.5 suggested for a quality gearset"
        else:
            s, msg = "pass", "contact ratio adequate"
        out.append(_r(s, "pair", f"transverse contact ratio {eps:.4f}: {msg}", method="transverse_contact_ratio",
                      measured=(eps, "1"), limit=(1.0, "1"), comparator=">=",
                      refs=(F.REF_KHK + " (transverse contact ratio; keep >= 1)", F.REF_SHI + " §14-1 (~1.5)")))
        for name, z, x in (("pinion", p.z1, p.x1), ("gear", p.z2, p.x2)):
            ra = F.tip_diameter(m, z, x) / 2
            sa = F.tooth_thickness_at(ra, m, z, x, ALPHA, p.backlash_allowance)
            out.append(_r("pass" if sa > 0 else "fail", f"pair.{name}", f"tip land thickness {sa:.4f} mm",
                          method="involute_tooth_thickness", measured=(sa, "mm"), limit=(0.0, "mm"), comparator=">",
                          refs=("involute tooth-thickness relation (" + F.REF_KHK + ")",)))
        c1, c2 = F.root_clearance(m, p.z1, p.z2, p.x1, p.x2, ALPHA)
        for name, c in (("pinion_tip", c1), ("gear_tip", c2)):
            out.append(_r("pass" if c > 0 else "fail", f"pair.{name}", f"tip-to-root radial clearance {c:.4f} mm",
                          method="root_clearance", measured=(c, "mm"), limit=(0.0, "mm"), comparator=">",
                          assumptions=("no tip shortening",)))
        if not undercut:
            out.extend(self._fillet_contact(p, st))
        lo, hi = F.face_width_guideline(m)
        ok = lo <= p.face_width <= hi
        out.append(_r("pass" if ok else "warn", "pair", f"face width {p.face_width:.3f} mm vs 3p..5p "
                      f"[{lo:.3f}, {hi:.3f}] mm", method="face_width_guideline", measured=(p.face_width, "mm"),
                      refs=(F.REF_SHI + " §14-1",)))
        out.extend(self._lewis(p))
        out.append(_r("not_evaluated", "pair", "AGMA 2001-D04 bending and pitting resistance rating not implemented",
                      method="agma_2001_d04"))
        return out

    def _fillet_contact(self, p: Parameters, st: dict[str, float]) -> list[ValidationResult]:
        """The mating tip must not engage below the form circle (into the trochoid fillet) [GEOM]."""
        res = []
        ln = st["a_w"] * math.sin(st["alpha_w"])
        for name, z, x, rb, ra_o, rb_o in (("pinion", p.z1, p.x1, st["rb1"], st["ra2"], st["rb2"]),
                                          ("gear", p.z2, p.x2, st["rb2"], st["ra1"], st["rb1"])):
            try:
                r_form = generate_flank(p.module, z, x, ALPHA, 4, 4).r_form
            except ValueError as e:
                res.append(_r("fail", f"pair.{name}", f"tooth profile cannot be generated: {e}",
                              method="rack_generation"))
                continue
            t = ln - math.sqrt(ra_o**2 - rb_o**2)
            r_low = math.sqrt(rb**2 + max(t, 0.0) ** 2)
            ok = t >= 0 and r_low >= r_form - 1e-9
            res.append(_r("pass" if ok else "fail", f"pair.{name}",
                          f"lowest contact radius {r_low:.4f} mm vs form radius {r_form:.4f} mm",
                          method="fillet_contact", measured=(r_low, "mm"), limit=(r_form, "mm"), comparator=">=",
                          refs=("line-of-action geometry; form radius from simulated rack generation",)))
        return res

    def _lewis(self, p: Parameters) -> list[ValidationResult]:
        target = "pair"
        method = "lewis_barth_preliminary"
        refs = (F.REF_SHI + " Eqs. 14-6, 14-8, Table 14-2",)
        assumptions = ("load at the tip, one tooth carries the load, no stress concentration (Lewis)",
                       "preliminary estimate only; not an AGMA/ISO rating")
        if p.pinion_torque is None or p.pinion_speed is None:
            return [_r("not_evaluated", target, "Lewis bending stress: pinion torque and speed not given",
                       method=method, refs=refs)]
        if p.x1 != 0 or p.x2 != 0:
            return [_r("not_evaluated", target, "Lewis bending stress: Table 14-2 Y applies to standard (x = 0) "
                       "teeth only", method=method, refs=refs)]
        d1 = p.module * p.z1
        v = F.pitch_line_velocity(d1, p.pinion_speed)
        kv = F.VELOCITY_FACTORS[p.manufacturing](v)
        wt = F.tangential_load(p.pinion_torque, d1)
        out = []
        for name, z in (("pinion", p.z1), ("gear", p.z2)):
            try:
                y, interp = F.lewis_form_factor(z)
            except ValueError as e:
                out.append(_r("not_evaluated", f"pair.{name}", f"Lewis: {e}", method=method, refs=refs))
                continue
            sigma = F.lewis_bending_stress(wt, p.face_width, p.module, y, kv)
            a = assumptions + ((f"Y interpolated for z={z}",) if interp else ()) + (
                f"K_v={kv:.4f} ({p.manufacturing}), V={v:.4f} m/s, W_t={wt:.3f} N",)
            if p.allowable_bending_stress is None:
                out.append(_r("not_evaluated", f"pair.{name}", f"Lewis stress {sigma:.3f} MPa; no allowable given",
                              method=method, measured=(sigma, "MPa"), refs=refs, assumptions=a))
            else:
                ok = sigma <= p.allowable_bending_stress
                out.append(_r("pass" if ok else "fail", f"pair.{name}", f"Lewis bending stress {sigma:.3f} MPa",
                              method=method, measured=(sigma, "MPa"), limit=(p.allowable_bending_stress, "MPa"),
                              comparator="<=", refs=refs, assumptions=a))
        return out

    def quantities(self, p: Parameters) -> dict[str, Quantity]:  # type: ignore[override]
        st = self.pair_state(p)
        q = {"ratio": Quantity.of(st["ratio"]), "center_distance": Quantity.of(st["a_w"], "mm"),
             "contact_ratio": Quantity.of(st["eps"]), "module": Quantity.of(p.module, "mm"),
             "pinion_teeth": Quantity.of(p.z1), "gear_teeth": Quantity.of(p.z2),
             "face_width": Quantity.of(p.face_width, "mm"),
             "working_pressure_angle": Quantity.of(st["alpha_w"], "rad"),
             "pinion_tip_diameter": Quantity.of(2 * st["ra1"], "mm"),
             "gear_tip_diameter": Quantity.of(2 * st["ra2"], "mm")}
        for r in self._lewis(p):
            if r.measured is not None:
                q[f"{r.target.split('.')[-1]}_bending_stress"] = r.measured.q()
        return q

    # ------------------------------------------------------------------ geometry
    def expand(self, p: Parameters) -> ComponentGeometry:  # type: ignore[override]
        st = self.pair_state(p)
        m, a_w = p.module, st["a_w"]
        # pinion: a tooth centred on +X (towards the gear); gear: a space centred on -X (towards the pinion)
        o1 = gear_outline_segments(m, p.z1, p.x1, ALPHA, p.backlash_allowance, -(math.pi / 2 - math.pi / p.z1))
        o2 = gear_outline_segments(m, p.z2, p.x2, ALPHA, p.backlash_allowance, math.pi / 2)
        ops: list[dict] = [
            {"id": "pinion_profile", "op": "profile", "start": o1["start"], "segments": o1["segments"],
             "tags": ["tooth_flank", "root_fillet"]},
            {"id": "pinion_blank", "op": "extrude", "profile": "pinion_profile", "height": p.face_width},
            {"id": "gear_profile", "op": "profile", "start": o2["start"], "segments": o2["segments"],
             "tags": ["tooth_flank", "root_fillet"]},
            {"id": "gear_blank", "op": "extrude", "profile": "gear_profile", "height": p.face_width},
        ]
        pin, gear = "pinion_blank", "gear_blank"
        for nm, bore, rf in (("pinion", p.bore_diameter_pinion, F.root_diameter(m, p.z1, p.x1) / 2),
                             ("gear", p.bore_diameter_gear, F.root_diameter(m, p.z2, p.x2) / 2)):
            if bore:
                if bore / 2 >= rf:
                    raise ValueError(f"{nm} bore exceeds root circle")
                ops += [{"id": f"{nm}_bore_tool", "op": "cylinder", "radius": bore / 2,
                         "height": p.face_width + 2.0, "tags": ["shaft_bore"]},
                        {"id": f"{nm}_bore_tool_t", "op": "translate", "input": f"{nm}_bore_tool",
                         "offset": [0.0, 0.0, -1.0]},
                        {"id": f"{nm}_body", "op": "difference",
                         "base": pin if nm == "pinion" else gear, "tools": [f"{nm}_bore_tool_t"]}]
                if nm == "pinion":
                    pin = "pinion_body"
                else:
                    gear = "gear_body"
        ops.append({"id": "gear_placed", "op": "translate", "input": gear, "offset": [a_w, 0.0, 0.0],
                    "tags": ["mesh_position"]})
        graph = GeometryGraph.model_validate({"ops": ops, "outputs": {"pinion": pin, "gear": "gear_placed"}})
        ports = [Port(id="pinion_shaft", kind="shaft_bore", frame=Frame()),
                 Port(id="gear_shaft", kind="shaft_bore", frame=Frame(origin=[a_w, 0.0, 0.0]))]
        rel = Relation(id="mesh", kind="mesh", a="pair.pinion_shaft", b="pair.gear_shaft",
                       params={"center_distance": QuantityModel(value=a_w, unit="mm"),
                               "working_pressure_angle": QuantityModel(value=math.degrees(st["alpha_w"]), unit="deg"),
                               "ratio": st["ratio"]})
        return ComponentGeometry(graph=graph, bodies={}, ports=ports, relations=[rel], notes=[
            "flanks: interpolating B-splines through exact generated involute/trochoid points"])

    def geometry_validators(self):
        return [self._mesh_interference]

    def _mesh_interference(self, ctx: GeometryContext) -> list[ValidationResult]:
        b = ctx.backend
        v = b.intersection_volume(ctx.bodies["pinion"], ctx.bodies["gear"])
        ref = max(1e-9, min(ctx.measurements["pinion"].volume_mm3, ctx.measurements["gear"].volume_mm3))
        tol_rel = 1e-6 if b.representation == "brep" else 1e-3
        ok = v <= tol_rel * ref
        return [ValidationResult(
            validator=VALIDATOR, level=4, status="pass" if ok else "fail", target="pair.mesh",
            message=f"pinion/gear overlap volume {v:.6g} mm^3 at the working centre distance (one meshing position)",
            measured=QuantityModel(value=v, unit="mm^3"), limit=QuantityModel(value=tol_rel * ref, unit="mm^3"),
            comparator="<=", evidence=["pinion", "gear"],
            fidelity=Fidelity(kind="geometric", method=f"boolean_common@{b.name}",
                              assumptions=["checked at a single rotational position"]))]

    def analyses(self) -> dict[str, str]:
        return {"geometry_checks": "analytic", "lewis_bending": "analytic", "agma_rating": "unsupported",
                "contact_stress": "unsupported", "dynamic_load": "solver_required"}

    # ------------------------------------------------------------------ sampling (dataset factory)
    def sample(self, rng: np.random.Generator, mode: str) -> Parameters:  # type: ignore[override]
        for _ in range(500):
            p = self._draw(rng, mode)
            if p is None:
                continue
            statuses = [r.status for r in self.check(p)]
            if mode in ("valid", "boundary") and "fail" not in statuses:
                return p
            if mode == "invalid" and "fail" in statuses:
                return p
        raise RuntimeError(f"could not sample a {mode} spur gear pair")

    def _draw(self, rng: np.random.Generator, mode: str) -> Parameters | None:
        m = float(rng.choice(DEFAULT_CANDIDATE_MODULES[2:9]))
        load = rng.random() < 0.5
        kw: dict[str, Any] = {}
        if load:
            kw = {"pinion_torque": float(np.round(rng.uniform(2e3, 2e5), 1)),
                  "pinion_speed": float(rng.choice([300, 600, 900, 1200, 1800, 3000])),
                  "allowable_bending_stress": float(rng.choice([60, 90, 120, 160, 200]))}
        if mode == "valid":
            z1 = int(rng.integers(18, 41))
            z2 = int(min(400, max(z1, round(z1 * rng.uniform(1.0, 5.0)))))
            x1 = x2 = 0.0
            if rng.random() < 0.3:
                x1 = float(np.round(rng.uniform(0.0, 0.5), 3))
                x2 = float(np.round(rng.uniform(-0.2, 0.3), 3))
        elif mode == "boundary":
            z1 = int(rng.integers(12, 18))
            x1 = float(np.round(F.min_shift_no_undercut(z1, ALPHA) + rng.uniform(0.0, 0.02), 4))
            z2 = int(rng.integers(max(z1, 18), 80))
            x2 = 0.0
        else:
            kind = rng.integers(0, 3)
            if kind == 0:   # undercut pinion
                z1, x1 = int(rng.integers(8, 17)), 0.0
                z2, x2 = int(rng.integers(20, 90)), 0.0
            elif kind == 1:  # pointed teeth from excessive shift
                z1, x1 = int(rng.integers(10, 16)), float(np.round(rng.uniform(1.0, 1.4), 3))
                z2, x2 = int(rng.integers(20, 60)), 0.0
            else:            # overloaded (Lewis) with valid geometry
                z1, x1, z2, x2 = 18, 0.0, 36, 0.0
                kw = {"pinion_torque": 5e5, "pinion_speed": 1200.0, "allowable_bending_stress": 60.0}
        face = float(np.round(4.0 * math.pi * m * rng.uniform(0.8, 1.2), 3))
        try:
            return Parameters(module=m, z1=z1, z2=z2, x1=x1, x2=x2, face_width=face, **kw)
        except ValueError:
            return None
