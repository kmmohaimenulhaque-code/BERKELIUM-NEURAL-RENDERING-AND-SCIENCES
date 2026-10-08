"""ModelEnv benchmark: scenarios with INDEPENDENT reference models (explicit relation sets authored per scenario)
and explicit expected statuses for the non-determined variants. Domain split: TRAIN = structures, heat, fluids;
HELD-OUT = vessel, gas dynamics, rocket (cross-domain composition). Seeded, reproducible."""

from __future__ import annotations

import numpy as np

from ..laws import LawSet, solve
from ..synthesis.library import FRAGMENTS_SRC, VARS
from ..units import Quantity
from .model_env import ModelTask

TRAIN_DOMAINS = {"structures", "heat", "fluids"}
CANT = {"support:cantilever", "load:end", "section:rect"}
SS = {"support:simply_supported", "load:mid", "section:rect"}
VES = {"vessel:cylinder", "load:internal_pressure", "end:plane_strain"}
PIPE = {"flow:pipe", "flow:fully_developed"}
ROCKET = {"flow:isentropic", "throat:choked", "nozzle:matched_expansion", "section:circular", "system:rocket_chamber",
          "vessel:cylinder", "load:internal_pressure", "end:plane_strain"}
_REL = {f.relation.id: f.relation for f in FRAGMENTS_SRC}


def _q(v, u):
    return Quantity.of(float(v), u)


def _scen(rng):
    u = rng.uniform
    P, L, E = u(50, 500), u(80, 300), u(70, 210)
    h = L / u(6, 20)
    b = h / u(1, 3)
    d = {}
    d["beam_cantilever_deflection"] = ("structures", CANT, {"P": _q(P, "N"), "L": _q(L, "mm"), "E": _q(E, "GPa"),
                                       "b": _q(b, "mm"), "h": _q(h, "mm")}, "delta",
                                       ["cantilever_end_load_deflection", "section_inertia_rect"], None)
    d["beam_simply_supported_deflection"] = ("structures", SS, dict(d["beam_cantilever_deflection"][2]), "delta",
                                             ["simply_supported_mid_load_deflection", "section_inertia_rect"], None)
    sig = 6 * P * L / (b * h * h)     # MPa: consistent with a valid (L/h in 6..20) section by construction
    d["beam_size_height_from_stress"] = ("structures", CANT, {"P": _q(P, "N"), "L": _q(L, "mm"), "sigma": _q(sig, "MPa"),
                                         "b": _q(b, "mm")}, "h", ["cantilever_root_stress_rect"], None)
    d["beam_underdetermined"] = ("structures", CANT, {"P": _q(P, "N"), "L": _q(L, "mm"), "E": _q(E, "GPa"),
                                 "b": _q(b, "mm")}, "delta", None, "underdetermined")
    rho = u(2700, 7900)
    d["beam_contradictory_mass"] = ("structures", CANT, {**d["beam_cantilever_deflection"][2], "rho": _q(rho, "kg/m^3"),
                                    "mass": _q(rho * L * b * h * 1e-9 * 1.2, "kg")}, "delta", None, "contradictory")
    Ti, Ta, k, ho = u(400, 600), u(280, 310), u(0.03, 0.08), u(5, 25)
    Ts = Ta + u(10, 30)
    d["heat_insulation_thickness"] = ("heat", {"heat:plane_wall_1d", "outer:convection"},
                                      {"T_in": _q(Ti, "K"), "T_amb": _q(Ta, "K"), "T_s": _q(Ts, "K"),
                                       "k_ins": _q(k, "W/(m*K)"), "h_o": _q(ho, "W/(m^2*K)")}, "t_ins",
                                      ["plane_wall_series", "convective_surface"], None)
    d["heat_underdetermined"] = ("heat", {"heat:plane_wall_1d", "outer:convection"},
                                 {"T_in": _q(Ti, "K"), "T_amb": _q(Ta, "K"), "T_s": _q(Ts, "K"), "k_ins": _q(k, "W/(m*K)")},
                                 "t_ins", None, "underdetermined")
    rf, mu, D, Lp = 998.0, 1.002e-3, u(20, 100), u(2, 50)
    base = {"rho_f": _q(rf, "kg/m^3"), "mu": _q(mu, "Pa*s"), "D": _q(D, "mm"), "Lp": _q(Lp, "m"), "rough": _q(u(0, 0.05), "mm")}
    vel = lambda re: re * mu / (rf * D * 1e-3)  # noqa: E731
    d["pipe_laminar_dp"] = ("fluids", PIPE, {**base, "Vf": _q(vel(u(300, 2000)), "m/s")}, "dp",
                            ["darcy_weisbach", "laminar_friction", "reynolds"], None)
    d["pipe_turbulent_dp"] = ("fluids", PIPE, {**base, "Vf": _q(vel(u(5e3, 2e5)), "m/s")}, "dp",
                              ["darcy_weisbach", "colebrook", "reynolds"], None)
    d["pipe_transitional_dp"] = ("fluids", PIPE, {**base, "Vf": _q(vel(u(2500, 3800)), "m/s")}, "dp", None,
                                 "outside_validity")
    p, ri = u(2, 40), u(10, 100)
    ro = ri * u(1.3, 2.5)
    nu = u(0.25, 0.35)
    d["vessel_thick_von_mises"] = ("vessel", VES, {"p": _q(p, "MPa"), "ri": _q(ri, "mm"), "ro": _q(ro, "mm"), "nu": _q(nu, "1")},
                                   "svm", ["lame_hoop_inner", "plane_strain_axial", "radial_stress_inner",
                                           "von_mises_principal"], None)
    d["vessel_contradictory_wall"] = ("vessel", VES, {"p": _q(p, "MPa"), "ri": _q(ri, "mm"), "ro": _q(ro, "mm"),
                                      "w": _q((ro - ri) * 1.25, "mm")}, "sig_t", None, "contradictory")
    d["vessel_thin_hoop"] = ("vessel", VES, {"p": _q(p, "MPa"), "ri": _q(ri, "mm"), "ro": _q(ri * u(1.01, 1.05), "mm")},
                             "sig_t", ["lame_hoop_inner"], None)
    gam = u(1.15, 1.4)
    d["nozzle_exit_mach"] = ("gas_dynamics", {"flow:isentropic"}, {"gamma": _q(gam, "1"), "p0": _q(u(1, 10), "MPa"),
                             "pe": _q(u(50, 150), "kPa")}, "M_e", ["isentropic_pressure_ratio"], None)
    d["nozzle_mach_from_area_ratio"] = ("gas_dynamics", {"flow:isentropic"}, {"gamma": _q(gam, "1"),
                                        "eps": _q(u(1.5, 8), "1")}, "M_e", None, "ambiguous")
    d["nozzle_thrust"] = ("gas_dynamics", {"flow:isentropic", "throat:choked", "nozzle:matched_expansion"},
                          {"gamma": _q(gam, "1"), "R": _q(u(280, 400), "J/(kg*K)"), "T0": _q(u(2500, 3500), "K"),
                           "p0": _q(u(1, 10), "MPa"), "pe": _q(101.325, "kPa"), "At": _q(u(50, 500), "mm^2")}, "F",
                          ["choked_mass_flow", "thrust_matched", "exit_velocity", "isentropic_temperature",
                           "isentropic_pressure_ratio"], None)
    d["rocket_chamber_wall_stress"] = ("rocket", ROCKET, {"gamma": _q(gam, "1"), "R": _q(u(280, 400), "J/(kg*K)"),
                                       "T0": _q(u(2500, 3500), "K"), "p0": _q(u(1, 5), "MPa"), "pe": _q(101.325, "kPa"),
                                       "F": _q(u(200, 5000), "N"), "CR": _q(u(3, 6), "1"), "nu": _q(0.3, "1"),
                                       "ro": _q(u(60, 120), "mm")}, "svm",
                                      ["chamber_is_vessel", "choked_mass_flow", "contraction_ratio", "exit_velocity",
                                       "isentropic_pressure_ratio", "isentropic_temperature", "lame_hoop_inner",
                                       "plane_strain_axial", "radial_stress_inner", "throat_diameter",
                                       "thrust_matched", "von_mises_principal"], None)
    return d


def tasks(n_per: int = 4, seed: int = 0) -> list[ModelTask]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_per):
        for name, (dom, ctx, known, target, ref, status) in _scen(rng).items():
            if ref is not None:
                rels = [_REL[r] for r in ref]
                names = set().union(*(r.vars for r in rels)) | set(known)
                s = solve(LawSet("reference", {n: VARS[n] for n in names}, rels), known)
                if target not in s.values or s.status != "solved":
                    raise RuntimeError(f"reference for {name} not solvable: {s.status} {s.diagnostics}")
                st, val = "determined", s.values[target].to(VARS[target].unit)
            else:
                st, val = status, None
            out.append(ModelTask(f"{name}#{i}", dom, frozenset(ctx), known, target, st, val,
                                 "train" if dom in TRAIN_DOMAINS else "heldout"))
    return out
