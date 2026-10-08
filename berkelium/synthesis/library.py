"""The relation library the agent constructs models FROM (ADR-023).

A Fragment is a relation plus the CONTEXT it presupposes (support conditions, flow regime, geometry class...).
Prior art: this is the 'model fragment + assumption' idea of compositional modelling (Falkenhainer & Forbus,
"Compositional modeling: finding the right model for the job", Artificial Intelligence 51, 1991). Berkelium adds:
dimension checking at load, acausal solving, validity domains checked at the solution, model-form bounds, and
competition between alternative models resolved by evidence.

The library deliberately contains HARD NEGATIVES:
  * plausible-but-wrong-context fragments (simply-supported deflection offered for a cantilever problem)
  * fragments valid only in a sub-domain (thin-wall hoop stress; laminar friction factor)
  * a dimensionally invalid fragment, which the loader must quarantine (never usable)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..laws import LawError, LawSet, Relation, Var
from ..laws.library import GERE, HEAT, ISEN, LAME

COLE = "Colebrook (1939)"


@dataclass(frozen=True)
class Fragment:
    relation: Relation
    domain: str
    context: frozenset[str] = frozenset()     # facts the problem must assert for this fragment to apply
    note: str = ""


def V(n, u="1", lo=None, hi=None):
    return Var(n, u, lo, hi)


VARS: dict[str, Var] = {v.name: v for v in [
    # structures
    V("P", "N", 0), V("L", "mm", 0), V("E", "GPa", 0), V("b", "mm", 0), V("h", "mm", 0), V("I", "mm^4", 0),
    V("delta", "mm", 0), V("sigma", "MPa", 0), V("rho", "kg/m^3", 0), V("mass", "kg", 0),
    # pressure vessel
    V("p", "MPa", 0), V("ri", "mm", 0), V("ro", "mm", 0), V("nu", "1", 0, 0.5), V("sig_t", "MPa"),
    V("sig_r", "MPa"), V("sig_z", "MPa"), V("svm", "MPa", 0), V("w", "mm", 0),
    # gas dynamics
    V("gamma", "1", 1.0001, 2), V("M_e", "1", 0), V("p0", "MPa", 0), V("pe", "kPa", 0), V("T0", "K", 0),
    V("R", "J/(kg*K)", 0), V("mdot", "kg/s", 0), V("At", "mm^2", 0), V("Ae", "mm^2", 0), V("eps", "1", 0),
    V("d_t", "mm", 0), V("d_e", "mm", 0), V("Te", "K", 0), V("ve", "m/s", 0), V("F", "N", 0), V("CR", "1", 1),
    # heat
    V("T_in", "K", 0), V("T_amb", "K", 0), V("T_s", "K", 0), V("q", "W/m^2"), V("t_ins", "mm", 0),
    V("k_ins", "W/(m*K)", 0), V("h_o", "W/(m^2*K)", 0),
    # pipe flow
    V("rho_f", "kg/m^3", 0), V("mu", "Pa*s", 0), V("D", "mm", 0), V("Lp", "m", 0), V("Vf", "m/s", 0),
    V("Qf", "m^3/s", 0), V("Re", "1", 0), V("f_D", "1", 0), V("dp", "Pa", 0), V("rough", "mm", 0)]}


def _R(i, t, validity=(), fid="analytic", mf=0.0, refs=(), ass=()):
    return Relation(i, t, tuple(validity), fid, mf, tuple(refs), tuple(ass))


g = "gamma"
FRAGMENTS_SRC: list[Fragment] = [
    # ---- structures
    Fragment(_R("section_inertia_rect", "I == b*h^3/12", fid="definition"), "structures", frozenset({"section:rect"})),
    Fragment(_R("cantilever_end_load_deflection", "delta == P*L^3/(3*E*I)", ["L/h >= 5"], refs=[GERE]),
             "structures", frozenset({"support:cantilever", "load:end"})),
    Fragment(_R("simply_supported_mid_load_deflection", "delta == P*L^3/(48*E*I)", ["L/h >= 5"], refs=[GERE]),
             "structures", frozenset({"support:simply_supported", "load:mid"}), "hard negative for cantilevers"),
    Fragment(_R("cantilever_root_stress_rect", "sigma == 6*P*L/(b*h^2)", ["L/h >= 5"], refs=[GERE]), "structures",
             frozenset({"support:cantilever", "load:end", "section:rect"})),
    Fragment(_R("simply_supported_mid_stress_rect", "sigma == 3*P*L/(2*b*h^2)", ["L/h >= 5"], refs=[GERE]),
             "structures", frozenset({"support:simply_supported", "load:mid", "section:rect"})),
    Fragment(_R("prism_mass", "mass == rho*L*b*h", fid="definition"), "structures", frozenset({"section:rect"})),
    Fragment(_R("BROKEN_deflection_units", "delta == P*L^2/(3*E*I)"), "structures",
             frozenset({"support:cantilever", "load:end"}), "dimensionally invalid: must be quarantined"),
    # ---- pressure vessel
    Fragment(_R("lame_hoop_inner", "sig_t == p*(ro^2 + ri^2)/(ro^2 - ri^2)", ["ro > ri"], refs=[LAME]), "vessel",
             frozenset({"vessel:cylinder", "load:internal_pressure"})),
    Fragment(_R("thin_wall_hoop", "sig_t == p*ri/(ro - ri)", ["(ro - ri)/ri <= 0.1"], fid="analytic", mf=0.05,
                refs=["thin-wall cylinder (membrane) approximation"]), "vessel",
             frozenset({"vessel:cylinder", "load:internal_pressure"}), "valid only for thin walls"),
    Fragment(_R("radial_stress_inner", "sig_r == -p", fid="exact"), "vessel",
             frozenset({"vessel:cylinder", "load:internal_pressure"})),
    Fragment(_R("plane_strain_axial", "sig_z == nu*(sig_r + sig_t)", refs=[LAME]), "vessel",
             frozenset({"vessel:cylinder", "end:plane_strain"})),
    Fragment(_R("von_mises_principal", "svm == sqrt(0.5*((sig_r - sig_t)^2 + (sig_t - sig_z)^2 + (sig_z - sig_r)^2))",
                fid="definition"), "mechanics", frozenset()),
    Fragment(_R("wall_thickness", "w == ro - ri", fid="definition"), "vessel", frozenset({"vessel:cylinder"})),
    # ---- gas dynamics
    Fragment(_R("isentropic_pressure_ratio", f"pe/p0 == (1 + ({g} - 1)/2*M_e^2)^(-{g}/({g} - 1))", refs=[ISEN]),
             "gas_dynamics", frozenset({"flow:isentropic"})),
    Fragment(_R("isentropic_area_ratio",
                f"eps == (1/M_e)*((2/({g} + 1))*(1 + ({g} - 1)/2*M_e^2))^(({g} + 1)/(2*({g} - 1)))", refs=[ISEN]),
             "gas_dynamics", frozenset({"flow:isentropic"})),
    Fragment(_R("choked_mass_flow", f"mdot == At*p0/sqrt(R*T0)*sqrt({g})*(2/({g} + 1))^(({g} + 1)/(2*({g} - 1)))",
                refs=[ISEN]), "gas_dynamics", frozenset({"flow:isentropic", "throat:choked"})),
    Fragment(_R("exit_area", "Ae == eps*At", fid="definition"), "gas_dynamics", frozenset({"flow:isentropic"})),
    Fragment(_R("isentropic_temperature", f"Te == T0/(1 + ({g} - 1)/2*M_e^2)", refs=[ISEN]), "gas_dynamics",
             frozenset({"flow:isentropic"})),
    Fragment(_R("exit_velocity", f"ve == M_e*sqrt({g}*R*Te)", fid="exact"), "gas_dynamics", frozenset({"flow:isentropic"})),
    Fragment(_R("thrust_matched", "F == mdot*ve"), "gas_dynamics", frozenset({"nozzle:matched_expansion"})),
    Fragment(_R("throat_diameter", "At == pi*d_t^2/4", fid="definition"), "geometry", frozenset({"section:circular"})),
    Fragment(_R("exit_diameter", "Ae == pi*d_e^2/4", fid="definition"), "geometry", frozenset({"section:circular"})),
    Fragment(_R("chamber_is_vessel", "p == p0", fid="definition"), "interface", frozenset({"system:rocket_chamber"})),
    Fragment(_R("contraction_ratio", "(2*ri)^2 == CR*d_t^2", fid="definition"), "interface",
             frozenset({"system:rocket_chamber"})),
    # ---- heat transfer
    Fragment(_R("plane_wall_series", "q == (T_in - T_amb)/(t_ins/k_ins + 1/h_o)", refs=[HEAT]), "heat",
             frozenset({"heat:plane_wall_1d", "outer:convection"})),
    Fragment(_R("convective_surface", "T_s == T_amb + q/h_o", refs=[HEAT]), "heat", frozenset({"outer:convection"})),
    # ---- pipe flow
    Fragment(_R("reynolds", "Re == rho_f*Vf*D/mu", fid="definition"), "fluids", frozenset({"flow:pipe"})),
    Fragment(_R("continuity_pipe", "Qf == Vf*pi*D^2/4", fid="definition"), "fluids", frozenset({"flow:pipe"})),
    Fragment(_R("darcy_weisbach", "dp == f_D*Lp/D*rho_f*Vf^2/2", refs=["Darcy-Weisbach"]), "fluids",
             frozenset({"flow:pipe", "flow:fully_developed"})),
    Fragment(_R("laminar_friction", "f_D == 64/Re", ["Re <= 2300"], fid="exact"), "fluids", frozenset({"flow:pipe"})),
    Fragment(_R("colebrook", "1/sqrt(f_D) == -2*log(rough/D/3.7 + 2.51/(Re*sqrt(f_D)))/log(10)", ["Re >= 4000"],
                fid="empirical", mf=0.15, refs=[COLE]), "fluids", frozenset({"flow:pipe"})),
]


def load() -> tuple[list[Fragment], list[tuple[str, str]]]:
    """Dimension-check every fragment; quarantine the invalid ones (returned with the reason)."""
    ok, quarantined = [], []
    for f in FRAGMENTS_SRC:
        try:
            LawSet(f.relation.id, {n: VARS[n] for n in f.relation.vars}, [f.relation])
            ok.append(f)
        except (LawError, KeyError) as e:
            quarantined.append((f.relation.id, str(e)))
    return ok, quarantined
