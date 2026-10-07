"""Engineering knowledge as DATA: four domains written only as variables + relations (ADR-020).

No domain below contains a single line of procedural code. Sizing, checking, inverting, sensitivity,
uncertainty and optimisation all come from the domain-free machinery in ``laws.core``.
"""

from __future__ import annotations

from .core import LawSet, Relation, Var

GERE = "Gere & Goodno, Mechanics of Materials"
LAME = "Lame thick-walled cylinder (e.g. Timoshenko & Goodier, Theory of Elasticity)"
ISEN = "NACA Report 1135 (1953), Equations, Tables and Charts for Compressible Flow"
HEAT = "Bergman et al., Fundamentals of Heat and Mass Transfer (plane wall, thermal resistance)"


def V(name, unit="1", lo=None, hi=None, d=""):
    return Var(name, unit, lo, hi, d)


def beam() -> LawSet:
    return LawSet("cantilever_beam", {v.name: v for v in [
        V("P", "N", 0), V("L", "mm", 0), V("E", "GPa", 0), V("b", "mm", 0), V("h", "mm", 0), V("a", "1", 0),
        V("I", "mm^4", 0), V("delta", "mm", 0), V("sigma", "MPa", 0), V("rho", "kg/m^3", 0), V("mass", "kg", 0),
        V("delta_max", "mm", 0), V("sigma_allow", "MPa", 0)]}, [
        Relation("section_inertia", "I == b*h^3/12", fidelity="definition"),
        Relation("aspect", "h == a*b", fidelity="definition"),
        Relation("eb_deflection", "delta == P*L^3/(3*E*I)", ("L/h >= 5",), model_form_rel=None,
                 references=(GERE,), assumptions=("Euler-Bernoulli: shear deformation neglected",)),
        Relation("flexure", "sigma == 6*P*L/(b*h^2)", ("L/h >= 5",), model_form_rel=None, references=(GERE,)),
        Relation("mass_def", "mass == rho*L*b*h", fidelity="definition"),
        Relation("stiffness_req", "delta <= delta_max", fidelity="requirement"),
        Relation("strength_req", "sigma <= sigma_allow", fidelity="requirement"),
    ])


def vessel() -> LawSet:
    return LawSet("thick_cylinder", {v.name: v for v in [
        V("p", "MPa", 0), V("ri", "mm", 0), V("ro", "mm", 0), V("nu", "1", 0, 0.5), V("sig_t", "MPa"),
        V("sig_r", "MPa"), V("sig_z", "MPa"), V("svm", "MPa", 0), V("S_allow", "MPa", 0), V("n_sf", "1", 1),
        V("w", "mm", 0), V("rho_s", "kg/m^3", 0), V("mass_per_len", "kg/m", 0)]}, [
        Relation("lame_hoop_inner", "sig_t == p*(ro^2 + ri^2)/(ro^2 - ri^2)", ("ro > ri",), references=(LAME,),
                 assumptions=("linear elastic, axisymmetric, internal pressure only",)),
        Relation("radial_inner", "sig_r == -p", fidelity="exact"),
        Relation("plane_strain_axial", "sig_z == nu*(sig_r + sig_t)", references=(LAME,),
                 assumptions=("plane strain (long, axially restrained)",)),
        Relation("von_mises", "svm == sqrt(0.5*((sig_r - sig_t)^2 + (sig_t - sig_z)^2 + (sig_z - sig_r)^2))",
                 fidelity="definition"),
        Relation("wall", "w == ro - ri", fidelity="definition"),
        Relation("mass_def", "mass_per_len == rho_s*pi*(ro^2 - ri^2)", fidelity="definition"),
        Relation("strength_req", "svm*n_sf <= S_allow", fidelity="requirement"),
    ])


def nozzle() -> LawSet:
    g = "gamma"
    return LawSet("isentropic_nozzle", {v.name: v for v in [
        V("gamma", "1", 1.0001, 2), V("M_e", "1", 0), V("p0", "MPa", 0), V("pe", "kPa", 0), V("T0", "K", 0),
        V("R", "J/(kg*K)", 0), V("mdot", "kg/s", 0), V("At", "mm^2", 0), V("Ae", "mm^2", 0), V("eps", "1", 0),
        V("d_t", "mm", 0), V("d_e", "mm", 0), V("Te", "K", 0), V("ve", "m/s", 0), V("F", "N", 0)]}, [
        Relation("pressure_ratio", f"pe/p0 == (1 + ({g} - 1)/2*M_e^2)^(-{g}/({g} - 1))", references=(ISEN,),
                 assumptions=("calorically perfect gas, isentropic, quasi-1-D, frozen composition",)),
        Relation("area_ratio", f"eps == (1/M_e)*((2/({g} + 1))*(1 + ({g} - 1)/2*M_e^2))^(({g} + 1)/(2*({g} - 1)))",
                 references=(ISEN,)),
        Relation("choked_mass_flow", f"mdot == At*p0/sqrt(R*T0)*sqrt({g})*(2/({g} + 1))^(({g} + 1)/(2*({g} - 1)))",
                 references=(ISEN,), assumptions=("choked throat",)),
        Relation("exit_area", "Ae == eps*At", fidelity="definition"),
        Relation("exit_temperature", f"Te == T0/(1 + ({g} - 1)/2*M_e^2)", references=(ISEN,)),
        Relation("exit_velocity", f"ve == M_e*sqrt({g}*R*Te)", fidelity="exact"),
        Relation("thrust_matched", "F == mdot*ve", assumptions=("optimally expanded: pe = ambient",)),
        Relation("throat_diameter", "At == pi*d_t^2/4", fidelity="definition"),
        Relation("exit_diameter", "Ae == pi*d_e^2/4", fidelity="definition"),
    ])


def wall() -> LawSet:
    return LawSet("insulated_wall", {v.name: v for v in [
        V("T_in", "K", 0), V("T_amb", "K", 0), V("T_s", "K", 0), V("q", "W/m^2"), V("t_ins", "mm", 0),
        V("k_ins", "W/(m*K)", 0), V("h_o", "W/(m^2*K)", 0), V("T_s_max", "K", 0)]}, [
        Relation("series_resistance", "q == (T_in - T_amb)/(t_ins/k_ins + 1/h_o)", references=(HEAT,),
                 assumptions=("steady 1-D conduction, inner surface held at T_in, convection outside",)),
        Relation("surface_temperature", "T_s == T_amb + q/h_o", references=(HEAT,)),
        Relation("touch_safety_req", "T_s <= T_s_max", fidelity="requirement"),
    ])


def rocket_chamber() -> LawSet:
    """Cross-domain composition: the nozzle's chamber IS a pressure vessel. Only interface relations are
    new; everything else is reused verbatim."""
    link = LawSet("chamber_interface", {v.name: v for v in [
        V("p", "MPa", 0), V("p0", "MPa", 0), V("ri", "mm", 0), V("d_t", "mm", 0), V("CR", "1", 1)]}, [
        Relation("chamber_pressure", "p == p0", fidelity="definition"),
        Relation("contraction_ratio", "(2*ri)^2 == CR*d_t^2", fidelity="definition",
                 assumptions=("cylindrical chamber; CR = chamber/throat area ratio",)),
    ])
    return nozzle().compose(vessel(), link, name="rocket_chamber")
