"""Small, SOURCED material library. Values are nominal handbook values for a material CLASS, not a grade
certificate; a design that matters must use the supplier's data. Every entry states its source."""

from __future__ import annotations

from .schema import Material

SHIGLEY_A5 = "Budynas & Nisbett, Shigley's Mechanical Engineering Design, Table A-5 (nominal, room temperature)"
INCROPERA_A1 = "Bergman, Lavine, Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, Table A.1 (300 K)"

LIBRARY: dict[str, Material] = {m.id: m for m in [
    Material(id="carbon_steel", name="Carbon steel (class)", source=SHIGLEY_A5,
             density={"value": 76.5e3 / 9.80665, "unit": "kg/m^3"},
             elastic={"youngs_modulus": {"value": 207.0, "unit": "GPa"}, "poisson_ratio": 0.292},
             notes=["density from tabulated unit weight 76.5 kN/m^3 / g0"]),
    Material(id="aluminum_alloy", name="Aluminum alloys (class)", source=SHIGLEY_A5,
             density={"value": 26.6e3 / 9.80665, "unit": "kg/m^3"},
             elastic={"youngs_modulus": {"value": 71.7, "unit": "GPa"}, "poisson_ratio": 0.333},
             notes=["density from tabulated unit weight 26.6 kN/m^3 / g0"]),
    Material(id="aisi_1010", name="Plain carbon steel AISI 1010", source=INCROPERA_A1,
             density={"value": 7832.0, "unit": "kg/m^3"},
             thermal={"conductivity": {"value": 63.9, "unit": "W/(m*K)"}}),
    Material(id="aluminum_pure", name="Aluminum, pure", source=INCROPERA_A1,
             density={"value": 2702.0, "unit": "kg/m^3"},
             thermal={"conductivity": {"value": 237.0, "unit": "W/(m*K)"}}),
]}
