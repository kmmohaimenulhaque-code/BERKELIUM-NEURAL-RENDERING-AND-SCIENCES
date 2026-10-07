"""Deterministic natural-language intent rendering from requirement dicts (template paraphrases).
Teacher-model paraphrases can be added later; they are kept only if the target still verifies (§10.1)."""

from __future__ import annotations

import numpy as np

OPENERS = ["Design a spur gear pair", "I need a pair of spur gears", "Create a spur gear reduction",
           "Make two meshing spur gears", "Generate a spur gearset", "Please size a spur gear pair"]


def gear_intent(req: dict, rng: np.random.Generator) -> str:
    parts = [str(rng.choice(OPENERS))]
    ratio = req["ratio"]
    parts.append(str(rng.choice([f"with a {ratio:g}:1 reduction", f"for a gear ratio of {ratio:g}",
                                 f"reducing speed by a factor of {ratio:g}"])))
    if "module" in req:
        parts.append(str(rng.choice([f"module {req['module']['value']:g} mm", f"using a {req['module']['value']:g} mm module"])))
    if "center_distance" in req:
        parts.append(f"at {req['center_distance']['value']:g} mm centre distance")
    if "min_pinion_teeth" in req:
        parts.append(f"allowing a pinion as small as {req['min_pinion_teeth']} teeth")
    if "pinion_torque" in req:
        parts.append(f"transmitting {req['pinion_torque']['value'] / 1000:g} N·m at "
                     f"{req['pinion_speed']['value']:g} rpm on the pinion")
        parts.append(f"with an allowable bending stress of {req['allowable_bending_stress']['value']:g} MPa")
    if "face_width" in req:
        parts.append(f"{req['face_width']['value']:g} mm face width")
    if "bore_diameter_pinion" in req:
        parts.append(f"a {req['bore_diameter_pinion']['value']:g} mm pinion bore")
    head, rest = parts[0], parts[1:]
    order = rng.permutation(len(rest))
    return head + " " + ", ".join(rest[i] for i in order) + "."
