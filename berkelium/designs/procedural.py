"""Creative procedural designs — no CEM, authored directly as Geometry IR with expression arguments.

These prove the second creation mechanism: geometry nobody catalogued, expressed as data that a model
can author, diff and patch, realised and validated by the same deterministic pipeline."""

from __future__ import annotations

from ..schema.design import DesignProposal


def lantern_proposal(height_mm: float = 120, radius_mm: float = 45, wall_mm: float = 3, slots: int = 18) -> DesignProposal:
    """A perforated lantern: a revolved, bellied shell with a twisted ring of slots and a top handle ring."""
    return DesignProposal.model_validate({
        "intent": {"prompt": "A lantern-like vessel with a bulging wall, a spiral band of slots and a ring on top.",
                   "assumptions": ["dimensions in mm", "open top, closed 3 mm base"]},
        "specification": {
            "parameters": [
                {"id": "H", "kind": "length", "value": height_mm, "unit": "mm", "lower": 40, "upper": 400},
                {"id": "R", "kind": "length", "value": radius_mm, "unit": "mm", "lower": 15, "upper": 200},
                {"id": "t", "kind": "length", "value": wall_mm, "unit": "mm", "lower": 1.2, "upper": 10},
            ],
            "constraints": [
                {"id": "wall_vs_radius", "expr": "t < R / 5", "cls": "geometric"},
                {"id": "slender", "expr": "lantern.body.extent_z <= 3 * lantern.body.extent_x",
                 "cls": "geometric", "strength": "soft"},
            ],
        },
        "structure": {"components": [{
            "id": "lantern", "kind": "procedural", "tags": ["creative"],
            "geometry": {"ops": [
                {"id": "wall", "op": "profile", "start": ["0.6*R", 0], "tags": ["shell"], "segments": [
                    {"kind": "spline", "points": [["R", "0.45*H"], ["0.7*R", "0.9*H"], ["0.72*R", "H"]]},
                    {"kind": "line", "to": ["0.72*R - t", "H"]},
                    {"kind": "spline", "points": [["0.7*R - t", "0.9*H"], ["R - t", "0.45*H"], ["0.6*R - t", "t"]]},
                    {"kind": "line", "to": [0, "t"]}, {"kind": "line", "to": [0, 0]},
                    {"kind": "line", "to": ["0.6*R", 0]}]},
                {"id": "shell_y", "op": "revolve", "profile": "wall"},
                {"id": "shell", "op": "rotate", "input": "shell_y", "axis": [1, 0, 0], "angle": 90},
                {"id": "slot", "op": "box", "size": ["3*t", "0.18*R", "0.28*H"], "center": True, "tags": ["slot"]},
                {"id": "slot_tilt", "op": "rotate", "input": "slot", "axis": [1, 0, 0], "angle": 25},
                {"id": "slot_out", "op": "translate", "input": "slot_tilt", "offset": ["R", 0, "0.45*H"]},
                {"id": "slots", "op": "polar_pattern", "input": "slot_out", "count": slots},
                {"id": "perforated", "op": "difference", "base": "shell", "tools": ["slots"]},
                {"id": "ring", "op": "torus", "major_radius": "0.72*R - t/2", "minor_radius": "t"},
                {"id": "ring_top", "op": "translate", "input": "ring", "offset": [0, 0, "H"], "tags": ["handle"]},
                {"id": "body", "op": "union", "inputs": ["perforated", "ring_top"]},
            ], "outputs": {"body": "body"}}}]},
    })


def gyroid_orb_proposal(radius_mm: float = 20, cell_mm: float = 6, voxel_mm: float = 0.5) -> DesignProposal:
    """A field-defined orb: a sphere filled with a gyroid TPMS sheet (implicit/SDF path, mesh-only output)."""
    k = 6.283185307179586 / cell_mm
    expr = (f"max(abs(sin({k}*x)*cos({k}*y) + sin({k}*y)*cos({k}*z) + sin({k}*z)*cos({k}*x)) - 0.35, "
            f"sqrt(x^2 + y^2 + z^2) - {radius_mm})")
    b = radius_mm + 2 * voxel_mm
    return DesignProposal.model_validate({
        "intent": {"prompt": "An orb made of a gyroid lattice sheet."},
        "structure": {"components": [{
            "id": "orb", "kind": "procedural", "tags": ["creative", "multi_body"],
            "geometry": {"ops": [{"id": "g", "op": "implicit", "expr": expr, "bounds_min": [-b, -b, -b],
                                  "bounds_max": [b, b, b], "voxel": voxel_mm, "tags": ["tpms"]}],
                         "outputs": {"body": "g"}}}]},
    })
