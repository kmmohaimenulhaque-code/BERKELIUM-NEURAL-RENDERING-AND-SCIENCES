"""Test geometry built through Berkelium's own OCCT backend (STEP bytes, mm)."""
from berkelium.geometry.ir import GeometryGraph
from berkelium.geometry.occt import OCCTBackend


def _step(ops, out):
    be = OCCTBackend()
    r = be.execute(GeometryGraph(ops=ops, outputs={"b": out}))
    return be.export(r.bodies["b"], "step")[0]


def tube_step(ri, ro, h):
    return _step([{"op": "cylinder", "id": "o", "radius": ro, "height": h},
                  {"op": "cylinder", "id": "i", "radius": ri, "height": h},
                  {"op": "difference", "id": "t", "base": "o", "tools": ["i"]}], "t")


def quarter_tube_step(ri, ro, h):
    return _step([{"op": "cylinder", "id": "o", "radius": ro, "height": h},
                  {"op": "cylinder", "id": "i", "radius": ri, "height": h},
                  {"op": "difference", "id": "t", "base": "o", "tools": ["i"]},
                  {"op": "box", "id": "q", "size": [ro + 5, ro + 5, h]},
                  {"op": "intersection", "id": "qt", "inputs": ["t", "q"]}], "qt")
