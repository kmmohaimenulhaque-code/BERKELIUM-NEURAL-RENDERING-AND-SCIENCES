"""Mesh export helpers (STL / GLB via trimesh). Deterministic for identical inputs."""

from __future__ import annotations

import numpy as np


def mesh_bytes(vertices: np.ndarray, faces: np.ndarray, fmt: str) -> bytes:
    import trimesh

    m = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces), process=False)
    if fmt == "stl":
        return m.export(file_type="stl")
    if fmt == "glb":
        return m.export(file_type="glb")
    raise ValueError(f"unsupported mesh format {fmt}")
