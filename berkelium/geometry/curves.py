"""Shared curve semantics so every backend realises the *same* curve.

IR spline definition: the chord-length-parameterised natural cubic spline through the current point and
the listed points. If a segment already lists >= DENSE points (e.g. exact samples of a CEM involute), the
points themselves are taken as the curve samples (their spacing is the producer's responsibility)."""

from __future__ import annotations

import numpy as np

DENSE = 6
SAMPLES_PER_SPAN = 24


def natural_cubic(ctrl: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """Chord-length parameterised natural cubic spline through ``ctrl``, sampled at n+1 points."""
    P = np.asarray(ctrl, dtype=float)
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    k = len(P)
    h = np.diff(t)
    A = np.zeros((k, k))
    B = np.zeros((k, 2))
    A[0, 0] = A[-1, -1] = 1.0
    for i in range(1, k - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        B[i] = 3 * ((P[i + 1] - P[i]) / h[i] - (P[i] - P[i - 1]) / h[i - 1])
    c = np.linalg.solve(A, B)
    out = []
    for s_ in np.linspace(0.0, t[-1], n + 1):
        i = min(int(np.searchsorted(t, s_, side="right")) - 1, k - 2)
        d = s_ - t[i]
        b = (P[i + 1] - P[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        dd = (c[i + 1] - c[i]) / (3 * h[i])
        p = P[i] + b * d + c[i] * d * d + dd * d**3
        out.append((float(p[0]), float(p[1])))
    return out




def spline_points(ctrl: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(ctrl) >= DENSE + 1:
        return list(ctrl)
    return natural_cubic(ctrl, SAMPLES_PER_SPAN * (len(ctrl) - 1))
