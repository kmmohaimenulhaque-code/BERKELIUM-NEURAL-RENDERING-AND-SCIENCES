"""Relation discovery from verified evidence (ADR-021): hypothesis -> candidate law -> independent validation
-> validity domain -> engineering memory.

1. ``pi_groups`` derives a complete set of dimensionless groups from the variables' units (Buckingham Pi, exact
   rational nullspace of the dimension matrix). Laws discovered in Pi groups are unit-consistent by construction
   and transfer across scales.
2. ``discover`` fits sparse models  y = sum_k c_k * prod_i Pi_i^e_ik  (<= ``max_terms`` terms, integer exponents)
   by exhaustive enumeration + least squares on TRAIN evidence only, scores each on a HELD-OUT family, and selects
   the simplest model within ``occam`` of the best held-out error (sparse regression in the spirit of SINDy /
   symbolic regression, but exhaustive over a small, explicit library — so the result is reproducible).
3. ``promote`` is the gate. A candidate becomes a Relation only if its held-out error is within the evidence's
   own numerical error plus an explicit tolerance. The promoted relation carries: validity = box hull of the
   training groups (no extrapolation), model-form bound = max observed |residual| + max evidence error, and the
   ids of every evidence record used. Rejected candidates are stored too.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from ..units import Quantity


def pi_groups(units: dict[str, str]) -> list[dict[str, int]]:
    """Integer exponent vectors of a basis of dimensionless products of the given variables."""
    import sympy
    names = sorted(units)
    D = sympy.Matrix([[Quantity.of(1, units[n]).dim[k] for n in names] for k in range(7)])
    out = []
    for v in D.nullspace():
        den = sympy.ilcm(*[x.q for x in v])
        w = [int(x * den) for x in v]
        g = math.gcd(*w) or 1
        w = [x // g for x in w]
        if next(x for x in w if x) < 0:
            w = [-x for x in w]
        out.append({n: e for n, e in zip(names, w, strict=True) if e})
    return out


@dataclass(frozen=True)
class Candidate:
    terms: tuple[tuple[int, ...], ...]     # exponent vector per term (over ``groups``)
    coef: tuple[float, ...]
    groups: tuple[str, ...]
    train_rmse: float
    heldout_max_abs: float
    heldout_rmse: float

    @property
    def complexity(self) -> int:
        return sum(1 + sum(abs(e) for e in t) for t in self.terms)

    def predict(self, G: np.ndarray) -> np.ndarray:
        return _design(G, self.terms) @ np.array(self.coef)

    def formula(self, names: dict[str, str] | None = None) -> str:
        nm = names or {g: g for g in self.groups}
        parts = []
        for c, t in zip(self.coef, self.terms, strict=True):
            f = "*".join(f"({nm[g]})^{e}" if e != 1 else f"({nm[g]})" for g, e in zip(self.groups, t, strict=True)
                         if e)
            parts.append(f"{c!r}*{f}" if f else f"{c!r}")
        return " + ".join(parts) if parts else "0"


def _design(G: np.ndarray, terms) -> np.ndarray:
    return np.stack([np.prod([G[:, j] ** e for j, e in enumerate(t)], axis=0) for t in terms], axis=1)


def vanishes_at_zero(terms, group_index: int) -> bool:
    """Limiting-case check, exact for monomial models: y -> 0 as Pi_g -> 0 (others fixed) iff every term has a
    POSITIVE exponent of Pi_g."""
    return all(t[group_index] > 0 for t in terms)


def discover(G_train: np.ndarray, y_train: np.ndarray, G_held: np.ndarray, y_held: np.ndarray,
             groups: tuple[str, ...], exponents=(-2, -1, 1, 2, 3), max_terms: int = 2, max_factors: int = 2,
             occam: float = 1.25, limits: tuple[str, ...] = (), report: dict | None = None
             ) -> tuple[Candidate, list[Candidate]]:
    """``limits``: names of groups g for which physics requires y -> 0 as g -> 0 (e.g. the slender limit where
    the base theory is exact). Candidates violating a declared limit are falsified before any fitting."""
    n = len(groups)
    monos = []
    for k in range(1, max_factors + 1):
        for idx in itertools.combinations(range(n), k):
            for es in itertools.product(exponents, repeat=k):
                t = [0] * n
                for i, e in zip(idx, es, strict=True):
                    t[i] = e
                monos.append(tuple(t))
    cands = []
    killed = 0
    li = [groups.index(g) for g in limits]
    for m in range(1, max_terms + 1):
        for terms in itertools.combinations(monos, m):
            if any(not vanishes_at_zero(terms, i) for i in li):
                killed += 1
                continue
            A = _design(G_train, terms)
            if not np.all(np.isfinite(A)) or np.linalg.matrix_rank(A) < m:
                continue
            c, *_ = np.linalg.lstsq(A, y_train, rcond=None)
            rt = float(np.sqrt(np.mean((A @ c - y_train) ** 2)))
            ph = _design(G_held, terms) @ c
            cands.append(Candidate(terms, tuple(map(float, c)), groups, rt, float(np.max(np.abs(ph - y_held))),
                                   float(np.sqrt(np.mean((ph - y_held) ** 2)))))
    if report is not None:
        report["falsified_by_limits"] = killed
        report["fitted"] = len(cands)
    cands.sort(key=lambda c: (c.heldout_rmse, c.complexity, c.terms))
    best = cands[0].heldout_rmse
    chosen = min((c for c in cands if c.heldout_rmse <= best * occam), key=lambda c: (c.complexity, c.heldout_rmse,
                                                                                         c.terms))
    return chosen, cands


@dataclass
class Gate:
    """Promotion criterion: held-out |residual| <= k * (held-out evidence error) + tol (all in y units)."""
    k: float = 2.0
    tol: float = 0.0
    min_train: int = 8
    min_heldout: int = 4
    log: list[str] = field(default_factory=list)

    def passes(self, c: Candidate, y_held_err: np.ndarray, n_train: int) -> bool:
        self.log.clear()
        if n_train < self.min_train:
            self.log.append(f"too few training points ({n_train} < {self.min_train})")
        if len(y_held_err) < self.min_heldout:
            self.log.append(f"too few held-out points ({len(y_held_err)} < {self.min_heldout})")
        limit = self.k * float(np.max(y_held_err)) + self.tol
        if c.heldout_max_abs > limit:
            self.log.append(f"held-out max |residual| {c.heldout_max_abs:.3e} > limit {limit:.3e}")
        return not self.log
