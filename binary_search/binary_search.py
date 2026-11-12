"""
Binary search over the spacing radius x, using the dNN feasibility oracle
as the yes/no check at each guess.

This is Stage 3 of the pipeline: it knows NOTHING about conflict graphs,
dials, or training -- it just tracks a (lo, hi) range, guesses the
midpoint, calls the oracle, and reacts to True/False. All of the actual
solving happens inside the oracle (Stage 2); this loop is deliberately
simple so it's easy to trust and easy to debug independently of the oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "oracle"))

from geometry.points import PointSet
from geometry.conflict_graph import build_conflict_graph

from feasibility_oracle import OracleConfig, feasibility_oracle


@dataclass
class SearchConfig:
    oracle_cfg: OracleConfig = None
    max_iterations: int = 40     # hard cap, in case epsilon convergence is slow
    epsilon: float | None = None  # if None, derived from points.diameter() (see below)
    mode: str = "discrete"       # "discrete": search over the sorted pairwise distances
                                 # (the optimum is always one of them, so the answer is
                                 # exact whenever the oracle makes no false negatives).
                                 # "continuous": the original epsilon-bisection on [0, diameter],
                                 # kept only so older results can be reproduced.

    def __post_init__(self):
        if self.oracle_cfg is None:
            self.oracle_cfg = OracleConfig()


@dataclass
class SearchResult:
    best_radius: float           # largest x confirmed feasible
    best_witness: list[int]      # the k points achieving best_radius
    iterations_run: int
    # diagnostics: one entry per binary-search iteration, useful for
    # instrumenting where/why things are slow or uncertain later
    history: list[dict] = field(default_factory=list)


def _witness_radius(points: PointSet, witness) -> float:
    w = list(witness)
    if len(w) < 2:
        return float("inf")
    sub = points.dist[np.ix_(w, w)]
    return float(sub[np.triu_indices(len(w), k=1)].min())


def _solve_discrete(points: PointSet, k: int, cfg: SearchConfig,
                    base_seed: int | None) -> SearchResult:
    """
    Binary search over the sorted distinct pairwise distances d_0 < ... < d_{m-1}.

    The optimal k-dispersion radius is always one of these distances, so
    searching them directly (a) removes the epsilon-resolution error of
    continuous bisection and (b) needs only ceil(log2 m) oracle calls.

    Invariant: cands[lo] is known feasible (we hold a witness for it) and
    every index >= hi is known (or declared) infeasible. When the oracle
    returns a witness, lo jumps straight to the witness's real radius, which
    is >= the guessed value -- a free improvement.
    """
    n = points.n
    cands = np.unique(points.dist[np.triu_indices(n, k=1)])

    # Any k points are a valid witness for the smallest distance.
    best_witness = list(range(k))
    lo = int(np.searchsorted(cands, _witness_radius(points, best_witness), side="right") - 1)
    hi = len(cands)
    history = []
    iterations = 0

    while hi - lo > 1 and iterations < cfg.max_iterations:
        mid = (lo + hi) // 2
        x = float(cands[mid])
        g = build_conflict_graph(points, x)
        result = feasibility_oracle(g, k, cfg.oracle_cfg, base_seed=base_seed)

        history.append({
            "iteration": iterations,
            "x_guess": x,
            "feasible": result.feasible,
            "restarts_used": result.restarts_used,
            "lo_before": float(cands[lo]),
            "hi_before": float(cands[hi]) if hi < len(cands) else float("inf"),
        })

        if result.feasible:
            r = _witness_radius(points, result.witness)
            lo = max(mid, int(np.searchsorted(cands, r, side="right") - 1))
            best_witness = list(result.witness)
        else:
            hi = mid
        iterations += 1

    return SearchResult(
        best_radius=_witness_radius(points, best_witness),
        best_witness=best_witness,
        iterations_run=iterations,
        history=history,
    )


def solve_k_dispersion(points: PointSet, k: int, cfg: SearchConfig,
                        base_seed: int | None = None) -> SearchResult:
    """
    Binary search over x in [lo, hi] using feasibility_oracle as the check.

    lo starts at 0 (trivially feasible: k=1 always works, and for k>1, the
    oracle will confirm smaller radii are fine too). hi starts at the point
    set's diameter (the largest possible meaningful spacing).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > points.n:
        raise ValueError(f"k={k} exceeds number of points n={points.n}")
    if k == 1:
        # no pairwise distances: any single point is optimal
        return SearchResult(best_radius=float("inf"), best_witness=[0], iterations_run=0)
    if cfg.mode == "discrete":
        return _solve_discrete(points, k, cfg, base_seed)
    if cfg.mode != "continuous":
        raise ValueError(f"unknown search mode: {cfg.mode}")

    lo = 0.0
    hi = points.diameter()
    # FOUND DURING TESTING: min_positive_gap() is often too coarse to use
    # directly as epsilon -- on the 6-point chain example it produced
    # epsilon=1.0, causing the search to stop noticeably short of the true
    # optimum (4.5 vs true 5.0). Use a small fraction of the diameter
    # instead, which scales sensibly with the actual search range,
    # independent of how close together any two specific points happen to be.
    epsilon = cfg.epsilon if cfg.epsilon is not None else max(hi * 1e-4, 1e-9)

    best_radius = 0.0
    best_witness: list[int] | None = None
    history = []

    iterations = 0
    while (hi - lo) > epsilon and iterations < cfg.max_iterations:
        mid = (lo + hi) / 2.0
        g = build_conflict_graph(points, mid)

        result = feasibility_oracle(g, k, cfg.oracle_cfg, base_seed=base_seed)

        history.append({
            "iteration": iterations,
            "x_guess": mid,
            "feasible": result.feasible,
            "restarts_used": result.restarts_used,
            "lo_before": lo,
            "hi_before": hi,
        })

        if result.feasible:
            lo = mid
            best_radius = mid
            best_witness = result.witness
        else:
            hi = mid

        iterations += 1

    if best_witness is None:
        # k=1 edge case safeguard: should always be feasible at x=0, but if
        # somehow no iteration ever ran (e.g. lo==hi already), check once
        # directly at x=0 rather than returning an empty answer.
        g = build_conflict_graph(points, 0.0)
        result = feasibility_oracle(g, k, cfg.oracle_cfg, base_seed=base_seed)
        if result.feasible:
            best_radius = 0.0
            best_witness = result.witness

    return SearchResult(
        best_radius=best_radius,
        best_witness=best_witness if best_witness is not None else [],
        iterations_run=iterations,
        history=history,
    )
