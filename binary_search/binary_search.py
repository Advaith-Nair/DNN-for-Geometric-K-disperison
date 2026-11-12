"""
Binary search for the k-dispersion value.

The optimal value x* is the largest x for which the conflict graph G_x has an
independent set of size k. Feasibility is monotone in x, so x* can be found
by binary search, using the feasibility oracle as the decision procedure at
each step. Since x* is always one of the pairwise distances, the search runs
over the sorted distinct distances, and the result is exact whenever the
oracle answers correctly. All problem-specific work is done by the oracle;
this module only maintains the search interval and the best solution found.
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
    """Parameters of the binary search.

    Attributes:
        oracle_cfg: Configuration passed to the feasibility oracle.
        max_iterations: Upper bound on the number of oracle calls.
    """

    oracle_cfg: OracleConfig = None
    max_iterations: int = 40

    def __post_init__(self):
        if self.oracle_cfg is None:
            self.oracle_cfg = OracleConfig()


@dataclass
class SearchResult:
    """Outcome of a binary search run.

    Attributes:
        best_radius: Minimum pairwise distance of the best set found.
        best_witness: Indices of the k selected points.
        iterations_run: Number of oracle calls made.
        history: One record per iteration (guess, answer, interval bounds).
    """

    best_radius: float
    best_witness: list[int]
    iterations_run: int
    history: list[dict] = field(default_factory=list)


def _witness_radius(points: PointSet, witness) -> float:
    """Return the minimum pairwise distance within a set of point indices."""
    w = list(witness)
    if len(w) < 2:
        return float("inf")
    sub = points.dist[np.ix_(w, w)]
    return float(sub[np.triu_indices(len(w), k=1)].min())


def _solve_discrete(points: PointSet, k: int, cfg: SearchConfig,
                    base_seed: int | None) -> SearchResult:
    """Binary search over the sorted distinct pairwise distances.

    Invariant: cands[lo] is feasible (a witness is held for it) and every
    index >= hi has been answered infeasible. When the oracle returns a
    witness, lo is advanced to the witness's actual minimum distance, which
    is never smaller than the value queried. The search needs at most
    ceil(log2(m)) oracle calls for m candidate distances.
    """
    n = points.n
    cands = np.unique(points.dist[np.triu_indices(n, k=1)])

    # Any k points form a valid solution for the smallest candidate.
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
    """Solve k-dispersion by binary search with the feasibility oracle.

    Args:
        points: The point set.
        k: Number of points to select (1 <= k <= n).
        cfg: Search configuration.
        base_seed: Seed passed to the oracle for reproducibility.

    Returns:
        The best set found and its minimum pairwise distance. For k = 1 the
        radius is reported as infinity, since a single point has no pairs.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > points.n:
        raise ValueError(f"k={k} exceeds number of points n={points.n}")
    if k == 1:
        return SearchResult(best_radius=float("inf"), best_witness=[0], iterations_run=0)
    return _solve_discrete(points, k, cfg, base_seed)