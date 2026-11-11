"""
Greedy farthest-point k-dispersion heuristic, as described in Ravi,
Rosenkrantz & Tayi (1994) and referenced in the k-dispersion paper's
"Related work" section. Proven 1/2-approximation ratio for points
satisfying the triangle inequality (true here, since we use Euclidean
distance).

This is the PRACTICAL fast baseline for your evaluation -- unlike the
paper's "Algorithm B" (0.99-approximation), which only applies for
k >= 3*10^5 by its own stated constraint, this greedy method works at any
n, k and runs in O(n*k) time, making it the realistic speed/quality
comparison point at the scales you'll actually test.

Algorithm:
    1. Start with the diameter pair (the two farthest-apart points).
    2. Repeatedly add the point whose distance to its CLOSEST currently-
       selected point is largest (i.e. the point that is "most isolated"
       from what's already chosen).
    3. Stop once k points are selected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.points import PointSet


@dataclass
class GreedyResult:
    radius: float             # the achieved minimum pairwise distance
    witness: tuple[int, ...]


def greedy_k_dispersion(points: PointSet, k: int) -> GreedyResult:
    """
    O(n*k) greedy farthest-point selection. Fast, no training, no
    randomness -- deterministic given the point set.
    """
    n = points.n
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    if k == 1:
        return GreedyResult(radius=float("inf"), witness=(0,))

    # Step 1: diameter pair
    dist = points.dist
    i_idx, j_idx = np.unravel_index(np.argmax(dist), dist.shape)
    selected = [int(i_idx), int(j_idx)]

    if k == 2:
        return GreedyResult(radius=float(dist[i_idx, j_idx]), witness=tuple(selected))

    # Step 2: repeatedly add the point maximizing distance-to-nearest-selected
    remaining = set(range(n)) - set(selected)
    # track, for each remaining point, its distance to the nearest selected point
    min_dist_to_selected = {p: min(dist[p, s] for s in selected) for p in remaining}

    while len(selected) < k:
        # pick the remaining point that is farthest from its nearest selected neighbor
        best_p = max(remaining, key=lambda p: min_dist_to_selected[p])
        selected.append(best_p)
        remaining.remove(best_p)

        # update nearest-distance tracking for all still-remaining points
        for p in remaining:
            d = dist[p, best_p]
            if d < min_dist_to_selected[p]:
                min_dist_to_selected[p] = d

    achieved_radius = min(
        dist[a, b] for idx_a, a in enumerate(selected) for b in selected[idx_a + 1:]
    )
    return GreedyResult(radius=float(achieved_radius), witness=tuple(selected))
