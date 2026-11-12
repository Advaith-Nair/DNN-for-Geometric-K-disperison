"""
Greedy farthest-point heuristic for k-dispersion.

The heuristic of Ravi, Rosenkrantz and Tayi (1994) starts from a diameter
pair and repeatedly adds the point whose distance to the nearest selected
point is largest, until k points are selected. For distances that satisfy
the triangle inequality it is a 1/2-approximation. It runs in O(nk) time
and is deterministic for a given point set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.points import PointSet


@dataclass
class GreedyResult:
    """Minimum pairwise distance of the selected set and the selected indices."""

    radius: float
    witness: tuple[int, ...]


def greedy_k_dispersion(points: PointSet, k: int) -> GreedyResult:
    """Select k points with the farthest-point heuristic.

    Args:
        points: The point set.
        k: Number of points to select.
    """
    n = points.n
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    if k == 1:
        return GreedyResult(radius=float("inf"), witness=(0,))

    # Start from a diameter pair.
    dist = points.dist
    i_idx, j_idx = np.unravel_index(np.argmax(dist), dist.shape)
    selected = [int(i_idx), int(j_idx)]

    if k == 2:
        return GreedyResult(radius=float(dist[i_idx, j_idx]), witness=tuple(selected))

    # Distance from each unselected point to its nearest selected point.
    remaining = set(range(n)) - set(selected)
    min_dist_to_selected = {p: min(dist[p, s] for s in selected) for p in remaining}

    while len(selected) < k:
        best_p = max(remaining, key=lambda p: min_dist_to_selected[p])
        selected.append(best_p)
        remaining.remove(best_p)

        for p in remaining:
            d = dist[p, best_p]
            if d < min_dist_to_selected[p]:
                min_dist_to_selected[p] = d

    achieved_radius = min(
        dist[a, b] for idx_a, a in enumerate(selected) for b in selected[idx_a + 1:]
    )
    return GreedyResult(radius=float(achieved_radius), witness=tuple(selected))