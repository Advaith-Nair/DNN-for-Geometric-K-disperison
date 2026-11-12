"""
Exact solver for k-dispersion.

Implements Algorithm A(k) of Chen and Dumitrescu, "A Couple of Simple
Algorithms for k-Dispersion" (arXiv:2511.00692, 2025), Theorem 1. The solver
is used to obtain reference optima for small instances.

For each pair (a, b) with x = d(a, b), the algorithm keeps the points at
distance at least x from both a and b, recursively solves (k-2)-dispersion
on them, and accepts the pair if the recursive solution has minimum distance
at least x. The best accepted x is optimal, since the closest pair of an
optimal solution is one of the pairs examined.

The base cases k <= 3 are solved by enumeration instead of the specialised
algorithms cited in the paper. This does not affect correctness, only the
running time.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.points import PointSet


@dataclass
class ExactResult:
    """Optimal value and an optimal subset of point indices."""

    radius: float
    witness: tuple[int, ...]


def _brute_force_small_k(points: PointSet, k: int, allowed_indices: list[int]) -> ExactResult:
    """Solve k-dispersion on `allowed_indices` by enumeration (used for k <= 3)."""
    if k == 1:
        # A single point has no pairwise distance; the value is reported as infinity.
        return ExactResult(radius=float("inf"), witness=(allowed_indices[0],))

    best_radius = -1.0
    best_witness = None
    for combo in itertools.combinations(allowed_indices, k):
        pairs = list(itertools.combinations(combo, 2))
        min_dist = min(points.dist[i, j] for i, j in pairs)
        if min_dist > best_radius:
            best_radius = min_dist
            best_witness = combo
    return ExactResult(radius=best_radius, witness=best_witness)


def exact_k_dispersion(points: PointSet, k: int,
                        allowed_indices: list[int] | None = None) -> ExactResult:
    """Solve k-dispersion exactly with Algorithm A(k).

    The running time is O(n^(k-1)) up to logarithmic factors, so the solver
    is intended for small n and k.

    Args:
        points: The point set.
        k: Number of points to select.
        allowed_indices: Subset of points to choose from. Used by the
            recursion; leave as None to use all points.
    """
    if allowed_indices is None:
        allowed_indices = list(range(points.n))

    if k > len(allowed_indices):
        raise ValueError(f"k={k} exceeds available points ({len(allowed_indices)})")

    if k <= 3:
        return _brute_force_small_k(points, k, allowed_indices)

    best_radius = -1.0
    best_witness = None

    n_allowed = len(allowed_indices)
    for i in range(n_allowed):
        for j in range(i + 1, n_allowed):
            a, b = allowed_indices[i], allowed_indices[j]
            x = points.dist[a, b]

            # Points that can be selected together with a and b at value x.
            remaining = [
                p for p in allowed_indices
                if p != a and p != b and points.dist[p, a] >= x and points.dist[p, b] >= x
            ]

            if len(remaining) < k - 2:
                continue

            sub_result = exact_k_dispersion(points, k - 2, allowed_indices=remaining)

            if sub_result.radius < x:
                continue

            if x > best_radius:
                best_radius = x
                best_witness = (a, b) + sub_result.witness

    if best_witness is None:
        # Not reached when k <= n; kept as a safeguard.
        return ExactResult(radius=0.0, witness=tuple(allowed_indices[:k]))

    return ExactResult(radius=best_radius, witness=best_witness)