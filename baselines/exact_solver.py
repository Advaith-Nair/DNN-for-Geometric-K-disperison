"""
Exact k-dispersion solver, implementing Algorithm A(k) from Theorem 1 of
Chen & Dumitrescu (2025), "A Couple of Simple Algorithms for k-Dispersion".

This is a GRADING TOOL, not part of the actual solver pipeline. It exists
purely to give ground-truth answers to compare the dNN + binary search
pipeline against, at a scale beyond what brute force can handle.

Recursive structure (paraphrasing the paper's Algorithm A(k)):
    For each pair (a, b) in P:
        x = dist(a, b)
        P' = points in P at distance >= x from BOTH a and b
        if |P'| < k-2: skip this pair
        recursively find the best (k-2)-subset K' of P' (Algorithm A(k-2))
        if min pairwise distance in K' < x: skip (a,b can't anchor this)
        record max(x) achieved this way, with witness {a,b} u K'

Base cases (k=1, 2, 3) are solved by direct brute force, since brute-forcing
pairs/triples is cheap (O(n^2), O(n^3)) and the paper's own base-case
citations aren't reproduced here in enough detail to reimplement directly.
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
    radius: float
    witness: tuple[int, ...]


def _brute_force_small_k(points: PointSet, k: int, allowed_indices: list[int]) -> ExactResult:
    """Brute force over all k-subsets of allowed_indices. Used for k=1,2,3 base cases."""
    if k == 1:
        # any single point; radius is degenerate (no pairs), report +inf
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
    """
    Exact solver via Algorithm A(k) (Theorem 1). Correct but combinatorial --
    only practical for small k (roughly k <= 6-8 depending on n), since the
    recursion effectively branches over all pairs at each of k/2 levels.

    allowed_indices restricts the point universe considered -- used
    internally by the recursion; pass None at the top level to use all
    points.
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

            # P' = points at distance >= x from BOTH a and b
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
        # no valid (a,b) anchor pair found -- k-dispersion infeasible at any
        # positive radius with this many points (shouldn't normally happen
        # if k <= n, but guard against it)
        return ExactResult(radius=0.0, witness=tuple(allowed_indices[:k]))

    return ExactResult(radius=best_radius, witness=best_witness)
