"""
Tests for the exact solver and the greedy heuristic.

The exact solver is compared with full enumeration and must agree up to
floating-point error. The greedy heuristic is checked against its
1/2-approximation guarantee.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import itertools
import numpy as np

from geometry.points import make_point_set, random_uniform_points
from exact_solver import exact_k_dispersion
from greedy_solver import greedy_k_dispersion


def brute_force_k_dispersion(points, k):
    """Return the optimal value and an optimal subset by full enumeration."""
    n = points.n
    best_min_dist = -1.0
    best_subset = None
    for combo in itertools.combinations(range(n), k):
        pairs = list(itertools.combinations(combo, 2))
        if not pairs:
            continue
        min_dist = min(points.dist[i, j] for i, j in pairs)
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_subset = combo
    return best_min_dist, best_subset


def make_chain_example():
    """Six collinear points at positions 0, 2, 5, 7, 10, 12."""
    positions = np.array([[0.0], [2.0], [5.0], [7.0], [10.0], [12.0]])
    return make_point_set(positions)


def test_exact_matches_brute_force_on_chain():
    """The exact solver agrees with enumeration on the chain with k = 3."""
    ps = make_chain_example()
    true_radius, _ = brute_force_k_dispersion(ps, k=3)
    result = exact_k_dispersion(ps, k=3)

    assert abs(result.radius - true_radius) < 1e-9, (
        f"exact solver {result.radius}, enumeration {true_radius}"
    )


def test_exact_matches_brute_force_on_random_instances():
    """The exact solver agrees with enumeration on random instances, k = 2 to 5."""
    for seed in range(5):
        ps = random_uniform_points(8, dim=2, seed=seed)
        for k in [2, 3, 4, 5]:
            true_radius, _ = brute_force_k_dispersion(ps, k)
            result = exact_k_dispersion(ps, k)
            assert abs(result.radius - true_radius) < 1e-9, (
                f"seed={seed} k={k}: exact={result.radius} enumeration={true_radius}"
            )


def test_exact_witness_is_valid():
    """The set returned by the exact solver attains the reported value."""
    for seed in range(3):
        ps = random_uniform_points(10, dim=2, seed=seed)
        for k in [3, 4]:
            result = exact_k_dispersion(ps, k)
            pairs = list(itertools.combinations(result.witness, 2))
            actual_min = min(ps.dist[i, j] for i, j in pairs)
            assert abs(actual_min - result.radius) < 1e-9, (
                f"reported {result.radius}, witness attains {actual_min}"
            )


def test_greedy_approximation_ratio():
    """The greedy value is at least half of the optimum."""
    for seed in range(6):
        ps = random_uniform_points(10, dim=2, seed=seed)
        for k in [3, 4, 5]:
            exact = exact_k_dispersion(ps, k)
            greedy = greedy_k_dispersion(ps, k)
            ratio = greedy.radius / exact.radius if exact.radius > 0 else 1.0
            assert ratio >= 0.5 - 1e-9, (
                f"seed={seed} k={k}: greedy={greedy.radius:.4f} "
                f"exact={exact.radius:.4f} ratio={ratio:.4f}"
            )


def test_greedy_witness_is_valid():
    """The set returned by the greedy heuristic attains the reported value."""
    ps = random_uniform_points(12, dim=2, seed=1)
    result = greedy_k_dispersion(ps, k=5)
    pairs = list(itertools.combinations(result.witness, 2))
    actual_min = min(ps.dist[i, j] for i, j in pairs)
    assert abs(actual_min - result.radius) < 1e-9


def test_k1_and_k2():
    """For k = 1 the value is infinite; for k = 2 both solvers return the diameter."""
    ps = random_uniform_points(6, dim=2, seed=0)

    assert exact_k_dispersion(ps, k=1).radius == float("inf")
    assert greedy_k_dispersion(ps, k=1).radius == float("inf")

    assert abs(exact_k_dispersion(ps, k=2).radius - ps.diameter()) < 1e-9
    assert abs(greedy_k_dispersion(ps, k=2).radius - ps.diameter()) < 1e-9