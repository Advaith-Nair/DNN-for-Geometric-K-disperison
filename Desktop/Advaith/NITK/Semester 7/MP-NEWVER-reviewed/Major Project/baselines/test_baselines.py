"""
Tests for Stage 4: ground-truth baselines.

Two things need verifying, independently of each other and independently
of the main solver pipeline:
  1. The exact solver (Algorithm A(k)) must EXACTLY match brute force
     wherever brute force is tractable -- no ratio slack allowed here,
     since this is supposed to be an exact algorithm.
  2. The greedy heuristic must stay at or above its proven 1/2-approximation
     bound relative to the exact solver, across many instances.
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
    """Same brute force as used in Stage 3 tests -- independent ground truth."""
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
    positions = np.array([[0.0], [2.0], [5.0], [7.0], [10.0], [12.0]])
    return make_point_set(positions)


def test_exact_matches_brute_force_chain():
    """Known example: chain, k=3. Exact solver should match brute force EXACTLY."""
    ps = make_chain_example()
    true_radius, _ = brute_force_k_dispersion(ps, k=3)
    result = exact_k_dispersion(ps, k=3)

    assert abs(result.radius - true_radius) < 1e-9, (
        f"exact solver got {result.radius}, brute force says {true_radius}"
    )
    print(f"PASS: exact solver matches brute force exactly on chain k=3 "
          f"(radius={result.radius:.4f}, witness={result.witness})")


def test_exact_matches_brute_force_random_instances():
    """
    Several random small instances, several k values, exact solver vs
    brute force. Since both are supposed to be exact, tolerance should be
    essentially zero (floating point only).
    """
    for seed in range(5):
        ps = random_uniform_points(8, dim=2, seed=seed)
        for k in [2, 3, 4, 5]:
            true_radius, _ = brute_force_k_dispersion(ps, k)
            result = exact_k_dispersion(ps, k)
            assert abs(result.radius - true_radius) < 1e-9, (
                f"seed={seed} k={k}: exact={result.radius} brute_force={true_radius}"
            )
    print("PASS: exact solver matches brute force exactly across 5 seeds x 4 k-values (20 checks)")


def test_exact_solver_witness_is_valid():
    """The exact solver's witness must actually achieve its claimed radius."""
    for seed in range(3):
        ps = random_uniform_points(10, dim=2, seed=seed)
        for k in [3, 4]:
            result = exact_k_dispersion(ps, k)
            pairs = list(itertools.combinations(result.witness, 2))
            actual_min = min(ps.dist[i, j] for i, j in pairs)
            assert abs(actual_min - result.radius) < 1e-9, (
                f"claimed radius {result.radius} but witness achieves {actual_min}"
            )
    print("PASS: exact solver witness sets genuinely achieve their claimed radius")


def test_greedy_meets_half_approximation_bound():
    """
    Greedy's proven worst-case bound is 1/2. Check this actually holds
    empirically against the exact solver, across several instances --
    and report how much better than the worst case it typically does
    (usually much better in practice).
    """
    ratios = []
    for seed in range(6):
        ps = random_uniform_points(10, dim=2, seed=seed)
        for k in [3, 4, 5]:
            exact = exact_k_dispersion(ps, k)
            greedy = greedy_k_dispersion(ps, k)
            ratio = greedy.radius / exact.radius if exact.radius > 0 else 1.0
            ratios.append(ratio)
            assert ratio >= 0.5 - 1e-9, (
                f"seed={seed} k={k}: greedy={greedy.radius:.4f} exact={exact.radius:.4f} "
                f"ratio={ratio:.4f} -- violates proven 1/2 bound!"
            )

    worst = min(ratios)
    avg = sum(ratios) / len(ratios)
    print(f"PASS: greedy stays >= 0.5 bound across {len(ratios)} checks -- "
          f"worst_ratio={worst:.4f} avg_ratio={avg:.4f}")


def test_greedy_witness_is_valid():
    """Greedy's witness must also genuinely achieve its claimed radius."""
    ps = random_uniform_points(12, dim=2, seed=1)
    result = greedy_k_dispersion(ps, k=5)
    pairs = list(itertools.combinations(result.witness, 2))
    actual_min = min(ps.dist[i, j] for i, j in pairs)
    assert abs(actual_min - result.radius) < 1e-9
    print(f"PASS: greedy witness genuinely achieves claimed radius "
          f"(radius={result.radius:.4f})")


def test_k1_and_k2_edge_cases():
    """Base cases for both solvers."""
    ps = random_uniform_points(6, dim=2, seed=0)

    exact_k1 = exact_k_dispersion(ps, k=1)
    greedy_k1 = greedy_k_dispersion(ps, k=1)
    assert exact_k1.radius == float("inf")
    assert greedy_k1.radius == float("inf")

    exact_k2 = exact_k_dispersion(ps, k=2)
    greedy_k2 = greedy_k_dispersion(ps, k=2)
    # k=2 for BOTH exact and greedy should be exactly the diameter pair
    assert abs(exact_k2.radius - ps.diameter()) < 1e-9
    assert abs(greedy_k2.radius - ps.diameter()) < 1e-9
    print("PASS: k=1 and k=2 edge cases correct for both exact and greedy solvers")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} baseline tests passed.")
