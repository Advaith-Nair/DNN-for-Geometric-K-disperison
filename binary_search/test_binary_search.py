"""
Tests for the binary search.

Results are compared against a brute-force solver that enumerates every
k-subset, which is feasible for the small instances used here.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "oracle"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import itertools
import numpy as np

from geometry.points import make_point_set, random_uniform_points
from dnn_core import TrainConfig
from feasibility_oracle import OracleConfig
from binary_search import SearchConfig, solve_k_dispersion


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


def default_search_cfg():
    return SearchConfig(
        oracle_cfg=OracleConfig(train_cfg=TrainConfig(target_mode="k", max_steps=400)),
        max_iterations=30,
    )


def test_chain_k3():
    """On the chain with k = 3 the result is close to the brute-force optimum."""
    ps = make_chain_example()
    true_radius, _ = brute_force_k_dispersion(ps, k=3)

    result = solve_k_dispersion(ps, k=3, cfg=default_search_cfg(), base_seed=0)

    ratio = result.best_radius / true_radius if true_radius > 0 else 1.0
    assert ratio >= 0.95, f"binary search radius {result.best_radius} vs true {true_radius}"


def test_witness_matches_reported_radius():
    """The returned set attains at least the reported radius."""
    ps = make_chain_example()
    result = solve_k_dispersion(ps, k=3, cfg=default_search_cfg(), base_seed=1)

    pairs = list(itertools.combinations(result.best_witness, 2))
    actual_min_dist = min(ps.dist[i, j] for i, j in pairs)

    assert actual_min_dist >= result.best_radius - 1e-6, (
        f"witness attains {actual_min_dist}, reported {result.best_radius}"
    )


def test_k_equals_1():
    """For k = 1 a single point is returned."""
    ps = random_uniform_points(10, dim=2, seed=3)
    result = solve_k_dispersion(ps, k=1, cfg=default_search_cfg(), base_seed=0)
    assert len(result.best_witness) == 1


def test_k_equals_n():
    """For k = n all points are selected and the value is the minimum pairwise distance."""
    ps = random_uniform_points(6, dim=2, seed=4)
    true_min_gap = ps.min_positive_gap()

    result = solve_k_dispersion(ps, k=ps.n, cfg=default_search_cfg(), base_seed=0)

    assert len(result.best_witness) == ps.n
    ratio = result.best_radius / true_min_gap if true_min_gap > 0 else 1.0
    assert ratio >= 0.9, f"got {result.best_radius}, true minimum distance {true_min_gap}"


def test_against_brute_force():
    """Random instances with n = 9 and k = 2, 3, 4 are within 15% of the optimum."""
    cfg = default_search_cfg()

    for seed in range(4):
        ps = random_uniform_points(9, dim=2, seed=seed)
        for k in [2, 3, 4]:
            true_radius, _ = brute_force_k_dispersion(ps, k)
            if true_radius <= 0:
                continue
            result = solve_k_dispersion(ps, k=k, cfg=cfg, base_seed=seed)
            ratio = result.best_radius / true_radius
            assert ratio >= 0.85, (
                f"seed={seed} k={k}: search={result.best_radius:.4f} "
                f"true={true_radius:.4f} ratio={ratio:.4f}"
            )


def test_discrete_search_is_exact_with_exact_oracle():
    """With an exact oracle, the discrete search returns the optimum.

    The neural oracle is replaced by exhaustive search, so this test checks
    the search procedure independently of the oracle.
    """
    import binary_search as bs
    from feasibility_oracle import OracleResult

    def exact_oracle(g, k, cfg, base_seed=None):
        for combo in itertools.combinations(range(g.n), k):
            if g.is_independent_set(combo):
                return OracleResult(True, list(combo), 1, [], 0.0, [], [], [])
        return OracleResult(False, None, 1, [], 0.0, [], [], [])

    original = bs.feasibility_oracle
    bs.feasibility_oracle = exact_oracle
    try:
        for seed in range(6):
            ps = random_uniform_points(10, dim=2, seed=seed)
            for k in [2, 3, 4, 5]:
                true_radius, _ = brute_force_k_dispersion(ps, k)
                res = bs.solve_k_dispersion(ps, k, default_search_cfg())
                assert abs(res.best_radius - true_radius) < 1e-12, (seed, k, res.best_radius, true_radius)
                assert res.iterations_run <= int(np.ceil(np.log2(45))) + 1
    finally:
        bs.feasibility_oracle = original