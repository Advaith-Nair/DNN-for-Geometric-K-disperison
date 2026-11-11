"""
Tests for the binary search driver (Stage 3).

Key check: run binary search on a random point set, and compare its answer
against an independent brute-force solver that tries every possible
k-subset directly. This is the first point in the pipeline where we can
verify the FULL k-dispersion answer (radius + witness set), not just the
MIS sub-problem the oracle solves.
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
    """
    Try every k-subset directly, return the true optimal minimum pairwise
    distance. Only tractable for small n -- this is a ground-truth checker,
    not part of the actual pipeline.
    """
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


def default_search_cfg():
    return SearchConfig(
        oracle_cfg=OracleConfig(train_cfg=TrainConfig(target_mode="k", max_steps=400)),
        max_iterations=30,
    )


def test_chain_example_k3():
    """
    Known structure: 6-point chain, k=3. Compare binary search's radius
    against brute force's true optimal radius.
    """
    ps = make_chain_example()
    true_radius, true_subset = brute_force_k_dispersion(ps, k=3)

    cfg = default_search_cfg()
    result = solve_k_dispersion(ps, k=3, cfg=cfg, base_seed=0)

    ratio = result.best_radius / true_radius if true_radius > 0 else 1.0
    assert ratio >= 0.95, f"binary search radius {result.best_radius} vs true {true_radius}"
    print(f"PASS: chain k=3 -> binary_search={result.best_radius:.4f} "
          f"true={true_radius:.4f} ratio={ratio:.4f} witness={result.best_witness} "
          f"iterations={result.iterations_run}")


def test_witness_actually_achieves_claimed_radius():
    """
    Hard invariant: the witness set's OWN minimum pairwise distance must be
    >= the claimed best_radius (allow tiny floating point slack). If this
    fails, the search is lying about what it found.
    """
    ps = make_chain_example()
    cfg = default_search_cfg()
    result = solve_k_dispersion(ps, k=3, cfg=cfg, base_seed=1)

    pairs = list(itertools.combinations(result.best_witness, 2))
    actual_min_dist = min(ps.dist[i, j] for i, j in pairs)

    assert actual_min_dist >= result.best_radius - 1e-6, (
        f"witness achieves {actual_min_dist} but search claimed {result.best_radius}"
    )
    print(f"PASS: witness set genuinely achieves claimed radius "
          f"(claimed={result.best_radius:.4f}, actual={actual_min_dist:.4f})")


def test_k1_trivial():
    """k=1: any single point works, radius is meaningless/max (no pairs)."""
    ps = random_uniform_points(10, dim=2, seed=3)
    cfg = default_search_cfg()
    result = solve_k_dispersion(ps, k=1, cfg=cfg, base_seed=0)
    assert len(result.best_witness) == 1
    print(f"PASS: k=1 trivial case, witness={result.best_witness}")


def test_k_equals_n():
    """
    k=n: must use every point. The optimal radius is then just the graph's
    own true minimum pairwise distance (no choice involved).
    """
    ps = random_uniform_points(6, dim=2, seed=4)
    true_min_gap = ps.min_positive_gap()

    cfg = default_search_cfg()
    result = solve_k_dispersion(ps, k=ps.n, cfg=cfg, base_seed=0)

    assert len(result.best_witness) == ps.n
    ratio = result.best_radius / true_min_gap if true_min_gap > 0 else 1.0
    assert ratio >= 0.9, f"k=n case: got {result.best_radius}, true min gap {true_min_gap}"
    print(f"PASS: k=n uses all points, radius={result.best_radius:.4f} "
          f"vs true min gap={true_min_gap:.4f}")


def test_against_brute_force_multiple_instances():
    """
    The main trust-building test: several small random instances, several
    k values each, compared directly against brute force. Reports a
    worst-case ratio across all of them -- this is the same metric you'll
    use for full evaluation later, just on a small scale first.
    """
    cfg = default_search_cfg()
    ratios = []

    for seed in range(4):
        ps = random_uniform_points(9, dim=2, seed=seed)
        for k in [2, 3, 4]:
            true_radius, _ = brute_force_k_dispersion(ps, k)
            if true_radius <= 0:
                continue
            result = solve_k_dispersion(ps, k=k, cfg=cfg, base_seed=seed)
            ratio = result.best_radius / true_radius
            ratios.append(ratio)
            assert ratio >= 0.85, (
                f"seed={seed} k={k}: search={result.best_radius:.4f} "
                f"true={true_radius:.4f} ratio={ratio:.4f} -- too far off"
            )

    worst = min(ratios)
    avg = sum(ratios) / len(ratios)
    print(f"PASS: {len(ratios)} instance/k combos vs brute force -- "
          f"worst_ratio={worst:.4f} avg_ratio={avg:.4f}")


def test_discrete_search_is_exact_with_perfect_oracle():
    """
    Checks the SEARCH logic on its own: replace the dNN oracle with a perfect
    brute-force oracle. Discrete search over pairwise distances must then
    return the exact optimum every time (no epsilon slack). Any remaining
    miss in the full pipeline can therefore only come from oracle false
    negatives, never from the search.
    """
    import binary_search as bs
    from feasibility_oracle import OracleResult

    def perfect_oracle(g, k, cfg, base_seed=None):
        for combo in itertools.combinations(range(g.n), k):
            if g.is_independent_set(combo):
                return OracleResult(True, list(combo), 1, [], 0.0, [], [], [])
        return OracleResult(False, None, 1, [], 0.0, [], [], [])

    real = bs.feasibility_oracle
    bs.feasibility_oracle = perfect_oracle
    try:
        for seed in range(6):
            ps = random_uniform_points(10, dim=2, seed=seed)
            for k in [2, 3, 4, 5]:
                true_radius, _ = brute_force_k_dispersion(ps, k)
                res = bs.solve_k_dispersion(ps, k, default_search_cfg())
                assert abs(res.best_radius - true_radius) < 1e-12, (seed, k, res.best_radius, true_radius)
                assert res.iterations_run <= int(np.ceil(np.log2(45))) + 1
    finally:
        bs.feasibility_oracle = real
    print("PASS: discrete search returns the exact optimum with a perfect oracle")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} binary search tests passed.")
