"""
Test the oracle in isolation, before wiring into binary search.

Uses the same 6-point chain example from the geometry tests, where we
already know the correct answer by hand:
  - At x=4: conflict graph is A-B-C-D-E-F (chain). Max independent set = 3
    (e.g. {A,C,E}). So k=3 should be FEASIBLE, k=4 should be INFEASIBLE.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from geometry.points import make_point_set, random_uniform_points
from geometry.conflict_graph import build_conflict_graph
from dnn_core import TrainConfig
from feasibility_oracle import OracleConfig, feasibility_oracle


def make_chain_example():
    positions = np.array([[0.0], [2.0], [5.0], [7.0], [10.0], [12.0]])
    return make_point_set(positions)


def test_chain_k3_feasible_at_x4():
    """Known answer: at x=4, {A,C,E} is a valid independent set of size 3."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=4.0)

    cfg = OracleConfig(train_cfg=TrainConfig(target_mode="k", max_steps=300))
    result = feasibility_oracle(g, k=3, cfg=cfg, base_seed=0)

    assert result.feasible is True, f"expected feasible, got infeasible. diagnostics: {result}"
    assert g.is_independent_set(result.witness), "witness set must be conflict-free"
    assert len(result.witness) == 3
    print(f"PASS: k=3 at x=4 correctly found feasible, witness={result.witness}, "
          f"restarts_used={result.restarts_used}")


def test_chain_k4_infeasible_at_x4():
    """
    Known answer: at x=4, the graph is a 6-node path graph. Max independent
    set in a path of 6 nodes is 3 (ceil(6/2)), so k=4 must be infeasible.
    """
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=4.0)

    cfg = OracleConfig(
        train_cfg=TrainConfig(target_mode="k", max_steps=300),
        max_restarts=5,
    )
    result = feasibility_oracle(g, k=4, cfg=cfg, base_seed=0)

    assert result.feasible is False, (
        f"expected infeasible (max IS in 6-path is 3), got feasible with witness "
        f"{result.witness}"
    )
    print(f"PASS: k=4 at x=4 correctly found infeasible after "
          f"{result.restarts_used} restarts")


def test_chain_k1_always_feasible():
    """k=1 should always be trivially feasible at any x (just pick any single point)."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=4.0)

    cfg = OracleConfig(train_cfg=TrainConfig(target_mode="k", max_steps=200))
    result = feasibility_oracle(g, k=1, cfg=cfg, base_seed=0)

    assert result.feasible is True
    print(f"PASS: k=1 trivially feasible, witness={result.witness}")


def test_empty_graph_all_feasible_up_to_n():
    """At x=1 (no edges at all in the chain example), any k up to n should be feasible."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=1.0)   # no edges, see geometry tests
    assert g.num_edges() == 0

    cfg = OracleConfig(train_cfg=TrainConfig(target_mode="k", max_steps=300))
    for k in [1, 3, 6]:
        result = feasibility_oracle(g, k=k, cfg=cfg, base_seed=0)
        assert result.feasible is True, f"k={k} should be feasible in an empty graph"
        assert g.is_independent_set(result.witness)
    print("PASS: empty conflict graph -> feasible for k=1, 3, 6 (all of n)")


def test_complete_graph_only_k1_feasible():
    """At x=13 (> diameter), the graph is complete. Only k=1 can be independent."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=13.0)
    n = ps.n
    assert g.num_edges() == n * (n - 1) // 2   # complete graph

    cfg = OracleConfig(
        train_cfg=TrainConfig(target_mode="k", max_steps=300),
        max_restarts=5,
    )
    r1 = feasibility_oracle(g, k=1, cfg=cfg, base_seed=0)
    r2 = feasibility_oracle(g, k=2, cfg=cfg, base_seed=0)

    assert r1.feasible is True
    assert r2.feasible is False
    print("PASS: complete graph -> k=1 feasible, k=2 infeasible")


def test_witness_is_always_valid_when_feasible():
    """
    Regression-style check across several random instances: whenever the
    oracle claims feasible, the witness it returns must actually be a valid,
    correctly-sized independent set. This is a hard correctness invariant
    -- must never be violated regardless of how the dNN's continuous
    optimization landed.
    """
    for seed in range(5):
        ps = random_uniform_points(15, dim=2, seed=seed)
        # pick a moderate x so the graph has a mix of edges/non-edges
        x = np.median(ps.dist[ps.dist > 0])
        g = build_conflict_graph(ps, x=x)

        cfg = OracleConfig(train_cfg=TrainConfig(target_mode="k", max_steps=300))
        for k in [2, 3]:
            result = feasibility_oracle(g, k=k, cfg=cfg, base_seed=seed)
            if result.feasible:
                assert g.is_independent_set(result.witness), (
                    f"INVALID witness returned for seed={seed}, k={k}: {result.witness}"
                )
                assert len(result.witness) == k
    print("PASS: witness sets are always valid & correctly sized across random instances")


def test_two_improvement_grows_set():
    """
    Path 0-1-2 (points at 0, 1, 2 with x=1.5). Starting from {1}, plain
    repair + greedy augment cannot grow it (0 and 2 both conflict with 1),
    but the 2-improvement move swaps 1 out for {0, 2}.
    """
    import numpy as np
    from repair import repair
    ps = make_point_set(np.array([[0.0], [1.0], [2.0]]))
    g = build_conflict_graph(ps, x=1.5)
    assert repair([1], g, local_search=False) == [1]
    out = repair([1], g, local_search=True)
    assert out == [0, 2] and g.is_independent_set(out)
    print("PASS: 2-improvement local search turns {1} into {0, 2}")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} oracle tests passed.")
