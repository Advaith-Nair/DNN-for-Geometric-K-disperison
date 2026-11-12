"""
Tests for the feasibility oracle and the post-processing.

The chain instance (points at 0, 2, 5, 7, 10, 12) has a path on six vertices
as conflict graph at x = 4, whose independence number is 3.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from geometry.points import make_point_set, random_uniform_points
from geometry.conflict_graph import build_conflict_graph
from dnn_core import TrainConfig
from feasibility_oracle import OracleConfig, feasibility_oracle
from repair import repair


def make_chain_example():
    """Six collinear points at positions 0, 2, 5, 7, 10, 12."""
    positions = np.array([[0.0], [2.0], [5.0], [7.0], [10.0], [12.0]])
    return make_point_set(positions)


def test_path_k3_is_feasible():
    """On the path at x = 4 an independent set of size 3 is found."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=4.0)

    cfg = OracleConfig(train_cfg=TrainConfig(target_mode="k", max_steps=300))
    result = feasibility_oracle(g, k=3, cfg=cfg, base_seed=0)

    assert result.feasible is True, f"expected feasible: {result}"
    assert g.is_independent_set(result.witness)
    assert len(result.witness) == 3


def test_path_k4_is_infeasible():
    """On the path at x = 4 no independent set of size 4 exists."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=4.0)

    cfg = OracleConfig(
        train_cfg=TrainConfig(target_mode="k", max_steps=300),
        max_restarts=5,
    )
    result = feasibility_oracle(g, k=4, cfg=cfg, base_seed=0)

    assert result.feasible is False, f"unexpected witness {result.witness}"


def test_k1_is_feasible():
    """A single vertex is always an independent set."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=4.0)

    cfg = OracleConfig(train_cfg=TrainConfig(target_mode="k", max_steps=200))
    result = feasibility_oracle(g, k=1, cfg=cfg, base_seed=0)

    assert result.feasible is True


def test_empty_graph():
    """Without edges every k up to n is feasible."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=1.0)
    assert g.num_edges() == 0

    cfg = OracleConfig(train_cfg=TrainConfig(target_mode="k", max_steps=300))
    for k in [1, 3, 6]:
        result = feasibility_oracle(g, k=k, cfg=cfg, base_seed=0)
        assert result.feasible is True, f"k={k} should be feasible"
        assert g.is_independent_set(result.witness)


def test_complete_graph():
    """In a complete graph only k = 1 is feasible."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=13.0)
    n = ps.n
    assert g.num_edges() == n * (n - 1) // 2

    cfg = OracleConfig(
        train_cfg=TrainConfig(target_mode="k", max_steps=300),
        max_restarts=5,
    )
    assert feasibility_oracle(g, k=1, cfg=cfg, base_seed=0).feasible is True
    assert feasibility_oracle(g, k=2, cfg=cfg, base_seed=0).feasible is False


def test_witness_is_valid_on_random_graphs():
    """Every reported witness is an independent set of size exactly k."""
    for seed in range(5):
        ps = random_uniform_points(15, dim=2, seed=seed)
        x = np.median(ps.dist[ps.dist > 0])
        g = build_conflict_graph(ps, x=x)

        cfg = OracleConfig(train_cfg=TrainConfig(target_mode="k", max_steps=300))
        for k in [2, 3]:
            result = feasibility_oracle(g, k=k, cfg=cfg, base_seed=seed)
            if result.feasible:
                assert g.is_independent_set(result.witness), (
                    f"invalid witness for seed={seed}, k={k}: {result.witness}"
                )
                assert len(result.witness) == k


def test_two_improvement():
    """On the path 0-1-2, local search replaces {1} by {0, 2}."""
    ps = make_point_set(np.array([[0.0], [1.0], [2.0]]))
    g = build_conflict_graph(ps, x=1.5)
    assert repair([1], g, local_search=False) == [1]
    out = repair([1], g, local_search=True)
    assert out == [0, 2] and g.is_independent_set(out)