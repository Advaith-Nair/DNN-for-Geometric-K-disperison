"""
Hand-checkable tests for the geometry layer.

The 1D point set below reproduces the "6-point chain" example we worked
through by hand: points on a line spaced so we know the exact pairwise
distances and can predict the conflict graph at chosen x values.
"""

import numpy as np
from points import make_point_set, random_uniform_points, random_clustered_points
from conflict_graph import build_conflict_graph


def make_chain_example():
    """
    Points A..F on a line at positions chosen so pairwise distances match
    the worked example:
      A-B=2, B-C=3, C-D=2, D-E=3, E-F=2   (chain, small gaps)
      and everything else (A-C, A-D, ... ) is >= 7.
    Positions: A=0, B=2, C=5, D=7, E=10, F=12
    """
    positions = np.array([[0.0], [2.0], [5.0], [7.0], [10.0], [12.0]])
    return make_point_set(positions)


def test_diameter_matches_hand_calc():
    ps = make_chain_example()
    # A=0, F=12 -> diameter should be 12
    assert abs(ps.diameter() - 12.0) < 1e-9
    print("PASS: diameter matches hand calculation (12.0)")


def test_min_positive_gap():
    ps = make_chain_example()
    # smallest gap is A-B = 2, C-D = 2, E-F = 2 -> min should be 2
    assert abs(ps.min_positive_gap() - 2.0) < 1e-9
    print("PASS: min_positive_gap matches hand calculation (2.0)")


def test_conflict_graph_at_x_4_is_chain():
    """
    At x=4: edges should be exactly A-B(2), B-C(3), C-D(2), D-E(3), E-F(2)
    since all of these are < 4, and the next-smallest distance (A-C=5) is not.
    """
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=4.0)
    expected_edges = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)}
    assert set(g.edges) == expected_edges, f"got {set(g.edges)}"
    assert g.num_edges() == 5
    print("PASS: conflict graph at x=4 is exactly the expected 5-edge chain")


def test_conflict_graph_at_x_1_is_empty():
    """At x=1, nothing is closer than 1, so no edges at all."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=1.0)
    assert g.num_edges() == 0
    print("PASS: conflict graph at x=1 has zero edges (nothing that close)")


def test_conflict_graph_at_x_13_is_complete():
    """At x=13 (> diameter of 12), every pair is 'too close', graph is complete."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=13.0)
    n = ps.n
    assert g.num_edges() == n * (n - 1) // 2
    print(f"PASS: conflict graph at x=13 is complete ({g.num_edges()} edges)")


def test_independent_set_check():
    """
    At x=4 the graph is the chain A-B-C-D-E-F. {A, C, E} (indices 0,2,4)
    should be independent (no direct edges among them); {A, B, C} should not
    (A-B and B-C are edges).
    """
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=4.0)
    assert g.is_independent_set([0, 2, 4]) is True
    assert g.is_independent_set([0, 1, 2]) is False
    print("PASS: is_independent_set correctly validates {A,C,E} vs {A,B,C}")


def test_strict_inequality_boundary():
    """
    A pair at distance EXACTLY x should NOT be an edge (spacing constraint
    satisfied at the boundary). A-B distance is exactly 2, so at x=2 there
    should be no A-B edge.
    """
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=2.0)
    assert not g.has_edge(0, 1), "distance exactly == x should not be an edge"
    print("PASS: pairs at distance exactly x are correctly NOT edges (strict <)")


def test_degree_and_neighbors():
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=4.0)
    # B (index 1) is connected to A and C -> degree 2
    assert g.degree(1) == 2
    assert set(g.neighbors(1).tolist()) == {0, 2}
    # A (index 0) is only connected to B -> degree 1
    assert g.degree(0) == 1
    print("PASS: degree/neighbors correct for chain graph")


def test_monotonicity_more_edges_as_x_grows():
    """
    Sanity check on the core assumption binary search relies on: as x grows,
    the conflict graph can only gain edges, never lose them.
    """
    ps = random_uniform_points(30, dim=2, seed=42)
    xs = np.linspace(0.0, ps.diameter(), 15)
    prev_edges = -1
    for x in xs:
        g = build_conflict_graph(ps, x)
        assert g.num_edges() >= prev_edges, "edge count must be non-decreasing in x"
        prev_edges = g.num_edges()
    print("PASS: conflict graph edge count is monotonically non-decreasing in x")


def test_caching_reused_not_recomputed():
    """
    Confirm build_conflict_graph only re-thresholds the cached distance
    matrix -- i.e. the PointSet's dist matrix is untouched/read-only across
    repeated calls at different x.
    """
    ps = random_uniform_points(20, dim=2, seed=1)
    dist_before = ps.dist.copy()
    for x in [0.1, 0.3, 0.5, 0.9]:
        build_conflict_graph(ps, x)
    assert np.array_equal(dist_before, ps.dist), "distance cache was mutated"
    print("PASS: distance cache is not mutated across repeated conflict graph builds")


def test_random_generators_run_and_produce_valid_pointsets():
    ps1 = random_uniform_points(50, dim=2, seed=0)
    ps2 = random_clustered_points(50, dim=2, n_clusters=4, seed=0)
    for ps in (ps1, ps2):
        assert ps.n == 50
        assert ps.dist.shape == (50, 50)
        assert np.allclose(np.diag(ps.dist), 0.0)
        assert np.allclose(ps.dist, ps.dist.T)
    print("PASS: random point generators produce valid, symmetric, zero-diagonal PointSets")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} geometry tests passed.")
