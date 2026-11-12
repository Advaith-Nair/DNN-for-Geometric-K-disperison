"""
Unit tests for point sets and conflict graphs.

Most tests use a small one-dimensional instance (the "chain") whose pairwise
distances and conflict graphs can be worked out by hand.
"""

import numpy as np
from points import make_point_set, random_uniform_points, random_clustered_points
from conflict_graph import build_conflict_graph


def make_chain_example():
    """Six collinear points A-F at positions 0, 2, 5, 7, 10, 12.

    Consecutive gaps are 2, 3, 2, 3, 2; every non-consecutive pair is at
    distance at least 5.
    """
    positions = np.array([[0.0], [2.0], [5.0], [7.0], [10.0], [12.0]])
    return make_point_set(positions)


def test_diameter():
    """The diameter of the chain is the distance from A to F."""
    ps = make_chain_example()
    assert abs(ps.diameter() - 12.0) < 1e-9


def test_min_positive_gap():
    """The smallest positive distance in the chain is 2."""
    ps = make_chain_example()
    assert abs(ps.min_positive_gap() - 2.0) < 1e-9


def test_conflict_graph_is_path_at_x4():
    """At x = 4 only consecutive points conflict, giving a path on six vertices."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=4.0)
    expected_edges = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)}
    assert set(g.edges) == expected_edges, f"got {set(g.edges)}"
    assert g.num_edges() == 5


def test_conflict_graph_is_empty_below_min_distance():
    """At x = 1 no pair is closer than x, so the graph has no edges."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=1.0)
    assert g.num_edges() == 0


def test_conflict_graph_is_complete_above_diameter():
    """At x = 13, larger than the diameter, every pair conflicts."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=13.0)
    n = ps.n
    assert g.num_edges() == n * (n - 1) // 2


def test_independent_set_check():
    """On the path at x = 4, {A, C, E} is independent and {A, B, C} is not."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=4.0)
    assert g.is_independent_set([0, 2, 4]) is True
    assert g.is_independent_set([0, 1, 2]) is False


def test_pair_at_distance_x_is_not_an_edge():
    """Edges use a strict inequality: A and B (distance 2) do not conflict at x = 2."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=2.0)
    assert not g.has_edge(0, 1)


def test_degree_and_neighbors():
    """Degrees and neighbourhoods on the path at x = 4."""
    ps = make_chain_example()
    g = build_conflict_graph(ps, x=4.0)
    assert g.degree(1) == 2
    assert set(g.neighbors(1).tolist()) == {0, 2}
    assert g.degree(0) == 1


def test_edge_count_is_monotone_in_x():
    """Increasing x never removes edges; binary search relies on this."""
    ps = random_uniform_points(30, dim=2, seed=42)
    xs = np.linspace(0.0, ps.diameter(), 15)
    prev_edges = -1
    for x in xs:
        g = build_conflict_graph(ps, x)
        assert g.num_edges() >= prev_edges
        prev_edges = g.num_edges()


def test_distance_matrix_is_not_modified():
    """Building conflict graphs leaves the stored distance matrix unchanged."""
    ps = random_uniform_points(20, dim=2, seed=1)
    dist_before = ps.dist.copy()
    for x in [0.1, 0.3, 0.5, 0.9]:
        build_conflict_graph(ps, x)
    assert np.array_equal(dist_before, ps.dist)


def test_random_generators():
    """Both generators return symmetric distance matrices with a zero diagonal."""
    ps1 = random_uniform_points(50, dim=2, seed=0)
    ps2 = random_clustered_points(50, dim=2, n_clusters=4, seed=0)
    for ps in (ps1, ps2):
        assert ps.n == 50
        assert ps.dist.shape == (50, 50)
        assert np.allclose(np.diag(ps.dist), 0.0)
        assert np.allclose(ps.dist, ps.dist.T)