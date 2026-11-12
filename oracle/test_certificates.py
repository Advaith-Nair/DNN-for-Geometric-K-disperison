"""
Tests for the clique-cover bound.
"""

import sys, os, itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from geometry.points import make_point_set, random_uniform_points
from geometry.conflict_graph import build_conflict_graph
from certificates import greedy_clique_cover_bound, certified_infeasible


def true_alpha(g):
    """Independence number by enumeration (small graphs only)."""
    for s in range(g.n, 0, -1):
        for c in itertools.combinations(range(g.n), s):
            if g.is_independent_set(c):
                return s
    return 0


def test_path_is_not_certified_infeasible():
    """On the path with six vertices (alpha = 3), k = 3 is not rejected."""
    ps = make_point_set(np.array([[0.0], [2.0], [5.0], [7.0], [10.0], [12.0]]))
    g = build_conflict_graph(ps, 4.0)
    assert greedy_clique_cover_bound(g.adj) >= 3
    assert not certified_infeasible(g.adj, 3)


def test_complete_graph():
    """A complete graph is covered by one clique, so k = 2 is rejected."""
    ps = make_point_set(np.array([[0.0], [1.0], [2.0], [3.0]]))
    g = build_conflict_graph(ps, 10.0)
    assert greedy_clique_cover_bound(g.adj) == 1
    assert certified_infeasible(g.adj, 2)


def test_bound_is_valid_on_random_graphs():
    """The bound is never smaller than the independence number."""
    bad = 0
    for seed in range(30):
        ps = random_uniform_points(11, seed=seed)
        for x in np.linspace(0.05, ps.diameter(), 8):
            g = build_conflict_graph(ps, x)
            bad += greedy_clique_cover_bound(g.adj) < true_alpha(g)
    assert bad == 0, f"bound below the independence number in {bad} graphs"