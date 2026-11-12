"""Network-free tests for the infeasibility certificate (needs only numpy)."""
import sys, os, itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from geometry.points import make_point_set, random_uniform_points
from geometry.conflict_graph import build_conflict_graph
from certificates import greedy_clique_cover_bound, certified_infeasible


def true_alpha(g):
    for s in range(g.n, 0, -1):
        for c in itertools.combinations(range(g.n), s):
            if g.is_independent_set(c):
                return s
    return 0


def test_chain_example():
    ps = make_point_set(np.array([[0.0], [2.0], [5.0], [7.0], [10.0], [12.0]]))
    g = build_conflict_graph(ps, 4.0)           # path of 6 nodes, alpha = 3
    assert greedy_clique_cover_bound(g.adj) >= 3
    assert not certified_infeasible(g.adj, 3)   # k=3 IS feasible: must NOT be certified away
    print("PASS: chain k=3 is not wrongly certified infeasible")


def test_complete_graph():
    ps = make_point_set(np.array([[0.0], [1.0], [2.0], [3.0]]))
    g = build_conflict_graph(ps, 10.0)          # complete graph, alpha = 1
    assert greedy_clique_cover_bound(g.adj) == 1
    assert certified_infeasible(g.adj, 2)
    print("PASS: complete graph certifies k=2 infeasible")


def test_bound_never_below_true_alpha():
    bad = 0
    total = 0
    for seed in range(30):
        ps = random_uniform_points(11, seed=seed)
        for x in np.linspace(0.05, ps.diameter(), 8):
            g = build_conflict_graph(ps, x)
            total += 1
            bad += greedy_clique_cover_bound(g.adj) < true_alpha(g)
    assert bad == 0, f"bound fell below the true independence number in {bad} graphs"
    print(f"PASS: bound >= true independence number in all {total} random graphs")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} certificate tests passed.")
