"""
Cheap, network-free certificates that an independent set of size k CANNOT exist.

Idea: if the vertices can be covered by c cliques, an independent set can
contain at most one vertex per clique, so alpha(G) <= c. If c < k, then
"no independent set of size k" is PROVEN, and the dNN never needs to run.

This only ever certifies "infeasible" (it never claims "feasible"), and is
a valid upper bound, so it cannot introduce wrong answers.
"""
from __future__ import annotations
import numpy as np


def greedy_clique_cover_bound(adj: np.ndarray) -> int:
    """Number of cliques in a greedy clique cover = an upper bound on alpha(G)."""
    n = adj.shape[0]
    unassigned = np.ones(n, dtype=bool)
    # process low-degree vertices first (they have the fewest clique options)
    order = np.argsort(adj.sum(axis=1))
    cliques = 0
    for v in order:
        if not unassigned[v]:
            continue
        clique = [int(v)]
        unassigned[v] = False
        # candidates: unassigned vertices adjacent to every member so far
        cand = unassigned & adj[v]
        while cand.any():
            u = int(np.flatnonzero(cand)[0])
            clique.append(u)
            unassigned[u] = False
            cand = cand & adj[u] & unassigned
        cliques += 1
    return cliques


def certified_infeasible(adj: np.ndarray, k: int) -> bool:
    """True only if it is PROVEN that no independent set of size k exists."""
    return greedy_clique_cover_bound(adj) < k
