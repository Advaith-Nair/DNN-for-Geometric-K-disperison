"""
Clique-cover upper bound on the independence number.

If the vertices of a graph are partitioned into c cliques, an independent set
contains at most one vertex from each clique, so alpha(G) <= c. When a greedy
clique cover has fewer than k cliques, the graph has no independent set of
size k and the neural oracle does not need to be run. The bound can only
prove infeasibility; it never reports a feasible instance as infeasible.
"""

from __future__ import annotations

import numpy as np


def greedy_clique_cover_bound(adj: np.ndarray) -> int:
    """Return the number of cliques in a greedy clique cover.

    Vertices are processed in order of increasing degree. Each unassigned
    vertex starts a new clique, which is extended greedily by unassigned
    vertices adjacent to all of its members.
    """
    n = adj.shape[0]
    unassigned = np.ones(n, dtype=bool)
    order = np.argsort(adj.sum(axis=1))
    cliques = 0
    for v in order:
        if not unassigned[v]:
            continue
        clique = [int(v)]
        unassigned[v] = False
        # Unassigned vertices adjacent to every member of the clique.
        cand = unassigned & adj[v]
        while cand.any():
            u = int(np.flatnonzero(cand)[0])
            clique.append(u)
            unassigned[u] = False
            cand = cand & adj[u] & unassigned
        cliques += 1
    return cliques


def certified_infeasible(adj: np.ndarray, k: int) -> bool:
    """Return True if the clique-cover bound shows that alpha(G) < k."""
    return greedy_clique_cover_bound(adj) < k