"""
Conflict graph construction for the k-dispersion -> MIS reduction.

Given a PointSet and a candidate spacing radius x, build the graph G_x where
two points are connected iff their distance is STRICTLY LESS than x (i.e.
they'd violate the "all pairwise distances >= x" requirement if both were
selected). An independent set of size k in G_x is exactly a set of k points
that are pairwise >= x apart.

This module only re-thresholds the cached distance matrix -- it never
recomputes geometry. That's what makes repeated calls during binary search
cheap.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

try:
    from .points import PointSet
except ImportError:
    from points import PointSet


@dataclass
class ConflictGraph:
    """
    Conflict graph at a fixed radius x, expressed as both an adjacency matrix
    and an edge list (both are useful downstream: the dNN oracle wants a
    compact edge list to build its loss function; repair logic wants fast
    O(1) adjacency lookups).
    """

    n: int
    x: float
    adj: np.ndarray          # (n, n) boolean, symmetric, zero diagonal
    edges: list[tuple[int, int]]   # i < j for every edge

    def degree(self, v: int) -> int:
        return int(self.adj[v].sum())

    def degrees(self) -> np.ndarray:
        return self.adj.sum(axis=1)

    def neighbors(self, v: int) -> np.ndarray:
        return np.nonzero(self.adj[v])[0]

    def has_edge(self, u: int, v: int) -> bool:
        return bool(self.adj[u, v])

    def num_edges(self) -> int:
        return len(self.edges)

    def is_independent_set(self, nodes) -> bool:
        """True iff no two nodes in `nodes` are connected."""
        nodes = list(nodes)
        for a_idx in range(len(nodes)):
            for b_idx in range(a_idx + 1, len(nodes)):
                if self.adj[nodes[a_idx], nodes[b_idx]]:
                    return False
        return True


def build_conflict_graph(points: PointSet, x: float) -> ConflictGraph:
    """
    Build G_x from the cached distance matrix. Edge (i, j) exists iff
    dist(i, j) < x. Strict inequality matches the problem definition: a pair
    at distance exactly x is fine (their spacing constraint is satisfied at
    exactly the boundary).
    """
    n = points.n
    adj = points.dist < x
    np.fill_diagonal(adj, False)

    # upper triangle only, to get each edge once
    iu, ju = np.triu_indices(n, k=1)
    mask = adj[iu, ju]
    edges = list(zip(iu[mask].tolist(), ju[mask].tolist()))

    return ConflictGraph(n=n, x=x, adj=adj, edges=edges)
