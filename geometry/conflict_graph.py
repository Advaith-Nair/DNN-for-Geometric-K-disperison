"""
Conflict graphs for the decision version of k-dispersion.

For a threshold x, the conflict graph G_x has one vertex per point and an
edge between two points whenever their distance is strictly less than x.
A set of k points with pairwise distance at least x is therefore exactly an
independent set of size k in G_x. The graph is obtained by thresholding the
precomputed distance matrix of the PointSet.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# Support both package imports and running modules from inside this folder.
try:
    from .points import PointSet
except ImportError:
    from points import PointSet


@dataclass
class ConflictGraph:
    """Conflict graph G_x for a fixed threshold x.

    The graph is stored both as an adjacency matrix, for constant-time
    adjacency queries, and as an edge list, which is used to build the
    loss function of the neural oracle.

    Attributes:
        n: Number of vertices.
        x: Distance threshold that defines the edges.
        adj: Boolean (n, n) adjacency matrix, symmetric with a zero diagonal.
        edges: List of edges (i, j) with i < j.
    """

    n: int
    x: float
    adj: np.ndarray
    edges: list[tuple[int, int]]

    def degree(self, v: int) -> int:
        """Return the degree of vertex v."""
        return int(self.adj[v].sum())

    def degrees(self) -> np.ndarray:
        """Return the degree of every vertex."""
        return self.adj.sum(axis=1)

    def neighbors(self, v: int) -> np.ndarray:
        """Return the indices of the neighbours of vertex v."""
        return np.nonzero(self.adj[v])[0]

    def has_edge(self, u: int, v: int) -> bool:
        """Return True if u and v are adjacent."""
        return bool(self.adj[u, v])

    def num_edges(self) -> int:
        """Return the number of edges."""
        return len(self.edges)

    def is_independent_set(self, nodes) -> bool:
        """Return True if no two vertices in `nodes` are adjacent."""
        nodes = list(nodes)
        for a_idx in range(len(nodes)):
            for b_idx in range(a_idx + 1, len(nodes)):
                if self.adj[nodes[a_idx], nodes[b_idx]]:
                    return False
        return True


def build_conflict_graph(points: PointSet, x: float) -> ConflictGraph:
    """Build the conflict graph G_x of a point set.

    Vertices i and j are adjacent if and only if dist(i, j) < x. The
    inequality is strict, so two points at distance exactly x may both be
    selected, in line with the constraint "all pairwise distances >= x".

    Args:
        points: The point set, with precomputed distances.
        x: Distance threshold.
    """
    n = points.n
    adj = points.dist < x
    np.fill_diagonal(adj, False)

    # Read edges from the upper triangle so that each appears once.
    iu, ju = np.triu_indices(n, k=1)
    mask = adj[iu, ju]
    edges = list(zip(iu[mask].tolist(), ju[mask].tolist()))

    return ConflictGraph(n=n, x=x, adj=adj, edges=edges)