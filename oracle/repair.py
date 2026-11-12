"""
Post-processing of the rounded dNN output.

The vertex set obtained by rounding theta may contain adjacent vertices and
may not be maximal. It is processed in three stages:

1. Conflict repair. For each adjacent pair in the set, try to replace one
   endpoint by a neighbour that has no other neighbour in the set (a swap,
   following Algorithm 2 of Alkhouri et al.); if no swap exists, delete the
   endpoint involved in more conflicts.
2. Greedy augmentation. Add vertices with no neighbour in the set until the
   set is maximal.
3. 2-improvement local search (Andrade, Resende and Werneck, 2012), as also
   used in Section 5.1 of Alkhouri et al.
"""

from __future__ import annotations

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.conflict_graph import ConflictGraph


def _conflicts_within(candidate: set[int], g: ConflictGraph) -> list[tuple[int, int]]:
    """Return all adjacent pairs inside the candidate set."""
    conflicts = []
    cand_list = list(candidate)
    for i in range(len(cand_list)):
        for j in range(i + 1, len(cand_list)):
            a, b = cand_list[i], cand_list[j]
            if g.has_edge(a, b):
                conflicts.append((a, b))
    return conflicts


def _find_1tight_replacement(node: int, candidate: set[int], g: ConflictGraph) -> int | None:
    """Find a neighbour w of `node` that can replace it.

    w must be outside the candidate set and have no neighbour in
    candidate minus {node}. Returns None if no such vertex exists.
    """
    others = candidate - {node}
    for w in g.neighbors(node):
        w = int(w)
        if w in candidate:
            continue
        if all(not g.has_edge(w, other) for other in others):
            return w
    return None


def _greedy_augment(cand: set[int], g: ConflictGraph) -> set[int]:
    """Add vertices with no neighbour in the set until it is maximal."""
    added = True
    while added:
        added = False
        for v in range(g.n):
            if v in cand:
                continue
            if all(not g.has_edge(v, u) for u in cand):
                cand.add(v)
                added = True
    return cand


def _two_improvement(cand: set[int], g: ConflictGraph) -> set[int]:
    """2-improvement local search for independent sets.

    For a selected vertex v, consider the unselected vertices whose only
    selected neighbour is v (1-tight vertices). If two of them are not
    adjacent, replace v by both, which enlarges the set by one. The move is
    repeated, followed by greedy augmentation, until no improving move exists.
    """
    adj = g.adj
    improved = True
    while improved:
        improved = False
        in_set = list(cand)
        if not in_set:
            break
        # Number of selected neighbours of each vertex.
        tight = adj[:, in_set].sum(axis=1)
        for v in in_set:
            ones = [u for u in g.neighbors(v) if u not in cand and tight[u] == 1]
            for i in range(len(ones)):
                for j in range(i + 1, len(ones)):
                    a, b = int(ones[i]), int(ones[j])
                    if not adj[a, b]:
                        cand.discard(v)
                        cand.update((a, b))
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            cand = _greedy_augment(cand, g)
    return cand


def repair(candidate: list[int], g: ConflictGraph, max_iters: int = 1000,
           local_search: bool = True) -> list[int]:
    """Turn a rounded vertex set into a maximal independent set.

    Args:
        candidate: Vertices selected by rounding.
        g: Conflict graph.
        max_iters: Maximum number of conflict-repair steps.
        local_search: Whether to run 2-improvement after augmentation.

    Returns:
        A sorted list of vertices forming an independent set.
    """
    cand = set(candidate)

    for _ in range(max_iters):
        conflicts = _conflicts_within(cand, g)
        if not conflicts:
            break

        a, b = conflicts[0]
        replaced = False

        for node in (a, b):
            w = _find_1tight_replacement(node, cand, g)
            if w is not None:
                cand.discard(node)
                cand.add(w)
                replaced = True
                break

        if not replaced:
            # Remove the endpoint that occurs in more conflicting pairs.
            counts = {}
            for (x, y) in conflicts:
                counts[x] = counts.get(x, 0) + 1
                counts[y] = counts.get(y, 0) + 1
            to_remove = max((a, b), key=lambda node: counts.get(node, 0))
            cand.discard(to_remove)

    cand = _greedy_augment(cand, g)
    if local_search:
        cand = _two_improvement(cand, g)
    return sorted(cand)