"""
Repair the extracted candidate set when it still has internal conflicts
(edges between two 'selected' nodes) after thresholding theta.

THIS IS FIX #2: the paper's own Algorithm 2 pattern is swap-before-delete,
not delete-only. For every conflicting pair, first look for a "1-tight"
replacement -- a neighbor of the conflicting node whose only conflict
WITH THE CURRENT SET is the node it would replace. Only delete outright if
no such replacement exists.

This recovers solutions that a naive greedy-delete would throw away.
"""

from __future__ import annotations

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.conflict_graph import ConflictGraph


def _conflicts_within(candidate: set[int], g: ConflictGraph) -> list[tuple[int, int]]:
    """All conflicting (edge) pairs currently inside the candidate set."""
    conflicts = []
    cand_list = list(candidate)
    for i in range(len(cand_list)):
        for j in range(i + 1, len(cand_list)):
            a, b = cand_list[i], cand_list[j]
            if g.has_edge(a, b):
                conflicts.append((a, b))
    return conflicts


def _find_1tight_replacement(node: int, candidate: set[int], g: ConflictGraph) -> int | None:
    """
    Look for a neighbor `w` of `node` such that `w` is NOT already in the
    candidate set, and `w`'s only conflict with (candidate - {node}) would
    be resolved by swapping node -> w. Concretely: w must have zero edges
    into candidate minus {node}. If such w exists, swapping is strictly safe.
    """
    others = candidate - {node}
    for w in g.neighbors(node):
        w = int(w)
        if w in candidate:
            continue
        # check w has no edge into the rest of the candidate set
        if all(not g.has_edge(w, other) for other in others):
            return w
    return None


def _greedy_augment(cand: set[int], g: ConflictGraph) -> set[int]:
    """
    GAP FOUND DURING TESTING: the dNN often converges to a valid but
    UNDER-SIZED independent set (e.g. it finds {A, F} when {A, C, F} is
    also perfectly valid and available) -- extraction and conflict-repair
    alone never try to ADD more nodes, they only ever remove/swap.

    This pass greedily adds any currently-unselected node that has zero
    edges into the current candidate set, repeating until no more can be
    added. Cheap (this is just standard greedy MIS augmentation) and
    directly targets the "found a valid but shrinkable solution" failure
    mode we observed empirically.
    """
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
    """
    The "2-improvement" local search (Andrade, Resende & Werneck 2012) that
    the reference paper also applies after the dNN (Sec. 5.1).

    Take a selected node v. Look at the unselected nodes whose ONLY conflict
    with the current set is v (1-tight nodes w.r.t. v). If two of them are
    not in conflict with each other, swap v out and both of them in: the set
    grows by one and stays conflict-free. Repeat until no such move exists.
    """
    adj = g.adj
    improved = True
    while improved:
        improved = False
        in_set = list(cand)
        if not in_set:
            break
        # number of selected neighbours of every node
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
    """
    Iteratively resolve conflicts in `candidate` using swap-before-delete,
    THEN greedily augment with any compatible unselected nodes.

    Returns a conflict-free subset, generally as large as easily achievable
    from the given starting candidate. Conflict resolution never deletes
    more than necessary (swap is tried first); augmentation never removes
    anything, only adds.
    """
    cand = set(candidate)

    for _ in range(max_iters):
        conflicts = _conflicts_within(cand, g)
        if not conflicts:
            break

        # process one conflicting pair
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
            # no swap available for either endpoint -- delete the one that
            # appears in more conflicting pairs (paper's tie-break rule)
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
