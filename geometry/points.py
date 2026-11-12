"""
Point generation and pairwise distance caching.

Design principle: compute the full pairwise distance matrix ONCE per problem
instance, then reuse it for every binary-search iteration. Binary search will
call the conflict-graph builder many times (once per candidate radius x), and
we never want to recompute point-to-point distances from raw coordinates more
than once.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PointSet:
    """A set of n points in R^d, plus their cached pairwise distance matrix."""

    coords: np.ndarray          # shape (n, d)
    dist: np.ndarray            # shape (n, n), symmetric, zero diagonal

    @property
    def n(self) -> int:
        return self.coords.shape[0]

    @property
    def dim(self) -> int:
        return self.coords.shape[1]

    def diameter(self) -> float:
        """Largest pairwise distance. Used as the upper bound for binary search."""
        return float(self.dist.max())

    def min_positive_gap(self) -> float:
        """
        Smallest strictly-positive pairwise distance. Useful for setting a
        meaningful binary-search epsilon (no point resolving x finer than
        the data itself can distinguish).
        """
        off_diag = self.dist[~np.eye(self.n, dtype=bool)]
        positive = off_diag[off_diag > 0]
        if positive.size == 0:
            return 0.0
        return float(positive.min())


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Vectorized Euclidean pairwise distance matrix. O(n^2) space, computed once."""
    diffs = coords[:, None, :] - coords[None, :, :]   # (n, n, d)
    sq_dists = np.sum(diffs ** 2, axis=-1)
    # clip tiny negative values from floating point error before sqrt
    sq_dists = np.clip(sq_dists, 0.0, None)
    return np.sqrt(sq_dists)


def make_point_set(coords: np.ndarray) -> PointSet:
    """Wrap raw coordinates into a PointSet with cached distances."""
    coords = np.asarray(coords, dtype=float)
    dist = _pairwise_distances(coords)
    return PointSet(coords=coords, dist=dist)


def random_uniform_points(n: int, dim: int = 2, low: float = 0.0, high: float = 1.0,
                           seed: int | None = None) -> PointSet:
    """n points drawn uniformly at random from [low, high]^dim."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(low, high, size=(n, dim))
    return make_point_set(coords)


def random_clustered_points(n: int, dim: int = 2, n_clusters: int = 5,
                             cluster_std: float = 0.05, low: float = 0.0,
                             high: float = 1.0, seed: int | None = None) -> PointSet:
    """
    n points drawn from a mixture of n_clusters Gaussian blobs, cluster centers
    themselves uniform in [low, high]^dim. Useful for testing non-uniform,
    harder instances (points bunched together stress-test the conflict graph
    at small x).
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(low, high, size=(n_clusters, dim))
    assignments = rng.integers(0, n_clusters, size=n)
    coords = centers[assignments] + rng.normal(0.0, cluster_std, size=(n, dim))
    coords = np.clip(coords, low, high)
    return make_point_set(coords)
