"""
Point sets and pairwise distances.

A problem instance is a set of n points in R^d. The full Euclidean distance
matrix is computed once when the instance is created and is reused by every
later step (conflict-graph construction, the solvers and the evaluation),
so coordinates are never processed again after construction.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PointSet:
    """A set of n points in R^d together with its pairwise distance matrix.

    Attributes:
        coords: Array of shape (n, d) holding the point coordinates.
        dist: Array of shape (n, n) holding Euclidean distances. The matrix
            is symmetric with a zero diagonal.
    """

    coords: np.ndarray
    dist: np.ndarray

    @property
    def n(self) -> int:
        """Number of points."""
        return self.coords.shape[0]

    @property
    def dim(self) -> int:
        """Dimension of the ambient space."""
        return self.coords.shape[1]

    def diameter(self) -> float:
        """Return the largest pairwise distance."""
        return float(self.dist.max())


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Return the (n, n) Euclidean distance matrix of the given coordinates."""
    diffs = coords[:, None, :] - coords[None, :, :]
    sq_dists = np.sum(diffs ** 2, axis=-1)
    # Guard against small negative values caused by rounding.
    sq_dists = np.clip(sq_dists, 0.0, None)
    return np.sqrt(sq_dists)


def make_point_set(coords: np.ndarray) -> PointSet:
    """Create a PointSet from an (n, d) coordinate array."""
    coords = np.asarray(coords, dtype=float)
    dist = _pairwise_distances(coords)
    return PointSet(coords=coords, dist=dist)


def random_uniform_points(n: int, dim: int = 2, low: float = 0.0, high: float = 1.0,
                           seed: int | None = None) -> PointSet:
    """Sample n points uniformly at random from the box [low, high]^dim.

    Args:
        n: Number of points.
        dim: Dimension of the space.
        low, high: Bounds of the sampling box.
        seed: Seed for the random number generator.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(low, high, size=(n, dim))
    return make_point_set(coords)


def random_clustered_points(n: int, dim: int = 2, n_clusters: int = 5,
                             cluster_std: float = 0.05, low: float = 0.0,
                             high: float = 1.0, seed: int | None = None) -> PointSet:
    """Sample n points from a mixture of isotropic Gaussian clusters.

    Cluster centres are drawn uniformly from [low, high]^dim, each point is
    assigned to a centre uniformly at random, and the resulting coordinates
    are clipped to the box.

    Args:
        n: Number of points.
        dim: Dimension of the space.
        n_clusters: Number of Gaussian clusters.
        cluster_std: Standard deviation of each cluster.
        low, high: Bounds of the box.
        seed: Seed for the random number generator.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(low, high, size=(n_clusters, dim))
    assignments = rng.integers(0, n_clusters, size=n)
    coords = centers[assignments] + rng.normal(0.0, cluster_std, size=(n, dim))
    coords = np.clip(coords, low, high)
    return make_point_set(coords)