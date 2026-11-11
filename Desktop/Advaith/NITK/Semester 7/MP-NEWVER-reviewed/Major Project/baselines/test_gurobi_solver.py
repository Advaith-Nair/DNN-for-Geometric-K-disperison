"""Tests for the direct Gurobi MILP (skipped when gurobipy is unavailable)."""

import importlib.util
import itertools

import numpy as np
import pytest

if importlib.util.find_spec("gurobipy") is None:
    pytest.skip("gurobipy is not installed", allow_module_level=True)

from geometry.points import make_point_set, random_uniform_points
from baselines.exact_solver import exact_k_dispersion
from baselines.gurobi_solver import GurobiConfig, solve_k_dispersion_gurobi


def _radius(points, witness):
    return min(points.dist[i, j] for i, j in itertools.combinations(witness, 2))


def test_gurobi_matches_exact_on_handcrafted_instance():
    points = make_point_set(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [3.0, 4.0]
    ]))
    k = 3
    exact = exact_k_dispersion(points, k)
    result = solve_k_dispersion_gurobi(
        points, k, GurobiConfig(time_limit=20, mip_gap=0, threads=1)
    )

    assert result.is_optimal
    assert result.witness is not None
    assert len(result.witness) == k
    assert len(set(result.witness)) == k
    assert result.radius == pytest.approx(exact.radius, rel=1e-7, abs=1e-8)
    assert result.radius == pytest.approx(_radius(points, result.witness))


def test_gurobi_matches_exact_on_seeded_random_instances():
    for seed, k in [(13, 3), (21, 4), (34, 5)]:
        points = random_uniform_points(n=9, dim=2, seed=seed)
        exact = exact_k_dispersion(points, k)
        result = solve_k_dispersion_gurobi(
            points, k, GurobiConfig(time_limit=20, mip_gap=0, threads=1)
        )
        assert result.is_optimal
        assert result.radius == pytest.approx(exact.radius, rel=1e-7, abs=1e-8)
