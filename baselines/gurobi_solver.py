"""Direct MILP formulation of the Euclidean k-dispersion problem.

Given pairwise distances d[i,j], choose exactly k points and maximize the
minimum distance between any selected pair:

    maximize    r
    subject to  sum_i y[i] = k
                r <= d[i,j] + M[i,j] * (2 - y[i] - y[j])  for i < j
                0 <= r <= diameter
                y[i] in {0, 1}

Here y[i] indicates whether point i is selected.  We use the tight valid
coefficient M[i,j] = diameter - d[i,j].  If both endpoints are selected,
the constraint reduces to r <= d[i,j].  Otherwise its right-hand side is
at least the diameter, so it is inactive because r is bounded by diameter.

This model is independent of the dNN/conflict-graph pipeline and therefore
serves as a direct mathematical-programming comparison baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import time
from typing import Any

import numpy as np

try:
    from geometry.points import PointSet
except ImportError:  # allow running from a module's own directory
    from points import PointSet


@dataclass
class GurobiConfig:
    """Gurobi solver settings used in the benchmark."""

    time_limit: float = 60.0
    mip_gap: float = 0.0
    threads: int = 1
    output_flag: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.time_limit <= 0:
            raise ValueError("time_limit must be > 0 seconds")
        if self.mip_gap < 0:
            raise ValueError("mip_gap must be >= 0")
        if self.threads < 1:
            raise ValueError("threads must be >= 1")


@dataclass
class GurobiResult:
    """Solver output plus independently measured witness quality."""

    radius: float | None
    witness: tuple[int, ...] | None
    status: int
    status_name: str
    is_optimal: bool
    runtime: float
    wall_time: float
    obj_val: float | None
    obj_bound: float | None
    mip_gap: float | None
    node_count: float | None
    num_vars: int
    num_constraints: int


def _load_gurobi():
    """Import Gurobi lazily so the rest of the project works without it."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError(
            "Gurobi's Python package is not installed. Activate your project "
            "environment and run: python -m pip install gurobipy"
        ) from exc
    return gp, GRB


def gurobi_available() -> bool:
    """Return whether gurobipy can be imported (does not test the license)."""
    try:
        import gurobipy  # noqa: F401
        return True
    except ImportError:
        return False


def _status_name(status: int, GRB: Any) -> str:
    names = [
        "LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED",
        "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT",
        "SOLUTION_LIMIT", "INTERRUPTED", "NUMERIC", "SUBOPTIMAL",
        "INPROGRESS", "USER_OBJ_LIMIT", "WORK_LIMIT", "MEM_LIMIT",
    ]
    for name in names:
        value = getattr(GRB, name, None)
        if value is not None and value == status:
            return name
    return f"STATUS_{status}"


def solve_k_dispersion_gurobi(
    points: PointSet,
    k: int,
    cfg: GurobiConfig | None = None,
) -> GurobiResult:
    """Solve max-min k-dispersion with a direct Gurobi MILP.

    A feasible incumbent is returned even when the time limit is reached.
    ``radius`` is recomputed from the extracted point set rather than copied
    from Gurobi's objective, so the reported solution quality is independent
    of the solver's floating-point objective value.
    """
    if cfg is None:
        cfg = GurobiConfig()
    if k < 2:
        raise ValueError("k must be >= 2; k=1 has no pairwise distance objective")
    if k > points.n:
        raise ValueError(f"k={k} exceeds n={points.n}")

    gp, GRB = _load_gurobi()
    n = points.n
    dist = np.asarray(points.dist, dtype=float)
    diameter = float(np.max(dist))
    start = time.perf_counter()

    model = gp.Model("k_dispersion_direct_milp")
    try:
        model.Params.OutputFlag = int(cfg.output_flag)
        model.Params.TimeLimit = float(cfg.time_limit)
        model.Params.MIPGap = float(cfg.mip_gap)
        model.Params.Threads = int(cfg.threads)
        model.Params.Seed = int(cfg.seed)

        selected = model.addVars(n, vtype=GRB.BINARY, name="selected")
        radius = model.addVar(lb=0.0, ub=diameter, vtype=GRB.CONTINUOUS, name="radius")

        model.addConstr(gp.quicksum(selected[i] for i in range(n)) == k,
                        name="choose_exactly_k")

        # A pair only constrains the radius when BOTH endpoints are selected.
        # M_ij = diameter - d_ij is sufficient and tighter than a single
        # global big-M constant because radius is bounded above by diameter.
        for i in range(n):
            for j in range(i + 1, n):
                dij = float(dist[i, j])
                big_m = max(0.0, diameter - dij)
                model.addConstr(
                    radius <= dij + big_m * (2 - selected[i] - selected[j]),
                    name=f"pair_spacing_{i}_{j}",
                )

        model.setObjective(radius, GRB.MAXIMIZE)
        model.optimize()

        status = int(model.Status)
        sol_count = int(model.SolCount)
        witness: tuple[int, ...] | None = None
        achieved_radius: float | None = None
        obj_val: float | None = None
        mip_gap: float | None = None

        if sol_count > 0:
            chosen = tuple(i for i in range(n) if selected[i].X >= 0.5)
            if len(chosen) == k:
                witness = chosen
                achieved_radius = float(min(
                    dist[i, j] for i, j in itertools.combinations(chosen, 2)
                ))
            obj_val = float(model.ObjVal)
            try:
                mip_gap = float(model.MIPGap)
            except (AttributeError, gp.GurobiError):
                mip_gap = None

        try:
            obj_bound = float(model.ObjBound)
        except (AttributeError, gp.GurobiError):
            obj_bound = None

        try:
            node_count = float(model.NodeCount)
        except (AttributeError, gp.GurobiError):
            node_count = None

        runtime = float(model.Runtime)
        num_vars = int(model.NumVars)
        num_constraints = int(model.NumConstrs)
        return GurobiResult(
            radius=achieved_radius,
            witness=witness,
            status=status,
            status_name=_status_name(status, GRB),
            is_optimal=(status == GRB.OPTIMAL),
            runtime=runtime,
            wall_time=time.perf_counter() - start,
            obj_val=obj_val,
            obj_bound=obj_bound,
            mip_gap=mip_gap,
            node_count=node_count,
            num_vars=num_vars,
            num_constraints=num_constraints,
        )
    finally:
        model.dispose()
