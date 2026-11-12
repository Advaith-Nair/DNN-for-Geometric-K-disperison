"""
Check the Gurobi installation: import gurobipy, confirm that a licence is
available and solve a five-point instance to proven optimality.
"""

from __future__ import annotations

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, os.path.join(ROOT, "baselines")):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import gurobipy as gp
except ImportError as exc:
    raise SystemExit(
        "gurobipy is not installed. Run: python -m pip install -r requirements.txt"
    ) from exc

from geometry.points import make_point_set
from gurobi_solver import GurobiConfig, solve_k_dispersion_gurobi


def main() -> None:
    """Solve a small test instance and print the solver output."""
    print("Gurobi version:", ".".join(map(str, gp.gurobi.version())))
    points = make_point_set(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [3.0, 4.0]
    ]))
    result = solve_k_dispersion_gurobi(
        points, k=3,
        cfg=GurobiConfig(time_limit=15.0, mip_gap=0.0, threads=1, output_flag=1),
    )
    print("Status:", result.status_name)
    print("Witness:", result.witness)
    print("Achieved radius:", result.radius)
    print("Best upper bound:", result.obj_bound)
    print("MIP gap:", result.mip_gap)
    print("Runtime:", result.runtime, "seconds")
    if not result.is_optimal or result.witness is None:
        raise SystemExit("Gurobi ran, but the test instance was not solved to proven optimality.")
    print("Gurobi installation and license check passed.")


if __name__ == "__main__":
    main()