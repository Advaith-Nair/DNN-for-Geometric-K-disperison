"""
Check what the workstation offers before running the experiments:
CPU cores, PyTorch and CUDA (GPU), and whether the Gurobi licence is the
free size-limited one (models up to 2000 variables/constraints, i.e. n <= 63)
or a full licence that can solve the large instances.
"""

from __future__ import annotations

import os
import platform
import sys


def main() -> None:
    print("Python      :", sys.version.split()[0], "on", platform.system())
    print("CPU cores   :", os.cpu_count())

    try:
        import torch
        print("PyTorch     :", torch.__version__)
        if torch.cuda.is_available():
            print("CUDA        : available,", torch.cuda.device_count(), "GPU(s):",
                  ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))
        else:
            print("CUDA        : not available (the dNN will run on CPU)")
    except ImportError:
        print("PyTorch     : not installed")

    try:
        import gurobipy as gp
    except ImportError:
        print("Gurobi      : gurobipy not installed")
        return
    print("Gurobi      :", ".".join(map(str, gp.gurobi.version())))
    try:
        env = gp.Env(params={"OutputFlag": 0})
        model = gp.Model(env=env)
        x = model.addVars(2100)           # more than the size-limited licence allows
        model.setObjective(x.sum())
        model.optimize()
        print("Licence     : FULL - large instances (n = 100-300) can be solved with Gurobi")
    except gp.GurobiError as exc:
        if "size-limited" in str(exc):
            print("Licence     : SIZE-LIMITED (pip licence) - Gurobi only up to n = 63;")
            print("              run the large instances without --with-gurobi")
        else:
            print("Licence     : problem -", exc)


if __name__ == "__main__":
    main()