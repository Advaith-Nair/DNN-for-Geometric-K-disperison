# Gurobi baseline: installation and comparison

This project now includes a direct mixed-integer linear programming (MILP)
formulation of Euclidean k-dispersion in `baselines/gurobi_solver.py`.

## 1. Install in the same Python environment as the project

Open PowerShell in the project root (the folder containing `geometry`,
`oracle`, `binary_search`, `baselines`, and `eval`) and run:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy torch pytest
python -m pip install -r requirements-gurobi.txt
```

Use a 64-bit Python installation. The `python -m pip` form is intentional:
it installs Gurobi into the interpreter that will run the project. Verify it:

```powershell
python -c "import gurobipy as gp; print(gp.gurobi.version())"
python eval/check_gurobi_setup.py
```

The PyPI `gurobipy` package includes a free size-limited license for small
models. Gurobi currently documents limits of 2,000 variables and 2,000 linear
constraints for this license. The present MILP has `n + 1` variables and
`n*(n-1)/2 + 1` linear constraints. Thus, the intended `n <= 60` experiments
fit within those size limits (at n=60: 61 variables and 1,771 constraints).
For larger experiments, or to avoid the size cap, activate a full academic
license.

## 2. Free academic license (recommended for unrestricted project work)

Students at recognized degree-granting institutions can request a free
academic license for research/coursework. Create an Academic account on
Gurobi's website, connect to the university network (or a VPN that routes
through it), and request an Academic Named-User license in the User Portal.
Run the `grbgetkey` command shown in the portal to install the license file.
The Named-User license is tied to one user and one machine, but after
activation it can be used offline. If you are using containers or multiple
machines, check the Academic WLS option instead; WLS needs internet access
while solving and has a shorter renewal period.

Important: `pip install gurobipy` installs the Python interface but does not
install the `grbgetkey` license utility. Follow Gurobi's current license-tool
instructions or use its full/conda installation when you need to activate a
Named-User license.

## 3. Smoke test

From the project root:

```powershell
python -c "import gurobipy as gp; m=gp.Model(); x=m.addVar(); m.setObjective(x, gp.GRB.MAXIMIZE); m.optimize(); print('status=', m.Status, 'obj=', m.ObjVal); m.dispose()"
```

If the package imports but model creation fails, inspect the license error.
For an academic-domain error, connect to the campus network or its VPN before
requesting/activating the license. Do not put `gurobi.lic` in source control.

## 4. Run the comparison

The existing exact recursive solver remains the ground-truth reference on
small instances. Add Gurobi with a fixed 60-second cap and a zero requested
MIP gap:

```powershell
python eval/run_eval.py --with-gurobi --gurobi-time-limit 60 --gurobi-mip-gap 0 `
  --gurobi-threads 1 --restarts 60 --certify --ns 15 20 --ks 3 4 5 --seeds 10 `
  --seed-start 500 --dists uniform clustered --out eval/results_gurobi_small.csv
```

For medium instances, exact recursion can become the bottleneck. Skip that
reference and use Gurobi's incumbent and best bound as the comparison interval:

```powershell
python eval/run_eval.py --with-gurobi --skip-exact --gurobi-time-limit 60 `
  --gurobi-mip-gap 0 --gurobi-threads 1 --restarts 60 --certify `
  --ns 30 40 50 60 --ks 6 8 10 --seeds 1 --seed-start 1000 --dists uniform `
  --out eval/results_gurobi_medium.csv
```

The medium command is only a one-distribution pilot. Expand to more seeds and
clustered points after checking runtime and solver gaps. Use a new seed range
for the final held-out results. Do not tune the dNN hyperparameters on those
seeds. Keep the Gurobi time limit, MIP gap, and thread count identical across
all instances in a comparison batch.

## 5. What the output means

The CSV adds:

- `gurobi_radius`: the minimum pairwise distance recomputed from Gurobi's
  returned set (the actual feasible solution quality).
- `gurobi_obj`: Gurobi's incumbent objective value.
- `gurobi_bound`: Gurobi's best upper bound for this maximization problem.
- `gurobi_gap`: the solver-reported relative MIP gap.
- `gurobi_status`: e.g. `OPTIMAL` or `TIME_LIMIT`.
- `gurobi_runtime` and `gurobi_wall_time`: optimizer time and end-to-end
  Python wall time, respectively.
- `dnn_gap_to_gurobi_bound`: a conservative certificate for the dNN result
  when exact recursion is skipped. Since dNN returns a feasible set and
  Gurobi's `ObjBound` is an upper bound, this is
  `max(0, bound - dnn_radius) / max(abs(bound), 1e-12)`.

Interpretation:

- If Gurobi is `OPTIMAL`, its value is the proven optimum (up to solver
  tolerances), and compare all returned radii against it.
- If Gurobi hits `TIME_LIMIT` with an incumbent, the incumbent is a lower
  bound on the optimum and `gurobi_bound` is an upper bound. Report both and
  the MIP gap; do not call the incumbent the proven optimum.
- If it hits the time limit without an incumbent, there is no Gurobi witness
  to compare, but its bound is still useful when available.

For the report, present the small-instance exact-match comparison separately
from medium-instance time-limited comparisons. Report the distribution of
Gurobi statuses and gaps, not just the mean runtime.
