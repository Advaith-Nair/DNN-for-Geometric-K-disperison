# DNN for Geometric k-Dispersion

Major project, Semester 7, NITK Surathkal.
Advaith Nair, Jessica Mariam Chandy, Sampreet Vasishta, guided by Professor Manjanna B.

Given n points in the plane, pick k of them so that the closest pair among the
picked points is as far apart as possible. We call that smallest distance x.
This is the max-min k-dispersion problem and it is NP-hard.

We solve it with the dataless neural network (dNN) from Alkhouri, Atia and
Velasquez, "A differentiable approach to the maximum independent set problem
using dataless neural networks" (Neural Networks, 2022). The dNN in that paper
finds independent sets in graphs, so most of this project is about turning
k-dispersion into a sequence of independent set questions.

## How it works

For a guessed distance x, join every two points that are closer than x. Call
this the conflict graph. Picking k points that are all at least x apart is the
same as finding an independent set of size k in this graph.

So the solver is a binary search over x:

1. Collect all pairwise distances. The optimal x is always one of them.
2. Take the middle candidate x and build the conflict graph.
3. Ask the dNN for an independent set of size k.
4. If it finds one, move to larger x. If not, move to smaller x.

The dNN has one value θ in [0, 1] per point. We minimise the loss from Eq. 7
of the paper with Adam and round θ at 0.5. Things we changed or added on top
of the paper:

- Target is -k/2 instead of -n/2, since we only need k points, not the
  largest possible set.
- Leaky ReLU (annealed towards ReLU) so dials at 0 don't get stuck, plus a
  small term pushing θ towards 0 or 1.
- Restarts that alternate between degree-based starts (Eq. 12 of the paper)
  and random starts.
- After rounding: remove conflicts, greedily add any point that fits, then
  run the 2-improvement local search (swap one chosen point for two).
- A clique-cover bound that proves some x values infeasible without running
  the dNN at all.

## Layout

```
geometry/        point generators, distance matrix, conflict graph
oracle/          dNN loss and training, restarts, repair + local search, clique bound
binary_search/   search over pairwise distances
baselines/       exact recursive solver, Gurobi MILP, greedy farthest-point
eval/            experiment runner, analysis/plots, result CSVs
```

Each folder has its own `test_*.py`.

## Setup

Python 3.10, 64-bit.

```
pip install -r requirements.txt
```

Gurobi is only needed for the Gurobi baseline; the rest runs without it.

The pip version of Gurobi comes with a size-limited licence (2000 variables /
2000 constraints). Our MILP has n(n-1)/2 + 1 constraints, so that covers
n up to about 63. Bigger instances need an academic licence.

Check that Gurobi works with `python eval/check_gurobi_setup.py`.

## Running

Tests (run inside each folder):

```
cd geometry
python -m pytest -q
```

Same for `oracle`, `baselines` and `binary_search`. For `eval`, run
`python -m pytest -q eval` from the root.

Small instances, checked against the exact solver and Gurobi:

```
python eval/run_eval.py --with-gurobi --restarts 60 --certify --ns 15 20 --ks 3 4 5 --seeds 10 --seed-start 3100 --dists uniform clustered --out eval/final_small.csv
```

Medium instances against Gurobi:

```
python eval/run_eval.py --with-gurobi --skip-exact --independent-sizes --gurobi-time-limit 60 --gurobi-threads 1 --restarts 60 --certify --ns 30 40 50 60 --ks 6 8 10 --seeds 5 --seed-start 3000 --dists uniform clustered --out eval/final_full.csv
```

Useful flags for ablations: `--no-local-search`, `--max-steps 0` (skip
training, use the initial θ directly), `--target-mode k2` (the paper's
enhanced loss, Eq. 10).

Plots and summary tables:

```
python eval/analyze_results.py --input eval/final_full.csv
python eval/compare_runs.py full=eval/final_full.csv no_ls=eval/final_no_ls.csv untrained=eval/final_untrained.csv
```

`analyze_results.py` writes into `eval/analysis_<csv name>/`. Every plot is
saved as a 300 dpi PNG and as a PDF (use the PDFs in LaTeX).

## Results

Uniform and clustered points, fresh seeds not used during tuning. Every
returned set is re-checked to be valid.

Small, n = 15-20, k = 3-5, 120 instances:

- dNN found the optimum on 120/120
- Gurobi and the exact solver agreed on all 120
- greedy was optimal on 33%, worst case 62% of optimum

Medium, n = 30-60, k = 6-10, 120 instances, Gurobi proved optimality on all:

| Version                         | Optimal | Mean ratio | Worst  | Time/instance |
|---------------------------------|---------|------------|--------|---------------|
| full (training + local search)  | 100%    | 1.0000     | 1.0000 | 65 s          |
| no local search                 | 75.0%   | 0.9908     | 0.8776 | 92 s          |
| no training                     | 96.7%   | 0.9993     | 0.9506 | 0.7 s         |
| greedy                          | 11.7%   | 0.8859     | 0.7153 | -             |

Large, n = 100-150, k = 10-15, 40 instances. Gurobi's free licence can't run
these, so the ratio is against the best answer any version found, not a
proven optimum:

| Version                      | Matched best | Mean ratio | Worst  | Time/instance |
|------------------------------|--------------|------------|--------|---------------|
| training + local search      | 60.0%        | 0.9936     | 0.9402 | 18.6 s        |
| no training + local search   | 72.5%        | 0.9932     | 0.9479 | 1.2 s         |
| training, no local search    | 7.5%         | 0.9483     | 0.8802 | 21.3 s        |
| no training, no local search | 15.0%        | 0.9647     | 0.9056 | 1.4 s         |
| greedy                       | 0%           | 0.8431     | 0.7006 | -             |

What we took from this:

- The local search step makes the biggest difference.
- Training the dNN helps a little on medium instances (it beat the untrained
  version on 4 instances and never lost) and is roughly a tie on large ones.
- Gurobi is much faster at these sizes (under 0.5 s). Speed wasn't the goal
  here.

## Limitations

- No optimality guarantee. If the dNN misses a feasible set, the search
  settles on a smaller x.
- Proven optima only up to n ≈ 63 with the free Gurobi licence.
- Only tested on random 2D points. The method only uses pairwise distances, so
  other dimensions should work without changes, but we haven't tried.

## Reference

I. R. Alkhouri, G. K. Atia, A. Velasquez. A differentiable approach to the
maximum independent set problem using dataless neural networks. Neural
Networks, 155:168-176, 2022. https://doi.org/10.1016/j.neunet.2022.08.008