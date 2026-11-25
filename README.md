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
same as finding an independent set of size k in this graph. This link
between dispersion and independent sets is classical (Erkut 1990;
Pisinger 2006; Sayyady and Fathi 2016); what is new here is answering each
independent set question with a dataless neural network.

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
- Leaky ReLU (annealed towards ReLU) so values below 0.5 still get a
  gradient, plus a small term pushing θ towards 0 or 1.
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
eval/            experiment runner, analysis and comparison scripts, results
```

Each folder has its own `test_*.py`.

The exact solver implements Algorithm A(k) of Chen and Dumitrescu (2025),
with brute force for the k <= 3 base cases. The greedy baseline is the
farthest-point heuristic of Ravi et al. (1994).

The Gurobi baseline is the MILP of Kuby (1987), exactly as stated by Sayah
and Irnich (2017), Eq. (4):

```
maximize    d
subject to  sum_i x_i = k
            d <= M (2 - x_i - x_j) + d_ij    for all i < j
            x_i in {0, 1},  d >= 0
```

x_i says whether point i is picked, d is the smallest distance between picked
points, and M is the largest pairwise distance. If both i and j are picked,
the constraint becomes d <= d_ij; otherwise it does nothing. The model has
n + 1 variables and n(n-1)/2 + 1 constraints.

Result files in `eval/`:

| File | Experiment |
|---|---|
| `final_small.csv` | small instances, full method, vs exact solver and Gurobi |
| `final_full.csv` | medium instances, full method, vs Gurobi |
| `final_no_ls.csv` | medium, without local search |
| `final_untrained.csv` | medium, without training |
| `final_large_trained_ls.csv` | large, full method |
| `final_large_untrained_ls.csv` | large, without training |
| `final_large_trained.csv` | large, without local search |
| `final_large_untrained.csv` | large, without training or local search |

`analysis_final_*/` holds the summary tables and plots for the four
small/medium CSVs.

## Setup

Python 3.10 or newer, 64-bit. The results below were produced with
Python 3.10, PyTorch 2.10 (CPU) and Gurobi 13.0.3.

Gurobi is only needed for the Gurobi baseline; the rest runs without it.
The pip version of Gurobi comes with a size-limited licence (2000 variables /
2000 constraints), so no licence setup is needed. Our MILP has
n(n-1)/2 + 1 constraints, which covers n up to about 63, enough for every
Gurobi run below.

## Reproducing all results

Every run is seeded, so re-running a command reproduces the solutions in the
CSVs exactly. Only the timing columns change, because they depend on the
machine. All commands are run from the repository root. On macOS and Linux,
use `python3` where the commands say `python`.

### 1. Get the code and install the packages

```
git clone https://github.com/Advaith-Nair/DNN-for-Geometric-K-disperison.git
cd DNN-for-Geometric-K-disperison
python -m venv .venv
```

Activate the environment. Windows (PowerShell): `.venv\Scripts\Activate.ps1`.
macOS / Linux: `source .venv/bin/activate`. Then:

```
pip install -r requirements.txt
python eval/check_gurobi_setup.py
```

The last command should end with "Gurobi installation and license check
passed."

### 2. Run the tests

```
cd geometry
python -m pytest -q
cd ../oracle
python -m pytest -q
cd ../baselines
python -m pytest -q
cd ../binary_search
python -m pytest -q
cd ..
python -m pytest -q eval
```

There are 45 tests in total; the three Gurobi tests are skipped if gurobipy
is not installed.

### 3. Run the experiments

The commands write into a new folder `repro/`, so the original results in
`eval/` stay untouched for comparison. Create it first: `mkdir repro`.
The times are what each run took on our laptop; expect about 7 hours in total.

Small instances, against the exact solver and Gurobi (about 40 min):

```
python eval/run_eval.py --with-gurobi --restarts 60 --certify --ns 15 20 --ks 3 4 5 --seeds 10 --seed-start 3100 --dists uniform clustered --out repro/final_small.csv
```

Medium instances, full method, against Gurobi (about 2.2 h):

```
python eval/run_eval.py --with-gurobi --skip-exact --independent-sizes --gurobi-time-limit 60 --gurobi-threads 1 --restarts 60 --certify --ns 30 40 50 60 --ks 6 8 10 --seeds 5 --seed-start 3000 --dists uniform clustered --out repro/final_full.csv
```

Medium instances without local search (about 3 h):

```
python eval/run_eval.py --with-gurobi --skip-exact --independent-sizes --gurobi-time-limit 60 --gurobi-threads 1 --restarts 60 --certify --no-local-search --ns 30 40 50 60 --ks 6 8 10 --seeds 5 --seed-start 3000 --dists uniform clustered --out repro/final_no_ls.csv
```

Medium instances without training (about 2 min). `--max-steps 0` skips the
Adam steps, so the starting θ is rounded directly:

```
python eval/run_eval.py --with-gurobi --skip-exact --independent-sizes --gurobi-time-limit 60 --gurobi-threads 1 --restarts 60 --certify --max-steps 0 --ns 30 40 50 60 --ks 6 8 10 --seeds 5 --seed-start 3000 --dists uniform clustered --out repro/final_untrained.csv
```

Large instances, all four versions (about 30 min together). There is no
reference solver here; the versions are compared against each other.

```
python eval/run_eval.py --skip-exact --independent-sizes --restarts 10 --certify --ns 100 150 --ks 10 15 --seeds 5 --seed-start 4000 --dists uniform clustered --out repro/final_large_trained_ls.csv
python eval/run_eval.py --skip-exact --independent-sizes --restarts 10 --certify --max-steps 0 --ns 100 150 --ks 10 15 --seeds 5 --seed-start 4000 --dists uniform clustered --out repro/final_large_untrained_ls.csv
python eval/run_eval.py --skip-exact --independent-sizes --restarts 10 --certify --no-local-search --ns 100 150 --ks 10 15 --seeds 5 --seed-start 4000 --dists uniform clustered --out repro/final_large_trained.csv
python eval/run_eval.py --skip-exact --independent-sizes --restarts 10 --certify --max-steps 0 --no-local-search --ns 100 150 --ks 10 15 --seeds 5 --seed-start 4000 --dists uniform clustered --out repro/final_large_untrained.csv
```

`--target-mode k2` switches to the enhanced loss of the paper (Eq. 10); it
was not used for these results.

### 4. Make the tables and plots

```
python eval/analyze_results.py --input repro/final_small.csv
python eval/analyze_results.py --input repro/final_full.csv
python eval/analyze_results.py --input repro/final_no_ls.csv
python eval/analyze_results.py --input repro/final_untrained.csv
python eval/compare_runs.py full=repro/final_full.csv no_ls=repro/final_no_ls.csv untrained=repro/final_untrained.csv
python eval/compare_runs.py trained_ls=repro/final_large_trained_ls.csv untrained_ls=repro/final_large_untrained_ls.csv trained=repro/final_large_trained.csv untrained=repro/final_large_untrained.csv
```

`analyze_results.py` writes into `repro/analysis_<csv name>/`: a
`summary.md`, summary CSVs, and every plot as a 300 dpi PNG and a PDF. The
two `compare_runs.py` commands print the medium and large tables below,
including how often each version beats each other one.

Where each result in the paper comes from:

| Paper | Source |
|---|---|
| Table 1, small rows | `repro/analysis_final_small/summary.md` |
| Table 1, medium rows | `repro/analysis_final_full/summary.md` |
| Table 2 (medium ablation) | first `compare_runs.py` command |
| Table 3 (large instances) | second `compare_runs.py` command |
| Fig. 1 (quality without local search) | `repro/analysis_final_no_ls/quality_by_n.pdf` |
| Fig. 2 (runtime, dNN vs Gurobi) | `repro/analysis_final_full/runtime_by_n.pdf` |

### 5. Check that the rerun matches

Compare each new file with the original in `eval/`, for example:

```
python eval/compare_runs.py original=eval/final_full.csv rerun=repro/final_full.csv
```

If the rerun reproduces the original, both rows show the same optimal
percentage and mean and worst ratios, and the "strictly better" table shows
0 in both directions. Repeat for the other seven files. The mean times will
differ from ours, because they depend on the machine.

## Results

Uniform and clustered points, fresh seeds not used during tuning. Every
returned set is re-checked to be valid. Times are from our laptop.

Small, n = 15-20, k = 3-5, 120 instances:

- dNN found the optimum on 120/120
- Gurobi and the exact solver agreed on all 120
- time per instance: dNN 18.6 s, exact solver 0.02 s, Gurobi 0.04 s
- greedy was optimal on 33%, worst case 62% of optimum

Medium, n = 30-60, k = 6-10, 120 instances. Gurobi proved optimality on all
of them, in 0.15 s per instance on average (at most 0.43 s):

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
- Gurobi is much faster at these sizes (0.15 s per medium instance on
  average, against 65 s for the dNN). Speed wasn't the goal here.

## Limitations

- No optimality guarantee. If the dNN misses a feasible set, the search
  settles on a smaller x.
- Proven optima only up to n ≈ 63 with the free Gurobi licence.
- The Gurobi baseline is the plain Kuby model. Stronger formulations, such as
  the compact model of Sayah and Irnich (2017), would make Gurobi even faster.
- Only tested on random 2D points. The method only uses pairwise distances, so
  other dimensions should work without changes, but we haven't tried.

## References

I. R. Alkhouri, G. K. Atia, A. Velasquez. A differentiable approach to the
maximum independent set problem using dataless neural networks. Neural
Networks, 155:168-176, 2022. https://doi.org/10.1016/j.neunet.2022.08.008

K. Chen, A. Dumitrescu. A couple of simple algorithms for k-dispersion.
arXiv:2511.00692, 2025.

S. S. Ravi, D. J. Rosenkrantz, G. K. Tayi. Heuristic and special case
algorithms for dispersion problems. Operations Research, 42(2):299-310, 1994.

D. V. Andrade, M. G. C. Resende, R. F. Werneck. Fast local search for the
maximum independent set problem. Journal of Heuristics, 18(4):525-547, 2012.

M. J. Kuby. Programming models for facility dispersion: the p-dispersion and
maxisum dispersion problems. Geographical Analysis, 19(4):315-329, 1987.

D. Sayah, S. Irnich. A new compact formulation for the discrete p-dispersion
problem. European Journal of Operational Research, 256(1):62-67, 2017.

E. Erkut. The discrete p-dispersion problem. European Journal of Operational
Research, 46(1):48-60, 1990.

D. Pisinger. Upper bounds and exact algorithms for p-dispersion problems.
Computers & Operations Research, 33(5):1380-1398, 2006.
https://doi.org/10.1016/j.cor.2004.09.033

F. Sayyady, Y. Fathi. An integer programming approach for solving the
p-dispersion problem. European Journal of Operational Research,
253(1):216-225, 2016.