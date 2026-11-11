# Review: changes made and checks still to do

## Code changes (all 45 tests pass)

| # | File | What changed | Why |
|---|---|---|---|
| 1 | `binary_search/binary_search.py` | New default `mode="discrete"`: binary search over the sorted distinct pairwise distances; when the oracle returns a witness, `lo` jumps to that witness's real radius. Old bisection kept as `mode="continuous"`. | The optimum is always a pairwise distance. Bisection on `[0, diameter]` with `eps = 1e-4*diameter` could stop just below it, so some old "misses" (ratios 0.99994, 0.99976 in `results*.csv`) were search-resolution artefacts, not dNN failures. Oracle calls also drop from 14 to about 6–10 per instance. |
| 2 | `oracle/dnn_core.py` | `enhanced_penalty` (`target_mode="k2"`) now receives the annealed `leak`. Added `target_mode="n"` (the paper's original target `-n/2`). | `k2` silently used a fixed leak of 0.05 and ignored the schedule. |
| 3 | `oracle/repair.py`, `feasibility_oracle.py` | Added the paper's 2-improvement local search (Sec. 5.1) after repair (`OracleConfig.local_search`, on by default). | This targets the remaining failure mode, where the oracle says "infeasible" at radii that are actually feasible. Medium benchmark: 16/24 → 23/24 proven optima. |
| 4 | `eval/run_eval.py` | New flags `--search-mode`, `--target-mode`, `--no-local-search`. Every CSV row now records its `cfg_*` settings. k<2 is skipped. `--skip-exact` is allowed without Gurobi (for large-n head-to-heads). | Provenance: the old CSVs do not say which restarts/steps/certify settings produced them. |
| 5 | `eval/compare_runs.py` | New: compares several CSVs from the same instances (exact %, mean/worst ratio, head-to-head wins). | For the ablation table. |
| 6 | tests | Added a perfect-oracle test showing the discrete search is exact, plus a 2-improvement unit test. | Shows the search logic is correct independently of the dNN. |

Note: CSVs made before these changes used continuous search and no local search.
Label them as "v1 pipeline" in the report or regenerate them.

## Key finding: an ablation you must report

`eval/ablation_medium_1100/`: same 24 instances as `results_gurobi_medium_independent.csv`
(uniform, n = 30–60, k = 6/8/10, replicates 1100–1101, 60 restarts, certify on), all with Gurobi proven optima:

| Pipeline | Exact (proven optimum) | Mean ratio | Worst |
|---|---|---|---|
| trained dNN + local search (new default) | 95.8% | 0.9997 | 0.9923 |
| trained dNN, no local search (your original) | 66.7% | 0.9946 | 0.9643 |
| **no training** (init → threshold → repair), + local search | 100% | 1.0000 | 1.0000 |
| **no training**, no local search | 87.5% | 0.9991 | 0.9916 |
| paper target −n/2, no local search | 54.2% | 0.9901 | 0.9434 |
| greedy farthest-point | 0% | 0.8817 | 0.7313 |

On these sizes, random/degree initialisation plus repair, restarted 60 times, is at least as good as
training the dNN. At larger sizes (n = 100/150, 10 restarts, no LS) the trained dNN beat the
untrained version on 8 of 12 instances, by small margins (`large_*.csv`).
An examiner can easily ask "what does the network add?", so put this ablation in the report and discuss it.
Don't hide it.

## Checks still to do (on your machine)

1. **Final experiments with enough replicates and both distributions** (currently only 2 seeds, uniform only).
   Use fresh seeds you never tuned on:
   ```
   python eval/run_eval.py --with-gurobi --skip-exact --independent-sizes --gurobi-time-limit 60 --gurobi-threads 1 --restarts 60 --certify --ns 30 40 50 60 --ks 6 8 10 --seeds 5 --seed-start 3000 --dists uniform clustered --out eval/final_full.csv
   ```
   Then the same command with `--no-local-search --out eval/final_no_ls.csv`, and with
   `--max-steps 0 --out eval/final_untrained.csv` (no training at all), then:
   ```
   python eval/compare_runs.py full=eval/final_full.csv no_ls=eval/final_no_ls.csv untrained=eval/final_untrained.csv
   python eval/analyze_results.py --input eval/final_full.csv
   ```
2. **Small instances vs the exact solver** (cross-checks Gurobi too):
   `--ns 15 20 --ks 3 4 5 --seeds 10 --seed-start 3100 --dists uniform clustered --with-gurobi` (without `--skip-exact`).
3. **Gurobi free-licence limit**: the MILP has n(n-1)/2+1 constraints, so n ≤ 63 under the pip licence.
   Larger n needs the academic licence. Otherwise compare against greedy and the best-known radius only.
4. **Report write-up**: the progress report proposed a max-sum dNN, but the implementation solves max-min via
   binary search + an MIS-dNN feasibility oracle. Explain this switch, and list every deviation from the
   paper: target −k/2, leaky-ReLU annealing, binarisation term, mixed init, restarts, certificate, local search.
   Record machine, Python, PyTorch and Gurobi versions.
