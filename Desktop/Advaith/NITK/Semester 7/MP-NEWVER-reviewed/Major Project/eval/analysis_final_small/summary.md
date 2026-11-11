# k-dispersion experiment analysis

**Input:** `eval\final_small.csv`  
**Instances:** 120  
**Distributions:** clustered, uniform  
**n values:** 15, 20  
**k values:** 3, 4, 5

## Overall results

| Metric | Result |
|---|---:|
| Valid DNN witnesses | 120/120 (100.00%) |
| Instances with proven reference | 120 |
| DNN exact-match rate (proven-reference subset) | 100.00% |
| Mean DNN / proven optimum | 1.000000 |
| Worst DNN / proven optimum | 1.000000 |
| Mean DNN gap to proven optimum | 0.000% |
| Mean DNN conservative gap to Gurobi bound | 0.000% |
| Mean DNN runtime | 18.625 s |
| Mean Gurobi solver runtime | 0.0208 s |
| Gurobi proven-optimal count | 120/120 |

## Results by distribution, n and k

| Distribution | n | k | Runs | DNN valid % | Proven refs | Exact match % | Mean ratio | Worst ratio | Mean DNN s | Mean Gurobi s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clustered | 15 | 3 | 10 | 100.0 | 10 | 100.0 | 1.00000 | 1.00000 | 5.69 | 0.0153 |
| clustered | 15 | 4 | 10 | 100.0 | 10 | 100.0 | 1.00000 | 1.00000 | 5.59 | 0.0195 |
| clustered | 15 | 5 | 10 | 100.0 | 10 | 100.0 | 1.00000 | 1.00000 | 0.31 | 0.0143 |
| clustered | 20 | 3 | 10 | 100.0 | 10 | 100.0 | 1.00000 | 1.00000 | 0.74 | 0.0110 |
| clustered | 20 | 4 | 10 | 100.0 | 10 | 100.0 | 1.00000 | 1.00000 | 0.74 | 0.0189 |
| clustered | 20 | 5 | 10 | 100.0 | 10 | 100.0 | 1.00000 | 1.00000 | 1.89 | 0.0238 |
| uniform | 15 | 3 | 10 | 100.0 | 10 | 100.0 | 1.00000 | 1.00000 | 45.76 | 0.0133 |
| uniform | 15 | 4 | 10 | 100.0 | 10 | 100.0 | 1.00000 | 1.00000 | 17.97 | 0.0201 |
| uniform | 15 | 5 | 10 | 100.0 | 10 | 100.0 | 1.00000 | 1.00000 | 14.48 | 0.0154 |
| uniform | 20 | 3 | 10 | 100.0 | 10 | 100.0 | 1.00000 | 1.00000 | 41.03 | 0.0181 |
| uniform | 20 | 4 | 10 | 100.0 | 10 | 100.0 | 1.00000 | 1.00000 | 49.59 | 0.0246 |
| uniform | 20 | 5 | 10 | 100.0 | 10 | 100.0 | 1.00000 | 1.00000 | 39.70 | 0.0555 |

## Dataset integrity audit

| Check | Status | Details |
|---|---|---|
| one_point_set_per_distribution_size_replicate | **PASS** | 40 dataset identities checked; each should have one hash across k. Inconsistent identities: 0. |
| no_hash_reused_across_dataset_identities | **NOT_APPLICABLE** | Independent-size mode: False. Repeated hashes across distinct identities: 0. |
| independent_size_mode_declared | **WARNING** | seed_mode values found: nested. |

## Interpretation notes

- A DNN/optimum ratio is reported only when the CSV contains a custom exact radius or Gurobi has proved optimality.
- For time-limited/non-optimal Gurobi runs, the incumbent is not labelled as the optimum. The DNN-to-bound metric is conservative because the Gurobi bound is an upper bound on the optimum.
- Grouped metrics are descriptive. Two or three replicates are not enough for strong statistical generalization claims; increase the replicate count before final reporting.
- Runtime comparisons are hardware- and configuration-dependent. Record machine, Python, PyTorch and Gurobi versions in the final methodology.
