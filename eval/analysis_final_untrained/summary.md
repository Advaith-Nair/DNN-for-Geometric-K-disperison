# k-dispersion experiment analysis

**Input:** `eval\final_untrained.csv`  
**Instances:** 120  
**Distributions:** clustered, uniform  
**n values:** 30, 40, 50, 60  
**k values:** 6, 8, 10

## Overall results

| Metric | Result |
|---|---:|
| Valid DNN witnesses | 120/120 (100.00%) |
| Instances with proven reference | 120 |
| DNN exact-match rate (proven-reference subset) | 96.67% |
| Mean DNN / proven optimum | 0.999251 |
| Worst DNN / proven optimum | 0.950629 |
| Mean DNN gap to proven optimum | 0.075% |
| Mean DNN conservative gap to Gurobi bound | 0.075% |
| Mean DNN runtime | 0.718 s |
| Mean Gurobi solver runtime | 0.2456 s |
| Gurobi proven-optimal count | 120/120 |

## Results by distribution, n and k

| Distribution | n | k | Runs | DNN valid % | Proven refs | Exact match % | Mean ratio | Worst ratio | Mean DNN s | Mean Gurobi s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clustered | 30 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.11 | 0.0630 |
| clustered | 30 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.14 | 0.0658 |
| clustered | 30 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.10 | 0.0886 |
| clustered | 40 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.24 | 0.1116 |
| clustered | 40 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.28 | 0.1512 |
| clustered | 40 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.19 | 0.1278 |
| clustered | 50 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.32 | 0.1554 |
| clustered | 50 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.33 | 0.2278 |
| clustered | 50 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.63 | 0.3448 |
| clustered | 60 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 1.04 | 0.3180 |
| clustered | 60 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 1.00 | 0.3080 |
| clustered | 60 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 1.20 | 0.4228 |
| uniform | 30 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 1.03 | 0.1452 |
| uniform | 30 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.22 | 0.1130 |
| uniform | 30 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.14 | 0.1080 |
| uniform | 40 | 6 | 5 | 100.0 | 5 | 80.0 | 0.99608 | 0.98040 | 0.61 | 0.1762 |
| uniform | 40 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.57 | 0.1846 |
| uniform | 40 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.35 | 0.1894 |
| uniform | 50 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 1.06 | 0.2734 |
| uniform | 50 | 8 | 5 | 100.0 | 5 | 80.0 | 0.99013 | 0.95063 | 0.94 | 0.4264 |
| uniform | 50 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 0.97 | 0.4022 |
| uniform | 60 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 2.79 | 0.2922 |
| uniform | 60 | 8 | 5 | 100.0 | 5 | 80.0 | 0.99713 | 0.98566 | 1.54 | 0.4528 |
| uniform | 60 | 10 | 5 | 100.0 | 5 | 80.0 | 0.99868 | 0.99341 | 1.46 | 0.7472 |

## Dataset integrity audit

| Check | Status | Details |
|---|---|---|
| one_point_set_per_distribution_size_replicate | **PASS** | 40 dataset identities checked; each should have one hash across k. Inconsistent identities: 0. |
| no_hash_reused_across_dataset_identities | **PASS** | Independent-size mode: True. Repeated hashes across distinct identities: 0. |
| independent_size_mode_declared | **PASS** | seed_mode values found: independent_sizes. |

## Interpretation notes

- A DNN/optimum ratio is reported only when the CSV contains a custom exact radius or Gurobi has proved optimality.
- For time-limited/non-optimal Gurobi runs, the incumbent is not labelled as the optimum. The DNN-to-bound metric is conservative because the Gurobi bound is an upper bound on the optimum.
- Runtimes depend on hardware and settings (CPU only, single-threaded Gurobi).
