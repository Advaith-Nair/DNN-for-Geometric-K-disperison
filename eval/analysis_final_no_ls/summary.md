# k-dispersion experiment analysis

**Input:** `eval\final_no_ls.csv`  
**Instances:** 120  
**Distributions:** clustered, uniform  
**n values:** 30, 40, 50, 60  
**k values:** 6, 8, 10

## Overall results

| Metric | Result |
|---|---:|
| Valid DNN witnesses | 120/120 (100.00%) |
| Instances with proven reference | 120 |
| DNN exact-match rate (proven-reference subset) | 75.00% |
| Mean DNN / proven optimum | 0.990770 |
| Worst DNN / proven optimum | 0.877553 |
| Mean DNN gap to proven optimum | 0.923% |
| Mean DNN conservative gap to Gurobi bound | 0.923% |
| Mean DNN runtime | 92.016 s |
| Mean Gurobi solver runtime | 0.1922 s |
| Gurobi proven-optimal count | 120/120 |

## Results by distribution, n and k

| Distribution | n | k | Runs | DNN valid % | Proven refs | Exact match % | Mean ratio | Worst ratio | Mean DNN s | Mean Gurobi s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clustered | 30 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 27.82 | 0.0472 |
| clustered | 30 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 29.70 | 0.0594 |
| clustered | 30 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 16.91 | 0.0658 |
| clustered | 40 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 29.30 | 0.0868 |
| clustered | 40 | 8 | 5 | 100.0 | 5 | 40.0 | 0.96205 | 0.87755 | 59.30 | 0.1084 |
| clustered | 40 | 10 | 5 | 100.0 | 5 | 80.0 | 0.99623 | 0.98113 | 42.91 | 0.0914 |
| clustered | 50 | 6 | 5 | 100.0 | 5 | 80.0 | 0.99253 | 0.96267 | 36.93 | 0.1524 |
| clustered | 50 | 8 | 5 | 100.0 | 5 | 60.0 | 0.96612 | 0.88927 | 64.05 | 0.1674 |
| clustered | 50 | 10 | 5 | 100.0 | 5 | 80.0 | 0.99467 | 0.97337 | 33.32 | 0.1826 |
| clustered | 60 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 59.66 | 0.1918 |
| clustered | 60 | 8 | 5 | 100.0 | 5 | 80.0 | 0.99086 | 0.95430 | 49.03 | 0.2068 |
| clustered | 60 | 10 | 5 | 100.0 | 5 | 40.0 | 0.99013 | 0.96260 | 63.26 | 0.2842 |
| uniform | 30 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 119.87 | 0.1144 |
| uniform | 30 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 73.60 | 0.0878 |
| uniform | 30 | 10 | 5 | 100.0 | 5 | 60.0 | 0.99091 | 0.96499 | 44.36 | 0.0848 |
| uniform | 40 | 6 | 5 | 100.0 | 5 | 60.0 | 0.99356 | 0.97034 | 96.45 | 0.1622 |
| uniform | 40 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 111.12 | 0.1748 |
| uniform | 40 | 10 | 5 | 100.0 | 5 | 80.0 | 0.99944 | 0.99719 | 136.15 | 0.3582 |
| uniform | 50 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 85.68 | 0.2284 |
| uniform | 50 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 97.84 | 0.2720 |
| uniform | 50 | 10 | 5 | 100.0 | 5 | 60.0 | 0.99251 | 0.98091 | 151.61 | 0.4416 |
| uniform | 60 | 6 | 5 | 100.0 | 5 | 60.0 | 0.98711 | 0.96222 | 50.77 | 0.1694 |
| uniform | 60 | 8 | 5 | 100.0 | 5 | 20.0 | 0.95400 | 0.88997 | 640.01 | 0.3398 |
| uniform | 60 | 10 | 5 | 100.0 | 5 | 0.0 | 0.96837 | 0.90979 | 88.76 | 0.5358 |

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
