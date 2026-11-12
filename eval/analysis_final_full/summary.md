# k-dispersion experiment analysis

**Input:** `eval\final_full.csv`  
**Instances:** 120  
**Distributions:** clustered, uniform  
**n values:** 30, 40, 50, 60  
**k values:** 6, 8, 10

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
| Mean DNN runtime | 65.192 s |
| Mean Gurobi solver runtime | 0.2156 s |
| Gurobi proven-optimal count | 120/120 |

## Results by distribution, n and k

| Distribution | n | k | Runs | DNN valid % | Proven refs | Exact match % | Mean ratio | Worst ratio | Mean DNN s | Mean Gurobi s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clustered | 30 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 35.48 | 0.0624 |
| clustered | 30 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 31.89 | 0.0604 |
| clustered | 30 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 19.19 | 0.0674 |
| clustered | 40 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 25.28 | 0.1116 |
| clustered | 40 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 56.66 | 0.1304 |
| clustered | 40 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 36.50 | 0.1128 |
| clustered | 50 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 25.28 | 0.1390 |
| clustered | 50 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 54.10 | 0.1956 |
| clustered | 50 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 37.05 | 0.2162 |
| clustered | 60 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 84.08 | 0.2468 |
| clustered | 60 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 69.99 | 0.2672 |
| clustered | 60 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 84.86 | 0.3288 |
| uniform | 30 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 59.67 | 0.0550 |
| uniform | 30 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 39.67 | 0.0978 |
| uniform | 30 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 34.42 | 0.1318 |
| uniform | 40 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 160.74 | 0.3198 |
| uniform | 40 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 119.48 | 0.2194 |
| uniform | 40 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 70.24 | 0.2672 |
| uniform | 50 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 55.49 | 0.1332 |
| uniform | 50 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 90.30 | 0.3732 |
| uniform | 50 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 98.24 | 0.4144 |
| uniform | 60 | 6 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 89.86 | 0.2414 |
| uniform | 60 | 8 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 101.85 | 0.3966 |
| uniform | 60 | 10 | 5 | 100.0 | 5 | 100.0 | 1.00000 | 1.00000 | 84.29 | 0.5850 |

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
