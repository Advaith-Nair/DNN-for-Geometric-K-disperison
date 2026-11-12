"""
Diagnose WHY the dNN pipeline missed on certain instances.

Key logic: the oracle can never return a false "yes", because every witness
is a real set of k points that we verify. So lo in the binary search only
ever moves up to a truly feasible radius. Therefore, if the final radius is
below the true optimum, at some iteration the oracle must have answered
"infeasible" for an x that was actually <= the true optimum. That is a
FALSE NEGATIVE, and it permanently pushes hi below the optimum.

This script:
  1. Re-runs the search on each missed instance from results.csv
  2. Lists every false negative in the search history (x_guess <= true radius
     but oracle said infeasible)
  3. Re-asks the oracle at exactly those x values with a much bigger budget
     (more restarts, more ADAM steps) to see if the false negative goes away

Interpretation:
  - false negatives that FLIP to feasible with a bigger budget -> the oracle
    just needed more effort near the threshold (tuning problem)
  - false negatives that STAY infeasible -> a deeper weakness in the
    optimization itself (needs a better method, not just more effort)

Usage (from the eval folder, after run_eval.py produced results.csv):
    python diagnose_misses.py
    python diagnose_misses.py --limit 12 --big-restarts 60 --big-steps 1000
"""

import argparse
import csv
import time

from run_eval import make_points   # also sets up sys.path for the other folders
from dnn_core import TrainConfig
from feasibility_oracle import OracleConfig, feasibility_oracle
from binary_search import SearchConfig, solve_k_dispersion
from geometry.conflict_graph import build_conflict_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--limit", type=int, default=8, help="max missed instances to analyse")
    ap.add_argument("--big-restarts", type=int, default=60)
    ap.add_argument("--big-steps", type=int, default=1000)
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = [r for r in csv.DictReader(f) if r["dnn_exact"] == "0"]
    print(f"{len(rows)} missed instances in {args.csv}; analysing {min(len(rows), args.limit)}\n")

    base_cfg = SearchConfig(
        oracle_cfg=OracleConfig(train_cfg=TrainConfig(target_mode="k", max_steps=400),
                                max_restarts=15),
        max_iterations=40,
    )
    big_cfg = OracleConfig(train_cfg=TrainConfig(target_mode="k", max_steps=args.big_steps),
                           max_restarts=args.big_restarts)

    n_with_fn = 0
    n_fn_total = 0
    n_fn_flipped = 0

    for row in rows[: args.limit]:
        dist, n, k, seed = row["dist"], int(row["n"]), int(row["k"]), int(row["seed"])
        true_r = float(row["true_radius"])
        points = make_points(dist, n, seed)

        res = solve_k_dispersion(points, k, base_cfg, base_seed=seed)
        fns = [h for h in res.history if (not h["feasible"]) and h["x_guess"] <= true_r]

        print(f"{dist} n={n} k={k} seed={seed}: true={true_r:.4f} "
              f"search_best={res.best_radius:.4f}  false negatives={len(fns)}")
        if fns:
            n_with_fn += 1

        for h in fns:
            n_fn_total += 1
            x = h["x_guess"]
            g = build_conflict_graph(points, x)
            t0 = time.time()
            r = feasibility_oracle(g, k, big_cfg, base_seed=seed + 1000)
            dt = time.time() - t0
            if r.feasible:
                n_fn_flipped += 1
            gap = (true_r - x) / true_r
            print(f"    iter {h['iteration']:>2}: x={x:.4f} ({100*gap:.2f}% below optimum) "
                  f"restarts_used_first_try={h['restarts_used']} -> "
                  f"bigger budget: {'FLIPPED to feasible' if r.feasible else 'still infeasible'} "
                  f"({dt:.1f}s)")

    print("\nSUMMARY")
    print(f"  missed instances analysed        : {min(len(rows), args.limit)}")
    print(f"  of those, with >=1 false negative: {n_with_fn}")
    print(f"  total false negatives            : {n_fn_total}")
    print(f"  fixed by bigger budget           : {n_fn_flipped}")
    if n_fn_total:
        print(f"  -> {100*n_fn_flipped/n_fn_total:.0f}% of false negatives were just under-searched")


if __name__ == "__main__":
    main()
