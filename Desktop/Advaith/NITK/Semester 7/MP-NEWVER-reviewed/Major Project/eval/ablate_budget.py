"""
Which factor actually fixes the oracle's false 'infeasible' answers?

diagnose_misses.py changed THREE things at once when it re-asked the oracle:
ADAM steps (400 -> 1000), restarts (15 -> 60) and the random seed. This script
re-asks each false-negative x under controlled variants that change ONE thing
at a time, so we know what the real fix is:

  control : steps=400,  restarts=15, same seeds as the original run
            (should still fail -- confirms these really are failures)
  seed    : steps=400,  restarts=15, DIFFERENT seed
            (flips => it was just seed luck; restarts weren't independent enough)
  steps   : steps=1000, restarts=15, different seed
            (flips => longer/gentler annealing is the key)
  restarts: steps=400,  restarts=60, different seed
            (flips => simply needed more attempts)

Usage (from the eval folder, after run_eval.py produced results.csv):
    python ablate_budget.py
    python ablate_budget.py --limit 8
"""

import argparse
import csv
import time

from run_eval import make_points   # also sets up sys.path for the other folders
from dnn_core import TrainConfig
from feasibility_oracle import OracleConfig, feasibility_oracle
from binary_search import SearchConfig, solve_k_dispersion
from geometry.conflict_graph import build_conflict_graph


def oracle_cfg(steps, restarts):
    return OracleConfig(train_cfg=TrainConfig(target_mode="k", max_steps=steps),
                        max_restarts=restarts)


VARIANTS = [
    # name,       steps, restarts, seed_shift
    ("control",    400,   15,       0),
    ("seed",       400,   15,       1000),
    ("steps",      1000,  15,       1000),
    ("restarts",   400,   60,       1000),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--limit", type=int, default=6, help="missed instances to analyse")
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = [r for r in csv.DictReader(f) if r["dnn_exact"] == "0"][: args.limit]

    base_cfg = SearchConfig(oracle_cfg=oracle_cfg(400, 15), max_iterations=40)

    flips = {name: 0 for name, *_ in VARIANTS}
    secs = {name: 0.0 for name, *_ in VARIANTS}
    total = 0

    for row in rows:
        dist, n, k, seed = row["dist"], int(row["n"]), int(row["k"]), int(row["seed"])
        true_r = float(row["true_radius"])
        points = make_points(dist, n, seed)

        res = solve_k_dispersion(points, k, base_cfg, base_seed=seed)
        fns = [h for h in res.history if (not h["feasible"]) and h["x_guess"] <= true_r]
        print(f"{dist} n={n} k={k} seed={seed}: {len(fns)} false negatives", flush=True)

        for h in fns:
            total += 1
            g = build_conflict_graph(points, h["x_guess"])
            line = []
            for name, steps, restarts, shift in VARIANTS:
                t0 = time.time()
                r = feasibility_oracle(g, k, oracle_cfg(steps, restarts), base_seed=seed + shift)
                secs[name] += time.time() - t0
                flips[name] += int(r.feasible)
                line.append(f"{name}={'Y' if r.feasible else 'n'}")
            print(f"    iter {h['iteration']:>2} x={h['x_guess']:.4f}: " + "  ".join(line), flush=True)

    print("\nSUMMARY (how many of the false negatives became feasible)")
    for name, *_ in VARIANTS:
        print(f"  {name:<9}: {flips[name]:>3}/{total}   total time {secs[name]:.0f}s")


if __name__ == "__main__":
    main()
