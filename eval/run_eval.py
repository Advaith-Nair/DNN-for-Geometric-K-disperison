"""
Experiment runner.

For every combination of distribution, n, k and seed, a point set is
generated and solved with the dNN method, the greedy heuristic and,
optionally, the exact solver and Gurobi. One CSV row is written per instance.

The value of each method is recomputed from the returned point set (its
minimum pairwise distance), independently of the value the method reports.
Each row also records the run configuration (cfg_* columns).

Examples (from the repository root):
    python eval/run_eval.py --ns 15 20 --ks 3 4 5 --seeds 10 --dists uniform clustered --with-gurobi
    python eval/run_eval.py --with-gurobi --skip-exact --independent-sizes \
        --ns 30 40 50 60 --ks 6 8 10 --seeds 5 --restarts 60 --certify --out eval/final_full.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ["", "oracle", "baselines", "binary_search"]:
    p = os.path.join(ROOT, sub) if sub else ROOT
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

from geometry.points import random_uniform_points, random_clustered_points
from dnn_core import TrainConfig
from feasibility_oracle import OracleConfig
from binary_search import SearchConfig, solve_k_dispersion
from exact_solver import exact_k_dispersion
from greedy_solver import greedy_k_dispersion
from gurobi_solver import GurobiConfig, solve_k_dispersion_gurobi


def achieved_radius(points, witness) -> float:
    """Minimum pairwise distance of a set of point indices."""
    return min(points.dist[i, j] for i, j in itertools.combinations(witness, 2))


def witness_is_valid(witness, k, n) -> bool:
    """Check that a solution has k distinct, in-range point indices."""
    return (
        len(witness) == k
        and len(set(witness)) == k
        and all(0 <= v < n for v in witness)
    )


def derive_independent_size_seed(dist_name: str, n: int, replicate_seed: int) -> int:
    """Return a deterministic seed for (distribution, n, replicate).

    k is not part of the seed, so all values of k share the same point set.
    n is part of the seed, so instances of different sizes are generated
    independently rather than as prefixes of one another.
    """
    payload = f"kdispersion-v1|{dist_name}|{n}|{replicate_seed}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def point_set_fingerprint(points) -> str:
    """Return a short hash of the coordinates, used to check dataset consistency."""
    coords = points.coords.astype("<f8", copy=False)
    return hashlib.sha256(coords.tobytes(order="C")).hexdigest()[:20]


def make_points(dist_name: str, n: int, seed: int):
    """Generate a uniform or clustered instance (4 clusters, standard deviation 0.08)."""
    if dist_name == "uniform":
        return random_uniform_points(n, dim=2, seed=seed)
    if dist_name == "clustered":
        return random_clustered_points(n, dim=2, n_clusters=4, cluster_std=0.08, seed=seed)
    raise ValueError(f"unknown distribution {dist_name}")


def run_instance(dist_name, n, k, seed, search_cfg, gurobi_cfg=None, use_exact=True,
                 replicate_seed=None, seed_mode="nested"):
    """Solve one instance with every requested method and return a CSV row."""
    points = make_points(dist_name, n, seed)

    true_r = None
    t_exact = None
    if use_exact:
        t0 = time.perf_counter()
        exact = exact_k_dispersion(points, k)
        t_exact = time.perf_counter() - t0
        true_r = float(exact.radius)
        if not (true_r > 0):
            return None  # degenerate instance, e.g. duplicate points

    t0 = time.perf_counter()
    greedy = greedy_k_dispersion(points, k)
    t_greedy = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = solve_k_dispersion(points, k, search_cfg, base_seed=seed)
    t_dnn = time.perf_counter() - t0

    dnn_valid = witness_is_valid(res.best_witness, k, n)
    dnn_r = achieved_radius(points, res.best_witness) if dnn_valid else 0.0
    greedy_r = greedy.radius

    dnn_ratio = (dnn_r / true_r) if true_r is not None else None
    dnn_exact = (int(dnn_r >= true_r * (1 - 1e-6)) if true_r is not None else None)
    greedy_ratio = (greedy_r / true_r) if true_r is not None else None
    greedy_exact = (int(greedy_r >= true_r * (1 - 1e-6)) if true_r is not None else None)

    row = {
        "dist": dist_name, "n": n, "k": k,
        "seed_mode": seed_mode,
        "seed": seed,
        "replicate_seed": (seed if replicate_seed is None else replicate_seed),
        "point_set_hash": point_set_fingerprint(points),
        "true_radius": true_r,
        "dnn_radius": dnn_r,
        "dnn_ratio": dnn_ratio,
        "dnn_exact": dnn_exact,
        "dnn_valid": int(dnn_valid),
        "greedy_radius": greedy_r,
        "greedy_ratio": greedy_ratio,
        "greedy_exact": greedy_exact,
        "t_exact": t_exact, "t_greedy": t_greedy, "t_dnn": t_dnn,
        "bs_iters": res.iterations_run,
        "oracle_calls_total_restarts": sum(h["restarts_used"] for h in res.history),
        "oracle_max_restarts_in_one_call": max(
            (h["restarts_used"] for h in res.history), default=0
        ),
        # Run configuration.
        "cfg_search_mode": search_cfg.mode,
        "cfg_target_mode": search_cfg.oracle_cfg.train_cfg.target_mode,
        "cfg_max_steps": search_cfg.oracle_cfg.train_cfg.max_steps,
        "cfg_restarts": search_cfg.oracle_cfg.max_restarts,
        "cfg_certify": int(search_cfg.oracle_cfg.use_certificate),
        "cfg_local_search": int(search_cfg.oracle_cfg.local_search),
    }

    if gurobi_cfg is not None:
        t0 = time.perf_counter()
        gr = solve_k_dispersion_gurobi(points, k, gurobi_cfg)
        gurobi_total = time.perf_counter() - t0
        gurobi_valid = witness_is_valid(gr.witness, k, n) if gr.witness is not None else False
        gurobi_r = gr.radius if gurobi_valid else None
        gurobi_ratio = (gurobi_r / true_r) if (gurobi_r is not None and true_r is not None) else None
        gurobi_exact = (
            int(gurobi_r >= true_r * (1 - 1e-6))
            if (gurobi_r is not None and true_r is not None) else None
        )
        bound_gap = None
        if gr.obj_bound is not None and gr.obj_bound > 0 and dnn_valid:
            bound_gap = max(0.0, gr.obj_bound - dnn_r) / max(abs(gr.obj_bound), 1e-12)
        row.update({
            "gurobi_radius": gurobi_r,
            "gurobi_ratio": gurobi_ratio,
            "gurobi_exact": gurobi_exact,
            "gurobi_valid": int(gurobi_valid),
            "gurobi_status": gr.status_name,
            "gurobi_optimal": int(gr.is_optimal),
            "gurobi_obj": gr.obj_val,
            "gurobi_bound": gr.obj_bound,
            "gurobi_gap": gr.mip_gap,
            "gurobi_nodes": gr.node_count,
            "gurobi_runtime": gr.runtime,
            "gurobi_wall_time": gurobi_total,
            "gurobi_nvars": gr.num_vars,
            "gurobi_nconstraints": gr.num_constraints,
            "dnn_gap_to_gurobi_bound": bound_gap,
        })

    return row


def summarize(rows):
    """Print a summary of the results to the console."""
    exact_rows = [r for r in rows if r.get("true_radius") is not None]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dist"], r["n"], r["k"])].append(r)

    if exact_rows:
        header = (f"{'dist':<10}{'n':>4}{'k':>3}{'#':>4} | "
                  f"{'dNN exact%':>10}{'dNN worst':>10}{'dNN avg':>9} | "
                  f"{'greedy exact%':>13}{'greedy worst':>13} | "
                  f"{'t_dnn(s)':>9}{'t_exact(s)':>11}")
        print(header)
        print("-" * len(header))
        for key in sorted(groups):
            g = [r for r in groups[key] if r.get("true_radius") is not None]
            if not g:
                continue
            m = len(g)
            print(f"{key[0]:<10}{key[1]:>4}{key[2]:>3}{m:>4} | "
                  f"{100*sum(r['dnn_exact'] for r in g)/m:>10.1f}"
                  f"{min(r['dnn_ratio'] for r in g):>10.4f}"
                  f"{sum(r['dnn_ratio'] for r in g)/m:>9.4f} | "
                  f"{100*sum(r['greedy_exact'] for r in g)/m:>13.1f}"
                  f"{min(r['greedy_ratio'] for r in g):>13.4f} | "
                  f"{sum(r['t_dnn'] for r in g)/m:>9.2f}"
                  f"{sum(r['t_exact'] for r in g)/m:>11.3f}")

        m = len(exact_rows)
        print("\nComparison with the exact solver over", m, "instances")
        print(f"  dNN    : exact-match {100*sum(r['dnn_exact'] for r in exact_rows)/m:.1f}%   "
              f"worst ratio {min(r['dnn_ratio'] for r in exact_rows):.4f}   "
              f"avg ratio {sum(r['dnn_ratio'] for r in exact_rows)/m:.4f}   "
              f"invalid witnesses {sum(1 - r['dnn_valid'] for r in exact_rows)}")
        print(f"  greedy : exact-match {100*sum(r['greedy_exact'] for r in exact_rows)/m:.1f}%   "
              f"worst ratio {min(r['greedy_ratio'] for r in exact_rows):.4f}   "
              f"avg ratio {sum(r['greedy_ratio'] for r in exact_rows)/m:.4f}")
    else:
        print("\nNo exact solver results; reporting values found and Gurobi bounds.")
        print(f"{'dist':<10}{'n':>4}{'k':>3}{'#':>4} | {'dNN radius':>11}{'Gurobi inc.':>12}{'dNN-bound gap':>15}")
        print("-" * 70)
        for key in sorted(groups):
            g = groups[key]
            avg_dnn = sum(r["dnn_radius"] for r in g) / len(g)
            inc = [r["gurobi_radius"] for r in g if r.get("gurobi_radius") is not None]
            gaps = [r["dnn_gap_to_gurobi_bound"] for r in g if r.get("dnn_gap_to_gurobi_bound") is not None]
            print(f"{key[0]:<10}{key[1]:>4}{key[2]:>3}{len(g):>4} | "
                  f"{avg_dnn:>11.5f}"
                  f"{(sum(inc)/len(inc) if inc else float('nan')):>12.5f}"
                  f"{(sum(gaps)/len(gaps) if gaps else float('nan')):>15.5f}")

    if rows and "gurobi_status" in rows[0]:
        gurobi_rows = [r for r in rows if r.get("gurobi_status") is not None]
        incumbents = [r for r in gurobi_rows if r.get("gurobi_valid") == 1]
        optimal = [r for r in gurobi_rows if r.get("gurobi_optimal") == 1]
        gaps = [r["gurobi_gap"] for r in incumbents if r.get("gurobi_gap") is not None]
        bound_gaps = [r["dnn_gap_to_gurobi_bound"] for r in gurobi_rows
                      if r.get("dnn_gap_to_gurobi_bound") is not None]
        statuses = defaultdict(int)
        for r in gurobi_rows:
            statuses[r["gurobi_status"]] += 1
        print("\nGurobi results over", len(gurobi_rows), "instances")
        print(f"  feasible incumbents : {len(incumbents)}/{len(gurobi_rows)}")
        print(f"  proven optimal      : {len(optimal)}/{len(gurobi_rows)}")
        print(f"  mean solver runtime : {sum(r['gurobi_runtime'] for r in gurobi_rows)/len(gurobi_rows):.4f}s")
        if exact_rows:
            g_exact = [r for r in exact_rows if r.get("gurobi_exact") is not None]
            if g_exact:
                g_ratio = [r["gurobi_ratio"] for r in g_exact if r.get("gurobi_ratio") is not None]
                print(f"  exact-reference match: {100*sum(r['gurobi_exact'] for r in g_exact)/len(g_exact):.1f}% "
                      f"({sum(r['gurobi_exact'] for r in g_exact)}/{len(g_exact)})")
                if g_ratio:
                    print(f"  worst/mean Gurobi ratio: {min(g_ratio):.6f} / "
                          f"{sum(g_ratio)/len(g_ratio):.6f}")
        if gaps:
            print(f"  mean incumbent MIP gap: {sum(gaps)/len(gaps):.6f}")
        if bound_gaps:
            print(f"  mean dNN-to-Gurobi-bound gap: {sum(bound_gaps)/len(bound_gaps):.6f}")
        print("  statuses            : " + ", ".join(f"{s}={c}" for s, c in sorted(statuses.items())))


def main():
    ap = argparse.ArgumentParser(description="Run k-dispersion experiments and write a results CSV.")
    ap.add_argument("--ns", type=int, nargs="+", default=[12])
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds per config")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first replicate seed")
    ap.add_argument("--independent-sizes", action="store_true",
                    help="derive a separate point-set seed for each (distribution, n, replicate); "
                         "the same point set is used for every k")
    ap.add_argument("--dists", nargs="+", default=["uniform"])
    ap.add_argument("--max-steps", type=int, default=400, help="ADAM steps per oracle attempt")
    ap.add_argument("--restarts", type=int, default=15, help="max oracle restarts")
    ap.add_argument("--search-mode", choices=["discrete", "continuous"], default="discrete",
                    help="discrete: search over pairwise distances (default); "
                         "continuous: bisection on [0, diameter]")
    ap.add_argument("--target-mode", choices=["k", "n", "k2"], default="k",
                    help="k: loss f (Eq. 7), target -k/2; n: loss f, target -n/2; "
                         "k2: loss h (Eq. 10), target -k^2/2")
    ap.add_argument("--no-local-search", action="store_true",
                    help="disable the 2-improvement local search")
    ap.add_argument("--certify", action="store_true",
                    help="use the clique-cover bound to reject infeasible x without running the network")
    ap.add_argument("--with-gurobi", action="store_true",
                    help="also solve every instance with the Gurobi MILP")
    ap.add_argument("--gurobi-time-limit", type=float, default=60.0,
                    help="Gurobi time limit in seconds per instance")
    ap.add_argument("--gurobi-mip-gap", type=float, default=0.0,
                    help="relative MIP gap (0 = solve to proven optimality)")
    ap.add_argument("--gurobi-threads", type=int, default=1,
                    help="number of Gurobi threads")
    ap.add_argument("--skip-exact", action="store_true",
                    help="do not run the exact solver (use for n > 20)")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    if args.skip_exact and not args.with_gurobi:
        print("Note: neither the exact solver nor Gurobi is used; only the values found are reported.")

    gurobi_cfg = None
    if args.with_gurobi:
        gurobi_cfg = GurobiConfig(
            time_limit=args.gurobi_time_limit,
            mip_gap=args.gurobi_mip_gap,
            threads=args.gurobi_threads,
            output_flag=0,
        )

    search_cfg = SearchConfig(
        oracle_cfg=OracleConfig(
            train_cfg=TrainConfig(target_mode=args.target_mode, max_steps=args.max_steps),
            max_restarts=args.restarts,
            use_certificate=args.certify,
            local_search=not args.no_local_search,
        ),
        max_iterations=40,
        mode=args.search_mode,
    )

    rows = []
    configs = list(itertools.product(args.dists, args.ns, args.ks,
                                      range(args.seed_start, args.seed_start + args.seeds)))
    for idx, (d, n, k, replicate_seed) in enumerate(configs, 1):
        if k > n or k < 2:
            continue  # k = 1 has no pairwise distance
        seed = (derive_independent_size_seed(d, n, replicate_seed)
                if args.independent_sizes else replicate_seed)
        row = run_instance(
            d, n, k, seed, search_cfg,
            gurobi_cfg=gurobi_cfg,
            use_exact=not args.skip_exact,
            replicate_seed=replicate_seed,
            seed_mode=("independent_sizes" if args.independent_sizes else "nested"),
        )
        if row is None:
            print(f"[{idx}/{len(configs)}] {d} n={n} k={k} seed={seed}: skipped (degenerate)")
            continue
        rows.append(row)
        if row["dnn_exact"] is None:
            flag = "RUN"
            quality = f"dnn={row['dnn_radius']:.4f}"
        else:
            flag = "OK " if row["dnn_exact"] else "MISS"
            quality = (f"dnn={row['dnn_radius']:.4f} true={row['true_radius']:.4f} "
                       f"ratio={row['dnn_ratio']:.4f}")
        solver = (f" Gurobi={row['gurobi_status']} "
                  f"r={row['gurobi_radius']} gap={row['gurobi_gap']}"
                  if gurobi_cfg is not None else "")
        seed_label = f"seed={seed}" if not args.independent_sizes else f"replicate={replicate_seed} instance_seed={seed}"
        print(f"[{idx}/{len(configs)}] {d} n={n} k={k} {seed_label}: {flag} "
              f"{quality} ({row['t_dnn']:.1f}s){solver}", flush=True)

    if not rows:
        print("no instances were run")
        return

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote per-instance results to {args.out}\n")
    summarize(rows)


if __name__ == "__main__":
    main()