"""
Compare several run_eval.py CSVs that were run on the SAME instances
(same --ns/--ks/--seeds/--seed-start/--dists/--independent-sizes), e.g. the
full pipeline vs. the ablations.

Reference radius per instance = exact solver radius if present, otherwise
Gurobi's radius when Gurobi proved optimality, otherwise the best radius any
compared run found (then "exact %" means "matched the best known").

Usage (from the project root):
    python eval/compare_runs.py full=eval/final_full.csv no_ls=eval/final_no_ls.csv \
        untrained=eval/final_untrained.csv
"""

import csv
import sys


def key(r):
    return (r["dist"], r["n"], r["k"], r.get("replicate_seed") or r["seed"])


def reference(r):
    if r.get("true_radius"):
        return float(r["true_radius"]), "exact"
    if r.get("gurobi_optimal") == "1" and r.get("gurobi_radius"):
        return float(r["gurobi_radius"]), "gurobi"
    return None, None


def main(argv):
    runs = {}
    for arg in argv:
        label, path = arg.split("=", 1)
        runs[label] = {key(r): r for r in csv.DictReader(open(path))}

    common = set.intersection(*(set(v) for v in runs.values()))
    if not common:
        sys.exit("no common instances -- were the runs made with the same settings?")

    ref = {}
    ref_kind = {}
    for kk in common:
        any_row = next(iter(runs.values()))[kk]
        val, kind = reference(any_row)
        if val is None:
            val = max(float(rs[kk]["dnn_radius"]) for rs in runs.values())
            kind = "best-known"
        ref[kk], ref_kind[kk] = val, kind

    kinds = sorted(set(ref_kind.values()))
    print(f"{len(common)} common instances; reference = {', '.join(kinds)}\n")
    print(f"{'run':<14}{'exact %':>9}{'mean ratio':>12}{'worst':>9}{'mean t_dnn(s)':>15}")
    rows_greedy = None
    for label, rs in runs.items():
        ratios = [float(rs[kk]["dnn_radius"]) / ref[kk] for kk in common]
        exact = sum(x >= 1 - 1e-9 for x in ratios)
        t = sum(float(rs[kk]["t_dnn"]) for kk in common) / len(common)
        print(f"{label:<14}{100*exact/len(common):>8.1f}%{sum(ratios)/len(ratios):>12.4f}"
              f"{min(ratios):>9.4f}{t:>15.2f}")
        rows_greedy = rs
    g = [float(rows_greedy[kk]["greedy_radius"]) / ref[kk] for kk in common]
    print(f"{'greedy':<14}{100*sum(x >= 1-1e-9 for x in g)/len(g):>8.1f}%"
          f"{sum(g)/len(g):>12.4f}{min(g):>9.4f}{'':>15}")

    labels = list(runs)
    if len(labels) >= 2:
        print("\nHead-to-head (row beats column on how many instances):")
        print(" " * 14 + "".join(f"{l:>14}" for l in labels))
        for a in labels:
            line = f"{a:<14}"
            for b in labels:
                if a == b:
                    line += f"{'-':>14}"
                    continue
                wins = sum(float(runs[a][kk]["dnn_radius"]) > float(runs[b][kk]["dnn_radius"]) + 1e-12
                           for kk in common)
                line += f"{wins:>14}"
            print(line)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
