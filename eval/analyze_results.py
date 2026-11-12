"""
Summary tables and figures for a results CSV produced by run_eval.py.

The reference value of an instance is chosen as follows:

* true_radius (exact solver), when present;
* otherwise gurobi_radius, but only if Gurobi proved optimality;
* otherwise no optimum is assumed. If a Gurobi bound is available, only the
  gap between the dNN value and that upper bound is reported.

Usage from the project root:
    python eval/analyze_results.py --input eval/final_full.csv

Outputs go to eval/analysis_<input-stem>/ by default. Each plot is saved as
PNG (300 dpi) and PDF.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TOLERANCE = 1e-6


def _number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _flag(value: Any) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def read_results(path: str | Path) -> list[dict[str, Any]]:
    """Read a results CSV and check that the required columns are present."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"CSV contains no result rows: {path}")
    required = {"dist", "n", "k", "dnn_radius"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    return rows


def _reference(row: dict[str, Any]) -> tuple[float | None, str]:
    """Return a proven optimum and its source, if one is available."""
    exact = _number(row.get("true_radius"))
    if exact is not None and exact > 0:
        return exact, "custom_exact"

    gurobi_optimal = _flag(row.get("gurobi_optimal")) is True
    gurobi_valid = _flag(row.get("gurobi_valid"))
    gurobi_radius = _number(row.get("gurobi_radius"))
    if gurobi_optimal and gurobi_radius is not None and gurobi_radius > 0:
        if gurobi_valid is not False:
            return gurobi_radius, "gurobi_proven_optimal"
    return None, "unproven"


def enrich_rows(rows: list[dict[str, Any]], tolerance: float = TOLERANCE) -> list[dict[str, Any]]:
    """Add the reference value, ratios, gaps and exact-match flags to each row."""
    enriched: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        n, k = _integer(row.get("n")), _integer(row.get("k"))
        dnn_radius = _number(row.get("dnn_radius"))
        dnn_valid = _flag(row.get("dnn_valid"))
        if dnn_valid is None:
            # CSVs from earlier versions have no validity column.
            dnn_valid = dnn_radius is not None and dnn_radius > 0
        optimum, source = _reference(row)
        bound = _number(row.get("gurobi_bound"))
        dnn_ratio = None
        dnn_gap_pct = None
        dnn_match = None
        if optimum is not None and optimum > 0:
            if dnn_valid and dnn_radius is not None:
                dnn_ratio = dnn_radius / optimum
                dnn_gap_pct = max(0.0, 1.0 - dnn_ratio) * 100.0
                dnn_match = int(dnn_ratio >= 1.0 - tolerance)
            else:
                # An invalid or missing solution counts as a miss when a
                # proven reference exists.
                dnn_match = 0

        dnn_bound_ratio = None
        dnn_bound_gap_pct = None
        if dnn_valid and dnn_radius is not None and bound is not None and bound > 0:
            dnn_bound_ratio = dnn_radius / bound
            dnn_bound_gap_pct = max(0.0, 1.0 - dnn_bound_ratio) * 100.0

        row.update({
            "n_int": n,
            "k_int": k,
            "dnn_valid_derived": int(bool(dnn_valid)),
            "reference_radius": optimum,
            "reference_source": source,
            "dnn_to_proven_optimum_ratio": dnn_ratio,
            "dnn_gap_to_proven_optimum_pct": dnn_gap_pct,
            "dnn_exact_match_derived": dnn_match,
            "dnn_to_gurobi_bound_ratio": dnn_bound_ratio,
            "dnn_gap_to_gurobi_bound_pct": dnn_bound_gap_pct,
        })
        enriched.append(row)
    return enriched


def _mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(clean) if clean else None


def _median(values: Iterable[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(clean) if clean else None


def _min(values: Iterable[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return min(clean) if clean else None


def _max(values: Iterable[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return max(clean) if clean else None


def _summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [r.get("dnn_to_proven_optimum_ratio") for r in group]
    gaps = [r.get("dnn_gap_to_proven_optimum_pct") for r in group]
    bound_gaps = [r.get("dnn_gap_to_gurobi_bound_pct") for r in group]
    proven = [r for r in group if r.get("reference_radius") is not None]
    valid_count = sum(int(r.get("dnn_valid_derived", 0)) for r in group)
    matches = [r["dnn_exact_match_derived"] for r in proven
               if r.get("dnn_exact_match_derived") is not None]
    gurobi_rows = [r for r in group if _flag(r.get("gurobi_optimal")) is not None
                   or r.get("gurobi_status")]
    gurobi_optimal_count = sum(_flag(r.get("gurobi_optimal")) is True for r in gurobi_rows)

    return {
        "instances": len(group),
        "replicates": len({str(r.get("replicate_seed", r.get("seed", ""))) for r in group}),
        "dnn_valid_count": valid_count,
        "dnn_valid_pct": 100.0 * valid_count / len(group) if group else None,
        "proven_reference_count": len(proven),
        "dnn_exact_match_count": sum(matches) if matches else None,
        "dnn_exact_match_pct": 100.0 * statistics.mean(matches) if matches else None,
        "dnn_ratio_mean": _mean(ratios),
        "dnn_ratio_median": _median(ratios),
        "dnn_ratio_worst": _min(ratios),
        "dnn_ratio_best": _max(ratios),
        "dnn_gap_mean_pct": _mean(gaps),
        "dnn_gap_worst_pct": _max(gaps),
        "dnn_bound_gap_mean_pct": _mean(bound_gaps),
        "dnn_runtime_mean_s": _mean(_number(r.get("t_dnn")) for r in group),
        "dnn_runtime_median_s": _median(_number(r.get("t_dnn")) for r in group),
        "greedy_ratio_mean": _mean(_number(r.get("greedy_ratio")) for r in group),
        "greedy_ratio_worst": _min(_number(r.get("greedy_ratio")) for r in group),
        "gurobi_optimal_count": gurobi_optimal_count if gurobi_rows else None,
        "gurobi_optimal_pct": 100.0 * gurobi_optimal_count / len(gurobi_rows) if gurobi_rows else None,
        "gurobi_solver_runtime_mean_s": _mean(_number(r.get("gurobi_runtime")) for r in group),
        "gurobi_wall_time_mean_s": _mean(_number(r.get("gurobi_wall_time")) for r in group),
        "gurobi_mip_gap_mean": _mean(_number(r.get("gurobi_gap")) for r in group),
    }


def summarize(enriched: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Aggregate per (distribution, n, k), per distribution and overall."""
    by_nk: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    by_dist: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        dist = str(row.get("dist", "unknown"))
        n, k = row.get("n_int"), row.get("k_int")
        if n is None or k is None:
            continue
        by_nk[(dist, n, k)].append(row)
        by_dist[dist].append(row)

    group_rows = []
    for (dist, n, k), group in sorted(by_nk.items()):
        group_rows.append({"dist": dist, "n": n, "k": k, **_summarize_group(group)})

    dist_rows = []
    for dist, group in sorted(by_dist.items()):
        dist_rows.append({"dist": dist, **_summarize_group(group)})

    all_stats = _summarize_group(enriched)
    total_gurobi = [r for r in enriched if r.get("gurobi_status")]
    status_counts: dict[str, int] = defaultdict(int)
    for row in total_gurobi:
        status_counts[str(row.get("gurobi_status"))] += 1
    all_stats["gurobi_status_counts"] = dict(sorted(status_counts.items()))
    all_stats["instances_total"] = len(enriched)
    all_stats["distributions"] = sorted({str(r.get("dist")) for r in enriched})
    all_stats["sizes"] = sorted({r.get("n_int") for r in enriched if r.get("n_int") is not None})
    all_stats["k_values"] = sorted({r.get("k_int") for r in enriched if r.get("k_int") is not None})
    return group_rows, dist_rows, all_stats


def audit_point_sets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check the point-set fingerprints for consistency.

    Rows with the same (distribution, n, replicate) must share one point set
    across all k, and in independent-size runs different (distribution, n,
    replicate) combinations must not share a point set.
    """
    audit: list[dict[str, Any]] = []
    if not any(str(r.get("point_set_hash", "")).strip() for r in rows):
        return [{"check": "point_set_hash_available", "status": "NOT_AVAILABLE",
                 "details": "CSV has no point_set_hash column/values; dataset independence cannot be audited."}]

    dataset_hashes: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    hash_owners: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    seed_modes = {str(r.get("seed_mode", "")).strip() for r in rows if r.get("seed_mode")}
    for row in rows:
        dist = str(row.get("dist", ""))
        n = str(row.get("n", ""))
        replicate = str(row.get("replicate_seed", row.get("seed", "")))
        identity = (dist, n, replicate)
        fingerprint = str(row.get("point_set_hash", "")).strip()
        if fingerprint:
            dataset_hashes[identity].add(fingerprint)
            hash_owners[fingerprint].add(identity)

    inconsistent = [(key, values) for key, values in dataset_hashes.items() if len(values) != 1]
    duplicate_hashes = [(fingerprint, owners) for fingerprint, owners in hash_owners.items() if len(owners) > 1]
    independent_mode = "independent_sizes" in seed_modes
    audit.append({
        "check": "one_point_set_per_distribution_size_replicate",
        "status": "PASS" if not inconsistent else "FAIL",
        "details": (f"{len(dataset_hashes)} dataset identities checked; each should have one hash across k. "
                    f"Inconsistent identities: {len(inconsistent)}."),
    })
    audit.append({
        "check": "no_hash_reused_across_dataset_identities",
        "status": ("PASS" if not duplicate_hashes else "FAIL") if independent_mode else "NOT_APPLICABLE",
        "details": (f"Independent-size mode: {independent_mode}. "
                    f"Repeated hashes across distinct identities: {len(duplicate_hashes)}."),
    })
    audit.append({
        "check": "independent_size_mode_declared",
        "status": "PASS" if independent_mode else "WARNING",
        "details": f"seed_mode values found: {', '.join(sorted(seed_modes)) or 'not recorded'}.",
    })
    return audit


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV, using the union of all keys as the header."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (ValueError, TypeError):
        return str(value)


def write_markdown_report(path: Path, stats: dict[str, Any], groups: list[dict[str, Any]], audit: list[dict[str, Any]], input_path: Path) -> None:
    """Write summary.md with the overall table, per-group table and audit."""
    lines = [
        "# k-dispersion experiment analysis",
        "",
        f"**Input:** `{input_path}`  ",
        f"**Instances:** {stats.get('instances_total', 0)}  ",
        f"**Distributions:** {', '.join(stats.get('distributions', []))}  ",
        f"**n values:** {', '.join(map(str, stats.get('sizes', [])))}  ",
        f"**k values:** {', '.join(map(str, stats.get('k_values', [])))}",
        "",
        "## Overall results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Valid DNN witnesses | {stats.get('dnn_valid_count', 0)}/{stats.get('instances_total', 0)} ({_fmt(stats.get('dnn_valid_pct'), 2)}%) |",
        f"| Instances with proven reference | {stats.get('proven_reference_count', 0)} |",
        f"| DNN exact-match rate (proven-reference subset) | {_fmt(stats.get('dnn_exact_match_pct'), 2)}% |",
        f"| Mean DNN / proven optimum | {_fmt(stats.get('dnn_ratio_mean'), 6)} |",
        f"| Worst DNN / proven optimum | {_fmt(stats.get('dnn_ratio_worst'), 6)} |",
        f"| Mean DNN gap to proven optimum | {_fmt(stats.get('dnn_gap_mean_pct'), 3)}% |",
        f"| Mean DNN conservative gap to Gurobi bound | {_fmt(stats.get('dnn_bound_gap_mean_pct'), 3)}% |",
        f"| Mean DNN runtime | {_fmt(stats.get('dnn_runtime_mean_s'), 3)} s |",
        f"| Mean Gurobi solver runtime | {_fmt(stats.get('gurobi_solver_runtime_mean_s'), 4)} s |",
        f"| Gurobi proven-optimal count | {stats.get('gurobi_optimal_count', '-')}/{stats.get('instances_total', 0)} |",
        "",
        "## Results by distribution, n and k",
        "",
        "| Distribution | n | k | Runs | DNN valid % | Proven refs | Exact match % | Mean ratio | Worst ratio | Mean DNN s | Mean Gurobi s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        lines.append(
            f"| {g['dist']} | {g['n']} | {g['k']} | {g['instances']} | "
            f"{_fmt(g.get('dnn_valid_pct'), 1)} | {g.get('proven_reference_count', 0)} | "
            f"{_fmt(g.get('dnn_exact_match_pct'), 1)} | {_fmt(g.get('dnn_ratio_mean'), 5)} | "
            f"{_fmt(g.get('dnn_ratio_worst'), 5)} | {_fmt(g.get('dnn_runtime_mean_s'), 2)} | "
            f"{_fmt(g.get('gurobi_solver_runtime_mean_s'), 4)} |"
        )
    lines.extend(["", "## Dataset integrity audit", "", "| Check | Status | Details |", "|---|---|---|"])
    for item in audit:
        details = str(item.get("details", "")).replace("|", "\\|")
        lines.append(f"| {item['check']} | **{item['status']}** | {details} |")
    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- A DNN/optimum ratio is reported only when the CSV contains a custom exact radius or Gurobi has proved optimality.",
        "- For time-limited/non-optimal Gurobi runs, the incumbent is not labelled as the optimum. The DNN-to-bound metric is conservative because the Gurobi bound is an upper bound on the optimum.",
        "- Runtimes depend on hardware and settings (CPU only, single-threaded Gurobi).",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


# Figure style: serif text, full frame and inward ticks. The colours are
# distinguishable under colour-vision deficiency, and each series also has
# its own marker so that the figures remain readable in greyscale.
INK = "#000000"
MUTED = "#4d4d4d"
GRID = "#d9d9d9"
SERIES = ["#1f5fae", "#d1572a", "#2e8b57"]
MARKERS = ["o", "s", "^"]


def _style(plt) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "axes.labelsize": 9,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.edgecolor": INK,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "patch.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, out_dir: Path, stem: str, created: list[str]) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    fig.savefig(out_dir / f"{stem}.pdf")
    created.append(f"{stem}.png")


def make_plots(enriched: list[dict[str, Any]], groups: list[dict[str, Any]], out_dir: Path) -> list[str]:
    """Create all figures in out_dir and return the PNG file names."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed - skipping plots (pip install matplotlib)")
        return []

    _style(plt)
    created: list[str] = []
    dists = sorted({g["dist"] for g in groups})
    letters = "abcdefgh"

    # Quality vs n: one panel per distribution, one line per k.
    # Error bars span the worst and best replicate; lines are offset slightly
    # in n so the bars don't sit on top of each other.
    eligible = [g for g in groups if g.get("dnn_ratio_mean") is not None]
    if eligible:
        panels = [d for d in dists if any(g["dist"] == d for g in eligible)]
        ks = sorted({g["k"] for g in eligible})
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.4),
                                 sharey=True, squeeze=False)
        worst_all = min((g.get("dnn_ratio_worst") or g["dnn_ratio_mean"]) for g in eligible)
        all_optimal = worst_all >= 1.0 - TOLERANCE
        lo = 0.98 if all_optimal else worst_all - 0.01
        for p_i, (ax, dist) in enumerate(zip(axes[0], panels)):
            for i, k in enumerate(ks):
                data = sorted([g for g in eligible if g["dist"] == dist and g["k"] == k],
                              key=lambda g: g["n"])
                if not data:
                    continue
                xs = [g["n"] for g in data]
                ys = [g["dnn_ratio_mean"] for g in data]
                below = [max(0.0, y - (g.get("dnn_ratio_worst") or y)) for y, g in zip(ys, data)]
                above = [max(0.0, (g.get("dnn_ratio_max") or y) - y) for y, g in zip(ys, data)]
                shift = (i - (len(ks) - 1) / 2) * 0.6
                c = SERIES[i % len(SERIES)]
                ax.errorbar([x + shift for x in xs], ys, yerr=[below, above], color=c,
                            marker=MARKERS[i % 3], elinewidth=0.6, capsize=1.8,
                            capthick=0.6, label=f"$k={k}$")
            ax.axhline(1.0, color=MUTED, linewidth=0.5, linestyle="--", zorder=0)
            ax.set_xlabel("Number of points $n$")
            ax.set_xticks(sorted({g["n"] for g in eligible}))
            ax.set_ylim(lo, 1.0 + (1.0 - lo) * 0.06)
            ax.grid(True, axis="y")
            ax.set_title(f"({letters[p_i]}) {dist.capitalize()}")
            if all_optimal:
                ax.text(0.5, 0.4, "all instances optimal", transform=ax.transAxes,
                        ha="center", color=MUTED, fontsize=8, style="italic")
        axes[0][0].set_ylabel("$x_{\\mathrm{dNN}}\\,/\\,x^{*}$")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(ks),
                   bbox_to_anchor=(0.5, 1.07))
        fig.tight_layout()
        _save(fig, out_dir, "quality_by_n", created)
        plt.close(fig)

    # Runtime vs n (log scale): color = method, line/marker = distribution.
    runtime_rows = [g for g in groups if g.get("dnn_runtime_mean_s") is not None]
    if runtime_rows:
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        styles = [("-", "o"), ("--", "s")]
        for j, dist in enumerate(sorted({g["dist"] for g in runtime_rows})):
            by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for g in runtime_rows:
                if g["dist"] == dist:
                    by_n[g["n"]].append(g)
            xs = sorted(by_n)
            dnn = [_mean(r.get("dnn_runtime_mean_s") for r in by_n[n]) for n in xs]
            gur = [_mean(r.get("gurobi_solver_runtime_mean_s") for r in by_n[n]) for n in xs]
            ls, mk = styles[j % 2]
            ax.plot(xs, dnn, color=SERIES[0], linestyle=ls, marker=mk, label=f"dNN ({dist})")
            if any(v is not None for v in gur):
                ax.plot(xs, gur, color=SERIES[1], linestyle=ls, marker=mk,
                        markerfacecolor="white", label=f"Gurobi ({dist})")
            ax.set_xticks(xs)
        ax.set_yscale("log")
        ax.set_xlabel("Number of points $n$")
        ax.set_ylabel("Mean runtime (s)")
        ax.grid(True, axis="y", which="major")
        ax.legend(loc="center right", ncol=1)
        fig.tight_layout()
        _save(fig, out_dir, "runtime_by_n", created)
        plt.close(fig)

    # Mean gap heatmap per distribution (k rows, n columns). Uses the proven
    # optimum when available, otherwise the conservative gap to Gurobi's bound.
    for dist in dists:
        data = [g for g in groups if g["dist"] == dist and (
            g.get("dnn_gap_mean_pct") is not None or g.get("dnn_bound_gap_mean_pct") is not None)]
        if not data:
            continue
        ns = sorted({g["n"] for g in data})
        ks = sorted({g["k"] for g in data})
        matrix = np.full((len(ks), len(ns)), np.nan)
        for g in data:
            v = g.get("dnn_gap_mean_pct")
            if v is None:
                v = g.get("dnn_bound_gap_mean_pct")
            matrix[ks.index(g["k"]), ns.index(g["n"])] = v
        uses_bound_only = all(g.get("dnn_gap_mean_pct") is None for g in data)
        vmax = max(1.0, float(np.nanmax(matrix)))
        fig, ax = plt.subplots(figsize=(0.7 * len(ns) + 1.4, 0.5 * len(ks) + 0.9))
        image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="Greys",
                          vmin=0.0, vmax=vmax * 1.15)
        ax.set_xticks(range(len(ns)), labels=[str(n) for n in ns])
        ax.set_yticks(range(len(ks)), labels=[str(k) for k in ks])
        ax.tick_params(which="both", length=0, top=False, right=False)
        ax.set_xlabel("Number of points $n$")
        ax.set_ylabel("$k$")
        for i in range(len(ks)):
            for j in range(len(ns)):
                if not np.isnan(matrix[i, j]):
                    dark = matrix[i, j] > 0.6 * vmax
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if dark else INK)
        cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
        what = "gap to Gurobi bound (%)" if uses_bound_only else "Mean gap to optimum (%)"
        cbar.set_label(what)
        cbar.ax.tick_params(direction="in", length=2, width=0.6)
        cbar.outline.set_linewidth(0.6)
        ax.set_title(dist.capitalize())
        fig.tight_layout()
        _save(fig, out_dir, f"optimality_gap_{dist}", created)
        plt.close(fig)

    # Per-instance gap histogram (proven references only).
    gaps = [r.get("dnn_gap_to_proven_optimum_pct") for r in enriched
            if r.get("dnn_gap_to_proven_optimum_pct") is not None]
    if gaps:
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        top = max(gaps)
        if top > 0:
            w = top / 12
            bins = np.arange(-w / 2, top + w, w)
        else:
            bins = np.array([-0.15, 0.15])
        ax.hist(gaps, bins=bins, color="#bfbfbf", edgecolor=INK, linewidth=0.6)
        ax.set_xlabel("Gap to optimal $x$ (%)")
        ax.set_ylabel("Number of instances")
        ax.grid(True, axis="y")
        exact = sum(1 for v in gaps if v <= TOLERANCE * 100)
        ax.text(0.97, 0.93, f"{exact} of {len(gaps)} exactly optimal", transform=ax.transAxes,
                ha="right", va="top", fontsize=8)
        if top == 0:
            ax.set_xticks([0])
            ax.set_xlim(-1, 1)
        fig.tight_layout()
        _save(fig, out_dir, "dnn_optimality_gap_distribution", created)
        plt.close(fig)

    return created


def analyze(input_path: str | Path, out_dir: str | Path | None = None,
            tolerance: float = TOLERANCE, plots: bool = True) -> Path:
    """Analyse one results CSV and write all tables and figures."""
    input_path = Path(input_path)
    if out_dir is None:
        out_dir = input_path.parent / f"analysis_{input_path.stem}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_results(input_path)
    enriched = enrich_rows(raw, tolerance=tolerance)
    groups, distributions, stats = summarize(enriched)
    audit = audit_point_sets(raw)

    _write_csv(out_dir / "per_instance_analysis.csv", enriched)
    _write_csv(out_dir / "summary_by_n_k.csv", groups)
    _write_csv(out_dir / "summary_by_distribution.csv", distributions)
    _write_csv(out_dir / "data_integrity_audit.csv", audit)

    plot_files = make_plots(enriched, groups, out_dir) if plots else []
    write_markdown_report(out_dir / "summary.md", stats, groups, audit, input_path)

    metadata = [
        {"metric": "instances_total", "value": stats.get("instances_total")},
        {"metric": "dnn_valid_pct", "value": stats.get("dnn_valid_pct")},
        {"metric": "proven_reference_count", "value": stats.get("proven_reference_count")},
        {"metric": "dnn_exact_match_pct", "value": stats.get("dnn_exact_match_pct")},
        {"metric": "dnn_ratio_mean", "value": stats.get("dnn_ratio_mean")},
        {"metric": "dnn_ratio_worst", "value": stats.get("dnn_ratio_worst")},
        {"metric": "dnn_gap_mean_pct", "value": stats.get("dnn_gap_mean_pct")},
        {"metric": "dnn_bound_gap_mean_pct", "value": stats.get("dnn_bound_gap_mean_pct")},
        {"metric": "dnn_runtime_mean_s", "value": stats.get("dnn_runtime_mean_s")},
        {"metric": "gurobi_solver_runtime_mean_s", "value": stats.get("gurobi_solver_runtime_mean_s")},
        {"metric": "gurobi_optimal_count", "value": stats.get("gurobi_optimal_count")},
        {"metric": "gurobi_status_counts", "value": stats.get("gurobi_status_counts")},
        {"metric": "plots_generated", "value": ";".join(plot_files)},
    ]
    _write_csv(out_dir / "overall_summary.csv", metadata)

    print(f"Analyzed {len(raw)} instances from {input_path}")
    print(f"Output directory: {out_dir}")
    print(f"Valid DNN witnesses: {stats.get('dnn_valid_count')}/{stats.get('instances_total')}")
    print(f"Proven references: {stats.get('proven_reference_count')}/{stats.get('instances_total')}")
    print(f"Mean DNN/proven-optimum ratio: {_fmt(stats.get('dnn_ratio_mean'), 6)}")
    print(f"Worst DNN/proven-optimum ratio: {_fmt(stats.get('dnn_ratio_worst'), 6)}")
    print(f"Mean DNN runtime: {_fmt(stats.get('dnn_runtime_mean_s'), 3)} s")
    print(f"Mean Gurobi solver runtime: {_fmt(stats.get('gurobi_solver_runtime_mean_s'), 4)} s")
    print(f"Plots generated: {len(plot_files)}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze k-dispersion experiment CSV results.")
    parser.add_argument("--input", required=True, help="Per-instance results CSV from eval/run_eval.py")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: eval/analysis_<input-stem>)")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE,
                        help="Relative tolerance for exact-match classification (default: 1e-6)")
    parser.add_argument("--no-plots", action="store_true", help="Write tables/report without requiring matplotlib")
    args = parser.parse_args()
    analyze(args.input, args.out_dir, tolerance=args.tolerance, plots=not args.no_plots)


if __name__ == "__main__":
    main()