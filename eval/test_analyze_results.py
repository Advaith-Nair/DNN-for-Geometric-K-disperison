"""
Tests for the analysis script: choice of reference value and dataset audit.
"""

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from analyze_results import audit_point_sets, enrich_rows, summarize


def sample_rows():
    """Three result rows: two values of k on one point set and one larger instance."""
    common = {
        "dist": "uniform", "seed_mode": "independent_sizes", "replicate_seed": "10",
        "dnn_valid": "1", "dnn_radius": "0.9", "t_dnn": "2.0",
        "gurobi_valid": "1", "gurobi_optimal": "1", "gurobi_status": "OPTIMAL",
        "gurobi_radius": "1.0", "gurobi_bound": "1.0", "gurobi_runtime": "0.1",
    }
    return [
        {**common, "n": "20", "k": "3", "point_set_hash": "hash-a"},
        {**common, "n": "20", "k": "4", "point_set_hash": "hash-a"},
        {**common, "n": "30", "k": "3", "point_set_hash": "hash-b", "replicate_seed": "10",
         "dnn_radius": "0.95", "gurobi_radius": "1.0", "gurobi_bound": "1.0"},
    ]


def test_proven_gurobi_optimum_is_used_as_reference():
    """A Gurobi value proven optimal is used as the reference."""
    enriched = enrich_rows(sample_rows())
    assert enriched[0]["reference_source"] == "gurobi_proven_optimal"
    assert abs(enriched[0]["dnn_to_proven_optimum_ratio"] - 0.9) < 1e-12
    assert abs(enriched[0]["dnn_gap_to_proven_optimum_pct"] - 10.0) < 1e-12


def test_unproven_gurobi_value_is_not_used_as_reference():
    """A Gurobi value from a time-limited run is not treated as optimal."""
    row = sample_rows()[0].copy()
    row["gurobi_optimal"] = "0"
    row["gurobi_status"] = "TIME_LIMIT"
    row["gurobi_radius"] = "0.95"
    row["gurobi_bound"] = "1.0"
    enriched = enrich_rows([row])[0]
    assert enriched["reference_radius"] is None
    assert enriched["reference_source"] == "unproven"
    assert enriched["dnn_to_proven_optimum_ratio"] is None
    assert abs(enriched["dnn_gap_to_gurobi_bound_pct"] - 10.0) < 1e-12


def test_audit_passes_on_consistent_data():
    """The audit passes when every k uses the same point set."""
    audit = audit_point_sets(sample_rows())
    assert all(item["status"] == "PASS" for item in audit)


def test_audit_detects_different_point_sets_across_k():
    """The audit fails when two values of k use different point sets."""
    rows = sample_rows()
    rows[1]["point_set_hash"] = "different-hash"
    audit = audit_point_sets(rows)
    assert audit[0]["status"] == "FAIL"


def test_summary_counts():
    """Group and overall counts are computed correctly."""
    enriched = enrich_rows(sample_rows())
    groups, distributions, overall = summarize(enriched)
    assert len(groups) == 3
    assert len(distributions) == 1
    assert overall["instances_total"] == 3
    assert overall["proven_reference_count"] == 3
    assert overall["dnn_valid_count"] == 3