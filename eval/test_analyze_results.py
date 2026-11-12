import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from analyze_results import audit_point_sets, enrich_rows, summarize


def sample_rows():
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


def test_enrich_uses_proven_gurobi_optimum():
    enriched = enrich_rows(sample_rows())
    assert enriched[0]["reference_source"] == "gurobi_proven_optimal"
    assert abs(enriched[0]["dnn_to_proven_optimum_ratio"] - 0.9) < 1e-12
    assert abs(enriched[0]["dnn_gap_to_proven_optimum_pct"] - 10.0) < 1e-12


def test_unproven_incumbent_is_not_called_optimum():
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


def test_dataset_hash_audit_passes_expected_k_reuse_and_size_independence():
    audit = audit_point_sets(sample_rows())
    assert all(item["status"] == "PASS" for item in audit)


def test_dataset_hash_audit_detects_mismatch_across_k():
    rows = sample_rows()
    rows[1]["point_set_hash"] = "different-hash"
    audit = audit_point_sets(rows)
    assert audit[0]["status"] == "FAIL"


def test_summary_counts_and_grouping():
    enriched = enrich_rows(sample_rows())
    groups, distributions, overall = summarize(enriched)
    assert len(groups) == 3
    assert len(distributions) == 1
    assert overall["instances_total"] == 3
    assert overall["proven_reference_count"] == 3
    assert overall["dnn_valid_count"] == 3
