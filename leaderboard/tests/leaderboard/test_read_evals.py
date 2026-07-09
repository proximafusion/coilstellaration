import json

from src.display.utils import AutoEvalColumn, COLS
from src.leaderboard.read_evals import get_raw_eval_results, result_to_row


def _record(name, overall, with_strata=False):
    def track(mean, median, p10, feas, strict):
        t = {"mean": mean, "median": median, "p10": p10,
             "feasibility_rate_0p9": feas, "strict_feasibility": strict, "n": 2}
        if with_strata:
            t["strata"] = {
                "tight": {"mean": mean / 10, "median": median / 10, "p10": p10 / 10,
                          "feasibility_rate_0p9": 0.0, "strict_feasibility": 0.0, "n": 1},
                "loose": {"mean": mean * 2, "median": median * 2, "p10": p10 * 2,
                          "feasibility_rate_0p9": feas, "strict_feasibility": strict, "n": 1},
            }
        return t
    return {
        "model": f"org/{name}", "revision": "abc", "submitter": "alice",
        "submitted_time": "2026-06-11T00:00:00Z", "stochastic": True, "k_samples": 4,
        "status": "FINISHED", "overall_score": overall,
        "tracks": {
            "fixed_shape": track(overall, 0.5, 0.2, 0.3, 0.1),
            "variable_shape": track(overall, 0.4, 0.1, 0.2, 0.0),
        },
    }


def _write(dirpath, name, overall, with_strata=False):
    sub = dirpath / "org"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"{name}.json").write_text(json.dumps(_record(name, overall, with_strata)))


def test_get_raw_eval_results_returns_finished_records(tmp_path):
    _write(tmp_path, "m1", 0.8)
    recs = get_raw_eval_results(str(tmp_path))
    assert len(recs) == 1
    assert recs[0]["model"] == "org/m1"
    assert recs[0]["overall_score"] == 0.8


def test_get_raw_eval_results_skips_unfinished(tmp_path):
    _write(tmp_path, "m1", 0.8)
    (tmp_path / "org" / "m2.json").write_text(json.dumps({"model": "org/m2", "status": "FAILED"}))
    recs = get_raw_eval_results(str(tmp_path))
    assert [r["model"] for r in recs] == ["org/m1"]


def test_result_to_row_all_stratum_uses_headline(tmp_path):
    row = result_to_row(_record("m1", 0.8, with_strata=True), stratum="All")
    assert row[AutoEvalColumn.overall.name] == 0.8
    assert row[AutoEvalColumn.fixed_mean.name] == 0.8          # headline (all strata)
    assert row[AutoEvalColumn.fixed_feas09.name] == 0.3
    assert row[AutoEvalColumn.stochastic.name] == "✓ (K=4)"
    assert "org/m1" in row[AutoEvalColumn.model.name]
    for col in COLS:
        assert col in row


def test_result_to_row_specific_stratum_uses_breakdown(tmp_path):
    rec = _record("m1", 0.8, with_strata=True)
    row = result_to_row(rec, stratum="tight")
    # tight numbers, not the headline
    assert row[AutoEvalColumn.fixed_mean.name] == 0.08
    assert row[AutoEvalColumn.var_mean.name] == 0.08
    # Overall for a stratum is the mean across tracks of that stratum's mean.
    assert row[AutoEvalColumn.overall.name] == 0.08


def test_result_to_row_missing_stratum_is_nan(tmp_path):
    # An old record without a `strata` breakdown yields NaN metrics for a
    # specific stratum (so the frame-builder can drop it from that view).
    import math
    rec = _record("m1", 0.8, with_strata=False)
    row = result_to_row(rec, stratum="tight")
    assert math.isnan(row[AutoEvalColumn.fixed_mean.name])


def test_result_to_row_defaults_when_fields_absent():
    rec = _record("m1", 0.8)
    del rec["stochastic"]
    row = result_to_row(rec)
    assert row[AutoEvalColumn.commit.name] == ""
    assert row[AutoEvalColumn.onnx_file.name] == "model.onnx"


def test_result_to_row_labels_both_tracks():
    row = result_to_row(_record("m1", 0.8))
    assert row[AutoEvalColumn.tracks.name] == "both"


def test_result_to_row_labels_fixed_only():
    rec = _record("m1", 0.8)
    rec["tracks"]["variable_shape"] = {}   # unscored track
    row = result_to_row(rec)
    assert row[AutoEvalColumn.tracks.name] == "fixed"
