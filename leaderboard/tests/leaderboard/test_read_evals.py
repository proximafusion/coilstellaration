import json

from src.display.utils import AutoEvalColumn, COLS
from src.leaderboard.read_evals import get_raw_eval_results


def _write_result(dirpath, name, overall):
    record = {
        "model": f"org/{name}", "revision": "abc", "submitter": "alice",
        "submitted_time": "2026-06-11T00:00:00Z", "stochastic": True, "k_samples": 4,
        "status": "FINISHED", "overall_score": overall,
        "tracks": {
            "fixed_shape": {"mean": overall, "median": 0.5, "p10": 0.2,
                            "feasibility_rate_0p9": 0.3, "strict_feasibility": 0.1, "n": 2},
            "variable_shape": {"mean": overall, "median": 0.4, "p10": 0.1,
                               "feasibility_rate_0p9": 0.2, "strict_feasibility": 0.0, "n": 2},
        },
    }
    sub = dirpath / "org"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"{name}.json").write_text(json.dumps(record))


def test_parses_records_into_rows(tmp_path):
    _write_result(tmp_path, "m1", 0.8)
    rows = get_raw_eval_results(str(tmp_path))
    assert len(rows) == 1
    row = rows[0]
    assert row[AutoEvalColumn.overall.name] == 0.8
    assert row[AutoEvalColumn.fixed_feas09.name] == 0.3
    assert row[AutoEvalColumn.stochastic.name] == "✓ (K=4)"
    assert "org/m1" in row[AutoEvalColumn.model.name]  # clickable markdown
    # every displayed column is present
    for col in COLS:
        assert col in row


def test_skips_unfinished(tmp_path):
    _write_result(tmp_path, "m1", 0.8)
    bad = tmp_path / "org" / "m2.json"
    bad.write_text(json.dumps({"model": "org/m2", "status": "FAILED"}))
    rows = get_raw_eval_results(str(tmp_path))
    assert [r[AutoEvalColumn.model.name] for r in rows if "m2" in r[AutoEvalColumn.model.name]] == []
