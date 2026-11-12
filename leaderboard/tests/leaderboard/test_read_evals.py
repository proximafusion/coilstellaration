import json
import math

from leaderboard.src.display.utils import COLS, AutoEvalColumn
from leaderboard.src.leaderboard.read_evals import get_raw_eval_results, result_to_row


def _record(name, overall, with_strata=False, tracks=("fixed_shape", "variable_shape")):
    def track(mean, median, p10, feas, strict):
        t = {
            "mean": mean,
            "median": median,
            "p10": p10,
            "feasibility_rate_0p9": feas,
            "strict_feasibility": strict,
            "n": 2,
        }
        if with_strata:
            t["strata"] = {
                "tight": {
                    "mean": mean / 10,
                    "median": median / 10,
                    "p10": p10 / 10,
                    "feasibility_rate_0p9": 0.0,
                    "strict_feasibility": 0.0,
                    "n": 1,
                },
                "loose": {
                    "mean": mean * 2,
                    "median": median * 2,
                    "p10": p10 * 2,
                    "feasibility_rate_0p9": feas,
                    "strict_feasibility": strict,
                    "n": 1,
                },
            }
        return t

    # A track that wasn't scored is recorded as an empty dict.
    scored = {
        "fixed_shape": track(overall, 0.5, 0.2, 0.3, 0.1),
        "variable_shape": track(overall, 0.4, 0.1, 0.2, 0.0),
    }
    return {
        "model": f"org/{name}",
        "revision": "abc",
        "submitter": "alice",
        "submitted_time": "2026-06-11T00:00:00Z",
        "status": "FINISHED",
        "overall_score": overall,
        "tracks": {
            t: (scored[t] if t in tracks else {})
            for t in ("fixed_shape", "variable_shape")
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
    (tmp_path / "org" / "m2.json").write_text(
        json.dumps({"model": "org/m2", "status": "FAILED"})
    )
    recs = get_raw_eval_results(str(tmp_path))
    assert [r["model"] for r in recs] == ["org/m1"]


def test_result_to_row_all_stratum_uses_headline():
    row = result_to_row(_record("m1", 0.8, with_strata=True), "fixed_shape", "all")
    assert row[AutoEvalColumn.mean.name] == 0.8  # headline (all strata)
    assert row[AutoEvalColumn.feas09.name] == 0.3
    assert "org/m1" in row[AutoEvalColumn.model.name]
    for col in COLS:
        assert col in row


def test_result_to_row_picks_the_selected_track():
    rec = _record("m1", 0.8)
    fixed = result_to_row(rec, "fixed_shape", "all")
    var = result_to_row(rec, "variable_shape", "all")
    assert fixed[AutoEvalColumn.median.name] == 0.5  # fixed track's median
    assert var[AutoEvalColumn.median.name] == 0.4  # variable track's median


def test_result_to_row_specific_stratum_uses_breakdown():
    rec = _record("m1", 0.8, with_strata=True)
    row = result_to_row(rec, "fixed_shape", "tight")
    assert row[AutoEvalColumn.mean.name] == 0.08  # tight, not the headline


def test_result_to_row_none_when_track_unscored():
    # A fixed-only record has no data for the variable track -> no row there.
    rec = _record("m1", 0.8, tracks=("fixed_shape",))
    assert result_to_row(rec, "variable_shape", "all") is None
    assert result_to_row(rec, "fixed_shape", "all") is not None


def test_result_to_row_none_when_track_missing():
    # A record may omit an unscored track instead of storing it as an empty dict.
    rec = _record("m1", 0.8, with_strata=True)
    del rec["tracks"]["variable_shape"]
    assert result_to_row(rec, "variable_shape", "all") is None
    assert result_to_row(rec, "variable_shape", "tight") is None


def test_result_to_row_none_when_stratum_absent():
    # A record without a per-stratum breakdown has no data for a specific stratum.
    rec = _record("m1", 0.8, with_strata=False)
    assert result_to_row(rec, "fixed_shape", "tight") is None
    assert result_to_row(rec, "fixed_shape", "all") is not None


def test_result_to_row_defaults_when_fields_absent():
    rec = _record("m1", 0.8)
    row = result_to_row(rec, "fixed_shape", "all")
    assert row[AutoEvalColumn.commit.name] == ""
    assert row[AutoEvalColumn.onnx_file.name] == "model.onnx"


def test_result_to_row_reports_n_and_strict_standard_error():
    rec = {
        "model": "org/m",
        "tracks": {"fixed_shape": {"mean": 0.5, "strict_feasibility": 0.2, "n": 100}},
    }
    row = result_to_row(rec, "fixed_shape", "all")
    assert row[AutoEvalColumn.n.name] == 100
    # sqrt(0.2 * 0.8 / 100) = 0.04
    assert abs(row[AutoEvalColumn.strict_se.name] - 0.04) < 1e-12


def test_result_to_row_without_n_has_no_standard_error():
    rec = {"model": "org/m", "tracks": {"fixed_shape": {"strict_feasibility": 0.2}}}
    row = result_to_row(rec, "fixed_shape", "all")
    assert math.isnan(row[AutoEvalColumn.strict_se.name])
    assert math.isnan(row[AutoEvalColumn.n.name])


def test_result_to_row_links_user_profile_and_full_results(tmp_path):
    from leaderboard.src.envs import RESULTS_REPO

    (tmp_path / "org").mkdir()
    (tmp_path / "org" / "m_abc_model.json").write_text(
        json.dumps(
            {
                "model": "org/m",
                "submitter": "alice",
                "status": "FINISHED",
                "tracks": {"fixed_shape": {"mean": 0.5}},
            }
        )
    )
    [rec] = get_raw_eval_results(str(tmp_path))
    row = result_to_row(rec, "fixed_shape", "all")

    assert 'href="https://huggingface.co/alice"' in row[AutoEvalColumn.user.name]
    assert row[AutoEvalColumn.user_name.name] == "alice"
    url = f"https://huggingface.co/datasets/{RESULTS_REPO}/blob/main/org/m_abc_model.parquet"
    assert f'href="{url}"' in row[AutoEvalColumn.full_results.name]
    assert row[AutoEvalColumn.model.name] == "org/m"
