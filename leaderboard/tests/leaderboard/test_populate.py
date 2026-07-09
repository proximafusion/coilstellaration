import json

from src.display.utils import AutoEvalColumn, BENCHMARK_COLS, COLS, EVAL_COLS, STRATUM_VIEWS
from src.populate import get_evaluation_queue_df, get_leaderboard_df, get_leaderboard_frames


def _write(tmp_path, name, overall, with_strata=True):
    def track(mean):
        t = {"mean": mean, "median": 0.5, "p10": 0.2,
             "feasibility_rate_0p9": 0.3, "strict_feasibility": 0.1, "n": 2}
        if with_strata:
            t["strata"] = {"tight": {"mean": mean / 10, "median": 0.05, "p10": 0.02,
                                     "feasibility_rate_0p9": 0.0, "strict_feasibility": 0.0, "n": 1}}
        return t
    rec = {
        "model": f"org/{name}", "revision": "abc", "submitter": "alice",
        "submitted_time": "2026-06-11T00:00:00Z", "stochastic": False, "k_samples": 1,
        "status": "FINISHED", "overall_score": overall,
        "tracks": {"fixed_shape": track(overall), "variable_shape": track(overall)},
    }
    sub = tmp_path / "org"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"{name}.json").write_text(json.dumps(rec))


def test_empty_results_returns_columned_frame(tmp_path):
    # A fresh leaderboard (no result files) must yield an empty frame that still
    # carries the display columns, so the Leaderboard renders instead of crashing.
    df = get_leaderboard_df(str(tmp_path), COLS, BENCHMARK_COLS)
    assert df.empty
    assert list(df.columns) == COLS


def test_frames_keyed_by_stratum_view(tmp_path):
    _write(tmp_path, "m1", 0.8, with_strata=True)
    frames = get_leaderboard_frames(str(tmp_path), COLS, BENCHMARK_COLS)
    assert set(frames) == set(STRATUM_VIEWS)
    # 'All' shows the headline mean; 'tight' shows the tight breakdown.
    assert frames["All"][AutoEvalColumn.fixed_mean.name].iloc[0] == 0.8
    assert frames["tight"][AutoEvalColumn.fixed_mean.name].iloc[0] == 0.08


def test_frame_drops_rows_missing_stratum(tmp_path):
    # A record with no per-stratum breakdown appears in 'All' but not in a
    # specific-stratum view (its metric is NaN there and gets filtered).
    _write(tmp_path, "m1", 0.8, with_strata=False)
    frames = get_leaderboard_frames(str(tmp_path), COLS, BENCHMARK_COLS)
    assert not frames["All"].empty
    assert frames["tight"].empty


def test_empty_results_frames_carry_columns(tmp_path):
    frames = get_leaderboard_frames(str(tmp_path), COLS, BENCHMARK_COLS)
    assert set(frames) == set(STRATUM_VIEWS)
    for view in STRATUM_VIEWS:
        assert frames[view].empty
        assert list(frames[view].columns) == COLS


def test_running_queue_surfaces_started_at():
    # A stuck RUNNING job should expose its claim time so it is visible in the UI.
    _finished, running, _pending = get_evaluation_queue_df(
        [{
            "model": "org/m", "revision": "main", "submitter": "a", "stochastic": False,
            "status": "RUNNING", "started_at": "2026-06-12T00:00:00+00:00",
        }],
        EVAL_COLS,
    )
    assert "2026-06-12T00:00:00+00:00" in running["started"].tolist()


def test_queue_buckets_by_status():
    records = [
        {"model": "org/p", "status": "PENDING"},
        {"model": "org/r", "status": "RUNNING"},
        {"model": "org/f", "status": "FINISHED"},
    ]
    finished, running, pending = get_evaluation_queue_df(records, EVAL_COLS)
    assert len(pending) == 1 and len(running) == 1 and len(finished) == 1


def test_queue_tolerates_missing_status():
    # A stray/partial record without a status must not crash the queue render.
    finished, running, pending = get_evaluation_queue_df([{"model": "org/x"}], EVAL_COLS)
    assert len(finished) == 0 and len(running) == 0 and len(pending) == 0


def test_queue_shows_supported_tracks():
    finished, _running, pending = get_evaluation_queue_df(
        [
            {"model": "org/f", "status": "PENDING", "tracks": ["fixed_shape"]},
            {"model": "org/b", "status": "PENDING",
             "tracks": ["fixed_shape", "variable_shape"]},
        ],
        EVAL_COLS,
    )
    assert set(pending["tracks"]) == {"fixed", "both"}


def test_queue_defaults_tracks_to_both_when_absent():
    _finished, _running, pending = get_evaluation_queue_df(
        [{"model": "org/legacy", "status": "PENDING"}], EVAL_COLS,
    )
    assert pending["tracks"].iloc[0] == "both"
