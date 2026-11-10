import json

from leaderboard.src.display.utils import (
    COLS,
    EVAL_COLS,
    STRATUM_VIEWS,
    TRACKS,
    AutoEvalColumn,
)
from leaderboard.src.populate import (
    get_evaluation_queue_df,
    get_leaderboard_df,
    get_leaderboard_frames,
)


def _write(
    tmp_path,
    name,
    mean,
    with_strata=True,
    tracks=("fixed_shape", "variable_shape"),
    strict=0.1,
):
    def track(m):
        t = {
            "mean": m,
            "median": 0.5,
            "p10": 0.2,
            "feasibility_rate_0p9": 0.3,
            "strict_feasibility": strict,
            "n": 2,
        }
        if with_strata:
            t["strata"] = {
                "tight": {
                    "mean": m / 10,
                    "median": 0.05,
                    "p10": 0.02,
                    "feasibility_rate_0p9": 0.0,
                    "strict_feasibility": strict / 10,
                    "n": 1,
                }
            }
        return t

    # A track that wasn't scored is recorded as an empty dict.
    rec = {
        "model": f"org/{name}",
        "revision": "abc",
        "submitter": "alice",
        "submitted_time": "2026-06-11T00:00:00Z",
        "stochastic": False,
        "k_samples": 1,
        "status": "FINISHED",
        "overall_score": mean,
        "tracks": {
            t: (track(mean) if t in tracks else {})
            for t in ("fixed_shape", "variable_shape")
        },
    }
    sub = tmp_path / "org"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"{name}.json").write_text(json.dumps(rec))


ALL_KEYS = [(t, s) for t in TRACKS for s in STRATUM_VIEWS]


def test_empty_results_returns_columned_frame(tmp_path):
    # A fresh leaderboard (no result files) must yield an empty frame that still
    # carries the display columns, so the Leaderboard renders instead of crashing.
    df = get_leaderboard_df(str(tmp_path), COLS)
    assert df.empty
    assert list(df.columns) == COLS


def test_frames_keyed_by_track_and_stratum(tmp_path):
    _write(tmp_path, "m1", 0.8, with_strata=True)
    frames = get_leaderboard_frames(str(tmp_path), COLS)
    assert set(frames) == set(ALL_KEYS)
    # 'All' shows the headline mean; 'tight' shows the tight breakdown.
    assert frames[("fixed_shape", "All")][AutoEvalColumn.mean.name].iloc[0] == 0.8
    assert frames[("fixed_shape", "tight")][AutoEvalColumn.mean.name].iloc[0] == 0.08


def test_fixed_only_absent_from_variable_frames(tmp_path):
    # A submission scored only on fixed_shape shows in the fixed frames and in
    # none of the variable frames (no data there).
    _write(tmp_path, "fixedonly", 0.7, tracks=("fixed_shape",))
    frames = get_leaderboard_frames(str(tmp_path), COLS)
    assert not frames[("fixed_shape", "All")].empty
    assert frames[("variable_shape", "All")].empty


def test_frame_drops_rows_missing_stratum(tmp_path):
    # A record with no per-stratum breakdown appears in 'All' but not in a
    # specific-stratum view (no data there).
    _write(tmp_path, "m1", 0.8, with_strata=False)
    frames = get_leaderboard_frames(str(tmp_path), COLS)
    assert not frames[("fixed_shape", "All")].empty
    assert frames[("fixed_shape", "tight")].empty


def test_rows_sorted_by_strict_then_mean(tmp_path):
    # Higher strict feasibility ranks first; mean breaks ties.
    _write(tmp_path, "hi_strict", 0.1, strict=0.9)
    _write(tmp_path, "lo_strict_hi_mean", 0.9, strict=0.2)
    _write(tmp_path, "lo_strict_lo_mean", 0.5, strict=0.2)
    frames = get_leaderboard_frames(str(tmp_path), COLS)
    models = frames[("fixed_shape", "All")][AutoEvalColumn.model.name].tolist()
    order = [
        m
        for name in ["hi_strict", "lo_strict_hi_mean", "lo_strict_lo_mean"]
        for m in models
        if name in m
    ]
    assert order == models


def test_empty_results_frames_carry_columns(tmp_path):
    frames = get_leaderboard_frames(str(tmp_path), COLS)
    assert set(frames) == set(ALL_KEYS)
    for key in ALL_KEYS:
        assert frames[key].empty
        assert list(frames[key].columns) == COLS


def test_running_queue_surfaces_started_at():
    # A stuck RUNNING job should expose its claim time so it is visible in the UI.
    _finished, running, _pending, _failed = get_evaluation_queue_df(
        [
            {
                "model": "org/m",
                "revision": "main",
                "submitter": "a",
                "stochastic": False,
                "status": "RUNNING",
                "started_at": "2026-06-12T00:00:00+00:00",
            }
        ],
        EVAL_COLS,
    )
    assert "2026-06-12T00:00:00+00:00" in running["started"].tolist()


def test_queue_buckets_by_status():
    records = [
        {"model": "org/p", "status": "PENDING"},
        {"model": "org/r", "status": "RUNNING"},
        {"model": "org/f", "status": "FINISHED"},
    ]
    finished, running, pending, _failed = get_evaluation_queue_df(records, EVAL_COLS)
    assert len(pending) == 1 and len(running) == 1 and len(finished) == 1


def test_queue_tolerates_missing_status():
    # A stray/partial record without a status must not crash the queue render.
    finished, running, pending, failed = get_evaluation_queue_df(
        [{"model": "org/x"}], EVAL_COLS
    )
    assert (
        len(finished) == 0
        and len(running) == 0
        and len(pending) == 0
        and len(failed) == 0
    )


def test_queue_shows_supported_tracks():
    finished, _running, pending, _failed = get_evaluation_queue_df(
        [
            {"model": "org/f", "status": "PENDING", "tracks": ["fixed_shape"]},
            {
                "model": "org/b",
                "status": "PENDING",
                "tracks": ["fixed_shape", "variable_shape"],
            },
        ],
        EVAL_COLS,
    )
    assert set(pending["tracks"]) == {"fixed", "both"}


def test_queue_defaults_tracks_to_both_when_absent():
    _finished, _running, pending, _failed = get_evaluation_queue_df(
        [{"model": "org/legacy", "status": "PENDING"}],
        EVAL_COLS,
    )
    assert pending["tracks"].iloc[0] == "both"
