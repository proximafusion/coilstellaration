from src.display.utils import BENCHMARK_COLS, COLS, EVAL_COLS
from src.populate import get_evaluation_queue_df, get_leaderboard_df


def test_empty_results_returns_columned_frame(tmp_path):
    # A fresh leaderboard (no result files) must yield an empty frame that still
    # carries the display columns, so the Leaderboard renders instead of crashing.
    df = get_leaderboard_df(str(tmp_path), COLS, BENCHMARK_COLS)
    assert df.empty
    assert list(df.columns) == COLS


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
