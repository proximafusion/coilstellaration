import json

from src.display.utils import BENCHMARK_COLS, COLS, EVAL_COLS
from src.populate import get_evaluation_queue_df, get_leaderboard_df


def test_empty_results_returns_columned_frame(tmp_path):
    # A fresh leaderboard (no result files) must yield an empty frame that still
    # carries the display columns, so the Leaderboard renders instead of crashing.
    df = get_leaderboard_df(str(tmp_path), str(tmp_path), COLS, BENCHMARK_COLS)
    assert df.empty
    assert list(df.columns) == COLS


def test_running_queue_surfaces_started_at(tmp_path):
    # A stuck RUNNING job should expose its claim time so it is visible in the UI.
    (tmp_path / "r.json").write_text(json.dumps({
        "model": "org/m", "revision": "main", "submitter": "a", "stochastic": False,
        "status": "RUNNING", "started_at": "2026-06-12T00:00:00+00:00",
    }))
    _finished, running, _pending = get_evaluation_queue_df(str(tmp_path), EVAL_COLS)
    assert "2026-06-12T00:00:00+00:00" in running["started"].tolist()
