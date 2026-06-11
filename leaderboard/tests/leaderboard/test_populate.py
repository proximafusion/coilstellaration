from src.display.utils import BENCHMARK_COLS, COLS
from src.populate import get_leaderboard_df


def test_empty_results_returns_columned_frame(tmp_path):
    # A fresh leaderboard (no result files) must yield an empty frame that still
    # carries the display columns, so the Leaderboard renders instead of crashing.
    df = get_leaderboard_df(str(tmp_path), str(tmp_path), COLS, BENCHMARK_COLS)
    assert df.empty
    assert list(df.columns) == COLS
