import pandas as pd

from leaderboard.src.display.leaderboard import search_rows
from leaderboard.src.leaderboard.read_evals import result_to_row


def _frame():
    records = [
        {
            "model": model,
            "submitter": submitter,
            "tracks": {"fixed_shape": {"mean": 0.5, "strict_feasibility": 0.1}},
        }
        for model, submitter in [("org/alpha", "alice"), ("lab/beta", "bob")]
    ]
    return pd.DataFrame([result_to_row(r, "fixed_shape", "All") for r in records])


def test_search_matches_plain_model_name_not_link_markup():
    df = _frame()
    columns = ["model_name", "Submitter"]
    # The Model column's link contains "huggingface" and "href" in every row.
    assert search_rows(df, "huggingface", columns).empty
    assert search_rows(df, "href", columns).empty
    assert list(search_rows(df, "ALPHA", columns)["model_name"]) == ["org/alpha"]
    assert list(search_rows(df, "bob", columns)["model_name"]) == ["lab/beta"]


def test_empty_query_keeps_every_row():
    df = _frame()
    assert len(search_rows(df, "", ["model_name"])) == 2
