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


def test_style_frame_marks_column_maxima_and_rounds():
    from leaderboard.src.display.leaderboard import BEST_VALUE_CSS, style_frame

    df = pd.DataFrame({"Model": ["a", "b"], "mean": [0.5, 0.512], "p10": [0.2, None]})
    styled = style_frame(df, ["mean", "p10", "not-shown"])
    html = styled.to_html()
    assert "0.500" in html
    assert "0.512" in html
    # One highlighted cell per metric column: b's mean and a's p10.
    ctx = styled._compute().ctx
    highlighted = {cell for cell, props in ctx.items() if props}
    assert highlighted == {(1, 1), (0, 2)}
    assert BEST_VALUE_CSS.startswith("font-weight: bold")


def test_style_frame_formats_counts_as_integers_and_other_numbers_to_3dp():
    from leaderboard.src.display.leaderboard import style_frame

    df = pd.DataFrame(
        {"strict SE": [0.0412345], "n": [2771.0], "Params": [float("nan")]}
    )
    html = style_frame(df, [], ["n", "Params"]).to_html()
    assert "0.041" in html
    assert "2,771" in html
    assert "2771.0" not in html
    assert "nan" not in html


def test_write_csv_strips_links_to_their_text():
    from leaderboard.src.display.leaderboard import write_csv

    df = _frame()[["Model", "Submitter", "mean"]]
    path = write_csv(df, "board-fixed_shape-All")
    assert path.endswith("board-fixed_shape-All.csv")
    out = pd.read_csv(path)
    assert list(out["Model"]) == ["org/alpha", "lab/beta"]
    assert list(out["mean"]) == [0.5, 0.5]
