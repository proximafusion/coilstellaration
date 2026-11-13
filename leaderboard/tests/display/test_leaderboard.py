import pandas as pd

from leaderboard.src.display.leaderboard import search_rows
from leaderboard.src.leaderboard.read_evals import result_to_row


def _frame():
    records = [
        {
            "model": model,
            "submitter": submitter,
            "raw_scores_file": f"{model}_abc_model.parquet",
            "tracks": {"fixed_shape": {"mean": 0.5, "strict_feasibility": 0.1}},
        }
        for model, submitter in [("org/alpha", "alice"), ("lab/beta", "bob")]
    ]
    return pd.DataFrame([result_to_row(r, "fixed_shape", "all") for r in records])


def test_search_matches_plain_names_not_link_markup():
    df = _frame()
    columns = ["model id", "user_name"]
    # The user and full results links contain "huggingface" and "href" in every
    # row; search matches the plain model id and username instead.
    assert search_rows(df, "huggingface", columns).empty
    assert search_rows(df, "href", columns).empty
    assert list(search_rows(df, "ALPHA", columns)["model id"]) == ["org/alpha"]
    assert list(search_rows(df, "bob", columns)["model id"]) == ["lab/beta"]


def test_empty_query_keeps_every_row():
    df = _frame()
    assert len(search_rows(df, "", ["model id"])) == 2


def test_style_frame_marks_column_maxima_and_rounds():
    from leaderboard.src.display.leaderboard import BEST_VALUE_CSS, style_frame

    df = pd.DataFrame({"Model": ["a", "b"], "mean": [0.5, 0.512], "p10": [0.2, None]})
    styled = style_frame(df, ["mean", "p10", "not-shown"])
    html = styled.to_html()
    assert "0.500" in html
    assert "0.512" in html
    # One highlighted cell per metric column: b's mean and a's p10.
    # Private pandas API, so not in pandas-stubs; it's how pandas' own tests
    # read a Styler's per-cell CSS.
    ctx = styled._compute().ctx  # pyright: ignore[reportAttributeAccessIssue]
    highlighted = {cell for cell, props in ctx.items() if props}
    assert highlighted == {(1, 1), (0, 2)}
    # Bolder than the Medium headers, on a pale surface; never teal text.
    assert "font-weight: 600" in BEST_VALUE_CSS


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


def test_write_csv_turns_links_into_urls():
    from leaderboard.src.display.leaderboard import write_csv
    from leaderboard.src.envs import RESULTS_REPO

    df = _frame()[["user", "full results", "model id", "mean"]]
    path = write_csv(df, "board-fixed_shape-all")
    assert path.endswith("board-fixed_shape-all.csv")
    out = pd.read_csv(path)
    # A "link" cell becomes the URL it points to; other links become their text.
    base = f"https://huggingface.co/datasets/{RESULTS_REPO}/blob/main"
    assert list(out["full results"]) == [
        f"{base}/org/alpha_abc_model.parquet",
        f"{base}/lab/beta_abc_model.parquet",
    ]
    assert list(out["user"]) == ["alice", "bob"]
    assert list(out["model id"]) == ["org/alpha", "lab/beta"]
    assert list(out["mean"]) == [0.5, 0.5]


def test_leaderboard_opens_on_default_view():
    import gradio as gr

    from leaderboard.src.display.leaderboard import Leaderboard
    from leaderboard.src.display.utils import DEFAULT_VIEW, STRATUM_VIEWS, TRACKS

    # Each frame names its own view, so the rendered table shows which one loaded.
    frames = {
        (t, s): pd.DataFrame({"view": [f"{t}/{s}"]})
        for t in TRACKS
        for s in STRATUM_VIEWS
    }
    with gr.Blocks() as demo:
        table = Leaderboard(
            value=frames[DEFAULT_VIEW],
            frames=frames,
            tracks=TRACKS,
            strata=STRATUM_VIEWS,
            track_labels={t: t for t in TRACKS},
            default_view=DEFAULT_VIEW,
        )
    assert table.value["data"] == [["/".join(DEFAULT_VIEW)]]
    radios = {b.label: b.value for b in demo.blocks.values() if isinstance(b, gr.Radio)}
    assert radios == {"Track": DEFAULT_VIEW[0], "Stratum": DEFAULT_VIEW[1]}


def test_default_view_is_variable_track_tight_stratum():
    from leaderboard.src.display.utils import DEFAULT_VIEW

    assert DEFAULT_VIEW == ("variable_shape", "tight")


def _named_endpoints(demo):
    """The API endpoints `demo` publishes: name to (parameters, returns)."""
    info = demo.get_api_info(all_endpoints=True)["named_endpoints"]
    return {
        name: (
            [p["parameter_name"] for p in api["parameters"]],
            [r["label"] for r in api["returns"]],
        )
        for name, api in info.items()
    }


def test_leaderboard_publishes_one_view_endpoint():
    import gradio as gr

    from leaderboard.src.display.leaderboard import Leaderboard
    from leaderboard.src.display.utils import DEFAULT_VIEW, STRATUM_VIEWS, TRACKS

    frames = {
        (t, s): pd.DataFrame({"model id": ["org/alpha"]})
        for t in TRACKS
        for s in STRATUM_VIEWS
    }
    with gr.Blocks() as demo:
        Leaderboard(
            value=frames[DEFAULT_VIEW],
            frames=frames,
            tracks=TRACKS,
            strata=STRATUM_VIEWS,
            default_view=DEFAULT_VIEW,
            search_columns=["model id"],
            label="Leaderboard",
        )
    assert _named_endpoints(demo) == {
        "/leaderboard": (
            ["columns", "search", "track", "stratum"],
            ["Leaderboard", "Download CSV"],
        )
    }


def test_leaderboard_reads_frames_from_a_function_on_each_view():
    import gradio as gr

    from leaderboard.src.display.leaderboard import Leaderboard

    current = {(None, None): pd.DataFrame({"model id": ["org/alpha"]})}
    with gr.Blocks() as demo:
        table = Leaderboard(
            value=current[(None, None)],
            frames=lambda: current,
            reload_on=demo.load,
        )
    assert table.value["data"] == [["org/alpha"]]
    # The data changes while the app runs; the next view shows it.
    current[(None, None)] = pd.DataFrame({"model id": ["org/alpha", "lab/beta"]})
    view = next(
        fn.fn
        for fn in demo.fns.values()
        if any(trigger == "load" for _, trigger in fn.targets)
    )
    assert view is not None
    update, _csv = view(["model id"], "", None, None)
    assert len(update["value"].data) == 2
