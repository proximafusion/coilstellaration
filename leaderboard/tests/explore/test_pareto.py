import math
from typing import Any

import numpy as np

from leaderboard.src.explore import pareto


def _record(onnx_path, mean, strict, time, n_parameters=None, strata=None):
    track = {
        "mean": mean,
        "strict_feasibility": strict,
        "inference_time": {"mean": time},
    }
    if strata:
        track["strata"] = strata
    return {
        "model": "org/models",
        "onnx_path": onnx_path,
        "submitter": "alice",
        "n_parameters": n_parameters,
        "tracks": {"fixed_shape": track, "variable_shape": {}},
    }


def test_pareto_mask_keeps_points_no_other_point_beats():
    cost = np.array([1.0, 2.0, 3.0, 4.0, 2.0])
    score = np.array([0.1, 0.3, 0.2, 0.5, 0.3])
    # 3 costs more than 2 for a lower score; the second (2, 0.3) ties the first.
    assert list(pareto.pareto_mask(cost, score)) == [True, True, False, True, False]


def test_pareto_mask_of_nothing_is_empty():
    assert len(pareto.pareto_mask(np.array([]), np.array([]))) == 0


def test_points_reads_the_view_and_marks_the_front():
    records = [
        _record("a.onnx", mean=0.5, strict=0.1, time=1e-3),
        _record("b.onnx", mean=0.4, strict=0.2, time=1e-4),
        _record("c.onnx", mean=0.3, strict=0.05, time=2e-3),
    ]
    df, missing = pareto.points(records, "fixed_shape", "all", "mean", "inference time")
    assert missing == 0
    assert list(df["x"]) == [1.0, 0.1, 2.0]  # milliseconds
    assert dict(zip(df["name"].str.split(" · ").str[1], df["on_front"])) == {
        "a.onnx": True,
        "b.onnx": True,
        "c.onnx": False,
    }
    df, _ = pareto.points(records, "fixed_shape", "all", "strict", "inference time")
    assert dict(zip(df["name"].str.split(" · ").str[1], df["on_front"])) == {
        "a.onnx": False,
        "b.onnx": True,
        "c.onnx": False,
    }


def test_points_counts_results_without_a_cost():
    records = [
        _record("a.onnx", 0.5, 0.1, 1e-3, n_parameters=1000),
        _record("b.onnx", 0.4, 0.2, 1e-4),
    ]
    df, missing = pareto.points(records, "fixed_shape", "all", "mean", "params")
    assert list(df["name"].str.split(" · ").str[1]) == ["a.onnx"]
    assert df["x"].iloc[0] == 1000
    assert missing == 1


def test_points_skips_tracks_and_strata_without_data():
    strata = {"loose": {"mean": 0.9, "strict_feasibility": 0.5}}
    records = [_record("a.onnx", 0.5, 0.1, 1e-3, strata=strata)]
    # The stratum has no inference time recorded, so it counts as missing.
    df, missing = pareto.points(
        records, "fixed_shape", "loose", "mean", "inference time"
    )
    assert df.empty
    assert missing == 1
    # A stratum the record doesn't have, and a track that wasn't scored, are
    # skipped without counting as missing.
    df, missing = pareto.points(records, "fixed_shape", "tight", "mean", "params")
    assert df.empty
    assert missing == 0
    df, missing = pareto.points(records, "variable_shape", "all", "mean", "params")
    assert df.empty
    assert missing == 0


def test_figure_draws_the_front_as_the_second_trace():
    # The dark-mode CSS recolours the front by its position, the second trace.
    records = [
        _record("a.onnx", 0.5, 0.1, 1e-3),
        _record("b.onnx", 0.4, 0.2, 1e-4),
        _record("c.onnx", 0.3, 0.05, 2e-3),
    ]
    df, _ = pareto.points(records, "fixed_shape", "all", "mean", "inference time")
    # plotly builds trace and layout attributes at runtime; pyright can't see them.
    fig: Any = pareto.figure(df, "inference time", "mean")
    assert [t.name for t in fig.data] == [pareto.DOMINATED, pareto.FRONT]
    assert list(fig.data[1].customdata[:, 0]) == [  # front, cheapest first
        "org/models · b.onnx",
        "org/models · a.onnx",
    ]
    table = pareto.front_markdown(df, "inference time", "mean")
    assert table.index("b.onnx") < table.index("a.onnx")
    assert "c.onnx" not in table
    assert fig.layout.xaxis.type == "log"
    assert not math.isnan(fig.data[0].y[0])
