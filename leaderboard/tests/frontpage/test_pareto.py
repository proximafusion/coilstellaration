import math
from typing import Any

import numpy as np

from leaderboard.src.frontpage import pareto


def _record(onnx_path, mean, strict, time, n_parameters=None, strata=None):
    track = {
        "mean": mean,
        "strict_feasibility": strict,
        "inference_time": {"mean": time},
        "n": 10,
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


def _ranked(records, score="mean", cost="inference time"):
    df = pareto.view_frame(records, "fixed_shape", "all")
    return pareto.rank(df, score, cost)


def test_front_ranks_peels_successive_fronts():
    cost = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 5.0])
    score = np.array([0.1, 0.3, 0.2, 0.5, 0.3, 0.1])
    # 3 is beaten by 2 (cheaper, better); 6 by 1, 2, 3 and 4. 2 and 5 are equal,
    # so they share front 1.
    assert list(pareto.front_ranks(cost, score)) == [1, 1, 2, 1, 1, 3]


def test_front_ranks_leaves_out_missing_and_zero_costs():
    cost = np.array([1.0, np.nan, 0.0, 2.0])
    score = np.array([0.1, 0.9, 0.9, np.nan])
    assert list(pareto.front_ranks(cost, score)) == [1, 0, 0, 0]


def test_front_ranks_of_nothing_is_empty():
    assert len(pareto.front_ranks(np.array([]), np.array([]))) == 0


def test_rank_orders_by_front_then_score_with_unranked_last():
    records = [
        _record("slow_best.onnx", mean=0.9, strict=0.1, time=1.0),
        _record("fast.onnx", mean=0.2, strict=0.2, time=1e-4),
        _record("mid.onnx", mean=0.5, strict=0.05, time=1e-2),
        _record("beaten.onnx", mean=0.4, strict=0.05, time=2e-2),
        _record("old.onnx", mean=0.99, strict=0.9, time=None),
    ]
    ranked = _ranked(records)
    assert list(ranked["onnx file"]) == [
        "slow_best.onnx",
        "mid.onnx",
        "fast.onnx",
        "beaten.onnx",
        "old.onnx",
    ]
    assert ranked[pareto.FRONT].tolist()[:4] == [1, 1, 1, 2]
    assert ranked[pareto.FRONT].isna().tolist() == [False] * 4 + [True]
    # The front column comes first.
    assert ranked.columns[0] == pareto.FRONT


def test_rank_follows_the_score_and_cost_axes():
    records = [
        _record("a.onnx", 0.5, 0.1, 1e-3, n_parameters=10),
        _record("b.onnx", 0.4, 0.2, 1e-4, n_parameters=1000),
    ]
    by_time = _ranked(records, "mean", "inference time")
    assert by_time[pareto.FRONT].tolist() == [1, 1]
    by_strict = _ranked(records, "strict", "inference time")
    assert list(by_strict["onnx file"]) == ["b.onnx", "a.onnx"]
    assert by_strict[pareto.FRONT].tolist() == [1, 2]
    by_params = _ranked(records, "mean", "params")
    assert list(by_params["onnx file"]) == ["a.onnx", "b.onnx"]
    assert by_params[pareto.FRONT].tolist() == [1, 2]


def test_view_frame_skips_tracks_and_strata_without_data():
    strata = {"loose": {"mean": 0.9, "strict_feasibility": 0.5}}
    records = [_record("a.onnx", 0.5, 0.1, 1e-3, strata=strata)]
    assert len(pareto.view_frame(records, "fixed_shape", "loose")) == 1
    assert pareto.view_frame(records, "fixed_shape", "tight").empty
    assert pareto.view_frame(records, "variable_shape", "all").empty


def test_short_name_shares_one_budget_between_owner_and_repo():
    assert pareto.LABEL_MAX == 28
    assert pareto.short_name("org/my-model") == "org/my-model"
    # "coilstellaration" is abbreviated, in any case, before counting.
    assert pareto.short_name("phanerozoic/coilstellaration-direct") == (
        "phanerozoic/CS-direct"
    )
    assert pareto.short_name("Coilstellaration/x") == "CS/x"
    assert pareto.short_name("CoilStellaration/COILSTELLARATION-y") == "CS/CS-y"
    # A short owner leaves its unused half to the repo.
    assert pareto.short_name("phanerozoic/coilstellaration-circle-baseline") == (
        "phanerozoic/CS-circle-baselin..."
    )
    # Two long parts get half each.
    assert pareto.short_name("o" * 20 + "/" + "r" * 20) == (
        "o" * 14 + ".../" + "r" * 14 + "..."
    )
    # A long owner can use what a short repo leaves.
    assert pareto.short_name("o" * 30 + "/abc") == "o" * 25 + ".../abc"
    assert pareto.short_name("no-owner") == "no-owner"


def _three():
    return _ranked(
        [
            _record("a.onnx", 0.5, 0.1, 1e-3),
            _record("b.onnx", 0.4, 0.2, 1e-4),
            _record("c.onnx", 0.3, 0.05, 2e-3),
        ]
    )


def test_figure_draws_the_front_as_the_second_trace():
    # The dark-mode CSS recolours the front by its position, the second trace.
    fig: Any = pareto.figure(_three(), "mean", "inference time")
    assert [t.name for t in fig.data] == [pareto.DOMINATED_TRACE, pareto.FRONT_TRACE]
    assert list(fig.data[1].customdata[:, 0]) == [  # front, cheapest first
        "org/models · b.onnx",
        "org/models · a.onnx",
    ]
    assert list(fig.data[1].text) == ["org/models", "org/models"]
    assert fig.layout.xaxis.type == "log"
    assert not math.isnan(fig.data[0].y[0])
    assert not fig.layout.shapes


def test_figure_with_a_budget_fades_points_over_it_and_rings_the_best():
    # Times are 1, 0.1 and 2 ms; a 0.5 ms budget leaves only b.
    fig: Any = pareto.figure(_three(), "mean", "inference time", budget=0.5)
    assert [t.name for t in fig.data][2] == pareto.BEST_TRACE
    assert list(fig.data[2].customdata[:, 0]) == ["org/models · b.onnx"]
    assert list(fig.data[1].marker.opacity) == [1.0, pareto.OVER_BUDGET_OPACITY]
    # The budget line sits at the budget itself; Plotly handles the log axis.
    # It has no annotation, which Plotly would place by the log of its position.
    assert fig.layout.shapes[0].x0 == 0.5
    assert fig.layout.shapes[0].showlegend
    assert not fig.layout.annotations


def test_note_names_the_best_within_budget_and_counts_unranked():
    ranked = _ranked(
        [
            _record("a.onnx", 0.5, 0.1, 1e-3),
            _record("b.onnx", 0.4, 0.2, 1e-4),
            _record("old.onnx", 0.9, 0.9, None),
        ]
    )
    note = pareto.note_markdown(ranked, "mean", "inference time", budget=2)
    assert "**org/models · a.onnx**" in note
    assert "1 result without a recorded inference time" in note
    note = pareto.note_markdown(ranked, "mean", "inference time", budget=0.01)
    assert "No ranked result" in note
    assert pareto.note_markdown(ranked.iloc[:2], "mean", "inference time") == ""


def test_note_footnotes_the_abbreviation_only_when_a_label_uses_it():
    record = _record("a.onnx", 0.5, 0.1, 1e-3)
    ranked = _ranked([record])
    assert pareto.ABBREVIATION_NOTE not in pareto.note_markdown(
        ranked, "mean", "inference time"
    )
    ranked = _ranked([{**record, "model": "org/coilstellaration-direct"}])
    note = pareto.note_markdown(ranked, "mean", "inference time")
    assert note.endswith(pareto.ABBREVIATION_NOTE)
