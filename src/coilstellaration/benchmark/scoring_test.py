"""Unit tests for `scoring`."""

import math
from unittest import mock

import pandas as pd
import pydantic
import pytest

from coilstellaration import types
from coilstellaration.benchmark import scoring
from coilstellaration.benchmark import types as benchmark_types


def _make_target(
    *,
    coil_to_coil: float = 0.5,
    coil_to_plasma: float = 0.4,
    curvature: float = 2.0,
    field_error: float = 0.01,
) -> benchmark_types.RequirementMetrics:
    return benchmark_types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=coil_to_coil,
        min_normalized_coil_to_plasma_distance=coil_to_plasma,
        max_normalized_coil_curvature=curvature,
        max_normalized_field_error=field_error,
    )


def _make_achieved_feasible() -> benchmark_types.RequirementMetrics:
    return _make_target(
        coil_to_coil=0.6,
        coil_to_plasma=0.5,
        curvature=1.5,
        field_error=0.005,
    )


# --- compute_violations ---


def test_compute_violations_feasible_is_zero():
    achieved = _make_achieved_feasible()
    target = _make_target()
    v = scoring.compute_violations(achieved, target, target_floor=1e-6)
    assert all(val == 0.0 for val in v.values())
    assert set(v) == set(benchmark_types.REQUIREMENT_METRIC_DIRECTIONS)


def test_compute_violations_upper_bound_violation_is_relative():
    target = _make_target(curvature=2.0)
    # Curvature 20% above the cap.
    achieved = _make_target(
        coil_to_coil=0.6, coil_to_plasma=0.5, curvature=2.4, field_error=0.005
    )
    v = scoring.compute_violations(achieved, target, target_floor=1e-6)
    assert v["max_normalized_coil_curvature"] == pytest.approx(0.2)
    assert v["min_normalized_coil_to_coil_distance"] == 0.0


def test_compute_violations_lower_bound_violation_is_relative():
    target = _make_target(coil_to_plasma=0.4)
    # Coil-to-plasma 25% below the floor (lower-bound violation).
    achieved = _make_target(
        coil_to_coil=0.6, coil_to_plasma=0.3, curvature=1.5, field_error=0.005
    )
    v = scoring.compute_violations(achieved, target, target_floor=1e-6)
    assert v["min_normalized_coil_to_plasma_distance"] == pytest.approx(0.25)


def test_compute_violations_target_floor_caps_near_zero_target():
    target = _make_target(field_error=1e-12)
    achieved = _make_target(
        coil_to_coil=0.6, coil_to_plasma=0.5, curvature=1.5, field_error=1e-9
    )
    # Without the floor v would be ~999. With floor=1e-3 it is (1e-9 - 1e-12)/1e-3.
    v = scoring.compute_violations(achieved, target, target_floor=1e-3)
    assert v["max_normalized_field_error"] == pytest.approx(
        (1e-9 - 1e-12) / 1e-3, rel=1e-9
    )


# --- apply_soft_score ---


def test_apply_soft_score_exponential_at_zero_and_alpha():
    settings = benchmark_types.ScoringSettings(soft_score="exponential")
    violations = {
        "min_normalized_coil_to_coil_distance": 0.0,
        "min_normalized_coil_to_plasma_distance": settings.tolerances[
            "min_normalized_coil_to_plasma_distance"
        ],
        "max_normalized_coil_curvature": 0.0,
        "max_normalized_field_error": 0.0,
    }
    s = scoring.apply_soft_score(violations, settings)
    assert s["min_normalized_coil_to_coil_distance"] == pytest.approx(1.0)
    assert s["min_normalized_coil_to_plasma_distance"] == pytest.approx(math.exp(-1.0))


def test_apply_soft_score_linear_ramp_hits_zero_at_alpha():
    settings = benchmark_types.ScoringSettings(soft_score="linear_ramp")
    alpha = settings.tolerances["max_normalized_field_error"]
    violations = {
        "min_normalized_coil_to_coil_distance": 0.0,
        "min_normalized_coil_to_plasma_distance": 0.0,
        "max_normalized_coil_curvature": 0.0,
        "max_normalized_field_error": alpha,
    }
    s = scoring.apply_soft_score(violations, settings)
    assert s["max_normalized_field_error"] == pytest.approx(0.0)
    # Beyond alpha the ramp clips to 0 (does not go negative).
    violations["max_normalized_field_error"] = 2 * alpha
    s = scoring.apply_soft_score(violations, settings)
    assert s["max_normalized_field_error"] == 0.0


# --- aggregate_metrics ---


def _all_ones_except(field: str, value: float) -> dict[str, float]:
    return {
        f: (value if f == field else 1.0)
        for f in benchmark_types.REQUIREMENT_METRIC_DIRECTIONS
    }


def test_aggregate_metrics_geometric_mean_zero_collapses_to_zero():
    settings = benchmark_types.ScoringSettings(metric_aggregation="geometric_mean")
    soft = _all_ones_except("max_normalized_field_error", 0.0)
    assert scoring.aggregate_metrics(soft, settings) == 0.0


def test_aggregate_metrics_geometric_mean_matches_formula():
    settings = benchmark_types.ScoringSettings(metric_aggregation="geometric_mean")
    soft = {
        "min_normalized_coil_to_coil_distance": 0.5,
        "min_normalized_coil_to_plasma_distance": 0.5,
        "max_normalized_coil_curvature": 0.5,
        "max_normalized_field_error": 0.5,
    }
    assert scoring.aggregate_metrics(soft, settings) == pytest.approx(0.5)


def test_aggregate_metrics_min():
    settings = benchmark_types.ScoringSettings(metric_aggregation="min")
    soft = _all_ones_except("max_normalized_coil_curvature", 0.3)
    assert scoring.aggregate_metrics(soft, settings) == pytest.approx(0.3)


def test_aggregate_metrics_arithmetic_mean():
    settings = benchmark_types.ScoringSettings(metric_aggregation="arithmetic_mean")
    soft = _all_ones_except("max_normalized_coil_curvature", 0.0)
    # Three 1.0s and a 0.0 average to 0.75 (the case the agent flagged).
    assert scoring.aggregate_metrics(soft, settings) == pytest.approx(0.75)


def test_aggregate_metrics_weighted_geometric_mean():
    weights = {
        "min_normalized_coil_to_coil_distance": 0.1,
        "min_normalized_coil_to_plasma_distance": 0.1,
        "max_normalized_coil_curvature": 0.2,
        "max_normalized_field_error": 0.6,
    }
    settings = benchmark_types.ScoringSettings(
        metric_aggregation="weighted_geometric_mean",
        metric_weights=weights,
    )
    soft = {
        "min_normalized_coil_to_coil_distance": 0.9,
        "min_normalized_coil_to_plasma_distance": 0.9,
        "max_normalized_coil_curvature": 0.9,
        "max_normalized_field_error": 0.5,
    }
    expected = math.exp(sum(weights[k] * math.log(soft[k]) for k in soft))
    assert scoring.aggregate_metrics(soft, settings) == pytest.approx(expected)


def test_aggregate_metrics_weighted_geometric_mean_zero_collapses():
    weights = {
        "min_normalized_coil_to_coil_distance": 0.25,
        "min_normalized_coil_to_plasma_distance": 0.25,
        "max_normalized_coil_curvature": 0.25,
        "max_normalized_field_error": 0.25,
    }
    settings = benchmark_types.ScoringSettings(
        metric_aggregation="weighted_geometric_mean",
        metric_weights=weights,
    )
    soft = _all_ones_except("max_normalized_field_error", 0.0)
    assert scoring.aggregate_metrics(soft, settings) == 0.0


# --- score_instance ---


def test_score_instance_feasible_yields_one_and_strictly_feasible():
    settings = benchmark_types.ScoringSettings()
    instance = scoring.score_instance(
        boundary_id="b0",
        achieved=_make_achieved_feasible(),
        target=_make_target(),
        settings=settings,
    )
    assert instance.boundary_id == "b0"
    assert instance.aggregate == pytest.approx(1.0)
    assert instance.metric_scores.strictly_feasible
    assert all(v == 0.0 for v in instance.metric_scores.violations.values())


def test_score_instance_infeasible_drops_aggregate():
    settings = benchmark_types.ScoringSettings()
    target = _make_target(field_error=0.01)
    achieved = _make_target(
        coil_to_coil=0.6,
        coil_to_plasma=0.5,
        curvature=1.5,
        field_error=0.02,  # 100% over the cap; well beyond alpha=0.25.
    )
    instance = scoring.score_instance(
        boundary_id="b1", achieved=achieved, target=target, settings=settings
    )
    assert not instance.metric_scores.strictly_feasible
    assert instance.aggregate < 1.0
    assert instance.metric_scores.violations[
        "max_normalized_field_error"
    ] == pytest.approx(1.0)


# --- aggregate_instances + score_benchmark ---


def _instance(
    boundary_id: str, aggregate: float, feasible: bool
) -> benchmark_types.InstanceScore:
    zeros = {f: 0.0 for f in benchmark_types.REQUIREMENT_METRIC_DIRECTIONS}
    return benchmark_types.InstanceScore(
        boundary_id=boundary_id,
        metric_scores=benchmark_types.MetricScores(
            violations=zeros,
            soft_scores=zeros,
            strictly_feasible=feasible,
        ),
        aggregate=aggregate,
    )


def test_aggregate_instances_reports_each_summary():
    settings = benchmark_types.ScoringSettings()
    instances = [
        _instance("a", 1.0, feasible=True),
        _instance("b", 0.95, feasible=False),
        _instance("c", 0.5, feasible=False),
        _instance("d", 0.0, feasible=False),
    ]
    summaries = scoring.aggregate_instances(instances, settings)
    assert summaries["mean"] == pytest.approx((1.0 + 0.95 + 0.5 + 0.0) / 4)
    assert summaries["median"] == pytest.approx((0.95 + 0.5) / 2)
    assert summaries["p10"] == pytest.approx(0.15)  # 10th percentile of the four.
    assert summaries["feasibility_rate_0p9"] == pytest.approx(0.5)
    assert summaries["strict_feasibility"] == pytest.approx(0.25)


def test_score_benchmark_round_trip():
    settings = benchmark_types.ScoringSettings()
    target = _make_target()
    achieved_good = _make_achieved_feasible()
    achieved_bad = _make_target(
        coil_to_coil=0.6, coil_to_plasma=0.5, curvature=1.5, field_error=0.02
    )
    instances = [
        scoring.score_instance("g", achieved_good, target, settings),
        scoring.score_instance("b", achieved_bad, target, settings),
    ]
    benchmark = scoring.score_benchmark(instances, settings)
    assert [i.boundary_id for i in benchmark.per_instance] == ["g", "b"]
    assert set(benchmark.summaries) == set(settings.instance_aggregations)
    assert benchmark.summaries["strict_feasibility"] == pytest.approx(0.5)


# --- ScoringSettings validation ---


def test_scoring_settings_rejects_missing_tolerance_fields():
    with pytest.raises(pydantic.ValidationError, match="tolerances keys"):
        benchmark_types.ScoringSettings(
            tolerances={"min_normalized_coil_to_coil_distance": 0.1},
        )


def test_scoring_settings_rejects_nonpositive_tolerance():
    bad = {f: 0.1 for f in benchmark_types.REQUIREMENT_METRIC_DIRECTIONS}
    bad["max_normalized_field_error"] = 0.0
    with pytest.raises(pydantic.ValidationError, match="strictly positive"):
        benchmark_types.ScoringSettings(tolerances=bad)


def test_scoring_settings_weighted_requires_weights():
    with pytest.raises(pydantic.ValidationError, match="requires metric_weights"):
        benchmark_types.ScoringSettings(metric_aggregation="weighted_geometric_mean")


def test_scoring_settings_weights_must_sum_to_one():
    with pytest.raises(pydantic.ValidationError, match="must sum to 1"):
        benchmark_types.ScoringSettings(
            metric_aggregation="weighted_geometric_mean",
            metric_weights={
                f: 0.1 for f in benchmark_types.REQUIREMENT_METRIC_DIRECTIONS
            },
        )


# --- score_eval_data ---


def _stub_eval_data(
    boundary_id: str,
    target: benchmark_types.RequirementMetrics,
) -> mock.MagicMock:
    eval_data = mock.MagicMock(spec=types.EvalData)
    eval_data.boundary_id = boundary_id
    eval_data.requirement_metrics = target
    return eval_data


def test_score_eval_data_uses_eval_data_target_and_id():
    settings = benchmark_types.ScoringSettings()
    target = _make_target()
    achieved = _make_achieved_feasible()
    eval_data = _stub_eval_data("boundary-42", target)

    instance = scoring.score_eval_data(eval_data, achieved, settings)

    assert instance.boundary_id == "boundary-42"
    assert instance.aggregate == pytest.approx(1.0)
    assert instance.metric_scores.strictly_feasible


def test_score_eval_data_matches_score_instance_directly():
    settings = benchmark_types.ScoringSettings()
    target = _make_target()
    achieved = _make_target(
        coil_to_coil=0.6, coil_to_plasma=0.5, curvature=1.5, field_error=0.02
    )
    eval_data = _stub_eval_data("b", target)

    via_eval = scoring.score_eval_data(eval_data, achieved, settings)
    direct = scoring.score_instance(
        boundary_id="b", achieved=achieved, target=target, settings=settings
    )
    assert via_eval == direct


# --- instance_score_to_row ---


def test_instance_score_to_row_keys_and_value_benchmark_types():
    settings = benchmark_types.ScoringSettings()
    instance = scoring.score_instance(
        boundary_id="b0",
        achieved=_make_achieved_feasible(),
        target=_make_target(),
        settings=settings,
    )

    row = scoring.instance_score_to_row(instance)

    # boundary_id deliberately omitted: dataframe carries it as its own column.
    assert "boundary_id" not in row
    assert set(row) == (
        {"score", "strictly_feasible"}
        | {f"violation/{f}" for f in benchmark_types.REQUIREMENT_METRIC_DIRECTIONS}
        | {f"soft_score/{f}" for f in benchmark_types.REQUIREMENT_METRIC_DIRECTIONS}
    )
    # Value benchmark_types match the slack contract: int | float | bool only.
    for k, v in row.items():
        assert isinstance(v, (int, float, bool)), f"{k} -> {type(v).__name__}"


def test_instance_score_to_row_values_match_instance():
    settings = benchmark_types.ScoringSettings()
    achieved = _make_target(
        coil_to_coil=0.6, coil_to_plasma=0.5, curvature=1.5, field_error=0.02
    )
    instance = scoring.score_instance(
        boundary_id="b1",
        achieved=achieved,
        target=_make_target(),
        settings=settings,
    )

    row = scoring.instance_score_to_row(instance)

    assert row["score"] == instance.aggregate
    assert row["strictly_feasible"] == instance.metric_scores.strictly_feasible
    for field in benchmark_types.REQUIREMENT_METRIC_DIRECTIONS:
        assert row[f"violation/{field}"] == instance.metric_scores.violations[field]
        assert row[f"soft_score/{field}"] == instance.metric_scores.soft_scores[field]


def test_instance_score_to_row_is_dataframe_friendly():
    settings = benchmark_types.ScoringSettings()
    instances = [
        scoring.score_instance(
            "good", _make_achieved_feasible(), _make_target(), settings
        ),
        scoring.score_instance(
            "bad",
            _make_target(
                coil_to_coil=0.6,
                coil_to_plasma=0.5,
                curvature=1.5,
                field_error=0.02,
            ),
            _make_target(),
            settings,
        ),
    ]

    df = pd.DataFrame(
        [
            {"boundary_id": s.boundary_id, **scoring.instance_score_to_row(s)}
            for s in instances
        ]
    )

    assert list(df["boundary_id"]) == ["good", "bad"]
    feasible = df["strictly_feasible"].to_numpy()
    scores = df["score"].to_numpy()
    assert feasible[0]
    assert not feasible[1]
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] < 1.0


# --- instance_scores_to_dataframe + summarize_scores ---


def _scored_pair(
    settings: benchmark_types.ScoringSettings,
) -> list[benchmark_types.InstanceScore]:
    """A 2-instance set: one feasible, one with a single bad metric."""
    target = _make_target()
    achieved_good = _make_achieved_feasible()
    achieved_bad = _make_target(
        coil_to_coil=0.6,
        coil_to_plasma=0.5,
        curvature=1.5,
        field_error=0.02,  # 100% over the cap.
    )
    return [
        scoring.score_instance("good", achieved_good, target, settings),
        scoring.score_instance("bad", achieved_bad, target, settings),
    ]


def test_instance_scores_to_dataframe_columns_and_shape():
    settings = benchmark_types.ScoringSettings()
    instances = _scored_pair(settings)

    df = scoring.instance_scores_to_dataframe(instances)

    assert len(df) == 2
    expected_columns = (
        {"boundary_id", "strictly_feasible"}
        | {f"violation/{f}" for f in benchmark_types.REQUIREMENT_METRIC_DIRECTIONS}
        | {f"soft_score/{f}" for f in benchmark_types.REQUIREMENT_METRIC_DIRECTIONS}
        | {"score/geometric_mean", "score/min", "score/arithmetic_mean"}
    )
    assert set(df.columns) == expected_columns
    assert list(df["boundary_id"]) == ["good", "bad"]


def test_instance_scores_to_dataframe_geometric_mean_matches_instance():
    """`score/geometric_mean` column equals the per-instance aggregate when settings use
    that aggregation."""
    settings = benchmark_types.ScoringSettings(metric_aggregation="geometric_mean")
    instances = _scored_pair(settings)

    df = scoring.instance_scores_to_dataframe(instances)

    np_assert = pytest.approx
    for inst, expected in zip(instances, df["score/geometric_mean"].to_numpy()):
        assert expected == np_assert(inst.aggregate)


def test_instance_scores_to_dataframe_min_and_arithmetic_consistent():
    """Min and arithmetic_mean values match what `aggregate_metrics` would compute."""
    settings = benchmark_types.ScoringSettings()
    instances = _scored_pair(settings)

    df = scoring.instance_scores_to_dataframe(instances)

    for inst, row in zip(instances, df.to_dict("records")):
        soft = inst.metric_scores.soft_scores
        assert row["score/min"] == pytest.approx(min(soft.values()))
        assert row["score/arithmetic_mean"] == pytest.approx(
            sum(soft.values()) / len(soft)
        )


def test_instance_scores_to_dataframe_weighted_column_only_when_weights_given():
    settings = benchmark_types.ScoringSettings()
    instances = _scored_pair(settings)

    df_no_weights = scoring.instance_scores_to_dataframe(instances)
    assert "score/weighted_geometric_mean" not in df_no_weights.columns

    weights = {
        "min_normalized_coil_to_coil_distance": 0.1,
        "min_normalized_coil_to_plasma_distance": 0.1,
        "max_normalized_coil_curvature": 0.2,
        "max_normalized_field_error": 0.6,
    }
    df_w = scoring.instance_scores_to_dataframe(instances, weights=weights)
    assert "score/weighted_geometric_mean" in df_w.columns
    # Verify formula on the first instance.
    soft = instances[0].metric_scores.soft_scores
    expected = math.exp(sum(weights[k] * math.log(soft[k]) for k in soft))
    assert df_w["score/weighted_geometric_mean"].iloc[0] == pytest.approx(expected)


def test_instance_scores_to_dataframe_rejects_bad_weights():
    settings = benchmark_types.ScoringSettings()
    instances = _scored_pair(settings)
    with pytest.raises(ValueError, match="must sum to 1"):
        scoring.instance_scores_to_dataframe(
            instances,
            weights={f: 0.1 for f in benchmark_types.REQUIREMENT_METRIC_DIRECTIONS},
        )


def test_summarize_scores_rows_and_columns():
    settings = benchmark_types.ScoringSettings()
    instances = _scored_pair(settings)
    df = scoring.instance_scores_to_dataframe(instances)

    summary = scoring.summarize_scores(df, feasibility_threshold=0.9)

    assert set(summary.index) == {"geometric_mean", "min", "arithmetic_mean"}
    assert {
        "mean",
        "median",
        "p10",
        "feasibility_rate_0p9",
        "strict_feasibility_rate",
    } == set(summary.columns)
    # strict_feasibility_rate is constant across rows (one instance feasible / two).
    assert (
        summary["strict_feasibility_rate"]
        .eq(summary["strict_feasibility_rate"].iloc[0])
        .all()
    )
    assert summary["strict_feasibility_rate"].iloc[0] == pytest.approx(0.5)


def test_summarize_scores_can_rank_aggregations():
    """For the test pair, arithmetic_mean is more lenient than geometric_mean."""
    settings = benchmark_types.ScoringSettings()
    instances = _scored_pair(settings)
    df = scoring.instance_scores_to_dataframe(instances)

    summary = scoring.summarize_scores(df)
    means = summary["mean"].to_dict()

    assert means["arithmetic_mean"] >= means["geometric_mean"]
    assert means["min"] <= means["geometric_mean"]


def test_summarize_scores_rejects_dataframe_without_score_columns():
    df = pd.DataFrame({"boundary_id": ["a"], "strictly_feasible": [True]})
    with pytest.raises(ValueError, match="score/"):
        scoring.summarize_scores(df)
