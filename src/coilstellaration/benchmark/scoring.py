"""Soft-feasibility scoring of predicted coilsets against `RequirementMetrics`.

See `_types.ScoringSettings` for the knobs and the package `README.md` for
the full design rationale. Pipeline:

1. `compute_violations`: per-metric normalized violation `v_i >= 0`.
2. `apply_soft_score`: saturating map `v_i -> s_i in [0, 1]`.
3. `aggregate_metrics`: per-metric scores -> per-instance scalar.
4. `aggregate_instances`: per-instance scalars -> headline + diagnostics.

Top-level entry points are `score_instance`, `score_eval_data`, and
`score_benchmark`. Use `instance_score_to_row` to flatten an `InstanceScore`
into a dict suitable for an analysis dataframe.
"""

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from coilstellaration import types
from coilstellaration.benchmark import _types


def score_instance(
    boundary_id: str,
    achieved: types.RequirementMetrics,
    target: types.RequirementMetrics,
    settings: _types.ScoringSettings,
) -> _types.InstanceScore:
    """End-to-end scoring of one test instance.

    Args:
        boundary_id: Identifier propagated onto the returned `InstanceScore`.
        achieved: Metrics evaluated on the predicted coilset.
        target: Requirement thresholds for this test case.
        settings: Scoring knobs (soft map, tolerances, metric aggregation).

    Returns:
        `InstanceScore` with per-metric violations + soft scores and the
        aggregated per-instance scalar.
    """
    violations = compute_violations(achieved, target, settings.target_floor)
    soft = apply_soft_score(violations, settings)
    aggregate = aggregate_metrics(soft, settings)
    metric_scores = _types.MetricScores(
        violations=violations,
        soft_scores=soft,
        strictly_feasible=all(v == 0.0 for v in violations.values()),
    )
    return _types.InstanceScore(
        boundary_id=boundary_id,
        metric_scores=metric_scores,
        aggregate=aggregate,
    )


def score_eval_data(
    eval_data: types.EvalData,
    achieved: types.RequirementMetrics,
    settings: _types.ScoringSettings,
) -> _types.InstanceScore:
    """`score_instance` driven by an `EvalData`.

    Uses `eval_data.boundary_id` and `eval_data.requirement_metrics` (the
    target) so callers only have to supply the achieved metrics evaluated on
    the predicted coilset.
    """
    return score_instance(
        boundary_id=eval_data.boundary_id,
        achieved=achieved,
        target=eval_data.requirement_metrics,
        settings=settings,
    )


def score_benchmark(
    instance_scores: Iterable[_types.InstanceScore],
    settings: _types.ScoringSettings,
) -> _types.BenchmarkScore:
    """Aggregate per-instance scores into headline + diagnostic summaries."""
    instances = list(instance_scores)
    return _types.BenchmarkScore(
        per_instance=instances,
        summaries=aggregate_instances(instances, settings),
    )


def compute_violations(
    achieved: types.RequirementMetrics,
    target: types.RequirementMetrics,
    target_floor: float,
) -> dict[str, float]:
    """Normalized violation per metric. `0.0` means feasible at-or-beyond threshold.

    For upper-bound constraints `v_i = max(0, (g - tau) / max(|tau|, floor))`;
    for lower-bound constraints `v_i = max(0, (tau - g) / max(|tau|, floor))`.
    """
    violations: dict[str, float] = {}
    for field, direction in _types.REQUIREMENT_METRIC_DIRECTIONS.items():
        g = float(getattr(achieved, field))
        tau = float(getattr(target, field))
        denom = max(abs(tau), target_floor)
        if direction == "upper_bound":
            violations[field] = max(0.0, (g - tau) / denom)
        else:
            violations[field] = max(0.0, (tau - g) / denom)
    return violations


def apply_soft_score(
    violations: dict[str, float],
    settings: _types.ScoringSettings,
) -> dict[str, float]:
    """Map `v_i >= 0` to `s_i in [0, 1]` per `settings.soft_score`."""
    soft: dict[str, float] = {}
    for field, v in violations.items():
        alpha = settings.tolerances[field]
        if settings.soft_score == "exponential":
            soft[field] = math.exp(-v / alpha)
        else:
            soft[field] = max(0.0, 1.0 - v / alpha)
    return soft


def aggregate_metrics(
    soft_scores: dict[str, float],
    settings: _types.ScoringSettings,
) -> float:
    """Collapse per-metric soft scores to a per-instance scalar."""
    fields = list(soft_scores)
    values = np.asarray([soft_scores[f] for f in fields], dtype=float)
    match settings.metric_aggregation:
        case "geometric_mean":
            # Log-space for stability; an exact zero collapses the product to 0.
            if np.any(values <= 0.0):
                return 0.0
            return float(np.exp(np.log(values).mean()))
        case "weighted_geometric_mean":
            assert settings.metric_weights is not None
            weights = np.asarray(
                [settings.metric_weights[f] for f in fields], dtype=float
            )
            if np.any(values <= 0.0):
                return 0.0
            return float(np.exp((weights * np.log(values)).sum()))
        case "min":
            return float(values.min())
        case "arithmetic_mean":
            return float(values.mean())


def aggregate_instances(
    instance_scores: Iterable[_types.InstanceScore],
    settings: _types.ScoringSettings,
) -> dict[str, float]:
    """Compute the diagnostics named in `settings.instance_aggregations`."""
    instances = list(instance_scores)
    aggregates = np.asarray([s.aggregate for s in instances], dtype=float)
    feasible = np.asarray(
        [s.metric_scores.strictly_feasible for s in instances], dtype=bool
    )
    out: dict[str, float] = {}
    for kind in settings.instance_aggregations:
        match kind:
            case "mean":
                out["mean"] = float(aggregates.mean())
            case "median":
                out["median"] = float(np.median(aggregates))
            case "p10":
                out["p10"] = float(np.quantile(aggregates, 0.10))
            case "feasibility_rate_0p9":
                out["feasibility_rate_0p9"] = float((aggregates >= 0.9).mean())
            case "strict_feasibility":
                out["strict_feasibility"] = float(feasible.mean())
    return out


def instance_score_to_row(
    instance_score: _types.InstanceScore,
) -> dict[str, int | float | bool]:
    """Flatten an `InstanceScore` into a row for an analysis dataframe.

    `boundary_id` is intentionally omitted so the dataframe can carry it as
    its own column/index. Per-metric entries use `violation/<field>` and
    `soft_score/<field>` to match the `/`-separator convention used elsewhere
    in the analysis pipeline.
    """
    row: dict[str, int | float | bool] = {
        "score": instance_score.aggregate,
        "strictly_feasible": instance_score.metric_scores.strictly_feasible,
    }
    for field, v in instance_score.metric_scores.violations.items():
        row[f"violation/{field}"] = v
    for field, s in instance_score.metric_scores.soft_scores.items():
        row[f"soft_score/{field}"] = s
    return row


def instance_scores_to_dataframe(
    instance_scores: Iterable[_types.InstanceScore],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Wide dataframe with every per-metric value and every aggregation variant.

    One row per instance. Columns:

    - `boundary_id` (str)
    - `strictly_feasible` (bool)
    - `violation/<field>` and `soft_score/<field>` for each `RequirementMetrics` field
    - `score/geometric_mean`, `score/min`, `score/arithmetic_mean`
    - `score/weighted_geometric_mean` (only when `weights` is provided)

    Useful for plotting and for comparing aggregations side-by-side without
    re-running the scoring pipeline.

    Args:
        instance_scores: Per-instance scores from `score_instance` /
            `score_eval_data`.
        weights: Optional per-metric weights for `weighted_geometric_mean`. Keys
            must match `RequirementMetrics` fields; values must sum to 1.
    """
    rows: list[dict[str, int | float | bool | str]] = []
    for inst in instance_scores:
        row: dict[str, int | float | bool | str] = {
            "boundary_id": inst.boundary_id,
            "strictly_feasible": inst.metric_scores.strictly_feasible,
        }
        for field, v in inst.metric_scores.violations.items():
            row[f"violation/{field}"] = v
        for field, s in inst.metric_scores.soft_scores.items():
            row[f"soft_score/{field}"] = s
        for name, value in _all_aggregates(
            inst.metric_scores.soft_scores, weights
        ).items():
            row[f"score/{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_scores(
    scores_df: pd.DataFrame,
    feasibility_threshold: float = 0.9,
) -> pd.DataFrame:
    """Summary statistics per aggregation method.

    Returned dataframe is indexed by aggregation name (e.g. `geometric_mean`,
    `min`) and has columns:

    - `mean`, `median`, `p10` of the per-instance score
    - `feasibility_rate_<threshold>`: fraction of instances with score >=
      `feasibility_threshold`
    - `strict_feasibility_rate`: fraction with `strictly_feasible == True`
      (constant across aggregation methods, repeated for convenience)

    Sort by `mean` (or any other column) to rank models. Input must be the
    output of `instance_scores_to_dataframe` (it looks for `score/...` columns).
    """
    score_columns = [c for c in scores_df.columns if c.startswith("score/")]
    if not score_columns:
        raise ValueError(
            "scores_df has no `score/...` columns; was it produced by "
            "`instance_scores_to_dataframe`?"
        )
    feas_col = f"feasibility_rate_{str(feasibility_threshold).replace('.', 'p')}"
    strict_rate = (
        float(scores_df["strictly_feasible"].mean())
        if "strictly_feasible" in scores_df.columns
        else float("nan")
    )
    rows: dict[str, dict[str, float]] = {}
    for col in score_columns:
        agg_name = col[len("score/") :]
        values = scores_df[col].to_numpy()
        rows[agg_name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.10)),
            feas_col: float((values >= feasibility_threshold).mean()),
            "strict_feasibility_rate": strict_rate,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("aggregation")


def _all_aggregates(
    soft_scores: dict[str, float],
    weights: dict[str, float] | None,
) -> dict[str, float]:
    """All metric-aggregation variants computed from per-metric soft scores.

    Single source of truth for the formulas used by both `aggregate_metrics`
    (which picks one) and `instance_scores_to_dataframe` (which reports all).
    """
    fields = list(soft_scores)
    values = np.asarray([soft_scores[f] for f in fields], dtype=float)
    has_zero = bool(np.any(values <= 0.0))
    out: dict[str, float] = {
        "geometric_mean": (0.0 if has_zero else float(np.exp(np.log(values).mean()))),
        "min": float(values.min()),
        "arithmetic_mean": float(values.mean()),
    }
    if weights is not None:
        if set(weights) != set(soft_scores):
            raise ValueError(
                "weights keys must match soft_scores keys; got "
                f"{sorted(weights)}, expected {sorted(soft_scores)}."
            )
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1; got {total:.6f}.")
        w = np.asarray([weights[f] for f in fields], dtype=float)
        out["weighted_geometric_mean"] = (
            0.0 if has_zero else float(np.exp((w * np.log(values)).sum()))
        )
    return out
