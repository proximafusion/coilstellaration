"""Types for the soft-feasibility scoring pipeline.

See `scoring.score_instance` for the pipeline that consumes these types and
the package `README.md` for the rationale behind each design choice.
"""

from typing import Literal, Self

import dapper
import pydantic

ConstraintDirection = Literal["lower_bound", "upper_bound"]
"""Direction of a `RequirementMetrics` constraint.

`lower_bound`: achieved value must be `>=` target (higher is better).
`upper_bound`: achieved value must be `<=` target (lower is better).
"""

REQUIREMENT_METRIC_DIRECTIONS: dict[str, ConstraintDirection] = {
    "min_normalized_coil_to_coil_distance": "lower_bound",
    "min_normalized_coil_to_plasma_distance": "lower_bound",
    "max_normalized_coil_curvature": "upper_bound",
    "max_normalized_field_error": "upper_bound",
}
"""Per-field constraint direction for `RequirementMetrics`.

Keys must match `RequirementMetrics` field names; iteration order defines the
canonical order used by all scoring outputs.
"""


class ScoringSettings(dapper.DapperData):
    """Soft-feasibility scoring of a predicted coilset against `RequirementMetrics`.

    See `scoring.score_instance` for the full pipeline: per-metric normalized
    violation, saturating soft-score map, aggregation across metrics, and
    aggregation across test instances.
    """

    soft_score: Literal["exponential", "linear_ramp"] = "exponential"
    """Per-metric saturating map from normalized violation `v_i` to `s_i in [0, 1]`.

    `exponential`: `s_i = exp(-v_i / alpha_i)` — smooth, never reaches 0.
    `linear_ramp`: `s_i = max(0, 1 - v_i / alpha_i)` — hits 0 at `v_i = alpha_i`.
    """

    tolerances: dict[str, float] = pydantic.Field(
        default_factory=lambda: {
            "min_normalized_coil_to_coil_distance": 0.10,
            "min_normalized_coil_to_plasma_distance": 0.10,
            "max_normalized_coil_curvature": 0.10,
            "max_normalized_field_error": 0.25,
        }
    )
    """Per-metric `alpha_i`: relative violation that cuts the score to `1/e`
    (`exponential`) or to `0` (`linear_ramp`).

    Physics judgment, not tuned.
    """

    target_floor: float = 1e-6
    """Lower clamp on `|tau_i|` when normalizing violation, to keep near-zero targets
    from blowing up `v_i`."""

    metric_aggregation: Literal[
        "geometric_mean", "weighted_geometric_mean", "min", "arithmetic_mean"
    ] = "geometric_mean"
    """How per-metric scores collapse to a per-instance scalar.

    `geometric_mean` is the headline — one zero zeros the whole instance, matching
    feasibility semantics. `min` is the strict worst-case diagnostic.
    `arithmetic_mean` is reported only as a secondary diagnostic.
    """

    metric_weights: dict[str, float] | None = None
    """Required iff `metric_aggregation == 'weighted_geometric_mean'`.

    Keys must match `RequirementMetrics` fields; values must sum to 1.
    """

    instance_aggregations: list[
        Literal["mean", "median", "p10", "feasibility_rate_0p9", "strict_feasibility"]
    ] = pydantic.Field(
        default_factory=lambda: [
            "mean",
            "median",
            "p10",
            "feasibility_rate_0p9",
            "strict_feasibility",
        ]
    )
    """Diagnostics computed across test instances.

    `mean` is the headline ranking
    column; `p10` and `feasibility_rate_0p9` are tail diagnostics;
    `strict_feasibility` is the hard-bound rate (`v_i = 0` for all metrics).
    """

    @pydantic.model_validator(mode="after")
    def _check_tolerances_and_weights(self) -> Self:
        expected_fields = set(REQUIREMENT_METRIC_DIRECTIONS)
        if set(self.tolerances) != expected_fields:
            raise ValueError(
                "tolerances keys must match RequirementMetrics fields: "
                f"{sorted(expected_fields)}; got {sorted(self.tolerances)}."
            )
        if any(a <= 0.0 for a in self.tolerances.values()):
            raise ValueError("tolerances values must be strictly positive.")
        if self.metric_aggregation == "weighted_geometric_mean":
            if self.metric_weights is None:
                raise ValueError(
                    "weighted_geometric_mean requires metric_weights to be set."
                )
            if set(self.metric_weights) != expected_fields:
                raise ValueError(
                    "metric_weights keys must match RequirementMetrics fields: "
                    f"{sorted(expected_fields)}; got {sorted(self.metric_weights)}."
                )
            total = sum(self.metric_weights.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"metric_weights must sum to 1; got {total:.6f}.")
        return self


class MetricScores(dapper.DapperData):
    """Per-metric soft-feasibility scores for one instance."""

    violations: dict[str, float]
    """Normalized violation `v_i >= 0` per metric (0 = constraint satisfied)."""
    soft_scores: dict[str, float]
    """Per-metric `s_i in [0, 1]` after the saturating map."""
    strictly_feasible: bool
    """True iff every `v_i == 0` (all hard bounds satisfied)."""


class InstanceScore(dapper.DapperData):
    """Per-instance score for one test boundary + requirements pair."""

    boundary_id: str
    metric_scores: MetricScores
    aggregate: float
    """Per-instance scalar produced by `ScoringSettings.metric_aggregation`."""


class BenchmarkScore(dapper.DapperData):
    """Aggregate score across the eval set for a single model."""

    per_instance: list[InstanceScore]
    summaries: dict[str, float]
    """Keyed by entries in `ScoringSettings.instance_aggregations`."""


# Note: `ScoreModelBatchSpec` and `ScoreModelBatchResult` (and `EvalSet`) live
# in `_batch_types.py` to keep this module's import graph light for notebook
# users. Cloud-only callers should import them directly from `_batch_types`.
