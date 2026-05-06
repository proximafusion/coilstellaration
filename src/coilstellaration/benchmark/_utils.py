"""Private helpers shared by the scoring example, batch tasks, and runner.

Not intended as a public API — import via the calling module instead.
"""

import logging
import warnings
from typing import Literal

import dapper
import jax
import numpy as np
from geometry.surface import rz_fourier_desc
from util import pytree
from util.types import runtime_check_array_sizes

from coilstellaration import (
    coilset_utils,
    flax_nnx_checkpoint,
    metrics_utils,
    types,
)
from coilstellaration.benchmark import (
    _diagnostics,
    _types,
    scoring,
)
from coilstellaration.machine_learning import model_definition, types

logger = logging.getLogger(__name__)


def read_checkpoint(
    model_type: Literal["mlp", "mlp_ensemble", "res_mlp", "res_mlp_ensemble"],
    checkpoint_id: str,
) -> (
    types.CoilPredictorCheckpoint
    | types.MlpEnsembleCoilPredictorCheckpoint
    | types.ResMlpCoilPredictorCheckpoint
    | types.ResMlpEnsembleCoilPredictorCheckpoint
):
    """Read a coil-predictor checkpoint of the declared architecture."""
    match model_type:
        case "mlp":
            return dapper.read(types.CoilPredictorCheckpoint, checkpoint_id)
        case "mlp_ensemble":
            return dapper.read(types.MlpEnsembleCoilPredictorCheckpoint, checkpoint_id)
        case "res_mlp":
            return dapper.read(types.ResMlpCoilPredictorCheckpoint, checkpoint_id)
        case "res_mlp_ensemble":
            return dapper.read(
                types.ResMlpEnsembleCoilPredictorCheckpoint, checkpoint_id
            )


def read_model_from_checkpoint(
    checkpoint: (
        types.CoilPredictorCheckpoint
        | types.MlpEnsembleCoilPredictorCheckpoint
        | types.ResMlpCoilPredictorCheckpoint
        | types.ResMlpEnsembleCoilPredictorCheckpoint
    ),
) -> (
    model_definition.CoilPredictor
    | model_definition.MlpEnsembleCoilPredictor
    | model_definition.ResMlpCoilPredictor
    | model_definition.ResMlpEnsembleCoilPredictor
):
    """Restore the live model from its stored checkpoint."""
    if isinstance(checkpoint.config, types.CoilPredictorConfig):
        return flax_nnx_checkpoint.from_checkpoint(
            checkpoint, model_definition.CoilPredictor
        )
    if isinstance(checkpoint.config, types.MlpEnsembleCoilPredictorConfig):
        return flax_nnx_checkpoint.from_checkpoint(
            checkpoint, model_definition.MlpEnsembleCoilPredictor
        )
    if isinstance(checkpoint.config, types.ResMlpCoilPredictorConfig):
        return flax_nnx_checkpoint.from_checkpoint(
            checkpoint, model_definition.ResMlpCoilPredictor
        )
    if isinstance(checkpoint.config, types.ResMlpEnsembleCoilPredictorConfig):
        return flax_nnx_checkpoint.from_checkpoint(
            checkpoint, model_definition.ResMlpEnsembleCoilPredictor
        )
    raise TypeError(
        f"Unsupported checkpoint config type: {type(checkpoint.config).__name__}"
    )


@runtime_check_array_sizes
def predict_coilsets(
    model: (
        model_definition.CoilPredictor
        | model_definition.MlpEnsembleCoilPredictor
        | model_definition.ResMlpCoilPredictor
        | model_definition.ResMlpEnsembleCoilPredictor
    ),
    eval_dataset: list[types.EvalData],
) -> list[types.EvalData]:
    """Run the model on every boundary in `eval_dataset` in one batched call.

    Sets the model to eval mode first so dropout / layer-norm running stats are
    deterministic — flax NNX raises a TraceContextError otherwise, since dropout tries
    to mutate its RNG counter inside the JIT trace.
    """
    model.eval()
    n_modes_coils_max = model.config.n_modes_coils_max
    n_max_fourier_order = (n_modes_coils_max - 1) // 2
    batched_call = jax.jit(
        jax.vmap(jax.tree_util.Partial(model, fourier_order=n_max_fourier_order))
    )
    batched_boundaries = pytree.tree_stack([e.boundary for e in eval_dataset])
    batched_requirement_metrics = pytree.tree_stack(
        [e.requirement_metrics for e in eval_dataset]
    )
    predicted_batched = batched_call(batched_boundaries, batched_requirement_metrics)
    predicted_list = pytree.tree_unstack(predicted_batched)
    return [
        eval_data.model_copy(update=dict(predicted_coilset=predicted))
        for eval_data, predicted in zip(eval_dataset, predicted_list, strict=True)
    ]


def to_requirement_metrics(
    metrics: types.ConStellarationUpdateMetrics,
) -> types.RequirementMetrics:
    """Reduce `ConStellarationUpdateMetrics` to the four `RequirementMetrics`
    scalars."""
    return types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=float(
            np.min(np.asarray(metrics.normalized_coil_to_coil_min_distances))
        ),
        min_normalized_coil_to_plasma_distance=float(
            np.min(np.asarray(metrics.normalized_coil_to_plasma_min_distances))
        ),
        max_normalized_coil_curvature=float(
            np.max(np.asarray(metrics.normalized_coil_curvatures))
        ),
        max_normalized_field_error=float(
            np.max(np.asarray(metrics.normalized_field_error))
        ),
    )


def score_predictions(
    predictions: list[types.EvalData],
    settings: _types.ScoringSettings,
    log_every: int = 16,
) -> tuple[list[_types.InstanceScore], list[str]]:
    """Score predictions sequentially.

    Returns ``(instance_scores, failed_boundary_ids)``. A boundary is recorded as
    failed if its DESC evaluation raises, so a single bad coilset doesn't kill
    the whole batch.
    """
    instance_scores: list[_types.InstanceScore] = []
    failed_boundary_ids: list[str] = []
    _diagnostics.write("score_loop_start", f"n={len(predictions)}")
    for i, ed in enumerate(predictions):
        try:
            with warnings.catch_warnings(action="ignore"):
                assert ed.predicted_coilset is not None
                predicted_metrics = (
                    metrics_utils.evaluate_coilset_metrics_from_boundary(
                        boundary=rz_fourier_desc.to_desc_fourier_rz_toroidal_surface(
                            ed.boundary
                        ),
                        coilset=coilset_utils.constellaration_update_to_desc(
                            ed.predicted_coilset
                        ),
                    )
                )
            achieved = to_requirement_metrics(predicted_metrics)
            instance_scores.append(scoring.score_eval_data(ed, achieved, settings))
        except Exception:
            logger.exception("Failed to score boundary %s", ed.boundary_id)
            failed_boundary_ids.append(ed.boundary_id)
        if i % log_every == 0:
            jax.clear_caches()
            logger.info("Scored %d / %d", i + 1, len(predictions))
            _diagnostics.write(
                f"score_progress_{i:03d}",
                f"i={i + 1} of {len(predictions)} ok={len(instance_scores)} "
                f"failed={len(failed_boundary_ids)}",
            )
    logger.info(
        "Finished: %d scored, %d failed.",
        len(instance_scores),
        len(failed_boundary_ids),
    )
    return instance_scores, failed_boundary_ids
