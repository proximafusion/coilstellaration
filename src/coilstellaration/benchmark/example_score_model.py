r"""Example: score one ML model checkpoint against the public test set.

Edit `MODEL_CHECKPOINT_ID` (and `MODEL_TYPE` if scoring an mlp_ensemble
checkpoint) below, then run:

    pants run \\
        beta/constellaration_update/machine_learning/scoring/example_score_model.py

Outputs:
- `scores_<id>.csv`: per-instance flat row dict (see `instance_score_to_row`).
- `soft_feasibility_<id>.png`: 2x3 grid of per-metric soft-score histograms.
- `BenchmarkScore.summaries` printed to stdout.

Authentication for the private `proxima-fusion/constellaration_update`
HuggingFace dataset is required before running. Either set `HF_TOKEN` in the
environment or call `huggingface_hub.login(token=...)` once.
"""

import logging
import os
import pathlib
import warnings
from typing import Literal

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import dapper
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from geometry.surface import rz_fourier_desc
from util import logging_utils, pytree
from util.types import runtime_check_array_sizes

from coilstellaration import (
    coilset_utils,
    flax_nnx_checkpoint,
    metrics_utils,
    types,
)
from coilstellaration.benchmark import _types, scoring
from coilstellaration.machine_learning import (
    model_definition,
    train,
    types,
)

# --- USER CONFIG -------------------------------------------------------------

MODEL_CHECKPOINT_ID = "D2CjX6A6BHAdwhhYWPz3S7B"
"""Dapper ID of the checkpoint to score."""

MODEL_TYPE: Literal["mlp", "mlp_ensemble", "res_mlp", "res_mlp_ensemble"] = "mlp"
"""Architecture matching `MODEL_CHECKPOINT_ID`."""

OUTPUTS_DIR = pathlib.Path("/home/devuser/tmp/outputs/constellaration_update/")
"""Where to write the CSV + histogram PNG."""

EVAL_N = 32
"""Public-test-set cap.

Use `0` to load the full eval split (slow — DESC eval
runs once per instance).
"""

EVAL_THRESHOLD = -0.5
"""`relative_min_norm_coil_to_plasma_distance_error_threshold` for
`train.load_dataset`."""

SCORING_SETTINGS = _types.ScoringSettings()
"""Scoring knobs.

Defaults: exponential soft-score, geometric-mean aggregation across metrics,
all five instance summaries on. See `scoring/README.md` for the full design.
"""

# --- PIPELINE ----------------------------------------------------------------


def main() -> None:
    logging_utils.configure_color_logging()
    logging.getLogger("geometry.surface.surface_types").setLevel(logging.ERROR)
    logging.getLogger("storage.google_cloud").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading eval split (n=%d).", EVAL_N)
    eval_dataset = train.load_dataset(
        "eval",
        relative_min_norm_coil_to_plasma_distance_error_threshold=EVAL_THRESHOLD,
        n=EVAL_N,
    )

    logger.info("Reading checkpoint %s (%s).", MODEL_CHECKPOINT_ID, MODEL_TYPE)
    checkpoint = _read_checkpoint(MODEL_TYPE, MODEL_CHECKPOINT_ID)
    model = _read_model_from_checkpoint(checkpoint)

    logger.info("Predicting %d coilsets...", len(eval_dataset))
    predictions = _predict_coilsets(model, eval_dataset)

    logger.info("Scoring predictions (heavy DESC eval per instance)...")
    instance_scores = _score_predictions(predictions, logger)

    scores_df = pd.DataFrame(
        [
            {"boundary_id": s.boundary_id, **scoring.instance_score_to_row(s)}
            for s in instance_scores
        ]
    )
    scores_csv = OUTPUTS_DIR / f"scores_{MODEL_CHECKPOINT_ID}.csv"
    scores_df.to_csv(scores_csv, index=False)
    logger.info("Wrote %s", scores_csv)

    histogram_png = OUTPUTS_DIR / f"soft_feasibility_{MODEL_CHECKPOINT_ID}.png"
    _plot_histograms(scores_df, histogram_png)
    logger.info("Wrote %s", histogram_png)

    benchmark = scoring.score_benchmark(instance_scores, SCORING_SETTINGS)
    print(
        f"\nAggregate score for {MODEL_TYPE} {MODEL_CHECKPOINT_ID} "
        f"on the public test set (n={len(instance_scores)}):\n"
    )
    print(pd.Series(benchmark.summaries, name="value").to_frame())


def _read_checkpoint(
    model_type: Literal["mlp", "mlp_ensemble", "res_mlp", "res_mlp_ensemble"],
    checkpoint_id: str,
) -> (
    types.CoilPredictorCheckpoint
    | types.MlpEnsembleCoilPredictorCheckpoint
    | types.ResMlpCoilPredictorCheckpoint
    | types.ResMlpEnsembleCoilPredictorCheckpoint
):
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


def _read_model_from_checkpoint(
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
def _predict_coilsets(
    model: (
        model_definition.CoilPredictor
        | model_definition.MlpEnsembleCoilPredictor
        | model_definition.ResMlpCoilPredictor
        | model_definition.ResMlpEnsembleCoilPredictor
    ),
    eval_dataset: list[types.EvalData],
) -> list[types.EvalData]:
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


def _score_predictions(
    predictions: list[types.EvalData],
    logger: logging.Logger,
) -> list[_types.InstanceScore]:
    instance_scores: list[_types.InstanceScore] = []
    for i, ed in enumerate(predictions):
        with warnings.catch_warnings(action="ignore"):
            assert ed.predicted_coilset is not None
            predicted_metrics = metrics_utils.evaluate_coilset_metrics_from_boundary(
                boundary=rz_fourier_desc.to_desc_fourier_rz_toroidal_surface(
                    ed.boundary
                ),
                coilset=coilset_utils.constellaration_update_to_desc(
                    ed.predicted_coilset
                ),
            )
        achieved = _to_requirement_metrics(predicted_metrics)
        instance_scores.append(scoring.score_eval_data(ed, achieved, SCORING_SETTINGS))
        if i % 16 == 0:
            jax.clear_caches()
            logger.info("Scored %d / %d", i, len(predictions))
    logger.info("Finished scoring %d instances.", len(instance_scores))
    return instance_scores


def _to_requirement_metrics(
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


def _plot_histograms(scores_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    soft_score_columns = [
        f"soft_score/{f}" for f in _types.REQUIREMENT_METRIC_DIRECTIONS
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, col in zip(axes[:4], soft_score_columns, strict=True):
        sns.histplot(scores_df[col].to_numpy(), ax=ax, bins=20)
        ax.set_title(col.replace("soft_score/", ""))
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("soft feasibility")
    sns.histplot(scores_df["score"].to_numpy(), ax=axes[4], bins=20, color="C2")
    axes[4].set_title("aggregate (geometric mean)")
    axes[4].set_xlim(-0.05, 1.05)
    axes[4].set_xlabel("score")
    axes[5].axis("off")
    fig.suptitle(f"Soft-feasibility distribution — {MODEL_TYPE} {MODEL_CHECKPOINT_ID}")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
