"""Score one ML model checkpoint against the public eval set.

Outputs (under `paths.OUTPUTS_PATH`):
- `scores_<id>.csv`: per-instance flat row dict (see `instance_score_to_row`).
- `soft_feasibility_<id>.png`: 2x3 grid of per-metric soft-score histograms.
- `BenchmarkScore.summaries` printed to stdout.
"""

import logging
import pathlib
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from constellaration.geometry import surface_utils_desc

from coilstellaration import (
    coilset_utils,
    data_utils,
    flax_nnx_checkpoint_util,
    metrics_utils_v2,
    paths,
    types,
)
from coilstellaration.benchmark import scoring
from coilstellaration.benchmark import types as benchmark_types
from coilstellaration.machine_learning import model_definition, utils

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("geometry.surface.surface_types").setLevel(logging.ERROR)
logging.getLogger("storage.google_cloud").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

MODEL_CHECKPOINT_ID = "D2HbzeYjo57Aif48z5T6axt"
EVAL_N = 4  # Public-test-set cap. Use 0 to score the full eval split (slow).


def plot_soft_feasibility(scores_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    fields = list(benchmark_types.REQUIREMENT_METRIC_DIRECTIONS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, field in zip(axes[:4], fields, strict=True):
        ax.hist(
            scores_df[f"soft_score/{field}"].to_numpy(),
            bins=20,
            range=(0.0, 1.0),
        )
        ax.set_title(field)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("soft feasibility")
    axes[4].hist(scores_df["score"].to_numpy(), bins=20, range=(0.0, 1.0), color="C2")
    axes[4].set_title("aggregate (geometric mean)")
    axes[4].set_xlim(-0.05, 1.05)
    axes[4].set_xlabel("score")
    axes[5].axis("off")
    fig.suptitle(f"Soft-feasibility distribution — {MODEL_CHECKPOINT_ID}")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    settings = benchmark_types.ScoringSettings()

    logger.info("Loading eval split (n=%d)...", EVAL_N)
    eval_dataset = data_utils.load_benchmark_dataset(
        track="fixed_shape", stratum="tight", split="eval", n=EVAL_N
    )

    model_path = paths.model_path(MODEL_CHECKPOINT_ID)
    logger.info("Loading model checkpoint from %s", model_path)
    checkpoint = types.CoilPredictorCheckpoint.model_validate_json(
        model_path.read_text()
    )
    model = flax_nnx_checkpoint_util.from_checkpoint(
        checkpoint, module_cls=model_definition.CoilPredictor
    )
    model.eval()

    logger.info("Predicting %d coilsets...", len(eval_dataset))
    predictions = model_definition.predict_coilsets(model, eval_dataset)

    logger.info("Scoring predictions (heavy DESC eval per instance)...")
    instance_scores: list[benchmark_types.InstanceScore] = []
    for i, eval_data in enumerate(predictions):
        assert eval_data.predicted_coilset is not None
        with warnings.catch_warnings(action="ignore"):
            achieved_metrics = metrics_utils_v2.evaluate_coilset_metrics_from_boundary(
                boundary=surface_utils_desc.to_desc_fourier_rz_toroidal_surface(
                    eval_data.boundary
                ),
                coilset=coilset_utils.coilstellaration_to_desc(
                    eval_data.predicted_coilset
                ),
            )
            achieved = data_utils.metrics_to_requirement_metrics(achieved_metrics)
        instance_scores.append(scoring.score_eval_data(eval_data, achieved, settings))
        if (i + 1) % 16 == 0:
            logger.info("Scored %d / %d", i + 1, len(predictions))
    logger.info("Finished scoring %d instances.", len(instance_scores))

    scores_df = pd.DataFrame(
        [
            {"boundary_id": s.boundary_id, **scoring.instance_score_to_row(s)}
            for s in instance_scores
        ]
    )
    scores_csv = paths.OUTPUTS_PATH / f"scores_{MODEL_CHECKPOINT_ID}.csv"
    logger.info("Writing per-instance scores to %s", scores_csv)
    scores_df.to_csv(scores_csv, index=False)

    histogram_png = paths.OUTPUTS_PATH / f"soft_feasibility_{MODEL_CHECKPOINT_ID}.png"
    logger.info("Writing soft-feasibility histograms to %s", histogram_png)
    plot_soft_feasibility(scores_df, histogram_png)

    benchmark = scoring.score_benchmark(instance_scores, settings)
    logger.info(
        "Aggregate score for %s on the public test set (n=%d):",
        MODEL_CHECKPOINT_ID,
        len(instance_scores),
    )
    print(pd.Series(benchmark.summaries, name="value").to_frame())
