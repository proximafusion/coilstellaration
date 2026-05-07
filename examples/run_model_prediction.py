import logging

from constellaration.geometry import surface_utils_desc

from coilstellaration import (
    coilset_utils,
    data_utils,
    flax_nnx_checkpoint_util,
    metrics_utils_v2,
    paths,
    plot_utils,
    types,
)
from coilstellaration.machine_learning import model_definition

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("geometry.surface.surface_types").setLevel(logging.ERROR)
logging.getLogger("storage.google_cloud").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Load one of the MLP models
    model_path = paths.model_path("D2HbzeYjo57Aif48z5T6axt")

    logger.info("Loading model checkpoint from %s", model_path)
    model_checkpoint = types.CoilPredictorCheckpoint.model_validate_json(
        model_path.read_text()
    )
    coil_predictor = flax_nnx_checkpoint_util.from_checkpoint(
        model_checkpoint,
        module_cls=model_definition.CoilPredictor,
    )

    logger.info("Running model prediction on one example from the eval set...")
    eval_data = data_utils.load_benchmark_dataset(
        track="fixed_shape", stratum="tight", split="eval", n=1
    )
    predictions = model_definition.predict_coilsets(coil_predictor, eval_data)
    prediction = predictions[0]

    predicted_coilset_path = paths.OUTPUTS_PATH / "predicted_coilset.json"
    logger.info("Writing predicted coilset to %s", predicted_coilset_path)
    with predicted_coilset_path.open("w") as f:
        f.write(prediction.predicted_coilset.model_dump_json())

    logger.info("Plotting initial coilset and equilibrium...")
    figure = plot_utils.plot_surface(prediction.boundary)
    figure = plot_utils.plot_coilset(
        prediction.predicted_coilset, figure=figure, color="blue"
    )
    figure = plot_utils.plot_coilset(
        prediction.true_coilset, figure=figure, color="green"
    )
    figure.write_html(paths.OUTPUTS_PATH / "predicted_coilset_and_equilibrium.html")

    logger.info("Evaluating coilset metrics...")
    metrics = metrics_utils_v2.evaluate_coilset_metrics_from_boundary(
        boundary=surface_utils_desc.to_desc_fourier_rz_toroidal_surface(
            prediction.boundary
        ),
        coilset=coilset_utils.coilstellaration_to_desc(prediction.predicted_coilset),
        surf_eval_m=24,
        surf_eval_n=24,
        coil_eval_n=100,
    )

    predicted_coilset_metrics_path = (
        paths.OUTPUTS_PATH / "predicted_coilset_metrics.json"
    )
    logger.info("Writing metrics to %s", predicted_coilset_metrics_path)
    with predicted_coilset_metrics_path.open("w") as f:
        f.write(metrics.model_dump_json())
