# ruff: noqa: E402
import logging
import os
import time
from collections import defaultdict

import jax

jax.config.update("jax_enable_x64", False)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

logging.basicConfig(level=logging.INFO)
logging.getLogger("geometry.surface.surface_types").setLevel(logging.ERROR)
logging.getLogger("storage.google_cloud").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

from coilstellaration import types
from coilstellaration.machine_learning import train_tasks


def main() -> None:
    base_config = types.TrainConfig(
        steps=40000,
        warmup_steps=100,
        batch_size=64,
        eval_batch_size=2213,
        eval_every=10,
        max_wall_time_s=60 * 60,
        early_stopping_patience=4000,
    )

    configs_to_run = [
        base_config.model_copy(update={"model_type": "mlp_ensemble"}),
        base_config.model_copy(update={"model_type": "mlp"}),
        base_config.model_copy(update={"model_type": "res_mlp_ensemble"}),
        base_config.model_copy(update={"model_type": "res_mlp"}),
    ] * 3

    checkpoints = defaultdict(list)
    for train_config in configs_to_run:
        train_config = train_config.model_copy(
            update={
                "seed": int(time.time_ns() % (2**32)),
            }
        )
        match train_config.model_type:
            case "mlp":
                train_task = train_tasks.train_coil_predictor
            case "mlp_ensemble":
                train_task = train_tasks.train_mlp_ensemble_coil_predictor
            case "res_mlp":
                train_task = train_tasks.train_res_mlp_coil_predictor
            case "res_mlp_ensemble":
                train_task = train_tasks.train_res_mlp_ensemble_coil_predictor
            case _:
                raise ValueError(f"Unsupported model_type: {train_config.model_type}")
        checkpoint = train_task.stored(train_config)
        logger.info("Saved model checkpoint with ID: %s", checkpoint.dapper_storage_id)
        checkpoints[train_config.model_type].append(checkpoint)
        logger.info(
            "Checkpoints: %s",
            {k: [c.dapper_storage_id for c in v] for k, v in checkpoints.items()},
        )
        jax.clear_caches()
    logger.info("All done!")
    return


if __name__ == "__main__":
    main()
