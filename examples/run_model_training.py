# ruff: noqa: E402
import logging
import os

import jax

jax.config.update("jax_enable_x64", False)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

logging.basicConfig(level=logging.INFO)
logging.getLogger("geometry.surface.surface_types").setLevel(logging.ERROR)
logging.getLogger("storage.google_cloud").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

from coilstellaration import data_utils, paths, types
from coilstellaration.machine_learning import train

if __name__ == "__main__":
    train_config = types.TrainConfig(
        steps=4000,
        warmup_steps=100,
        batch_size=64,
        eval_batch_size=2266,  # full eval set
        eval_every=10,
        max_wall_time_s=60 * 60,
        model_type="mlp",
    )

    checkpoint = train.train(train_config)

    model_path = (paths.OUTPUTS_PATH / data_utils.get_unique_id()).with_suffix(".json")
    logger.info("Saving trained model checkpoint to %s", model_path)
    with model_path.open("w") as f:
        f.write(checkpoint.model_dump_json())

    logger.info("All done!")
