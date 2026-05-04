"""Example multi-config training driver."""

# ruff: noqa: E402
import logging
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from constellaration_update.machine_learning import train_tasks, types


def main() -> None:
    base_config = types.TrainConfig(
        steps=10_000,
        warmup_steps=250,
        eval_batch_size=512,
        eval_every=500,
        relative_min_coil_to_plasma_distance_error_threshold=-0.4,
        early_stopping_patience=1000,
    )

    configs_to_run = [
        base_config.model_copy(update={"model_type": "mlp"}),
        base_config.model_copy(update={"model_type": "attention"}),
    ] * 5

    checkpoints = []
    for train_config in configs_to_run:
        train_config = train_config.model_copy(
            update={
                "seed": int(time.time_ns() % (2**32)),
            }
        )
        match train_config.model_type:
            case "mlp":
                train_fn = train_tasks.train_coil_predictor
            case "attention":
                train_fn = train_tasks.train_attention_coil_predictor
            case _:
                raise ValueError(f"Unsupported model_type: {train_config.model_type}")
        checkpoint = train_fn(train_config)
        logger.info("Trained checkpoint for model_type=%s", train_config.model_type)
        checkpoints.append(checkpoint)
    logger.info("All done! Trained %d checkpoints.", len(checkpoints))


if __name__ == "__main__":
    main()
