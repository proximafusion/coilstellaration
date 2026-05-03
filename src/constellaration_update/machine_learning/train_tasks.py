"""Plain Python wrappers around ``train`` for typed model-checkpoint contracts."""

import logging

from constellaration_update.machine_learning import train, types

logger = logging.getLogger(__name__)


def train_coil_predictor(
    train_config: types.TrainConfig,
) -> types.CoilPredictorCheckpoint:
    """Train an MLP coil predictor; returns the final-state model checkpoint."""
    if train_config.model_type != "mlp":
        logger.warning(
            "train_coil_predictor expects model_type='mlp' but got %r; "
            "overriding to 'mlp'.",
            train_config.model_type,
        )
        train_config = train_config.model_copy(update={"model_type": "mlp"})
    checkpoint = train.train(train_config)
    assert isinstance(
        checkpoint.config, types.CoilPredictorConfig
    ), "train() returned a non-MLP checkpoint despite model_type='mlp'"
    return checkpoint  # type: ignore[return-value]


def train_attention_coil_predictor(
    train_config: types.TrainConfig,
) -> types.AttentionCoilPredictorCheckpoint:
    """Train an attention coil predictor; returns the final-state checkpoint."""
    if train_config.model_type != "attention":
        logger.warning(
            "train_attention_coil_predictor expects model_type='attention' but "
            "got %r; overriding to 'attention'.",
            train_config.model_type,
        )
        train_config = train_config.model_copy(update={"model_type": "attention"})
    checkpoint = train.train(train_config)
    assert isinstance(
        checkpoint.config, types.AttentionCoilPredictorConfig
    ), "train() returned a non-attention checkpoint despite model_type='attention'"
    return checkpoint  # type: ignore[return-value]
