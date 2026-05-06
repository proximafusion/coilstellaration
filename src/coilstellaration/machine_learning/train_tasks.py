"""Dapper-task wrappers around ``train`` and the evaluation pipeline."""

import logging

import dapper
from coilstellaration.machine_learning import (
    train,
    types,
)

logger = logging.getLogger(__name__)


@dapper.task
def train_coil_predictor(
    train_config: types.TrainConfig,
) -> types.CoilPredictorCheckpoint:
    """Cache-keyed MLP training run; returns the final-state model checkpoint."""
    if train_config.model_type != "mlp":
        logger.warning(
            "train_coil_predictor expects model_type='mlp' but got %r; "
            "overriding to 'mlp'.",
            train_config.model_type,
        )
        train_config = train_config.model_copy(update={"model_type": "mlp"})
    checkpoint = train.train(train_config)
    assert isinstance(checkpoint.config, types.CoilPredictorConfig), (
        "train() returned a non-MLP checkpoint despite model_type='mlp'"
    )
    return checkpoint  # type: ignore[return-value]


@dapper.task
def train_mlp_ensemble_coil_predictor(
    train_config: types.TrainConfig,
) -> types.MlpEnsembleCoilPredictorCheckpoint:
    """Cache-keyed MLP ensemble training run; returns the final-state checkpoint."""
    if train_config.model_type != "mlp_ensemble":
        logger.warning(
            "train_mlp_ensemble_coil_predictor expects model_type='mlp_ensemble' "
            "but got %r; overriding to 'mlp_ensemble'.",
            train_config.model_type,
        )
        train_config = train_config.model_copy(update={"model_type": "mlp_ensemble"})
    checkpoint = train.train(train_config)
    assert isinstance(checkpoint.config, types.MlpEnsembleCoilPredictorConfig), (
        "train() returned a non-MLP-ensemble checkpoint despite "
        "model_type='mlp_ensemble'"
    )
    return checkpoint  # type: ignore[return-value]


@dapper.task
def train_res_mlp_coil_predictor(
    train_config: types.TrainConfig,
) -> types.ResMlpCoilPredictorCheckpoint:
    """Cache-keyed residual-MLP training run; returns the final-state checkpoint."""
    if train_config.model_type != "res_mlp":
        logger.warning(
            "train_res_mlp_coil_predictor expects model_type='res_mlp' "
            "but got %r; overriding to 'res_mlp'.",
            train_config.model_type,
        )
        train_config = train_config.model_copy(update={"model_type": "res_mlp"})
    checkpoint = train.train(train_config)
    assert isinstance(checkpoint.config, types.ResMlpCoilPredictorConfig), (
        "train() returned a non-ResMLP checkpoint despite model_type='res_mlp'"
    )
    return checkpoint  # type: ignore[return-value]


@dapper.task
def train_res_mlp_ensemble_coil_predictor(
    train_config: types.TrainConfig,
) -> types.ResMlpEnsembleCoilPredictorCheckpoint:
    """Cache-keyed residual-MLP ensemble training run; returns the final checkpoint."""
    if train_config.model_type != "res_mlp_ensemble":
        logger.warning(
            "train_res_mlp_ensemble_coil_predictor expects "
            "model_type='res_mlp_ensemble' but got %r; overriding.",
            train_config.model_type,
        )
        train_config = train_config.model_copy(
            update={"model_type": "res_mlp_ensemble"}
        )
    checkpoint = train.train(train_config)
    assert isinstance(checkpoint.config, types.ResMlpEnsembleCoilPredictorConfig), (
        "train() returned a non-ResMLP-ensemble checkpoint despite "
        "model_type='res_mlp_ensemble'"
    )
    return checkpoint  # type: ignore[return-value]
