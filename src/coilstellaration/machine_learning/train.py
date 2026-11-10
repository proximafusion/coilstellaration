from __future__ import annotations

import hashlib
import itertools
import logging
import queue
import threading
import time
import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import optax
import orjson
import pandas as pd
import pydantic
import wandb
from beartype.typing import Any
from constellaration.geometry import surface_rz_fourier
from flax import nnx

from coilstellaration import (
    coilset_utils,
    data_utils,
    flax_nnx_checkpoint_util,
    types,
)
from coilstellaration.machine_learning import (
    model_definition,
    utils,
)
from coilstellaration.types import NpOrJaxArray

logger = logging.getLogger(__name__)

_VARIABLE_SHAPE_CONFIG_TYPES: dict[
    str, type[model_definition.AnyVariableShapeConfig]
] = {
    "per_coil": types.PerCoilCoilPredictorConfig,
    "masked": types.MaskedCoilPredictorConfig,
    "per_coil_ensemble": types.PerCoilEnsembleCoilPredictorConfig,
    "masked_ensemble": types.MaskedEnsembleCoilPredictorConfig,
}


class TrainSummary(pydantic.BaseModel):
    """Outcome of a training run besides the checkpoint."""

    best_eval_loss: float
    best_step: int
    final_eval_loss: float
    """Evaluation loss of the final weights."""
    n_train_rows: int
    n_eval_rows: int


def split_train_holdout(
    df: pd.DataFrame, fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits rows into `(train, holdout)` such that no boundary appears in both.

    A boundary is held out based on a stable hash of its
    `constellaration_boundary_id`, so the split is reproducible across processes and
    independent of row order. About `fraction` of the boundaries are held out.
    """
    is_holdout = (
        df["constellaration_boundary_id"].astype(str).map(_stable_unit_interval)
        < fraction
    )
    return df.loc[~is_holdout], df.loc[is_holdout]


def compute_coil_currents_stats(train_df: pd.DataFrame) -> tuple[float, float]:
    """Robust mean and std of the training coil currents, see `robust_mean_and_std`.

    Currents are taken in units of the boundary's major radius, like the models
    predict them (see `model_definition.CoilTransform`).
    """
    logger.info("Computing currents stats from %d training rows...", len(train_df))
    currents = [_currents_per_major_radius(row) for _, row in train_df.iterrows()]
    mean, std = robust_mean_and_std(np.concatenate(currents))
    logger.info("Computed currents mean %s and std %s", mean, std)
    return float(mean), float(std)


def drop_outlier_currents(
    df: pd.DataFrame,
    currents_mean: float,
    currents_std: float,
    max_deviation: float = 10.0,
) -> pd.DataFrame:
    """Drops the rows with a coil current more than `max_deviation` stds from the mean.

    `currents_mean` and `currents_std` are those of `compute_coil_currents_stats`. Such
    coilsets are broken, e.g. with all currents ten times too large or one coil
    reversed. The loss divides the currents error by `currents_std`, so a single one of
    them would dominate the loss of its batch.
    """
    keep = [
        bool(
            np.all(
                np.abs(_currents_per_major_radius(row) - currents_mean)
                <= max_deviation * currents_std
            )
        )
        for _, row in df.iterrows()
    ]
    logger.info(
        "Dropping %d of %d rows with outlier currents", keep.count(False), len(keep)
    )
    return df.loc[keep]


def compute_coil_curvature_stats(train_df: pd.DataFrame) -> tuple[float, float]:
    """Robust mean and std of the training coilsets' max curvature, see
    `robust_mean_and_std`."""
    mean, std = robust_mean_and_std(_max_normalized_coil_curvatures(train_df))
    return float(mean), float(std)


def drop_outlier_curvatures(
    df: pd.DataFrame,
    curvature_mean: float,
    curvature_std: float,
    max_deviation: float = 10.0,
) -> pd.DataFrame:
    """Drops the rows whose max coil curvature is more than `max_deviation` stds from
    the mean.

    `curvature_mean` and `curvature_std` are those of `compute_coil_curvature_stats`.
    Such coilsets are broken, with a coil folded into a near-cusp. Their curvature is
    tens to thousands of times the usual and would reach the model as an input that
    many stds out.
    """
    keep = (
        np.abs(_max_normalized_coil_curvatures(df) - curvature_mean)
        <= max_deviation * curvature_std
    )
    logger.info(
        "Dropping %d of %d rows with outlier curvatures", int(np.sum(~keep)), len(keep)
    )
    return df.loc[keep]


def compute_requirement_metrics_stats(
    train_df: pd.DataFrame,
) -> tuple[jt.Float[np.ndarray, " 4"], jt.Float[np.ndarray, " 4"]]:
    """Robust means and stds of the metric inputs, see `robust_mean_and_std`.

    Ordered like the model inputs, `model_definition.requirement_metrics_vector`.
    """
    columns = [*data_utils.REQUIREMENT_METRICS_COLUMNS, data_utils.N_COILS_COLUMN]
    # `np.stack` rather than `tree_stack`: stacking ~10^5 scalars into device arrays one
    # by one takes ~30 times longer.
    requirement_metrics = jax.tree.map(
        lambda *values: np.stack(values),
        *(
            data_utils.row_to_requirement_metrics(row)
            for row in train_df[columns].to_dict("records")
        ),
    )
    return robust_mean_and_std(
        np.asarray(
            jax.vmap(model_definition.requirement_metrics_vector)(requirement_metrics)
        )
    )


@types.runtime_check_array_sizes
def robust_mean_and_std(
    values: jt.Float[np.ndarray, " n *features"],
) -> tuple[jt.Float[np.ndarray, " *features"], jt.Float[np.ndarray, " *features"]]:
    """Median and normal-consistent interquartile range along the first axis.

    Both equal the mean and std for normally distributed values, but a few rows that are
    orders of magnitude off do not move them. With a plain mean and std, such rows set
    the scale and squash every other row onto the same standardized value. The std is
    floored at `1e-8`.
    """
    q25, median, q75 = np.nanpercentile(values, [25, 50, 75], axis=0)
    # The interquartile range of the standard normal distribution.
    normal_iqr = 1.3489795003921634
    return np.asarray(median), np.asarray(np.maximum((q75 - q25) / normal_iqr, 1e-8))


def _decode_coilset(
    row: pd.Series,
) -> types.Coilset:
    return types.Coilset.model_validate(orjson.loads(row["json_desc_coilset"]))


def _decode_boundary(row: pd.Series) -> surface_rz_fourier.SurfaceRZFourier:
    return surface_rz_fourier.SurfaceRZFourier.model_validate(
        orjson.loads(row["json_constellaration_boundary"])
    )


def _currents_per_major_radius(row: pd.Series) -> jt.Float[np.ndarray, " n_coils"]:
    return np.asarray(_decode_coilset(row).currents) / float(
        utils.major_radius(_decode_boundary(row))
    )


def _max_normalized_coil_curvatures(
    df: pd.DataFrame,
) -> jt.Float[np.ndarray, " n_rows"]:
    columns = [*data_utils.REQUIREMENT_METRICS_COLUMNS, data_utils.N_COILS_COLUMN]
    return np.array(
        [
            float(
                data_utils.row_to_requirement_metrics(row).max_normalized_coil_curvature
            )
            for row in df[columns].to_dict("records")
        ]
    )


def tree_stack[T](trees: Sequence[T], axis: int = 0) -> T:
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=axis), *trees)


def tree_unstack[T](tree: T, axis: int = 0) -> list[T]:
    leaves, treedef = jax.tree.flatten(tree)

    leaves = [jnp.moveaxis(x, axis, 0) for x in leaves]
    stack_length = leaves[0].shape[0]
    return [
        jax.tree.unflatten(treedef, [x[i] for x in leaves]) for i in range(stack_length)
    ]


def tree_map[T](
    f: Callable[..., Any],
    tree: T,
    *rest,
    is_leaf: Callable[[Any], bool] | None = None,
) -> T:
    if not isinstance(tree, pydantic.BaseModel):
        return jax.tree_util.tree_map(f, tree, *rest, is_leaf=is_leaf)

    TreeType = type(tree)
    rest = [
        (
            TreeType.model_construct(**r.__dict__)
            if not jax.tree.structure(tree) == jax.tree.structure(r)
            else r
        )
        for r in rest
    ]
    return jax.tree.map(f, tree, *rest, is_leaf=is_leaf)


def make_batches(
    df: pd.DataFrame,
    batch_size: int,
    seed: int,
    pad_to_fourier_order: int,
    pad_to_n_coils: int | None = None,
) -> Iterator[types.Batch]:
    """Yields infinite training batches in seeded shuffled order.

    If `pad_to_n_coils` is given, every coilset is zero-padded to that many unique coils
    and `Batch.coil_mask` marks the real coil slots. Otherwise coilsets are not padded
    and the mask is all ones.
    """
    key = jax.random.PRNGKey(seed)
    json_df = df[
        [
            *data_utils.JSON_COLUMNS,
            *data_utils.REQUIREMENT_METRICS_COLUMNS,
            data_utils.N_COILS_COLUMN,
        ]
    ].reset_index(drop=True)
    n = len(json_df)
    if n < batch_size:
        raise ValueError(
            f"dataset has {n} rows but batch_size={batch_size}; "
            "iterator would never yield"
        )
    indices = jnp.arange(n)

    logger.info("Decoding %d rows to build in-memory batch artifacts...", n)

    all_coilsets = [
        coilset_utils.pad_coilset_to_fourier_order(
            _decode_coilset(row), pad_to_fourier_order
        )
        for _, row in json_df.iterrows()
    ]
    all_coil_masks = jnp.stack(
        [
            coilset_utils.make_coil_mask(
                int(c.n_unique_coils),
                pad_to_n_coils if pad_to_n_coils is not None else int(c.n_unique_coils),
            )
            for c in all_coilsets
        ]
    )
    if pad_to_n_coils is not None:
        all_coilsets = [
            coilset_utils.pad_coilset_to_n_coils(c, pad_to_n_coils)
            for c in all_coilsets
        ]
    all_masks = jnp.stack(
        [
            coilset_utils.make_fourier_order_mask(
                int(c.fourier_order), pad_to_fourier_order
            )
            for c in all_coilsets
        ]
    )
    all_coilsets = tree_stack(all_coilsets)
    all_boundaries = tree_stack(
        [_decode_boundary(row) for _, row in json_df.iterrows()]
    )
    all_requirement_metrics = tree_stack(
        [
            data_utils.row_to_requirement_metrics(row.to_dict())
            for _, row in json_df.iterrows()
        ]
    )
    del json_df

    while True:
        key, subkey = jax.random.split(key)
        shuffled_indices = jax.random.permutation(subkey, indices)
        for batch_indices in itertools.batched(shuffled_indices, batch_size):
            batch_indices = jnp.array(batch_indices)
            (
                subset_boundaries,
                subset_coilsets,
                subset_masks,
                subset_coil_masks,
                subset_requirement_metrics,
            ) = tree_map(
                lambda a: a[batch_indices],
                (
                    all_boundaries,
                    all_coilsets,
                    all_masks,
                    all_coil_masks,
                    all_requirement_metrics,
                ),
            )

            yield types.Batch.model_construct(
                boundaries=subset_boundaries,
                coilsets=subset_coilsets,
                requirement_metrics=subset_requirement_metrics,
                fourier_order_mask=subset_masks,
                coil_mask=subset_coil_masks,
                batch_max_fourier_order=pad_to_fourier_order,
            )


def prefetched(
    iterator: Iterator[types.Batch], queue_depth: int = 10
) -> Iterator[types.Batch]:
    """Wrap ``iterator`` in a daemon-thread prefetch queue."""
    q: queue.Queue[types.Batch | Exception | None] = queue.Queue(maxsize=queue_depth)

    def _producer() -> None:
        try:
            for batch in iterator:
                q.put(batch)
        except Exception as exc:  # pragma: no cover
            q.put(exc)
        else:
            q.put(None)

    threading.Thread(target=_producer, daemon=True).start()

    while True:
        item = q.get()
        if item is None:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def infer_max_fourier_order(train_df: pd.DataFrame) -> int:
    """Return the maximum per-sample fourier order N over the train set.

    Scans `train_df["json_coilset"]`, decodes each row, and returns the
    maximum `coilset.fourier_order`. Raises `ValueError` if the column is
    missing, no rows have a decodable coilset, or any row fails to decode.
    """
    if "json_desc_coilset" not in train_df.columns:
        raise ValueError("train_df has no 'json_desc_coilset' column")
    series = train_df["json_desc_coilset"].dropna()
    if len(series) == 0:
        raise ValueError("train_df is empty (no decodable coilsets)")
    max_order = 0
    for _, json_str in series.items():
        coilset = types.Coilset.model_validate(orjson.loads(json_str))
        max_order = max(max_order, int(coilset.fourier_order))
    return max_order


def _resolve_model_config(
    initial: types.AnyModelConfig | None,
    model_kind: types.AnyModelLiteral,
    train_df: pd.DataFrame,
    coil_currents_mean: float,
    coil_currents_std: float,
    requirement_metrics_means: jt.Float[NpOrJaxArray, " n_requirements"],
    requirement_metrics_stds: jt.Float[NpOrJaxArray, " n_requirements"],
    max_fourier_order: int,
) -> types.AnyModelConfig:
    """Fills data-derived fields on the architecture config from the training set.

    Architecture knobs (``hidden_dim``, layer counts, spectral params) come
    from ``initial``; defaults applied if ``initial`` is ``None`` for the
    architecture selected by ``model_kind``. Data-derived fields are taken
    from a single training row and cross-checked against a second row to
    surface dataset inconsistency early. For variable-shape model types, the
    supported numbers of field periods and coils are collected over all training
    rows, which must share one boundary resolution and symmetry.
    """
    if model_kind in types.VARIABLE_SHAPE_MODEL_KIND:
        return _resolve_variable_shape_config(
            initial,
            model_kind,
            train_df,
            coil_currents_mean,
            coil_currents_std,
            requirement_metrics_means,
            requirement_metrics_stds,
            max_fourier_order,
        )
    if initial is None:
        match model_kind:
            case "mlp":
                initial = types.CoilPredictorConfig(
                    is_stellarator_symmetric=False,
                    n_field_periods=0,
                    n_poloidal_modes=0,
                    n_toroidal_modes=0,
                    n_unique_coils=0,
                    n_modes_coils_max=0,
                    coil_currents_mean=0.0,
                    coil_currents_std=0.0,
                    requirement_metrics_means=[],
                    requirement_metrics_stds=[],
                )
            case "res_mlp":
                initial = types.ResMlpCoilPredictorConfig(
                    is_stellarator_symmetric=False,
                    n_field_periods=0,
                    n_poloidal_modes=0,
                    n_toroidal_modes=0,
                    n_unique_coils=0,
                    n_modes_coils_max=0,
                    coil_currents_mean=0.0,
                    coil_currents_std=0.0,
                    requirement_metrics_means=[],
                    requirement_metrics_stds=[],
                )
            case "mlp_ensemble":
                initial = types.MlpEnsembleCoilPredictorConfig(
                    is_stellarator_symmetric=False,
                    n_field_periods=0,
                    n_poloidal_modes=0,
                    n_toroidal_modes=0,
                    n_unique_coils=0,
                    n_modes_coils_max=0,
                    coil_currents_mean=0.0,
                    coil_currents_std=0.0,
                    requirement_metrics_means=[],
                    requirement_metrics_stds=[],
                )
            case "res_mlp_ensemble":
                initial = types.ResMlpEnsembleCoilPredictorConfig(
                    is_stellarator_symmetric=False,
                    n_field_periods=0,
                    n_poloidal_modes=0,
                    n_toroidal_modes=0,
                    n_unique_coils=0,
                    n_modes_coils_max=0,
                    coil_currents_mean=0.0,
                    coil_currents_std=0.0,
                    requirement_metrics_means=[],
                    requirement_metrics_stds=[],
                )
            case _:
                raise ValueError(f"unsupported model_kind {model_kind!r}")
    rows = train_df.dropna(subset=list(data_utils.JSON_COLUMNS)).head(2)
    if len(rows) < 1:
        raise RuntimeError("training set is empty after filtering")
    boundary_a = surface_rz_fourier.SurfaceRZFourier.model_validate(
        orjson.loads(rows.iloc[0]["json_constellaration_boundary"])
    )
    coilset_a = types.Coilset.model_validate(
        orjson.loads(rows.iloc[0]["json_desc_coilset"])
    )
    if len(rows) >= 2:
        boundary_b = surface_rz_fourier.SurfaceRZFourier.model_validate(
            orjson.loads(rows.iloc[1]["json_constellaration_boundary"])
        )
        coilset_b = types.Coilset.model_validate(
            orjson.loads(rows.iloc[1]["json_desc_coilset"])
        )
        if (
            boundary_a.n_field_periods != boundary_b.n_field_periods
            or boundary_a.is_stellarator_symmetric
            != boundary_b.is_stellarator_symmetric
            or coilset_a.coil_x_n.shape[0] != coilset_b.coil_x_n.shape[0]
        ):
            raise RuntimeError(
                "training rows disagree on architecture-defining shape/scalars "
                "(only fourier_order is allowed to vary across rows)"
            )
    assert initial is not None
    return initial.model_copy(
        update={
            "is_stellarator_symmetric": bool(boundary_a.is_stellarator_symmetric),
            "n_field_periods": int(boundary_a.n_field_periods),
            "n_poloidal_modes": int(boundary_a.n_poloidal_modes),
            "n_toroidal_modes": int(boundary_a.n_toroidal_modes),
            "n_unique_coils": int(coilset_a.coil_x_n.shape[0]),
            "n_modes_coils_max": 2 * max_fourier_order + 1,
            "coil_currents_mean": coil_currents_mean,
            "coil_currents_std": coil_currents_std,
            "requirement_metrics_means": [float(x) for x in requirement_metrics_means],
            "requirement_metrics_stds": [float(x) for x in requirement_metrics_stds],
        }
    )


def train(
    train_config: types.TrainConfig,
) -> types.AnyModelCheckpoint:
    """Run training end-to-end and return the final-state model checkpoint.

    The concrete model architecture is selected by `train_config.model_kind`.
    The returned checkpoint type matches the chosen architecture.
    """
    checkpoint, _ = train_with_summary(train_config)
    return checkpoint


def train_with_summary(
    train_config: types.TrainConfig,
) -> tuple[types.AnyModelCheckpoint, TrainSummary]:
    """Loads the dataset for `train_config.model_kind` and trains on it.

    Variable-shape model types train on the variable-shape track, all others on the
    fixed-shape track. With `train_config.validation == "train_holdout"`, the model is
    validated on boundaries held out of the train split instead of the eval split.
    """
    track: Literal["fixed_shape", "variable_shape"] = (
        "variable_shape"
        if train_config.model_kind in types.VARIABLE_SHAPE_MODEL_KIND
        else "fixed_shape"
    )
    logger.info("Loading data frames (track=%s)...", track)
    train_df = data_utils.load_ml_baseline_dataframe(track=track, split="train")
    if train_config.validation == "train_holdout":
        train_df, eval_df = split_train_holdout(train_df, train_config.holdout_fraction)
        if len(eval_df) < train_config.eval_batch_size:
            raise ValueError(
                f"The holdout has {len(eval_df)} rows, fewer than "
                f"eval_batch_size={train_config.eval_batch_size}; raise "
                "holdout_fraction or lower eval_batch_size."
            )
    else:
        eval_df = data_utils.load_ml_baseline_dataframe(track=track, split="eval")
    return train_on_dataframes(train_config, train_df, eval_df)


def train_on_dataframes(
    train_config: types.TrainConfig,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> tuple[types.AnyModelCheckpoint, TrainSummary]:
    """Trains the model selected by `train_config.model_kind` on the given rows.

    Rows with outlier currents or curvatures are dropped from both frames, see
    `drop_outlier_currents` and `drop_outlier_curvatures`. Returns the checkpoint and a
    summary of the evaluation losses on `eval_df`. The checkpoint holds the final
    weights, or the weights at the best evaluation with `train_config.checkpoint_best`.
    """
    if jax.config.read("jax_enable_x64"):
        warnings.warn(
            "JAX is configured for float64 (jax_enable_x64=True). Training in "
            "float64 roughly halves CPU throughput vs float32. Unset "
            "JAX_ENABLE_X64 or call jax.config.update('jax_enable_x64', False) "
            "before importing this module."
        )

    logger.info("Train rows: %d, eval rows: %d", len(train_df), len(eval_df))

    coil_currents_mean, coil_currents_std = compute_coil_currents_stats(train_df)
    train_df = drop_outlier_currents(train_df, coil_currents_mean, coil_currents_std)
    eval_df = drop_outlier_currents(eval_df, coil_currents_mean, coil_currents_std)
    curvature_mean, curvature_std = compute_coil_curvature_stats(train_df)
    train_df = drop_outlier_curvatures(train_df, curvature_mean, curvature_std)
    eval_df = drop_outlier_curvatures(eval_df, curvature_mean, curvature_std)

    logger.info("Computing train-set requirement-metrics stats...")
    requirement_metrics_means, requirement_metrics_stds = (
        compute_requirement_metrics_stats(train_df)
    )

    dataset_max_fourier_order = max(
        infer_max_fourier_order(train_df),
        infer_max_fourier_order(eval_df),
    )

    logger.info("Resolving architecture config from data...")
    model_config = _resolve_model_config(
        train_config.model_config_overrides,
        train_config.model_kind,
        train_df,
        coil_currents_mean,
        coil_currents_std,
        requirement_metrics_means,
        requirement_metrics_stds,
        max_fourier_order=dataset_max_fourier_order,
    )
    resolved_train_config = train_config.model_copy(
        update={"model_config_overrides": model_config}
    )
    # Variable-shape rows mix coil counts, so coilsets are padded to common slots.
    pad_to_n_coils = (
        model_config.max_coils
        if isinstance(model_config, model_definition.AnyVariableShapeConfig)
        else None
    )

    eval_iter = make_batches(
        eval_df,
        train_config.eval_batch_size,
        train_config.seed,
        dataset_max_fourier_order,
        pad_to_n_coils=pad_to_n_coils,
    )
    eval_batch = next(eval_iter)
    del eval_iter
    train_iter = prefetched(
        make_batches(
            train_df,
            train_config.batch_size,
            train_config.seed,
            dataset_max_fourier_order,
            pad_to_n_coils=pad_to_n_coils,
        )
    )

    model: model_definition.AnyModel
    rngs = nnx.Rngs(train_config.seed)
    if train_config.model_kind == "mlp":
        assert isinstance(model_config, types.CoilPredictorConfig)
        model = model_definition.CoilPredictor(model_config, rngs=rngs)
    elif train_config.model_kind == "res_mlp":
        assert isinstance(model_config, types.ResMlpCoilPredictorConfig)
        model = model_definition.ResMlpCoilPredictor(model_config, rngs=rngs)
    elif train_config.model_kind == "mlp_ensemble":
        assert isinstance(model_config, types.MlpEnsembleCoilPredictorConfig)
        model = model_definition.MlpEnsembleCoilPredictor(model_config, rngs=rngs)
    elif train_config.model_kind == "res_mlp_ensemble":
        assert isinstance(model_config, types.ResMlpEnsembleCoilPredictorConfig)
        model = model_definition.ResMlpEnsembleCoilPredictor(model_config, rngs=rngs)
    elif train_config.model_kind == "per_coil":
        assert isinstance(model_config, types.PerCoilCoilPredictorConfig)
        model = model_definition.PerCoilCoilPredictor(model_config, rngs=rngs)
    elif train_config.model_kind == "masked":
        assert isinstance(model_config, types.MaskedCoilPredictorConfig)
        model = model_definition.MaskedCoilPredictor(model_config, rngs=rngs)
    elif train_config.model_kind == "per_coil_ensemble":
        assert isinstance(model_config, types.PerCoilEnsembleCoilPredictorConfig)
        model = model_definition.PerCoilEnsembleCoilPredictor(model_config, rngs=rngs)
    elif train_config.model_kind == "masked_ensemble":
        assert isinstance(model_config, types.MaskedEnsembleCoilPredictorConfig)
        model = model_definition.MaskedEnsembleCoilPredictor(model_config, rngs=rngs)
    else:
        raise NotImplementedError(
            f"model_kind {train_config.model_kind!r} not implemented"
        )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_config.learning_rate,
        warmup_steps=train_config.warmup_steps,
        decay_steps=train_config.steps,
        end_value=train_config.end_lr,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(train_config.grad_clip),
        optax.adamw(schedule, weight_decay=train_config.weight_decay),
    )
    _, init_params, _ = nnx.split(model, nnx.Param, ...)
    opt_state = tx.init(init_params)  # pyright: ignore[reportArgumentType]

    @nnx.jit
    def update_step(model, opt_state, batch: types.Batch):
        model.train()

        def loss_fn(model):
            loss, aux = _loss_fn(model, batch)
            return loss, aux

        (total, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        grad_norm = optax.global_norm(grads)
        _, params, _ = nnx.split(model, nnx.Param, ...)
        updates, opt_state = tx.update(grads, opt_state, params)  # pyright: ignore[reportArgumentType]
        new_params = optax.apply_updates(params, updates)  # pyright: ignore[reportArgumentType, reportAssignmentType]
        nnx.update(model, new_params)
        return opt_state, total, aux, grad_norm

    @nnx.jit
    def eval_step(model, batch):
        model.eval()
        return _loss_fn(model, batch)

    owns_wandb_run = wandb.run is None
    if not owns_wandb_run:
        assert wandb.run is not None
        run = wandb.run
    else:
        run = wandb.init(project="coilstellaration")
    run.config.update(
        {
            "train_config": resolved_train_config.model_dump(mode="json"),
            "jax_version": jax.__version__,
            "jax_devices": [str(d) for d in jax.devices()],
            "param_count": int(
                sum(int(jnp.size(x)) for x in jax.tree_util.tree_leaves(init_params))
            ),
            "n_train_rows": (len(train_df)),
            "n_eval_rows": (len(eval_df)),
        }
    )

    best_eval = float("inf")
    best_step = -1
    final_eval = float("inf")
    best_model = model
    loss_components = ("coil_x_n", "coil_y_n", "coil_z_n", "currents")

    logger.info("Starting training...")
    pending_train_payloads: list[dict[str, object]] = []
    time_start = time.time_ns()
    for step in range(train_config.steps):
        batch = next(train_iter)
        opt_state, train_loss, train_aux, grad_norm = update_step(
            model, opt_state, batch
        )
        train_payload: dict[str, object] = {
            "step": step,
            "train/loss": train_loss,
            "train/lr": schedule(step),
            "train/grad_norm": grad_norm,
        }
        for c in loss_components:
            train_payload[f"train/loss_{c}"] = train_aux[c]
        pending_train_payloads.append(train_payload)

        if (
            step == 0
            or (step + 1) % train_config.eval_every == 0
            or (step + 1) == train_config.steps
        ):
            eval_loss, eval_aux = eval_step(model, eval_batch)
            eval_payload: dict[str, object] = {"eval/loss": eval_loss}
            for c in loss_components:
                eval_payload[f"eval/loss_{c}"] = eval_aux[c]

            # Single bulk host sync for the whole window.
            train_synced, eval_synced = jax.device_get(
                (pending_train_payloads, eval_payload)
            )
            for p in train_synced:
                s = int(p.pop("step"))
                wandb.log({k: float(v) for k, v in p.items()}, step=s, commit=False)
            wandb.log(
                {k: float(v) for k, v in eval_synced.items()},
                step=step,
                commit=True,
            )
            pending_train_payloads = []

            eval_loss_f = float(eval_synced["eval/loss"])
            train_loss_f = float(train_synced[-1]["train/loss"])
            final_eval = eval_loss_f
            if eval_loss_f < best_eval:
                logger.info(
                    "Step %6d  train=%.6f  eval=%.6f",
                    step + 1,
                    train_loss_f,
                    eval_loss_f,
                )
                best_eval = eval_loss_f
                best_step = step
                if train_config.checkpoint_best:
                    best_model = nnx.clone(model)
            elif (
                train_config.early_stopping_patience is not None
                and (step - best_step) >= train_config.early_stopping_patience
            ):
                logger.info("Early stopping at step %d", step)
                break

            if time.time_ns() - time_start > train_config.max_wall_time_s * 1e9:
                logger.info("Max wall time exceeded; stopping at step %d", step)
                break

    if wandb.run is None:
        raise RuntimeError("wandb.run is None — wandb.init must have failed silently")
    wandb.run.summary["best_eval_loss"] = best_eval
    wandb.run.summary["best_step"] = best_step
    wandb.run.summary["final_eval_loss"] = final_eval

    final_model = best_model if train_config.checkpoint_best else model
    checkpoint: types.AnyModelCheckpoint
    if train_config.model_kind == "mlp":
        checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
            final_model, model_config, types.CoilPredictorCheckpoint
        )
    elif train_config.model_kind == "res_mlp":
        checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
            final_model, model_config, types.ResMlpCoilPredictorCheckpoint
        )
    elif train_config.model_kind == "mlp_ensemble":
        checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
            final_model, model_config, types.MlpEnsembleCoilPredictorCheckpoint
        )
    elif train_config.model_kind == "res_mlp_ensemble":
        checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
            final_model, model_config, types.ResMlpEnsembleCoilPredictorCheckpoint
        )
    elif train_config.model_kind == "per_coil":
        checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
            final_model, model_config, types.PerCoilCoilPredictorCheckpoint
        )
    elif train_config.model_kind == "masked":
        checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
            final_model, model_config, types.MaskedCoilPredictorCheckpoint
        )
    elif train_config.model_kind == "per_coil_ensemble":
        checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
            final_model, model_config, types.PerCoilEnsembleCoilPredictorCheckpoint
        )
    elif train_config.model_kind == "masked_ensemble":
        checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
            final_model, model_config, types.MaskedEnsembleCoilPredictorCheckpoint
        )
    else:
        raise ValueError(f"unsupported model_kind {train_config.model_kind!r}")
    if owns_wandb_run:
        wandb.finish()

    logger.info("Best eval loss %.6f at step %d", best_eval, best_step)
    return checkpoint, TrainSummary(
        best_eval_loss=best_eval,
        best_step=best_step,
        final_eval_loss=final_eval,
        n_train_rows=len(train_df),
        n_eval_rows=len(eval_df),
    )


def _stable_unit_interval(key: str) -> float:
    # sha256 instead of `hash`, which is salted per process.
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _resolve_variable_shape_config(
    initial: types.AnyModelConfig | None,
    model_kind: types.AnyModelLiteral,
    train_df: pd.DataFrame,
    coil_currents_mean: float,
    coil_currents_std: float,
    requirement_metrics_means: jt.Float[NpOrJaxArray, " n_requirements"],
    requirement_metrics_stds: jt.Float[NpOrJaxArray, " n_requirements"],
    max_fourier_order: int,
) -> model_definition.AnyVariableShapeConfig:
    config_type = _VARIABLE_SHAPE_CONFIG_TYPES[model_kind]
    rows = train_df.dropna(subset=list(data_utils.JSON_COLUMNS))
    if len(rows) < 1:
        raise RuntimeError("training set is empty after filtering")
    boundaries = [orjson.loads(s) for s in rows["json_constellaration_boundary"]]
    first = boundaries[0]
    boundary_shape = np.shape(first["r_cos"])
    for boundary in boundaries:
        if np.shape(boundary["r_cos"]) != boundary_shape:
            raise ValueError(
                "training rows disagree on the boundary resolution: "
                f"{np.shape(boundary['r_cos'])} vs {boundary_shape}"
            )
        if boundary["is_stellarator_symmetric"] != first["is_stellarator_symmetric"]:
            raise ValueError("training rows disagree on boundary stellarator symmetry")
    n_coils_values = sorted({int(n) for n in rows[data_utils.N_COILS_COLUMN]})
    base = initial.model_dump() if initial is not None else {}
    return config_type.model_validate(
        {
            **base,
            "is_stellarator_symmetric": bool(first["is_stellarator_symmetric"]),
            "n_poloidal_modes": int(boundary_shape[0]),
            "n_toroidal_modes": int(boundary_shape[1]),
            "n_modes_coils_max": 2 * max_fourier_order + 1,
            "coil_currents_mean": coil_currents_mean,
            "coil_currents_std": coil_currents_std,
            "n_field_periods_values": sorted(
                {int(b["n_field_periods"]) for b in boundaries}
            ),
            "n_coils_values": n_coils_values,
            "max_coils": max(n_coils_values),
            "requirement_metrics_means": [float(x) for x in requirement_metrics_means],
            "requirement_metrics_stds": [float(x) for x in requirement_metrics_stds],
        }
    )


def _masked_mse_coef(
    p: jt.Float[NpOrJaxArray, "batch n_unique n_modes"],
    t: jt.Float[NpOrJaxArray, "batch n_unique n_modes"],
    mask: jt.Bool[NpOrJaxArray, "batch n_modes"],
    coil_mask: jt.Float[NpOrJaxArray, "batch n_unique"],
    eps: float = 1e-8,
) -> jt.Float[NpOrJaxArray, " *dims"]:
    # Exact-one weights keep this bit-identical to the unpadded computation.
    mask_b = mask[:, None, :] * coil_mask[:, :, None]  # (B, U, 2N+1)
    n_real = jnp.maximum(mask_b.sum(), 1.0)
    sum_real = (t * mask_b).sum()
    mean_t = sum_real / n_real
    var_t = (((t - mean_t) ** 2) * mask_b).sum() / n_real
    std = jnp.where(var_t > eps, jnp.sqrt(var_t), 1.0)
    sq_err = ((p - t) / std) ** 2 * mask_b
    return sq_err


def _current_mse(
    p: jt.Float[NpOrJaxArray, "batch n_unique"],
    t: jt.Float[NpOrJaxArray, "batch n_unique"],
    coil_mask: jt.Float[NpOrJaxArray, "batch n_unique"],
    std: jt.Float[NpOrJaxArray, " "] | None = None,
    eps: float = 1e-8,
) -> jt.Float[NpOrJaxArray, " *dims"]:
    if std is None:
        n_real = jnp.maximum(coil_mask.sum(), 1.0)
        mean_t = (t * coil_mask).sum() / n_real
        std = jnp.sqrt((((t - mean_t) ** 2) * coil_mask).sum() / n_real)
    scale = jnp.where(std > eps, std, jnp.ones_like(std))
    return ((p - t) / scale) ** 2 * coil_mask


_PER_SAMPLE_RNGS = nnx.StateAxes({nnx.RngState: 0, ...: None})


def _eval_model(
    model: nnx.Module,
    boundaries: surface_rz_fourier.SurfaceRZFourier,
    requirement_metrics: types.RequirementMetrics,
    max_fourier_order: int,
) -> types.Coilset:
    @nnx.split_rngs(splits=len(boundaries.r_cos))
    @nnx.vmap(in_axes=(_PER_SAMPLE_RNGS, 0, 0, None))
    def eval_sample(
        model: nnx.Module,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        max_fourier_order: int,
    ) -> types.Coilset:
        return model(boundary, requirement_metrics, fourier_order=max_fourier_order)

    return eval_sample(model, boundaries, requirement_metrics, max_fourier_order)


def _eval_per_member(
    model: model_definition.AnyEnsembleModel,
    boundaries: surface_rz_fourier.SurfaceRZFourier,
    requirement_metrics: types.RequirementMetrics,
    max_fourier_order: int,
) -> types.Coilset:
    @nnx.split_rngs(splits=len(boundaries.r_cos))
    @nnx.vmap(in_axes=(_PER_SAMPLE_RNGS, 0, 0, None))
    def eval_sample(
        model: model_definition.AnyEnsembleModel,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        max_fourier_order: int,
    ) -> types.Coilset:
        return model.predict_per_member(
            boundary, requirement_metrics, fourier_order=max_fourier_order
        )

    return eval_sample(model, boundaries, requirement_metrics, max_fourier_order)


def _loss_fn(
    model: model_definition.AnyModel,
    batch: types.Batch,
) -> tuple[jt.Float[NpOrJaxArray, " "], dict[str, jt.Float[NpOrJaxArray, " "]]]:
    """Per-component masked MSE; the aggregate is the mean over all entries.

    Every Fourier coefficient and every current of a real coil slot is one entry, so
    each of the three coefficient components outweighs the currents by its number of
    modes. The fourier-order mask suppresses contributions from zero-padded
    coefficient indices so small-`N` and large-`N` samples weigh equally per
    real coefficient. Per-component std is computed over real (mask=1)
    entries only to avoid bias from zero pad. Padded coil slots (`coil_mask` = 0)
    contribute nothing to errors, means, stds or gradients. Currents are not
    Fourier-masked; their errors are divided by `coil_currents_std`.

    For ensemble models (`model_definition.AnyEnsembleModel`), members are graded
    independently against the target and the per-member losses are averaged: this
    trains each member as if it were a standalone model (no cross-member gradient
    coupling), preserving ensemble diversity that loss-of-mean would erode.

    Returns `(scalar_loss, aux)` where `aux[c]` is the masked MSE for each coil
    component (`coil_x_n`, `coil_y_n`, `coil_z_n`, `currents`).
    """

    currents_std = jnp.asarray(model.config.coil_currents_std)

    if isinstance(
        model,
        (
            model_definition.MlpEnsembleCoilPredictor,
            model_definition.ResMlpEnsembleCoilPredictor,
            model_definition.PerCoilEnsembleCoilPredictor,
            model_definition.MaskedEnsembleCoilPredictor,
        ),
    ):
        pred = _eval_per_member(
            model,
            batch.boundaries,
            batch.requirement_metrics,
            batch.batch_max_fourier_order,
        )
        per_member = True
    else:
        pred = _eval_model(
            model,
            batch.boundaries,
            batch.requirement_metrics,
            batch.batch_max_fourier_order,
        )
        per_member = False
    return _coilset_loss(
        pred,
        batch.coilsets,
        batch.fourier_order_mask,
        batch.coil_mask,
        currents_std,
        per_member=per_member,
    )


def _coilset_loss(
    pred: types.Coilset,
    target: types.Coilset,
    mask: jt.Float[NpOrJaxArray, "batch n_modes"],
    coil_mask: jt.Float[NpOrJaxArray, "batch n_unique"],
    currents_std: jt.Float[NpOrJaxArray, " "],
    *,
    per_member: bool,
) -> tuple[jt.Float[NpOrJaxArray, " "], dict[str, jt.Float[NpOrJaxArray, " "]]]:
    """Masked loss of predictions against targets.

    With `per_member`, `pred` leaves carry an ensemble axis at position 1
    (`(B, M, U, 2N+1)` and `(B, M, U)`); otherwise `(B, U, 2N+1)` and `(B, U)`.
    """
    eps = 1e-8
    if per_member:
        # vmap over the member axis, broadcasting target/mask.
        member_coef = jax.vmap(_masked_mse_coef, in_axes=(1, None, None, None, None))
        member_current = jax.vmap(_current_mse, in_axes=(1, None, None, None, None))
        aux = {
            "coil_x_n": member_coef(
                pred.coil_x_n, target.coil_x_n, mask, coil_mask, eps
            ),
            "coil_y_n": member_coef(
                pred.coil_y_n, target.coil_y_n, mask, coil_mask, eps
            ),
            "coil_z_n": member_coef(
                pred.coil_z_n, target.coil_z_n, mask, coil_mask, eps
            ),
            "currents": member_current(
                pred.currents, target.currents, coil_mask, currents_std, eps
            ),
        }
    else:
        aux = {
            "coil_x_n": _masked_mse_coef(
                pred.coil_x_n, target.coil_x_n, mask, coil_mask, eps
            ),
            "coil_y_n": _masked_mse_coef(
                pred.coil_y_n, target.coil_y_n, mask, coil_mask, eps
            ),
            "coil_z_n": _masked_mse_coef(
                pred.coil_z_n, target.coil_z_n, mask, coil_mask, eps
            ),
            "currents": _current_mse(
                pred.currents, target.currents, coil_mask, currents_std, eps=eps
            ),
        }
    # Weights count each real coil slot once per element; padded slots drop out of the
    # denominators so they do not dilute the means.
    weights = {
        name: jnp.broadcast_to(
            coil_mask[..., None] if name != "currents" else coil_mask,
            err.shape,
        )
        for name, err in aux.items()
    }
    total = jnp.sum(jnp.concatenate([aux[k].flatten() for k in aux])) / jnp.sum(
        jnp.concatenate([weights[k].flatten() for k in aux])
    )
    means = {k: aux[k].sum() / weights[k].sum() for k in aux}
    return total, means
