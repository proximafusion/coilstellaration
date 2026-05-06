from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax
import orjson
import pandas as pd
import wandb
from flax import nnx
from geometry.surface import surface_types
from util import pytree
from util.types import NpOrJaxArray
from util.wandb import wandb_types
from version_control import git

from coilstellaration import (
    coilset_utils,
    data_utils,
    flax_nnx_checkpoint_util,
    types,
)
from coilstellaration.machine_learning import (
    model_definition,
    types,
)

logger = logging.getLogger(__name__)


def compute_coil_currents_stats(
    train_df: pd.DataFrame,
) -> nnx.metrics.Statistics:
    """Population-mean currents over the training rows, decoding coilsets once."""
    welford = nnx.metrics.Welford()

    logger.info("Computing mean currents from %d training rows...", len(train_df))
    for json_str in train_df["json_desc_coilset"].dropna():
        c = types.CoilStellarationCoilset.model_validate(orjson.loads(json_str))
        currents = jnp.asarray(c.currents)
        welford.update(values=currents)
    stats = welford.compute()
    mean = stats.mean
    logger.info("Computed mean currents to be %s", mean)
    return stats


def _decode_coilset(
    row: pd.Series,
) -> types.Coilset:
    return types.CoilStellarationCoilset.model_validate(
        orjson.loads(row["json_desc_coilset"])
    )


def _decode_boundary(row: pd.Series) -> surface_types.SurfaceRZFourier:
    return surface_types.SurfaceRZFourier.model_validate(
        orjson.loads(row["json_constellaration_boundary"])
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


def make_batches(
    df: pd.DataFrame,
    batch_size: int,
    seed: int,
    pad_to_fourier_order: int,
) -> Iterator[types.Batch]:
    """Yields infinite training batches in seeded shuffled order."""
    key = jax.random.PRNGKey(seed)
    json_df = df[
        [*data_utils.JSON_COLUMNS, *data_utils.REQUIREMENT_METRICS_COLUMNS]
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
                subset_requirement_metrics,
            ) = pytree.tree_map(
                lambda a: a[batch_indices],
                (all_boundaries, all_coilsets, all_masks, all_requirement_metrics),
            )

            yield types.Batch.model_construct(
                boundaries=subset_boundaries,
                coilsets=subset_coilsets,
                requirement_metrics=subset_requirement_metrics,
                fourier_order_mask=subset_masks,
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
        coilset = types.CoilStellarationCoilset.model_validate(orjson.loads(json_str))
        max_order = max(max_order, int(coilset.fourier_order))
    return max_order


def _resolve_model_config(
    initial: types.AnyModelConfig | None,
    model_type: Literal["mlp", "res_mlp", "mlp_ensemble", "res_mlp_ensemble"],
    train_df: pd.DataFrame,
    coil_currents_mean: float,
    coil_currents_std: float,
    requirement_metrics_means: jt.Float[jt.Array, " n_requirements"],
    requirement_metrics_stds: jt.Float[jt.Array, " n_requirements"],
    max_fourier_order: int,
) -> types.AnyModelConfig:
    """Fills data-derived fields on the architecture config from the training set.

    Architecture knobs (``hidden_dim``, layer counts, spectral params) come
    from ``initial``; defaults applied if ``initial`` is ``None`` for the
    architecture selected by ``model_type``. Data-derived fields are taken
    from a single training row and cross-checked against a second row to
    surface dataset inconsistency early.
    """
    if initial is None:
        match model_type:
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
                raise ValueError(f"unsupported model_type {model_type!r}")
    rows = train_df.dropna(subset=list(data_utils.JSON_COLUMNS)).head(2)
    if len(rows) < 1:
        raise RuntimeError("training set is empty after filtering")
    boundary_a = surface_types.SurfaceRZFourier.model_validate(
        orjson.loads(rows.iloc[0]["json_constellaration_boundary"])
    )
    coilset_a = types.CoilStellarationCoilset.model_validate(
        orjson.loads(rows.iloc[0]["json_desc_coilset"])
    )
    if len(rows) >= 2:
        boundary_b = surface_types.SurfaceRZFourier.model_validate(
            orjson.loads(rows.iloc[1]["json_constellaration_boundary"])
        )
        coilset_b = types.CoilStellarationCoilset.model_validate(
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
) -> (
    types.CoilPredictorCheckpoint
    | types.ResMlpCoilPredictorCheckpoint
    | types.MlpEnsembleCoilPredictorCheckpoint
    | types.ResMlpEnsembleCoilPredictorCheckpoint
):
    """Run training end-to-end and return the final-state model checkpoint.

    The concrete model architecture is selected by `train_config.model_type`.
    The returned checkpoint type matches the chosen architecture.
    """
    if jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "JAX is configured for float64 (jax_enable_x64=True). Training in "
            "float64 roughly halves CPU throughput vs float32. Unset "
            "JAX_ENABLE_X64 or call jax.config.update('jax_enable_x64', False) "
            "before importing this module."
        )

    wandb_settings = train_config.wandb_settings
    if wandb_settings is None:
        wandb_settings = wandb_types.WandBSettings(
            project="constellaration_update_coil_predictor"
        )

    logger.info("Loading data frames...")
    train_df = data_utils.load_dataframes(
        "train",
        filtered=True,
    )
    eval_df = data_utils.load_dataframes(
        "eval",
        filtered=True,
    )
    logger.info("Train rows: %d, eval rows: %d", len(train_df), len(eval_df))

    logger.info("Computing train-set mean currents (one streaming pass)...")
    coil_current_stats = compute_coil_currents_stats(train_df)

    logger.info("Computing train-set requirement-metrics stats...")
    stacked_requirements = jnp.asarray(
        train_df[list(data_utils.REQUIREMENT_METRICS_COLUMNS)]
        .dropna()
        .to_numpy(dtype=float)
    )
    requirement_metrics_means = stacked_requirements.mean(axis=0)
    requirement_metrics_stds = jnp.maximum(stacked_requirements.std(axis=0), 1e-8)

    dataset_max_fourier_order = max(
        infer_max_fourier_order(train_df),
        infer_max_fourier_order(eval_df),
    )

    logger.info("Resolving architecture config from data...")
    model_config = _resolve_model_config(
        train_config.model_config_overrides,
        train_config.model_type,
        train_df,
        float(coil_current_stats.mean),
        float(coil_current_stats.standard_deviation),
        requirement_metrics_means,
        requirement_metrics_stds,
        max_fourier_order=dataset_max_fourier_order,
    )
    resolved_train_config = train_config.model_copy(
        update={"model_config_overrides": model_config}
    )

    eval_iter = make_batches(
        eval_df,
        train_config.eval_batch_size,
        train_config.seed,
        dataset_max_fourier_order,
    )
    eval_batch = next(eval_iter)
    del eval_iter
    train_iter = prefetched(
        make_batches(
            train_df,
            train_config.batch_size,
            train_config.seed,
            dataset_max_fourier_order,
        )
    )

    model: model_definition.AnyModel
    rngs = nnx.Rngs(train_config.seed)
    if train_config.model_type == "mlp":
        assert isinstance(model_config, types.CoilPredictorConfig)
        model = model_definition.CoilPredictor(model_config, rngs=rngs)
    elif train_config.model_type == "res_mlp":
        assert isinstance(model_config, types.ResMlpCoilPredictorConfig)
        model = model_definition.ResMlpCoilPredictor(model_config, rngs=rngs)
    elif train_config.model_type == "mlp_ensemble":
        assert isinstance(model_config, types.MlpEnsembleCoilPredictorConfig)
        model = model_definition.MlpEnsembleCoilPredictor(model_config, rngs=rngs)
    elif train_config.model_type == "res_mlp_ensemble":
        assert isinstance(model_config, types.ResMlpEnsembleCoilPredictorConfig)
        model = model_definition.ResMlpEnsembleCoilPredictor(model_config, rngs=rngs)
    else:
        raise NotImplementedError(
            f"model_type {train_config.model_type!r} not implemented"
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
        updates, opt_state = tx.update(
            grads, opt_state, params
        )  # pyright: ignore[reportArgumentType]
        new_params = optax.apply_updates(
            params, updates
        )  # pyright: ignore[reportArgumentType, reportAssignmentType]
        nnx.update(model, new_params)
        return opt_state, total, aux, grad_norm

    @nnx.jit
    def eval_step(model, batch):
        model.eval()
        return _loss_fn(model, batch)

    wandb.init(
        project=wandb_settings.project,
        id=wandb_settings.id,
        name=wandb_settings.id,
        mode=wandb_settings.mode,
        config={
            "train_config": resolved_train_config.model_dump(mode="json"),
            "git_sha": git.get_git_info().commit or "unknown",
            "jax_version": jax.__version__,
            "jax_devices": [str(d) for d in jax.devices()],
            "param_count": int(
                sum(int(jnp.size(x)) for x in jax.tree_util.tree_leaves(init_params))
            ),
            "n_train_rows": int(len(train_df)),
            "n_eval_rows": int(len(eval_df)),
        },
    )

    best_eval = float("inf")
    best_step = -1
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
            if eval_loss_f < best_eval:
                logger.info(
                    "Step %6d  train=%.6f  eval=%.6f",
                    step + 1,
                    train_loss_f,
                    eval_loss_f,
                )
                best_eval = eval_loss_f
                best_step = step
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

    final_model = model
    checkpoint: (
        types.CoilPredictorCheckpoint
        | types.ResMlpCoilPredictorCheckpoint
        | types.MlpEnsembleCoilPredictorCheckpoint
        | types.ResMlpEnsembleCoilPredictorCheckpoint
    )
    if train_config.model_type == "mlp":
        checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
            final_model, model_config, types.CoilPredictorCheckpoint
        )
    elif train_config.model_type == "res_mlp":
        checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
            final_model, model_config, types.ResMlpCoilPredictorCheckpoint
        )
    elif train_config.model_type == "mlp_ensemble":
        checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
            final_model, model_config, types.MlpEnsembleCoilPredictorCheckpoint
        )
    elif train_config.model_type == "res_mlp_ensemble":
        checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
            final_model, model_config, types.ResMlpEnsembleCoilPredictorCheckpoint
        )
    else:
        raise ValueError(f"unsupported model_type {train_config.model_type!r}")
    wandb.finish()

    logger.info("Best eval loss %.6f at step %d", best_eval, best_step)
    return checkpoint


def _masked_mse_coef(
    p: jt.Float[NpOrJaxArray, "batch n_unique n_modes"],
    t: jt.Float[NpOrJaxArray, "batch n_unique n_modes"],
    mask: jt.Bool[NpOrJaxArray, "batch n_unique"],
    eps: float = 1e-8,
) -> jt.Float[NpOrJaxArray, " *dims"]:
    mask_b = mask[:, None, :]  # (B, 1, 2N+1) broadcasts over n_unique
    n_real = jnp.maximum(mask_b.sum() * t.shape[1], 1.0)
    sum_real = (t * mask_b).sum()
    mean_t = sum_real / n_real
    var_t = (((t - mean_t) ** 2) * mask_b).sum() / n_real
    std = jnp.where(var_t > eps, jnp.sqrt(var_t), 1.0)
    sq_err = ((p - t) / std) ** 2 * mask_b
    return sq_err


def _current_mse_unmasked(
    p: jt.Float[NpOrJaxArray, " *dims"],
    t: jt.Float[NpOrJaxArray, " *dims"],
    std: jt.Float[NpOrJaxArray, " "] | None = None,
    eps: float = 1e-8,
) -> jt.Float[NpOrJaxArray, " *dims"]:
    if std is None:
        std = jnp.std(t)
    scale = jnp.where(std > eps, std, jnp.ones_like(std))
    return ((p - t) / scale) ** 2


@nnx.vmap(in_axes=(None, 0, 0, None))
def _eval_model(
    model: nnx.Module,
    boundary: surface_types.SurfaceRZFourier,
    requirement_metrics: types.RequirementMetrics,
    max_fourier_order: int,
) -> types.Coilset:
    pred = model(boundary, requirement_metrics, fourier_order=max_fourier_order)
    return pred


@nnx.vmap(in_axes=(None, 0, 0, None))
def _eval_per_member(
    model: model_definition.AnyEnsembleModel,
    boundary: surface_types.SurfaceRZFourier,
    requirement_metrics: types.RequirementMetrics,
    max_fourier_order: int,
) -> types.Coilset:
    return model.predict_per_member(
        boundary, requirement_metrics, fourier_order=max_fourier_order
    )


def _loss_fn(
    model: model_definition.AnyModel,
    batch: types.Batch,
) -> tuple[jt.Float[NpOrJaxArray, " "], dict[str, jt.Float[NpOrJaxArray, " "]]]:
    """Per-component masked MSE; aggregate is the unweighted mean.

    The fourier-order mask suppresses contributions from zero-padded
    coefficient indices so small-`N` and large-`N` samples weigh equally per
    real coefficient. Per-component std is computed over real (mask=1)
    entries only to avoid bias from zero pad. Currents are unmasked.

    For ensemble models (`MlpEnsembleCoilPredictor`,
    `ResMlpEnsembleCoilPredictor`), members are graded independently against
    the target and the per-member losses are averaged: this trains each
    member as if it were a standalone model (no cross-member gradient
    coupling), preserving ensemble diversity that loss-of-mean would erode.

    Returns ``(scalar_loss, aux)`` where ``aux[c]`` is the masked MSE for
    each coil component (``coil_x_n``, ``coil_y_n``, ``coil_z_n``,
    ``currents``).
    """

    target = batch.coilsets
    mask = batch.fourier_order_mask  # (B, 2*N+1)
    eps = 1e-8
    currents_std = jnp.asarray(model.config.coil_currents_std)

    if isinstance(
        model,
        (
            model_definition.MlpEnsembleCoilPredictor,
            model_definition.ResMlpEnsembleCoilPredictor,
        ),
    ):
        # pred.coil_*: (B, M, U, 2N+1); pred.currents: (B, M, U).
        # vmap masked-MSE over the member axis, broadcasting target/mask.
        pred = _eval_per_member(
            model,
            batch.boundaries,
            batch.requirement_metrics,
            batch.batch_max_fourier_order,
        )
        member_masked_mse = jax.vmap(_masked_mse_coef, in_axes=(1, None, None, None))
        member_current_mse = jax.vmap(
            _current_mse_unmasked, in_axes=(1, None, None, None)
        )
        aux = {
            "coil_x_n": member_masked_mse(pred.coil_x_n, target.coil_x_n, mask, eps),
            "coil_y_n": member_masked_mse(pred.coil_y_n, target.coil_y_n, mask, eps),
            "coil_z_n": member_masked_mse(pred.coil_z_n, target.coil_z_n, mask, eps),
            "currents": member_current_mse(
                pred.currents, target.currents, currents_std, eps
            ),
        }
    else:
        pred = _eval_model(
            model,
            batch.boundaries,
            batch.requirement_metrics,
            batch.batch_max_fourier_order,
        )
        aux = {
            "coil_x_n": _masked_mse_coef(pred.coil_x_n, target.coil_x_n, mask, eps),
            "coil_y_n": _masked_mse_coef(pred.coil_y_n, target.coil_y_n, mask, eps),
            "coil_z_n": _masked_mse_coef(pred.coil_z_n, target.coil_z_n, mask, eps),
            "currents": _current_mse_unmasked(
                pred.currents, target.currents, currents_std, eps=eps
            ),
        }
    total = jnp.mean(
        jnp.concatenate(
            [
                aux["coil_x_n"].flatten(),
                aux["coil_y_n"].flatten(),
                aux["coil_z_n"].flatten(),
                aux["currents"].flatten(),
            ]
        )
    )
    return total, jax.tree_util.tree_map(jnp.mean, aux)
