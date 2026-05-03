from __future__ import annotations

import functools
import logging
import queue
import threading
from collections.abc import Iterator
from typing import Literal

import datasets
import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax
import orjson
import pandas as pd
import pyarrow.fs as pf
from constellaration.geometry import surface_rz_fourier
from constellaration.utils import pytree
from flax import nnx

from constellaration_update import types as constellaration_update_types
from constellaration_update.checkpoint import flax_nnx as flax_nnx_checkpoint
from constellaration_update.coilset import utils as coilset_utils
from constellaration_update.machine_learning import model as model_definition
from constellaration_update.machine_learning import types
from constellaration_update.utils.types import NpOrJaxArray

logger = logging.getLogger(__name__)


JSON_COLUMNS = (
    "json_desc_coilset",
    "json_constellaration_boundary",
)

REQUIREMENT_METRICS_COLUMNS = (
    "desc_metrics/normalized_coil_to_coil_min_distances/min",
    "desc_metrics/normalized_coil_to_plasma_min_distances/min",
    "desc_metrics/normalized_coil_curvatures/max",
    "desc_metrics/normalized_field_error/max",
)
FIELD_ERROR_MEAN_COLUMN = "desc_metrics/normalized_field_error/mean"

gcs = pf.GcsFileSystem()


def load_dataframes(
    split: Literal["train", "eval"],
    relative_min_coil_to_plasma_distance_error_threshold: float,
) -> pd.DataFrame:
    constellaration_dataset = datasets.load_dataset(
        "proxima-fusion/constellaration", name="default", split="train"
    )
    assert type(constellaration_dataset) is datasets.Dataset
    _constellaration_update_dataset = datasets.load_dataset(
        "proxima-fusion/constellaration_update", split=split
    )
    assert type(_constellaration_update_dataset) is datasets.Dataset
    constellaration_update_dataset = _constellaration_update_dataset
    _constellaration_update_coilsets = datasets.load_dataset(
        "proxima-fusion/constellaration_update", name="coilsets", split="train"
    )
    assert type(_constellaration_update_coilsets) is datasets.Dataset
    constellaration_update_coilsets = _constellaration_update_coilsets
    _constellaration_update_requirements = datasets.load_dataset(
        "proxima-fusion/constellaration_update", name="requirements", split="train"
    )
    assert type(_constellaration_update_requirements) is datasets.Dataset
    constellaration_update_requirements = _constellaration_update_requirements

    _results_df = constellaration_update_dataset.to_pandas(batched=False)
    assert type(_results_df) is pd.DataFrame
    results_df = _results_df.rename(
        columns={"plasma_configuration_id": "constellaration_boundary_id"}
    )
    _coilsets_df = constellaration_update_coilsets.to_pandas(batched=False)
    assert type(_coilsets_df) is pd.DataFrame
    coilsets_df = _coilsets_df.set_index("id").rename(
        columns={"json": "json_desc_coilset"}
    )
    _requirements_df = constellaration_update_requirements.to_pandas(batched=False)
    assert type(_requirements_df) is pd.DataFrame
    requirements_df = _requirements_df.set_index("id").rename(
        columns={"json": "json_requirements"}
    )
    _constellaration_df = constellaration_dataset.to_pandas(batched=False)
    assert type(_constellaration_df) is pd.DataFrame
    constellaration_df = _constellaration_df.set_index("plasma_config_id").rename(
        columns={"boundary.json": "json_constellaration_boundary"}
    )
    constellaration_df = constellaration_df[
        ["json_constellaration_boundary", "boundary.n_field_periods"]
        + [
            col
            for col in constellaration_df.columns
            if col.startswith("metric") and col not in {"metrics.id", "metrics.json"}
        ]
    ].rename(columns=lambda x: x.replace("metrics.", "constellaration_metrics/"))
    merged_df = (
        results_df.join(coilsets_df, on="desc_coilset_id")
        .join(requirements_df, on="requirements_id")
        .join(constellaration_df, on="constellaration_boundary_id")
    )
    filtered_df = merged_df.loc[
        (
            (
                (
                    merged_df[
                        "desc_metrics/normalized_coil_to_plasma_min_distances/mean"
                    ]
                    - merged_df["reqs/normalized_min_coil_plasma_distance"]
                )
                / merged_df["reqs/normalized_min_coil_plasma_distance"]
            )
            >= relative_min_coil_to_plasma_distance_error_threshold
        )
        & (merged_df["boundary.n_field_periods"] == 3)
        & (merged_df["reqs/n_coils_per_half_period"] == 5)
    ].dropna(axis=0, how="any", subset=list(JSON_COLUMNS))

    return filtered_df


@functools.lru_cache(maxsize=10)
def load_dataset(
    split: Literal["train", "eval"],
    relative_min_coil_to_plasma_distance_error_threshold: float,
    n: int = 0,
) -> list[types.EvalData]:
    """Materialize the deterministic eval split as per-sample lists."""
    logger.info(
        "Loading dataset for split=%r with "
        "relative_min_coil_to_plasma_distance_error_threshold=%.3g...",
        split,
        relative_min_coil_to_plasma_distance_error_threshold,
    )
    eval_df = load_dataframes(
        split,
        relative_min_coil_to_plasma_distance_error_threshold=relative_min_coil_to_plasma_distance_error_threshold,
    )
    eval_datas = []
    logger.info("Processing eval rows to build dataset artifacts...")
    for i, (_, row) in enumerate(eval_df.iterrows()):
        if n > 0 and i >= n:
            break
        eval_data = types.EvalData(
            boundary=surface_rz_fourier.SurfaceRZFourier.model_validate(
                orjson.loads(row["json_constellaration_boundary"])
            ),
            boundary_id=row["constellaration_boundary_id"],
            true_coilset=constellaration_update_types.ConStellarationUpdateCoilset.model_validate(
                orjson.loads(row["json_desc_coilset"])
            ),
            true_metrics=row["desc_metrics_id"],
            requirement_metrics=row_to_requirement_metrics(row),
        )
        eval_datas.append(eval_data)
    logger.info("Assembled evaluation dataset with %d samples.", len(eval_datas))
    return eval_datas


def row_to_requirement_metrics(
    row: pd.Series,
) -> constellaration_update_types.RequirementMetrics:
    """Build a `RequirementMetrics` from `results.parquet`'s flattened columns."""
    values = [float(row[c]) for c in REQUIREMENT_METRICS_COLUMNS]
    return constellaration_update_types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=values[0],
        min_normalized_coil_to_plasma_distance=values[1],
        max_normalized_coil_curvature=values[2],
        max_normalized_field_error=values[3],
    )


def compute_mean_currents(
    train_df: pd.DataFrame,
) -> jt.Float[jt.Array, " n_unique_coils"]:
    """Population-mean currents over the training rows, decoding coilsets once."""
    currents_acc = None
    n = 0
    logger.info("Computing mean currents from %d training rows...", len(train_df))
    for json_str in train_df["json_desc_coilset"].dropna():
        c = constellaration_update_types.ConStellarationUpdateCoilset.model_validate(
            orjson.loads(json_str)
        )
        currents = jnp.asarray(c.currents)
        if currents_acc is None:
            currents_acc = jnp.zeros_like(currents)
        currents_acc += currents
        n += 1
    if currents_acc is None:
        return jnp.zeros(0, dtype=float)
    logger.info("Computed mean currents to be %s", currents_acc / max(n, 1))
    return currents_acc / max(n, 1)


def _decode_row(
    row: pd.Series,
) -> tuple[
    surface_rz_fourier.SurfaceRZFourier,
    constellaration_update_types.ConStellarationUpdateCoilset,
    constellaration_update_types.RequirementMetrics,
]:
    boundary = surface_rz_fourier.SurfaceRZFourier.model_validate(
        orjson.loads(row["json_constellaration_boundary"])
    )
    coilset = constellaration_update_types.ConStellarationUpdateCoilset.model_validate(
        orjson.loads(row["json_desc_coilset"])
    )
    return boundary, coilset, row_to_requirement_metrics(row)


def _stack_batch(
    items: list[
        tuple[
            surface_rz_fourier.SurfaceRZFourier,
            constellaration_update_types.ConStellarationUpdateCoilset,
            constellaration_update_types.RequirementMetrics,
        ]
    ],
) -> types.Batch:
    """Stack a list of (boundary, coilset, requirement_metrics) triples.

    Coilsets are padded symmetrically to the per-batch maximum fourier order
    so the leading-dim stack is well-defined for variable-N datasets. The
    batch's `fourier_order_mask` carries 1.0 at each sample's real
    coefficient indices and 0.0 at zero-padded indices, for use in the loss.
    """
    boundaries = [b for b, _, _ in items]
    coilsets = [c for _, c, _ in items]
    requirement_metrics = [r for _, _, r in items]
    batch_max_fourier_order = max(int(c.fourier_order) for c in coilsets)
    padded_coilsets = [
        coilset_utils.pad_coilset_to_fourier_order(c, batch_max_fourier_order)
        for c in coilsets
    ]
    masks = jnp.stack(
        [
            coilset_utils.make_fourier_order_mask(
                int(c.fourier_order), batch_max_fourier_order
            )
            for c in coilsets
        ]
    )
    return types.Batch.model_construct(
        boundaries=pytree.tree_stack(boundaries),
        coilsets=pytree.tree_stack(padded_coilsets),
        requirement_metrics=pytree.tree_stack(requirement_metrics),
        fourier_order_mask=masks,
        batch_max_fourier_order=batch_max_fourier_order,
    )


def make_train_batches(
    train_df: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> Iterator[types.Batch]:
    """Yields infinite training batches in seeded shuffled order."""
    key = jax.random.PRNGKey(seed)
    json_df = train_df[[*JSON_COLUMNS, *REQUIREMENT_METRICS_COLUMNS]].reset_index(
        drop=True
    )
    n = len(json_df)
    if n < batch_size:
        raise ValueError(
            f"train set has {n} rows but batch_size={batch_size}; "
            "iterator would never yield"
        )
    indices = jnp.arange(n)
    while True:
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, indices)
        items: list[
            tuple[
                surface_rz_fourier.SurfaceRZFourier,
                constellaration_update_types.ConStellarationUpdateCoilset,
                constellaration_update_types.RequirementMetrics,
            ]
        ] = []
        for idx in indices:
            row = json_df.iloc[int(idx)]
            # Upstream _filtered() already called dropna(subset=JSON_COLUMNS).
            items.append(_decode_row(row))
            if len(items) == batch_size:
                yield _stack_batch(items)
                items = []


def make_eval_batch(eval_df: pd.DataFrame, eval_batch_size: int) -> types.Batch:
    """One fixed eval batch, deterministic across runs."""
    json_df = (
        eval_df[["desc_coilset_id", *JSON_COLUMNS, *REQUIREMENT_METRICS_COLUMNS]]
        .dropna(subset=list(JSON_COLUMNS))
        .sort_values("desc_coilset_id")
        .reset_index(drop=True)
    )
    take = min(eval_batch_size, len(json_df))
    items = [_decode_row(json_df.iloc[i]) for i in range(take)]
    if not items:
        raise RuntimeError("eval set has no usable rows after filtering")
    return _stack_batch(items)


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
    for i, (idx, json_str) in enumerate(series.items()):
        coilset = (
            constellaration_update_types.ConStellarationUpdateCoilset.model_validate(
                orjson.loads(json_str)
            )
        )
        # Sanity-check the first 25 samples to catch any deserialization issues early
        if i < 25:
            coilset.model_rebuild(raise_errors=True)
        max_order = max(max_order, int(coilset.fourier_order))
    return max_order


def _resolve_model_config(
    initial: types.CoilPredictorConfig | types.AttentionCoilPredictorConfig | None,
    model_type: Literal["mlp", "attention"],
    train_df: pd.DataFrame,
    expected_currents: jt.Float[jt.Array, " n_unique_coils"],
    requirement_metrics_means: jt.Float[jt.Array, " n_requirements"],
    requirement_metrics_stds: jt.Float[jt.Array, " n_requirements"],
) -> types.CoilPredictorConfig | types.AttentionCoilPredictorConfig:
    """Fills data-derived fields on the architecture config from the training set.

    Architecture knobs (``hidden_dim``, layer counts, spectral params) come
    from ``initial``; defaults applied if ``initial`` is ``None`` for the
    architecture selected by ``model_type``. Data-derived fields are taken
    from a single training row and cross-checked against a second row to
    surface dataset inconsistency early.
    """
    if initial is None:
        if model_type == "mlp":
            initial = types.CoilPredictorConfig(
                is_stellarator_symmetric=False,
                n_field_periods=0,
                n_poloidal_modes=0,
                n_toroidal_modes=0,
                n_unique_coils=0,
                n_modes_coils_max=0,
                expected_currents=[],
                hidden_dim=512,
                num_layers=4,
                requirement_metrics_means=[],
                requirement_metrics_stds=[],
            )
        else:
            initial = types.AttentionCoilPredictorConfig(
                is_stellarator_symmetric=False,
                n_field_periods=0,
                n_poloidal_modes=0,
                n_toroidal_modes=0,
                n_unique_coils=0,
                n_modes_coils_max=0,
                expected_currents=[],
                requirement_metrics_means=[],
                requirement_metrics_stds=[],
            )
    rows = train_df.dropna(subset=list(JSON_COLUMNS)).head(2)
    if len(rows) < 1:
        raise RuntimeError("training set is empty after filtering")
    boundary_a = surface_rz_fourier.SurfaceRZFourier.model_validate(
        orjson.loads(rows.iloc[0]["json_constellaration_boundary"])
    )
    coilset_a = (
        constellaration_update_types.ConStellarationUpdateCoilset.model_validate(
            orjson.loads(rows.iloc[0]["json_desc_coilset"])
        )
    )
    if len(rows) >= 2:
        boundary_b = surface_rz_fourier.SurfaceRZFourier.model_validate(
            orjson.loads(rows.iloc[1]["json_constellaration_boundary"])
        )
        coilset_b = (
            constellaration_update_types.ConStellarationUpdateCoilset.model_validate(
                orjson.loads(rows.iloc[1]["json_desc_coilset"])
            )
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

    return initial.model_copy(
        update={
            "is_stellarator_symmetric": bool(boundary_a.is_stellarator_symmetric),
            "n_field_periods": int(boundary_a.n_field_periods),
            "n_poloidal_modes": int(boundary_a.n_poloidal_modes),
            "n_toroidal_modes": int(boundary_a.n_toroidal_modes),
            "n_unique_coils": int(coilset_a.coil_x_n.shape[0]),
            "n_modes_coils_max": 2 * infer_max_fourier_order(train_df) + 1,
            "expected_currents": [float(x) for x in expected_currents],
            "requirement_metrics_means": [float(x) for x in requirement_metrics_means],
            "requirement_metrics_stds": [float(x) for x in requirement_metrics_stds],
        }
    )


def train(
    train_config: types.TrainConfig,
) -> types.CoilPredictorCheckpoint | types.AttentionCoilPredictorCheckpoint:
    """Run training end-to-end and return the final-state model checkpoint.

    The concrete model architecture is selected by `train_config.model_type`.
    The returned checkpoint type matches the chosen architecture.
    """
    logger.info("Loading data frames...")
    train_df = load_dataframes(
        "train",
        relative_min_coil_to_plasma_distance_error_threshold=train_config.relative_min_coil_to_plasma_distance_error_threshold,
    )
    eval_df = load_dataframes(
        "eval",
        relative_min_coil_to_plasma_distance_error_threshold=train_config.relative_min_coil_to_plasma_distance_error_threshold,
    )
    logger.info("Train rows: %d, eval rows: %d", len(train_df), len(eval_df))

    logger.info("Computing train-set mean currents (one streaming pass)...")
    mean_currents = compute_mean_currents(train_df)

    logger.info("Computing train-set requirement-metrics stats...")
    stacked_requirements = jnp.asarray(
        train_df[list(REQUIREMENT_METRICS_COLUMNS)].dropna().to_numpy(dtype=float)
    )
    requirement_metrics_means = stacked_requirements.mean(axis=0)
    requirement_metrics_stds = jnp.maximum(stacked_requirements.std(axis=0), 1e-8)

    logger.info("Resolving architecture config from data...")
    model_config = _resolve_model_config(
        train_config.model_config_overrides,
        train_config.model_type,
        train_df,
        mean_currents,
        requirement_metrics_means,
        requirement_metrics_stds,
    )
    resolved_train_config = train_config.model_copy(
        update={"model_config_overrides": model_config}
    )

    eval_batch = make_eval_batch(eval_df, train_config.eval_batch_size)
    train_iter = prefetched(
        make_train_batches(train_df, train_config.batch_size, train_config.seed)
    )

    model: model_definition.CoilPredictor | model_definition.AttentionCoilPredictor
    if train_config.model_type == "mlp":
        assert isinstance(model_config, types.CoilPredictorConfig)
        model = model_definition.CoilPredictor(
            model_config, rngs=nnx.Rngs(train_config.seed)
        )
    else:
        assert isinstance(model_config, types.AttentionCoilPredictorConfig)
        model = model_definition.AttentionCoilPredictor(
            model_config, rngs=nnx.Rngs(train_config.seed)
        )
    # Split params away from RngState/BatchStat/etc. so that `value_and_grad`
    # only sees float Param leaves — RngState leaves are uint32 PRNG keys and
    # would otherwise trigger `grad requires real- or complex-valued inputs`.
    graphdef, params, other_state = nnx.split(model, nnx.Param, ...)

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
    opt_state = tx.init(params)  # pyright: ignore[reportArgumentType]

    logger.info(
        "Starting training: train_config=%s jax_version=%s jax_devices=%s"
        " param_count=%d n_train_rows=%d n_eval_rows=%d",
        resolved_train_config.model_dump(mode="json"),
        jax.__version__,
        [str(d) for d in jax.devices()],
        int(sum(int(jnp.size(x)) for x in jax.tree_util.tree_leaves(params))),
        int(len(train_df)),
        int(len(eval_df)),
    )

    def _loss_for_grad(params, other_state, batch):
        model = nnx.merge(graphdef, params, other_state)
        loss, aux = _loss_fn(model, batch)
        _, _, new_other_state = nnx.split(model, nnx.Param, ...)
        return loss, (aux, new_other_state)

    @jax.jit
    def update_step(params, other_state, opt_state, batch: types.Batch):
        (total, (aux, new_other_state)), grads = jax.value_and_grad(
            _loss_for_grad, argnums=0, has_aux=True
        )(params, other_state, batch)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, new_other_state, opt_state, total, aux, grad_norm

    @jax.jit
    def eval_step(params, other_state, batch):
        model = nnx.merge(graphdef, params, other_state)
        return _loss_fn(model, batch)

    best_eval = float("inf")
    best_step = -1
    evals_without_improvement = 0
    loss_components = ("coil_x_n", "coil_y_n", "coil_z_n", "currents")

    logger.info("Starting training...")
    for step in range(train_config.steps):
        batch = next(train_iter)
        params, other_state, opt_state, train_loss, train_aux, grad_norm = update_step(
            params, other_state, opt_state, batch
        )
        log_payload = {
            "train/loss": float(train_loss),
            "train/lr": float(schedule(step)),  # pyright: ignore[reportArgumentType]
            "train/grad_norm": float(grad_norm),
        }
        for c in loss_components:
            log_payload[f"train/loss_{c}"] = float(train_aux[c])
        logger.info("step=%d %s", step, log_payload)

        if (step + 1) % train_config.eval_every == 0 or step == train_config.steps - 1:
            eval_loss, eval_aux = eval_step(params, other_state, eval_batch)
            eval_payload = {"eval/loss": float(eval_loss)}
            for c in loss_components:
                eval_payload[f"eval/loss_{c}"] = float(eval_aux[c])
            logger.info("eval step=%d %s", step, eval_payload)

            logger.info(
                "Step %6d  train=%.6f  eval=%.6f",
                step,
                float(train_loss),
                float(eval_loss),
            )
            if float(eval_loss) < best_eval:
                best_eval = float(eval_loss)
                best_step = step
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
                if (
                    train_config.early_stopping_patience is not None
                    and evals_without_improvement
                    >= train_config.early_stopping_patience
                ):
                    logger.info("Early stopping at step %d", step)
                    break

    final_model = nnx.merge(graphdef, params, other_state)
    checkpoint: types.CoilPredictorCheckpoint | types.AttentionCoilPredictorCheckpoint
    if train_config.model_type == "mlp":
        checkpoint = flax_nnx_checkpoint.to_checkpoint(
            final_model, model_config, types.CoilPredictorCheckpoint
        )
    else:
        checkpoint = flax_nnx_checkpoint.to_checkpoint(
            final_model, model_config, types.AttentionCoilPredictorCheckpoint
        )

    logger.info("Best eval loss %.6f at step %d", best_eval, best_step)
    return checkpoint


def _masked_mse_coef(
    p: jt.Float[NpOrJaxArray, "batch n_unique n_modes"],
    t: jt.Float[NpOrJaxArray, "batch n_unique n_modes"],
    mask: jt.Bool[NpOrJaxArray, "batch n_unique"],
    eps: float = 1e-8,
) -> jt.Float[NpOrJaxArray, " "]:
    mask_b = mask[:, None, :]  # (B, 1, 2N+1) broadcasts over n_unique
    n_real = jnp.maximum(mask_b.sum() * t.shape[1], 1.0)
    sum_real = (t * mask_b).sum()
    mean_t = sum_real / n_real
    var_t = (((t - mean_t) ** 2) * mask_b).sum() / n_real
    std = jnp.where(var_t > eps, jnp.sqrt(var_t), 1.0)
    sq_err = ((p - t) / std) ** 2 * mask_b
    return sq_err.sum() / n_real


def _component_mse_unmasked(
    p: jt.Float[NpOrJaxArray, " *dims"],
    t: jt.Float[NpOrJaxArray, " *dims"],
    eps: float = 1e-8,
) -> jt.Float[NpOrJaxArray, " "]:
    std = jnp.std(t)
    scale = jnp.where(std > eps, std, jnp.ones_like(std))
    return jnp.mean(((p - t) / scale) ** 2)


def _loss_fn(
    model: model_definition.CoilPredictor | model_definition.AttentionCoilPredictor,
    batch: types.Batch,
) -> tuple[jt.Float[NpOrJaxArray, " "], dict[str, jt.Float[NpOrJaxArray, " "]]]:
    """Per-component masked MSE; aggregate is the unweighted mean.

    The fourier-order mask suppresses contributions from zero-padded
    coefficient indices so small-`N` and large-`N` samples weigh equally per
    real coefficient. Per-component std is computed over real (mask=1)
    entries only to avoid bias from zero pad. Currents are unmasked.

    Returns ``(scalar_loss, aux)`` where ``aux[c]`` is the masked MSE for
    each coil component (``coil_x_n``, ``coil_y_n``, ``coil_z_n``,
    ``currents``).
    """
    pred = jax.vmap(
        lambda b, r: model(b, r, fourier_order=batch.batch_max_fourier_order)
    )(batch.boundaries, batch.requirement_metrics)
    target = batch.coilsets

    mask = batch.fourier_order_mask  # (B, 2*N+1)
    eps = 1e-8

    aux = {
        "coil_x_n": _masked_mse_coef(pred.coil_x_n, target.coil_x_n, mask, eps),
        "coil_y_n": _masked_mse_coef(pred.coil_y_n, target.coil_y_n, mask, eps),
        "coil_z_n": _masked_mse_coef(pred.coil_z_n, target.coil_z_n, mask, eps),
        "currents": _component_mse_unmasked(pred.currents, target.currents, eps),
    }
    total = (aux["coil_x_n"] + aux["coil_y_n"] + aux["coil_z_n"] + aux["currents"]) / 4
    return total, aux
