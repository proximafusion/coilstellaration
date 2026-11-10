"""Generates `fixed_shape_golden.npz`, the reference outputs of the fixed-shape models.

The golden file pins the predictions of a bundled checkpoint and the training loss of a
seeded tiny `CoilPredictor`, so that refactors can prove they leave the fixed-shape
path bit-identical.

Regenerate (only when a change of the fixed-shape numerics is intended) with:

    python -m coilstellaration.testing_data.make_fixed_shape_golden
"""  # noqa: E501

import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from coilstellaration import types
from coilstellaration.machine_learning import (
    model_definition,
    train,
    utils,
)
from constellaration.geometry import surface_rz_fourier
from flax import nnx

CHECKPOINT_ID = "DHmCARHZ4ay9rsMDPdjVCu3"
GOLDEN_FILENAME = "fixed_shape_golden.npz"

_N_FIELD_PERIODS = 3
_N_UNIQUE_COILS = 5
_N_MODES_COILS_MAX = 7
_BATCH_SIZE = 4


def compute_golden() -> dict[str, np.ndarray]:
    """Computes the golden arrays with the current code."""
    return {**_checkpoint_predictions(), "loss": _tiny_model_loss()}


def _synthetic_boundary(seed_offset: int) -> surface_rz_fourier.SurfaceRZFourier:
    r_cos = 0.1 * jax.random.normal(jax.random.PRNGKey(seed_offset), (5, 9))
    z_sin = 0.1 * jax.random.normal(jax.random.PRNGKey(seed_offset + 1), (5, 9))
    # Stellarator symmetry requires these m=0 entries to vanish.
    r_cos = r_cos.at[0, :4].set(0.0).at[0, 4].set(1.0)
    z_sin = z_sin.at[0, :5].set(0.0)
    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.asarray(r_cos),
        z_sin=np.asarray(z_sin),
        n_field_periods=_N_FIELD_PERIODS,
        is_stellarator_symmetric=True,
    )


def _checkpoint_predictions() -> dict[str, np.ndarray]:
    checkpoint = utils.read_checkpoint_from_id("mlp", CHECKPOINT_ID)
    model = model_definition.read_model_from_checkpoint(checkpoint)
    requirements = types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=0.4,
        min_normalized_coil_to_plasma_distance=1.2,
        max_normalized_coil_curvature=3.8,
        max_normalized_field_error=0.01,
        n_coils_per_half_period=5,
    )
    coilset = model(_synthetic_boundary(0), requirements)
    return {
        "coil_x_n": np.asarray(coilset.coil_x_n),
        "coil_y_n": np.asarray(coilset.coil_y_n),
        "coil_z_n": np.asarray(coilset.coil_z_n),
        "currents": np.asarray(coilset.currents),
    }


def _tiny_model_loss() -> np.ndarray:
    config = types.CoilPredictorConfig(
        is_stellarator_symmetric=True,
        n_field_periods=_N_FIELD_PERIODS,
        n_poloidal_modes=5,
        n_toroidal_modes=9,
        n_unique_coils=_N_UNIQUE_COILS,
        n_modes_coils_max=_N_MODES_COILS_MAX,
        coil_currents_mean=0.0,
        coil_currents_std=1e6,
        hidden_dim=16,
        num_layers=2,
        requirement_metrics_means=[0.4, 1.2, 3.8, 0.01],
        requirement_metrics_stds=[0.1, 0.2, 0.5, 0.01],
    )
    model = model_definition.CoilPredictor(config, rngs=nnx.Rngs(0))

    boundaries = [_synthetic_boundary(10 + 2 * i) for i in range(_BATCH_SIZE)]
    key = jax.random.PRNGKey(100)
    target_shape = (_BATCH_SIZE, _N_UNIQUE_COILS, _N_MODES_COILS_MAX)
    kx, ky, kz, kc = jax.random.split(key, 4)
    coilsets = types.Coilset.model_construct(
        coil_x_n=np.asarray(jax.random.normal(kx, target_shape)),
        coil_y_n=np.asarray(jax.random.normal(ky, target_shape)),
        coil_z_n=np.asarray(jax.random.normal(kz, target_shape)),
        currents=np.asarray(
            1e6 * jax.random.normal(kc, (_BATCH_SIZE, _N_UNIQUE_COILS))
        ),
        n_field_periods=_N_FIELD_PERIODS,
        is_stellarator_symmetric=True,
    )
    batch = types.Batch.model_construct(
        boundaries=train.tree_stack(boundaries),
        coilsets=coilsets,
        requirement_metrics=types.RequirementMetrics.model_construct(
            min_normalized_coil_to_coil_distance=jnp.full(_BATCH_SIZE, 0.4),
            min_normalized_coil_to_plasma_distance=jnp.full(_BATCH_SIZE, 1.2),
            max_normalized_coil_curvature=jnp.full(_BATCH_SIZE, 3.8),
            max_normalized_field_error=jnp.full(_BATCH_SIZE, 0.01),
            n_coils_per_half_period=jnp.full(_BATCH_SIZE, 5),
        ),
        fourier_order_mask=np.ones((_BATCH_SIZE, _N_MODES_COILS_MAX)),
        coil_mask=np.ones((_BATCH_SIZE, _N_UNIQUE_COILS)),
        batch_max_fourier_order=(_N_MODES_COILS_MAX - 1) // 2,
    )
    loss, _ = train._loss_fn(model, batch)
    return np.asarray(loss)


if __name__ == "__main__":
    output = pathlib.Path(__file__).resolve().parent / GOLDEN_FILENAME
    arrays: dict[str, Any] = dict(compute_golden())
    np.savez(output, **arrays)
    print(f"Wrote {output}")
