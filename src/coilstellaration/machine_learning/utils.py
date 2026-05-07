import logging
import pathlib

import jax.numpy as jnp
import jaxtyping as jt
from constellaration.geometry import surface_rz_fourier

from coilstellaration import paths, types

logger = logging.getLogger(__name__)


@types.runtime_check_array_sizes
def calculate_exponential_spectral_scaling_coilset(
    n_modes: int,
    alpha: float = 1.2,
    ess_min_value: float = 1e-7,
) -> jt.Float[jt.Array, " {n_modes}"]:
    """DESC-style exponential spectral scaling factors for coilset Fourier modes.

    Returns ``alpha ** (-|n|)`` (floored at ``ess_min_value``) for ``n`` ranging
    over the symmetric Fourier index, suitable for elementwise scaling of
    ``coil_[xyz]_n`` arrays.
    """
    fourier_order = (n_modes - 1) // 2
    abs_n = jnp.abs(jnp.arange(n_modes) - fourier_order).astype(jnp.float32)
    scale = jnp.maximum(alpha ** (-abs_n), ess_min_value)
    return scale


@types.runtime_check_array_sizes
def calculate_exponential_spectral_scaling_surface(
    reference_surface: surface_rz_fourier.SurfaceRZFourier,
    alpha: float = 1.2,
    order: float = jnp.inf,
    ess_min_value: float = 1e-7,
) -> surface_rz_fourier.SurfaceRZFourier:
    """DESC-style exponential spectral scaling for a ``SurfaceRZFourier``.

    Forward divides ``r_cos``/``z_sin`` (and ``r_sin``/``z_cos`` if present)
    by ``alpha ** (-||(m, n)||_p)`` (floored at ``ess_min_value``). Backward
    multiplies.
    """
    m = jnp.abs(reference_surface.poloidal_modes).astype(jnp.float32)
    n = jnp.abs(reference_surface.toroidal_modes).astype(jnp.float32)
    if order == jnp.inf:
        pnorm = jnp.maximum(m, n)
    else:
        pnorm = (m**order + n**order) ** (1.0 / order)
    scale = jnp.maximum(alpha ** (-pnorm), ess_min_value)

    updates: dict = dict(r_cos=scale, z_sin=scale)
    if reference_surface.r_sin is not None:
        updates["r_sin"] = scale
    if reference_surface.z_cos is not None:
        updates["z_cos"] = scale
    return reference_surface.model_copy(update=updates)


@types.runtime_check_array_sizes
def forward_boundary_to_flat(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    *,
    alpha: float,
    order: float,
    ess_min_value: float,
) -> jt.Float[jt.Array, " n_flat_boundary"]:
    """Spectrally-scale a boundary surface and flatten it to a 1D vector.

    Output is the concatenation of `r_cos`, `z_sin`, and (if non-symmetric)
    `r_sin`, `z_cos`, each divided by their exponential spectral scaling.
    """
    scaled = calculate_exponential_spectral_scaling_surface(
        boundary, alpha=alpha, order=order, ess_min_value=ess_min_value
    )
    parts = [
        (boundary.r_cos / scaled.r_cos).ravel(),
        (boundary.z_sin / scaled.z_sin).ravel(),
    ]
    if boundary.r_sin is not None and scaled.r_sin is not None:
        parts.append((boundary.r_sin / scaled.r_sin).ravel())
    if boundary.z_cos is not None and scaled.z_cos is not None:
        parts.append((boundary.z_cos / scaled.z_cos).ravel())
    return jnp.concatenate(parts)


@types.runtime_check_array_sizes
def forward_coilset_to_flat(
    coilset: types.Coilset,
    *,
    n_modes_coils_max: int,
    coil_currents_mean: float,
    coil_currents_std: float,
    alpha: float,
    ess_min_value: float,
) -> jt.Float[jt.Array, " n_flat_coilset"]:
    """Spectrally-scale a coilset and flatten to a 1D vector."""
    scaling = calculate_exponential_spectral_scaling_coilset(
        n_modes_coils_max, alpha=alpha, ess_min_value=ess_min_value
    )
    return jnp.concatenate(
        [
            (coilset.coil_x_n / scaling).ravel(),
            (coilset.coil_y_n / scaling).ravel(),
            (coilset.coil_z_n / scaling).ravel(),
            ((coilset.currents - coil_currents_mean) / coil_currents_std).ravel(),
        ]
    )


@types.runtime_check_array_sizes
def backward_flat_to_coilset(
    flat: jt.Float[jt.Array, " n_flat_coilset"],
    *,
    n_unique_coils: int,
    n_modes_coils_max: int,
    coil_currents_mean: float,
    coil_currents_std: float,
    n_field_periods: int,
    is_stellarator_symmetric: bool,
    alpha: float,
    ess_min_value: float,
) -> types.Coilset:
    """Inverse of `forward_coilset_to_flat`: flat scaled vector -> coilset."""
    coeff_size = n_unique_coils * n_modes_coils_max
    shape = (n_unique_coils, n_modes_coils_max)
    scaling = calculate_exponential_spectral_scaling_coilset(
        n_modes_coils_max, alpha=alpha, ess_min_value=ess_min_value
    )
    coil_x_n = flat[:coeff_size].reshape(shape) * scaling
    coil_y_n = flat[coeff_size : 2 * coeff_size].reshape(shape) * scaling
    coil_z_n = flat[2 * coeff_size : 3 * coeff_size].reshape(shape) * scaling
    currents = flat[3 * coeff_size :] * coil_currents_std + coil_currents_mean
    return types.Coilset.model_construct(
        coil_x_n=coil_x_n,
        coil_y_n=coil_y_n,
        coil_z_n=coil_z_n,
        currents=currents,
        n_field_periods=n_field_periods,
        is_stellarator_symmetric=is_stellarator_symmetric,
    )


@types.runtime_check_array_sizes
def read_checkpoint_from_id(
    model_type: types.AnyModelLiteral,
    checkpoint_id: str,
    model_data_path: pathlib.Path | None = None,
) -> types.AnyModelCheckpoint:
    """Read the checkpoint stored at `checkpoint_id` with its declared type."""

    if model_data_path is None:
        model_data_path = paths.model_path(checkpoint_id)

    match model_type:
        case "mlp":
            ckpt = types.CoilPredictorCheckpoint
        case "mlp_ensemble":
            ckpt = types.MlpEnsembleCoilPredictorCheckpoint
        case "res_mlp":
            ckpt = types.ResMlpCoilPredictorCheckpoint
        case "res_mlp_ensemble":
            ckpt = types.ResMlpEnsembleCoilPredictorCheckpoint
        case _:
            raise ValueError(f"Unsupported model_type: {model_type}")
    ckpt = ckpt.model_validate_json(model_data_path.read_text())
    return ckpt
