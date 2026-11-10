import logging
import pathlib

import jax.numpy as jnp
import jaxtyping as jt
from constellaration.geometry import surface_rz_fourier

from coilstellaration import paths, types
from coilstellaration.types import NpOrJaxArray

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


def major_radius(
    boundary: surface_rz_fourier.SurfaceRZFourier,
) -> types.ScalarFloat:
    """The `(m, n) = (0, 0)` coefficient of `r_cos`, the models' unit of length."""
    return jnp.asarray(boundary.r_cos[0, boundary.max_toroidal_mode])


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
    `r_sin`, `z_cos`, each in units of the `major_radius` and divided by their
    exponential spectral scaling.
    """
    scaled = calculate_exponential_spectral_scaling_surface(
        boundary, alpha=alpha, order=order, ess_min_value=ess_min_value
    )
    radius = major_radius(boundary)
    parts = [
        (boundary.r_cos / radius / scaled.r_cos).ravel(),
        (boundary.z_sin / radius / scaled.z_sin).ravel(),
    ]
    if boundary.r_sin is not None and scaled.r_sin is not None:
        parts.append((boundary.r_sin / radius / scaled.r_sin).ravel())
    if boundary.z_cos is not None and scaled.z_cos is not None:
        parts.append((boundary.z_cos / radius / scaled.z_cos).ravel())
    return jnp.concatenate(parts)


@types.runtime_check_array_sizes
def forward_coilset_to_flat(
    coilset: types.Coilset,
    *,
    major_radius: types.ScalarFloat,
    n_modes_coils_max: int,
    coil_currents_mean: float,
    coil_currents_std: float,
    alpha: float,
    ess_min_value: float,
) -> jt.Float[jt.Array, " n_flat_coilset"]:
    """Spectrally-scale a coilset and flatten to a 1D vector.

    Coefficients and currents are taken in units of `major_radius`: scaling a device
    by a factor scales the currents that produce the same field by that factor.
    """
    scaling = calculate_exponential_spectral_scaling_coilset(
        n_modes_coils_max, alpha=alpha, ess_min_value=ess_min_value
    )
    return jnp.concatenate(
        [
            (coilset.coil_x_n / major_radius / scaling).ravel(),
            (coilset.coil_y_n / major_radius / scaling).ravel(),
            (coilset.coil_z_n / major_radius / scaling).ravel(),
            (
                (coilset.currents / major_radius - coil_currents_mean)
                / coil_currents_std
            ).ravel(),
        ]
    )


@types.runtime_check_array_sizes
def backward_flat_to_coilset(
    flat: jt.Float[jt.Array, " n_flat_coilset"],
    *,
    major_radius: types.ScalarFloat,
    n_unique_coils: int,
    n_modes_coils_max: int,
    coil_currents_mean: float,
    coil_currents_std: float,
    n_field_periods: types.ScalarInt,
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
    coil_x_n = flat[:coeff_size].reshape(shape) * scaling * major_radius
    coil_y_n = flat[coeff_size : 2 * coeff_size].reshape(shape) * scaling * major_radius
    coil_z_n = (
        flat[2 * coeff_size : 3 * coeff_size].reshape(shape) * scaling * major_radius
    )
    currents = (
        flat[3 * coeff_size :] * coil_currents_std + coil_currents_mean
    ) * major_radius
    return types.Coilset.model_construct(
        coil_x_n=coil_x_n,
        coil_y_n=coil_y_n,
        coil_z_n=coil_z_n,
        currents=currents,
        n_field_periods=n_field_periods,
        is_stellarator_symmetric=is_stellarator_symmetric,
    )


def truncate_to_fourier_order(
    full: types.Coilset,
    *,
    n_modes_coils_max: int,
    fourier_order: int | None,
) -> types.Coilset:
    """Centrally slice a full-band coilset to the requested fourier order."""
    max_fourier_order = (n_modes_coils_max - 1) // 2
    if fourier_order is None or fourier_order == max_fourier_order:
        return full
    if fourier_order > max_fourier_order:
        raise ValueError(
            f"fourier_order={fourier_order} exceeds n_modes_coils_max="
            f"{n_modes_coils_max} (max supported fourier_order={max_fourier_order})"
        )
    width = 2 * fourier_order + 1
    pad = (n_modes_coils_max - width) // 2
    return full.model_copy(
        update={
            "coil_x_n": full.coil_x_n[:, pad : pad + width],
            "coil_y_n": full.coil_y_n[:, pad : pad + width],
            "coil_z_n": full.coil_z_n[:, pad : pad + width],
        }
    )


@types.runtime_check_array_sizes
def read_checkpoint_from_id(
    model_kind: types.AnyModelLiteral,
    checkpoint_id: str,
    model_data_path: pathlib.Path | None = None,
) -> types.AnyModelCheckpoint:
    """Read the checkpoint stored at `checkpoint_id` with its declared type."""

    if model_data_path is None:
        model_data_path = paths.model_path(checkpoint_id)

    match model_kind:
        case "mlp":
            ckpt = types.CoilPredictorCheckpoint
        case "mlp_ensemble":
            ckpt = types.MlpEnsembleCoilPredictorCheckpoint
        case "res_mlp":
            ckpt = types.ResMlpCoilPredictorCheckpoint
        case "res_mlp_ensemble":
            ckpt = types.ResMlpEnsembleCoilPredictorCheckpoint
        case "per_coil":
            ckpt = types.PerCoilCoilPredictorCheckpoint
        case "masked":
            ckpt = types.MaskedCoilPredictorCheckpoint
        case "per_coil_ensemble":
            ckpt = types.PerCoilEnsembleCoilPredictorCheckpoint
        case "masked_ensemble":
            ckpt = types.MaskedEnsembleCoilPredictorCheckpoint
        case _:
            raise ValueError(f"Unsupported model_kind: {model_kind}")
    ckpt = ckpt.model_validate_json(model_data_path.read_text())
    return ckpt


@types.runtime_check_array_sizes
def z_score_normalization(
    metrics: jt.Float[NpOrJaxArray, " *dims"],
    means: jt.Float[NpOrJaxArray, " *dims"],
    stds: jt.Float[NpOrJaxArray, " *dims"],
) -> jt.Float[jt.Array, " *dims"]:
    return jnp.asarray((metrics - means) / stds)
