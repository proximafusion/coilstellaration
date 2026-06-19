import logging

import jax.numpy as jnp
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.grid import LinearGrid

logger = logging.getLogger(__name__)

# Coil source points = 2 * FIELD_SOURCE_N + 1. 64 -> 129 points is well past
# the exponential-convergence knee for the smooth Fourier coils used here (the
# surface field matches the 401-point baseline to ~1e-15); raise it if coils
# ever carry very high mode numbers.
FIELD_SOURCE_N = 64

_PATCHED = False


def apply_patch():
    """Patch `MetricsEvaluator._make_surface_field_fn` in place (idempotent).

    Rebuilds the surface-field closure identically to the upstream version but
    sourcing the Biot-Savart integral from a coarse `FIELD_SOURCE_N` grid
    instead of the (much denser) geometric-metric grid.
    """
    global _PATCHED
    if _PATCHED:
        return
    from coilstellaration import metrics_utils_v2
    from coilstellaration.metrics_utils_v2 import MetricsEvaluator

    def _make_surface_field_fn(self):
        surf_keys = ["x", "n_rho", "|e_theta x e_zeta|"]
        surf_transforms = get_transforms(
            surf_keys, obj=self._template_surface, grid=self._surf_grid
        )
        surf_profiles = get_profiles(
            surf_keys, obj=self._template_surface, grid=self._surf_grid
        )
        surf_shape = (self._surf_grid.num_theta, self._surf_grid.num_zeta)
        template_surface = self._template_surface
        template_coilset = self._template_coilset
        # Decoupled, coarse Biot-Savart source grid (the only change vs upstream).
        field_source_grid = LinearGrid(N=FIELD_SOURCE_N)

        def _surface_field_fn(surface_params, coil_params):
            data = compute_fun(
                template_surface,
                surf_keys,
                params=surface_params,
                transforms=surf_transforms,
                profiles=surf_profiles,
            )
            B_vec = template_coilset.compute_magnetic_field(
                data["x"],
                basis="rpz",
                source_grid=field_source_grid,
                params=coil_params,
            )
            Bn = jnp.sum(B_vec * data["n_rho"], axis=-1)
            B_mag = jnp.linalg.norm(B_vec, axis=-1)
            dS = data["|e_theta x e_zeta|"]
            normalized_field_error = (jnp.abs(Bn) / B_mag).reshape(surf_shape)
            local_quadratic_flux = (Bn**2 / B_mag**2 * dS).reshape(surf_shape)
            quadratic_flux = (Bn * jnp.sqrt(dS)).reshape(surf_shape)
            return normalized_field_error, local_quadratic_flux, quadratic_flux

        return _surface_field_fn

    MetricsEvaluator._make_surface_field_fn = _make_surface_field_fn
    # Drop any evaluators built with the original (dense-source) field function.
    metrics_utils_v2.clear_evaluator_cache()
    _PATCHED = True
    logger.info(
        "Patched surface-field Biot-Savart to coarse source grid (N=%d, %d points)",
        FIELD_SOURCE_N, 2 * FIELD_SOURCE_N + 1,
    )
