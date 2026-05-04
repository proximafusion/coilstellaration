"""Helpers for converting between DESC and Proxima coilset representations.

This is a separate module to isolate the expensive DESC import, which has side-effects
at import time.
"""

from typing import Any, cast

import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from desc.backend import tree_leaves
from desc.coils import CoilSet, FourierXYZCoil
from desc.equilibrium import Equilibrium
from desc.objectives._coils import _CoilObjective
from desc.objectives.objective_funs import _Objective, collect_docs

from constellaration_update import types as constellaration_update_types
from constellaration_update.metrics import metrics as metrics_utils


def pad_coilset_to_fourier_order(
    coilset: constellaration_update_types.ConStellarationUpdateCoilset,
    n_modes_coils_max: int,
) -> constellaration_update_types.ConStellarationUpdateCoilset:
    """Pad each coil's Fourier coefficients symmetrically around the constant term.

    DESC layout is `[sin(N), ..., sin(1), const, cos(1), ..., cos(N)]` with length
    `2N + 1`. Padding to `N_max` prepends `N_max - N` zeros (high-|n| sin) and
    appends `N_max - N` zeros (high-|n| cos), preserving the central `2N + 1`
    real coefficients exactly.

    Currents and scalar metadata are unchanged. Raises `ValueError` if
    `n_modes_coils_max` is smaller than the input fourier order.
    """
    fourier_order = coilset.fourier_order
    if n_modes_coils_max < fourier_order:
        raise ValueError(
            f"cannot shrink: requested n_modes_coils_max={n_modes_coils_max} < "
            f"input fourier_order={fourier_order}"
        )
    pad = int(n_modes_coils_max) - int(fourier_order)
    if pad == 0:
        return coilset
    pad_widths = ((0, 0), (pad, pad))
    return coilset.model_copy(
        update={
            "coil_x_n": jnp.pad(coilset.coil_x_n, pad_widths),
            "coil_y_n": jnp.pad(coilset.coil_y_n, pad_widths),
            "coil_z_n": jnp.pad(coilset.coil_z_n, pad_widths),
        }
    )


def make_fourier_order_mask(
    fourier_order: int,
    n_modes_coils_max: int,
) -> jt.Float[jt.Array, " n_modes"]:
    """1.0 at the central `2N + 1` real coefficient indices, 0.0 elsewhere.

    Shape is `(2 * n_modes_coils_max + 1,)`. Matches the symmetric padding
    layout of `pad_coilset_to_fourier_order`.
    """
    if fourier_order > n_modes_coils_max:
        raise ValueError(
            f"fourier_order={fourier_order} > n_modes_coils_max={n_modes_coils_max}"
        )
    n_modes = 2 * n_modes_coils_max + 1
    pad = n_modes_coils_max - fourier_order
    indices = jnp.arange(n_modes)
    return jnp.where((indices >= pad) & (indices < n_modes - pad), 1.0, 0.0)


def constellaration_update_from_desc(
    desc_coilset: CoilSet,
    n_field_periods: int,
    is_stellarator_symmetric: bool,
) -> constellaration_update_types.ConStellarationUpdateCoilset:
    """Convert a DESC CoilSet to a CoilsetDESC.

    DESC expands all symmetries, so total coils = n_unique * n_symmetrized.
    This extracts the first n_unique coils.

    Args:
        desc_coilset: DESC CoilSet with all symmetry-expanded coils.
        n_field_periods: Number of toroidal field periods.
        is_stellarator_symmetric: Whether the coilset has stellarator symmetry.
    """
    n_symmetrized = 2 * n_field_periods if is_stellarator_symmetric else n_field_periods
    n_unique = desc_coilset.num_coils // n_symmetrized

    x_n_list = []
    y_n_list = []
    z_n_list = []
    currents_list = []
    for i, desc_coil in enumerate(desc_coilset):
        if i >= n_unique:
            break
        x_n_list.append(np.asarray(desc_coil.X_n))
        y_n_list.append(np.asarray(desc_coil.Y_n))
        z_n_list.append(np.asarray(desc_coil.Z_n))
        currents_list.append(float(desc_coil.current))

    return constellaration_update_types.ConStellarationUpdateCoilset(
        coil_x_n=np.stack(x_n_list),
        coil_y_n=np.stack(y_n_list),
        coil_z_n=np.stack(z_n_list),
        currents=np.array(currents_list),
        n_field_periods=n_field_periods,
        is_stellarator_symmetric=is_stellarator_symmetric,
    )


def constellaration_update_to_desc(
    coilset_desc_data: constellaration_update_types.ConStellarationUpdateCoilset,
) -> CoilSet:
    """Convert a CoilsetDESC to a DESC CoilSet.

    Expands symmetries to produce the full set of coils that DESC expects.
    """
    desc_coils = []
    for i in range(coilset_desc_data.n_unique_coils):
        desc_coils.append(
            FourierXYZCoil(
                X_n=np.asarray(coilset_desc_data.coil_x_n[i]),
                Y_n=np.asarray(coilset_desc_data.coil_y_n[i]),
                Z_n=np.asarray(coilset_desc_data.coil_z_n[i]),
                current=float(coilset_desc_data.currents[i]),  # type: ignore[arg-type]
            )
        )

    return CoilSet(
        *desc_coils,
        NFP=int(coilset_desc_data.n_field_periods),
        sym=coilset_desc_data.is_stellarator_symmetric,
    )


def scale_coil_currents_to_B_0_equals_1_T(coilset: CoilSet, eq: Equilibrium) -> CoilSet:
    """Scale coil currents so the on-axis average magnetic field is 1 T."""
    B_old_avg = metrics_utils.compute_on_axis_average_magnetic_field(
        coilset=coilset, eq=eq, zeta_n=100
    )
    B_target_avg = 1.0
    k = B_target_avg / B_old_avg
    new_coilset = coilset.copy()
    new_coilset.current = jnp.multiply(jnp.asarray(new_coilset.current), k)
    return new_coilset


class CoilCurvatureResidual(_CoilObjective):
    """Coil curvature objective using a per-coil integrated exceedance penalty."""

    _coilset_tree: dict[str, Any]
    _static_attrs = _CoilObjective._static_attrs + ["_curvature_threshold"]

    __doc__ = (__doc__ or "").rstrip() + collect_docs(
        target_default="``0``.",
        bounds_default="``None``.",
        coil=True,
    )

    _scalar = False
    _units = "(m^-1)"
    _print_value_fmt = "Coil curvature: "
    _broadcast_input = "Coil"

    def __init__(
        self,
        coil,
        target: Any = None,
        bounds: Any = None,
        curvature_threshold: float | None = None,
        weight: float = 1.0,
        normalize: bool = True,
        normalize_target: bool = True,
        loss_function: str | None = None,
        deriv_mode: str = "auto",
        grid=None,
        name: str = "coil curvature",
        jac_chunk_size: int | None = None,
    ) -> None:
        assert bounds is None, "CoilCurvatureExcessIntegral does not support bounds."
        assert not normalize, "CoilCurvatureExcessIntegral requires normalize=False."
        if target is not None and target != 0.0:
            raise ValueError(
                "CoilCurvatureExcessIntegral only supports target=0.0 because the "
                "objective computes a residual directly."
            )
        if curvature_threshold is None or not np.isfinite(curvature_threshold):
            raise ValueError(
                "CoilCurvatureExcessIntegral requires a finite curvature_threshold."
            )
        self._curvature_threshold = curvature_threshold

        super().__init__(
            coil,
            ["curvature"],
            target=0.0,
            bounds=None,
            weight=cast(Any, weight),
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        super().build(use_jit=use_jit, verbose=verbose)

        self._constants["coil_quad_weights"] = self._constants["quad_weights"]
        self._constants["quad_weights"] = jnp.ones(self.dim_f)
        self._curvature_threshold = self._coilset_broadcast(self._curvature_threshold)

        if self._normalize:
            self._normalization = 1 / np.mean([scale["a"] for scale in self._scales])

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(
        self, params, constants=None
    ):  # pyright: ignore[reportIncompatibleMethodOverride]
        data = cast(
            Any,
            tree_leaves(
                cast(Any, super().compute(params, constants=constants)),
                is_leaf=lambda x: isinstance(x, dict),
            ),
        )
        quad_weights = cast(Any, constants or self._constants)["coil_quad_weights"]
        thresholds = jnp.asarray(self._curvature_threshold)
        residuals = []
        weight_start = 0
        for i, dat in enumerate(data):
            curvature = jnp.abs(dat["curvature"])
            weight_stop = weight_start + curvature.shape[0]
            coil_quad_weights = quad_weights[weight_start:weight_stop]
            weight_start = weight_stop
            coil_length = jnp.sum(coil_quad_weights)
            residuals.append(
                jnp.sum(
                    coil_quad_weights * jnp.maximum(curvature - thresholds[i], 0.0) ** 2
                )
                / coil_length
            )
        return jnp.asarray(residuals)[self._coilset_tree["coilset_mask"]]
