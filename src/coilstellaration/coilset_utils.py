import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from desc.coils import CoilSet, FourierXYZCoil
from desc.equilibrium import Equilibrium

from coilstellaration import metrics_utils, types


def pad_coilset_to_fourier_order(
    coilset: types.Coilset,
    n_modes_coils_max: int,
) -> types.Coilset:
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


def coilstellaration_from_desc(
    desc_coilset: CoilSet,
    n_field_periods: int,
    is_stellarator_symmetric: bool,
) -> types.Coilset:
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

    return types.Coilset(
        coil_x_n=np.stack(x_n_list),
        coil_y_n=np.stack(y_n_list),
        coil_z_n=np.stack(z_n_list),
        currents=np.array(currents_list),
        n_field_periods=n_field_periods,
        is_stellarator_symmetric=is_stellarator_symmetric,
    )


def coilstellaration_to_desc(
    coilset_desc_data: types.Coilset,
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
