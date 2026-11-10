import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from desc.coils import CoilSet, FourierXYZCoil
from desc.equilibrium import Equilibrium

from coilstellaration import metrics_utils, types

runtime_check_array_sizes = types.runtime_check_array_sizes


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


@runtime_check_array_sizes
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


def pad_coilset_to_n_coils(
    coilset: types.Coilset,
    max_coils: int,
) -> types.Coilset:
    """Pad a coilset to `max_coils` unique coils with zero rows.

    Appends zero-filled coil rows to reach `max_coils` total. Currents are
    padded with zeros as well. Raises `ValueError` if `max_coils` is smaller
    than the input's `n_unique_coils`.
    """
    n_unique = coilset.n_unique_coils
    if max_coils < n_unique:
        raise ValueError(
            f"cannot shrink: requested max_coils={max_coils} < "
            f"n_unique_coils={n_unique}"
        )
    pad = int(max_coils) - int(n_unique)
    if pad == 0:
        return coilset
    pad_widths = ((0, pad), (0, 0))
    return coilset.model_copy(
        update={
            "coil_x_n": jnp.pad(coilset.coil_x_n, pad_widths),
            "coil_y_n": jnp.pad(coilset.coil_y_n, pad_widths),
            "coil_z_n": jnp.pad(coilset.coil_z_n, pad_widths),
            "currents": jnp.pad(coilset.currents, (0, pad)),
        }
    )


def trim_to_n_coils(
    coilset: types.Coilset,
    n_coils: int,
) -> types.Coilset:
    """Trim a coilset to the first `n_coils` unique coils.

    Takes only the first `n_coils` rows from all coefficient arrays and
    currents. Raises `ValueError` if `n_coils` is larger than the input's
    `n_unique_coils`.
    """
    n_unique = coilset.n_unique_coils
    if n_coils > n_unique:
        raise ValueError(
            f"cannot grow: requested n_coils={n_coils} > n_unique_coils={n_unique}"
        )
    if n_coils == n_unique:
        return coilset
    return coilset.model_copy(
        update={
            "coil_x_n": coilset.coil_x_n[:n_coils],
            "coil_y_n": coilset.coil_y_n[:n_coils],
            "coil_z_n": coilset.coil_z_n[:n_coils],
            "currents": coilset.currents[:n_coils],
        }
    )


@runtime_check_array_sizes
def make_coil_mask(
    n_coils: int,
    max_coils: int,
) -> jt.Float[jt.Array, " max_coils"]:
    """1.0 for the first `n_coils`, 0.0 for the remainder.

    Shape is `(max_coils,)`. Used to mask padded coilset losses.
    """
    if n_coils > max_coils:
        raise ValueError(f"n_coils={n_coils} > max_coils={max_coils}")
    indices = jnp.arange(max_coils)
    return jnp.where(indices < n_coils, 1.0, 0.0)


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
    coilset_desc_data: types.Coilset, check_intersection: bool = True
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
        check_intersection=check_intersection,
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
