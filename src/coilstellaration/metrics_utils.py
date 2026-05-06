import typing

import jax.numpy as jnp
import jaxtyping as jt
from desc.coils import CoilSet as DescCoilSet
from desc.coils import LinearGrid
from desc.equilibrium import Equilibrium as DescEquilibrium
from desc.geometry.surface import (
    FourierRZToroidalSurface as DescFourierRZToroidalSurface,
)
from desc.grid import QuadratureGrid
from desc.objectives import (
    CoilArclengthVariance,
    CoilCurrentLength,
    CoilIntegratedCurvature,
    CoilSetLinkingNumber,
    CoilSetMinDistance,
    LinkingCurrentConsistency,
    PlasmaCoilSetMinDistance,
    QuadraticFlux,
    ToroidalFlux,
)

from coilstellaration import types

METRICS_FIELD_ORDER: tuple[str, ...] = (
    "min_normalized_coil_to_coil_distance",
    "min_normalized_coil_to_plasma_distance",
    "max_normalized_coil_curvature",
    "max_normalized_field_error",
    "on_axis_average_magnetic_field",
    "toroidal_flux",
)
"""Canonical order of `MetricsTargets` fields.

Frozen — do not reorder.
"""

METRICS_LOG1P_MASK: tuple[bool, ...] = (
    False,
    False,
    False,
    False,
    True,
    True,
)
"""Per-feature mask aligned with `METRICS_FIELD_ORDER`: True means apply signed-log1p
before z-scoring."""


@types.runtime_check_array_sizes
def compute_surface_field_metrics(
    coilset: DescCoilSet,
    eq: DescEquilibrium,
    surf_eval_grid: LinearGrid,
    coil_eval_grid: LinearGrid,
) -> tuple[
    jt.Float[jt.Array, " {surf_eval_grid.N}+1 2*{surf_eval_grid.M}+1"],
    jt.Float[jt.Array, " {surf_eval_grid.N}+1 2*{surf_eval_grid.M}+1"],
]:
    """Compute normalized field error and local quadratic flux on the plasma surface.

    Shares a single Biot-Savart evaluation for both metrics.

    Returns:
        (normalized_field, local_quadratic_flux) where:
        - normalized_field = |Bn| / |B|
        - local_quadratic_flux = Bn^2 / |B|^2 * dS
    """
    Bn, surf_coords = coilset.compute_Bnormal(
        eq.surface,
        eval_grid=surf_eval_grid,
        source_grid=coil_eval_grid,
    )
    B_vec = coilset.compute_magnetic_field(surf_coords, source_grid=coil_eval_grid)
    B_mag = jnp.linalg.norm(B_vec, axis=1)

    surface_data = eq.surface.compute(["|e_theta x e_zeta|"], grid=surf_eval_grid)
    dS = jnp.asarray(surface_data["|e_theta x e_zeta|"])

    shape = (surf_eval_grid.num_theta, surf_eval_grid.num_zeta)
    normalized_field_error = (jnp.abs(Bn) / B_mag).reshape(shape)
    local_quadratic_flux = (Bn**2 / B_mag**2 * dS).reshape(shape)

    return normalized_field_error, local_quadratic_flux


@types.runtime_check_array_sizes
def compute_on_axis_average_magnetic_field(
    coilset: DescCoilSet, eq: DescEquilibrium, zeta_n: int
) -> types.ScalarFloat:
    axis_grid = LinearGrid(
        rho=jnp.array([0.0]),
        theta=jnp.array([0.0]),
        zeta=jnp.linspace(0, 2 * jnp.pi / eq.NFP, zeta_n, endpoint=False),
        NFP=eq.NFP,
        sym=eq.sym,
    )
    axis_coords = eq.compute("x", grid=axis_grid)["x"]
    B_vec = coilset.compute_magnetic_field(axis_coords, basis="xyz")
    B_mag = jnp.linalg.norm(B_vec, axis=-1)
    B_avg = jnp.mean(B_mag)
    return B_avg


@types.runtime_check_array_sizes
def compute_coil_lengths(
    coilset: DescCoilSet, coil_eval_grid: LinearGrid
) -> jt.Float[jt.Array, " {len(coilset)}"]:
    _data = coilset.compute(["length"], grid=coil_eval_grid)
    data = typing.cast(list[dict[str, jt.Array]], _data)
    return jnp.array([d["length"] for d in data])


@types.runtime_check_array_sizes
def compute_coil_torsions(
    coilset: DescCoilSet, coil_eval_grid: LinearGrid
) -> jt.Float[jt.Array, "{len(coilset)} 2*{coil_eval_grid.N}+1"]:
    _data = coilset.compute(["torsion"], grid=coil_eval_grid)
    data = typing.cast(list[dict[str, jt.Array]], _data)
    return jnp.array([d["torsion"] for d in data])


@types.runtime_check_array_sizes
def compute_coil_curvatures(
    coilset: DescCoilSet, coil_eval_grid: LinearGrid
) -> jt.Float[jt.Array, "{len(coilset)} 2*{coil_eval_grid.N}+1"]:
    _data = coilset.compute(["curvature"], grid=coil_eval_grid)
    data = typing.cast(list[dict[str, jt.Array]], _data)
    return jnp.array([d["curvature"] for d in data])


@types.runtime_check_array_sizes
def compute_coil_to_coil_min_distances(
    coilset: DescCoilSet,
    coil_eval_grid: LinearGrid,
) -> jt.Float[jt.Array, " {coilset.num_coils}"]:
    obj = CoilSetMinDistance(coil=coilset, grid=coil_eval_grid)
    obj.build(verbose=0)
    return jnp.asarray(obj.compute(coilset.params_dict))


@types.runtime_check_array_sizes
def compute_coil_linking_numbers(
    coilset: DescCoilSet,
    coil_eval_grid: LinearGrid,
) -> jt.Float[jt.Array, " {coilset.num_coils}"]:
    obj = CoilSetLinkingNumber(coil=coilset, grid=coil_eval_grid)
    obj.build(verbose=0)
    return jnp.asarray(obj.compute(coilset.params_dict))


@types.runtime_check_array_sizes
def compute_coilset_arclength_variance(
    coilset: DescCoilSet,
    coil_eval_grid: LinearGrid,
) -> jt.Float[jt.Array, " {len(coilset)}"]:
    obj = CoilArclengthVariance(coilset, grid=coil_eval_grid)
    obj.build(verbose=0)
    return obj.compute(coilset.params_dict)


@types.runtime_check_array_sizes
def compute_coilset_quadratic_flux(
    coilset: DescCoilSet,
    eq: DescEquilibrium,
    surf_eval_grid: LinearGrid,
    coil_eval_grid: LinearGrid,
) -> jt.Float[jt.Array, " {surf_eval_grid.N}+1 2*{surf_eval_grid.M}+1"]:
    obj = QuadraticFlux(
        eq=eq,
        field=coilset,
        eval_grid=surf_eval_grid,
        source_grid=coil_eval_grid,
        field_grid=coil_eval_grid,
        vacuum=True,
    )
    obj.build(verbose=0)
    flat = obj.compute(coilset.params_dict)
    return flat.reshape(surf_eval_grid.num_theta, surf_eval_grid.num_zeta)


@types.runtime_check_array_sizes
def compute_coilset_toroidal_flux(
    coilset: DescCoilSet,
    eq: DescEquilibrium,
    coil_eval_grid: LinearGrid,
    surf_eval_m: int = 80,
) -> types.ScalarFloat:
    toroidal_flux_eval_grid = LinearGrid(
        L=surf_eval_m, M=surf_eval_m, zeta=jnp.array(0.0), NFP=eq.NFP
    )

    obj = ToroidalFlux(
        eq=eq,
        field=coilset,
        field_grid=coil_eval_grid,
        eval_grid=toroidal_flux_eval_grid,
        eq_fixed=True,
    )
    obj.build(verbose=0)
    return obj.compute(coilset.params_dict)


@types.runtime_check_array_sizes
def compute_coilset_linking_current_consistency(
    coilset: DescCoilSet,
    eq: DescEquilibrium,
    surf_eval_grid: LinearGrid,
) -> types.ScalarFloat:
    obj = LinkingCurrentConsistency(
        coil=coilset, eq=eq, eq_fixed=True, grid=surf_eval_grid
    )
    obj.build(verbose=0)
    return obj.compute(coilset.params_dict)


@types.runtime_check_array_sizes
def compute_coil_to_plasma_min_distances(
    coilset: DescCoilSet,
    eq: DescEquilibrium,
    coil_eval_grid: LinearGrid,
    surf_eval_grid: LinearGrid,
) -> jt.Float[jt.Array, " {coilset.num_coils}"]:
    obj = PlasmaCoilSetMinDistance(
        eq=eq,
        coil=coilset,
        coil_grid=coil_eval_grid,
        plasma_grid=surf_eval_grid,
        eq_fixed=True,
    )
    obj.build(verbose=0)
    return jnp.asarray(obj.compute(coilset.params_dict))


@types.runtime_check_array_sizes
def compute_coil_current_lengths(
    coilset: DescCoilSet,
    coil_eval_grid: LinearGrid,
) -> jt.Float[jt.Array, " {len(coilset)}"]:
    """Compute current * length for each coil (A*m).

    Proxy for HTS cost.
    """
    obj = CoilCurrentLength(coil=coilset, grid=coil_eval_grid)
    obj.build(verbose=0)
    return obj.compute(coilset.params_dict)


@types.runtime_check_array_sizes
def compute_coil_integrated_curvatures(
    coilset: DescCoilSet,
    coil_eval_grid: LinearGrid,
) -> jt.Float[jt.Array, " {len(coilset)}"]:
    """Compute integrated curvature for each coil.

    Equals 2*pi for convex coils.
    """
    obj = CoilIntegratedCurvature(coil=coilset, grid=coil_eval_grid)
    obj.build(verbose=0)
    return obj.compute(coilset.params_dict)


@types.runtime_check_array_sizes
def evaluate_coilset_metrics_from_boundary(
    boundary: DescFourierRZToroidalSurface,
    coilset: DescCoilSet,
    surf_eval_m: int = 80,
    surf_eval_n: int = 80,
    coil_eval_n: int = 200,
) -> types.Metrics:
    return evaluate_coilset_metrics(
        eq=DescEquilibrium(surface=boundary),
        coilset=coilset,
        surf_eval_m=surf_eval_m,
        surf_eval_n=surf_eval_n,
        coil_eval_n=coil_eval_n,
    )


@types.runtime_check_array_sizes
def evaluate_coilset_metrics(
    eq: DescEquilibrium,
    coilset: DescCoilSet,
    surf_eval_m: int = 80,
    surf_eval_n: int = 80,
    coil_eval_n: int = 200,
) -> types.Metrics:
    surf_eval_grid = LinearGrid(M=surf_eval_m, N=surf_eval_n, NFP=eq.NFP, sym=eq.sym)
    coil_eval_grid = LinearGrid(N=coil_eval_n)

    B_avg = compute_on_axis_average_magnetic_field(coilset, eq, zeta_n=coil_eval_n)

    coil_currents = jnp.asarray(coilset.current)
    coil_lengths = compute_coil_lengths(coilset, coil_eval_grid)
    coil_to_coil_min_distances = compute_coil_to_coil_min_distances(
        coilset, coil_eval_grid
    )
    coil_to_plasma_min_distances = compute_coil_to_plasma_min_distances(
        coilset, eq, coil_eval_grid, surf_eval_grid
    )
    coil_curvatures = compute_coil_curvatures(coilset, coil_eval_grid)
    coil_current_lengths = compute_coil_current_lengths(coilset, coil_eval_grid)
    coil_integrated_curvatures = compute_coil_integrated_curvatures(
        coilset, coil_eval_grid
    )
    coil_linking_numbers = compute_coil_linking_numbers(coilset, coil_eval_grid)
    coil_torsions = compute_coil_torsions(coilset, coil_eval_grid)
    coil_arclength_variances = compute_coilset_arclength_variance(
        coilset, coil_eval_grid
    )
    quadratic_flux = compute_coilset_quadratic_flux(
        coilset, eq, surf_eval_grid, coil_eval_grid
    )
    toroidal_flux = compute_coilset_toroidal_flux(
        coilset, eq, coil_eval_grid, surf_eval_m=surf_eval_m
    )
    linking_current_consistency = compute_coilset_linking_current_consistency(
        coilset, eq, surf_eval_grid
    )
    normalized_field_error, local_quadratic_flux = compute_surface_field_metrics(
        coilset, eq, surf_eval_grid, coil_eval_grid
    )
    minor_radius = eq.compute("a", grid=QuadratureGrid(L=16, M=16, N=16, NFP=eq.NFP))[
        "a"
    ]

    n_symmetrized = 2 * eq.NFP if eq.sym else eq.NFP
    n_coils_per_half_period = coilset.num_coils // n_symmetrized

    return types.Metrics(
        surf_eval_coords=surf_eval_grid.nodes[:, [1, 2]]
        .reshape(surf_eval_grid.num_theta, surf_eval_grid.num_zeta, 2)
        .transpose(2, 0, 1),
        coil_eval_params=coil_eval_grid.nodes[:, 2],
        coil_currents=coil_currents,
        n_coils_per_half_period=n_coils_per_half_period,
        on_axis_average_magnetic_field=B_avg,
        normalized_field_error=normalized_field_error,
        coil_linking_numbers=coil_linking_numbers,
        coil_to_coil_min_distances=coil_to_coil_min_distances,
        coil_to_plasma_min_distances=coil_to_plasma_min_distances,
        coil_lengths=coil_lengths,
        coil_curvatures=coil_curvatures,
        coil_current_lengths=coil_current_lengths,
        coil_integrated_curvatures=coil_integrated_curvatures,
        coil_torsions=coil_torsions,
        coil_arclength_variances=coil_arclength_variances,
        quadratic_flux=quadratic_flux,
        local_quadratic_flux=local_quadratic_flux,
        toroidal_flux=toroidal_flux,
        linking_current_consistency=linking_current_consistency,
        minor_radius=minor_radius,
    )
