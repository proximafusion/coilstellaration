import jax.numpy as jnp
import numpy as np
import pytest
from coilstellaration import (
    metrics_utils,
    metrics_utils_v2,
    types,
)
from desc.coils import initialize_modular_coils
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import LinearGrid


@pytest.fixture(params=["precise_QA", "W7-X"], scope="module")
def eq(request) -> Equilibrium:
    eq = get(request.param)
    assert isinstance(eq, Equilibrium)
    return eq


@pytest.fixture(params=[0], scope="module")
def seed(request) -> int:
    return int(request.param)


@pytest.fixture(params=[1e5], scope="module")
def current(request) -> int:
    return int(request.param)


@pytest.fixture(params=[3], scope="module")
def num_coils(request) -> int:
    return int(request.param)


@pytest.fixture(scope="module")
def R(eq: Equilibrium) -> float:
    """Major radius of the equilibrium."""
    return float(eq.compute("R0")["R0"])


@pytest.fixture(scope="module", params=[2.0])
def r_over_a(request) -> float:
    """Minor radius of the modular coils in units of the plasma minor radius."""
    return float(request.param)


@pytest.fixture(scope="module")
def a(eq: Equilibrium, r_over_a: float) -> float:
    """Minor radius of the modular coils."""
    return r_over_a * float(eq.compute("a")["a"])


@pytest.fixture(scope="module")
def coilset(seed: int, eq: Equilibrium, current: int, num_coils: int, r_over_a: float):
    """Non-planar, stellarator-symmetric coils, so each coil's distances differ within a
    field period while mirror partners stay equal."""
    rng = np.random.default_rng(seed)
    coilset = initialize_modular_coils(
        eq, num_coils=num_coils, r_over_a=r_over_a
    ).to_FourierXYZ(N=4)
    for i, coil in enumerate(coilset.coils):
        coil.change_resolution(N=4)
        coil.current = current
        coil.X_n = coil.X_n + 0.05 * rng.standard_normal(np.size(coil.X_n))
        coil.Y_n = coil.Y_n + 0.05 * rng.standard_normal(np.size(coil.Y_n))
        coil.Z_n = coil.Z_n + 0.05 * rng.standard_normal(np.size(coil.Z_n))
        coilset.coils[i] = coil
    return coilset


@pytest.fixture(params=[16], scope="module")
def surf_eval_m(request) -> int:
    return request.param


@pytest.fixture(params=[16], scope="module")
def surf_eval_n(request) -> int:
    return request.param


@pytest.fixture(params=[200], scope="module")
def coil_eval_n(request) -> int:
    return request.param


@pytest.fixture(scope="module")
def metrics_v1(eq, coilset, surf_eval_m, surf_eval_n, coil_eval_n):
    return metrics_utils.evaluate_coilset_metrics(
        coilset=coilset,
        eq=eq,
        surf_eval_m=surf_eval_m,
        surf_eval_n=surf_eval_n,
        coil_eval_n=coil_eval_n,
    )


@pytest.fixture(scope="module")
def metrics_v2(eq, coilset, surf_eval_m, surf_eval_n, coil_eval_n):
    return metrics_utils_v2.evaluate_coilset_metrics(
        coilset=coilset,
        eq=eq,
        surf_eval_m=surf_eval_m,
        surf_eval_n=surf_eval_n,
        coil_eval_n=coil_eval_n,
    )


@pytest.fixture(scope="module", params=["v1", "v2"])
def metrics_v1_or_v2(request, metrics_v1, metrics_v2):
    match request.param:
        case "v1":
            return metrics_v1
        case "v2":
            return metrics_v2
        case _:
            raise ValueError(f"{request.param} is not supported.")


def test_compute_surface_field_metrics_shapes(eq, coilset):
    surf_grid = LinearGrid(M=4, N=4, NFP=eq.NFP, sym=eq.sym)
    coil_grid = LinearGrid(N=8)

    normalized_field, local_qf = metrics_utils.compute_surface_field_metrics(
        coilset, eq, surf_grid, coil_grid
    )

    expected_shape = (surf_grid.num_theta, surf_grid.num_zeta)
    assert normalized_field.shape == expected_shape
    assert local_qf.shape == expected_shape
    np.testing.assert_array_less(-1e-15, normalized_field)
    np.testing.assert_array_less(-1e-15, local_qf)


def test_compute_surface_field_metrics_consistency(eq, coilset):
    """Verify local_qf = normalized_field^2 * dS."""
    surf_grid = LinearGrid(M=4, N=4, NFP=eq.NFP, sym=eq.sym)
    coil_grid = LinearGrid(N=8)

    normalized_field, local_qf = metrics_utils.compute_surface_field_metrics(
        coilset, eq, surf_grid, coil_grid
    )

    surface_data = eq.surface.compute(["|e_theta x e_zeta|"], grid=surf_grid)
    dS = jnp.asarray(surface_data["|e_theta x e_zeta|"]).reshape(
        surf_grid.num_theta, surf_grid.num_zeta
    )

    expected_local_qf = normalized_field**2 * dS
    np.testing.assert_allclose(local_qf, expected_local_qf, rtol=1e-6)


@pytest.mark.parametrize("eq", ["precise_QA"], indirect=True)
def test_compute_on_axis_average_magnetic_field(eq, coilset, current, a, R):
    zeta_n = 64

    B_avg = metrics_utils.compute_on_axis_average_magnetic_field(
        coilset, eq, zeta_n=zeta_n
    )

    # Analytical estimate: each coil is a circle of radius a centered on the
    # axis at major radius R.  In the magnetic-dipole limit (a << distance to
    # axis) the field per coil is mu_0*m/(4*pi*d^3) with m = I*pi*a^2.  The RMS
    # distance from the axis to any coil center averaged over toroidal angle
    # is sqrt(2)*R.
    mu_0 = 4 * np.pi * 1e-7
    total_coils_current = coilset.num_coils * coilset.NFP * current
    n_coils = len(coilset)
    d_avg = np.sqrt(2) * R
    expected_B = n_coils * mu_0 * total_coils_current * a**2 / (4 * d_avg**3)
    np.testing.assert_allclose(float(B_avg), expected_B, rtol=0.5)


@pytest.mark.parametrize(
    "metrics_key",
    [*types.Metrics.model_fields, *types.Metrics.model_computed_fields],
)
def test_equivalence(metrics_v1, metrics_v2, metrics_key):
    atol = 1e-12

    match metrics_key:
        case "quadratic_flux":
            # v1 evaluates `quadratic_flux` on `eq` at rho=1, v2 on `eq.surface`. The
            # two bases place the surface nodes ~1e-15 m apart, which the steep field of
            # high-current coils amplifies to a few 1e-12 in Bn*sqrt(dS).
            atol = 1e-11

    np.testing.assert_allclose(
        getattr(metrics_v1, metrics_key),
        getattr(metrics_v2, metrics_key),
        rtol=1e-12,
        atol=atol,
        err_msg=f"Mismatch in metric '{metrics_key}'",
    )


@pytest.mark.parametrize("num_pairs", [(512, 4096), (1,), (1, 1_000_000)])
def test_pruned_min_distances_are_bit_identical_to_brute_force(
    eq, coilset, num_pairs, monkeypatch, surf_eval_m, surf_eval_n, coil_eval_n
):
    def _distances(pairs):
        monkeypatch.setattr(metrics_utils_v2, "_PRUNED_NUM_PAIRS", pairs)
        metrics_utils_v2.clear_evaluator_cache()
        m = metrics_utils_v2.evaluate_coilset_metrics(
            eq=eq,
            coilset=coilset,
            subset_metrics=True,
            surf_eval_m=surf_eval_m,
            surf_eval_n=surf_eval_n,
            coil_eval_n=coil_eval_n,
        )
        return (
            np.asarray(m.coil_to_coil_min_distances),
            np.asarray(m.coil_to_plasma_min_distances),
        )

    brute_force = _distances(())
    pruned = _distances(num_pairs)
    np.testing.assert_array_equal(pruned[0], brute_force[0])
    np.testing.assert_array_equal(pruned[1], brute_force[1])


def test_min_distances_match_full_objectives(metrics_v1, metrics_v2):
    """V2's distances, computed over one field period and tiled, equal v1's full DESC
    objectives."""
    eps = float(np.finfo(float).eps)
    np.testing.assert_allclose(
        np.asarray(metrics_v1.coil_to_coil_min_distances),
        np.asarray(metrics_v2.coil_to_coil_min_distances),
        rtol=0.0,
        atol=1e3 * eps,
    )
    np.testing.assert_allclose(
        np.asarray(metrics_v1.coil_to_plasma_min_distances),
        np.asarray(metrics_v2.coil_to_plasma_min_distances),
        rtol=0.0,
        atol=1e3 * eps,
    )


@pytest.fixture(scope="module")
def coil_xyz(coilset, coil_eval_n) -> np.ndarray:
    return np.asarray(
        coilset._compute_position(grid=LinearGrid(N=coil_eval_n), basis="xyz")
    )


@pytest.mark.parametrize(
    ("surf_eval_m", "surf_eval_n", "coil_eval_n"),
    [(16, 16, 200), (8, 12, 64)],
    indirect=True,
)
def test_coil_to_plasma_distances_equal_for_mirror_partners(
    eq,
    coilset,
    coil_xyz,
    metrics_v1_or_v2,
):
    assert eq.sym
    assert coilset.sym
    mirrored = coil_xyz * np.array([1.0, -1.0, -1.0])
    mismatch = np.array(
        [
            [
                np.linalg.norm(image[:, None] - c[None], axis=-1).min(axis=1).max()
                for c in coil_xyz
            ]
            for image in mirrored
        ]
    )
    mirror_partners = np.argmin(mismatch, axis=1)
    np.testing.assert_allclose(
        mismatch[np.arange(len(mirror_partners)), mirror_partners],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    eps = float(np.finfo(float).eps)
    coil_to_plasma_v1_or_v2 = metrics_v1_or_v2.coil_to_plasma_min_distances
    np.testing.assert_allclose(
        coil_to_plasma_v1_or_v2,
        coil_to_plasma_v1_or_v2[mirror_partners],
        rtol=0.0,
        atol=1e3 * eps,
    )


@pytest.mark.parametrize(
    ("surf_eval_m", "surf_eval_n", "coil_eval_n"),
    [(16, 16, 200), (8, 12, 64)],
    indirect=True,
)
def test_coil_to_plasma_distances_match_full_surface_brute_force(
    metrics_v1_or_v2,
    coil_xyz,
    eq,
    surf_eval_m,
    surf_eval_n,
):
    grid = LinearGrid(M=surf_eval_m, N=surf_eval_n, NFP=eq.NFP, sym=False)
    rpz = np.asarray(eq.surface.compute("x", grid=grid, basis="rpz")["x"])
    rpz = np.concatenate(
        [rpz + np.array([0.0, 2 * np.pi * k / eq.NFP, 0.0]) for k in range(eq.NFP)]
    )
    full_plasma_xyz = np.stack(
        [rpz[:, 0] * np.cos(rpz[:, 1]), rpz[:, 0] * np.sin(rpz[:, 1]), rpz[:, 2]], -1
    )
    brute_force = np.array(
        [
            np.linalg.norm(c[:, None] - full_plasma_xyz[None], axis=-1).min()
            for c in coil_xyz
        ]
    )
    eps = float(np.finfo(float).eps)
    coil_to_plasma_v1_or_v2 = metrics_v1_or_v2.coil_to_plasma_min_distances
    np.testing.assert_allclose(
        coil_to_plasma_v1_or_v2, brute_force, rtol=0.0, atol=1e3 * eps
    )
