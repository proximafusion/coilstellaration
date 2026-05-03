import jax.numpy as jnp
import numpy as np
import pytest

_SKIP_REASON = "requires constellaration_update.types (Task 8)"

pytest.importorskip(
    "constellaration_update.types",
    reason=_SKIP_REASON,
    exc_type=ImportError,
)
metrics_utils = pytest.importorskip(
    "constellaration_update.metrics.metrics",
    reason=_SKIP_REASON,
    exc_type=ImportError,
)

pytestmark = pytest.mark.skip(reason=_SKIP_REASON)

try:
    from desc.coils import CoilSet, FourierXYZCoil
    from desc.equilibrium import Equilibrium
    from desc.examples import get
    from desc.grid import LinearGrid
except Exception:  # noqa: BLE001
    CoilSet = None  # type: ignore[assignment,misc]
    FourierXYZCoil = None  # type: ignore[assignment,misc]
    Equilibrium = None  # type: ignore[assignment,misc]
    get = None  # type: ignore[assignment]
    LinearGrid = None  # type: ignore[assignment,misc]


@pytest.fixture
def desc_eq_and_coilset():
    """Load a small DESC example equilibrium and build a simple coilset."""
    _eq = get("precise_QA")
    assert isinstance(_eq, Equilibrium)
    eq = _eq
    coil = FourierXYZCoil(current=100000)
    unique_coilset = CoilSet(coil)
    coilset = CoilSet.from_symmetry(unique_coilset, NFP=eq.NFP, sym=eq.sym)
    return eq, coilset


def test_compute_surface_field_metrics_shapes(desc_eq_and_coilset):
    eq, coilset = desc_eq_and_coilset
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


def test_compute_surface_field_metrics_consistency(desc_eq_and_coilset):
    """Verify local_qf = normalized_field^2 * dS."""
    eq, coilset = desc_eq_and_coilset
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


def test_compute_on_axis_average_magnetic_field(desc_eq_and_coilset):
    eq, coilset = desc_eq_and_coilset
    zeta_n = 64

    B_avg = metrics_utils.compute_on_axis_average_magnetic_field(
        coilset, eq, zeta_n=zeta_n
    )

    # Analytical estimate: each default FourierXYZCoil is a circle of radius
    # a=2 m centered at R_coil=10 m.  In the magnetic-dipole limit
    # (a << distance to axis) the field per coil is mu_0*m/(4*pi*d^3) with
    # m = I*pi*a^2.  The RMS distance from the axis (at R0) to any coil
    # center (at R_coil) averaged over toroidal angle is sqrt(R_coil^2+R0^2).
    R0 = float(eq.compute("R0")["R0"])
    mu_0 = 4 * np.pi * 1e-7
    total_coils_current = 100_000.0
    a_coil = 2.0
    R_coil = 10.0
    n_coils = len(coilset)
    d_avg = np.sqrt(R_coil**2 + R0**2)
    expected_B = n_coils * mu_0 * total_coils_current * a_coil**2 / (4 * d_avg**3)
    np.testing.assert_allclose(float(B_avg), expected_B, rtol=0.5)
