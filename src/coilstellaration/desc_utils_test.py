import numpy as np
import pytest
from coilstellaration import (
    desc_utils,
    metrics_utils_v2,
    types,
)
from constellaration.geometry import surface_rz_fourier, surface_utils_desc
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import QuadratureGrid
from desc.profiles import SplineProfile

_MACHINE_PRECISION = float(np.finfo(float).eps)


@pytest.fixture(scope="module")
def equilibrium() -> Equilibrium:
    # A shaped, non-axisymmetric boundary with a perturbed interior, so the minor
    # radius depends on the interior coefficients and not only on the boundary.
    surface = FourierRZToroidalSurface(
        R_lmn=[1.0, 0.125, 0.1, 0.02],
        Z_lmn=[-0.125, -0.1, 0.02],
        modes_R=[[0, 0], [1, 0], [0, 1], [1, 1]],
        modes_Z=[[-1, 0], [0, -1], [-1, 1]],
        NFP=3,
    )
    knots = np.linspace(0.0, 1.0, 5)
    eq = Equilibrium(
        surface=surface,
        L=4,
        M=4,
        N=2,
        NFP=3,
        Psi=0.3,
        pressure=SplineProfile(1e3 * (1 - knots**2), knots, name="pressure"),
        current=SplineProfile(1e2 * knots**2, knots, name="current"),
    )
    eq.R_lmn = eq.R_lmn + 1e-3 * np.cos(np.arange(np.asarray(eq.R_lmn).size))
    return eq


def test_desc_equilibrium_round_trips_through_json(equilibrium: Equilibrium) -> None:
    stored = types.DescEquilibrium.model_validate_json(
        desc_utils.desc_equilibrium_from_desc_object(equilibrium).model_dump_json()
    )
    restored = desc_utils.desc_equilibrium_to_desc_object(stored)

    for name in ("R_lmn", "Z_lmn", "L_lmn", "Ra_n", "Za_n", "Rb_lmn", "Zb_lmn"):
        np.testing.assert_allclose(
            restored.params_dict[name],
            equilibrium.params_dict[name],
            rtol=0.0,
            atol=0.0,
            err_msg=name,
        )
    # The minor radius normalizes the benchmark's distance and curvature metrics.
    grid = QuadratureGrid(L=16, M=16, N=16, NFP=equilibrium.NFP)
    np.testing.assert_allclose(
        np.asarray(restored.compute("a", grid=grid)["a"]),
        np.asarray(equilibrium.compute("a", grid=grid)["a"]),
        rtol=0.0,
        atol=4 * _MACHINE_PRECISION,
    )
    restored_current, current = restored.current, equilibrium.current
    assert restored_current is not None
    assert current is not None
    rho = np.linspace(0.0, 1.0, 7)
    np.testing.assert_allclose(
        np.asarray(restored_current(rho)),
        np.asarray(current(rho)),
        rtol=0.0,
        atol=0.0,
    )
    assert restored.Psi == equilibrium.Psi
    assert restored.iota is None


def test_boundary_equilibrium_matches_the_cached_one() -> None:
    # A shaped, non-axisymmetric boundary, so the interior guess is not trivial.
    boundary = surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.array([[0.0, 1.0, 0.1], [0.02, 0.125, 0.0]]),
        z_sin=np.array([[0.0, 0.0, -0.1], [0.02, -0.125, 0.0]]),
        n_field_periods=3,
        is_stellarator_symmetric=True,
    )
    restored = desc_utils.desc_equilibrium_to_desc_object(
        desc_utils.desc_equilibrium_from_boundary(boundary)
    )
    cached = metrics_utils_v2.get_cached_equilibrium(
        surface_utils_desc.to_desc_fourier_rz_toroidal_surface(boundary)
    )

    grid = QuadratureGrid(L=8, M=8, N=8, NFP=3)
    for name in ("a", "R0", "Psi"):
        np.testing.assert_allclose(
            np.asarray(restored.compute(name, grid=grid)[name]),
            np.asarray(cached.compute(name, grid=grid)[name]),
            rtol=0.0,
            atol=4 * _MACHINE_PRECISION,
            err_msg=name,
        )
