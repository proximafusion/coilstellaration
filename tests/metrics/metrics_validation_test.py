"""Validation tests for evaluate_coilset_metrics against reactor_design tools.

These tests verify that the DESC-based metrics computed in beta/constellaration_update
match the equivalent computations from the reactor_design geometry modules. This ensures
consistency between the two implementations.
"""

import numpy as np
import pytest

# TODO(migration): geometry.curve has no constellaration equivalent — escalate or
# rewrite tests against a DESC-only baseline when geometry.curve becomes available.

pytestmark = pytest.mark.skip(
    reason="requires Proxima-internal geometry.curve (no constellaration equivalent)"
)

metrics_utils = pytest.importorskip(
    "constellaration_update.metrics.metrics",
    reason="requires constellaration_update.types (Task 8)",
    exc_type=ImportError,
)

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
def desc_equilibrium_and_coilset():
    """Load a small DESC example equilibrium and build a coilset."""
    eq = get("precise_QA")
    assert isinstance(eq, Equilibrium)
    coil = FourierXYZCoil(current=100000)
    unique_coilset = CoilSet(coil)
    coilset = CoilSet.from_symmetry(unique_coilset, NFP=eq.NFP, sym=eq.sym)
    return eq, coilset


def test_coil_lengths_match_geometry_curve_utils(desc_equilibrium_and_coilset):
    """Validate coil_lengths against geometry.curve.curve_utils.compute_length.

    The DESC CoilSet computes coil lengths via its internal methods. This test verifies
    that these lengths match our independent computation using
    curve_utils.compute_length on the same coil geometry.
    """
    eq, coilset = desc_equilibrium_and_coilset
    coil_eval_n = 50
    coil_eval_grid = LinearGrid(N=coil_eval_n)

    desc_lengths = metrics_utils.compute_coil_lengths(coilset, coil_eval_grid)

    # TODO(migration): curve_utils.compute_length — geometry.curve not ported
    expected_lengths = [None]

    np.testing.assert_allclose(
        desc_lengths[0],
        expected_lengths[0],
        rtol=1e-4,
        err_msg="Coil lengths from DESC should match curve_utils.compute_length",
    )


def test_coil_curvatures_match_geometry_curve_utils(desc_equilibrium_and_coilset):
    """Validate coil_curvatures against geometry.curve.curve_utils.curvature.

    DESC computes curvature at evaluation points. This test verifies that curvature
    values match our independent computation using curve_utils.curvature.
    """
    eq, coilset = desc_equilibrium_and_coilset
    coil_eval_n = 30
    coil_eval_grid = LinearGrid(N=coil_eval_n)

    desc_curvatures = metrics_utils.compute_coil_curvatures(coilset, coil_eval_grid)

    for i, _desc_coil in enumerate(coilset):
        # TODO(migration): curve_utils.curvature unavailable — geometry.curve not ported
        expected_curvature = None

        np.testing.assert_allclose(
            desc_curvatures[i],
            expected_curvature,
            rtol=1e-3,
            err_msg=f"Coil {i} curvatures should match curve_utils.curvature",
        )
        # Only test the first unique coil; symmetry-expanded coils are duplicates
        break


def test_coil_torsions_match_geometry_curve_utils(desc_equilibrium_and_coilset):
    """Validate coil_torsions against geometry.curve.curve_utils.torsion.

    DESC computes torsion at evaluation points. This test verifies that torsion values
    match our independent computation using curve_utils.torsion.
    """
    eq, coilset = desc_equilibrium_and_coilset
    coil_eval_n = 30
    coil_eval_grid = LinearGrid(N=coil_eval_n)

    desc_torsions = metrics_utils.compute_coil_torsions(coilset, coil_eval_grid)

    for i, _desc_coil in enumerate(coilset):
        # TODO(migration): curve_utils.torsion unavailable — geometry.curve not ported
        expected_torsion = None

        np.testing.assert_allclose(
            desc_torsions[i],
            expected_torsion,
            rtol=1e-3,
            atol=1e-6,
            err_msg=f"Coil {i} torsions should match curve_utils.torsion",
        )
        # Only test the first unique coil; symmetry-expanded coils are duplicates
        break


def test_integrated_curvature_equals_2pi_for_planar_circle():
    """Validate that coil_integrated_curvatures equals 2*pi for planar convex coils.

    A circular coil (planar, convex) should have integrated curvature equal to 2*pi.
    This is a fundamental geometric property that should hold regardless of
    implementation.
    """
    radius = 5.0
    major_radius = 10.0
    coil = FourierXYZCoil(
        X_n=[major_radius, radius, 0],
        Y_n=[0, 0, radius],
        Z_n=[0, 0, 0],
        current=100000,
    )
    coilset = CoilSet(coil, NFP=1, sym=False)

    coil_eval_grid = LinearGrid(N=100)
    integrated_curvatures = metrics_utils.compute_coil_integrated_curvatures(
        coilset, coil_eval_grid
    )

    np.testing.assert_allclose(
        integrated_curvatures[0],
        2 * np.pi,
        rtol=1e-4,
        err_msg="Integrated curvature of planar circular coil should be 2*pi",
    )
