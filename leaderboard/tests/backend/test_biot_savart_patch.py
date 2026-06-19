"""Tests for the bench-local Biot-Savart speedup patch.

The patch lowers the surface-field Biot-Savart source resolution inside
`coilstellaration.metrics_utils_v2` (decoupled from the geometric-metric grid).
These tests pin the patched metric values to the unpatched baseline and verify
the eval path applies the patch.

Tests that construct DESC equilibria/coilsets and run physics are marked
``slow``.
"""

import numpy as np
import pytest
from desc.coils import CoilSet, FourierXYZCoil
from desc.examples import get

from coilstellaration import types
from constellaration.geometry import surface_rz_fourier


@pytest.fixture()
def eq_and_coilset():
    eq = get("precise_QA")
    rng = np.random.default_rng(0)
    coils = []
    for _ in range(3):
        c = FourierXYZCoil(current=1e5)
        c.change_resolution(N=4)
        c.X_n = c.X_n + 0.05 * rng.standard_normal(c.X_n.size)
        c.Y_n = c.Y_n + 0.05 * rng.standard_normal(c.Y_n.size)
        c.Z_n = c.Z_n + 0.05 * rng.standard_normal(c.Z_n.size)
        coils.append(c)
    coilset = CoilSet(*coils, NFP=eq.NFP, sym=eq.sym)
    return eq, coilset


@pytest.mark.slow
def test_patch_preserves_surface_field_metrics(eq_and_coilset):
    """Patched surface-field metrics match the unpatched baseline.

    The patch lowers the Biot-Savart source resolution; for smooth Fourier
    coils the field is fully converged, so the field-error and quadratic-flux
    metrics must be unchanged to well within scoring tolerance.
    """
    from coilstellaration import metrics_utils_v2
    from src.backend import biot_savart_patch

    eq, coilset = eq_and_coilset
    kwargs = dict(surf_eval_m=16, surf_eval_n=16, coil_eval_n=200, subset_metrics=True)

    orig_method = metrics_utils_v2.MetricsEvaluator._make_surface_field_fn
    orig_flag = biot_savart_patch._PATCHED
    try:
        metrics_utils_v2.clear_evaluator_cache()
        baseline = metrics_utils_v2.evaluate_coilset_metrics(eq, coilset, **kwargs)
        base_nfe = np.asarray(baseline.normalized_field_error)
        base_qf = np.asarray(baseline.quadratic_flux)

        biot_savart_patch.apply_patch()
        metrics_utils_v2.clear_evaluator_cache()
        patched = metrics_utils_v2.evaluate_coilset_metrics(eq, coilset, **kwargs)
    finally:
        metrics_utils_v2.MetricsEvaluator._make_surface_field_fn = orig_method
        biot_savart_patch._PATCHED = orig_flag
        metrics_utils_v2.clear_evaluator_cache()

    np.testing.assert_allclose(
        np.asarray(patched.normalized_field_error), base_nfe, rtol=1e-6, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(patched.quadratic_flux), base_qf, rtol=1e-6, atol=1e-9
    )


def test_evaluate_achieved_applies_patch(monkeypatch):
    """The eval path applies the speedup patch before running physics."""
    from coilstellaration import metrics_utils_v2, data_utils
    from src.backend import biot_savart_patch
    from src.backend import eval as beval

    calls = {"applied": 0}
    monkeypatch.setattr(
        biot_savart_patch,
        "apply_patch",
        lambda: calls.__setitem__("applied", calls["applied"] + 1),
    )
    sentinel = object()
    monkeypatch.setattr(
        metrics_utils_v2, "evaluate_coilset_metrics_from_boundary",
        lambda **kwargs: sentinel,
    )
    monkeypatch.setattr(
        data_utils, "metrics_to_requirement_metrics", lambda metrics: metrics
    )

    boundary = surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.zeros((2, 3)), z_sin=np.zeros((2, 3)),
        n_field_periods=3, is_stellarator_symmetric=True,
    )
    coilset = types.Coilset(
        coil_x_n=np.zeros((4, 5)), coil_y_n=np.zeros((4, 5)), coil_z_n=np.zeros((4, 5)),
        currents=np.ones((4,)), n_field_periods=3, is_stellarator_symmetric=True,
    )

    result = beval._evaluate_achieved(boundary, coilset)
    assert result is sentinel
    assert calls["applied"] == 1
