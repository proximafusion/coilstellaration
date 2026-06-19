"""Tests for the coil field-period symmetry speedup patch.

The patch makes `CoilSetMinDistance` / `PlasmaCoilSetMinDistance` evaluate only
one field period of coils and tile the result, exploiting the exact toroidal
periodicity of a DESC `CoilSet`. These tests pin the patched distance metrics
bit-for-bit to the unpatched baseline and verify the eval path applies it.
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
def test_patch_preserves_distance_metrics(eq_and_coilset):
    """Patched coil-distance metrics match the unpatched baseline bit-for-bit.

    The patch only drops the redundant per-field-period copies of the distance
    computation, so the per-coil min distances must be identical to within
    floating-point noise (not merely within scoring tolerance).
    """
    from coilstellaration import metrics_utils_v2
    from src.backend import coil_symmetry_patch

    eq, coilset = eq_and_coilset
    kwargs = dict(surf_eval_m=16, surf_eval_n=16, coil_eval_n=200, subset_metrics=True)

    orig_method = metrics_utils_v2.MetricsEvaluator._make_compute
    orig_flag = coil_symmetry_patch._PATCHED
    try:
        metrics_utils_v2.clear_evaluator_cache()
        baseline = metrics_utils_v2.evaluate_coilset_metrics(eq, coilset, **kwargs)
        base_cc = np.asarray(baseline.coil_to_coil_min_distances)
        base_cp = np.asarray(baseline.coil_to_plasma_min_distances)

        coil_symmetry_patch.apply_patch()
        metrics_utils_v2.clear_evaluator_cache()
        patched = metrics_utils_v2.evaluate_coilset_metrics(eq, coilset, **kwargs)
    finally:
        metrics_utils_v2.MetricsEvaluator._make_compute = orig_method
        coil_symmetry_patch._PATCHED = orig_flag
        metrics_utils_v2.clear_evaluator_cache()

    np.testing.assert_allclose(
        np.asarray(patched.coil_to_coil_min_distances), base_cc, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(patched.coil_to_plasma_min_distances), base_cp, rtol=0, atol=1e-12
    )


def test_evaluate_achieved_applies_patch(monkeypatch):
    """The eval path applies the coil-symmetry patch before running physics."""
    from coilstellaration import metrics_utils_v2, data_utils
    from src.backend import biot_savart_patch, coil_symmetry_patch
    from src.backend import eval as beval

    calls = {"applied": 0}
    monkeypatch.setattr(biot_savart_patch, "apply_patch", lambda: None)
    monkeypatch.setattr(
        coil_symmetry_patch,
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
