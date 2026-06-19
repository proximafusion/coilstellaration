"""Tests for the plasma-distance field-period reduction patch.

`PlasmaCoilSetMinDistance` tiles the one-field-period plasma grid across all
`NFP` periods (`desc.utils.copy_rpz_periods`) before the per-coil distance
reduction. Combined with `coil_symmetry_patch` (which evaluates only the first
field period's coils), the tiled copies in the other periods never win the
`jnp.min`, so the plasma can be tiled to fewer periods.

The tiling math (`_tile_periods`) is pinned exactly; the end-to-end bit-exact
property on real benchmark geometry is pinned by a slow integration test.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from desc.utils import copy_rpz_periods

from coilstellaration import types
from constellaration.geometry import surface_rz_fourier


def test_tile_periods_full_matches_copy_rpz_periods():
    """Keeping all NFP periods reproduces desc's copy_rpz_periods bit-for-bit."""
    from src.backend.plasma_distance_patch import _tile_periods

    rng = np.random.default_rng(0)
    rpz = jnp.asarray(rng.standard_normal((37, 3)))
    for nfp in (2, 3, 4, 5):
        expected = np.asarray(copy_rpz_periods(rpz, nfp))
        got = np.asarray(_tile_periods(rpz, nfp, nfp))
        np.testing.assert_array_equal(got, expected)


def test_tile_periods_one_keeps_only_period_zero():
    """keep=1 returns exactly the untiled (period-0) points, unrotated."""
    from src.backend.plasma_distance_patch import _tile_periods

    rng = np.random.default_rng(1)
    rpz = jnp.asarray(rng.standard_normal((11, 3)))
    got = np.asarray(_tile_periods(rpz, 3, 1))
    np.testing.assert_array_equal(got, np.asarray(rpz))


def test_tile_periods_partial_keeps_field_period_rotation():
    """keep=2 of NFP=3 keeps the 2*pi/3 spacing, not 2*pi/2.

    It must equal the first `keep * num_nodes` rows of the full tiling, i.e. a
    prefix of periods (0, 1) rather than a re-spaced two-period tiling.
    """
    from src.backend.plasma_distance_patch import _tile_periods

    rng = np.random.default_rng(2)
    n = 8
    rpz = jnp.asarray(rng.standard_normal((n, 3)))
    full = np.asarray(copy_rpz_periods(rpz, 3))
    got = np.asarray(_tile_periods(rpz, 3, 2))
    np.testing.assert_array_equal(got, full[: 2 * n])


def test_tile_periods_clamps_keep_above_nfp():
    """keep > NFP (or keep <= 0) falls back to the full NFP tiling."""
    from src.backend.plasma_distance_patch import _tile_periods

    rng = np.random.default_rng(3)
    rpz = jnp.asarray(rng.standard_normal((6, 3)))
    expected = np.asarray(copy_rpz_periods(rpz, 3))
    np.testing.assert_array_equal(np.asarray(_tile_periods(rpz, 3, 9)), expected)
    np.testing.assert_array_equal(np.asarray(_tile_periods(rpz, 3, 0)), expected)


@pytest.fixture(scope="module")
def real_instance():
    """A real benchmark boundary + a coilset from an example model.

    The plasma single-period reduction is a *geometric* property of realistic
    modular coils (each coil's nearest plasma point lies in its own field
    period), so it must be validated on real coils, not a synthetic fixture.
    """
    from constellaration.geometry import surface_utils_desc

    from src.backend import config, marshal
    from src.backend import data as bd
    from src.backend.eval import _to_desc_coilset
    from src.backend.onnx_model import OnnxModel

    ds = bd.load_benchmark_dataset(
        track="fixed_shape", stratum=config.STRATUM, split=config.SPLIT, n=1
    )
    d = ds[0]
    model = OnnxModel("examples/res_mlp_De3VR8.onnx")
    inputs = marshal.eval_data_to_inputs(d.boundary, d.requirement_metrics)
    coilset = marshal.outputs_to_coilset(model.run(inputs), d.boundary)
    surf = surface_utils_desc.to_desc_fourier_rz_toroidal_surface(d.boundary)
    return surf, _to_desc_coilset(coilset)


@pytest.mark.slow
def test_patch_preserves_plasma_distance_on_benchmark_instance(real_instance):
    """Single-period plasma matches the full-torus baseline bit-for-bit.

    Both runs apply the (provably exact) coil-loop symmetry patch; the only
    difference is the plasma tiling, so this isolates the geometric reduction.
    """
    from desc.objectives import _coils

    from coilstellaration import metrics_utils_v2
    from src.backend import coil_symmetry_patch, plasma_distance_patch

    surf, dc = real_instance
    kwargs = dict(surf_eval_m=80, surf_eval_n=80, coil_eval_n=200, subset_metrics=True)

    orig_make_compute = metrics_utils_v2.MetricsEvaluator._make_compute
    orig_copy = _coils.copy_rpz_periods
    orig_cs_flag = coil_symmetry_patch._PATCHED
    orig_pd_flag = plasma_distance_patch._PATCHED
    try:
        coil_symmetry_patch.apply_patch()
        metrics_utils_v2.clear_evaluator_cache()
        base = metrics_utils_v2.evaluate_coilset_metrics_from_boundary(
            surf, dc, **kwargs
        )
        base_cp = np.asarray(base.coil_to_plasma_min_distances)

        plasma_distance_patch.apply_patch()  # config.PLASMA_PERIODS (default 1)
        metrics_utils_v2.clear_evaluator_cache()
        patched = metrics_utils_v2.evaluate_coilset_metrics_from_boundary(
            surf, dc, **kwargs
        )
    finally:
        metrics_utils_v2.MetricsEvaluator._make_compute = orig_make_compute
        _coils.copy_rpz_periods = orig_copy
        coil_symmetry_patch._PATCHED = orig_cs_flag
        plasma_distance_patch._PATCHED = orig_pd_flag
        metrics_utils_v2.clear_evaluator_cache()

    np.testing.assert_allclose(
        np.asarray(patched.coil_to_plasma_min_distances), base_cp, rtol=0, atol=1e-12
    )


def test_evaluate_achieved_applies_plasma_patch(monkeypatch):
    """The eval path applies the plasma-distance patch before running physics."""
    from coilstellaration import data_utils, metrics_utils_v2
    from src.backend import (
        biot_savart_patch,
        coil_symmetry_patch,
        plasma_distance_patch,
    )
    from src.backend import eval as beval

    calls = {"applied": 0}
    monkeypatch.setattr(biot_savart_patch, "apply_patch", lambda: None)
    monkeypatch.setattr(coil_symmetry_patch, "apply_patch", lambda: None)
    monkeypatch.setattr(
        plasma_distance_patch,
        "apply_patch",
        lambda: calls.__setitem__("applied", calls["applied"] + 1),
    )
    sentinel = object()
    monkeypatch.setattr(
        metrics_utils_v2,
        "evaluate_coilset_metrics_from_boundary",
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
