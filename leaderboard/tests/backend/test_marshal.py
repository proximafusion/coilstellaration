import numpy as np
import pytest

from constellaration.geometry import surface_rz_fourier
from coilstellaration import types

from leaderboard.src.backend import marshal


def _symmetric_boundary():
    r_cos = np.array([[0.0, 1.0, 0.1], [0.0, 0.05, 0.0]], dtype=float)
    z_sin = np.array([[0.0, 0.0, 0.2], [0.0, 0.03, 0.0]], dtype=float)
    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos, z_sin=z_sin, n_field_periods=3, is_stellarator_symmetric=True
    )


def _requirements():
    return types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=0.1,
        min_normalized_coil_to_plasma_distance=0.2,
        max_normalized_coil_curvature=1.5,
        max_normalized_field_error=0.01,
        n_coils_per_half_period=5,
    )


def test_inputs_contain_required_names_and_shapes():
    boundary = _symmetric_boundary()
    inputs = marshal.eval_data_to_inputs(boundary, _requirements())
    assert set(inputs) == {
        "boundary_r_cos",
        "boundary_z_sin",
        "requirement_metrics",
        "n_field_periods",
        "n_coils_per_half_period",
    }
    assert inputs["boundary_r_cos"].shape == (2, 3)
    assert inputs["requirement_metrics"].shape == (4,)
    assert inputs["requirement_metrics"].dtype == np.float32
    # order matches RequirementMetrics.to_array()
    np.testing.assert_allclose(
        inputs["requirement_metrics"], [0.1, 0.2, 1.5, 0.01], rtol=1e-6
    )


def test_inputs_include_int64_counts():
    inputs = marshal.eval_data_to_inputs(_symmetric_boundary(), _requirements())
    for name, expected in (("n_field_periods", 3), ("n_coils_per_half_period", 5)):
        assert inputs[name].dtype == np.int64
        assert inputs[name].shape == ()
        assert int(inputs[name]) == expected


def test_asymmetric_inputs_include_r_sin_z_cos():
    r = np.zeros((2, 3))
    boundary = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r.copy(),
        z_sin=r.copy(),
        r_sin=r.copy(),
        z_cos=r.copy(),
        n_field_periods=1,
        is_stellarator_symmetric=False,
    )
    inputs = marshal.eval_data_to_inputs(boundary, _requirements())
    assert {"boundary_r_sin", "boundary_z_cos"} <= set(inputs)


def _desc_equilibrium():
    return types.DescEquilibrium(
        r_lmn=np.array([1.0, 0.1]),
        z_lmn=np.array([0.1]),
        lambda_lmn=np.array([0.0]),
        radial_resolution=1,
        poloidal_resolution=1,
        toroidal_resolution=0,
        n_field_periods=3,
        pressure_profile=types.DescSplineProfile(
            knots=np.array([0.0, 1.0]), params=np.array([1.0, 0.0])
        ),
    )


def test_outputs_to_coilset_uses_equilibrium_symmetry_and_nfp():
    outputs = {
        "coil_x_n": np.zeros((4, 5), dtype=np.float32),
        "coil_y_n": np.zeros((4, 5), dtype=np.float32),
        "coil_z_n": np.zeros((4, 5), dtype=np.float32),
        "currents": np.ones((4,), dtype=np.float32),
    }
    coilset = marshal.outputs_to_coilset(outputs, _desc_equilibrium())
    assert isinstance(coilset, types.Coilset)
    assert coilset.n_unique_coils == 4
    assert int(coilset.n_field_periods) == 3
    assert coilset.is_stellarator_symmetric is True
    np.testing.assert_allclose(np.asarray(coilset.currents), [1, 1, 1, 1])


def test_missing_output_key_raises():
    with pytest.raises(KeyError):
        marshal.outputs_to_coilset({"coil_x_n": np.zeros((4, 5))}, _desc_equilibrium())
