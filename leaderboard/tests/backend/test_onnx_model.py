import numpy as np
import pytest

from leaderboard.src.backend import config, onnx_model
from tests.backend.conftest import build_dummy_onnx


def test_validate_accepts_conforming_model(dummy_onnx_path):
    errors = onnx_model.validate_onnx_contract(dummy_onnx_path)
    assert errors == []


def test_validate_rejects_missing_required_input(tmp_path):
    path = build_dummy_onnx(
        tmp_path / "bad.onnx",
        input_names=("boundary_r_cos", "boundary_z_sin"),  # no requirement_metrics
    )
    errors = onnx_model.validate_onnx_contract(path)
    assert any("requirement_metrics" in e for e in errors)


def test_validate_rejects_missing_output(tmp_path):
    path = build_dummy_onnx(tmp_path / "bad.onnx", output_names=("coil_x_n", "coil_y_n", "coil_z_n"))
    errors = onnx_model.validate_onnx_contract(path)
    assert any("currents" in e for e in errors)


def test_run_returns_named_outputs_with_shapes(dummy_onnx_path):
    model = onnx_model.OnnxModel(dummy_onnx_path)
    out = model.run(
        {
            "boundary_r_cos": np.zeros((2, 3), np.float32),
            "boundary_z_sin": np.zeros((2, 3), np.float32),
            "requirement_metrics": np.zeros((4,), np.float32),
            "n_field_periods": np.asarray(3, np.int64),
            "n_coils_per_half_period": np.asarray(5, np.int64),
        }
    )
    assert set(out) == {"coil_x_n", "coil_y_n", "coil_z_n", "currents"}
    assert out["coil_x_n"].shape == (4, 5)
    assert out["currents"].shape == (4,)


def _session(path):
    return onnx_model.ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def test_read_supported_tracks_both(tmp_path):
    path = build_dummy_onnx(tmp_path / "m.onnx", supported_tracks="fixed_shape,variable_shape")
    assert onnx_model.read_supported_tracks(_session(path)) == ["fixed_shape", "variable_shape"]


def test_read_supported_tracks_fixed_only(tmp_path):
    path = build_dummy_onnx(tmp_path / "m.onnx", supported_tracks="fixed_shape")
    assert onnx_model.read_supported_tracks(_session(path)) == ["fixed_shape"]


def test_missing_metadata_is_contract_error(tmp_path):
    path = build_dummy_onnx(tmp_path / "m.onnx", supported_tracks=None)
    with pytest.raises(ValueError, match="supported_tracks"):
        onnx_model.read_supported_tracks(_session(path))
    errors, tracks = onnx_model.inspect_onnx(path)
    assert any("supported_tracks" in e for e in errors)
    assert tracks == []


@pytest.mark.parametrize(
    "value", ["fixed_shape, variable", "fixed_shape, variable_shape", "variable_shape", "bogus", ""]
)
def test_invalid_metadata_is_contract_error(tmp_path, value):
    path = build_dummy_onnx(tmp_path / "m.onnx", supported_tracks=value)
    with pytest.raises(ValueError, match="supported_tracks"):
        onnx_model.read_supported_tracks(_session(path))
    errors, tracks = onnx_model.inspect_onnx(path)
    assert any("supported_tracks" in e for e in errors)
    assert tracks == []


@pytest.mark.parametrize("missing", ["n_field_periods", "n_coils_per_half_period"])
def test_missing_count_input_is_contract_error(tmp_path, missing):
    names = tuple(n for n in config.REQUIRED_INPUT_NAMES if n != missing)
    path = build_dummy_onnx(tmp_path / "bad.onnx", input_names=names)
    errors = onnx_model.validate_onnx_contract(path)
    assert any(missing in e for e in errors)
