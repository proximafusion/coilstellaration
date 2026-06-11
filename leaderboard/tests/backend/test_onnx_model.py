import numpy as np

from src.backend import onnx_model
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
    path = build_dummy_onnx(
        tmp_path / "bad.onnx", output_names=("coil_x_n", "coil_y_n", "coil_z_n")
    )
    errors = onnx_model.validate_onnx_contract(path)
    assert any("currents" in e for e in errors)


def test_run_returns_named_outputs_with_shapes(dummy_onnx_path):
    model = onnx_model.OnnxModel(dummy_onnx_path)
    out = model.run({
        "boundary_r_cos": np.zeros((2, 3), np.float32),
        "boundary_z_sin": np.zeros((2, 3), np.float32),
        "requirement_metrics": np.zeros((4,), np.float32),
    })
    assert set(out) == {"coil_x_n", "coil_y_n", "coil_z_n", "currents"}
    assert out["coil_x_n"].shape == (4, 5)
    assert out["currents"].shape == (4,)
