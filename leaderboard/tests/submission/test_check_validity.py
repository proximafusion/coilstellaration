from src.submission.check_validity import validate_local_onnx
from tests.backend.conftest import build_dummy_onnx


def test_valid_model_passes(tmp_path):
    path = build_dummy_onnx(tmp_path / "model.onnx")
    ok, msg = validate_local_onnx(path)
    assert ok, msg


def test_missing_tensor_fails(tmp_path):
    path = build_dummy_onnx(tmp_path / "model.onnx", output_names=("coil_x_n", "coil_y_n", "coil_z_n"))
    ok, msg = validate_local_onnx(path)
    assert not ok
    assert "currents" in msg
