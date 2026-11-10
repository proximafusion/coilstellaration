from leaderboard.src.submission.check_validity import validate_local_onnx
from tests.backend.conftest import build_dummy_onnx


def test_valid_model_passes(tmp_path):
    path = build_dummy_onnx(tmp_path / "model.onnx")
    ok, msg, tracks = validate_local_onnx(path)
    assert ok, msg
    assert tracks == ["fixed_shape"]


def test_variable_model_reports_both_tracks(tmp_path):
    path = build_dummy_onnx(
        tmp_path / "model.onnx", supported_tracks="fixed_shape,variable_shape"
    )
    ok, msg, tracks = validate_local_onnx(path)
    assert ok, msg
    assert tracks == ["fixed_shape", "variable_shape"]


def test_missing_tensor_fails(tmp_path):
    path = build_dummy_onnx(
        tmp_path / "model.onnx", output_names=("coil_x_n", "coil_y_n", "coil_z_n")
    )
    ok, msg, tracks = validate_local_onnx(path)
    assert not ok
    assert "currents" in msg
    assert tracks == []


def test_invalid_track_metadata_fails(tmp_path):
    path = build_dummy_onnx(
        tmp_path / "model.onnx", supported_tracks="fixed_shape, variable"
    )
    ok, msg, tracks = validate_local_onnx(path)
    assert not ok
    assert "supported_tracks" in msg
    assert tracks == []
