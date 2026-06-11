import json

from src.submission.check_validity import (
    already_submitted_models,
    submission_key,
    validate_local_onnx,
)
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


def test_submission_key_combines_model_commit_and_path():
    assert submission_key("org/m", "abc123", "models/best.onnx") == "org/m@abc123:models/best.onnx"


def test_already_submitted_collects_keys(tmp_path):
    sub = tmp_path / "org"
    sub.mkdir()
    (sub / "a.json").write_text(json.dumps(
        {"model": "org/m", "commit_hash": "abc123", "onnx_path": "model.onnx"}))
    (sub / "b.json").write_text(json.dumps(
        {"model": "org/m", "commit_hash": "def456", "onnx_path": "alt.onnx"}))

    keys = already_submitted_models(str(tmp_path))
    assert keys == {"org/m@abc123:model.onnx", "org/m@def456:alt.onnx"}


def test_already_submitted_defaults_missing_onnx_path(tmp_path):
    sub = tmp_path / "org"
    sub.mkdir()
    (sub / "a.json").write_text(json.dumps({"model": "org/m", "commit_hash": "abc123"}))
    assert already_submitted_models(str(tmp_path)) == {"org/m@abc123:model.onnx"}
