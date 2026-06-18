import json

from src.submission import requests_dataset, submit
from tests.backend.conftest import build_dummy_onnx


class _Info:
    sha = "deadbeefcafebabe1234567890abcdef12345678"


def _wire(monkeypatch, tmp_path, captured, existing=()):
    """Stub all Hub I/O; `existing` seeds the requests-dataset file list."""
    monkeypatch.setattr(submit.API, "repo_info", lambda **kw: _Info())
    monkeypatch.setattr(requests_dataset, "list_request_files", lambda: list(existing))

    def fake_download(repo_id, filename, revision, local_dir, token):
        captured["download"] = {"filename": filename, "revision": revision}
        return build_dummy_onnx(tmp_path / "dl.onnx")

    monkeypatch.setattr(submit, "hf_hub_download", fake_download)

    def fake_upload(**kw):
        captured["entry"] = json.loads(open(kw["path_or_fileobj"]).read())
        captured["path_in_repo"] = kw["path_in_repo"]

    monkeypatch.setattr(submit.API, "upload_file", fake_upload)


def test_submission_records_commit_hash_and_default_path(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    out = submit.add_new_eval("org/m", "main", "alice", False, 1, -1)

    assert "submitted" in out.lower()
    assert captured["entry"]["commit_hash"] == _Info.sha
    assert captured["entry"]["onnx_path"] == "model.onnx"
    assert captured["path_in_repo"] == "org/m_eval_request_%s_model.json" % _Info.sha
    assert captured["download"]["revision"] == _Info.sha
    assert captured["download"]["filename"] == "model.onnx"


def test_submission_accepts_custom_onnx_path(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    out = submit.add_new_eval("org/m", "main", "alice", False, 1, -1, onnx_path="models/best.onnx")

    assert "submitted" in out.lower()
    assert captured["entry"]["onnx_path"] == "models/best.onnx"
    assert captured["download"]["filename"] == "models/best.onnx"


def test_submission_rejects_non_onnx_path(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    out = submit.add_new_eval("org/m", "main", "alice", False, 1, -1, onnx_path="weights.bin")

    assert ".onnx" in out
    assert "entry" not in captured
    assert "download" not in captured


def test_submission_rejects_duplicate(tmp_path, monkeypatch):
    captured = {}
    existing = [requests_dataset.request_path("org/m", _Info.sha, "model.onnx")]
    _wire(monkeypatch, tmp_path, captured, existing=existing)

    out = submit.add_new_eval("org/m", "main", "alice", False, 1, -1)

    assert "already" in out.lower()
    assert "entry" not in captured
    assert "download" not in captured  # dedup fires before the download


def test_submission_fails_closed_when_queue_unreadable(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    def boom():
        raise RuntimeError("hub down")

    monkeypatch.setattr(requests_dataset, "list_request_files", boom)

    out = submit.add_new_eval("org/m", "main", "alice", False, 1, -1)

    assert "retry" in out.lower()
    assert "entry" not in captured
    assert "download" not in captured  # never reaches validation
