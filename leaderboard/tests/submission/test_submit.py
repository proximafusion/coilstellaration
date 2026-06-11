import json

from src.submission import submit
from tests.backend.conftest import build_dummy_onnx


class _Info:
    sha = "deadbeefcafebabe1234567890abcdef12345678"


def _wire(monkeypatch, tmp_path, captured):
    """Point submit at tmp dirs and stub all Hub I/O."""
    monkeypatch.setattr(submit, "EVAL_REQUESTS_PATH", str(tmp_path))
    monkeypatch.setattr(submit.API, "repo_info", lambda **kw: _Info())

    def fake_download(repo_id, filename, revision, local_dir, token):
        captured["download"] = {"filename": filename, "revision": revision}
        return build_dummy_onnx(tmp_path / "dl.onnx")

    monkeypatch.setattr(submit, "hf_hub_download", fake_download)

    def fake_upload(**kw):
        captured["entry"] = json.loads(open(kw["path_or_fileobj"]).read())

    monkeypatch.setattr(submit.API, "upload_file", fake_upload)


def test_submission_records_commit_hash_and_default_path(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    out = submit.add_new_eval("org/m", "main", "alice", False, 1, -1)

    assert "submitted" in out.lower()
    assert captured["entry"]["commit_hash"] == _Info.sha
    assert captured["entry"]["onnx_path"] == "model.onnx"
    # The file is downloaded at the pinned SHA, not the named ref.
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
    assert "entry" not in captured  # nothing queued


def test_submission_rejects_duplicate(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)
    # Pre-seed an identical (model, commit, file) request in the queue dir.
    sub = tmp_path / "org"
    sub.mkdir()
    (sub / "existing.json").write_text(json.dumps(
        {"model": "org/m", "commit_hash": _Info.sha, "onnx_path": "model.onnx"}))

    out = submit.add_new_eval("org/m", "main", "alice", False, 1, -1)

    assert "already" in out.lower()
    assert "entry" not in captured
