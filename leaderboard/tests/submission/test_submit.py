import json
from pathlib import Path

from leaderboard.src.backend import config
from leaderboard.src.submission import requests_dataset, submit
from tests.backend.conftest import build_dummy_onnx


class _Info:
    sha = "deadbeefcafebabe1234567890abcdef12345678"


def _wire(
    monkeypatch,
    tmp_path,
    captured,
    existing=(),
    size=1024,
    supported_tracks="fixed_shape",
):
    """Stub all Hub I/O; `existing` seeds the requests-dataset file list.

    `size` is the file size get_paths_info reports; `supported_tracks` is the
    dummy ONNX's track metadata.
    """

    def fake_repo_info(**kw):
        captured["repo_info_token"] = kw.get("token", "unset")
        return _Info()

    monkeypatch.setattr(submit.API, "repo_info", fake_repo_info)
    monkeypatch.setattr(requests_dataset, "list_request_files", lambda: list(existing))

    class _PathInfo:
        def __init__(self, path, size):
            self.path = path
            self.size = size

    def fake_paths_info(repo_id, paths, **kw):
        captured["paths_info"] = {"repo_id": repo_id, "paths": list(paths)}
        captured["paths_info_token"] = kw.get("token", "unset")
        return [_PathInfo(paths[0], size)]

    monkeypatch.setattr(submit.API, "get_paths_info", fake_paths_info)

    def fake_download(repo_id, filename, revision, local_dir, token):
        captured["download"] = {
            "filename": filename,
            "revision": revision,
            "token": token,
        }
        return build_dummy_onnx(tmp_path / "dl.onnx", supported_tracks=supported_tracks)

    monkeypatch.setattr(submit, "hf_hub_download", fake_download)

    def fake_upload(**kw):
        captured["entry"] = json.loads(Path(kw["path_or_fileobj"]).read_text())
        captured["path_in_repo"] = kw["path_in_repo"]

    monkeypatch.setattr(submit.API, "upload_file", fake_upload)


def test_submission_records_commit_hash_and_default_path(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    out = submit.add_new_eval("org/m", "main", "alice")

    assert "submitted" in out.lower()
    assert captured["entry"]["commit_hash"] == _Info.sha
    assert captured["entry"]["onnx_path"] == "model.onnx"
    assert captured["path_in_repo"] == "org/m_eval_request_%s_model.json" % _Info.sha
    assert captured["download"]["revision"] == _Info.sha
    assert captured["download"]["filename"] == "model.onnx"


def test_submission_reads_the_repo_anonymously(tmp_path, monkeypatch):
    """The Space's token must not be used to read submitted repos, so private
    repos it can see are not submittable."""
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    submit.add_new_eval("org/m", "main", "alice")

    assert captured["repo_info_token"] is False
    assert captured["paths_info_token"] is False
    assert captured["download"]["token"] is False


def test_submission_accepts_custom_onnx_path(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    out = submit.add_new_eval("org/m", "main", "alice", onnx_path="models/best.onnx")

    assert "submitted" in out.lower()
    assert captured["entry"]["onnx_path"] == "models/best.onnx"
    assert captured["download"]["filename"] == "models/best.onnx"


def test_submission_rejects_non_onnx_path(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    out = submit.add_new_eval("org/m", "main", "alice", onnx_path="weights.bin")

    assert ".onnx" in out
    assert "entry" not in captured
    assert "download" not in captured


def test_submission_rejects_duplicate(tmp_path, monkeypatch):
    captured = {}
    existing = [requests_dataset.request_path("org/m", _Info.sha, "model.onnx")]
    _wire(monkeypatch, tmp_path, captured, existing=existing)

    out = submit.add_new_eval("org/m", "main", "alice")

    assert "already" in out.lower()
    assert "entry" not in captured
    assert "download" not in captured  # dedup fires before the download


def test_submission_fails_closed_when_queue_unreadable(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    def boom():
        raise RuntimeError("hub down")

    monkeypatch.setattr(requests_dataset, "list_request_files", boom)

    out = submit.add_new_eval("org/m", "main", "alice")

    assert "retry" in out.lower()
    assert "entry" not in captured
    assert "download" not in captured  # never reaches validation


def test_submission_rejects_oversized_model(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured, size=config.MAX_ONNX_BYTES + 1)

    out = submit.add_new_eval("org/m", "main", "alice")

    assert "limit" in out.lower()
    assert "entry" not in captured
    assert "download" not in captured  # rejected before download


def test_submission_records_fixed_only_tracks(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)  # default dummy declares fixed_shape

    out = submit.add_new_eval("org/m", "main", "alice")

    assert captured["entry"]["tracks"] == ["fixed_shape"]
    assert "fixed_shape" in out


def test_submission_records_both_tracks(tmp_path, monkeypatch):
    captured = {}
    _wire(
        monkeypatch, tmp_path, captured, supported_tracks="fixed_shape,variable_shape"
    )

    out = submit.add_new_eval("org/m", "main", "alice")

    assert captured["entry"]["tracks"] == ["fixed_shape", "variable_shape"]
    assert "both" in out.lower()


def test_submission_falls_back_to_getsize_when_metadata_lacks_size(
    tmp_path, monkeypatch
):
    captured = {}
    _wire(monkeypatch, tmp_path, captured, size=None)  # metadata reports no size

    out = submit.add_new_eval("org/m", "main", "alice")

    # A tiny dummy file is well under the limit, so it still queues.
    assert "submitted" in out.lower()
    assert captured["entry"]["tracks"] == ["fixed_shape"]
