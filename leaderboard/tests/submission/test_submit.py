import json

import gradio as gr
import pytest

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
    size: int | None = 1024,
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
    # Every submitter in these tests is a real Hub account unless a test says
    # otherwise.
    monkeypatch.setattr(submit, "hub_account_exists", lambda name: True)
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
        captured["entry"] = json.loads(kw["path_or_fileobj"])
        captured["path_in_repo"] = kw["path_in_repo"]

    monkeypatch.setattr(submit.API, "upload_file", fake_upload)


def test_submission_records_commit_hash_and_default_path(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    out = submit.add_new_eval("org/m", "main", "alice", no_eval_training=True)

    assert "submitted" in out.lower()
    assert captured["entry"]["commit_hash"] == _Info.sha
    assert captured["entry"]["onnx_path"] == "model.onnx"
    assert captured["path_in_repo"] == "org/m_eval_request_%s_model.json" % _Info.sha
    assert captured["download"]["revision"] == _Info.sha
    assert captured["download"]["filename"] == "model.onnx"
    downloaded = tmp_path / "dl.onnx"
    assert captured["entry"]["onnx_sha256"] == submit.file_sha256(downloaded)


def test_submission_reads_the_repo_anonymously(tmp_path, monkeypatch):
    """The Space's token must not be used to read submitted repos, so private
    repos it can see are not submittable."""
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    submit.add_new_eval("org/m", "main", "alice", no_eval_training=True)

    assert captured["repo_info_token"] is False
    assert captured["paths_info_token"] is False
    assert captured["download"]["token"] is False


def test_submission_accepts_custom_onnx_path(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    out = submit.add_new_eval(
        "org/m", "main", "alice", onnx_path="models/best.onnx", no_eval_training=True
    )

    assert "submitted" in out.lower()
    assert captured["entry"]["onnx_path"] == "models/best.onnx"
    assert captured["download"]["filename"] == "models/best.onnx"


def test_submission_rejects_non_onnx_path(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    with pytest.raises(gr.Error, match=r"(?i)\.onnx"):
        submit.add_new_eval(
            "org/m", "main", "alice", onnx_path="weights.bin", no_eval_training=True
        )

    assert "entry" not in captured
    assert "download" not in captured


def test_submission_rejects_duplicate(tmp_path, monkeypatch):
    captured = {}
    existing = [requests_dataset.request_path("org/m", _Info.sha, "model.onnx")]
    _wire(monkeypatch, tmp_path, captured, existing=existing)

    with pytest.raises(gr.Error, match=r"(?i)already"):
        submit.add_new_eval("org/m", "main", "alice", no_eval_training=True)

    assert "entry" not in captured
    assert "download" not in captured  # dedup fires before the download


def test_submission_fails_closed_when_queue_unreadable(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    def boom():
        raise RuntimeError("hub down")

    monkeypatch.setattr(requests_dataset, "list_request_files", boom)

    with pytest.raises(gr.Error, match=r"(?i)retry"):
        submit.add_new_eval("org/m", "main", "alice", no_eval_training=True)

    assert "entry" not in captured
    assert "download" not in captured  # never reaches validation


def test_submission_rejects_oversized_model(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured, size=config.MAX_ONNX_BYTES + 1)

    with pytest.raises(gr.Error, match=r"(?i)limit"):
        submit.add_new_eval("org/m", "main", "alice", no_eval_training=True)

    assert "entry" not in captured
    assert "download" not in captured  # rejected before download


def test_submission_records_fixed_only_tracks(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)  # default dummy declares fixed_shape

    out = submit.add_new_eval("org/m", "main", "alice", no_eval_training=True)

    assert captured["entry"]["tracks"] == ["fixed_shape"]
    assert "fixed_shape" in out


def test_submission_records_both_tracks(tmp_path, monkeypatch):
    captured = {}
    _wire(
        monkeypatch, tmp_path, captured, supported_tracks="fixed_shape,variable_shape"
    )

    out = submit.add_new_eval("org/m", "main", "alice", no_eval_training=True)

    assert captured["entry"]["tracks"] == ["fixed_shape", "variable_shape"]
    assert "both" in out.lower()


def test_submission_falls_back_to_getsize_when_metadata_lacks_size(
    tmp_path, monkeypatch
):
    captured = {}
    _wire(monkeypatch, tmp_path, captured, size=None)  # metadata reports no size

    out = submit.add_new_eval("org/m", "main", "alice", no_eval_training=True)

    # A tiny dummy file is well under the limit, so it still queues.
    assert "submitted" in out.lower()
    assert captured["entry"]["tracks"] == ["fixed_shape"]


def test_submission_reports_queue_upload_failure(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    def boom(**kw):
        raise RuntimeError("hub down")

    monkeypatch.setattr(submit.API, "upload_file", boom)

    with pytest.raises(gr.Error, match="retry"):
        submit.add_new_eval("org/m", "main", "alice", no_eval_training=True)


def test_validator_flags_each_invalid_field():
    results = submit.validate_submission("no-slash", "", "", "weights.bin", False)
    assert [r["is_valid"] for r in results] == [False, True, False, False, False]
    assert ".onnx" in results[3]["message"]
    assert "eval split" in results[4]["message"]

    results = submit.validate_submission("org/m", "main", "alice", "", True)
    assert all(r["is_valid"] for r in results)  # empty path means model.onnx


def test_submission_requires_the_eval_split_declaration(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    with pytest.raises(gr.Error, match="eval split"):
        submit.add_new_eval("org/m", "main", "alice")
    assert "entry" not in captured


def test_submission_records_declaration_and_parameter_count(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    submit.add_new_eval("org/m", "main", "alice", no_eval_training=True)

    assert captured["entry"]["no_eval_training"] is True
    expected = submit.count_parameters(str(tmp_path / "dl.onnx"))
    assert captured["entry"]["n_parameters"] == expected


class _NotFound(submit.HfHubHTTPError):
    def __init__(self, status=404):
        Exception.__init__(self, f"HTTP {status}")
        self.response = type("Response", (), {"status_code": status})()


def test_hub_account_exists_finds_users_and_organisations(monkeypatch):
    users, orgs = {"alice"}, {"proxima-fusion"}

    def lookup(names):
        def fake(name, token):
            assert token is False
            if name not in names:
                raise _NotFound()

        return fake

    monkeypatch.setattr(submit.API, "get_user_overview", lookup(users))
    monkeypatch.setattr(submit.API, "get_organization_overview", lookup(orgs))

    assert submit.hub_account_exists("alice")
    assert submit.hub_account_exists("proxima-fusion")
    assert not submit.hub_account_exists("nobody-here")


def test_hub_account_exists_raises_on_hub_errors(monkeypatch):
    def outage(name, token):
        raise _NotFound(status=503)

    monkeypatch.setattr(submit.API, "get_user_overview", outage)
    with pytest.raises(submit.HfHubHTTPError):
        submit.hub_account_exists("alice")


def test_submission_rejects_unknown_hub_account(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(submit, "hub_account_exists", lambda name: False)

    with pytest.raises(gr.Error, match="not a Hugging Face user"):
        submit.add_new_eval("org/m", "main", "ghost", no_eval_training=True)
    assert "entry" not in captured
    assert "download" not in captured  # rejected before touching the model


def test_submission_reports_account_lookup_failure(tmp_path, monkeypatch):
    captured = {}
    _wire(monkeypatch, tmp_path, captured)

    def boom(name):
        raise RuntimeError("hub down")

    monkeypatch.setattr(submit, "hub_account_exists", boom)

    with pytest.raises(gr.Error, match="retry"):
        submit.add_new_eval("org/m", "main", "alice", no_eval_training=True)
    assert "entry" not in captured
