import json

from leaderboard.src.submission import requests_dataset


def test_request_path_encodes_identity():
    assert (
        requests_dataset.request_path("org/m", "abc123", "models/best.onnx")
        == "org/m_eval_request_abc123_models-best.json"
    )


def test_request_path_defaults_match_default_onnx():
    assert (
        requests_dataset.request_path("org/m", "abc123", "model.onnx")
        == "org/m_eval_request_abc123_model.json"
    )


def test_list_request_files_keeps_only_json(monkeypatch):
    monkeypatch.setattr(
        requests_dataset.API,
        "list_repo_files",
        lambda **kw: ["org/a.json", "README.md", ".gitattributes", "org/b.json"],
    )
    assert requests_dataset.list_request_files() == ["org/a.json", "org/b.json"]


def test_submission_exists_matches_built_path(monkeypatch):
    path = requests_dataset.request_path("org/m", "abc123", "model.onnx")
    monkeypatch.setattr(requests_dataset, "list_request_files", lambda: [path])
    assert requests_dataset.submission_exists("org/m", "abc123", "model.onnx")
    assert not requests_dataset.submission_exists("org/m", "def456", "model.onnx")


def test_iter_requests_reads_one_snapshot_and_skips_malformed(monkeypatch, tmp_path):
    (tmp_path / "org").mkdir()
    (tmp_path / "org" / "good.json").write_text(
        json.dumps({"model": "org/m", "status": "PENDING"})
    )
    (tmp_path / "org" / "bad.json").write_text("{not json")
    calls = []

    def fake_snapshot(**kw):
        calls.append(kw)
        return str(tmp_path)

    monkeypatch.setattr(requests_dataset, "snapshot_download", fake_snapshot)

    records = list(requests_dataset.iter_requests())
    assert records == [{"model": "org/m", "status": "PENDING"}]
    assert len(calls) == 1
    assert calls[0]["allow_patterns"] == "*.json"
