import json

from src.backend import runner


def test_run_once_pins_commit_and_passes_onnx_path(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue" / "org"
    queue.mkdir(parents=True)
    (queue / "m_eval_request_main.json").write_text(json.dumps({
        "model": "org/m", "revision": "main",
        "commit_hash": "deadbeefcafebabe", "onnx_path": "models/best.onnx",
        "submitter": "alice", "stochastic": False, "k_samples": 1,
        "submitted_time": "2026-06-11T00:00:00Z", "n_instances": -1,
        "status": "PENDING",
    }))

    monkeypatch.setattr(runner, "EVAL_REQUESTS_PATH", str(tmp_path / "eval-queue"))
    monkeypatch.setattr(runner, "snapshot_download", lambda **kw: None)
    monkeypatch.setattr(runner, "OnnxModel", lambda path: object())
    monkeypatch.setattr(runner.API, "upload_file", lambda **kw: None)

    captured = {}

    def fake_download(repo_id, filename, revision, local_dir, token):
        captured["filename"] = filename
        captured["revision"] = revision
        return str(tmp_path / "x.onnx")

    monkeypatch.setattr(runner, "hf_hub_download", fake_download)

    def fake_eval(**kw):
        captured["commit_hash"] = kw["commit_hash"]
        captured["onnx_path"] = kw["onnx_path"]
        return {"model": "org/m", "revision": "main", "status": "FINISHED"}

    monkeypatch.setattr(runner.beval, "evaluate_model", fake_eval)

    runner.run_once()

    assert captured["revision"] == "deadbeefcafebabe"     # pinned SHA, not "main"
    assert captured["filename"] == "models/best.onnx"
    assert captured["commit_hash"] == "deadbeefcafebabe"
    assert captured["onnx_path"] == "models/best.onnx"


def test_push_result_path_disambiguates_by_commit_and_onnx(monkeypatch):
    captured = {}
    monkeypatch.setattr(runner.API, "upload_file",
                        lambda **kw: captured.update(path_in_repo=kw["path_in_repo"]))
    runner._push_result({
        "model": "org/m", "revision": "main",
        "commit_hash": "deadbeefcafebabe", "onnx_path": "models/best.onnx",
        "status": "FINISHED",
    })
    assert captured["path_in_repo"] == "org/m_deadbeefcafebabe_models-best.json"


def test_push_result_falls_back_for_legacy_record(monkeypatch):
    captured = {}
    monkeypatch.setattr(runner.API, "upload_file",
                        lambda **kw: captured.update(path_in_repo=kw["path_in_repo"]))
    runner._push_result({"model": "org/m", "revision": "main", "status": "FINISHED"})
    # No commit_hash/onnx_path -> falls back to revision + default model file slug.
    assert captured["path_in_repo"] == "org/m_main_model.json"
