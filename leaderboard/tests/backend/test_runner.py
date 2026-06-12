import datetime
import json

import pytest

from src.backend import runner

UTC = datetime.timezone.utc


# --- _run_eval: download/eval argument wiring (was test_run_once_pins_commit...) ---

def test_run_eval_pins_commit_and_passes_onnx_path(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(runner, "OnnxModel", lambda path: object())

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

    runner._run_eval({
        "model": "org/m", "revision": "main",
        "commit_hash": "deadbeefcafebabe", "onnx_path": "models/best.onnx",
        "submitter": "alice", "stochastic": False, "k_samples": 1,
        "submitted_time": "2026-06-11T00:00:00Z", "n_instances": -1,
    })

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


# --- per-eval timeout (item 1) ---
# The eval runs in a spawned child, so injected callables must be importable at
# module level (spawn pickles them by qualified name; monkeypatch wouldn't cross).

def _ok_eval(req):
    return {"model": req["model"], "status": "FINISHED"}


def _hang_eval(req):
    import time
    time.sleep(30)


def _boom_eval(req):
    raise ValueError("bad model")


def test_evaluate_with_timeout_returns_record():
    rec = runner._evaluate_with_timeout({"model": "org/m"}, timeout=30, _fn=_ok_eval)
    assert rec == {"model": "org/m", "status": "FINISHED"}


def test_evaluate_with_timeout_kills_hung_eval():
    with pytest.raises(TimeoutError):
        runner._evaluate_with_timeout({"model": "org/m"}, timeout=0.3, _fn=_hang_eval)


def test_evaluate_with_timeout_propagates_eval_error():
    with pytest.raises(RuntimeError, match="bad model"):
        runner._evaluate_with_timeout({"model": "org/m"}, timeout=30, _fn=_boom_eval)


# --- claim stamps attempts/started_at/worker_id (items 2 & 3) ---

def test_run_once_claims_with_attempt_and_timestamp(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue" / "org"
    queue.mkdir(parents=True)
    (queue / "m.json").write_text(json.dumps({
        "model": "org/m", "revision": "main", "commit_hash": "abc",
        "onnx_path": "model.onnx", "status": "PENDING",
    }))

    monkeypatch.setattr(runner, "EVAL_REQUESTS_PATH", str(tmp_path / "eval-queue"))
    monkeypatch.setattr(runner, "snapshot_download", lambda **kw: None)
    monkeypatch.setattr(runner, "_now", lambda: datetime.datetime(2026, 6, 12, tzinfo=UTC))
    monkeypatch.setattr(runner, "_worker_id", lambda: "host:123")
    pushed = []
    monkeypatch.setattr(runner, "_push_request", lambda req, rel: pushed.append(dict(req)))
    monkeypatch.setattr(runner, "_push_result", lambda rec: None)
    monkeypatch.setattr(runner, "_evaluate_with_timeout",
                        lambda req, timeout=None: {"model": "org/m", "status": "FINISHED"})

    runner.run_once()

    claim = pushed[0]
    assert claim["status"] == "RUNNING"
    assert claim["attempts"] == 1
    assert claim["started_at"] == "2026-06-12T00:00:00+00:00"
    assert claim["worker_id"] == "host:123"
    assert pushed[-1]["status"] == "FINISHED"


def test_run_once_marks_failed_on_eval_error_and_continues(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue" / "org"
    queue.mkdir(parents=True)
    for name in ("a", "b"):
        (queue / f"{name}.json").write_text(json.dumps({
            "model": f"org/{name}", "revision": "main", "status": "PENDING",
        }))

    monkeypatch.setattr(runner, "EVAL_REQUESTS_PATH", str(tmp_path / "eval-queue"))
    monkeypatch.setattr(runner, "snapshot_download", lambda **kw: None)
    pushed = []
    monkeypatch.setattr(runner, "_push_request", lambda req, rel: pushed.append(dict(req)))

    def boom(req, timeout=None):
        raise TimeoutError("Evaluation exceeded 1800s")

    monkeypatch.setattr(runner, "_evaluate_with_timeout", boom)

    runner.run_once()

    failed = [p for p in pushed if p["status"] == "FAILED"]
    assert {p["model"] for p in failed} == {"org/a", "org/b"}  # both processed, loop continued
    assert all("exceeded" in p["error"] for p in failed)


# --- reaper for stuck RUNNING (items 2 & 3) ---

def _write_running(queue_dir, name, started_at, attempts):
    org = queue_dir / "org"
    org.mkdir(parents=True, exist_ok=True)
    (org / f"{name}.json").write_text(json.dumps({
        "model": f"org/{name}", "revision": "main", "status": "RUNNING",
        "attempts": attempts, "started_at": started_at,
    }))


def test_reap_stale_requeues_running_under_attempt_cap(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue"
    _write_running(queue, "m", "2026-06-12T00:00:00+00:00", attempts=1)

    monkeypatch.setattr(runner.config, "LEASE_TTL", 3600)
    monkeypatch.setattr(runner.config, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(runner, "_now", lambda: datetime.datetime(2026, 6, 12, 2, 0, tzinfo=UTC))
    pushed = []
    monkeypatch.setattr(runner, "_push_request", lambda req, rel: pushed.append(dict(req)))

    runner._reap_stale(str(queue))

    assert len(pushed) == 1
    assert pushed[0]["status"] == "PENDING"


def test_reap_stale_fails_poison_pill_at_attempt_cap(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue"
    _write_running(queue, "m", "2026-06-12T00:00:00+00:00", attempts=3)

    monkeypatch.setattr(runner.config, "LEASE_TTL", 3600)
    monkeypatch.setattr(runner.config, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(runner, "_now", lambda: datetime.datetime(2026, 6, 12, 2, 0, tzinfo=UTC))
    pushed = []
    monkeypatch.setattr(runner, "_push_request", lambda req, rel: pushed.append(dict(req)))

    runner._reap_stale(str(queue))

    assert len(pushed) == 1
    assert pushed[0]["status"] == "FAILED"
    assert "attempt" in pushed[0]["error"].lower()


def test_reap_stale_ignores_fresh_running(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue"
    _write_running(queue, "m", "2026-06-12T01:59:00+00:00", attempts=1)  # 1 min ago

    monkeypatch.setattr(runner.config, "LEASE_TTL", 3600)
    monkeypatch.setattr(runner.config, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(runner, "_now", lambda: datetime.datetime(2026, 6, 12, 2, 0, tzinfo=UTC))
    pushed = []
    monkeypatch.setattr(runner, "_push_request", lambda req, rel: pushed.append(dict(req)))

    runner._reap_stale(str(queue))

    assert pushed == []
