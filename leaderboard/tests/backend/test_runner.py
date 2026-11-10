import datetime as dt
import json
import sys
import time

import pytest

from leaderboard.src.backend import evaluate_one, runner
from tests.backend.conftest import build_dummy_onnx

UTC = dt.timezone.utc


# --- _run_eval: download/eval argument wiring (was test_run_once_pins_commit...) ---


def test_run_eval_pins_commit_and_records_onnx_path(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(runner.jax_cache, "enable", lambda: None)

    def fake_download(repo_id, filename, revision, local_dir, token):
        captured["filename"] = filename
        captured["revision"] = revision
        return build_dummy_onnx(tmp_path / "x.onnx")

    monkeypatch.setattr(runner, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        runner.evaluate, "evaluate_model", lambda **kw: ({"status": "FINISHED"}, [])
    )

    record, _rows, model_bytes = runner._run_eval(
        {
            "model": "org/m",
            "revision": "main",
            "commit_hash": "deadbeefcafebabe",
            "onnx_path": "models/best.onnx",
            "submitter": "alice",
            "stochastic": False,
            "k_samples": 1,
            "submitted_time": "2026-06-11T00:00:00Z",
            "n_instances": -1,
        }
    )

    assert captured["revision"] == "deadbeefcafebabe"  # pinned SHA, not "main"
    assert captured["filename"] == "models/best.onnx"
    assert record["commit_hash"] == "deadbeefcafebabe"
    assert record["onnx_path"] == "models/best.onnx"
    assert record["model"] == "org/m"
    assert record["submitter"] == "alice"
    # Read before the download temp dir is torn down.
    assert model_bytes == (tmp_path / "x.onnx").read_bytes()


@pytest.mark.parametrize(
    ("declared", "queued", "expected"),
    [
        ("fixed_shape", ["fixed_shape", "variable_shape"], ["fixed_shape"]),
        (
            "fixed_shape,variable_shape",
            ["fixed_shape"],
            ["fixed_shape", "variable_shape"],
        ),
    ],
)
def test_run_eval_scores_only_metadata_declared_tracks(
    tmp_path, monkeypatch, declared, queued, expected
):
    captured = {}
    monkeypatch.setattr(runner.jax_cache, "enable", lambda: None)
    monkeypatch.setattr(
        runner,
        "hf_hub_download",
        lambda **kw: build_dummy_onnx(tmp_path / "x.onnx", supported_tracks=declared),
    )

    def fake_eval(**kw):
        captured["tracks"] = kw["tracks"]
        return {"model": "org/m", "status": "FINISHED"}, []

    monkeypatch.setattr(runner.evaluate, "evaluate_model", fake_eval)

    runner._run_eval({"model": "org/m", "revision": "main", "tracks": queued})

    assert captured["tracks"] == expected


def test_push_result_path_disambiguates_by_commit_and_onnx(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        runner.API,
        "upload_file",
        lambda **kw: captured.update(path_in_repo=kw["path_in_repo"]),
    )
    runner._push_result(
        {
            "model": "org/m",
            "revision": "main",
            "commit_hash": "deadbeefcafebabe",
            "onnx_path": "models/best.onnx",
            "status": "FINISHED",
        }
    )
    assert captured["path_in_repo"] == "org/m_deadbeefcafebabe_models-best.json"


def test_push_result_falls_back_for_legacy_record(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        runner.API,
        "upload_file",
        lambda **kw: captured.update(path_in_repo=kw["path_in_repo"]),
    )
    runner._push_result({"model": "org/m", "revision": "main", "status": "FINISHED"})
    # No commit_hash/onnx_path -> falls back to revision + default model file slug.
    assert captured["path_in_repo"] == "org/m_main_model.json"


def test_push_result_parquet_uses_sibling_path_and_uploads_bytes(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        runner.API,
        "upload_file",
        lambda **kw: captured.update(
            path=kw["path_in_repo"], repo=kw["repo_id"], obj=kw["path_or_fileobj"]
        ),
    )
    runner._push_result_parquet(
        {
            "model": "org/m",
            "revision": "main",
            "commit_hash": "deadbeefcafebabe",
            "onnx_path": "models/best.onnx",
            "status": "FINISHED",
        },
        b"PARQUET-BYTES",
    )
    # Same disambiguation as the JSON summary, .parquet extension.
    assert captured["path"] == "org/m_deadbeefcafebabe_models-best.parquet"
    assert captured["repo"] == runner.RESULTS_REPO
    assert captured["obj"] == b"PARQUET-BYTES"


# --- per-eval timeout (item 1) ---
# The eval runs in a plain freshly-exec'd subprocess
# (`python -m leaderboard.src.backend.evaluate_one <request_path> <result_path>`),
# so multiprocessing's spawn/forkserver main-module bootstrap machinery is never
# touched. The subprocess is its own session leader, so a
# timeout kills the whole process group (the eval plus any JAX/DESC helpers it spawned).
# These stubs stand in for that command: same contract (read the request from argv[1],
# write the result to argv[2]), without the heavy eval.
_STUB_OK = [
    sys.executable,
    "-c",
    "import sys, json; req = json.load(open(sys.argv[1])); "
    "json.dump({'model': req['model'], 'status': 'FINISHED'}, open(sys.argv[2], 'w'))",
]
_STUB_HANG = [sys.executable, "-c", "import time; time.sleep(30)"]
_STUB_ERR = [
    sys.executable,
    "-c",
    "import sys; sys.stderr.write('bad model'); sys.exit(1)",
]
_STUB_SEGV = [
    sys.executable,
    "-c",
    "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)",
]


def test_evaluate_with_timeout_returns_record(monkeypatch):
    monkeypatch.setattr(runner, "_EVAL_CMD", _STUB_OK)
    rec, parquet, model = runner._evaluate_with_timeout({"model": "org/m"}, timeout=30)
    assert rec == {"model": "org/m", "status": "FINISHED"}
    # The stub writes no parquet or ONNX sibling, so there is nothing to upload.
    assert parquet is None
    assert model is None


_STUB_DEADLINE = [
    sys.executable,
    "-c",
    "import sys, json, os; "
    "json.dump({'deadline': os.environ.get('COILBENCH_EVAL_DEADLINE')}, "
    "open(sys.argv[2], 'w'))",
]


def test_evaluate_with_timeout_passes_absolute_deadline_to_child(monkeypatch):
    # The child can't share the parent's monotonic clock, so the parent hands it
    # an absolute wall-clock deadline (now + timeout) for time-left logging.
    monkeypatch.setattr(runner, "_EVAL_CMD", _STUB_DEADLINE)
    before = time.time()
    rec, _parquet, _model = runner._evaluate_with_timeout(
        {"model": "org/m"}, timeout=120
    )
    assert rec["deadline"] is not None
    deadline = float(rec["deadline"])
    assert before + 120 <= deadline <= time.time() + 120


def test_evaluate_with_timeout_kills_hung_eval(monkeypatch):
    monkeypatch.setattr(runner, "_EVAL_CMD", _STUB_HANG)
    with pytest.raises(TimeoutError):
        runner._evaluate_with_timeout({"model": "org/m"}, timeout=0.5)


def test_evaluate_with_timeout_kills_whole_process_group(tmp_path, monkeypatch):
    # A hung eval may have spawned helper processes (JAX/DESC). Killing only the direct
    # child would orphan them; the session leader + killpg takes out the whole group.
    # The stub spawns a grandchild that writes `sentinel` after a delay, then hangs; if
    # the group kill works the grandchild dies first and the sentinel never appears.
    sentinel = tmp_path / "grandchild.txt"
    stub_script = tmp_path / "group_stub.py"
    stub_script.write_text(
        "import sys, json, subprocess, time\n"
        "sentinel = json.load(open(sys.argv[1]))['sentinel']\n"
        "subprocess.Popen([sys.executable, '-c',\n"
        "    'import sys, time; time.sleep(1.5); '\n"
        '    \'open(sys.argv[1], "w").write("x")\',\n'
        "    sentinel])\n"
        "time.sleep(30)\n"
    )
    monkeypatch.setattr(runner, "_EVAL_CMD", [sys.executable, str(stub_script)])
    with pytest.raises(TimeoutError):
        runner._evaluate_with_timeout(
            {"model": "org/m", "sentinel": str(sentinel)}, timeout=0.3
        )
    time.sleep(2.0)  # past the grandchild's write time, had it survived the kill
    assert not sentinel.exists()


def test_evaluate_with_timeout_reports_segfault(monkeypatch):
    monkeypatch.setattr(runner, "_EVAL_CMD", _STUB_SEGV)
    with pytest.raises(RuntimeError, match="(?i)segmentation"):
        runner._evaluate_with_timeout({"model": "org/m"}, timeout=30)


def test_evaluate_with_timeout_propagates_subprocess_error(capfd, monkeypatch):
    monkeypatch.setattr(runner, "_EVAL_CMD", _STUB_ERR)
    with pytest.raises(RuntimeError, match="exited 1"):
        runner._evaluate_with_timeout({"model": "org/m"}, timeout=30)
    # The child's diagnostics aren't folded into the exception any more; they go
    # straight to the console (inherited stderr).
    assert "bad model" in capfd.readouterr().err


def test_evaluate_with_timeout_streams_child_output(capfd, monkeypatch):
    # The child inherits the worker's stdout/stderr, so the eval's progress (and an
    # OOM-kill that leaves no exception) is visible in the console, not discarded.
    stub = [
        sys.executable,
        "-c",
        "import sys, json; print('hello-from-child'); sys.stdout.flush(); "
        "json.dump({'model': 'org/m', 'status': 'FINISHED'}, open(sys.argv[2], 'w'))",
    ]
    monkeypatch.setattr(runner, "_EVAL_CMD", stub)
    rec, _parquet, _model = runner._evaluate_with_timeout(
        {"model": "org/m"}, timeout=30
    )
    assert rec == {"model": "org/m", "status": "FINISHED"}
    assert "hello-from-child" in capfd.readouterr().out


def test_evaluate_with_timeout_runs_inline_when_disabled(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_run_eval",
        lambda req: ({"model": req["model"], "status": "FINISHED"}, [], b"ONNX"),
    )
    rec, parquet, model = runner._evaluate_with_timeout({"model": "org/m"}, timeout=0)
    assert rec == {"model": "org/m", "status": "FINISHED"}
    assert parquet is None  # no rows -> no parquet
    assert model == b"ONNX"


def test_eval_one_reads_request_file_and_writes_record(tmp_path, monkeypatch):
    # The subprocess entrypoint: request JSON at argv[1] -> record JSON at argv[2].
    monkeypatch.setattr(
        runner,
        "_run_eval",
        lambda req: ({"model": req["model"], "status": "FINISHED"}, [], b"ONNX"),
    )
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps({"model": "org/m"}))
    out = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["eval_one", str(req_path), str(out)])

    evaluate_one.main()

    assert json.loads(out.read_text()) == {"model": "org/m", "status": "FINISHED"}
    assert (tmp_path / "result.onnx").read_bytes() == b"ONNX"
    # No rows -> no parquet sibling written.
    assert not (tmp_path / "result.parquet").exists()


def test_eval_one_writes_parquet_sibling_when_rows_present(tmp_path, monkeypatch):
    import pandas as pd

    rows = [
        {
            "track": "fixed_shape",
            "stratum": "tight",
            "boundary_id": "b0",
            "score": 0.5,
            "strictly_feasible": True,
        }
    ]
    monkeypatch.setattr(
        runner,
        "_run_eval",
        lambda req: ({"model": req["model"], "status": "FINISHED"}, rows, b"ONNX"),
    )
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps({"model": "org/m"}))
    out = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["eval_one", str(req_path), str(out)])

    evaluate_one.main()

    parquet = tmp_path / "result.parquet"
    assert parquet.exists()
    assert pd.read_parquet(parquet).iloc[0]["boundary_id"] == "b0"


# --- claim stamps attempts/started_at/worker_id (items 2 & 3) ---


def test_run_once_claims_with_attempt_and_timestamp(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue" / "org"
    queue.mkdir(parents=True)
    (queue / "m.json").write_text(
        json.dumps(
            {
                "model": "org/m",
                "revision": "main",
                "commit_hash": "abc",
                "onnx_path": "model.onnx",
                "status": "PENDING",
            }
        )
    )

    monkeypatch.setattr(runner, "EVAL_REQUESTS_PATH", str(tmp_path / "eval-queue"))
    monkeypatch.setattr(runner, "snapshot_download", lambda **kw: None)
    monkeypatch.setattr(runner, "_now", lambda: dt.datetime(2026, 6, 12, tzinfo=UTC))
    monkeypatch.setattr(runner, "_worker_id", lambda: "host:123")
    pushed = []
    monkeypatch.setattr(
        runner, "_push_request", lambda req, rel: pushed.append(dict(req))
    )
    monkeypatch.setattr(runner, "_push_result", lambda rec: None)
    monkeypatch.setattr(runner, "_push_result_parquet", lambda rec, blob: None)
    monkeypatch.setattr(
        runner,
        "_evaluate_with_timeout",
        lambda req, timeout=None: (
            {"model": "org/m", "status": "FINISHED"},
            None,
            None,
        ),
    )

    runner.run_once()

    claim = pushed[0]
    assert claim["status"] == "RUNNING"
    assert claim["attempts"] == 1
    assert claim["started_at"] == "2026-06-12T00:00:00+00:00"
    assert claim["worker_id"] == "host:123"
    assert pushed[-1]["status"] == "FINISHED"


def test_run_once_pushes_parquet_when_present(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue" / "org"
    queue.mkdir(parents=True)
    (queue / "m.json").write_text(
        json.dumps(
            {
                "model": "org/m",
                "revision": "main",
                "status": "PENDING",
            }
        )
    )
    monkeypatch.setattr(runner, "EVAL_REQUESTS_PATH", str(tmp_path / "eval-queue"))
    monkeypatch.setattr(runner, "snapshot_download", lambda **kw: None)
    monkeypatch.setattr(runner, "_push_request", lambda req, rel: None)
    monkeypatch.setattr(runner, "_push_result", lambda rec: None)
    pushed_parquet = []
    monkeypatch.setattr(
        runner, "_push_result_parquet", lambda rec, blob: pushed_parquet.append(blob)
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_with_timeout",
        lambda req, timeout=None: (
            {"model": "org/m", "status": "FINISHED"},
            b"P",
            None,
        ),
    )

    runner.run_once()

    assert pushed_parquet == [b"P"]


def test_run_once_skips_parquet_when_absent(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue" / "org"
    queue.mkdir(parents=True)
    (queue / "m.json").write_text(
        json.dumps(
            {
                "model": "org/m",
                "revision": "main",
                "status": "PENDING",
            }
        )
    )
    monkeypatch.setattr(runner, "EVAL_REQUESTS_PATH", str(tmp_path / "eval-queue"))
    monkeypatch.setattr(runner, "snapshot_download", lambda **kw: None)
    monkeypatch.setattr(runner, "_push_request", lambda req, rel: None)
    monkeypatch.setattr(runner, "_push_result", lambda rec: None)
    pushed_parquet = []
    monkeypatch.setattr(
        runner, "_push_result_parquet", lambda rec, blob: pushed_parquet.append(blob)
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_with_timeout",
        lambda req, timeout=None: (
            {"model": "org/m", "status": "FINISHED"},
            None,
            None,
        ),
    )

    runner.run_once()

    assert pushed_parquet == []


def test_run_once_marks_failed_on_eval_error_and_continues(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue" / "org"
    queue.mkdir(parents=True)
    for name in ("a", "b"):
        (queue / f"{name}.json").write_text(
            json.dumps(
                {
                    "model": f"org/{name}",
                    "revision": "main",
                    "status": "PENDING",
                }
            )
        )

    monkeypatch.setattr(runner, "EVAL_REQUESTS_PATH", str(tmp_path / "eval-queue"))
    monkeypatch.setattr(runner, "snapshot_download", lambda **kw: None)
    pushed = []
    monkeypatch.setattr(
        runner, "_push_request", lambda req, rel: pushed.append(dict(req))
    )

    def boom(req, timeout=None):
        raise TimeoutError("Evaluation exceeded 1800s")

    monkeypatch.setattr(runner, "_evaluate_with_timeout", boom)

    runner.run_once()

    failed = [p for p in pushed if p["status"] == "FAILED"]
    assert {p["model"] for p in failed} == {
        "org/a",
        "org/b",
    }  # both processed, loop continued
    assert all("exceeded" in p["error"] for p in failed)


# --- reaper for stuck RUNNING (items 2 & 3) ---


def _write_running(queue_dir, name, started_at, attempts):
    org = queue_dir / "org"
    org.mkdir(parents=True, exist_ok=True)
    (org / f"{name}.json").write_text(
        json.dumps(
            {
                "model": f"org/{name}",
                "revision": "main",
                "status": "RUNNING",
                "attempts": attempts,
                "started_at": started_at,
            }
        )
    )


def test_reap_stale_requeues_running_under_attempt_cap(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue"
    _write_running(queue, "m", "2026-06-12T00:00:00+00:00", attempts=1)

    monkeypatch.setattr(runner.config, "LEASE_TTL", 3600)
    monkeypatch.setattr(runner.config, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(
        runner, "_now", lambda: dt.datetime(2026, 6, 12, 2, 0, tzinfo=UTC)
    )
    pushed = []
    monkeypatch.setattr(
        runner, "_push_request", lambda req, rel: pushed.append(dict(req))
    )

    runner._reap_stale(str(queue))

    assert len(pushed) == 1
    assert pushed[0]["status"] == "PENDING"


def test_reap_stale_fails_poison_pill_at_attempt_cap(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue"
    _write_running(queue, "m", "2026-06-12T00:00:00+00:00", attempts=3)

    monkeypatch.setattr(runner.config, "LEASE_TTL", 3600)
    monkeypatch.setattr(runner.config, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(
        runner, "_now", lambda: dt.datetime(2026, 6, 12, 2, 0, tzinfo=UTC)
    )
    pushed = []
    monkeypatch.setattr(
        runner, "_push_request", lambda req, rel: pushed.append(dict(req))
    )

    runner._reap_stale(str(queue))

    assert len(pushed) == 1
    assert pushed[0]["status"] == "FAILED"
    assert "attempt" in pushed[0]["error"].lower()


def test_reap_stale_ignores_fresh_running(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue"
    _write_running(queue, "m", "2026-06-12T01:59:00+00:00", attempts=1)  # 1 min ago

    monkeypatch.setattr(runner.config, "LEASE_TTL", 3600)
    monkeypatch.setattr(runner.config, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(
        runner, "_now", lambda: dt.datetime(2026, 6, 12, 2, 0, tzinfo=UTC)
    )
    pushed = []
    monkeypatch.setattr(
        runner, "_push_request", lambda req, rel: pushed.append(dict(req))
    )

    runner._reap_stale(str(queue))

    assert pushed == []


# --- a single malformed request file must not wedge the whole poll loop ---


def test_iter_requests_skips_empty_and_malformed_files(tmp_path):
    org = tmp_path / "org"
    org.mkdir(parents=True)
    (org / "empty.json").write_text("")  # zero-byte file (the production poison pill)
    (org / "garbage.json").write_text("{not json")
    (org / "good.json").write_text(json.dumps({"model": "org/m", "status": "PENDING"}))

    found = dict(runner._iter_requests(str(tmp_path)))

    assert [req["model"] for req in found.values()] == ["org/m"]


def test_run_once_processes_valid_requests_despite_poison_file(tmp_path, monkeypatch):
    queue = tmp_path / "eval-queue" / "org"
    queue.mkdir(parents=True)
    (queue / "empty.json").write_text("")  # would crash json.load
    (queue / "m.json").write_text(
        json.dumps(
            {
                "model": "org/m",
                "revision": "main",
                "status": "PENDING",
            }
        )
    )

    monkeypatch.setattr(runner, "EVAL_REQUESTS_PATH", str(tmp_path / "eval-queue"))
    monkeypatch.setattr(runner, "snapshot_download", lambda **kw: None)
    pushed = []
    monkeypatch.setattr(
        runner, "_push_request", lambda req, rel: pushed.append(dict(req))
    )
    monkeypatch.setattr(runner, "_push_result", lambda rec: None)
    monkeypatch.setattr(runner, "_push_result_parquet", lambda rec, blob: None)
    monkeypatch.setattr(
        runner,
        "_evaluate_with_timeout",
        lambda req, timeout=None: (
            {"model": "org/m", "status": "FINISHED"},
            None,
            None,
        ),
    )

    runner.run_once()

    assert pushed[-1]["status"] == "FINISHED"  # valid request still processed
