# Direct requests-dataset queue check — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Space-side local `eval-queue/` snapshot; serve duplicate-detection and the queue tables by reading the HF `requests` dataset directly.

**Architecture:** A new `src/submission/requests_dataset.py` module is the single place that talks to the requests dataset (via `HfApi.list_repo_files` + `hf_hub_download`). Submit's duplicate check becomes a live `list_repo_files` membership test on the deterministic request path; the queue tables read records from `iter_requests()` at boot. The backend `runner.py` is untouched.

**Tech Stack:** Python 3.13, `huggingface_hub` (`HfApi`, `hf_hub_download`), pandas, Gradio v6, pytest.

## Global Constraints

- Backend `runner.py` MUST remain unchanged — it keeps its own per-poll `snapshot_download`.
- `EVAL_REQUESTS_PATH` MUST stay defined in `src/envs.py` (the backend still uses it); only the Space stops referencing it.
- Dedup identity is the request file path `{org}/{model_path}_eval_request_{commit_hash}_{onnx_slug}.json` where `onnx_slug = onnx_path.replace("/", "-").removesuffix(".onnx")` — copied verbatim from the current `submit.py:84-87`. Submit-write and dedup-check MUST use the same builder.
- All Hub calls pass `token=TOKEN` and `repo_type="dataset"` against `QUEUE_REPO`.
- Dedup failure (Hub error during submit) MUST fail closed — reject the submission, never queue unchecked.

---

### Task 1: `requests_dataset` module

**Files:**
- Create: `src/submission/requests_dataset.py`
- Test: `tests/submission/test_requests_dataset.py`

**Interfaces:**
- Consumes: `src.envs.API`, `src.envs.QUEUE_REPO`, `src.envs.TOKEN`; `huggingface_hub.hf_hub_download`.
- Produces:
  - `request_path(model: str, commit_hash: str, onnx_path: str) -> str`
  - `list_request_files() -> list[str]`
  - `submission_exists(model: str, commit_hash: str, onnx_path: str) -> bool`
  - `iter_requests() -> Iterator[dict]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/submission/test_requests_dataset.py
import json

from src.submission import requests_dataset


def test_request_path_encodes_identity():
    assert (
        requests_dataset.request_path("org/m", "abc123", "models/best.onnx")
        == "org/m_eval_request_abc123_models-best.onnx.json"
    )


def test_request_path_defaults_match_default_onnx():
    assert (
        requests_dataset.request_path("org/m", "abc123", "model.onnx")
        == "org/m_eval_request_abc123_model.json"
    )


def test_list_request_files_keeps_only_json(monkeypatch):
    monkeypatch.setattr(
        requests_dataset.API, "list_repo_files",
        lambda **kw: ["org/a.json", "README.md", ".gitattributes", "org/b.json"],
    )
    assert requests_dataset.list_request_files() == ["org/a.json", "org/b.json"]


def test_submission_exists_matches_built_path(monkeypatch):
    path = requests_dataset.request_path("org/m", "abc123", "model.onnx")
    monkeypatch.setattr(requests_dataset, "list_request_files", lambda: [path])
    assert requests_dataset.submission_exists("org/m", "abc123", "model.onnx")
    assert not requests_dataset.submission_exists("org/m", "def456", "model.onnx")


def test_iter_requests_skips_malformed(monkeypatch, tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"model": "org/m", "status": "PENDING"}))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")

    monkeypatch.setattr(requests_dataset, "list_request_files", lambda: ["org/good.json", "org/bad.json"])
    downloads = {"org/good.json": str(good), "org/bad.json": str(bad)}
    monkeypatch.setattr(requests_dataset, "hf_hub_download", lambda **kw: downloads[kw["filename"]])

    records = list(requests_dataset.iter_requests())
    assert records == [{"model": "org/m", "status": "PENDING"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/submission/test_requests_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.submission.requests_dataset'`

- [ ] **Step 3: Write the module**

```python
# src/submission/requests_dataset.py
import json
import logging
from collections.abc import Iterator

from huggingface_hub import hf_hub_download

from src.envs import API, QUEUE_REPO, TOKEN

logger = logging.getLogger(__name__)


def request_path(model: str, commit_hash: str, onnx_path: str) -> str:
    """Canonical path of a request file within the requests dataset repo.

    The filename encodes the dedup identity, so existence of this path *is*
    the duplicate check.
    """
    user_name, model_path = model.split("/", 1)
    onnx_slug = onnx_path.replace("/", "-").removesuffix(".onnx")
    return f"{user_name}/{model_path}_eval_request_{commit_hash}_{onnx_slug}.json"


def list_request_files() -> list[str]:
    """All request JSON paths currently in the requests dataset."""
    files = API.list_repo_files(repo_id=QUEUE_REPO, repo_type="dataset", token=TOKEN)
    return [f for f in files if f.endswith(".json")]


def submission_exists(model: str, commit_hash: str, onnx_path: str) -> bool:
    """True if an identical (model, commit, file) request is already queued."""
    return request_path(model, commit_hash, onnx_path) in list_request_files()


def iter_requests() -> Iterator[dict]:
    """Yield each request record, downloading and parsing per file.

    A malformed or partially uploaded file is logged and skipped rather than
    aborting the whole queue view.
    """
    for path in list_request_files():
        try:
            local = hf_hub_download(repo_id=QUEUE_REPO, filename=path, repo_type="dataset", token=TOKEN)
            with open(local) as fp:
                yield json.load(fp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable request %s: %s", path, exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/submission/test_requests_dataset.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/submission/requests_dataset.py tests/submission/test_requests_dataset.py
git commit -m "Add requests_dataset module for direct queue reads"
```

---

### Task 2: Switch `submit.py` dedup + write to `requests_dataset`

**Files:**
- Modify: `src/submission/submit.py` (imports; dedup at `:50-52`; write/upload at `:84-98`)
- Modify: `tests/submission/test_submit.py` (rewrite `_wire` + duplicate test; add fail-closed test)

**Interfaces:**
- Consumes: `requests_dataset.submission_exists`, `requests_dataset.request_path` (Task 1).
- Produces: `add_new_eval(...)` unchanged signature and return contract.

- [ ] **Step 1: Rewrite the affected tests**

Replace the whole body of `tests/submission/test_submit.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/submission/test_submit.py -v`
Expected: FAIL — `test_submission_fails_closed_when_queue_unreadable` errors (no fail-closed path yet), and the duplicate/path tests fail against the old dir-walk logic.

- [ ] **Step 3: Update the imports in `submit.py`**

Replace lines 7-11:

```python
from huggingface_hub import hf_hub_download

from src.display.formatting import styled_error, styled_message
from src.envs import API, QUEUE_REPO, TOKEN
from src.submission import requests_dataset
from src.submission.check_validity import validate_local_onnx
```

(Drops `EVAL_REQUESTS_PATH`, `already_submitted_models`, `submission_key`; adds `requests_dataset`. Add `import tempfile` to the stdlib import block at the top if not already present — it is not, so add it.)

- [ ] **Step 4: Replace the dedup block (was `submit.py:50-52`)**

```python
    try:
        duplicate = requests_dataset.submission_exists(model, commit_hash, onnx_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not check the queue for duplicates: %s", exc)
        return styled_error("Couldn't verify against the queue — please retry.")
    if duplicate:
        logger.info("Duplicate submission for %s@%s:%s", model, commit_hash, onnx_path)
        return styled_error(
            f"This exact model commit ({commit_hash}) and file ({onnx_path}) has already been submitted."
        )
```

- [ ] **Step 5: Replace the write/upload block (was `submit.py:84-98`)**

```python
    rel_path = requests_dataset.request_path(model, commit_hash, onnx_path)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(eval_entry, f, indent=2)
        tmp = f.name
    API.upload_file(
        path_or_fileobj=tmp,
        path_in_repo=rel_path,
        repo_id=QUEUE_REPO,
        repo_type="dataset",
        commit_message=f"Add {model} to eval queue",
    )
    os.remove(tmp)
    logger.info("Submission %s queued (PENDING) at %s", model, current_time)
    return styled_message("Your model has been submitted to the evaluation queue!")
```

Also delete the now-unused `user_name, model_path = model.split("/", 1)` line (was `submit.py:34`); the `logger.info` immediately after it does not reference those names.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/submission/test_submit.py -v`
Expected: PASS (5 passed)

- [ ] **Step 7: Commit**

```bash
git add src/submission/submit.py tests/submission/test_submit.py
git commit -m "Dedup submissions against the live requests dataset"
```

---

### Task 3: `get_evaluation_queue_df` takes records; `get_leaderboard_df` drops unused path

**Files:**
- Modify: `src/populate.py` (`get_evaluation_queue_df` `:34-77`; `get_leaderboard_df` signature `:14`; drop unused `json`/`os` imports `:1-3`)
- Modify: `tests/leaderboard/test_populate.py`

**Interfaces:**
- Consumes: request dicts (from `requests_dataset.iter_requests()`, wired in Task 4).
- Produces:
  - `get_evaluation_queue_df(records: list[dict], cols: list) -> list[pd.DataFrame]` (finished, running, pending)
  - `get_leaderboard_df(results_path: str, cols: list, benchmark_cols: list) -> pd.DataFrame`

- [ ] **Step 1: Rewrite the tests**

Replace the whole body of `tests/leaderboard/test_populate.py` with:

```python
from src.display.utils import BENCHMARK_COLS, COLS, EVAL_COLS
from src.populate import get_evaluation_queue_df, get_leaderboard_df


def test_empty_results_returns_columned_frame(tmp_path):
    # A fresh leaderboard (no result files) must yield an empty frame that still
    # carries the display columns, so the Leaderboard renders instead of crashing.
    df = get_leaderboard_df(str(tmp_path), COLS, BENCHMARK_COLS)
    assert df.empty
    assert list(df.columns) == COLS


def test_running_queue_surfaces_started_at():
    # A stuck RUNNING job should expose its claim time so it is visible in the UI.
    _finished, running, _pending = get_evaluation_queue_df(
        [{
            "model": "org/m", "revision": "main", "submitter": "a", "stochastic": False,
            "status": "RUNNING", "started_at": "2026-06-12T00:00:00+00:00",
        }],
        EVAL_COLS,
    )
    assert "2026-06-12T00:00:00+00:00" in running["started"].tolist()


def test_queue_buckets_by_status():
    records = [
        {"model": "org/p", "status": "PENDING"},
        {"model": "org/r", "status": "RUNNING"},
        {"model": "org/f", "status": "FINISHED"},
    ]
    finished, running, pending = get_evaluation_queue_df(records, EVAL_COLS)
    assert len(pending) == 1 and len(running) == 1 and len(finished) == 1


def test_queue_tolerates_missing_status():
    # A stray/partial record without a status must not crash the queue render.
    finished, running, pending = get_evaluation_queue_df([{"model": "org/x"}], EVAL_COLS)
    assert len(finished) == 0 and len(running) == 0 and len(pending) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/leaderboard/test_populate.py -v`
Expected: FAIL — `get_leaderboard_df` still requires the `requests_path` positional arg and `get_evaluation_queue_df` still expects a path string.

- [ ] **Step 3: Edit `src/populate.py`**

Replace the imports (lines 1-3) — drop `json` and `os`, which are no longer used:

```python
import logging

import pandas as pd
```

Change `get_leaderboard_df` to drop the unused `requests_path` parameter (its body never referenced it):

```python
def get_leaderboard_df(results_path: str, cols: list, benchmark_cols: list) -> pd.DataFrame:
    """Creates a dataframe from all the individual experiment results"""
    all_data_json = get_raw_eval_results(results_path)
```

(Leave the rest of `get_leaderboard_df` unchanged.)

Replace `get_evaluation_queue_df` (the whole function) with a record-driven version:

```python
def get_evaluation_queue_df(records: list[dict], cols: list) -> list[pd.DataFrame]:
    """Bucket request records into finished/running/pending dataframes."""
    all_evals = []
    for record in records:
        data = dict(record)  # don't mutate the caller's dict
        data[EvalQueueColumn.model.name] = make_clickable_model(data["model"])
        data[EvalQueueColumn.revision.name] = data.get("revision", "main")
        data[EvalQueueColumn.started.name] = data.get("started_at", "")
        all_evals.append(data)

    pending_list = [e for e in all_evals if e.get("status") in ["PENDING", "RERUN"]]
    running_list = [e for e in all_evals if e.get("status") == "RUNNING"]
    finished_list = [
        e for e in all_evals
        if str(e.get("status", "")).startswith("FINISHED") or e.get("status") == "PENDING_NEW_EVAL"
    ]
    df_pending = pd.DataFrame.from_records(pending_list, columns=cols)
    df_running = pd.DataFrame.from_records(running_list, columns=cols)
    df_finished = pd.DataFrame.from_records(finished_list, columns=cols)
    logger.debug(
        "Eval queue: %d finished, %d running, %d pending",
        len(finished_list), len(running_list), len(pending_list),
    )
    return df_finished[cols], df_running[cols], df_pending[cols]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/leaderboard/test_populate.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/populate.py tests/leaderboard/test_populate.py
git commit -m "Feed queue dataframe from request records, not a local dir"
```

---

### Task 4: Wire the Space to `requests_dataset`; remove dead code

**Files:**
- Modify: `app_leaderboard.py` (imports `:8,26-27`; queue snapshot `:51-57`; leaderboard/queue df build `:67-73`)
- Modify: `src/submission/check_validity.py` (remove `already_submitted_models` `:54-67` and `submission_key` `:49-51`; drop unused `json`/`os` imports if now unused)
- Modify: `tests/submission/test_check_validity.py` (drop tests for removed functions)

**Interfaces:**
- Consumes: `requests_dataset.iter_requests` (Task 1); `get_evaluation_queue_df(records, cols)` and `get_leaderboard_df(results_path, cols, benchmark_cols)` (Task 3).
- Produces: a Space that no longer reads or writes `EVAL_REQUESTS_PATH`.

- [ ] **Step 1: Trim `tests/submission/test_check_validity.py`**

Replace its body with only the surviving ONNX-validation tests:

```python
from src.submission.check_validity import validate_local_onnx
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/submission/test_check_validity.py -v`
Expected: PASS for the two kept tests is NOT the failure point — this step's failure is the import check in Step 4. Proceed; the suite-wide run in Step 6 is the gate.

- [ ] **Step 3: Remove dead code from `src/submission/check_validity.py`**

Delete `submission_key` (lines 49-51) and `already_submitted_models` (lines 54-67) entirely. After deletion, `json` and `os` are unused — remove `import json` and `import os` from the top of the file. Keep `check_model_card` and `validate_local_onnx`.

- [ ] **Step 4: Confirm nothing else imports the removed symbols**

Run: `grep -rn "already_submitted_models\|submission_key" src/ app.py app_leaderboard.py app_runner.py`
Expected: no matches (empty output).

- [ ] **Step 5: Edit `app_leaderboard.py`**

In the import block, drop the now-unused `snapshot_download` import only if no longer used — it is still used for `RESULTS_REPO`, so keep line 8. Update the `src.envs` import (line 26) to drop `EVAL_REQUESTS_PATH` and `QUEUE_REPO` (the queue snapshot is gone; `QUEUE_REPO` is no longer referenced in this file):

```python
from src.envs import API, EVAL_RESULTS_PATH, REPO_ID, RESULTS_REPO, TOKEN
```

Add `requests_dataset` to the submission import (line 28 area):

```python
from src.submission import requests_dataset
from src.submission.submit import add_new_eval
```

Delete the QUEUE_REPO snapshot block (lines 51-57):

```python
try:
    print(EVAL_REQUESTS_PATH)
    snapshot_download(
        repo_id=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30, token=TOKEN
    )
except Exception:
    restart_space()
```

Replace the leaderboard/queue build (lines 67-73) with:

```python
LEADERBOARD_DF = get_leaderboard_df(EVAL_RESULTS_PATH, COLS, BENCHMARK_COLS)

try:
    request_records = list(requests_dataset.iter_requests())
except Exception:
    restart_space()
    request_records = []

(
    finished_eval_queue_df,
    running_eval_queue_df,
    pending_eval_queue_df,
) = get_evaluation_queue_df(request_records, EVAL_COLS)
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS (all tests green).

- [ ] **Step 7: Smoke-check the Space module imports**

Run: `.venv/bin/python -c "import ast; ast.parse(open('app_leaderboard.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 8: Commit**

```bash
git add app_leaderboard.py src/submission/check_validity.py tests/submission/test_check_validity.py
git commit -m "Read the queue directly from HF; drop local eval-queue snapshot"
```

---

## Self-Review notes

- **Spec coverage:** new module (Task 1), submit dedup + fail-closed + shared path builder (Task 2), populate record-driven split (Task 3), app wiring + snapshot removal + dead-code removal + `envs` constant retained (Task 4). Backend `runner.py` deliberately untouched. All spec sections map to a task.
- **Type consistency:** `request_path`/`list_request_files`/`submission_exists`/`iter_requests` signatures are identical across Tasks 1, 2, 4; `get_evaluation_queue_df(records, cols)` and `get_leaderboard_df(results_path, cols, benchmark_cols)` consistent across Tasks 3 and 4.
- **Edge cases:** malformed JSON skip (Task 1 test), missing-status tolerance (Task 3 test), fail-closed dedup (Task 2 test).
