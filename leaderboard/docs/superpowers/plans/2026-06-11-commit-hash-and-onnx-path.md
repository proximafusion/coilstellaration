# Commit Hash + Custom ONNX Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record and pin each submission's resolved git commit SHA, allow a custom `.onnx` file path other than `model.onnx`, deduplicate submissions on (model, commit, file), and surface commit + file on the leaderboard.

**Architecture:** Two new fields (`commit_hash`, `onnx_path`) flow through the request JSON (`submit.py`) and the result record (`eval.py`). The SHA is resolved from `API.repo_info(...).sha` at submission and used as the download ref end-to-end (pinning). Deduplication keys on `model@commit_hash:onnx_path`. Two hidden-by-default leaderboard columns display the commit (clickable, short SHA) and the ONNX filename.

**Tech Stack:** Python, Gradio, `gradio_leaderboard`, `huggingface_hub`, pytest, onnx.

---

## File Structure

- `src/display/formatting.py` — add `make_clickable_commit` (commit → HF commit link).
- `src/display/utils.py` — add `commit` and `onnx_file` leaderboard columns.
- `src/leaderboard/read_evals.py` — populate the two new columns in `result_to_row`.
- `src/backend/eval.py` — `evaluate_model` accepts and records `commit_hash` + `onnx_path`.
- `src/backend/runner.py` — download at the pinned SHA and the custom filename; pass both to `evaluate_model`.
- `src/submission/check_validity.py` — rewrite `already_submitted_models`; add `submission_key`.
- `src/submission/submit.py` — resolve SHA, validate `.onnx` path, dedup, store both fields.
- `app_leaderboard.py` — add the "ONNX file path" textbox and thread it through `submit_eval`.
- Tests: `tests/display/test_formatting.py` (new), `tests/leaderboard/test_read_evals.py`, `tests/backend/test_eval.py`, `tests/backend/test_runner.py` (new), `tests/submission/test_check_validity.py`, `tests/submission/test_submit.py` (new).

---

## Task 1: Clickable commit helper

**Files:**
- Modify: `src/display/formatting.py`
- Create: `tests/display/__init__.py`
- Test: `tests/display/test_formatting.py`

- [ ] **Step 1: Create the test package marker**

Create `tests/display/__init__.py` as an empty file:

```python
```

- [ ] **Step 2: Write the failing test**

Create `tests/display/test_formatting.py`:

```python
from src.display.formatting import make_clickable_commit


def test_clickable_commit_shows_short_sha_and_links_to_commit():
    sha = "deadbeefcafebabe1234567890abcdef12345678"
    html = make_clickable_commit("org/model", sha)
    assert "deadbee" in html                       # short 7-char SHA shown
    assert f"huggingface.co/org/model/commit/{sha}" in html
    assert html.startswith("<a")


def test_clickable_commit_empty_when_no_hash():
    assert make_clickable_commit("org/model", "") == ""
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/display/test_formatting.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_clickable_commit'`

- [ ] **Step 4: Implement the helper**

In `src/display/formatting.py`, add after `make_clickable_model`:

```python
def make_clickable_commit(model_name, commit_hash):
    if not commit_hash:
        return ""
    link = f"https://huggingface.co/{model_name}/commit/{commit_hash}"
    return model_hyperlink(link, commit_hash[:7])
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/display/test_formatting.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/display/formatting.py tests/display/__init__.py tests/display/test_formatting.py
git commit -m "Add make_clickable_commit helper"
```

---

## Task 2: Leaderboard columns

**Files:**
- Modify: `src/display/utils.py:37` (after the `revision` column)

This task has no standalone test; Task 3's test asserts the columns appear in `COLS` and in each row.

- [ ] **Step 1: Add the two columns**

In `src/display/utils.py`, immediately after the `revision` line (`auto_eval_column_dict.append(["revision", ...])`), add:

```python
auto_eval_column_dict.append(["commit", ColumnContent, ColumnContent("Commit", "markdown", False)])
auto_eval_column_dict.append(["onnx_file", ColumnContent, ColumnContent("ONNX file", "str", False)])
```

(`displayed_by_default=False` → hidden by default, selectable via the column picker, like `revision`. `hidden` stays `False`, so they remain part of `COLS`.)

- [ ] **Step 2: Verify the columns register**

Run: `python -c "from src.display.utils import AutoEvalColumn, COLS; print(AutoEvalColumn.commit.name, AutoEvalColumn.onnx_file.name, 'Commit' in COLS, 'ONNX file' in COLS)"`
Expected: `Commit ONNX file True True`

- [ ] **Step 3: Commit**

```bash
git add src/display/utils.py
git commit -m "Add Commit and ONNX file leaderboard columns"
```

---

## Task 3: Populate the new columns in result_to_row

**Files:**
- Modify: `src/leaderboard/read_evals.py`
- Test: `tests/leaderboard/test_read_evals.py`

- [ ] **Step 1: Write the failing test**

In `tests/leaderboard/test_read_evals.py`, add this test (and add `make_clickable_commit` usage knowledge — the import is already present indirectly). Append:

```python
def test_includes_commit_and_onnx_file(tmp_path):
    record = {
        "model": "org/m1", "revision": "main",
        "commit_hash": "deadbeefcafebabe1234567890abcdef12345678",
        "onnx_path": "models/best.onnx",
        "submitter": "alice", "submitted_time": "2026-06-11T00:00:00Z",
        "stochastic": False, "k_samples": 1, "status": "FINISHED",
        "overall_score": 0.5,
        "tracks": {
            "fixed_shape": {"mean": 0.5, "median": 0.5, "p10": 0.2,
                            "feasibility_rate_0p9": 0.3, "strict_feasibility": 0.1, "n": 2},
            "variable_shape": {"mean": 0.5, "median": 0.4, "p10": 0.1,
                               "feasibility_rate_0p9": 0.2, "strict_feasibility": 0.0, "n": 2},
        },
    }
    sub = tmp_path / "org"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "m1.json").write_text(json.dumps(record))

    rows = get_raw_eval_results(str(tmp_path))
    row = rows[0]
    assert "deadbee" in row[AutoEvalColumn.commit.name]
    assert row[AutoEvalColumn.onnx_file.name] == "models/best.onnx"


def test_defaults_when_fields_absent(tmp_path):
    _write_result(tmp_path, "m1", 0.8)  # record without commit_hash / onnx_path
    row = get_raw_eval_results(str(tmp_path))[0]
    assert row[AutoEvalColumn.commit.name] == ""
    assert row[AutoEvalColumn.onnx_file.name] == "model.onnx"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/leaderboard/test_read_evals.py -v`
Expected: FAIL — `KeyError` on `AutoEvalColumn.commit.name` (column not produced by `result_to_row`).

- [ ] **Step 3: Implement**

In `src/leaderboard/read_evals.py`, update the import line:

```python
from src.display.formatting import make_clickable_commit, make_clickable_model
```

In `result_to_row`, add two entries to the returned dict (after the `revision` entry):

```python
        AutoEvalColumn.revision.name: record.get("revision", ""),
        AutoEvalColumn.commit.name: make_clickable_commit(record["model"], record.get("commit_hash", "")),
        AutoEvalColumn.onnx_file.name: record.get("onnx_path", "model.onnx"),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/leaderboard/test_read_evals.py -v`
Expected: PASS (all tests, including the pre-existing `test_parses_records_into_rows` whose `for col in COLS: assert col in row` now covers the new columns).

- [ ] **Step 5: Commit**

```bash
git add src/leaderboard/read_evals.py tests/leaderboard/test_read_evals.py
git commit -m "Populate Commit and ONNX file columns in result rows"
```

---

## Task 4: evaluate_model records commit_hash + onnx_path

**Files:**
- Modify: `src/backend/eval.py`
- Test: `tests/backend/test_eval.py`

- [ ] **Step 1: Write the failing test**

In `tests/backend/test_eval.py`, append:

```python
def test_evaluate_model_records_commit_and_onnx_path(monkeypatch):
    data = [_eval_data("b0")]
    _patch_physics(monkeypatch, data[0].requirement_metrics)
    monkeypatch.setattr(beval, "_load_track", lambda track, n: data)

    record = beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="main",
        submitter="me", stochastic=False, k_samples=1,
        submitted_time="2026-06-11T00:00:00Z",
        commit_hash="deadbeefcafebabe", onnx_path="models/best.onnx",
    )
    assert record["commit_hash"] == "deadbeefcafebabe"
    assert record["onnx_path"] == "models/best.onnx"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/backend/test_eval.py::test_evaluate_model_records_commit_and_onnx_path -v`
Expected: FAIL — `TypeError: evaluate_model() got an unexpected keyword argument 'commit_hash'`

- [ ] **Step 3: Implement**

In `src/backend/eval.py`, change the `evaluate_model` signature to add the two keyword args (after `submitted_time`):

```python
def evaluate_model(
    model,
    model_name: str,
    revision: str,
    submitter: str,
    stochastic: bool,
    k_samples: int,
    submitted_time: str,
    n_instances: int = config.N_INSTANCES,
    commit_hash: str = "",
    onnx_path: str = "model.onnx",
) -> dict:
```

In the returned record dict, add the two fields (after `"revision": revision,`):

```python
        "model": model_name,
        "revision": revision,
        "commit_hash": commit_hash,
        "onnx_path": onnx_path,
        "submitter": submitter,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/backend/test_eval.py -v`
Expected: PASS (new test plus all pre-existing ones — they omit the new kwargs and rely on defaults).

- [ ] **Step 5: Commit**

```bash
git add src/backend/eval.py tests/backend/test_eval.py
git commit -m "Record commit_hash and onnx_path in eval result"
```

---

## Task 5: Deduplication key + rewrite already_submitted_models

**Files:**
- Modify: `src/submission/check_validity.py`
- Test: `tests/submission/test_check_validity.py`

- [ ] **Step 1: Write the failing test**

In `tests/submission/test_check_validity.py`, add the import at the top (keep the existing imports):

```python
import json

from src.submission.check_validity import (
    already_submitted_models,
    submission_key,
    validate_local_onnx,
)
```

Append these tests:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/submission/test_check_validity.py -v`
Expected: FAIL — `ImportError: cannot import name 'submission_key'`

- [ ] **Step 3: Implement**

In `src/submission/check_validity.py`, replace the entire `already_submitted_models` function (and its `defaultdict`-based body) with:

```python
def submission_key(model: str, commit_hash: str, onnx_path: str) -> str:
    """Canonical dedup identity: same model + commit + file = same evaluation."""
    return f"{model}@{commit_hash}:{onnx_path}"


def already_submitted_models(requested_models_dir: str) -> set[str]:
    """Set of dedup keys (model@commit_hash:onnx_path) already in the queue."""
    keys: set[str] = set()
    for root, _, files in os.walk(requested_models_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file)) as f:
                info = json.load(f)
            model = info.get("model")
            if not model:
                continue
            keys.add(submission_key(model, info.get("commit_hash", ""), info.get("onnx_path", "model.onnx")))
    return keys
```

If the `from collections import defaultdict` import is now unused, remove it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/submission/test_check_validity.py -v`
Expected: PASS (new tests plus the existing `validate_local_onnx` tests).

- [ ] **Step 5: Commit**

```bash
git add src/submission/check_validity.py tests/submission/test_check_validity.py
git commit -m "Key submission dedup on model, commit hash, and onnx path"
```

---

## Task 6: Wire commit hash, onnx path, and dedup into add_new_eval

**Files:**
- Modify: `src/submission/submit.py`
- Test: `tests/submission/test_submit.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/submission/test_submit.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/submission/test_submit.py -v`
Expected: FAIL — `add_new_eval` has no `onnx_path` parameter / does not record `commit_hash`.

- [ ] **Step 3: Implement**

In `src/submission/submit.py`, update the import of check_validity helpers:

```python
from src.submission.check_validity import already_submitted_models, submission_key, validate_local_onnx
```

Change the `add_new_eval` signature to add `onnx_path`:

```python
def add_new_eval(model: str, revision: str, submitter: str, stochastic: bool, k_samples: int, n_instances: int = -1, onnx_path: str = "model.onnx"):
```

Right after the existing `submitter` guard and `revision = revision or "main"` line, normalise and validate the path:

```python
    revision = revision or "main"
    onnx_path = (onnx_path or ONNX_FILENAME).strip()
    if not onnx_path.endswith(".onnx"):
        return styled_error("The model file path must end in `.onnx`.")
    k_samples = int(k_samples) if stochastic else 1
```

Capture the resolved SHA from `repo_info` — replace the existing repo-info block:

```python
    try:
        info = API.repo_info(repo_id=model, revision=revision, repo_type="model", token=TOKEN)
        commit_hash = info.sha
    except Exception as exc:  # noqa: BLE001
        logger.warning("Repo info check failed for %s: %s", model, exc)
        return styled_error(f'Model "{model}" was not found or is not accessible: {exc}')

    # Reject an identical (model, commit, file) submission already in the queue.
    if submission_key(model, commit_hash, onnx_path) in already_submitted_models(EVAL_REQUESTS_PATH):
        logger.info("Duplicate submission for %s@%s:%s", model, commit_hash, onnx_path)
        return styled_error("This exact model commit and file has already been submitted.")
```

Update the download-for-validation block to use the custom path **and the pinned SHA**:

```python
    try:
        with tempfile.TemporaryDirectory() as tmp:
            logger.debug("Downloading %s from %s for validation", onnx_path, model)
            local = hf_hub_download(
                repo_id=model, filename=onnx_path, revision=commit_hash,
                local_dir=tmp, token=TOKEN,
            )
            ok, msg = validate_local_onnx(local)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not download %s from %s: %s", onnx_path, model, exc)
        return styled_error(f"Could not download `{onnx_path}` from the repo: {exc}")
```

Add the two fields to `eval_entry` (after `"revision": revision,`):

```python
        "model": model,
        "revision": revision,
        "commit_hash": commit_hash,
        "onnx_path": onnx_path,
        "submitter": submitter,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/submission/test_submit.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/submission/submit.py tests/submission/test_submit.py
git commit -m "Resolve commit hash, validate onnx path, and dedup in add_new_eval"
```

---

## Task 7: Runner downloads the pinned commit + custom file

**Files:**
- Modify: `src/backend/runner.py`
- Test: `tests/backend/test_runner.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/backend/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/backend/test_runner.py -v`
Expected: FAIL — the runner downloads the hardcoded `ONNX_FILENAME` at `req["revision"]` and does not pass `commit_hash`/`onnx_path` to `evaluate_model`.

- [ ] **Step 3: Implement**

In `src/backend/runner.py`, inside `run_once`, replace the download + `evaluate_model` block within the `try`:

```python
            with tempfile.TemporaryDirectory() as tmp:
                onnx_path = req.get("onnx_path", ONNX_FILENAME)
                ref = req.get("commit_hash") or req["revision"]
                logger.debug("Downloading %s from %s at %s", onnx_path, req["model"], ref)
                local = hf_hub_download(repo_id=req["model"], filename=onnx_path,
                                        revision=ref, local_dir=tmp, token=TOKEN)
                model = OnnxModel(local)
                record = beval.evaluate_model(
                    model=model, model_name=req["model"], revision=req["revision"],
                    submitter=req.get("submitter", ""), stochastic=req.get("stochastic", False),
                    k_samples=req.get("k_samples", 1), submitted_time=req.get("submitted_time", ""),
                    n_instances=req.get("n_instances", config.N_INSTANCES),
                    commit_hash=req.get("commit_hash", ""), onnx_path=onnx_path,
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/backend/test_runner.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backend/runner.py tests/backend/test_runner.py
git commit -m "Pin runner download to commit hash and custom onnx path"
```

---

## Task 8: Submit form — ONNX file path field

**Files:**
- Modify: `app_leaderboard.py`

This is Gradio UI wiring; `app_leaderboard.py` performs network `snapshot_download` at import, so it is verified by `py_compile` rather than a unit test.

- [ ] **Step 1: Add the textbox to the form**

In `app_leaderboard.py`, in the first form column (after `revision_name_textbox`), add the field:

```python
                    model_name_textbox = gr.Textbox(label="Model repo (org/model)")
                    revision_name_textbox = gr.Textbox(label="Revision", placeholder="main")
                    onnx_path_textbox = gr.Textbox(label="ONNX file path", placeholder="model.onnx")
                    submitter_textbox = gr.Textbox(label="Submitter / contact")
```

- [ ] **Step 2: Thread it through `submit_eval`**

Change the `submit_eval` signature and its `add_new_eval` call:

```python
def submit_eval(model, revision, onnx_path, submitter, model_type, k_samples, instance_mode, n_instances):
    """Adapt the radio/slider controls to `add_new_eval`'s arguments.

    A disabled slider still submits its last value, so the radio is the source
    of truth: deterministic forces a single sample, full-set forces n=0.
    """
    stochastic = model_type == STOCHASTIC
    n = int(n_instances) if instance_mode == LIMIT_N else -1
    return add_new_eval(model, revision, submitter, stochastic, int(k_samples), n, onnx_path)
```

- [ ] **Step 3: Add the textbox to the click inputs**

In `submit_button.click`, add `onnx_path_textbox` to the inputs list, in the same position as the `submit_eval` signature (after `revision_name_textbox`):

```python
            submit_button.click(
                submit_eval,
                [
                    model_name_textbox,
                    revision_name_textbox,
                    onnx_path_textbox,
                    submitter_textbox,
                    stochastic_radio,
                    k_samples_slider,
                    instance_radio,
                    n_instances_slider,
                ],
                submission_result,
            )
```

- [ ] **Step 4: Verify the module compiles**

Run: `python -m py_compile app_leaderboard.py && echo OK`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add app_leaderboard.py
git commit -m "Add ONNX file path field to submission form"
```

---

## Task 9: Full suite + lint

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Run lint/format checks**

Run: `make quality`
Expected: no errors. If formatting fails, run `make style`, then re-run `make quality`, then `git commit -am "Apply formatting"`.

---

## Self-Review Notes

- **Spec coverage:** §1 commit hash → Tasks 4, 6, 7 (resolve at submit, record, pin in runner) + display Tasks 1–3. §2 custom onnx path → Tasks 6, 7, 8 (+ display Task 3). §3 leaderboard columns → Tasks 1–3. §4 dedup → Tasks 5, 6.
- **Type/name consistency:** `commit_hash` and `onnx_path` are the field names everywhere (request JSON, result record, `evaluate_model` kwargs). `submission_key(model, commit_hash, onnx_path)` is defined in Task 5 and consumed identically in Task 6. Column attrs `AutoEvalColumn.commit` / `AutoEvalColumn.onnx_file` (Task 2) match their use in Task 3.
- **Backward compatibility:** every downstream read uses `.get(..., default)` (`commit_hash` → `""`, `onnx_path` → `"model.onnx"`) so pre-existing requests/results still load.
