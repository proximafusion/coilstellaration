# Track commit hash + custom ONNX path

## Goal

Two additions to the submission/eval pipeline, both surfaced on the leaderboard:

1. **Commit hash** — record and pin the resolved git SHA of each submitted model
   so the leaderboard reflects exactly which commit was evaluated.
2. **Custom ONNX path** — let submitters point at an ONNX file other than the
   hardcoded `model.onnx`, as long as it ends in `.onnx`.

## Data model

Two new fields, carried through the request JSON (`submit.py`) and the result
record (`eval.py`):

- `commit_hash` — the resolved git SHA (string).
- `onnx_path` — path of the ONNX file within the repo (string; default
  `model.onnx`).

Both are read with `.get()` + defaults downstream so pre-existing requests and
results (which lack the fields) still load and render.

## 1. Commit hash (pinned to resolved SHA)

`add_new_eval` (`src/submission/submit.py`) already calls
`API.repo_info(repo_id=model, revision=revision)`, which returns the resolved
`.sha`. Capture it as `commit_hash`.

- The download-for-validation downloads `onnx_path` at `commit_hash` (not the
  named ref).
- The runner (`src/backend/runner.py`) downloads at `req["commit_hash"]`,
  falling back to `req["revision"]` when `commit_hash` is absent (old requests).
- `revision` is preserved as the human-friendly named ref (e.g. `main`) for
  display continuity.

This pins the exact commit end-to-end: a branch moving between submission and
evaluation no longer changes what gets scored.

## 2. Custom ONNX path

- Add an **"ONNX file path"** textbox to the submit form
  (`app_leaderboard.py`), placeholder `model.onnx`, defaulting to `model.onnx`
  when left blank.
- `submit_eval` passes the value through to `add_new_eval`.
- `add_new_eval` validates the path ends in `.onnx` (else `styled_error`),
  downloads/validates that file, and stores `onnx_path` in the request JSON.
- The runner downloads `req.get("onnx_path", "model.onnx")`.
- `ONNX_FILENAME = "model.onnx"` stays as the default constant.

## 3. Leaderboard columns

In `src/display/utils.py`, two new `ColumnContent` entries, both
`displayed_by_default=False` (hidden by default, like the existing `revision`
column; selectable via the column picker):

- `commit` → label `"Commit"`, type `"markdown"` — a clickable link to
  `https://huggingface.co/{model}/commit/{sha}` showing the short 7-char SHA.
- `onnx_file` → label `"ONNX file"`, type `"str"`.

`result_to_row` (`src/leaderboard/read_evals.py`) populates both from the result
record, using `.get()` defaults (`onnx_path` → `model.onnx`; `commit_hash` →
empty/falls back to revision) so pre-existing results render without the fields.

`evaluate_model` (`src/backend/eval.py`) includes `commit_hash` and `onnx_path`
in the result record; the runner passes both through from the request.

## Out of scope (YAGNI)

- No changes to the submission-queue tab columns (`EvalQueueColumn`).
- No model-card or path validation beyond the `.onnx` suffix check.
