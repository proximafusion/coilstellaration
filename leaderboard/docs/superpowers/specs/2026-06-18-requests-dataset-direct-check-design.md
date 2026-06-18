# Replace the local eval-queue cache with a direct check of the requests HF dataset

**Date:** 2026-06-18
**Status:** Approved, ready for implementation planning

## Problem

The Gradio Space keeps a local `eval-queue/` directory that is a one-time
mirror of the HF `requests` dataset (`QUEUE_REPO`). It is populated by a
`snapshot_download` at Space startup (`app_leaderboard.py:52-57`) and read in two
places:

- **Duplicate detection on submit** — `add_new_eval` calls
  `already_submitted_models(EVAL_REQUESTS_PATH)` (`submit.py:50`), which walks the
  local dir.
- **Queue tables** — `get_evaluation_queue_df(EVAL_REQUESTS_PATH, ...)`
  (`populate.py:34`) walks the same dir to build the PENDING/RUNNING/FINISHED
  views.

Because the snapshot is only refreshed at boot (and on the 30-minute
`restart_space` scheduler), the local cache goes stale during a session. The
correctness-critical consequence is duplicate detection: a model submitted after
boot is not seen, so an expensive DESC re-evaluation can be queued for a model
already in the queue.

The backend worker (`runner.py`) is **not** affected — it runs its own
`snapshot_download` every `run_once`, so its view is fresh per poll.

## Goal

Remove the Space-side local `eval-queue/` cache entirely. All Space-side reads of
the requests dataset go directly to HF:

- Duplicate detection becomes genuinely **live** (single list call per submit).
- Queue tables keep their current **boot cadence** (computed once at startup and
  on restart) but are sourced directly from HF instead of a local snapshot. No
  live-refresh UI wiring is added.

The backend `runner.py` is unchanged and keeps its `snapshot_download`.

## Approach (chosen)

`HfApi.list_repo_files` + `hf_hub_download` per file — the pattern `runner.py`
already uses.

The request file path is **deterministic**: submit writes
`{org}/{model_path}_eval_request_{commit_hash}_{onnx_slug}.json`. That filename
already encodes the dedup identity, so duplicate detection needs **no file
downloads** — only a `list_repo_files` call and a membership test on the expected
path. The queue tables do need contents, so they `hf_hub_download` each JSON
(etag-cached in `HF_HOME`, making restarts cheap).

Rejected alternatives:
- **`HfFileSystem` glob + read_text** — clean and zero local footprint, but a
  different API surface from the rest of the codebase and no benefit here since
  dedup avoids downloads entirely under the chosen approach.
- **On-demand `snapshot_download` to a temp dir** — still the snapshot/local-cache
  approach we are removing, and gives dedup no live path.

## Design

### New module `src/submission/requests_dataset.py`

The single place that talks to the HF requests dataset.

- `list_request_files() -> list[str]` — `API.list_repo_files(QUEUE_REPO,
  repo_type="dataset")`, filtered to `.json`.
- `request_path(model, commit_hash, onnx_path) -> str` — canonical request path
  builder, **extracted from `submit.py`** (currently inlined at `submit.py:84-87`)
  so submit-write and dedup-check share one source of truth.
- `submission_exists(model, commit_hash, onnx_path) -> bool` — builds the path via
  `request_path()` and membership-tests it against `list_request_files()`.
- `iter_requests() -> Iterator[dict]` — for each listed `.json`, `hf_hub_download`
  + `json.load`, yielding the parsed dict. Per-file parse is wrapped in
  try/except: a malformed/partial file is logged and skipped, not fatal.

### `src/submission/submit.py`

- Replace the `already_submitted_models(EVAL_REQUESTS_PATH)` walk (line 50) with
  `requests_dataset.submission_exists(model, commit_hash, onnx_path)`.
- The write path (lines 84-97) reuses `request_path()` to build the upload path.

### `src/populate.py`

- `get_evaluation_queue_df` changes signature to take the request **records**
  (`list[dict]`) instead of a `save_path`, and does only bucketing/df-shaping.
  This drops the `os.walk` and the dual folder/file handling, and makes the
  function testable without a filesystem.
- Bucketing uses `.get("status")` so a `.json` lacking expected fields is
  tolerated.

### `app_leaderboard.py`

- Delete the `snapshot_download(QUEUE_REPO -> EVAL_REQUESTS_PATH)` block
  (lines 52-57). Keep the `RESULTS_REPO` snapshot (60-62).
- Feed the queue df from `list(requests_dataset.iter_requests())`.

### `src/submission/check_validity.py`

- `already_submitted_models` (and its `os.walk`) becomes dead Space code; remove
  it. Remove `submission_key` if it is no longer referenced once dedup is
  filename-based (confirm during implementation).

### `src/envs.py`

- Keep `EVAL_REQUESTS_PATH` — backend `runner.py` still snapshots into it. Only
  the Space stops using it.

### `src/backend/runner.py`

- Unchanged. Keeps its per-poll `snapshot_download`.

## Error handling & edge cases

- **Dedup check fails (network/HF blip on submit)** — fail **closed**: catch the
  `list_repo_files` error and return
  `styled_error("Couldn't verify against the queue — please retry.")`. A missed
  duplicate would trigger an expensive DESC re-eval, so we do not let the
  submission through unchecked.
- **Queue listing/fetch fails at boot** — keep parity with the existing snapshot
  block: wrap in try/except and call `restart_space()`.
- **Malformed/partial request JSON** — `iter_requests()` logs and skips the bad
  file. Strictly more robust than the current `os.walk` version, which throws on
  the first bad file.
- **Non-request files in the dataset repo** (`README.md`, `.gitattributes`, stray
  `.json`) — filtered to `.json` in `list_request_files()`; bucketing tolerates
  missing fields via `.get`.
- **Slug collision** — `onnx_slug = onnx_path.replace("/","-").removesuffix(".onnx")`
  can in principle collide (`a/b.onnx` and `a-b.onnx` map to the same slug). This
  pre-exists in the current naming scheme and is out of scope. Sharing a single
  `request_path()` builder guarantees submit-write and dedup-check stay
  consistent.
- **commit_hash always present** — `submit.py` resolves it (and errors out)
  before the dedup check, so `request_path()` never builds with an empty commit.

## Testing

- **`requests_dataset`** — mock `list_repo_files` + `hf_hub_download`; assert
  `request_path()` determinism, `submission_exists` true/false, and that
  `iter_requests` skips malformed JSON.
- **`submit`** — mock `submission_exists` → True yields the duplicate error,
  False proceeds; mock the listing-failure path → fail-closed error.
- **`populate`** — pass record dicts directly (no temp dir), assert
  PENDING/RUNNING/FINISHED bucketing.
