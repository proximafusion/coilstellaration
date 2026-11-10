# Hosting

Two Spaces, both deployed from this repository by
`.github/workflows/sync-leaderboard-to-hf-spaces.yml` on every push to `main`
that touches `leaderboard/`, `src/`, `pyproject.toml` or `uv.lock`:

| Space | Front matter | `COILBENCH_ROLE` | Role |
|-------|--------------|------------------|------|
| `proxima-fusion/coilstellaration-bench` | `space.leaderboard.md` | `leaderboard` (default) | Display + accept submissions to the queue |
| `proxima-fusion/coilstellaration-bench-evaluator` | `space.evaluator.md` | `runner` | Poll the queue, run evaluations, push results |

Each Space receives the whole repo with `leaderboard/app.py` as its `app_file`.
Hugging Face reads Space config only from the root `README.md` front matter and
installs from a root `requirements.txt` / `packages.txt`, so the workflow writes
those at sync time: the README from the matching `space.*.md`, the requirements
from `uv.lock` (this package plus the `leaderboard` dependency group). Edit
`space.*.md`, not the README on the Hub; the next sync overwrites it.

The Spaces communicate only through two datasets, `<OWNER>/<REPO>-requests` (the
submission queue) and `<OWNER>/<REPO>-results` (scored results). `<OWNER>` is
`COILBENCH_OWNER` (default `jhewers-pf`) and `<REPO>` is `COILBENCH_REPO`
(default `coilstellaration-bench`), so by default:

- `jhewers-pf/coilstellaration-bench-requests`
- `jhewers-pf/coilstellaration-bench-results`

Create them once and make sure your token can write to them:

```bash
hf auth login
hf repos create jhewers-pf/coilstellaration-bench-requests --type dataset --exist-ok
hf repos create jhewers-pf/coilstellaration-bench-results  --type dataset --exist-ok
```

The workflow needs a GitHub secret `HF_TOKEN` with write access to both Spaces.

## Leaderboard Space

Set the Space secret `HF_TOKEN` (write access to the datasets) and, if your org
isn't the default, the variable `COILBENCH_OWNER`.

## Worker Space (the runner)

The runner does NOT run automatically — a request stays `PENDING` until the
worker picks it up. `leaderboard/app.py` is a single entrypoint for both roles and
branches on the `COILBENCH_ROLE` variable. Add these Space **secrets / variables**:

- `COILBENCH_ROLE` = `runner`  ← this flips the Space into worker mode, which
  launches `app_runner.py` instead of the leaderboard.
- `HF_TOKEN` — write access to the requests and results datasets, read access
  to submitted model repos.
- `COILBENCH_OWNER` — only if not `jhewers-pf`.
- `COILBENCH_RUNNER_INTERVAL` (optional) — seconds between polls (default 300).
- `COILBENCH_N` (optional) — instances per track; `0` = full split.
- `COILBENCH_EVAL_TIMEOUT` (optional) — wall-clock seconds before a single
  eval is killed (default 7200; `0` disables the subprocess guard). A full
  both-track evaluation takes ~1.2 h on cpu-basic (2 vCPU).
- `COILBENCH_LEASE_TTL` (optional) — seconds before a stuck `RUNNING` request
  is presumed orphaned and recovered (default 10800; keep above the timeout).
- `COILBENCH_MAX_ATTEMPTS` (optional) — how many times a request may be
  (re)claimed before it is failed permanently (default 3).

On boot the worker fires once, then polls every `COILBENCH_RUNNER_INTERVAL`
seconds, moving each request `PENDING → RUNNING → FINISHED/FAILED` and pushing
results. The worker page shows the last run time/result.

Each eval runs in a short-lived child process (its own session leader) with a
wall-clock timeout, so one hung or runaway model can't wedge the poll loop — on
timeout the worker SIGKILLs the child's whole process group, taking out any
JAX/DESC helper processes it spawned rather than leaking them. On every poll the worker also
sweeps for `RUNNING` requests whose lease has expired (a crashed/killed worker
leaves them stranded) and re-queues them to `PENDING`, up to `COILBENCH_MAX_ATTEMPTS`
claims before failing them permanently. A claim stamps `started_at`, `worker_id`
and `attempts` on the request; `started_at` is surfaced in the queue UI.

### Notes

- **Keep the worker awake.** A Space that sleeps stops polling. Use persistent
  hardware (or your own always-on machine running `uv run python -m leaderboard.src.backend.runner`
  on a cron) for continuous operation.
- **Run exactly one worker** against a given queue to avoid double-evaluating.
  `runner.run_once` flips a request to `RUNNING` before evaluating, so a second
  worker won't re-pick a running one, but don't run several pollers in lockstep.
- **No GPU needed**, but the physics eval is CPU-heavy; the full split takes a
  while. Tune `COILBENCH_N` while iterating.

## Run the worker without a Space

Any machine with the repo and deps (`uv sync`), from the repo root:

```bash
export HF_TOKEN=<write-token>
COILBENCH_OWNER=<OWNER> uv run python -m leaderboard.src.backend.runner   # single pass
```

Wrap it in a cron job or a loop for continuous draining.
