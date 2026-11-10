# Hosting

Two pieces, hosted as two separate Spaces (or the worker anywhere with the deps):

| Space | Entrypoint | Role | Needs physics deps |
|-------|------------|------|--------------------|
| Leaderboard | `app.py` | Display + accept submissions to the queue | no (only `onnxruntime`) |
| Worker | `runner_app.py` | Poll the queue, run evaluations, push results | yes (`coilstellaration`) |

The Spaces communicate only through two datasets under `<OWNER>` (set by
`COILBENCH_OWNER`, default `proximafusion`):

- `<OWNER>/requests` — the submission queue
- `<OWNER>/results` — scored results

Create them once and make sure your token can write to them:

```bash
huggingface-cli login
huggingface-cli repo create requests --type dataset --organization <OWNER>
huggingface-cli repo create results  --type dataset --organization <OWNER>
```

## Leaderboard Space

Already configured: `README.md` has `app_file: app.py`. Set the Space secret
`HF_TOKEN` (write access to the datasets) and, if your org isn't the default,
the variable `COILBENCH_OWNER`.

## Worker Space (the runner)

The runner does NOT run automatically — a request stays `PENDING` until the
worker picks it up.

`app.py` is a single entrypoint for both roles and branches on the
`COILBENCH_ROLE` variable, so the worker Space is a **pure mirror of this repo
with no file edits** — you only set Space variables:

1. **Duplicate/mirror this repo into a new Space** (Settings → Duplicate, or
   create a new Space pointing at this code). Leave `README.md` as-is
   (`app_file: app.py`).
2. Add Space **secrets / variables**:
   - `COILBENCH_ROLE` = `worker`  ← this flips the mirror into worker mode.
   - `HF_TOKEN` — write access to `<OWNER>/requests` + `<OWNER>/results`, read
     access to submitted model repos.
   - `COILBENCH_OWNER` — e.g. `jhewers-pf`.
   - `COILBENCH_RUNNER_INTERVAL` (optional) — seconds between polls (default 300).
   - `COILBENCH_N` (optional) — instances per track; `0` = full split.
   - `COILBENCH_EVAL_TIMEOUT` (optional) — wall-clock seconds before a single
     eval is killed (default 7200; `0` disables the subprocess guard). A full
     both-track evaluation takes ~1.2 h on cpu-basic (2 vCPU).
   - `COILBENCH_LEASE_TTL` (optional) — seconds before a stuck `RUNNING` request
     is presumed orphaned and recovered (default 10800; keep above the timeout).
   - `COILBENCH_MAX_ATTEMPTS` (optional) — how many times a request may be
     (re)claimed before it is failed permanently (default 3).

(With `COILBENCH_ROLE=worker`, `app.py` launches `runner_app.py` instead of the
leaderboard. You can also point a Space's `app_file` directly at `runner_app.py`
if you prefer an explicit entrypoint, but the variable approach needs no edits.)

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
  hardware (or your own always-on machine running `python -m src.backend.runner`
  on a cron) for continuous operation.
- **Run exactly one worker** against a given queue to avoid double-evaluating.
  `runner.run_once` flips a request to `RUNNING` before evaluating, so a second
  worker won't re-pick a running one, but don't run several pollers in lockstep.
- **No GPU needed**, but the physics eval is CPU-heavy; the full split takes a
  while. Tune `COILBENCH_N` while iterating.

## Run the worker without a Space

Any machine with the repo and deps:

```bash
export HF_TOKEN=<write-token>
COILBENCH_OWNER=<OWNER> python -m src.backend.runner   # single pass
```

Wrap it in a cron job or a loop for continuous draining.
