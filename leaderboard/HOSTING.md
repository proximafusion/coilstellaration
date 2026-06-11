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
worker picks it up. Host it like this:

1. **Duplicate this repo into a new Space** (Settings → Duplicate, or create a
   new Space pointing at this code).
2. In that Space's **README.md metadata**, set the entrypoint:
   ```yaml
   sdk: gradio
   app_file: runner_app.py
   ```
3. Add Space **secrets / variables**:
   - `HF_TOKEN` — write access to `<OWNER>/requests` + `<OWNER>/results`, read
     access to submitted model repos.
   - `COILBENCH_OWNER` — e.g. `jhewers-pf`.
   - `COILBENCH_RUNNER_INTERVAL` (optional) — seconds between polls (default 300).
   - `COILBENCH_N` (optional) — instances per track; `0` = full split.

On boot the worker fires once, then polls every `COILBENCH_RUNNER_INTERVAL`
seconds, moving each request `PENDING → RUNNING → FINISHED/FAILED` and pushing
results. The worker page shows the last run time/result.

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
