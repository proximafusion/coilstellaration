"""Backend-worker Space entrypoint.

Host this on a SEPARATE Hugging Face Space from the leaderboard. It runs a tiny
status page and a background scheduler that polls the `requests` queue every
`COILBENCH_RUNNER_INTERVAL` seconds, evaluating any PENDING submissions through
the coilstellaration physics pipeline and pushing results.

The recommended way to deploy is to mirror this repo unchanged and set the Space
variable COILBENCH_ROLE=worker — `app.py` then imports and launches this module
(see app.py). Alternatively, point the Space's README `app_file` here directly:
    sdk: gradio
    app_file: runner_app.py

Required secrets / variables on the worker Space:
    HF_TOKEN          write access to <OWNER>/requests and <OWNER>/results,
                      read access to submitted model repos
    COILBENCH_OWNER   org owning the requests/results datasets (e.g. jhewers-pf)
Optional:
    COILBENCH_RUNNER_INTERVAL   seconds between polls (default 300)
    COILBENCH_N                 instances per track (0 = full split)
"""

import datetime
import os
import traceback

import gradio as gr
from apscheduler.schedulers.background import BackgroundScheduler
from leaderboard.src.backend import runner
from leaderboard.src.envs import OWNER, QUEUE_REPO, RESULTS_REPO

INTERVAL = int(os.environ.get("COILBENCH_RUNNER_INTERVAL", "300"))

# Last-run status, surfaced on the page so you can see the worker is alive.
_status = {"last_run": "never", "last_result": "—", "finished": 0, "failed": 0}


def poll_once() -> None:
    """One queue-draining pass; never let an error kill the scheduler thread."""
    _status["last_run"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        finished, failed = runner.run_once()
        _status["finished"] += finished
        _status["failed"] += failed
        _status["last_result"] = f"{finished} finished, {failed} failed"
        print(f"[runner_app] poll at {_status['last_run']}: {finished} finished, {failed} failed")
    except Exception as exc:  # noqa: BLE001
        _status["last_result"] = f"error: {exc}"
        print(f"[runner_app] run_once FAILED: {exc}\n{traceback.format_exc()}")


def _status_md() -> str:
    return (
        f"# Coilstellaration backend worker\n\n"
        f"Polling `{QUEUE_REPO}` every **{INTERVAL}s**, pushing to `{RESULTS_REPO}`.\n\n"
        f"- Owner: `{OWNER}`\n"
        f"- Last run (UTC): `{_status['last_run']}`\n"
        f"- Last result: `{_status['last_result']}`\n"
        f"- Total finished (since program start): `{_status['finished']}`\n"
        f"- Total failed (since program start): `{_status['failed']}`\n"
    )


scheduler = BackgroundScheduler()
scheduler.add_job(
    poll_once,
    "interval",
    seconds=INTERVAL,
    next_run_time=datetime.datetime.now(),
)
scheduler.start()

with gr.Blocks() as app_runner:
    status = gr.Markdown(_status_md())
    refresh = gr.Button("Refresh status")
    refresh.click(_status_md, outputs=status)

if __name__ == "__main__":
    app_runner.launch()
