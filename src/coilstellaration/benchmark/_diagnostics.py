"""GCS-backed step-by-step diagnostics for cloud batch tasks.

Writes one tiny object per `write()` call to
`gs://proxima-experimental/scadena/cu_scoring_diagnostics/<job_id>/task_<idx>/`
so runner progress is visible from outside the VM even when log-read
permissions on `batch.googleapis.com/Job` resources are denied.

All calls are best-effort: they never raise, since a failed diagnostic write
must not bring down the task it is observing.
"""

import logging
import os
import time
import traceback

logger = logging.getLogger(__name__)

DIAGNOSTICS_GCS_PREFIX = "gs://proxima-experimental/scadena/cu_scoring_diagnostics"
"""Where diagnostic markers land.

Read via `gsutil ls -r <prefix>/<job_id>/`.
"""


def write(step: str, message: str = "") -> None:
    """Upload one diagnostic marker. Best-effort; never raises.

    Each marker becomes one small GCS object named with an epoch prefix so `gsutil ls`
    returns them in chronological order.
    """
    try:
        import gcsfs

        job_id = os.environ.get("BATCHED_JOB_ID", "unknown")
        task_index = os.environ.get("BATCH_TASK_INDEX", "0")
        epoch = int(time.time())
        path = (
            f"{DIAGNOSTICS_GCS_PREFIX}/{job_id}/"
            f"task_{task_index}/{epoch:010d}_{step}.txt"
        )
        body = f"step={step}\nepoch={epoch}\nmessage={message}\n"
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "w") as f:
            f.write(body)
        logger.info("diagnostic %s -> %s", step, path)
    except Exception:
        logger.exception("failed to write diagnostic %s", step)


def write_traceback(step: str) -> None:
    """Upload the currently-handled exception's traceback.

    Use inside `except`.
    """
    write(step, traceback.format_exc())
