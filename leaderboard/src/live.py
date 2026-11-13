"""The leaderboard Space's data, refreshed while the app runs.

`LiveData.refresh` downloads the results and requests datasets and swaps in a new
`Snapshot`. The app calls it on start and then on a schedule, and its event
handlers read `LiveData.current` on every call, so a page shows the latest
snapshot when it loads. No restart is needed to pick up new results, and a
refresh never interrupts a submission that is running.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from huggingface_hub import snapshot_download

from leaderboard.src.display.utils import COLS, EVAL_COLS
from leaderboard.src.envs import RESULTS_REPO, TOKEN
from leaderboard.src.leaderboard.read_evals import get_raw_eval_results
from leaderboard.src.populate import frames_from_records, get_evaluation_queue_df
from leaderboard.src.submission import requests_dataset

logger = logging.getLogger(__name__)

# The evaluation queue's tables, in the order the Submit tab shows them.
QUEUE_STATUSES = ("Running", "Pending", "Finished", "Failed")


def _empty_queue() -> dict[str, pd.DataFrame]:
    return {status: pd.DataFrame(columns=EVAL_COLS) for status in QUEUE_STATUSES}


@dataclass(frozen=True)
class Snapshot:
    """Everything the leaderboard shows, as of one refresh."""

    # One display frame per (track, stratum) view; see `frames_from_records`.
    frames: dict[tuple[str, str], pd.DataFrame] = field(
        default_factory=lambda: frames_from_records([], COLS)
    )
    # The raw result records, as read from the results dataset's JSON files.
    records: list[dict] = field(default_factory=list)
    # The evaluation queue's tables, keyed by `QUEUE_STATUSES`.
    queue: dict[str, pd.DataFrame] = field(default_factory=_empty_queue)


def download_snapshot() -> Snapshot:
    """Read the results and requests datasets into a new `Snapshot`.

    Both are fetched into the Hub cache, so only changed files are transferred,
    and the directory returned holds exactly the repo's current files: a result
    deleted from the repo disappears too.
    """
    results_path = snapshot_download(
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        # Only the JSON summaries; the parquet and ONNX files aren't displayed.
        allow_patterns="*.json",
        tqdm_class=None,
        etag_timeout=30,
        token=TOKEN,
    )
    records = get_raw_eval_results(results_path)
    finished, running, pending, failed = get_evaluation_queue_df(
        list(requests_dataset.iter_requests()), EVAL_COLS
    )
    return Snapshot(
        frames=frames_from_records(records, COLS),
        records=records,
        queue={
            "Running": running,
            "Pending": pending,
            "Finished": finished,
            "Failed": failed,
        },
    )


class LiveData:
    """The current `Snapshot`, replaced whole by each successful refresh.

    Readers take `current` once per event and use that snapshot throughout, so a
    refresh in the middle of an event can't mix two snapshots.
    """

    def __init__(self, load: Callable[[], Snapshot] = download_snapshot):
        self._load = load
        # Empty until the first refresh succeeds.
        self.current = Snapshot()

    def refresh(self) -> bool:
        """Load a new snapshot; on failure, log it and keep the current one."""
        try:
            snapshot = self._load()
        except Exception:
            logger.exception("Refreshing the leaderboard data failed")
            return False
        self.current = snapshot
        logger.info(
            "Refreshed the leaderboard data: %d results, queue %s",
            len(snapshot.records),
            {status: len(df) for status, df in snapshot.queue.items()},
        )
        return True
