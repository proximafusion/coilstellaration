import logging
from collections.abc import Sequence

import pandas as pd

from leaderboard.src.display.formatting import make_clickable_model
from leaderboard.src.display.utils import (
    SORT_COLS,
    STRATUM_VIEWS,
    TRACKS,
    EvalQueueColumn,
)
from leaderboard.src.leaderboard.read_evals import get_raw_eval_results, result_to_row

logger = logging.getLogger(__name__)

# The leaderboard's default view: variable track, all-strata pool.
DEFAULT_VIEW = ("variable_shape", "All")


def _frame_for_view(
    records: Sequence[dict], track: str, stratum: str, cols: list
) -> pd.DataFrame:
    """Build one display frame from raw records for a single track+stratum view.

    Each record contributes a row only when it has data for this track and
    stratum (`result_to_row` returns `None` otherwise), so a submission scored
    on one track never appears under the other, and records predating the
    per-stratum breakdown drop out of the specific-stratum views. Rows are
    ordered by strict feasibility then mean, both descending.
    """
    rows = [
        row
        for rec in records
        if (row := result_to_row(rec, track, stratum)) is not None
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame.from_records(rows)
    df = df.sort_values(by=SORT_COLS, ascending=False, na_position="last")
    return df[cols].round(decimals=3)


def get_leaderboard_frames(
    results_path: str, cols: list
) -> dict[tuple[str, str], pd.DataFrame]:
    """One display frame per `(track, stratum)` view.

    All frames carry identical columns; the values are the selected track's numbers for
    the selected stratum. The leaderboard's track and stratum selectors together pick
    which frame the table renders.
    """
    records = get_raw_eval_results(results_path)
    frames = {
        (track, stratum): _frame_for_view(records, track, stratum, cols)
        for track in TRACKS
        for stratum in STRATUM_VIEWS
    }
    logger.debug("Built leaderboard frames: %s", {k: len(f) for k, f in frames.items()})
    return frames


def get_leaderboard_df(results_path: str, cols: list) -> pd.DataFrame:
    """The default display frame (fixed track, all strata)."""
    return get_leaderboard_frames(results_path, cols)[DEFAULT_VIEW]


def _queue_entries(records: Sequence[dict]) -> Sequence[dict]:
    """Copy request records into display-ready dicts for the queue tables."""
    all_evals = []
    for record in records:
        data = dict(record)  # don't mutate the caller's dict
        data[EvalQueueColumn.model.name] = make_clickable_model(data["model"])
        data[EvalQueueColumn.revision.name] = data.get("revision", "main")
        data[EvalQueueColumn.started.name] = data.get("started_at", "")
        req_tracks = data.get("tracks") or ["fixed_shape", "variable_shape"]
        data[EvalQueueColumn.tracks.name] = "both" if len(req_tracks) == 2 else "fixed"
        all_evals.append(data)
    return all_evals


def get_evaluation_queue_df(
    records: Sequence[dict], cols: list
) -> Sequence[pd.DataFrame]:
    """Bucket request records into finished/running/pending dataframes."""
    all_evals = _queue_entries(records)

    pending_list = [e for e in all_evals if e.get("status") in ["PENDING", "RERUN"]]
    running_list = [e for e in all_evals if e.get("status") == "RUNNING"]
    finished_list = [
        e
        for e in all_evals
        if str(e.get("status", "")).startswith("FINISHED")
        or e.get("status") == "PENDING_NEW_EVAL"
    ]
    failed_list = [e for e in all_evals if e.get("status") == "FAILED"]
    df_pending = pd.DataFrame.from_records(pending_list, columns=cols)
    df_running = pd.DataFrame.from_records(running_list, columns=cols)
    df_finished = pd.DataFrame.from_records(finished_list, columns=cols)
    df_failed = pd.DataFrame.from_records(failed_list, columns=cols)
    logger.debug(
        "Eval queue: %d finished, %d running, %d pending, %d failed.",
        len(finished_list),
        len(running_list),
        len(pending_list),
        len(failed_list),
    )
    return df_finished[cols], df_running[cols], df_pending[cols], df_failed[cols]
