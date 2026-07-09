import logging

import pandas as pd

from src.display.formatting import has_no_nan_values, make_clickable_model
from src.display.utils import STRATUM_VIEWS, AutoEvalColumn, EvalQueueColumn
from src.leaderboard.read_evals import get_raw_eval_results, result_to_row

logger = logging.getLogger(__name__)


def _frame_for_stratum(records: list[dict], stratum: str, cols: list, benchmark_cols: list) -> pd.DataFrame:
    """Build one display frame from raw records for a single stratum view.

    Rows whose track means are NaN for this stratum (e.g. records predating the
    per-stratum breakdown, when viewing a specific stratum) are dropped so the
    view only shows models actually scored on it.
    """
    rows = [result_to_row(rec, stratum=stratum) for rec in records]
    df = pd.DataFrame.from_records(rows)
    df = df.sort_values(by=[AutoEvalColumn.overall.name], ascending=False)
    df = df[cols].round(decimals=3)
    df = df[has_no_nan_values(df, benchmark_cols)]
    return df


def get_leaderboard_frames(results_path: str, cols: list, benchmark_cols: list) -> dict[str, pd.DataFrame]:
    """One display frame per stratum view (keys: `STRATUM_VIEWS`).

    Each frame carries identical columns; only the per-track metric values (and
    the derived Overall/ranking) differ by stratum. The leaderboard's stratum
    selector swaps which frame the table renders.
    """
    records = get_raw_eval_results(results_path)
    if not records:
        logger.debug("No eval results found; returning empty leaderboard frames")
        empty = pd.DataFrame(columns=cols)
        return {view: empty.copy() for view in STRATUM_VIEWS}

    frames = {
        view: _frame_for_stratum(records, view, cols, benchmark_cols)
        for view in STRATUM_VIEWS
    }
    logger.debug("Built leaderboard frames: %s",
                 {v: len(f) for v, f in frames.items()})
    return frames


def get_leaderboard_df(results_path: str, cols: list, benchmark_cols: list) -> pd.DataFrame:
    """The all-strata display frame (the leaderboard's default view)."""
    return get_leaderboard_frames(results_path, cols, benchmark_cols)["All"]


def get_evaluation_queue_df(records: list[dict], cols: list) -> list[pd.DataFrame]:
    """Bucket request records into finished/running/pending dataframes."""
    all_evals = []
    for record in records:
        data = dict(record)  # don't mutate the caller's dict
        data[EvalQueueColumn.model.name] = make_clickable_model(data["model"])
        data[EvalQueueColumn.revision.name] = data.get("revision", "main")
        data[EvalQueueColumn.started.name] = data.get("started_at", "")
        req_tracks = data.get("tracks") or ["fixed_shape", "variable_shape"]
        data[EvalQueueColumn.tracks.name] = "both" if len(req_tracks) == 2 else "fixed"
        all_evals.append(data)

    pending_list = [e for e in all_evals if e.get("status") in ["PENDING", "RERUN"]]
    running_list = [e for e in all_evals if e.get("status") == "RUNNING"]
    finished_list = [
        e for e in all_evals
        if str(e.get("status", "")).startswith("FINISHED") or e.get("status") == "PENDING_NEW_EVAL"
    ]
    df_pending = pd.DataFrame.from_records(pending_list, columns=cols)
    df_running = pd.DataFrame.from_records(running_list, columns=cols)
    df_finished = pd.DataFrame.from_records(finished_list, columns=cols)
    logger.debug(
        "Eval queue: %d finished, %d running, %d pending",
        len(finished_list), len(running_list), len(pending_list),
    )
    return df_finished[cols], df_running[cols], df_pending[cols]
