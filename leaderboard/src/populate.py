import logging

import pandas as pd

from src.display.formatting import has_no_nan_values, make_clickable_model
from src.display.utils import AutoEvalColumn, EvalQueueColumn
from src.leaderboard.read_evals import get_raw_eval_results

logger = logging.getLogger(__name__)


def get_leaderboard_df(results_path: str, cols: list, benchmark_cols: list) -> pd.DataFrame:
    """Creates a dataframe from all the individual experiment results"""
    all_data_json = get_raw_eval_results(results_path)

    # No results yet (fresh leaderboard): return an empty frame that still has
    # the expected columns so the table renders instead of erroring.
    if not all_data_json:
        logger.debug("No eval results found; returning empty leaderboard")
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame.from_records(all_data_json)
    df = df.sort_values(by=[AutoEvalColumn.overall.name], ascending=False)
    df = df[cols].round(decimals=3)

    # filter out rows missing a track mean
    df = df[has_no_nan_values(df, benchmark_cols)]
    logger.debug("Built leaderboard dataframe with %d row(s)", len(df))
    return df


def get_evaluation_queue_df(records: list[dict], cols: list) -> list[pd.DataFrame]:
    """Bucket request records into finished/running/pending dataframes."""
    all_evals = []
    for record in records:
        data = dict(record)  # don't mutate the caller's dict
        data[EvalQueueColumn.model.name] = make_clickable_model(data["model"])
        data[EvalQueueColumn.revision.name] = data.get("revision", "main")
        data[EvalQueueColumn.started.name] = data.get("started_at", "")
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
