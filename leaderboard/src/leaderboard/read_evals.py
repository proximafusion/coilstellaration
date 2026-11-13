import json
import logging
import math
import os
from pathlib import Path

from leaderboard.src.display.formatting import (
    make_clickable_commit,
    make_results_link,
    make_user_link,
)
from leaderboard.src.display.utils import AutoEvalColumn
from leaderboard.src.envs import RESULTS_REPO

logger = logging.getLogger(__name__)


def track_summary(track_record: dict, stratum: str) -> dict | None:
    """The summary dict for one stratum view of a track.

    `"all"` maps to the track's headline (all-strata) keys; a specific stratum
    maps to its entry under `strata`, or `None` if this record predates the
    per-stratum breakdown.
    """
    if stratum == "all":
        return track_record
    return track_record.get("strata", {}).get(stratum)


def _get(summary: dict | None, key: str) -> float:
    """Metric value from a (possibly missing) summary; NaN when unavailable."""
    if not summary:
        return math.nan
    return summary.get(key, math.nan)


def _proportion_se(p: float, n) -> float:
    """Standard error of a proportion `p` over `n` instances; NaN without `n`."""
    if not n or math.isnan(p):
        return math.nan
    return math.sqrt(p * (1 - p) / n)


def _mean_time_ms(summary: dict) -> float:
    """Mean inference time per instance in ms; NaN when it wasn't recorded."""
    seconds = (summary.get("inference_time") or {}).get("mean")
    return math.nan if seconds is None else seconds * 1e3


def result_to_row(record: dict, track: str, stratum: str) -> dict | None:
    """Map one result record (spec §6) to a leaderboard row for one track+stratum view,
    or `None` when the record has no data for that view.

    The metric columns are that track's numbers: for `stratum == "all"` the
    all-strata headline, for a specific stratum that stratum's breakdown.
    Returns `None` when the track wasn't scored (empty or missing summary) or
    the stratum is absent from the record (predates the per-stratum breakdown),
    so the frame-builder can drop the row from that view.
    """
    summary = track_summary(record["tracks"].get(track, {}), stratum)
    if not summary:
        return None

    return {
        AutoEvalColumn.model.name: record["model"],
        AutoEvalColumn.user.name: make_user_link(record.get("submitter", "")),
        AutoEvalColumn.user_name.name: record.get("submitter", ""),
        AutoEvalColumn.full_results.name: make_results_link(
            RESULTS_REPO, record.get("raw_scores_file", "")
        ),
        AutoEvalColumn.mean.name: _get(summary, "mean"),
        AutoEvalColumn.feas09.name: _get(summary, "feasibility_rate_0p9"),
        AutoEvalColumn.strict.name: _get(summary, "strict_feasibility"),
        AutoEvalColumn.strict_se.name: _proportion_se(
            _get(summary, "strict_feasibility"), summary.get("n")
        ),
        AutoEvalColumn.time_ms.name: _mean_time_ms(summary),
        AutoEvalColumn.median.name: _get(summary, "median"),
        AutoEvalColumn.p10.name: _get(summary, "p10"),
        AutoEvalColumn.n.name: summary.get("n", math.nan),
        AutoEvalColumn.n_parameters.name: record.get("n_parameters") or math.nan,
        AutoEvalColumn.submitted.name: record.get("submitted_time", ""),
        AutoEvalColumn.bench_version.name: record.get("bench_version", ""),
        AutoEvalColumn.revision.name: record.get("revision", ""),
        AutoEvalColumn.commit.name: make_clickable_commit(
            record["model"], record.get("commit_hash", "")
        ),
        AutoEvalColumn.onnx_file.name: record.get("onnx_path", "model.onnx"),
    }


def get_raw_eval_results(results_path: str) -> list[dict]:
    """Walk the results dir; return the raw record dict for each FINISHED result.

    Each record gets `raw_scores_file`, the repo path of its per-instance scores.
    """
    logger.debug("Reading eval results from %s", results_path)
    records: list[dict] = []
    for root, _, files in os.walk(results_path):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = Path(root) / fname
            record = json.loads(path.read_text())
            if record.get("status") != "FINISHED":
                continue
            # The runner writes the per-instance scores next to the summary, as
            # the same path with a .parquet suffix.
            record["raw_scores_file"] = (
                path.relative_to(results_path).with_suffix(".parquet").as_posix()
            )
            records.append(record)
    logger.debug("Loaded %d FINISHED result record(s)", len(records))
    return records
