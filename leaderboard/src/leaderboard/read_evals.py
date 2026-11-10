import json
import logging
import math
import os

from src.display.formatting import make_clickable_commit, make_clickable_model
from src.display.utils import AutoEvalColumn

logger = logging.getLogger(__name__)


def _stochastic_label(record: dict) -> str:
    if record.get("stochastic"):
        return f"✓ (K={record.get('k_samples', 1)})"
    return "—"


def _track_summary(track_record: dict, stratum: str) -> dict | None:
    """The summary dict for one stratum view of a track.

    `"All"` maps to the track's headline (all-strata) keys; a specific stratum
    maps to its entry under `strata`, or `None` if this record predates the
    per-stratum breakdown.
    """
    if stratum == "All":
        return track_record
    return track_record.get("strata", {}).get(stratum)


def _get(summary: dict | None, key: str) -> float:
    """Metric value from a (possibly missing) summary; NaN when unavailable."""
    if not summary:
        return math.nan
    return summary.get(key, math.nan)


def result_to_row(record: dict, track: str, stratum: str) -> dict | None:
    """Map one result record (spec §6) to a leaderboard row for one track+stratum view,
    or `None` when the record has no data for that view.

    The metric columns are that track's numbers: for `stratum == "All"` the
    all-strata headline, for a specific stratum that stratum's breakdown.
    Returns `None` when the track wasn't scored (empty or missing summary) or
    the stratum is absent from the record (predates the per-stratum breakdown),
    so the frame-builder can drop the row from that view.
    """
    summary = _track_summary(record["tracks"].get(track, {}), stratum)
    if not summary:
        return None

    return {
        AutoEvalColumn.model.name: make_clickable_model(record["model"]),
        AutoEvalColumn.submitter.name: record.get("submitter", ""),
        AutoEvalColumn.mean.name: _get(summary, "mean"),
        AutoEvalColumn.feas09.name: _get(summary, "feasibility_rate_0p9"),
        AutoEvalColumn.strict.name: _get(summary, "strict_feasibility"),
        AutoEvalColumn.median.name: _get(summary, "median"),
        AutoEvalColumn.p10.name: _get(summary, "p10"),
        AutoEvalColumn.stochastic.name: _stochastic_label(record),
        AutoEvalColumn.submitted.name: record.get("submitted_time", ""),
        AutoEvalColumn.revision.name: record.get("revision", ""),
        AutoEvalColumn.commit.name: make_clickable_commit(record["model"], record.get("commit_hash", "")),
        AutoEvalColumn.onnx_file.name: record.get("onnx_path", "model.onnx"),
    }


def get_raw_eval_results(results_path: str) -> list[dict]:
    """Walk the results dir; return the raw record dict for each FINISHED result."""
    logger.debug("Reading eval results from %s", results_path)
    records: list[dict] = []
    for root, _, files in os.walk(results_path):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(root, fname)) as fp:
                record = json.load(fp)
            if record.get("status") != "FINISHED":
                continue
            records.append(record)
    logger.debug("Loaded %d FINISHED result record(s)", len(records))
    return records
