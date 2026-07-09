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


def result_to_row(record: dict, stratum: str = "All") -> dict:
    """Map one result record (spec §6) to a leaderboard dataframe row for one
    stratum view.

    For `stratum == "All"` the metric columns are the all-strata headline and
    `Overall` is the stored `overall_score`. For a specific stratum the columns
    are that stratum's breakdown and `Overall` is the mean across tracks of the
    stratum means (NaN if the stratum is absent from the record).
    """
    fixed = _track_summary(record["tracks"]["fixed_shape"], stratum)
    var = _track_summary(record["tracks"]["variable_shape"], stratum)

    if stratum == "All":
        overall = record["overall_score"]
    else:
        means = [_get(fixed, "mean"), _get(var, "mean")]
        overall = math.nan if any(math.isnan(m) for m in means) else sum(means) / len(means)

    return {
        AutoEvalColumn.model.name: make_clickable_model(record["model"]),
        AutoEvalColumn.submitter.name: record.get("submitter", ""),
        AutoEvalColumn.overall.name: overall,
        AutoEvalColumn.fixed_mean.name: _get(fixed, "mean"),
        AutoEvalColumn.fixed_feas09.name: _get(fixed, "feasibility_rate_0p9"),
        AutoEvalColumn.fixed_strict.name: _get(fixed, "strict_feasibility"),
        AutoEvalColumn.fixed_median.name: _get(fixed, "median"),
        AutoEvalColumn.fixed_p10.name: _get(fixed, "p10"),
        AutoEvalColumn.var_mean.name: _get(var, "mean"),
        AutoEvalColumn.var_feas09.name: _get(var, "feasibility_rate_0p9"),
        AutoEvalColumn.var_strict.name: _get(var, "strict_feasibility"),
        AutoEvalColumn.var_median.name: _get(var, "median"),
        AutoEvalColumn.var_p10.name: _get(var, "p10"),
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
