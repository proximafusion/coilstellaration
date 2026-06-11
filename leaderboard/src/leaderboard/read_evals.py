import json
import logging
import os

from src.display.formatting import make_clickable_model
from src.display.utils import AutoEvalColumn

logger = logging.getLogger(__name__)


def _stochastic_label(record: dict) -> str:
    if record.get("stochastic"):
        return f"✓ (K={record.get('k_samples', 1)})"
    return "—"


def result_to_row(record: dict) -> dict:
    """Map one result record (spec §6) to a leaderboard dataframe row."""
    fixed = record["tracks"]["fixed_shape"]
    var = record["tracks"]["variable_shape"]
    return {
        AutoEvalColumn.model.name: make_clickable_model(record["model"]),
        AutoEvalColumn.submitter.name: record.get("submitter", ""),
        AutoEvalColumn.overall.name: record["overall_score"],
        AutoEvalColumn.fixed_mean.name: fixed["mean"],
        AutoEvalColumn.fixed_feas09.name: fixed["feasibility_rate_0p9"],
        AutoEvalColumn.fixed_strict.name: fixed["strict_feasibility"],
        AutoEvalColumn.fixed_median.name: fixed["median"],
        AutoEvalColumn.fixed_p10.name: fixed["p10"],
        AutoEvalColumn.var_mean.name: var["mean"],
        AutoEvalColumn.var_feas09.name: var["feasibility_rate_0p9"],
        AutoEvalColumn.var_strict.name: var["strict_feasibility"],
        AutoEvalColumn.var_median.name: var["median"],
        AutoEvalColumn.var_p10.name: var["p10"],
        AutoEvalColumn.stochastic.name: _stochastic_label(record),
        AutoEvalColumn.submitted.name: record.get("submitted_time", ""),
        AutoEvalColumn.revision.name: record.get("revision", ""),
    }


def get_raw_eval_results(results_path: str) -> list[dict]:
    """Walk the results dir; return one display row per FINISHED result."""
    logger.debug("Reading eval results from %s", results_path)
    rows: list[dict] = []
    for root, _, files in os.walk(results_path):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(root, fname)) as fp:
                record = json.load(fp)
            if record.get("status") != "FINISHED":
                continue
            try:
                rows.append(result_to_row(record))
            except KeyError as exc:
                logger.warning("Skipping malformed result %s: missing key %s", fname, exc)
    logger.debug("Loaded %d FINISHED result row(s)", len(rows))
    return rows
