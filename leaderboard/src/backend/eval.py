"""Run an ONNX model through the coilstellaration eval + scoring pipeline."""
import io
import logging
import os
import time

import numpy as np
import pandas as pd
from coilstellaration import data_utils, metrics_utils_v2, types
from coilstellaration.benchmark import scoring
from coilstellaration.benchmark import types as benchmark_types
from constellaration.geometry import surface_rz_fourier, surface_utils_desc
from desc.coils import CoilSet, FourierXYZCoil

from src.backend import (
    biot_savart_patch,
    coil_symmetry_patch,
    config,
    marshal,
    plasma_distance_patch,
)
from src.backend import data as benchmark_data

logger = logging.getLogger(__name__)

SUMMARY_KEYS = ("mean", "median", "p10", "feasibility_rate_0p9", "strict_feasibility")

# Env carrying the absolute wall-clock deadline (epoch seconds) for the whole
# eval.
DEADLINE_ENV = "COILBENCH_EVAL_DEADLINE"

metrics_utils_v2._EVALUATOR_CACHE_MAX = 32


def _read_deadline() -> float | None:
    """Absolute epoch deadline for the eval, or None if no timeout is in force."""
    raw = os.environ.get(DEADLINE_ENV)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _fmt_duration(seconds: float) -> str:
    """Human-readable `1h23m` / `4m05s` / `42s`; signed so overruns read as negative."""
    total = int(round(seconds))
    sign = "-" if total < 0 else ""
    total = abs(total)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{sign}{h}h{m:02d}m"
    if m:
        return f"{sign}{m}m{s:02d}s"
    return f"{sign}{s}s"


def _timeout_suffix(deadline: float | None) -> str:
    """`` if no deadline, else ` [1h23m to timeout]` for appending to a log line."""
    if deadline is None:
        return ""
    return f" [{_fmt_duration(deadline - time.time())} to timeout]"


def _to_desc_coilset(coilset: types.Coilset) -> CoilSet:
    """Build a DESC `CoilSet` from a coilstellaration `Coilset`.

    Mirrors `coilset_utils.coilstellaration_to_desc` but passes
    `check_intersection=False`. The intersection check (`is_self_intersecting`,
    ~1.3s/instance) only computes a temporary `CoilSetMinDistance` to raise a
    user warning; its result is discarded and never feeds the metrics, so
    skipping it leaves the achieved metrics bit-for-bit unchanged while removing
    the dominant per-instance overhead.
    """
    desc_coils = [
        FourierXYZCoil(
            X_n=np.asarray(coilset.coil_x_n[i]),
            Y_n=np.asarray(coilset.coil_y_n[i]),
            Z_n=np.asarray(coilset.coil_z_n[i]),
            current=float(coilset.currents[i]),
        )
        for i in range(coilset.n_unique_coils)
    ]
    return CoilSet(
        *desc_coils,
        NFP=int(coilset.n_field_periods),
        sym=coilset.is_stellarator_symmetric,
        check_intersection=False,
    )


def _evaluate_achieved(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    coilset: types.Coilset,
) -> types.RequirementMetrics:
    """Predicted coilset -> achieved RequirementMetrics via DESC physics.

    Isolated so tests can monkeypatch the expensive physics.
    """
    biot_savart_patch.apply_patch()
    coil_symmetry_patch.apply_patch()
    # Must follow coil_symmetry_patch: the plasma single-period reduction is
    # only valid once the per-coil loop is restricted to the first field period.
    plasma_distance_patch.apply_patch()
    desc_surface = surface_utils_desc.to_desc_fourier_rz_toroidal_surface(boundary)
    desc_coilset = _to_desc_coilset(coilset)
    metrics = metrics_utils_v2.evaluate_coilset_metrics_from_boundary(
        boundary=desc_surface, coilset=desc_coilset, subset_metrics=True
    )
    return data_utils.metrics_to_requirement_metrics(metrics)


def _load_track(track: str, n: int) -> list[tuple[str, types.EvalData]]:
    """Load benchmark instances for a track, tagged with their stratum.

    Loads every stratum in `config.STRATA` and returns `(stratum, EvalData)`
    pairs so the caller can aggregate per-stratum as well as over the pool.
    Isolated for monkeypatching.

    `n` caps the instances *per stratum* (0 = full split). If `n` exceeds the
    number of available instances the loader simply stops early and returns
    them all.
    """
    tagged: list[tuple[str, types.EvalData]] = []
    for stratum in config.STRATA:
        datas = benchmark_data.load_benchmark_dataset(
            track=track, stratum=stratum, split=config.SPLIT, n=n
        )
        tagged.extend((stratum, d) for d in datas)
    return tagged


def _summarize(instances: list[benchmark_types.InstanceScore]) -> dict:
    """Aggregate a set of instance scores into the summary keys plus `n`."""
    summary = {
        k: float(v)
        for k, v in scoring.aggregate_instances(instances, config.SCORING_SETTINGS).items()
    }
    summary["n"] = len(instances)
    return summary


def _instance_rows(
    track: str,
    strata: list[str],
    per_instance: list[benchmark_types.InstanceScore],
) -> list[dict]:
    """Flatten per-instance scores into raw rows for the parquet artifact.

    Each row carries `track`, `stratum`, `boundary_id` and the flattened
    per-metric violations / soft scores / aggregate produced by
    `scoring.instance_score_to_row` (the same schema the paper's per-model
    score parquets use).
    """
    rows: list[dict] = []
    for stratum, inst in zip(strata, per_instance):
        row = {"track": track, "stratum": stratum, "boundary_id": inst.boundary_id}
        row.update(scoring.instance_score_to_row(inst))
        rows.append(row)
    return rows


def rows_to_parquet_bytes(rows: list[dict]) -> bytes:
    """Serialize raw per-instance rows to an in-memory parquet blob."""
    buf = io.BytesIO()
    pd.DataFrame.from_records(rows).to_parquet(buf, index=False)
    return buf.getvalue()


def _failed_instance(boundary_id: str) -> benchmark_types.InstanceScore:
    zeros = {f: 0.0 for f in benchmark_types.REQUIREMENT_METRIC_DIRECTIONS}
    return benchmark_types.InstanceScore(
        boundary_id=boundary_id,
        metric_scores=benchmark_types.MetricScores(
            violations=zeros, soft_scores=zeros, strictly_feasible=False
        ),
        aggregate=0.0,
    )


def evaluate_track(
    model,
    eval_datas: list[types.EvalData],
    settings: benchmark_types.ScoringSettings = config.SCORING_SETTINGS,
    deadline: float | None = None,
) -> tuple[benchmark_types.BenchmarkScore, int]:
    """Score one model on one track's instances. Single forward pass per instance.

    Returns the aggregated `BenchmarkScore` and the number of instances that
    failed (and were scored 0.0). The failure count lets `evaluate_model`
    distinguish a model that genuinely scores 0 from one that errored on every
    instance (which should be reported as FAILED, not FINISHED).

    `deadline` is the absolute epoch time at which the parent runner will kill
    the eval; when set, each instance logs the remaining budget and a warning
    fires if the run is projected to overrun it.
    """
    instances: list[benchmark_types.InstanceScore] = []
    n_failed = 0
    n_total = len(eval_datas)
    logger.info("Scoring %d instances", n_total)
    # Cheap running stats so progress is visible without a second pass: the
    # benchmark mean weights failed instances as 0.0 (they enter `instances`
    # with aggregate 0.0), so `running_score / i` is the running benchmark mean
    # over the first `i` instances.
    track_start = time.perf_counter()
    running_score = 0.0
    n_feasible = 0
    for i, data in enumerate(eval_datas, 1):
        t0 = time.perf_counter()
        try:
            inputs = marshal.eval_data_to_inputs(data.boundary, data.requirement_metrics)
            outputs = model.run(inputs)
            coilset = marshal.outputs_to_coilset(outputs, data.boundary)
            achieved = _evaluate_achieved(data.boundary, coilset)
            if not _is_finite(achieved):
                raise ValueError("achieved metrics contain NaN/inf")
            score_inst = scoring.score_eval_data(data, achieved, settings)
            instances.append(score_inst)
            running_score += score_inst.aggregate
            n_feasible += int(score_inst.metric_scores.strictly_feasible)
            logger.info(
                "[%d/%d] %s: score=%.4f feasible=%s in %.1fs "
                "(running mean=%.4f, feasible=%d/%d)%s",
                i, n_total, data.boundary_id, score_inst.aggregate,
                score_inst.metric_scores.strictly_feasible, time.perf_counter() - t0,
                running_score / i, n_feasible, i,
                _timeout_suffix(deadline),
            )
        except Exception as exc:  # noqa: BLE001 - one bad instance must not abort the run
            logger.warning(
                "[%d/%d] %s failed after %.1fs: %s",
                i, n_total, data.boundary_id, time.perf_counter() - t0, exc,
            )
            instances.append(_failed_instance(data.boundary_id))
            n_failed += 1
        # Project from the average so far whether the rest of the track fits the
        # budget; warn once we're behind so an impending kill isn't a surprise.
        if deadline is not None and i < n_total:
            remaining = deadline - time.time()
            projected = (time.perf_counter() - track_start) / i * (n_total - i)
            if projected > remaining:
                logger.warning(
                    "Behind schedule: ~%s to finish %d remaining instances but only "
                    "%s left before timeout",
                    _fmt_duration(projected), n_total - i, _fmt_duration(remaining),
                )
    elapsed = time.perf_counter() - track_start
    logger.info(
        "Track scored in %.1fs: %d/%d ok, %d failed, %d strictly feasible (%.2fs/instance)",
        elapsed, n_total - n_failed, n_total, n_failed, n_feasible,
        elapsed / n_total if n_total else 0.0,
    )
    return scoring.score_benchmark(instances, settings), n_failed


def _is_finite(achieved: types.RequirementMetrics) -> bool:
    return bool(np.all(np.isfinite(np.asarray(achieved.to_array(), dtype=float))))


def evaluate_model(
    model,
    model_name: str,
    revision: str,
    submitter: str,
    stochastic: bool,
    k_samples: int,
    submitted_time: str,
    n_instances: int = config.N_INSTANCES,
    commit_hash: str = "",
    onnx_path: str = "model.onnx",
) -> dict:
    """Evaluate a model on all tracks and assemble the result record (spec §6).

    `n_instances` caps the number of instances scored per track (0 = full
    split). Values larger than the available instance count harmlessly score
    the full split. Defaults to the `COILBENCH_N` env override.
    """
    deadline = _read_deadline()
    logger.info(
        "Evaluating model %s (revision %s, n_instances=%s)%s",
        model_name, revision, n_instances,
        f" [{_fmt_duration(deadline - time.time())} budget before timeout]"
        if deadline is not None else " [no timeout]",
    )
    model_start = time.perf_counter()
    tracks: dict[str, dict] = {}
    instance_rows: list[dict] = []
    total_instances = 0
    total_failed = 0
    for track in config.TRACKS:
        logger.info(
            "Loading track %s%s", track, _timeout_suffix(deadline),
        )
        load_start = time.perf_counter()
        tagged = _load_track(track, n_instances)
        strata = [stratum for stratum, _ in tagged]
        eval_datas = [data for _, data in tagged]
        logger.info(
            "Loaded %d instances for track %s in %.1fs",
            len(eval_datas), track, time.perf_counter() - load_start,
        )
        score, n_failed = evaluate_track(model, eval_datas, deadline=deadline)
        # `score.per_instance` is aligned 1:1 with `eval_datas` (and hence
        # `strata`), including the 0.0-scored failures, so we can slice it by
        # stratum without re-scoring.
        summary = _summarize(score.per_instance)  # over all strata
        summary["strata"] = {
            stratum: _summarize(
                [inst for inst, s in zip(score.per_instance, strata) if s == stratum]
            )
            for stratum in config.STRATA
            if stratum in strata
        }
        tracks[track] = summary
        instance_rows.extend(_instance_rows(track, strata, score.per_instance))
        total_instances += len(eval_datas)
        total_failed += n_failed
        logger.info(
            "Track %s done (all strata): mean=%.4f median=%.4f p10=%.4f "
            "feasibility@0.9=%.4f strict_feasibility=%.4f (n=%d, %d failed)",
            track, summary["mean"], summary["median"], summary["p10"],
            summary["feasibility_rate_0p9"], summary["strict_feasibility"],
            summary["n"], n_failed,
        )
        for stratum, sub in summary["strata"].items():
            logger.info(
                "  %s/%s: mean=%.4f median=%.4f feasibility@0.9=%.4f "
                "strict_feasibility=%.4f (n=%d)",
                track, stratum, sub["mean"], sub["median"],
                sub["feasibility_rate_0p9"], sub["strict_feasibility"], sub["n"],
            )

    # A model that errored on every instance is a broken submission, not a
    # legitimate 0.0 score: surface it as FAILED (the runner skips the results
    # push and marks the request FAILED).
    if total_instances > 0 and total_failed == total_instances:
        raise RuntimeError(
            f"Model failed on all {total_instances} instances across tracks"
        )

    overall = float(np.mean([tracks[t]["mean"] for t in config.TRACKS]))
    logger.info(
        "Model %s FINISHED: overall=%.4f in %.1fs "
        "(%d instances across %d tracks, %d failed)",
        model_name, overall, time.perf_counter() - model_start,
        total_instances, len(config.TRACKS), total_failed,
    )
    record = {
        "model": model_name,
        "revision": revision,
        "commit_hash": commit_hash,
        "onnx_path": onnx_path,
        "submitter": submitter,
        "submitted_time": submitted_time,
        "stochastic": stochastic,
        "k_samples": k_samples,
        "status": "FINISHED",
        "overall_score": overall,
        "tracks": tracks,
    }
    return record, instance_rows
