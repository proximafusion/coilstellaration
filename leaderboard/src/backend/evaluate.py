"""Run an ONNX model through the coilstellaration eval + scoring pipeline."""

import collections
import hashlib
import io
import itertools
import logging
import os
import time
from typing import Any, Literal, Sequence, get_args

import numpy as np
import pandas as pd
from desc.equilibrium import Equilibrium

from coilstellaration import (
    coilset_utils,
    data_utils,
    desc_utils,
    metrics_utils_v2,
    types,
)
from coilstellaration.benchmark import scoring, scoring_types
from leaderboard.src.backend import config, marshal, onnx_model

logger = logging.getLogger(__name__)

SUMMARY_KEYS = ("mean", "median", "p10", "feasibility_rate_0p9", "strict_feasibility")

DEADLINE_ENV = "COILBENCH_EVAL_DEADLINE"

_DESC_EQUILIBRIUM_CACHE_MAX = 512
_DESC_EQUILIBRIUM_CACHE: collections.OrderedDict[str, Equilibrium] = (
    collections.OrderedDict()
)


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


def _evaluate_achieved(
    coilset: types.Coilset,
    desc_equilibrium: types.DescEquilibrium,
) -> types.RequirementMetrics:
    """Predicted coilset -> achieved RequirementMetrics via DESC physics, on the
    equilibrium the targets were computed on.

    Isolated so tests can monkeypatch the expensive physics.
    """
    metrics = metrics_utils_v2.evaluate_coilset_metrics(
        eq=_cached_desc_equilibrium(desc_equilibrium),
        coilset=coilset_utils.coilstellaration_to_desc(
            coilset, check_intersection=False
        ),
        subset_metrics=True,
    )
    return data_utils.metrics_to_requirement_metrics(metrics)


def _desc_equilibrium_key(desc_equilibrium: types.DescEquilibrium) -> str:
    return hashlib.sha256(desc_equilibrium.model_dump_json().encode()).hexdigest()


def _cached_desc_equilibrium(desc_equilibrium: types.DescEquilibrium) -> Equilibrium:
    """`desc_equilibrium` as a DESC `Equilibrium`, LRU-cached by content.

    Every boundary is scored in each stratum and track it appears in, and rebuilding its
    equilibrium each time would redo DESC's basis and transform setup.
    """
    key = _desc_equilibrium_key(desc_equilibrium)
    eq = _DESC_EQUILIBRIUM_CACHE.get(key)
    if eq is not None:
        _DESC_EQUILIBRIUM_CACHE.move_to_end(key)
        return eq
    eq = desc_utils.desc_equilibrium_to_desc_object(desc_equilibrium)
    _DESC_EQUILIBRIUM_CACHE[key] = eq
    if len(_DESC_EQUILIBRIUM_CACHE) > _DESC_EQUILIBRIUM_CACHE_MAX:
        _DESC_EQUILIBRIUM_CACHE.popitem(last=False)
    return eq


def _achieved_key(
    coilset: types.Coilset,
    desc_equilibrium: types.DescEquilibrium,
) -> str:
    """Content hash of everything `_evaluate_achieved` reads.

    The `fixed_shape` rows reappear in `variable_shape` (about a quarter of all scored
    rows), so equal keys let the physics run once per distinct (plasma, coilset) pair.
    Keyed on the predicted coilset rather than the model inputs so a non-deterministic
    model can never be served another forward pass's metrics.
    """
    digest = hashlib.sha256()
    for arr in (
        coilset.coil_x_n,
        coilset.coil_y_n,
        coilset.coil_z_n,
        coilset.currents,
    ):
        a = np.ascontiguousarray(arr)
        digest.update(f"{a.dtype}{a.shape}".encode())
        digest.update(a.tobytes())
    digest.update(
        f"{coilset.n_field_periods},{coilset.is_stellarator_symmetric}".encode()
    )
    digest.update(_desc_equilibrium_key(desc_equilibrium).encode())
    return digest.hexdigest()


def _evaluate_achieved_cached(
    coilset: types.Coilset,
    desc_equilibrium: types.DescEquilibrium,
    cache: dict[str, types.RequirementMetrics] | None,
) -> types.RequirementMetrics:
    if cache is None:
        return _evaluate_achieved(coilset, desc_equilibrium)
    key = _achieved_key(coilset, desc_equilibrium)
    achieved = cache.get(key)
    if achieved is None:
        achieved = _evaluate_achieved(coilset, desc_equilibrium)
        cache[key] = achieved
    return achieved


def _time_stats(times: list[float]) -> dict[str, float | None]:
    """Summarize per-instance inference times into mean/max/min/std.

    Non-finite entries (NaN for instances whose inference never completed) are
    dropped. When nothing finite remains the four stats are `None` so the JSON
    summary stays valid (no non-standard NaN literals).
    """
    finite = [t for t in times if np.isfinite(t)]
    if not finite:
        return {"mean": None, "max": None, "min": None, "std": None}
    arr = np.asarray(finite, dtype=float)
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "std": float(arr.std()),
    }


def _summarize_instances(
    instances: list[scoring_types.InstanceScore],
) -> dict[str, Any]:
    """Aggregate a set of instance scores into the summary keys plus `n`."""
    summary: dict[str, Any] = {
        k: float(v)
        for k, v in scoring.aggregate_instances(
            instances, config.SCORING_SETTINGS
        ).items()
    }
    summary["n"] = len(instances)
    return summary


def _instance_scores_to_rows(
    track: str,
    stratum: str,
    per_instance: list[scoring_types.InstanceScore],
    times: list[float],
    coilsets: list[types.Coilset | None],
    **metadata: Any,
) -> list[dict]:
    """Flatten per-instance scores into raw rows for the parquet artifact."""
    rows: list[dict] = []
    for inst, t, coilset in zip(per_instance, times, coilsets, strict=True):
        row: dict = {
            "track": track,
            "stratum": stratum,
            "boundary_id": inst.boundary_id,
        }
        row |= scoring.instance_score_to_row(inst)
        row["inference_time_s"] = float(t)
        if coilset is None:
            row["coilset_json"] = None
        else:
            row["coilset_json"] = coilset.model_dump_json()
        row |= metadata
        rows.append(row)
    return rows


def rows_to_parquet_bytes(rows: list[dict]) -> bytes:
    """Serialize raw per-instance rows to an in-memory parquet blob."""
    buf = io.BytesIO()
    pd.DataFrame.from_records(rows).to_parquet(buf, index=False)
    return buf.getvalue()


def inference_model(
    model: onnx_model.OnnxModel,
    eval_datas: Sequence[types.EvalData],
    deadline: float | None = None,
) -> tuple[list[types.EvalData], list[float]]:
    eval_datas_maybe_with_inferenced_coilset: list[types.EvalData] = []
    inference_times: list[float] = []
    n_failed = 0
    n_total = len(eval_datas)
    logger.info("Inferencing %d coilsets", n_total)
    track_start = time.perf_counter()
    for i, eval_data in enumerate(eval_datas, 1):
        t0 = time.perf_counter()
        infer_s = float("nan")
        eval_data_maybe_with_inferenced_coilset = eval_data.model_copy()
        try:
            desc_equilibrium = eval_data.desc_equilibrium
            if desc_equilibrium is None:
                raise ValueError("no DESC equilibrium to score on")
            inputs = marshal.eval_data_to_inputs(
                eval_data.boundary, eval_data.requirement_metrics
            )
            t_infer = time.perf_counter()
            outputs = model.run(inputs)
            infer_s = time.perf_counter() - t_infer
            coilset = marshal.outputs_to_coilset(outputs, desc_equilibrium)
            eval_data_maybe_with_inferenced_coilset.predicted_coilset = coilset
            logger.info(
                "[%d/%d] %s: inferred in %.1fs (infer %.3fs)%s",
                i,
                n_total,
                eval_data.boundary_id,
                time.perf_counter() - t0,
                infer_s,
                _timeout_suffix(deadline),
            )
        except Exception as exc:  # noqa: BLE001 - one bad instance must not abort the run
            logger.warning(
                "[%d/%d] %s failed after %.1fs: %s",
                i,
                n_total,
                eval_data.boundary_id,
                time.perf_counter() - t0,
                exc,
            )
            n_failed += 1
        inference_times.append(infer_s)
        eval_datas_maybe_with_inferenced_coilset.append(
            eval_data_maybe_with_inferenced_coilset
        )

    elapsed = time.perf_counter() - track_start
    infer = _time_stats(inference_times)
    logger.info(
        "Coilsets inferenced in %.1fs: %d/%d ok, %d failed (%.2fs/instance; "
        "inference mean=%s max=%s min=%s std=%s)",
        elapsed,
        n_total - n_failed,
        n_total,
        n_failed,
        elapsed / n_total if n_total else 0.0,
        *(
            f"{infer[k]:.3f}s" if infer[k] is not None else "n/a"
            for k in ("mean", "max", "min", "std")
        ),
    )
    return eval_datas_maybe_with_inferenced_coilset, inference_times


def evaluate_data(
    eval_datas: list[types.EvalData],
    settings: scoring_types.ScoringSettings = config.SCORING_SETTINGS,
    deadline: float | None = None,
    achieved_cache: dict[str, types.RequirementMetrics] | None = None,
) -> tuple[scoring_types.BenchmarkScore, int]:
    instances: list[scoring_types.InstanceScore] = []
    n_failed = 0
    n_total = len(eval_datas)
    logger.info("Scoring %d instances", n_total)
    track_start = time.perf_counter()
    running_score = 0.0
    n_feasible = 0
    for i, data in enumerate(eval_datas, 1):
        t0 = time.perf_counter()
        try:
            predicted_coilset = data.predicted_coilset
            if predicted_coilset is None:
                raise ValueError("predicted coilset is None")
            desc_equilibrium = data.desc_equilibrium
            if desc_equilibrium is None:
                raise ValueError("desc_equilibrium is None")
            achieved = _evaluate_achieved_cached(
                predicted_coilset, desc_equilibrium, achieved_cache
            )
            if not _is_finite(achieved):
                raise ValueError("achieved metrics contain NaN/inf")
            score_inst = scoring.score_eval_data(data, achieved, settings)
            instances.append(score_inst)
            running_score += score_inst.aggregate
            n_feasible += int(score_inst.metric_scores.strictly_feasible)
            logger.info(
                "[%d/%d] %s: score=%.4f feasible=%s in %.1fs "
                "(running mean=%.4f, feasible=%d/%d)%s",
                i,
                n_total,
                data.boundary_id,
                score_inst.aggregate,
                score_inst.metric_scores.strictly_feasible,
                time.perf_counter() - t0,
                running_score / i,
                n_feasible,
                i,
                _timeout_suffix(deadline),
            )
        except Exception as exc:  # noqa: BLE001 - one bad instance must not abort the run
            logger.warning(
                "[%d/%d] %s failed after %.1fs: %s",
                i,
                n_total,
                data.boundary_id,
                time.perf_counter() - t0,
                exc,
            )
            instances.append(scoring.zero_instance_score(data.boundary_id))
            n_failed += 1

        if deadline is not None and i < n_total:
            remaining = deadline - time.time()
            projected = (time.perf_counter() - track_start) / i * (n_total - i)
            if projected > remaining:
                logger.warning(
                    "Behind schedule: ~%s to finish %d remaining instances "
                    "but only %s left before timeout",
                    _fmt_duration(projected),
                    n_total - i,
                    _fmt_duration(remaining),
                )
    elapsed = time.perf_counter() - track_start
    logger.info(
        "Track scored in %.1fs: %d/%d ok, %d failed, %d strictly feasible "
        "(%.2fs/instance)",
        elapsed,
        n_total - n_failed,
        n_total,
        n_failed,
        n_feasible,
        elapsed / n_total if n_total else 0.0,
    )
    return scoring.score_benchmark(instances, settings), n_failed


def _is_finite(achieved: types.RequirementMetrics) -> bool:
    return bool(np.all(np.isfinite(np.asarray(achieved.to_array(), dtype=float))))


def evaluate_model(
    model: onnx_model.OnnxModel,
    tracks: Sequence[Literal["fixed_shape", "variable_shape"]] = get_args(
        config.TRACKS
    ),
) -> tuple[dict, list[dict]]:
    """Score a model on every instance of the `eval` split of each track."""
    deadline = _read_deadline()
    logger.info(
        "Evaluating model on the full eval split%s",
        f" [{_fmt_duration(deadline - time.time())} budget before timeout]"
        if deadline is not None
        else " [no timeout]",
    )
    model_start = time.perf_counter()
    # Per track: the summary stats, `n`, `inference_time` and per-stratum `strata`.
    track_results: dict[str, dict[str, Any]] = {}
    track_scores: dict[str, list[scoring_types.InstanceScore]] = {}
    track_inference_times: dict[str, list[float]] = {}
    instance_rows: list[dict] = []
    total_instances = 0
    total_failed = 0
    achieved_cache: dict[str, types.RequirementMetrics] = {}
    for track, stratum in itertools.product(tracks, get_args(config.STRATA)):
        logger.info(
            "Loading track %s with stratum %s %s",
            track,
            stratum,
            _timeout_suffix(deadline),
        )
        load_start = time.perf_counter()
        eval_datas = data_utils.load_benchmark_dataset(
            track=track, stratum=stratum, split="eval", n=0
        )
        logger.info(
            "Loaded %d instances for track %s and stratum %s in %.1fs",
            len(eval_datas),
            track,
            stratum,
            time.perf_counter() - load_start,
        )
        eval_datas_maybe_with_inferenced_coilset, inference_times = inference_model(
            model=model, eval_datas=eval_datas, deadline=deadline
        )
        inferenced_coilsets = [
            f.predicted_coilset for f in eval_datas_maybe_with_inferenced_coilset
        ]

        score, n_failed = evaluate_data(
            eval_datas_maybe_with_inferenced_coilset,
            deadline=deadline,
            achieved_cache=achieved_cache,
        )
        track_scores[track] = track_scores.setdefault(track, [])
        track_scores[track].extend(score.per_instance)
        track_inference_times[track] = track_inference_times.setdefault(track, [])
        track_inference_times[track].extend(inference_times)

        instance_rows.extend(
            _instance_scores_to_rows(
                track, stratum, score.per_instance, inference_times, inferenced_coilsets
            )
        )
        track_results[track] = track_results.setdefault(track, {"strata": {}})
        summary = _summarize_instances(score.per_instance)
        time_stats = _time_stats(inference_times)
        summary["inference_time"] = time_stats
        track_results[track]["strata"][stratum] = summary
        total_instances += len(eval_datas)
        total_failed += n_failed
        logger.info(
            "Track %s / stratum %s done: mean=%.4f median=%.4f p10=%.4f "
            "feasibility@0.9=%.4f strict_feasibility=%.4f (n=%d, %d failed)",
            track,
            stratum,
            summary["mean"],
            summary["median"],
            summary["p10"],
            summary["feasibility_rate_0p9"],
            summary["strict_feasibility"],
            summary["n"],
            n_failed,
        )

    for track in tracks:
        summary = _summarize_instances(track_scores[track])
        time_stats = _time_stats(track_inference_times[track])
        summary["inference_time"] = time_stats
        track_results[track] |= summary

    if total_instances > 0 and total_failed == total_instances:
        raise RuntimeError(
            f"Model failed on all {total_instances} instances across tracks"
        )

    # Unscored tracks stay as `{}` so every record has the same keys as the ones already
    # published to the results repo.
    for track in get_args(config.TRACKS):
        track_results.setdefault(track, {})

    overall = float(np.mean([track_results[t]["mean"] for t in tracks]))
    logger.info(
        "Model FINISHED: overall=%.4f in %.1fs "
        "(%d instances across %d tracks, %d failed)",
        overall,
        time.perf_counter() - model_start,
        total_instances,
        len(tracks),
        total_failed,
    )
    record = {
        "status": "FINISHED",
        "bench_version": config.BENCH_VERSION,
        "dataset_revision": data_utils.COILSTELLARATION_REPO_REVISION,
        "overall_score": overall,
        "tracks": track_results,
    }
    return record, instance_rows
