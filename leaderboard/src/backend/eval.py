"""Run an ONNX model through the coilstellaration eval + scoring pipeline."""
import logging

import numpy as np
from coilstellaration import coilset_utils, data_utils, metrics_utils_v2, types
from coilstellaration.benchmark import scoring
from coilstellaration.benchmark import types as benchmark_types
from constellaration.geometry import surface_rz_fourier, surface_utils_desc

from src.backend import biot_savart_patch, config, marshal
from src.backend import data as benchmark_data

logger = logging.getLogger(__name__)

SUMMARY_KEYS = ("mean", "median", "p10", "feasibility_rate_0p9", "strict_feasibility")


def _evaluate_achieved(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    coilset: types.Coilset,
) -> types.RequirementMetrics:
    """Predicted coilset -> achieved RequirementMetrics via DESC physics.

    Isolated so tests can monkeypatch the expensive physics.
    """
    biot_savart_patch.apply_patch()
    desc_surface = surface_utils_desc.to_desc_fourier_rz_toroidal_surface(boundary)
    desc_coilset = coilset_utils.coilstellaration_to_desc(coilset)
    metrics = metrics_utils_v2.evaluate_coilset_metrics_from_boundary(
        boundary=desc_surface, coilset=desc_coilset, subset_metrics=True
    )
    return data_utils.metrics_to_requirement_metrics(metrics)


def _load_track(track: str, n: int) -> list[types.EvalData]:
    """Load benchmark instances for a track. Isolated for monkeypatching.

    `n` caps the instances per track (0 = full split). If `n` exceeds the
    number of available instances the loader simply stops early and returns
    them all.
    """
    return benchmark_data.load_benchmark_dataset(
        track=track, stratum=config.STRATUM, split=config.SPLIT, n=n
    )


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
) -> tuple[benchmark_types.BenchmarkScore, int]:
    """Score one model on one track's instances. Single forward pass per instance.

    Returns the aggregated `BenchmarkScore` and the number of instances that
    failed (and were scored 0.0). The failure count lets `evaluate_model`
    distinguish a model that genuinely scores 0 from one that errored on every
    instance (which should be reported as FAILED, not FINISHED).
    """
    instances: list[benchmark_types.InstanceScore] = []
    n_failed = 0
    logger.debug("Scoring %d instances", len(eval_datas))
    for data in eval_datas:
        try:
            inputs = marshal.eval_data_to_inputs(data.boundary, data.requirement_metrics)
            outputs = model.run(inputs)
            coilset = marshal.outputs_to_coilset(outputs, data.boundary)
            achieved = _evaluate_achieved(data.boundary, coilset)
            if not _is_finite(achieved):
                raise ValueError("achieved metrics contain NaN/inf")
            instances.append(scoring.score_eval_data(data, achieved, settings))
            logger.debug("Instance %s scored", data.boundary_id)
        except Exception as exc:  # noqa: BLE001 - one bad instance must not abort the run
            logger.warning("Instance %s failed: %s", data.boundary_id, exc)
            instances.append(_failed_instance(data.boundary_id))
            n_failed += 1
    logger.debug("Track scored: %d/%d instances failed", n_failed, len(eval_datas))
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
    logger.info(
        "Evaluating model %s (revision %s, n_instances=%s)",
        model_name, revision, n_instances,
    )
    tracks: dict[str, dict] = {}
    total_instances = 0
    total_failed = 0
    for track in config.TRACKS:
        logger.info("Loading track %s", track)
        eval_datas = _load_track(track, n_instances)
        score, n_failed = evaluate_track(model, eval_datas)
        summary = {k: float(score.summaries[k]) for k in SUMMARY_KEYS}
        summary["n"] = len(score.per_instance)
        tracks[track] = summary
        total_instances += len(eval_datas)
        total_failed += n_failed
        logger.info("Track %s mean=%.4f (%d instances)", track, summary["mean"], summary["n"])

    # A model that errored on every instance is a broken submission, not a
    # legitimate 0.0 score: surface it as FAILED (the runner skips the results
    # push and marks the request FAILED).
    if total_instances > 0 and total_failed == total_instances:
        raise RuntimeError(
            f"Model failed on all {total_instances} instances across tracks"
        )

    overall = float(np.mean([tracks[t]["mean"] for t in config.TRACKS]))
    logger.info("Model %s FINISHED with overall score %.4f", model_name, overall)
    return {
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
