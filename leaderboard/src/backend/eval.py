"""Run an ONNX model through the coilstellaration eval + scoring pipeline."""
import numpy as np

from coilstellaration import coilset_utils, data_utils, metrics_utils_v2
from coilstellaration import types
from coilstellaration.benchmark import scoring
from coilstellaration.benchmark import types as benchmark_types
from constellaration.geometry import surface_rz_fourier
from constellaration.geometry import surface_utils_desc

from src.backend import config, marshal

SUMMARY_KEYS = ("mean", "median", "p10", "feasibility_rate_0p9", "strict_feasibility")


def _evaluate_achieved(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    coilset: types.Coilset,
) -> types.RequirementMetrics:
    """Predicted coilset -> achieved RequirementMetrics via DESC physics.

    Isolated so tests can monkeypatch the expensive physics.
    """
    desc_surface = surface_utils_desc.to_desc_fourier_rz_toroidal_surface(boundary)
    desc_coilset = coilset_utils.coilstellaration_to_desc(coilset)
    metrics = metrics_utils_v2.evaluate_coilset_metrics_from_boundary(
        boundary=desc_surface, coilset=desc_coilset
    )
    return data_utils.metrics_to_requirement_metrics(metrics)


def _load_track(track: str) -> list[types.EvalData]:
    """Load benchmark instances for a track. Isolated for monkeypatching."""
    return data_utils.load_benchmark_dataset(
        track=track, stratum=config.STRATUM, split=config.SPLIT, n=config.N_INSTANCES
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
) -> benchmark_types.BenchmarkScore:
    """Score one model on one track's instances. Single forward pass per instance."""
    instances: list[benchmark_types.InstanceScore] = []
    for data in eval_datas:
        try:
            inputs = marshal.eval_data_to_inputs(data.boundary, data.requirement_metrics)
            outputs = model.run(inputs)
            coilset = marshal.outputs_to_coilset(outputs, data.boundary)
            achieved = _evaluate_achieved(data.boundary, coilset)
            if not _is_finite(achieved):
                raise ValueError("achieved metrics contain NaN/inf")
            instances.append(scoring.score_eval_data(data, achieved, settings))
        except Exception as exc:  # noqa: BLE001 - one bad instance must not abort the run
            print(f"[eval] instance {data.boundary_id} failed: {exc}")
            instances.append(_failed_instance(data.boundary_id))
    return scoring.score_benchmark(instances, settings)


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
) -> dict:
    """Evaluate a model on all tracks and assemble the result record (spec §6)."""
    tracks: dict[str, dict] = {}
    for track in config.TRACKS:
        eval_datas = _load_track(track)
        score = evaluate_track(model, eval_datas)
        summary = {k: float(score.summaries[k]) for k in SUMMARY_KEYS}
        summary["n"] = len(score.per_instance)
        tracks[track] = summary

    overall = float(np.mean([tracks[t]["mean"] for t in config.TRACKS]))
    return {
        "model": model_name,
        "revision": revision,
        "submitter": submitter,
        "submitted_time": submitted_time,
        "stochastic": stochastic,
        "k_samples": k_samples,
        "status": "FINISHED",
        "overall_score": overall,
        "tracks": tracks,
    }
