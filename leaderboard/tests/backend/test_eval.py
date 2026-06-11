import numpy as np

from constellaration.geometry import surface_rz_fourier
from coilstellaration import types

from src.backend import eval as beval


class FakeModel:
    def run(self, inputs):
        return {
            "coil_x_n": np.zeros((4, 5), np.float32),
            "coil_y_n": np.zeros((4, 5), np.float32),
            "coil_z_n": np.zeros((4, 5), np.float32),
            "currents": np.ones((4,), np.float32),
        }


def _eval_data(boundary_id="b0"):
    r = np.zeros((2, 3))
    boundary = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r.copy(), z_sin=r.copy(), n_field_periods=3, is_stellarator_symmetric=True
    )
    coilset = types.Coilset(
        coil_x_n=np.zeros((4, 5)), coil_y_n=np.zeros((4, 5)), coil_z_n=np.zeros((4, 5)),
        currents=np.ones((4,)), n_field_periods=3, is_stellarator_symmetric=True,
    )
    req = types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=0.1,
        min_normalized_coil_to_plasma_distance=0.2,
        max_normalized_coil_curvature=2.0,
        max_normalized_field_error=0.1,
    )
    return types.EvalData(
        boundary=boundary, boundary_id=boundary_id, true_coilset=coilset,
        true_metrics="metrics-id", requirement_metrics=req,
    )


def _patch_physics(monkeypatch, achieved):
    # achieved == target -> all violations 0 -> per-instance score 1.0
    monkeypatch.setattr(beval, "_evaluate_achieved", lambda boundary, coilset: achieved)


def test_evaluate_track_perfect_scores(monkeypatch):
    data = [_eval_data("b0"), _eval_data("b1")]
    _patch_physics(monkeypatch, data[0].requirement_metrics)
    score, n_failed = beval.evaluate_track(FakeModel(), data)
    assert score.summaries["mean"] == 1.0
    assert score.summaries["strict_feasibility"] == 1.0
    assert len(score.per_instance) == 2
    assert n_failed == 0


def test_failed_instance_scores_zero_and_continues(monkeypatch):
    data = [_eval_data("b0"), _eval_data("b1")]

    def flaky(boundary, coilset):
        if not flaky.called:
            flaky.called = True
            raise RuntimeError("physics blew up")
        return data[0].requirement_metrics
    flaky.called = False
    monkeypatch.setattr(beval, "_evaluate_achieved", flaky)

    score, n_failed = beval.evaluate_track(FakeModel(), data)
    aggregates = sorted(s.aggregate for s in score.per_instance)
    assert aggregates[0] == 0.0       # the failed one
    assert aggregates[1] == 1.0       # the good one
    assert n_failed == 1


def test_evaluate_model_builds_record(monkeypatch):
    data = [_eval_data("b0")]
    _patch_physics(monkeypatch, data[0].requirement_metrics)
    monkeypatch.setattr(beval, "_load_track", lambda track: data)

    record = beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="abc",
        submitter="me", stochastic=False, k_samples=1,
        submitted_time="2026-06-11T00:00:00Z",
    )
    assert record["status"] == "FINISHED"
    assert record["overall_score"] == 1.0
    assert set(record["tracks"]) == {"fixed_shape", "variable_shape"}
    assert record["tracks"]["fixed_shape"]["mean"] == 1.0
    assert record["tracks"]["fixed_shape"]["n"] == 1


def test_evaluate_model_all_instances_fail_raises(monkeypatch):
    import pytest

    data = [_eval_data("b0")]
    monkeypatch.setattr(beval, "_load_track", lambda track: data)

    def always_fail(boundary, coilset):
        raise RuntimeError("physics blew up")

    monkeypatch.setattr(beval, "_evaluate_achieved", always_fail)

    with pytest.raises(RuntimeError, match="failed on all"):
        beval.evaluate_model(
            model=FakeModel(), model_name="org/m", revision="abc",
            submitter="me", stochastic=False, k_samples=1,
            submitted_time="2026-06-11T00:00:00Z",
        )
