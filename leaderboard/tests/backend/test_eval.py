import time

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
    monkeypatch.setattr(beval, "_load_track", lambda track, n: data)

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


def test_evaluate_model_forwards_n_instances(monkeypatch):
    data = [_eval_data("b0")]
    _patch_physics(monkeypatch, data[0].requirement_metrics)
    seen_n = []
    monkeypatch.setattr(beval, "_load_track", lambda track, n: (seen_n.append(n) or data))

    beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="abc",
        submitter="me", stochastic=False, k_samples=1,
        submitted_time="2026-06-11T00:00:00Z", n_instances=7,
    )
    # One call per track, each receiving the requested cap.
    assert seen_n == [7, 7]


def test_evaluate_model_all_instances_fail_raises(monkeypatch):
    import pytest

    data = [_eval_data("b0")]
    monkeypatch.setattr(beval, "_load_track", lambda track, n: data)

    def always_fail(boundary, coilset):
        raise RuntimeError("physics blew up")

    monkeypatch.setattr(beval, "_evaluate_achieved", always_fail)

    with pytest.raises(RuntimeError, match="failed on all"):
        beval.evaluate_model(
            model=FakeModel(), model_name="org/m", revision="abc",
            submitter="me", stochastic=False, k_samples=1,
            submitted_time="2026-06-11T00:00:00Z",
        )


def test_fmt_duration_formats_at_each_scale():
    assert beval._fmt_duration(3661) == "1h01m"
    assert beval._fmt_duration(125) == "2m05s"
    assert beval._fmt_duration(42) == "42s"
    # Past the deadline reads as negative time-remaining, not a wrapped positive.
    assert beval._fmt_duration(-5) == "-5s"


def test_read_deadline_parses_env(monkeypatch):
    monkeypatch.setenv("COILBENCH_EVAL_DEADLINE", "1234.5")
    assert beval._read_deadline() == 1234.5


def test_read_deadline_none_when_unset_or_garbage(monkeypatch):
    monkeypatch.delenv("COILBENCH_EVAL_DEADLINE", raising=False)
    assert beval._read_deadline() is None
    monkeypatch.setenv("COILBENCH_EVAL_DEADLINE", "not-a-number")
    assert beval._read_deadline() is None


def test_evaluate_track_logs_time_to_timeout(monkeypatch, caplog):
    data = [_eval_data("b0")]
    _patch_physics(monkeypatch, data[0].requirement_metrics)
    with caplog.at_level("INFO", logger="src.backend.eval"):
        beval.evaluate_track(FakeModel(), data, deadline=time.time() + 100)
    assert any("to timeout" in r.message for r in caplog.records)


def test_evaluate_track_warns_when_projected_to_overrun(monkeypatch, caplog):
    # Two instances with the deadline already in the past: after the first
    # instance the projection for the remaining one must exceed the (negative)
    # budget, so a warning fires.
    data = [_eval_data("b0"), _eval_data("b1")]
    _patch_physics(monkeypatch, data[0].requirement_metrics)
    with caplog.at_level("WARNING", logger="src.backend.eval"):
        beval.evaluate_track(FakeModel(), data, deadline=time.time() - 1)
    assert any("timeout" in r.message.lower() for r in caplog.records
               if r.levelname == "WARNING")


def test_evaluate_track_silent_without_deadline(monkeypatch, caplog):
    data = [_eval_data("b0")]
    _patch_physics(monkeypatch, data[0].requirement_metrics)
    with caplog.at_level("INFO", logger="src.backend.eval"):
        beval.evaluate_track(FakeModel(), data)
    assert not any("to timeout" in r.message for r in caplog.records)


def test_evaluate_model_records_commit_and_onnx_path(monkeypatch):
    data = [_eval_data("b0")]
    _patch_physics(monkeypatch, data[0].requirement_metrics)
    monkeypatch.setattr(beval, "_load_track", lambda track, n: data)

    record = beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="main",
        submitter="me", stochastic=False, k_samples=1,
        submitted_time="2026-06-11T00:00:00Z",
        commit_hash="deadbeefcafebabe", onnx_path="models/best.onnx",
    )
    assert record["commit_hash"] == "deadbeefcafebabe"
    assert record["onnx_path"] == "models/best.onnx"
