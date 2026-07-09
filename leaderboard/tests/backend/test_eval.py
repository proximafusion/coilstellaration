import time

import numpy as np
import pytest

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


def _eval_data(boundary_id="b0", field_error=0.1):
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
        max_normalized_field_error=field_error,
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
    score, n_failed, _times = beval.evaluate_track(FakeModel(), data)
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

    score, n_failed, _times = beval.evaluate_track(FakeModel(), data)
    aggregates = sorted(s.aggregate for s in score.per_instance)
    assert aggregates[0] == 0.0       # the failed one
    assert aggregates[1] == 1.0       # the good one
    assert n_failed == 1


def test_evaluate_model_builds_record(monkeypatch):
    tagged = [("tight", _eval_data("b0"))]
    _patch_physics(monkeypatch, tagged[0][1].requirement_metrics)
    monkeypatch.setattr(beval, "_load_track", lambda track, n: tagged)

    record, _rows = beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="abc",
        submitter="me", stochastic=False, k_samples=1,
        submitted_time="2026-06-11T00:00:00Z",
    )
    assert record["status"] == "FINISHED"
    assert record["overall_score"] == 1.0
    assert set(record["tracks"]) == {"fixed_shape", "variable_shape"}
    fixed = record["tracks"]["fixed_shape"]
    # Headline keys are the aggregation over ALL strata.
    assert fixed["mean"] == 1.0
    assert fixed["n"] == 1
    # Per-stratum breakdown alongside the headline.
    assert set(fixed["strata"]) == {"tight"}
    assert fixed["strata"]["tight"]["mean"] == 1.0
    assert fixed["strata"]["tight"]["n"] == 1


def test_evaluate_model_aggregates_per_stratum(monkeypatch):
    # Two instances in different strata: tight matches the achieved metrics
    # (perfect), loose tightens its field-error bound below the achieved value
    # so it is infeasible. Per-stratum aggregation must isolate each.
    tight = _eval_data("b0", field_error=0.1)
    loose = _eval_data("b1", field_error=0.001)
    tagged = [("tight", tight), ("loose", loose)]
    _patch_physics(monkeypatch, tight.requirement_metrics)  # achieved == tight target
    monkeypatch.setattr(beval, "_load_track", lambda track, n: tagged)

    record, _rows = beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="abc",
        submitter="me", stochastic=False, k_samples=1,
        submitted_time="2026-06-11T00:00:00Z",
    )
    fixed = record["tracks"]["fixed_shape"]
    assert fixed["n"] == 2
    assert set(fixed["strata"]) == {"tight", "loose"}
    # The subsets are scored independently.
    assert fixed["strata"]["tight"]["mean"] == 1.0
    assert fixed["strata"]["tight"]["strict_feasibility"] == 1.0
    assert fixed["strata"]["tight"]["n"] == 1
    assert fixed["strata"]["loose"]["mean"] < 1.0
    assert fixed["strata"]["loose"]["strict_feasibility"] == 0.0
    assert fixed["strata"]["loose"]["n"] == 1
    # Headline mean is over all instances (both strata pooled).
    assert fixed["mean"] < 1.0
    assert fixed["mean"] == (fixed["strata"]["tight"]["mean"]
                             + fixed["strata"]["loose"]["mean"]) / 2


def test_evaluate_model_forwards_n_instances(monkeypatch):
    tagged = [("tight", _eval_data("b0"))]
    _patch_physics(monkeypatch, tagged[0][1].requirement_metrics)
    seen_n = []
    monkeypatch.setattr(beval, "_load_track", lambda track, n: (seen_n.append(n) or tagged))

    beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="abc",
        submitter="me", stochastic=False, k_samples=1,
        submitted_time="2026-06-11T00:00:00Z", n_instances=7,
    )
    # One call per track, each receiving the requested cap.
    assert seen_n == [7, 7]


def test_evaluate_model_all_instances_fail_raises(monkeypatch):
    import pytest

    tagged = [("tight", _eval_data("b0"))]
    monkeypatch.setattr(beval, "_load_track", lambda track, n: tagged)

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
    tagged = [("tight", _eval_data("b0"))]
    _patch_physics(monkeypatch, tagged[0][1].requirement_metrics)
    monkeypatch.setattr(beval, "_load_track", lambda track, n: tagged)

    record, _rows = beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="main",
        submitter="me", stochastic=False, k_samples=1,
        submitted_time="2026-06-11T00:00:00Z",
        commit_hash="deadbeefcafebabe", onnx_path="models/best.onnx",
    )
    assert record["commit_hash"] == "deadbeefcafebabe"
    assert record["onnx_path"] == "models/best.onnx"


def test_evaluate_model_returns_raw_instance_rows(monkeypatch):
    tight = _eval_data("b0")
    loose = _eval_data("b1")
    tagged = [("tight", tight), ("loose", loose)]
    _patch_physics(monkeypatch, tight.requirement_metrics)
    monkeypatch.setattr(beval, "_load_track", lambda track, n: tagged)

    _record, rows = beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="abc",
        submitter="me", stochastic=False, k_samples=1,
        submitted_time="2026-06-11T00:00:00Z",
    )
    # 2 instances scored on each of the 2 tracks.
    assert len(rows) == 4
    assert {r["track"] for r in rows} == {"fixed_shape", "variable_shape"}
    assert {r["stratum"] for r in rows} == {"tight", "loose"}
    r = rows[0]
    assert {"track", "stratum", "boundary_id", "score", "strictly_feasible"} <= set(r)
    assert any(k.startswith("violation/") for k in r)
    assert any(k.startswith("soft_score/") for k in r)


def test_evaluate_model_scores_only_requested_tracks(monkeypatch):
    tagged = [("tight", _eval_data("b0"))]
    _patch_physics(monkeypatch, tagged[0][1].requirement_metrics)
    monkeypatch.setattr(beval, "_load_track", lambda track, n: tagged)

    record, _rows = beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="r", submitter="a",
        stochastic=False, k_samples=1, submitted_time="t",
        tracks=["fixed_shape"],
    )
    # Only fixed_shape scored; variable_shape present but empty.
    assert record["tracks"]["fixed_shape"] != {}
    assert record["tracks"]["variable_shape"] == {}
    # Overall is the mean over scored tracks only == the fixed-shape mean.
    assert record["overall_score"] == record["tracks"]["fixed_shape"]["mean"]


def test_evaluate_model_defaults_to_all_tracks(monkeypatch):
    tagged = [("tight", _eval_data("b0"))]
    _patch_physics(monkeypatch, tagged[0][1].requirement_metrics)
    monkeypatch.setattr(beval, "_load_track", lambda track, n: tagged)

    record, _rows = beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="r", submitter="a",
        stochastic=False, k_samples=1, submitted_time="t",
    )
    assert record["tracks"]["fixed_shape"] != {}
    assert record["tracks"]["variable_shape"] != {}


def test_time_stats_computes_mean_max_min_std():
    stats = beval._time_stats([1.0, 2.0, 3.0])
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
    assert stats["mean"] == 2.0
    assert stats["std"] == pytest.approx(float(np.std([1.0, 2.0, 3.0])))


def test_time_stats_ignores_non_finite():
    # A failed instance contributes a NaN time; it must not poison the stats.
    stats = beval._time_stats([1.0, float("nan"), 3.0])
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
    assert stats["mean"] == 2.0


def test_time_stats_none_when_no_finite_times():
    expected = {"mean": None, "max": None, "min": None, "std": None}
    assert beval._time_stats([]) == expected
    assert beval._time_stats([float("nan")]) == expected


def test_evaluate_track_returns_inference_times(monkeypatch):
    data = [_eval_data("b0"), _eval_data("b1")]
    _patch_physics(monkeypatch, data[0].requirement_metrics)
    score, n_failed, times = beval.evaluate_track(FakeModel(), data)
    # One finite, non-negative time per instance, aligned with per_instance.
    assert len(times) == len(score.per_instance) == 2
    assert all(np.isfinite(t) and t >= 0 for t in times)


def test_evaluate_track_inference_time_nan_when_inference_fails(monkeypatch):
    data = [_eval_data("b0")]
    _patch_physics(monkeypatch, data[0].requirement_metrics)

    class BadModel:
        def run(self, inputs):
            raise RuntimeError("inference blew up")

    score, n_failed, times = beval.evaluate_track(BadModel(), data)
    assert n_failed == 1
    assert len(times) == 1
    # No valid inference happened, so no time is attributed to the instance.
    assert not np.isfinite(times[0])


def test_evaluate_model_stratum_includes_inference_time_stats(monkeypatch):
    tight = _eval_data("b0", field_error=0.1)
    loose = _eval_data("b1", field_error=0.001)
    tagged = [("tight", tight), ("loose", loose)]
    _patch_physics(monkeypatch, tight.requirement_metrics)
    monkeypatch.setattr(beval, "_load_track", lambda track, n: tagged)

    record, _rows = beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="abc",
        submitter="me", stochastic=False, k_samples=1,
        submitted_time="2026-06-11T00:00:00Z",
    )
    fixed = record["tracks"]["fixed_shape"]
    keys = {"mean", "max", "min", "std"}
    # Headline (all strata pooled) and each stratum carry a timing block.
    assert set(fixed["inference_time"]) == keys
    assert fixed["inference_time"]["mean"] >= 0
    for stratum in ("tight", "loose"):
        sub = fixed["strata"][stratum]["inference_time"]
        assert set(sub) == keys
        assert sub["mean"] >= 0
        assert sub["min"] <= sub["mean"] <= sub["max"]


def test_evaluate_model_rows_include_inference_time(monkeypatch):
    tagged = [("tight", _eval_data("b0"))]
    _patch_physics(monkeypatch, tagged[0][1].requirement_metrics)
    monkeypatch.setattr(beval, "_load_track", lambda track, n: tagged)

    _record, rows = beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="abc",
        submitter="me", stochastic=False, k_samples=1,
        submitted_time="2026-06-11T00:00:00Z",
    )
    assert all("inference_time_s" in r for r in rows)
    assert all(np.isfinite(r["inference_time_s"]) for r in rows)


def test_rows_to_parquet_bytes_roundtrips():
    import io

    import pandas as pd
    rows = [{"track": "fixed_shape", "stratum": "tight", "boundary_id": "b0",
             "score": 0.5, "strictly_feasible": True}]
    blob = beval.rows_to_parquet_bytes(rows)
    df = pd.read_parquet(io.BytesIO(blob))
    assert df.iloc[0]["boundary_id"] == "b0"
    assert df.iloc[0]["score"] == 0.5
