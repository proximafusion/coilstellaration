import numpy as np
import pytest
from leaderboard.src.backend import evaluate
from coilstellaration import types
from constellaration.geometry import surface_rz_fourier


class FakeModel:
    def run(self, inputs):
        return {
            "coil_x_n": np.zeros((4, 5), np.float32),
            "coil_y_n": np.zeros((4, 5), np.float32),
            "coil_z_n": np.zeros((4, 5), np.float32),
            "currents": np.ones((4,), np.float32),
        }


def _desc_equilibrium():
    return types.DescEquilibrium(
        r_lmn=np.array([1.0, 0.1]),
        z_lmn=np.array([0.1]),
        lambda_lmn=np.array([0.0]),
        radial_resolution=1,
        poloidal_resolution=1,
        toroidal_resolution=0,
        n_field_periods=3,
        pressure_profile=types.DescSplineProfile(knots=np.array([0.0, 1.0]), params=np.array([1.0, 0.0])),
        r_axis=np.array([1.0]),
    )


def _eval_data(boundary_id, field_error=0.1):
    r = np.zeros((2, 3))
    boundary = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r.copy(), z_sin=r.copy(), n_field_periods=3, is_stellarator_symmetric=True
    )
    coilset = types.Coilset(
        coil_x_n=np.zeros((4, 5)),
        coil_y_n=np.zeros((4, 5)),
        coil_z_n=np.zeros((4, 5)),
        currents=np.ones((4,)),
        n_field_periods=3,
        is_stellarator_symmetric=True,
    )
    req = types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=0.1,
        min_normalized_coil_to_plasma_distance=0.2,
        max_normalized_coil_curvature=2.0,
        max_normalized_field_error=field_error,
        n_coils_per_half_period=5,
    )
    return types.EvalData(
        boundary=boundary,
        boundary_id=boundary_id,
        true_coilset=coilset,
        true_metrics="metrics-id",
        requirement_metrics=req,
        desc_equilibrium=_desc_equilibrium(),
    )


def test_evaluate_model_summarizes_each_stratum_and_keeps_unscored_tracks(monkeypatch):
    # The achieved metrics match the medium and tight targets, but loose tightens its
    # field-error bound below them, so only loose is infeasible.
    datas = {
        "loose": _eval_data("b0", field_error=0.001),
        "medium": _eval_data("b1"),
        "tight": _eval_data("b2"),
    }
    monkeypatch.setattr(
        evaluate.data_utils, "load_benchmark_dataset", lambda track, stratum, split, n: [datas[stratum]]
    )
    monkeypatch.setattr(
        evaluate, "_evaluate_achieved", lambda coilset, desc_equilibrium: datas["tight"].requirement_metrics
    )

    record, rows = evaluate.evaluate_model(model=FakeModel(), stochastic=False, k_samples=1, tracks=["fixed_shape"])

    assert record["tracks"]["variable_shape"] == {}
    fixed = record["tracks"]["fixed_shape"]
    strata = fixed["strata"]
    assert set(strata) == {"loose", "medium", "tight"}
    assert strata["tight"]["mean"] == 1.0
    assert strata["medium"]["mean"] == 1.0
    assert strata["loose"]["strict_feasibility"] == 0.0
    assert all(summary["n"] == 1 for summary in strata.values())
    # The headline pools every stratum's instances.
    assert fixed["n"] == 3
    assert fixed["mean"] == pytest.approx(np.mean([summary["mean"] for summary in strata.values()]))
    assert record["overall_score"] == fixed["mean"]
    assert [row["stratum"] for row in rows] == ["loose", "medium", "tight"]
