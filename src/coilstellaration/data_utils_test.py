from collections.abc import Hashable

import numpy as np
import orjson
import pandas as pd
import pytest
from coilstellaration import data_utils, types
from desc import coils, grid

_FIXED_COLUMN = "baseline_labels/used_for_ml_baseline"
_QUALITY_COLUMN = "baseline_labels/meets_relative_error_thresholds"
_MACHINE_PRECISION = float(np.finfo(float).eps)
_MINOR_RADIUS = 0.5
# In polar form the coil is r(s) = 1 + 0.4 cos 2s, dented inwards at s = pi/2. Its
# curvature (r^2 + 2r'^2 - r r'') / (r^2 + r'^2)^(3/2) is -25/9 there, the largest
# magnitude on the coil; the convex peak at s = 0 is only 4.2 / 1.4^3.
_DENT_CURVATURE = 25 / 9


def _coilset_json(n_coils: int, nfp: int) -> str:
    return orjson.dumps(
        {"coil_x_n": [[0.0] * 7 for _ in range(n_coils)], "n_field_periods": nfp}
    ).decode()


def _row(
    coilset_id: str,
    nfp: int,
    n_coils: int,
    *,
    used_for_baseline: bool | None,
    meets: bool | None,
    has_coilset: bool = True,
) -> dict[str, object]:
    return {
        "desc_coilset_id": coilset_id,
        _FIXED_COLUMN: used_for_baseline,
        data_utils.VariableShapeTrackColumn: True,
        _QUALITY_COLUMN: meets,
        data_utils.N_COILS_COLUMN: n_coils,
        "json_desc_coilset": _coilset_json(n_coils, nfp) if has_coilset else None,
        "json_constellaration_boundary": orjson.dumps(
            {"n_field_periods": nfp}
        ).decode(),
    }


@pytest.fixture(scope="module")
def dented_coil_metrics() -> types.Metrics:
    # r(s) (cos s, sin s) = (1.2 cos s + 0.2 cos 3s, 0.8 sin s + 0.2 sin 3s), in DESC's
    # layout [sin 3s, sin 2s, sin s, 1, cos s, cos 2s, cos 3s].
    coil = coils.FourierXYZCoil(
        X_n=[0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.2],
        Y_n=[0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
        Z_n=[0.0] * 7,
    )
    # 400 points put one exactly on the dent.
    eval_grid = grid.LinearGrid(zeta=2 * np.pi * np.arange(400) / 400)
    curvatures = np.asarray(coil.compute("curvature", grid=eval_grid)["curvature"])
    return types.Metrics(
        surf_eval_coords=np.zeros((2, 1, 1)),
        coil_eval_params=np.asarray(eval_grid.nodes[:, 2]),
        coil_currents=np.ones(1),
        normalized_field_error=np.zeros((1, 1)),
        coil_to_coil_min_distances=np.ones(1),
        coil_to_plasma_min_distances=np.ones(1),
        coil_lengths=np.ones(1),
        coil_curvatures=curvatures[None, :],
        quadratic_flux=np.zeros((1, 1)),
        local_quadratic_flux=np.zeros((1, 1)),
        minor_radius=_MINOR_RADIUS,
    )


def test_curvature_from_metrics_counts_concave_bends(
    dented_coil_metrics: types.Metrics,
) -> None:
    requirement_metrics = data_utils.metrics_to_requirement_metrics(dented_coil_metrics)

    np.testing.assert_allclose(
        requirement_metrics.max_normalized_coil_curvature,
        _DENT_CURVATURE * _MINOR_RADIUS,
        rtol=0.0,
        atol=100 * _MACHINE_PRECISION,
    )


def test_curvature_from_stored_columns_counts_concave_bends(
    dented_coil_metrics: types.Metrics,
) -> None:
    curvatures = np.asarray(dented_coil_metrics.normalized_coil_curvatures)
    # The dataset stores only the min and max of the signed curvatures, and DESC gives
    # the dent negative curvature, so the max alone is the convex peak.
    assert curvatures.max() < _DENT_CURVATURE * _MINOR_RADIUS
    row: dict[Hashable, float] = {
        **dict.fromkeys(data_utils.REQUIREMENT_METRICS_COLUMNS, 1.0),
        "desc_metrics/normalized_coil_curvatures/min": float(curvatures.min()),
        "desc_metrics/normalized_coil_curvatures/max": float(curvatures.max()),
        data_utils.N_COILS_COLUMN: 1,
    }

    requirement_metrics = data_utils.row_to_requirement_metrics(row)

    np.testing.assert_allclose(
        requirement_metrics.max_normalized_coil_curvature,
        _DENT_CURVATURE * _MINOR_RADIUS,
        rtol=0.0,
        atol=100 * _MACHINE_PRECISION,
    )


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rows = [
        _row("c0", 3, 5, used_for_baseline=True, meets=True),
        _row("c1", 3, 5, used_for_baseline=True, meets=False),
        _row("c2", 4, 6, used_for_baseline=None, meets=True),
        _row("c3", 2, 8, used_for_baseline=False, meets=True),
        _row("c4", 3, 5, used_for_baseline=False, meets=None),
        _row("c5", 3, 5, used_for_baseline=False, meets=True, has_coilset=False),
    ]
    return pd.DataFrame(rows).astype({_FIXED_COLUMN: object, _QUALITY_COLUMN: object})


def test_select_fixed_shape_rows(synthetic_df: pd.DataFrame) -> None:
    selected = data_utils.select_ml_baseline_rows(synthetic_df, track="fixed_shape")
    assert list(selected["desc_coilset_id"]) == ["c0", "c1"]


def test_select_variable_shape_rows(synthetic_df: pd.DataFrame) -> None:
    selected = data_utils.select_ml_baseline_rows(synthetic_df, track="variable_shape")
    assert list(selected["desc_coilset_id"]) == ["c0", "c2", "c3"]


def test_validate_coil_counts_raises_on_mismatch(synthetic_df: pd.DataFrame) -> None:
    df = data_utils.select_ml_baseline_rows(synthetic_df, track="variable_shape").copy()
    df.loc[2, data_utils.N_COILS_COLUMN] = 7
    with pytest.raises(ValueError, match="c2"):
        data_utils.validate_coil_counts(df)


def test_validate_coil_counts_raises_on_nfp_mismatch(
    synthetic_df: pd.DataFrame,
) -> None:
    df = data_utils.select_ml_baseline_rows(synthetic_df, track="variable_shape").copy()
    df.loc[3, "json_constellaration_boundary"] = orjson.dumps(
        {"n_field_periods": 5}
    ).decode()
    with pytest.raises(ValueError, match="c3"):
        data_utils.validate_coil_counts(df)


def test_validate_coil_counts_accepts_consistent_rows(
    synthetic_df: pd.DataFrame,
) -> None:
    data_utils.validate_coil_counts(
        data_utils.select_ml_baseline_rows(synthetic_df, track="variable_shape")
    )


def test_sample_stratified_rows_caps_per_cell_and_is_deterministic() -> None:
    rows = [(3, 5)] * 6 + [(4, 6)] * 2 + [(2, 8)] * 1
    df = pd.DataFrame(
        {
            "boundary.n_field_periods": [nfp for nfp, _ in rows],
            data_utils.N_COILS_COLUMN: [nc for _, nc in rows],
            "id": range(len(rows)),
        }
    )
    sampled = data_utils.sample_stratified_rows(df, n_per_cell=2, seed=0)
    counts = sampled.groupby(["boundary.n_field_periods", data_utils.N_COILS_COLUMN])[
        "id"
    ].count()
    assert counts.to_dict() == {(3, 5): 2, (4, 6): 2, (2, 8): 1}
    again = data_utils.sample_stratified_rows(df, n_per_cell=2, seed=0)
    assert list(sampled["id"]) == list(again["id"])
