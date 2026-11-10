import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from constellaration.geometry import surface_rz_fourier
from flax import nnx

from coilstellaration import (
    flax_nnx_checkpoint_util,
    paths,
    types,
)
from coilstellaration.machine_learning import (
    model_definition,
    utils,
)
from coilstellaration.testing_data import (
    make_fixed_shape_golden,
)

AnyVariableModel = (
    model_definition.PerCoilCoilPredictor
    | model_definition.MaskedCoilPredictor
    | model_definition.PerCoilEnsembleCoilPredictor
    | model_definition.MaskedEnsembleCoilPredictor
)

VARIABLE_TYPES = {
    "per_coil": (
        types.PerCoilCoilPredictorConfig,
        model_definition.PerCoilCoilPredictor,
        types.PerCoilCoilPredictorCheckpoint,
    ),
    "masked": (
        types.MaskedCoilPredictorConfig,
        model_definition.MaskedCoilPredictor,
        types.MaskedCoilPredictorCheckpoint,
    ),
    "per_coil_ensemble": (
        types.PerCoilEnsembleCoilPredictorConfig,
        model_definition.PerCoilEnsembleCoilPredictor,
        types.PerCoilEnsembleCoilPredictorCheckpoint,
    ),
    "masked_ensemble": (
        types.MaskedEnsembleCoilPredictorConfig,
        model_definition.MaskedEnsembleCoilPredictor,
        types.MaskedEnsembleCoilPredictorCheckpoint,
    ),
}
FIXED_TYPES = {
    "mlp": (types.CoilPredictorConfig, model_definition.CoilPredictor),
    "res_mlp": (types.ResMlpCoilPredictorConfig, model_definition.ResMlpCoilPredictor),
    "mlp_ensemble": (
        types.MlpEnsembleCoilPredictorConfig,
        model_definition.MlpEnsembleCoilPredictor,
    ),
    "res_mlp_ensemble": (
        types.ResMlpEnsembleCoilPredictorConfig,
        model_definition.ResMlpEnsembleCoilPredictor,
    ),
}


@pytest.fixture(scope="module")
def fixed_shape_golden() -> dict[str, np.ndarray]:
    path = (
        paths.data_dir().parent
        / "testing_data"
        / make_fixed_shape_golden.GOLDEN_FILENAME
    )
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


@pytest.fixture(scope="module")
def fixed_shape_actual() -> dict[str, np.ndarray]:
    return make_fixed_shape_golden.compute_golden()


@pytest.fixture
def boundary() -> surface_rz_fourier.SurfaceRZFourier:
    r_cos = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (5, 9))
    z_sin = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (5, 9))
    r_cos = r_cos.at[0, :4].set(0.0).at[0, 4].set(1.0)
    z_sin = z_sin.at[0, :5].set(0.0)
    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.asarray(r_cos),
        z_sin=np.asarray(z_sin),
        n_field_periods=3,
        is_stellarator_symmetric=True,
    )


@pytest.fixture
def metrics() -> types.RequirementMetrics:
    return types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=0.4,
        min_normalized_coil_to_plasma_distance=1.2,
        max_normalized_coil_curvature=3.8,
        max_normalized_field_error=0.01,
        n_coils_per_half_period=5,
    )


@pytest.fixture
def eval_data(
    boundary: surface_rz_fourier.SurfaceRZFourier, metrics: types.RequirementMetrics
) -> types.EvalData:
    n_coils = int(metrics.n_coils_per_half_period)
    zeros = np.zeros((n_coils, 7))
    return types.EvalData(
        boundary=boundary,
        boundary_id="synthetic",
        true_coilset=types.Coilset(
            coil_x_n=zeros,
            coil_y_n=zeros,
            coil_z_n=zeros,
            currents=np.zeros(n_coils),
            n_field_periods=boundary.n_field_periods,
            is_stellarator_symmetric=True,
        ),
        true_metrics="",
        requirement_metrics=metrics,
    )


@pytest.fixture
def variable_config() -> types.PerCoilCoilPredictorConfig:
    return types.PerCoilCoilPredictorConfig(
        is_stellarator_symmetric=True,
        n_poloidal_modes=5,
        n_toroidal_modes=9,
        n_modes_coils_max=15,
        coil_currents_mean=0.0,
        coil_currents_std=1e6,
        n_field_periods_values=[2, 3, 4, 5],
        n_coils_values=[4, 5, 6, 7, 8],
        max_coils=8,
        hidden_dim=16,
        num_layers=2,
        requirement_metrics_means=[0.0] * 4,
        requirement_metrics_stds=[1.0] * 4,
    )


@pytest.fixture(params=list(VARIABLE_TYPES))
def variable_model(
    request: pytest.FixtureRequest, variable_config: types.PerCoilCoilPredictorConfig
) -> AnyVariableModel:
    config_type, model_cls, _ = VARIABLE_TYPES[request.param]
    fields = variable_config.model_dump(exclude={"model_kind"})
    if "n_ensemble_members" in config_type.model_fields:
        fields["n_ensemble_members"] = 3
    return model_cls(config_type(**fields), rngs=nnx.Rngs(0))  # type: ignore[arg-type]


@pytest.fixture
def variable_checkpoint(variable_model: AnyVariableModel) -> types.AnyModelCheckpoint:
    _, _, checkpoint_type = VARIABLE_TYPES[variable_model.config.model_kind]
    return flax_nnx_checkpoint_util.to_checkpoint(
        variable_model,
        variable_model.config,
        checkpoint_type,  # type: ignore[arg-type]
    )


@pytest.fixture(params=[*FIXED_TYPES, *VARIABLE_TYPES])
def any_model(
    request: pytest.FixtureRequest, variable_config: types.PerCoilCoilPredictorConfig
) -> model_definition.AnyModel:
    model: model_definition.AnyModel
    if request.param in VARIABLE_TYPES:
        config_type, model_cls, _ = VARIABLE_TYPES[request.param]
        config = config_type(**variable_config.model_dump(exclude={"model_kind"}))
    else:
        config_type, model_cls = FIXED_TYPES[request.param]
        config = config_type(
            is_stellarator_symmetric=True,
            n_field_periods=3,
            n_poloidal_modes=5,
            n_toroidal_modes=9,
            n_unique_coils=5,
            n_modes_coils_max=15,
            coil_currents_mean=3e5,
            coil_currents_std=1e5,
            hidden_dim=16,
            num_layers=2,
            requirement_metrics_means=[0.0] * 4,
            requirement_metrics_stds=[1.0] * 4,
        )
    model = model_cls(config, rngs=nnx.Rngs(0))  # type: ignore[arg-type]
    model.eval()
    return model


@pytest.mark.parametrize(
    "key", ["coil_x_n", "coil_y_n", "coil_z_n", "currents", "loss"]
)
def test_fixed_shape_matches_golden(
    key: str,
    fixed_shape_actual: dict[str, np.ndarray],
    fixed_shape_golden: dict[str, np.ndarray],
) -> None:
    # XLA compiles for the host CPU (vector width, FMA contraction), so the last bits
    # differ between the machine that wrote the golden and the one running the test.
    np.testing.assert_allclose(
        fixed_shape_actual[key], fixed_shape_golden[key], rtol=1e-12, atol=1e-13
    )


def test_predict_trims_each_instance(
    variable_model: AnyVariableModel, eval_data: types.EvalData
) -> None:
    dataset = [
        _with_shape(eval_data, n_field_periods=3, n_coils=4),
        _with_shape(eval_data, n_field_periods=5, n_coils=8),
        _with_shape(eval_data, n_field_periods=2, n_coils=6),
    ]

    result = model_definition.predict_coilsets(variable_model, dataset)

    row_counts = [
        int(np.asarray(item.predicted_coilset.currents).shape[0])  # type: ignore[union-attr]
        for item in result
    ]
    assert row_counts == [4, 8, 6]
    assert [item.boundary.n_field_periods for item in result] == [3, 5, 2]


@pytest.mark.parametrize(
    ("n_field_periods", "n_coils", "unsupported"), [(3, 9, "9"), (6, 5, "6")]
)
def test_predict_rejects_unsupported_shape(
    variable_model: AnyVariableModel,
    eval_data: types.EvalData,
    n_field_periods: int,
    n_coils: int,
    unsupported: str,
) -> None:
    dataset = [_with_shape(eval_data, n_field_periods, n_coils)]

    with pytest.raises(ValueError, match=unsupported):
        model_definition.predict_coilsets(variable_model, dataset)


def test_fixed_model_rejects_wrong_coil_count(eval_data: types.EvalData) -> None:
    checkpoint = utils.read_checkpoint_from_id("mlp", "DHmCARHZ4ay9rsMDPdjVCu3")
    model = model_definition.read_model_from_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="6"):
        model_definition.predict_coilsets(
            model, [_with_shape(eval_data, n_field_periods=3, n_coils=6)]
        )


@pytest.mark.parametrize("variable_model", ["per_coil", "masked"], indirect=True)
def test_output_padded_to_max_coils(
    variable_model: AnyVariableModel,
    boundary: surface_rz_fourier.SurfaceRZFourier,
    metrics: types.RequirementMetrics,
) -> None:
    coilset = variable_model(boundary, metrics)

    assert coilset.coil_x_n.shape == (8, 15)
    assert coilset.coil_y_n.shape == (8, 15)
    assert coilset.coil_z_n.shape == (8, 15)
    assert coilset.currents.shape == (8,)
    assert coilset.n_field_periods == boundary.n_field_periods


@pytest.mark.parametrize("variable_model", ["per_coil", "masked"], indirect=True)
def test_jit_vmap_over_mixed_nfp_and_nc(
    variable_model: AnyVariableModel,
    boundary: surface_rz_fourier.SurfaceRZFourier,
    metrics: types.RequirementMetrics,
) -> None:
    boundaries = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        boundary.model_copy(update={"n_field_periods": 2}),
        boundary.model_copy(update={"n_field_periods": 5}),
    )
    batched_metrics = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        metrics.model_copy(update={"n_coils_per_half_period": 4}),
        metrics.model_copy(update={"n_coils_per_half_period": 8}),
    )

    coilset = jax.jit(jax.vmap(variable_model))(boundaries, batched_metrics)

    assert coilset.coil_x_n.shape == (2, 8, 15)
    assert coilset.currents.shape == (2, 8)


@pytest.mark.parametrize("variable_model", ["per_coil", "masked"], indirect=True)
def test_onehot_changes_output(
    variable_model: AnyVariableModel,
    boundary: surface_rz_fourier.SurfaceRZFourier,
    metrics: types.RequirementMetrics,
) -> None:
    few = variable_model(
        boundary, metrics.model_copy(update={"n_coils_per_half_period": 4})
    )
    many = variable_model(
        boundary, metrics.model_copy(update={"n_coils_per_half_period": 8})
    )

    assert not np.allclose(few.coil_x_n, many.coil_x_n)


@pytest.mark.parametrize("variable_model", ["per_coil", "masked"], indirect=True)
def test_fourier_order_truncates(
    variable_model: AnyVariableModel,
    boundary: surface_rz_fourier.SurfaceRZFourier,
    metrics: types.RequirementMetrics,
) -> None:
    coilset = variable_model(boundary, metrics, fourier_order=3)

    assert coilset.coil_x_n.shape == (8, 7)


def test_prediction_scales_with_the_boundary(
    any_model: model_definition.AnyModel,
    boundary: surface_rz_fourier.SurfaceRZFourier,
    metrics: types.RequirementMetrics,
) -> None:
    # Scaling a device scales its coils, and the currents that produce the same field,
    # by the same factor.
    scale = 1.7
    scaled_boundary = boundary.model_copy(
        update={"r_cos": scale * boundary.r_cos, "z_sin": scale * boundary.z_sin}
    )

    coilset = any_model(boundary, metrics)
    scaled = any_model(scaled_boundary, metrics)

    for field in ("coil_x_n", "coil_y_n", "coil_z_n", "currents"):
        expected = scale * np.asarray(getattr(coilset, field))
        np.testing.assert_allclose(
            getattr(scaled, field),
            expected,
            rtol=1e-5,
            atol=1e-5 * np.abs(expected).max(),
            err_msg=field,
        )


def test_per_coil_param_count_independent_of_max_coils(
    variable_config: types.PerCoilCoilPredictorConfig,
) -> None:
    big = model_definition.PerCoilCoilPredictor(variable_config, rngs=nnx.Rngs(0))
    small = model_definition.PerCoilCoilPredictor(
        variable_config.model_copy(update={"max_coils": 6}),
        rngs=nnx.Rngs(0),
    )

    assert _n_params(big) - _n_params(small) == 2 * variable_config.hidden_dim


def test_train_config_json_roundtrip_keeps_variable_config_type(
    variable_model: AnyVariableModel,
) -> None:
    cfg = types.TrainConfig(
        model_kind=variable_model.config.model_kind,
        model_config_overrides=variable_model.config,  # type: ignore[arg-type]
    )

    restored = types.TrainConfig.model_validate_json(cfg.model_dump_json())

    assert type(restored.model_config_overrides) is type(cfg.model_config_overrides)


@pytest.mark.parametrize(
    "variable_model", ["per_coil_ensemble", "masked_ensemble"], indirect=True
)
def test_ensemble_mean_equals_mean_of_members(
    variable_model: AnyVariableModel,
    boundary: surface_rz_fourier.SurfaceRZFourier,
    metrics: types.RequirementMetrics,
) -> None:
    assert isinstance(
        variable_model,
        model_definition.PerCoilEnsembleCoilPredictor
        | model_definition.MaskedEnsembleCoilPredictor,
    )

    per_member = variable_model.predict_per_member(boundary, metrics)
    mean = variable_model(boundary, metrics)

    assert per_member.coil_x_n.shape == (3, 8, 15)
    assert per_member.currents.shape == (3, 8)
    assert not np.allclose(per_member.coil_x_n[0], per_member.coil_x_n[1])
    np.testing.assert_allclose(
        mean.coil_x_n, jnp.mean(per_member.coil_x_n, axis=0), rtol=1e-6
    )
    np.testing.assert_allclose(
        mean.currents, jnp.mean(per_member.currents, axis=0), rtol=1e-6
    )


class _PerCoilEnsembleConfigWithExtra(types.PerCoilEnsembleCoilPredictorConfig):
    stray: str = "not a member field"


class _MaskedEnsembleConfigWithExtra(types.MaskedEnsembleCoilPredictorConfig):
    stray: str = "not a member field"


@pytest.mark.parametrize(
    ("config_type", "model_cls", "member_config_type"),
    [
        (
            _PerCoilEnsembleConfigWithExtra,
            model_definition.PerCoilEnsembleCoilPredictor,
            types.PerCoilCoilPredictorConfig,
        ),
        (
            _MaskedEnsembleConfigWithExtra,
            model_definition.MaskedEnsembleCoilPredictor,
            types.MaskedCoilPredictorConfig,
        ),
    ],
)
def test_ensemble_members_get_only_member_fields(
    variable_config: types.PerCoilCoilPredictorConfig,
    config_type: type[_PerCoilEnsembleConfigWithExtra]
    | type[_MaskedEnsembleConfigWithExtra],
    model_cls: type[model_definition.PerCoilEnsembleCoilPredictor]
    | type[model_definition.MaskedEnsembleCoilPredictor],
    member_config_type: type[types.PerCoilCoilPredictorConfig]
    | type[types.MaskedCoilPredictorConfig],
) -> None:
    fields = variable_config.model_dump(exclude={"model_kind"})

    model = model_cls(config_type(**fields), rngs=nnx.Rngs(0))  # type: ignore[arg-type]

    expected = member_config_type(**fields)
    for member in model.members:
        assert type(member.config) is member_config_type
        assert member.config == expected


def test_checkpoint_roundtrip_identical_outputs(
    variable_model: AnyVariableModel,
    variable_checkpoint: types.AnyModelCheckpoint,
    boundary: surface_rz_fourier.SurfaceRZFourier,
    metrics: types.RequirementMetrics,
) -> None:
    variable_model.eval()
    restored_checkpoint = type(variable_checkpoint).model_validate_json(
        variable_checkpoint.model_dump_json()
    )

    restored = model_definition.read_model_from_checkpoint(restored_checkpoint)

    assert type(restored) is type(variable_model)
    expected = variable_model(boundary, metrics)
    actual = restored(boundary, metrics)
    np.testing.assert_array_equal(actual.coil_x_n, expected.coil_x_n)
    np.testing.assert_array_equal(actual.currents, expected.currents)


def test_read_checkpoint_from_id_dispatches_new_types(
    variable_model: AnyVariableModel,
    variable_checkpoint: types.AnyModelCheckpoint,
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text(variable_checkpoint.model_dump_json())

    restored = utils.read_checkpoint_from_id(
        variable_model.config.model_kind,
        "unused-id",
        model_data_path=path,
    )

    assert type(restored) is type(variable_checkpoint)
    assert type(restored.config) is type(variable_model.config)


def _with_shape(
    eval_data: types.EvalData, n_field_periods: int, n_coils: int
) -> types.EvalData:
    # `predict_coilsets` reads only the boundary and requirement metrics, so the
    # placeholder `true_coilset` can keep its original shape.
    return eval_data.model_copy(
        update={
            "boundary": eval_data.boundary.model_copy(
                update={"n_field_periods": n_field_periods}
            ),
            "requirement_metrics": eval_data.requirement_metrics.model_copy(
                update={"n_coils_per_half_period": n_coils}
            ),
        }
    )


def _n_params(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
