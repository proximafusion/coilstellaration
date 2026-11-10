import json

import numpy as np
import onnxruntime as ort
import pytest
from flax import nnx
from scripts import export_coilstellaration_onnx as export

from coilstellaration import flax_nnx_checkpoint_util, types
from coilstellaration.machine_learning import model_definition
from leaderboard.src.backend import onnx_model

_COMMON = dict(
    is_stellarator_symmetric=True,
    n_poloidal_modes=5,
    n_toroidal_modes=9,
    n_modes_coils_max=5,
    coil_currents_mean=1e6,
    coil_currents_std=1e5,
    hidden_dim=16,
    num_layers=2,
    requirement_metrics_means=[0.1, 1.0, 3.0, 0.01],
    requirement_metrics_stds=[0.1, 0.5, 1.0, 0.01],
)
_FIXED = dict(n_field_periods=3, n_unique_coils=5)
_VARIABLE = dict(n_field_periods_values=[3, 4], n_coils_values=[4, 5, 6], max_coils=6)

_MODELS = {
    "mlp": (model_definition.CoilPredictor, types.CoilPredictorConfig, _FIXED),
    "res_mlp_ensemble": (
        model_definition.ResMlpEnsembleCoilPredictor,
        types.ResMlpEnsembleCoilPredictorConfig,
        {**_FIXED, "n_ensemble_members": 2},
    ),
    "per_coil": (
        model_definition.PerCoilCoilPredictor,
        types.PerCoilCoilPredictorConfig,
        _VARIABLE,
    ),
    "masked": (
        model_definition.MaskedCoilPredictor,
        types.MaskedCoilPredictorConfig,
        _VARIABLE,
    ),
    "per_coil_ensemble": (
        model_definition.PerCoilEnsembleCoilPredictor,
        types.PerCoilEnsembleCoilPredictorConfig,
        {**_VARIABLE, "n_ensemble_members": 2},
    ),
    "masked_ensemble": (
        model_definition.MaskedEnsembleCoilPredictor,
        types.MaskedEnsembleCoilPredictorConfig,
        {**_VARIABLE, "n_ensemble_members": 2},
    ),
}


def _tiny_model(model_kind: str):
    module_cls, config_cls, extra = _MODELS[model_kind]
    model = module_cls(config_cls(**_COMMON, **extra), rngs=nnx.Rngs(0))
    model.eval()
    return model


@pytest.fixture(scope="module")
def sessions() -> dict[str, ort.InferenceSession]:
    return {
        kind: ort.InferenceSession(
            export.export_model(_tiny_model(kind)).SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        for kind in ("mlp", "per_coil")
    }


def _run(
    session: ort.InferenceSession, n_field_periods: int, n_coils: int
) -> dict[str, np.ndarray]:
    outputs = session.run(
        list(export.OUTPUT_NAMES),
        {
            "boundary_r_cos": np.full((5, 9), 0.1, np.float32),
            "boundary_z_sin": np.full((5, 9), 0.1, np.float32),
            "requirement_metrics": np.array([0.4, 1.2, 3.8, 0.01], np.float32),
            "n_field_periods": np.asarray(n_field_periods, np.int64),
            "n_coils_per_half_period": np.asarray(n_coils, np.int64),
        },
    )
    return {
        name: np.asarray(value)
        for name, value in zip(export.OUTPUT_NAMES, outputs, strict=True)
    }


@pytest.mark.parametrize("model_kind", sorted(_MODELS))
def test_export_matches_eager_for_all_pairs(model_kind):
    model = _tiny_model(model_kind)
    export.verify_export(model, export.export_model(model))


@pytest.mark.parametrize("n_coils", [4, 6])
def test_slice_returns_requested_rows(sessions, n_coils):
    outputs = _run(sessions["per_coil"], n_field_periods=4, n_coils=n_coils)
    assert outputs["coil_x_n"].shape == (n_coils, 5)
    assert outputs["currents"].shape == (n_coils,)


@pytest.mark.parametrize(
    ("model_kind", "n_field_periods", "n_coils"),
    [("mlp", 4, 6), ("mlp", 3, 6), ("per_coil", 5, 4), ("per_coil", 3, 8)],
)
def test_unsupported_counts_give_nan(sessions, model_kind, n_field_periods, n_coils):
    outputs = _run(sessions[model_kind], n_field_periods, n_coils)
    assert all(np.isnan(value).all() for value in outputs.values())


@pytest.mark.parametrize(
    ("model_kind", "tracks"),
    [("mlp", "fixed_shape"), ("per_coil", "fixed_shape,variable_shape")],
)
def test_metadata_written(sessions, model_kind, tracks):
    metadata = sessions[model_kind].get_modelmeta().custom_metadata_map
    assert metadata[export.SUPPORTED_TRACKS_METADATA_KEY] == tracks


@pytest.mark.parametrize("model_kind", ["mlp", "per_coil"])
def test_exported_metadata_accepted_by_space(sessions, model_kind):
    onnx_model.read_supported_tracks(sessions[model_kind])


def test_inputs_follow_contract(sessions):
    inputs = {i.name: i.type for i in sessions["per_coil"].get_inputs()}
    assert inputs == {
        "boundary_r_cos": "tensor(float)",
        "boundary_z_sin": "tensor(float)",
        "requirement_metrics": "tensor(float)",
        "n_field_periods": "tensor(int64)",
        "n_coils_per_half_period": "tensor(int64)",
    }


def _write_mlp_checkpoint_without_model_kind(json_path, dapper_type=None):
    model = _tiny_model("mlp")
    checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
        model, model.config, types.CoilPredictorCheckpoint
    )
    raw = json.loads(checkpoint.model_dump_json())
    del raw["config"]["model_kind"]
    if dapper_type is not None:
        raw["config"]["dapper_type"] = dapper_type
    json_path.write_text(json.dumps(raw))


def test_export_checkpoint_rejects_untagged_config(tmp_path):
    json_path = tmp_path / "Dold.json"
    _write_mlp_checkpoint_without_model_kind(json_path)

    with pytest.raises(ValueError, match="model_kind"):
        export.export_checkpoint(json_path, tmp_path)
