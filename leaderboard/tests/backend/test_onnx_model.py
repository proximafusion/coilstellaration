import numpy as np
import pytest

from leaderboard.src.backend import config, onnx_model
from tests.backend.conftest import build_dummy_onnx


def test_validate_accepts_conforming_model(dummy_onnx_path):
    errors = onnx_model.validate_onnx_contract(dummy_onnx_path)
    assert errors == []


def test_validate_rejects_missing_required_input(tmp_path):
    path = build_dummy_onnx(
        tmp_path / "bad.onnx",
        input_names=("boundary_r_cos", "boundary_z_sin"),  # no requirement_metrics
    )
    errors = onnx_model.validate_onnx_contract(path)
    assert any("requirement_metrics" in e for e in errors)


def test_validate_rejects_missing_output(tmp_path):
    path = build_dummy_onnx(
        tmp_path / "bad.onnx", output_names=("coil_x_n", "coil_y_n", "coil_z_n")
    )
    errors = onnx_model.validate_onnx_contract(path)
    assert any("currents" in e for e in errors)


def test_run_returns_named_outputs_with_shapes(dummy_onnx_path):
    model = onnx_model.OnnxModel(dummy_onnx_path)
    out = model.run(
        {
            "boundary_r_cos": np.zeros((2, 3), np.float32),
            "boundary_z_sin": np.zeros((2, 3), np.float32),
            "requirement_metrics": np.zeros((4,), np.float32),
            "n_field_periods": np.asarray(3, np.int64),
            "n_coils_per_half_period": np.asarray(5, np.int64),
        }
    )
    assert set(out) == {"coil_x_n", "coil_y_n", "coil_z_n", "currents"}
    assert out["coil_x_n"].shape == (4, 5)
    assert out["currents"].shape == (4,)


def _session(path):
    return onnx_model.ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def test_read_supported_tracks_both(tmp_path):
    path = build_dummy_onnx(
        tmp_path / "m.onnx", supported_tracks="fixed_shape,variable_shape"
    )
    assert onnx_model.read_supported_tracks(_session(path)) == [
        "fixed_shape",
        "variable_shape",
    ]


def test_read_supported_tracks_fixed_only(tmp_path):
    path = build_dummy_onnx(tmp_path / "m.onnx", supported_tracks="fixed_shape")
    assert onnx_model.read_supported_tracks(_session(path)) == ["fixed_shape"]


def test_missing_metadata_is_contract_error(tmp_path):
    path = build_dummy_onnx(tmp_path / "m.onnx", supported_tracks=None)
    with pytest.raises(ValueError, match="supported_tracks"):
        onnx_model.read_supported_tracks(_session(path))
    errors, tracks = onnx_model.inspect_onnx(path)
    assert any("supported_tracks" in e for e in errors)
    assert tracks == []


@pytest.mark.parametrize(
    "value",
    [
        "fixed_shape, variable",
        "fixed_shape, variable_shape",
        "variable_shape",
        "bogus",
        "",
    ],
)
def test_invalid_metadata_is_contract_error(tmp_path, value):
    path = build_dummy_onnx(tmp_path / "m.onnx", supported_tracks=value)
    with pytest.raises(ValueError, match="supported_tracks"):
        onnx_model.read_supported_tracks(_session(path))
    errors, tracks = onnx_model.inspect_onnx(path)
    assert any("supported_tracks" in e for e in errors)
    assert tracks == []


@pytest.mark.parametrize("missing", ["n_field_periods", "n_coils_per_half_period"])
def test_missing_count_input_is_contract_error(tmp_path, missing):
    names = tuple(n for n in config.REQUIRED_INPUT_NAMES if n != missing)
    path = build_dummy_onnx(tmp_path / "bad.onnx", input_names=names)
    errors = onnx_model.validate_onnx_contract(path)
    assert any(missing in e for e in errors)


def test_count_parameters_sums_initializers_and_constants(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    weight = helper.make_tensor("w", TensorProto.FLOAT, [3, 4], [0.0] * 12)
    bias = helper.make_node(
        "Constant",
        [],
        ["b"],
        value=helper.make_tensor("b_v", TensorProto.FLOAT, [4], [0.0] * 4),
    )
    graph = helper.make_graph(
        [bias, helper.make_node("Add", ["w", "b"], ["y"])],
        "g",
        [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4])],
        initializer=[weight],
    )
    path = tmp_path / "params.onnx"
    onnx.save(helper.make_model(graph), path)

    assert onnx_model.count_parameters(str(path)) == 12 + 4


def test_available_cpus_is_capped_by_the_cgroup_quota(monkeypatch, tmp_path):
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("800000 100000\n")
    monkeypatch.setattr(onnx_model, "_CGROUP_V2_CPU_MAX", cpu_max)
    monkeypatch.setattr(
        onnx_model.os, "sched_getaffinity", lambda _pid: set(range(96)), raising=False
    )

    assert onnx_model.available_cpus() == 8


def test_available_cpus_without_a_quota_is_the_affinity_mask(monkeypatch, tmp_path):
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("max 100000\n")
    monkeypatch.setattr(onnx_model, "_CGROUP_V2_CPU_MAX", cpu_max)
    monkeypatch.setattr(onnx_model, "_CGROUP_V1_CPU_QUOTA", tmp_path / "missing")
    monkeypatch.setattr(
        onnx_model.os, "sched_getaffinity", lambda _pid: set(range(6)), raising=False
    )

    assert onnx_model.available_cpus() == 6


def test_sessions_size_their_thread_pool_to_the_available_cpus(
    monkeypatch, dummy_onnx_path
):
    monkeypatch.setattr(onnx_model, "available_cpus", lambda: 3)
    threads = []
    inference_session = onnx_model.ort.InferenceSession

    def spy(path, sess_options=None, **kwargs):
        assert sess_options is not None
        threads.append(sess_options.intra_op_num_threads)
        return inference_session(path, sess_options=sess_options, **kwargs)

    monkeypatch.setattr(onnx_model.ort, "InferenceSession", spy)

    onnx_model.OnnxModel(dummy_onnx_path)
    onnx_model.inspect_onnx(dummy_onnx_path)

    assert threads == [3, 3]
