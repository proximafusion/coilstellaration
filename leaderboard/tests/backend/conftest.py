import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


def _const(name, shape):
    flat = np.zeros(int(np.prod(shape)), dtype=np.float32)
    val = helper.make_tensor(name + "_v", TensorProto.FLOAT, list(shape), flat)
    return helper.make_node("Constant", [], [name], value=val)


def build_dummy_onnx(
    path,
    n_pol=2,
    n_tor=3,
    n_coils=4,
    n_modes=5,
    input_names=(
        "boundary_r_cos",
        "boundary_z_sin",
        "requirement_metrics",
        "n_field_periods",
        "n_coils_per_half_period",
    ),
    output_names=("coil_x_n", "coil_y_n", "coil_z_n", "currents"),
    supported_tracks: str | None = "fixed_shape",
):
    """A contract-conforming ONNX graph whose outputs are zeros of the right shape.

    Inputs are declared (so onnxruntime accepts the feeds) but unused. The
    `supported_tracks` metadata is written verbatim; `None` omits it.
    """
    in_shapes = {
        "boundary_r_cos": [n_pol, n_tor],
        "boundary_z_sin": [n_pol, n_tor],
        "boundary_r_sin": [n_pol, n_tor],
        "boundary_z_cos": [n_pol, n_tor],
        "requirement_metrics": [4],
        "n_field_periods": [],
        "n_coils_per_half_period": [],
    }
    out_shapes = {
        "coil_x_n": [n_coils, n_modes],
        "coil_y_n": [n_coils, n_modes],
        "coil_z_n": [n_coils, n_modes],
        "currents": [n_coils],
    }
    int_inputs = {"n_field_periods", "n_coils_per_half_period"}
    inputs = [
        helper.make_tensor_value_info(
            n, TensorProto.INT64 if n in int_inputs else TensorProto.FLOAT, in_shapes[n]
        )
        for n in input_names
    ]
    nodes = [_const(n, out_shapes[n]) for n in output_names]
    outputs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, out_shapes[n])
        for n in output_names
    ]
    graph = helper.make_graph(nodes, "dummy", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    if supported_tracks is not None:
        helper.set_model_props(model, {"supported_tracks": supported_tracks})
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def dummy_onnx_path(tmp_path):
    return build_dummy_onnx(tmp_path / "model.onnx")
