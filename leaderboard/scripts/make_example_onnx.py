"""Generate an example, contract-conforming Coilstellaration submission model.

This builds an ONNX graph that satisfies the raw I/O contract (see the About
tab / design spec §4): it accepts the raw boundary Fourier coefficients and the
4 requirement metrics and emits a coilset (`coil_x_n`/`coil_y_n`/`coil_z_n`/
`currents`).

It is intentionally a *baseline*, not a good predictor: it ignores the boundary
and requirement inputs and computes `n_coils_per_half_period` planar circular
coils in-graph, evenly spaced within the first half period `[0, pi / NFP)`. Its purpose is to be a runnable
reference that (a) demonstrates the exact tensor names/shapes a submission must
expose, and (b) declares dynamic boundary axes so it runs on every benchmark
instance regardless of the boundary's spectral resolution. A real submission
should condition the coilset on the boundary and requirement metrics.

Run:  uv run python -m leaderboard.scripts.make_example_onnx
Output: examples/model.onnx
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper

# --- Coilset geometry (constant) -------------------------------------------------
# Fourier order 1 -> n_modes = 3, DESC layout [sin(1), const, cos(1)].
MAJOR_RADIUS = 1.0  # R0
MINOR_RADIUS = 0.3  # coil loop radius a
CURRENT = 1.0e6  # Amperes per coil

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
OUT_PATH = os.path.join(OUT_DIR, "dummy-constant-circle.onnx")


def _float_const(name: str, value: float) -> onnx.NodeProto:
    return helper.make_node(
        "Constant", [], [name], value=helper.make_tensor(name + "_v", TensorProto.FLOAT, [], [value])
    )


def _int64_const(name: str, values: list[int]) -> onnx.NodeProto:
    tensor = helper.make_tensor(name + "_v", TensorProto.INT64, [len(values)], values)
    return helper.make_node("Constant", [], [name], value=tensor)


def _int64_scalar(name: str, value: int) -> onnx.NodeProto:
    return helper.make_node(
        "Constant", [], [name], value=helper.make_tensor(name + "_v", TensorProto.INT64, [], [value])
    )


def _circular_coil_nodes() -> list[onnx.NodeProto]:
    """Planar circular loops computed in-graph, in the DESC [sin1, const, cos1] layout.

    For a loop of radius a centred at major radius R0 and toroidal angle phi:
        X(t) = R0 cos(phi) + a cos(phi) cos(t)
        Y(t) = R0 sin(phi) + a sin(phi) cos(t)
        Z(t) =                a sin(t)
    -> const = [R0 cos phi, R0 sin phi, 0], cos1 = [a cos phi, a sin phi, 0],
       sin1 = [0, 0, a].
    """
    mk = helper.make_node
    return [
        _int64_const("axes1", [1]),
        _float_const("half", 0.5),
        _float_const("pi", float(np.pi)),
        _float_const("r0", MAJOR_RADIUS),
        _float_const("a", MINOR_RADIUS),
        _float_const("zero_f", 0.0),
        _float_const("current", CURRENT),
        # Evenly spaced in the first half period [0, pi / NFP): coil k sits at
        # phi = (k + 0.5) * pi / (NFP * n).
        _int64_scalar("start", 0),
        _int64_scalar("step", 1),
        mk("Range", ["start", "n_coils_per_half_period", "step"], ["k_i"]),
        mk("Cast", ["k_i"], ["k"], to=TensorProto.FLOAT),
        mk("Mul", ["n_field_periods", "n_coils_per_half_period"], ["nfp_n_i"]),
        mk("Cast", ["nfp_n_i"], ["nfp_n"], to=TensorProto.FLOAT),
        mk("Div", ["pi", "nfp_n"], ["spacing"]),
        mk("Add", ["k", "half"], ["k_half"]),
        mk("Mul", ["k_half", "spacing"], ["phi"]),
        mk("Cos", ["phi"], ["cos_phi"]),
        mk("Sin", ["phi"], ["sin_phi"]),
        mk("Mul", ["phi", "zero_f"], ["zeros"]),
        mk("Mul", ["cos_phi", "r0"], ["x_const"]),
        mk("Mul", ["cos_phi", "a"], ["x_cos"]),
        mk("Mul", ["sin_phi", "r0"], ["y_const"]),
        mk("Mul", ["sin_phi", "a"], ["y_cos"]),
        mk("Add", ["zeros", "a"], ["z_sin"]),
        mk("Add", ["zeros", "current"], ["currents"]),
        *[
            mk("Unsqueeze", [c, "axes1"], [c + "_col"])
            for c in ["zeros", "x_const", "x_cos", "y_const", "y_cos", "z_sin"]
        ],
        mk("Concat", ["zeros_col", "x_const_col", "x_cos_col"], ["coil_x_n"], axis=1),
        mk("Concat", ["zeros_col", "y_const_col", "y_cos_col"], ["coil_y_n"], axis=1),
        mk("Concat", ["z_sin_col", "zeros_col", "zeros_col"], ["coil_z_n"], axis=1),
    ]


def build_model() -> onnx.ModelProto:
    # Inputs: boundary arrays have dynamic [n_pol, n_tor]; requirements is [4].
    inputs = [
        helper.make_tensor_value_info("boundary_r_cos", TensorProto.FLOAT, ["n_pol", "n_tor"]),
        helper.make_tensor_value_info("boundary_z_sin", TensorProto.FLOAT, ["n_pol", "n_tor"]),
        helper.make_tensor_value_info("requirement_metrics", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("n_field_periods", TensorProto.INT64, []),
        helper.make_tensor_value_info("n_coils_per_half_period", TensorProto.INT64, []),
    ]
    nodes = _circular_coil_nodes()
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, ["n_coils", 3])
        for name in ["coil_x_n", "coil_y_n", "coil_z_n"]
    ]
    outputs.append(helper.make_tensor_value_info("currents", TensorProto.FLOAT, ["n_coils"]))

    graph = helper.make_graph(nodes, "coilstellaration_example", inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="coilstellaration-bench-example",
    )
    model.doc_string = (
        "Baseline Coilstellaration submission: emits fixed planar circular coils. "
        "Conforms to the raw I/O contract; replace with a real conditioned model."
    )
    helper.set_metadata_props(model, {"supported_tracks": "fixed_shape,variable_shape"})
    onnx.checker.check_model(model)
    return model


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    model = build_model()
    onnx.save(model, OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
