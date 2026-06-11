"""Generate an example, contract-conforming Coilstellaration submission model.

This builds an ONNX graph that satisfies the raw I/O contract (see the About
tab / design spec §4): it accepts the raw boundary Fourier coefficients and the
4 requirement metrics and emits a coilset (`coil_x_n`/`coil_y_n`/`coil_z_n`/
`currents`).

It is intentionally a *baseline*, not a good predictor: it ignores its inputs
and emits a fixed set of planar circular coils. Its purpose is to be a runnable
reference that (a) demonstrates the exact tensor names/shapes a submission must
expose, and (b) declares dynamic boundary axes so it runs on every benchmark
instance regardless of the boundary's spectral resolution. A real submission
should condition the coilset on the boundary and requirement metrics.

Run:  python scripts/make_example_onnx.py
Output: examples/model.onnx
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper

# --- Coilset geometry (constant) -------------------------------------------------
# Fourier order 1 -> n_modes = 3, DESC layout [sin(1), const, cos(1)].
N_COILS = 4
N_MODES = 3
MAJOR_RADIUS = 1.0          # R0
MINOR_RADIUS = 0.3          # coil loop radius a
CURRENT = 1.0e6             # Amperes per coil
# Toroidal angles of the unique (pre-symmetry) coils.
PHI = [(k + 0.5) * 0.4 for k in range(N_COILS)]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
OUT_PATH = os.path.join(OUT_DIR, "model.onnx")


def _circular_coils() -> dict[str, np.ndarray]:
    """Planar circular loops, expressed in the DESC [sin1, const, cos1] layout.

    For a loop of radius a centred at major radius R0 and toroidal angle phi:
        X(t) = R0 cos(phi) + a cos(phi) cos(t)
        Y(t) = R0 sin(phi) + a sin(phi) cos(t)
        Z(t) =                a sin(t)
    -> const = [R0 cos phi, R0 sin phi, 0], cos1 = [a cos phi, a sin phi, 0],
       sin1 = [0, 0, a].
    """
    coil_x = np.zeros((N_COILS, N_MODES), np.float32)
    coil_y = np.zeros((N_COILS, N_MODES), np.float32)
    coil_z = np.zeros((N_COILS, N_MODES), np.float32)
    for k, phi in enumerate(PHI):
        coil_x[k, 1] = MAJOR_RADIUS * np.cos(phi)   # const
        coil_x[k, 2] = MINOR_RADIUS * np.cos(phi)   # cos(1)
        coil_y[k, 1] = MAJOR_RADIUS * np.sin(phi)   # const
        coil_y[k, 2] = MINOR_RADIUS * np.sin(phi)   # cos(1)
        coil_z[k, 0] = MINOR_RADIUS                 # sin(1)
    currents = np.full((N_COILS,), CURRENT, np.float32)
    return {
        "coil_x_n": coil_x,
        "coil_y_n": coil_y,
        "coil_z_n": coil_z,
        "currents": currents,
    }


def _const_node(name: str, arr: np.ndarray) -> onnx.NodeProto:
    tensor = helper.make_tensor(
        name + "_value", TensorProto.FLOAT, list(arr.shape), arr.flatten().tolist()
    )
    return helper.make_node("Constant", inputs=[], outputs=[name], value=tensor)


def build_model() -> onnx.ModelProto:
    coils = _circular_coils()

    # Inputs: boundary arrays have dynamic [n_pol, n_tor]; requirements is [4].
    inputs = [
        helper.make_tensor_value_info("boundary_r_cos", TensorProto.FLOAT, ["n_pol", "n_tor"]),
        helper.make_tensor_value_info("boundary_z_sin", TensorProto.FLOAT, ["n_pol", "n_tor"]),
        helper.make_tensor_value_info("requirement_metrics", TensorProto.FLOAT, [4]),
    ]
    nodes = [_const_node(name, arr) for name, arr in coils.items()]
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, list(arr.shape))
        for name, arr in coils.items()
    ]

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
    onnx.checker.check_model(model)
    return model


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    model = build_model()
    onnx.save(model, OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
