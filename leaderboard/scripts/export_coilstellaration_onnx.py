"""Export trained `coilstellaration` Flax NNX checkpoints to ONNX.

Checkpoints are the JSON files written by the `coilstellaration` trainer (bundled
under its `data/models/` directory, or the `checkpoint_json` of a training job).
Each file's `config.model_kind` (or, for checkpoints exported from dapper before it
existed, `config.dapper_type`) selects the architecture; all eight are supported:
the fixed-shape `mlp`, `res_mlp` and their ensembles, and the variable-shape
`per_coil`, `masked` and their ensembles.

Every graph follows the Space contract (see the About tab):

- inputs: `boundary_r_cos`, `boundary_z_sin` (float32, the model's boundary
  resolution), `requirement_metrics` (float32, shape (4,)), `n_field_periods` and
  `n_coils_per_half_period` (int64 scalars);
- outputs: `coil_x_n`, `coil_y_n`, `coil_z_n` (`n_coils_per_half_period` rows) and
  `currents`;
- metadata `supported_tracks`: `"fixed_shape"` or `"fixed_shape,variable_shape"`.

A field-period or coil count outside the model's training vocabulary makes every
output NaN, so the Space fails that instance instead of scoring a silently wrong
coilset. Every export is verified against the model's eager `__call__` for each
supported `(n_field_periods, n_coils_per_half_period)` pair before being written.

Run:  uv run python -m leaderboard.scripts.export_coilstellaration_onnx [ckpt.json ...]
      (pass checkpoint JSON paths as arguments to export only those)
Output: examples/<model_kind>_<id6>.onnx  (e.g. examples/per_coil_DXyz12.onnx)
"""

from __future__ import annotations

import argparse
import json
import pathlib

import jax
import jax.numpy as jnp
import jax2onnx
import numpy as np
import onnx
import onnxruntime as ort
from constellaration.geometry import surface_rz_fourier
from onnx import helper

from coilstellaration import coilset_utils, paths, types
from coilstellaration.machine_learning import model_definition
from leaderboard.src.backend import config as bench_config

# Gelu is a first-class ONNX op only from opset 20; onnxruntime >=1.17 supports it.
OPSET = 20
# float32 round-trip tolerance for the JAX-vs-ONNX verification gate, applied to
# each output's error normalized by that output's own magnitude scale:
#     max|expected - actual| <= REL_TOL * max|expected|
# Per-output scaling is essential for `currents`: they are reconstructed as
# `flat * std + mean` with std/mean ~1e6, so the float32 absolute-error floor is
# ~1 A regardless of a coil's final (possibly small) current. Grading on the
# output's overall scale reflects the physically meaningful (relative) error.
REL_TOL = 1e-4

INPUT_NAMES = (
    "boundary_r_cos",
    "boundary_z_sin",
    "requirement_metrics",
    "n_field_periods",
    "n_coils_per_half_period",
)
OUTPUT_NAMES = ("coil_x_n", "coil_y_n", "coil_z_n", "currents")
SUPPORTED_TRACKS_METADATA_KEY = bench_config.SUPPORTED_TRACKS_METADATA_KEY
# The fixed-shape track holds only NFP-3 boundaries with 5 coils per half period.
FIXED_SHAPE_TRACK_COUNTS = (3, 5)

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"

_CHECKPOINT_TYPES: dict[str, type[types.AnyModelCheckpoint]] = {
    "mlp": types.CoilPredictorCheckpoint,
    "res_mlp": types.ResMlpCoilPredictorCheckpoint,
    "mlp_ensemble": types.MlpEnsembleCoilPredictorCheckpoint,
    "res_mlp_ensemble": types.ResMlpEnsembleCoilPredictorCheckpoint,
    "per_coil": types.PerCoilCoilPredictorCheckpoint,
    "masked": types.MaskedCoilPredictorCheckpoint,
    "per_coil_ensemble": types.PerCoilEnsembleCoilPredictorCheckpoint,
    "masked_ensemble": types.MaskedEnsembleCoilPredictorCheckpoint,
}


def export_checkpoint(
    json_path: pathlib.Path, out_dir: pathlib.Path = OUT_DIR
) -> pathlib.Path:
    """Convert one checkpoint JSON to a verified ONNX file; return the output path."""
    config = json.loads(json_path.read_text())["config"]
    model_kind = config.get("model_kind")
    if model_kind not in _CHECKPOINT_TYPES:
        raise ValueError(
            f"{json_path.name}: cannot tell the architecture from config.model_kind "
            f"{config.get('model_kind')!r} or config.dapper_type "
            f"{config.get('dapper_type')!r}."
        )
    checkpoint = _CHECKPOINT_TYPES[model_kind].model_validate_json(
        json_path.read_text()
    )
    model = model_definition.read_model_from_checkpoint(checkpoint)

    proto = export_model(model)
    verify_export(model, proto)

    out_path = out_dir / f"{json_path.stem}.onnx"
    onnx.save(proto, out_path)
    return out_path


def export_model(model: model_definition.AnyModel) -> onnx.ModelProto:
    """Convert a loaded model to a contract-conforming ONNX graph."""
    config = model.config
    if not config.is_stellarator_symmetric:
        raise NotImplementedError("Only stellarator-symmetric models can be exported.")
    shape = (config.n_poloidal_modes, config.n_toroidal_modes)
    # Traced with JAX's default int32; `_widen_int_inputs` turns the graph inputs into
    # the contract's int64 afterwards, so the float32 model needs no x64 mode.
    proto = jax2onnx.to_onnx(
        _make_predict_fn(model),
        [
            jax.ShapeDtypeStruct(shape, jnp.float32),
            jax.ShapeDtypeStruct(shape, jnp.float32),
            jax.ShapeDtypeStruct((4,), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
        ],
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        model_name=f"coilstellaration_{config.model_kind}",
        opset=OPSET,
    )
    _batch_rank1_gemms(proto)
    _widen_int_inputs(proto, ("n_field_periods", "n_coils_per_half_period"))
    if _is_variable_shape(config):
        append_coil_slice(proto)
    set_supported_tracks(proto, supported_tracks(config))
    onnx.checker.check_model(proto)
    return proto


def verify_export(model: model_definition.AnyModel, proto: onnx.ModelProto) -> None:
    """Compare the ONNX graph with the eager model for every supported count pair.

    Raises:
        RuntimeError: If an output's shape differs or its error exceeds `REL_TOL`.
    """
    config = model.config
    shape = (config.n_poloidal_modes, config.n_toroidal_modes)
    r_cos = 0.1 * np.asarray(
        jax.random.normal(jax.random.PRNGKey(0), shape), np.float32
    )
    z_sin = 0.1 * np.asarray(
        jax.random.normal(jax.random.PRNGKey(1), shape), np.float32
    )
    # Stellarator symmetry requires zero `m = 0, n <= 0` coefficients (except the
    # `r_cos` major radius).
    n_toroidal = (config.n_toroidal_modes - 1) // 2
    r_cos[0, :n_toroidal] = 0.0
    r_cos[0, n_toroidal] = 1.0
    z_sin[0, : n_toroidal + 1] = 0.0
    metrics = np.array([0.4, 1.2, 3.8, 0.01], dtype=np.float32)
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    for n_field_periods, n_coils in supported_count_pairs(config):
        expected = _eager_prediction(
            model, r_cos, z_sin, metrics, n_field_periods, n_coils
        )
        actual = session.run(
            list(OUTPUT_NAMES),
            {
                "boundary_r_cos": r_cos,
                "boundary_z_sin": z_sin,
                "requirement_metrics": metrics,
                "n_field_periods": np.asarray(n_field_periods, np.int64),
                "n_coils_per_half_period": np.asarray(n_coils, np.int64),
            },
        )
        for name, exp, act in zip(
            OUTPUT_NAMES, expected, map(np.asarray, actual), strict=True
        ):
            where = (
                f"[{config.model_kind}, NFP {n_field_periods}, {n_coils} coils] "
                f"{name!r}"
            )
            if exp.shape != act.shape:
                raise RuntimeError(
                    f"{where}: ONNX shape {act.shape} != JAX shape {exp.shape}"
                )
            scale = float(np.max(np.abs(exp))) or 1.0
            max_abs = float(np.max(np.abs(exp - act)))
            if not max_abs <= REL_TOL * scale:
                raise RuntimeError(
                    f"{where}: ONNX disagrees with the JAX model "
                    f"(max|diff|={max_abs:g}, scale={scale:g}, "
                    f"rel={max_abs / scale:g} > rel_tol={REL_TOL})"
                )


def supported_count_pairs(config: types.AnyModelConfig) -> list[tuple[int, int]]:
    """All `(n_field_periods, n_coils_per_half_period)` pairs the model accepts."""
    nfp_values, n_coils_values = _supported_counts(config)
    return [(nfp, n_coils) for nfp in nfp_values for n_coils in n_coils_values]


def supported_tracks(config: types.AnyModelConfig) -> list[str]:
    tracks = []
    if FIXED_SHAPE_TRACK_COUNTS in supported_count_pairs(config):
        tracks.append("fixed_shape")
    if _is_variable_shape(config):
        tracks.append("variable_shape")
    if not tracks:
        raise ValueError(
            f"{config.model_kind} model supports no benchmark track: it does not "
            f"accept NFP {FIXED_SHAPE_TRACK_COUNTS[0]} with "
            f"{FIXED_SHAPE_TRACK_COUNTS[1]} coils."
        )
    if ",".join(tracks) not in bench_config.SUPPORTED_TRACKS_VALUES:
        raise ValueError(
            f"{config.model_kind} model would declare "
            f"supported_tracks={','.join(tracks)!r}, which the Space rejects; "
            f"accepted values: {list(bench_config.SUPPORTED_TRACKS_VALUES)}."
        )
    return tracks


def append_coil_slice(proto: onnx.ModelProto) -> None:
    """Keep only the first `n_coils_per_half_period` rows of every output, in place.

    Variable-shape models emit `max_coils` padded rows; JAX cannot slice to a
    traced length, so the slice is added to the exported graph instead.
    """
    graph = proto.graph
    ends = "coil_slice_ends"
    graph.initializer.extend(
        [
            helper.make_tensor("coil_slice_zero", onnx.TensorProto.INT64, [1], [0]),
        ]
    )
    graph.node.extend(
        [
            helper.make_node(
                "Unsqueeze",
                ["n_coils_per_half_period", "coil_slice_zero"],
                [ends],
                name="coil_slice_ends",
            )
        ]
    )
    for output in graph.output:
        padded = f"{output.name}_padded"
        _rename_value(graph, output.name, padded)
        graph.node.extend(
            [
                helper.make_node(
                    "Slice",
                    [padded, "coil_slice_zero", ends, "coil_slice_zero"],
                    [output.name],
                    name=f"{output.name}_coil_slice",
                )
            ]
        )
        output.type.tensor_type.shape.dim[0].ClearField("dim_value")
        output.type.tensor_type.shape.dim[0].dim_param = "n_coils_per_half_period"


def set_supported_tracks(proto: onnx.ModelProto, tracks: list[str]) -> None:
    helper.set_model_props(proto, {SUPPORTED_TRACKS_METADATA_KEY: ",".join(tracks)})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export coilstellaration checkpoints to ONNX."
    )
    parser.add_argument(
        "checkpoints",
        nargs="*",
        type=pathlib.Path,
        help="Checkpoint JSON files (default: every bundled checkpoint).",
    )
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    args = parser.parse_args()

    json_paths = args.checkpoints or sorted(paths.models_dir().glob("*.json"))
    if not json_paths:
        raise SystemExit(f"No checkpoints found in {paths.models_dir()}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {len(json_paths)} checkpoint(s) -> {args.out_dir}")
    for json_path in json_paths:
        out_path = export_checkpoint(json_path, args.out_dir)
        print(f"  {json_path.name}  ->  {out_path.name}  (verified)")
    print("Done.")


def _make_predict_fn(model: model_definition.AnyModel):
    """Build the pure-JAX contract-inputs -> padded-coilset-arrays function."""
    config = model.config
    nfp_values, n_coils_values = (jnp.asarray(v) for v in _supported_counts(config))
    shape = (config.n_poloidal_modes, config.n_toroidal_modes)
    # Pydantic validation rejects tracers, so the model inputs are built from
    # validated templates with `model_copy`, which skips validation.
    boundary_template = surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.zeros(shape),
        z_sin=np.zeros(shape),
        n_field_periods=FIXED_SHAPE_TRACK_COUNTS[0],
        is_stellarator_symmetric=True,
    )
    metrics_template = _requirement_metrics(np.zeros(4), FIXED_SHAPE_TRACK_COUNTS[1])

    def predict(
        boundary_r_cos, boundary_z_sin, requirement_metrics, n_field_periods, n_coils
    ):
        boundary = boundary_template.model_copy(
            update={
                "r_cos": boundary_r_cos,
                "z_sin": boundary_z_sin,
                "n_field_periods": n_field_periods,
            }
        )
        metrics = metrics_template.model_copy(
            update={
                **_metric_fields(requirement_metrics),
                "n_coils_per_half_period": n_coils,
            }
        )
        coilset = model(boundary, metrics)
        # Fixed-shape models ignore the counts and variable-shape models see an
        # all-zero one-hot for an unknown count; either would predict a plausible but
        # wrong coilset.
        supported = jnp.any(nfp_values == n_field_periods) & jnp.any(
            n_coils_values == n_coils
        )
        poison = jnp.where(supported, 1.0, jnp.nan).astype(jnp.float32)
        return tuple(getattr(coilset, name) * poison for name in OUTPUT_NAMES)

    return predict


def _eager_prediction(
    model: model_definition.AnyModel,
    r_cos: np.ndarray,
    z_sin: np.ndarray,
    metrics: np.ndarray,
    n_field_periods: int,
    n_coils: int,
) -> tuple[np.ndarray, ...]:
    boundary = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos,
        z_sin=z_sin,
        n_field_periods=n_field_periods,
        is_stellarator_symmetric=True,
    )
    coilset = model(boundary, _requirement_metrics(metrics, n_coils))
    if _is_variable_shape(model.config):
        coilset = coilset_utils.trim_to_n_coils(coilset, n_coils)
    return tuple(np.asarray(getattr(coilset, name)) for name in OUTPUT_NAMES)


def _requirement_metrics(metrics, n_coils) -> types.RequirementMetrics:
    return types.RequirementMetrics(
        **_metric_fields(metrics), n_coils_per_half_period=n_coils
    )


def _metric_fields(metrics) -> dict:
    # Field order of `RequirementMetrics.to_array()`.
    return {
        "min_normalized_coil_to_coil_distance": metrics[0],
        "min_normalized_coil_to_plasma_distance": metrics[1],
        "max_normalized_coil_curvature": metrics[2],
        "max_normalized_field_error": metrics[3],
    }


def _supported_counts(config: types.AnyModelConfig) -> tuple[list[int], list[int]]:
    if isinstance(config, model_definition.AnyVariableShapeConfig):
        return list(config.n_field_periods_values), list(config.n_coils_values)
    return [config.n_field_periods], [config.n_unique_coils]


def _is_variable_shape(config: types.AnyModelConfig) -> bool:
    return isinstance(config, model_definition.AnyVariableShapeConfig)


def _batch_rank1_gemms(proto: onnx.ModelProto) -> None:
    """Wrap every `Gemm` fed a vector in `Unsqueeze`/`Squeeze`, in place.

    jax2onnx lowers `nnx.Linear` to `Gemm`, which requires a rank-2 input, but the
    models apply their layers to rank-1 vectors.
    """
    graph = proto.graph
    ranks = _value_ranks(proto)
    nodes = []
    for node in graph.node:
        if node.op_type != "Gemm" or ranks.get(node.input[0]) != 1:
            nodes.append(node)
            continue
        vector_in, vector_out = node.input[0], node.output[0]
        node.input[0] = f"{node.name}_input_2d"
        node.output[0] = f"{node.name}_output_2d"
        nodes.extend(
            [
                helper.make_node(
                    "Unsqueeze",
                    [vector_in, "gemm_batch_axis"],
                    [node.input[0]],
                    name=f"{node.name}_unsqueeze",
                ),
                node,
                helper.make_node(
                    "Squeeze",
                    [node.output[0], "gemm_batch_axis"],
                    [vector_out],
                    name=f"{node.name}_squeeze",
                ),
            ]
        )
    graph.initializer.append(
        helper.make_tensor("gemm_batch_axis", onnx.TensorProto.INT64, [1], [0])
    )
    del graph.node[:]
    graph.node.extend(nodes)


def _widen_int_inputs(proto: onnx.ModelProto, names: tuple[str, ...]) -> None:
    """Declare the given int32 graph inputs as int64 and cast them back, in place."""
    graph = proto.graph
    casts = []
    for graph_input in graph.input:
        if graph_input.name not in names:
            continue
        narrow = f"{graph_input.name}_int32"
        _rename_value(graph, graph_input.name, narrow, rename_producers=False)
        graph_input.type.tensor_type.elem_type = onnx.TensorProto.INT64
        casts.append(
            helper.make_node(
                "Cast",
                [graph_input.name],
                [narrow],
                to=onnx.TensorProto.INT32,
                name=f"{narrow}_cast",
            )
        )
    nodes = [*casts, *graph.node]
    del graph.node[:]
    graph.node.extend(nodes)


def _rename_value(
    graph: onnx.GraphProto, old: str, new: str, *, rename_producers: bool = True
) -> None:
    for node in graph.node:
        node.input[:] = [new if name == old else name for name in node.input]
        if rename_producers:
            node.output[:] = [new if name == old else name for name in node.output]


def _value_ranks(proto: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(proto)
    graph = inferred.graph
    return {
        value.name: len(value.type.tensor_type.shape.dim)
        for value in (*graph.input, *graph.value_info, *graph.output)
        if value.type.tensor_type.HasField("shape")
    }


if __name__ == "__main__":
    main()
