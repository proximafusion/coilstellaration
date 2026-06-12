"""Export every bundled `coilstellaration` Flax NNX checkpoint to ONNX.

The `coilstellaration` package ships a handful of trained coil-predictor
checkpoints (Flax NNX, serialized as JSON) under its `data/models/` directory,
across four architectures: `CoilPredictor` (mlp), `ResMlpCoilPredictor`
(res_mlp), and their ensembles. This script turns each one into a
contract-conforming ONNX graph (see the About tab / design spec §4 and
`scripts/make_example_onnx.py`): raw boundary Fourier coefficients +
requirement metrics in, a coilset (`coil_x_n`/`coil_y_n`/`coil_z_n`/
`currents`) out.

Unlike the example baseline, these are the real trained predictors. We do not
transcribe weights by hand: we load each checkpoint and call its *own* trained
`nnx` layers from a small pure-JAX wrapper, then convert that wrapper with
`jax2onnx`. The wrapper only reimplements the cheap orchestration around the
layers (raw spectral-scaling of the boundary, metric standardization, ensemble
averaging) so that the matmuls see a rank-2 `(1, features)` input -- ONNX
`Gemm` requires rank 2, and the model itself runs on rank-1 vectors. Every
export is verified numerically against the model's eager `__call__` before
being written.

Note on shapes: these models have a *fixed* input resolution (the input Linear
has a fixed weight matrix), so each ONNX graph hard-codes its boundary shape.
Benchmark instances at a different spectral resolution will fail the model (and
be scored 0) -- this is inherent to the trained models, not a bug here.

Run:  uv run --with jax2onnx python scripts/export_coilstellaration_onnx.py
      (or just `uv run ...` once jax2onnx is in the dev dependency-group)
Output: examples/<arch>_<id6>.onnx  (e.g. examples/mlp_D2Hbze.onnx)
"""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
from constellaration.geometry import surface_rz_fourier
from flax import nnx
from jax2onnx import to_onnx

from coilstellaration import paths, types
from coilstellaration.machine_learning import model_definition, utils

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

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")

# dapper_type discriminator -> (checkpoint class, short architecture label).
_ARCH = {
    "CoilPredictorConfig": (types.CoilPredictorCheckpoint, "mlp"),
    "ResMlpCoilPredictorConfig": (types.ResMlpCoilPredictorCheckpoint, "res_mlp"),
    "MlpEnsembleCoilPredictorConfig": (
        types.MlpEnsembleCoilPredictorCheckpoint,
        "mlp_ensemble",
    ),
    "ResMlpEnsembleCoilPredictorConfig": (
        types.ResMlpEnsembleCoilPredictorCheckpoint,
        "res_mlp_ensemble",
    ),
}


def _reference_boundary(config) -> surface_rz_fourier.SurfaceRZFourier:
    """A zero boundary at the model's fixed resolution, for static mode layout.

    Only the spectral mode layout (poloidal/toroidal mode numbers, derived from
    array shape + field periods) is used; the coefficient values are overwritten
    by the real input at call time.
    """
    shape = (config.n_poloidal_modes, config.n_toroidal_modes)
    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.zeros(shape),
        z_sin=np.zeros(shape),
        r_sin=None,
        z_cos=None,
        n_field_periods=config.n_field_periods,
        is_stellarator_symmetric=config.is_stellarator_symmetric,
    )


def _boundary_scaling(config) -> tuple[jax.Array, jax.Array]:
    """Per-coefficient exponential spectral scaling for `r_cos`/`z_sin`.

    Mirrors `utils.forward_boundary_to_flat`, which divides the boundary
    coefficients by these factors before flattening.
    """
    scaled = utils.calculate_exponential_spectral_scaling_surface(
        _reference_boundary(config),
        alpha=config.spectral_scaling_alpha,
        order=config.spectral_scaling_order,
        ess_min_value=config.spectral_scaling_min,
    )
    return jnp.asarray(scaled.r_cos), jnp.asarray(scaled.z_sin)


def _run_mlp_member(member, x_2d: jax.Array) -> types.Coilset:
    """`CoilPredictor` trunk on a rank-2 input -> full-band coilset."""
    x = member.input_layer(x_2d)
    for lin in member.block_linears:
        x = nnx.gelu(x)
        x = lin(x)
    x = nnx.gelu(x)
    flat = member.output_layer(x)[0]
    return member.backward_coilset(flat)


def _run_resmlp_member(
    member, x_2d: jax.Array, standardized_2d: jax.Array
) -> types.Coilset:
    """`ResMlpCoilPredictor` trunk on a rank-2 input -> full-band coilset.

    Mirrors `ResMlpCoilPredictor.__call__` and `_CurrentsHead.__call__`, kept
    rank-2 so every matmul lowers to a rank-2 ONNX `Gemm`. Dropout is identity
    because the checkpoint was loaded in eval mode.
    """
    x = member.input_layer(x_2d)
    for norm, lin in zip(member.block_norms, member.block_linears):
        residual = x
        x = norm(x)
        x = nnx.gelu(x)
        x = lin(x)
        x = member.dropout(x)
        x = x + residual
    x = nnx.gelu(x)
    flat_coils = member.coil_head(x)  # (1, coil_flat)

    head = member.currents_head
    h = jnp.concatenate([flat_coils, standardized_2d], axis=1)
    h = head.norm(h)
    h = head.fc1(h)
    h = nnx.gelu(h)
    flat_currents = head.fc2(h)  # (1, n_unique_coils)

    flat = jnp.concatenate([flat_coils[0], flat_currents[0]])
    return member.backward_coilset(flat)


def _mean_coilset(coilsets: list[types.Coilset]) -> types.Coilset:
    """Elementwise mean across ensemble members (mirrors ensemble `__call__`)."""
    first = coilsets[0]
    return first.model_copy(
        update={
            "coil_x_n": jnp.mean(jnp.stack([c.coil_x_n for c in coilsets]), axis=0),
            "coil_y_n": jnp.mean(jnp.stack([c.coil_y_n for c in coilsets]), axis=0),
            "coil_z_n": jnp.mean(jnp.stack([c.coil_z_n for c in coilsets]), axis=0),
            "currents": jnp.mean(jnp.stack([c.currents for c in coilsets]), axis=0),
        }
    )


def _make_predict_fn(model, arch: str):
    """Build the pure-JAX raw-arrays -> coilset-arrays function for one model."""
    config = model.config
    r_cos_scale, z_sin_scale = _boundary_scaling(config)
    # Ensemble members carry the standardization Variables; single models carry
    # them directly.
    stats_owner = model.members[0] if arch.endswith("ensemble") else model
    means = stats_owner.requirement_metrics_means[...]
    stds = stats_owner.requirement_metrics_stds[...]
    is_resmlp = "res_mlp" in arch
    is_ensemble = arch.endswith("ensemble")

    def predict(boundary_r_cos, boundary_z_sin, requirement_metrics):
        flat_boundary = jnp.concatenate(
            [
                (boundary_r_cos / r_cos_scale).ravel(),
                (boundary_z_sin / z_sin_scale).ravel(),
            ]
        )
        standardized = (requirement_metrics - means) / stds
        x_2d = jnp.concatenate([flat_boundary, standardized]).reshape(1, -1)
        std_2d = standardized.reshape(1, -1)

        members = model.members if is_ensemble else [model]
        if is_resmlp:
            coilsets = [_run_resmlp_member(m, x_2d, std_2d) for m in members]
        else:
            coilsets = [_run_mlp_member(m, x_2d) for m in members]
        coilset = _mean_coilset(coilsets) if is_ensemble else coilsets[0]
        return (
            coilset.coil_x_n,
            coilset.coil_y_n,
            coilset.coil_z_n,
            coilset.currents,
        )

    return predict


def _eager_reference(model, config) -> tuple[np.ndarray, ...]:
    """Run the model's own `__call__` on a fixed pseudo-random input."""
    shape = (config.n_poloidal_modes, config.n_toroidal_modes)
    rc = np.asarray(jax.random.normal(jax.random.PRNGKey(0), shape)) * 0.1
    zs = np.asarray(jax.random.normal(jax.random.PRNGKey(1), shape)) * 0.1
    rm = np.array([0.4, 1.2, 3.8, 0.01], dtype=np.float32)
    boundary = _reference_boundary(config).model_copy(
        update={"r_cos": jnp.asarray(rc), "z_sin": jnp.asarray(zs)}
    )
    req = types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=rm[0],
        min_normalized_coil_to_plasma_distance=rm[1],
        max_normalized_coil_curvature=rm[2],
        max_normalized_field_error=rm[3],
    )
    nfo = (config.n_modes_coils_max - 1) // 2
    out = model(boundary, req, fourier_order=nfo)
    inputs = {
        "boundary_r_cos": rc.astype(np.float32),
        "boundary_z_sin": zs.astype(np.float32),
        "requirement_metrics": rm,
    }
    expected = tuple(
        np.asarray(a) for a in (out.coil_x_n, out.coil_y_n, out.coil_z_n, out.currents)
    )
    return inputs, expected


def export_checkpoint(json_path) -> str:
    """Convert one checkpoint JSON to an ONNX file; return the output path."""
    dapper_type = json.loads(json_path.read_text())["config"]["dapper_type"]
    checkpoint_cls, arch = _ARCH[dapper_type]
    ckpt = checkpoint_cls.model_validate_json(json_path.read_text())
    model = model_definition.read_model_from_checkpoint(ckpt)
    config = model.config

    predict = _make_predict_fn(model, arch)
    shape = (config.n_poloidal_modes, config.n_toroidal_modes)
    proto = to_onnx(
        predict,
        [
            jax.ShapeDtypeStruct(shape, jnp.float32),
            jax.ShapeDtypeStruct(shape, jnp.float32),
            jax.ShapeDtypeStruct((4,), jnp.float32),
        ],
        input_names=["boundary_r_cos", "boundary_z_sin", "requirement_metrics"],
        output_names=["coil_x_n", "coil_y_n", "coil_z_n", "currents"],
        model_name=f"coilstellaration_{arch}",
        opset=OPSET,
    )

    # Verification gate: ONNX must reproduce the eager model on a fixed input.
    inputs, expected = _eager_reference(model, config)
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    got = session.run(
        ["coil_x_n", "coil_y_n", "coil_z_n", "currents"], inputs
    )
    for name, exp, act in zip(
        ("coil_x_n", "coil_y_n", "coil_z_n", "currents"), expected, got
    ):
        scale = float(np.max(np.abs(exp))) or 1.0
        max_abs = float(np.max(np.abs(exp - act)))
        if max_abs > REL_TOL * scale:
            raise RuntimeError(
                f"{json_path.name} [{arch}]: ONNX output {name!r} disagrees with "
                f"the JAX model (max|diff|={max_abs:g}, scale={scale:g}, "
                f"rel={max_abs / scale:g} > rel_tol={REL_TOL})"
            )

    out_path = os.path.join(OUT_DIR, f"{arch}_{json_path.stem[:6]}.onnx")
    onnx.save(proto, out_path)
    return out_path


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    json_paths = sorted(paths.models_dir().glob("*.json"))
    if not json_paths:
        raise SystemExit(f"No checkpoints found in {paths.models_dir()}")
    print(f"Exporting {len(json_paths)} checkpoint(s) -> {OUT_DIR}")
    for json_path in json_paths:
        out_path = export_checkpoint(json_path)
        print(f"  {json_path.name}  ->  {os.path.basename(out_path)}  (verified)")
    print("Done.")


if __name__ == "__main__":
    main()
