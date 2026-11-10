# Example submission model

`dummy-constant-circle.onnx` is a minimal, **contract-conforming** Coilstellaration
submission. Regenerate it with:

```bash
uv run python -m leaderboard.scripts.make_example_onnx
```

It is a *baseline*, not a good predictor: it ignores the boundary and requirement
inputs and emits `n_coils_per_half_period` planar circular coils, so it runs
end-to-end and scores about 0. Use it as a template for the tensor names, shapes,
dynamic axes and metadata that a real submission must have.

## ONNX I/O contract

This is the same contract as in the leaderboard's **About** tab. All tensors are
**raw**: no scaling or normalization. Your model owns any internal encoding.

**Inputs**

| Name | dtype | Shape | Notes |
|---|---|---|---|
| `boundary_r_cos` | float32 | `[n_pol, n_tor]` | Boundary Fourier coefficients. Declare both axes dynamic. |
| `boundary_z_sin` | float32 | `[n_pol, n_tor]` | Dynamic. |
| `boundary_r_sin` | float32 | `[n_pol, n_tor]` | *Optional.* Fed only for non-stellarator-symmetric instances. |
| `boundary_z_cos` | float32 | `[n_pol, n_tor]` | *Optional.* Fed only for non-stellarator-symmetric instances. |
| `requirement_metrics` | float32 | `[4]` | In order: min normalized coil–coil distance, min normalized coil–plasma distance, max normalized coil curvature, max normalized field error. |
| `n_field_periods` | int64 | `[]` (scalar) | Number of field periods of the instance. |
| `n_coils_per_half_period` | int64 | `[]` (scalar) | Number of unique coils the output must have. |

**Outputs**

| Name | Shape | Notes |
|---|---|---|
| `coil_x_n` | `[n_unique_coils, n_modes]` | DESC layout `[sin(N)…sin(1), const, cos(1)…cos(N)]`, `n_modes = 2N+1`. |
| `coil_y_n` | `[n_unique_coils, n_modes]` | Same layout. |
| `coil_z_n` | `[n_unique_coils, n_modes]` | Same layout. |
| `currents` | `[n_unique_coils]` | Amperes, one per unique (pre-symmetry) coil. |

`n_unique_coils` must equal `n_coils_per_half_period`. A coilset with a different
coil count, or with NaN values, scores 0 on that instance. On `fixed_shape` this is
always 5. On `variable_shape`, declare the coil axis dynamic. You choose the
Fourier order `N`. Stellarator symmetry is taken from the instance, not from the
model.

**Metadata.** Set the ONNX `metadata_props` key `supported_tracks` to exactly
`fixed_shape` or `fixed_shape,variable_shape`. A model without it is rejected.
For example:

```python
import onnx

model = onnx.load("model.onnx")
onnx.helper.set_metadata_props(model, {"supported_tracks": "fixed_shape"})
onnx.save(model, "model.onnx")
```

**Size.** The model file must be at most 50 MiB.

**Repo.** The model must be in a **public** Hugging Face model repo. The
leaderboard reads submissions without a token, so it can't see private repos.

## Validate locally

From the repository root, check a local file:

```python
from leaderboard.src.backend.onnx_model import inspect_onnx

errors, tracks = inspect_onnx("leaderboard/examples/dummy-constant-circle.onnx")
print(errors)  # [] means the contract check passed
print(tracks)  # the tracks the model declares
```

Or check a Hub repo the way the leaderboard will, plus a dummy forward pass:

```bash
uv run python -m leaderboard.scripts.check_submission <org/model>
```
