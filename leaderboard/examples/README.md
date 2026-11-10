# Example submission model

`model.onnx` is a minimal, **contract-conforming** Coilstellaration submission.
Regenerate it with:

```bash
python scripts/make_example_onnx.py
```

It is a *baseline*, not a good predictor: it ignores its inputs and emits a
fixed set of planar circular coils, so it runs end-to-end and scores ~0. Use it
as a template for the exact tensor names, shapes, and dynamic axes a real
submission must expose.

## ONNX I/O contract

All values are **raw** (no scaling/normalisation — your model owns any internal
encoding).

**Inputs**

| Name                  | Shape                | Notes                                              |
|-----------------------|----------------------|----------------------------------------------------|
| `boundary_r_cos`      | `[n_pol, n_tor]`     | declare these axes **dynamic** — they vary per instance |
| `boundary_z_sin`      | `[n_pol, n_tor]`     | dynamic                                            |
| `boundary_r_sin`      | `[n_pol, n_tor]`     | only for non-stellarator-symmetric instances       |
| `boundary_z_cos`      | `[n_pol, n_tor]`     | only for non-stellarator-symmetric instances       |
| `requirement_metrics` | `[4]`                | min coil-coil dist, min coil-plasma dist, max coil curvature, max field error |

**Outputs**

| Name        | Shape                        | Notes                                  |
|-------------|------------------------------|----------------------------------------|
| `coil_x_n`  | `[n_unique_coils, n_modes]`  | DESC layout `[sin(N)…sin(1), const, cos(1)…cos(N)]`, `n_modes = 2N+1` |
| `coil_y_n`  | `[n_unique_coils, n_modes]`  |                                        |
| `coil_z_n`  | `[n_unique_coils, n_modes]`  |                                        |
| `currents`  | `[n_unique_coils]`           | Amperes, per unique (pre-symmetry) coil |

`n_field_periods` and stellarator symmetry are taken from the instance boundary
by the harness — the model does not emit them. For the `variable_shape` track,
declare the output coil dimensions dynamic too.

## Validate locally

```python
from src.backend.onnx_model import validate_onnx_contract

print(validate_onnx_contract("examples/model.onnx"))  # [] means valid
```
