# Your leaderboard name
TITLE = """<h1 align="center" id="space-title">Coilstellaration Benchmark Leaderboard</h1>"""

INTRODUCTION_TEXT = """
Submit an ONNX model that predicts a stellarator **coilset** from a plasma boundary
(`SurfaceRZFourier`) and a set of requirement metrics. Each model is evaluated on both
benchmark tracks (`fixed_shape`, `variable_shape`) at the `tight` stratum; the headline
**Overall** score is the mean across tracks of the per-instance geometric-mean soft score.
"""

LLM_BENCHMARKS_TEXT = """
## How it works
For each benchmark instance the harness feeds your model the raw boundary Fourier
coefficients and the 4 requirement metrics, reconstructs a coilset from the model output,
evaluates it with the `coilstellaration` physics pipeline, and scores feasibility against
the instance's requirements.

## ONNX contract
**Inputs** (raw, no scaling): `boundary_r_cos`, `boundary_z_sin` `[n_pol, n_tor]`
(+ `boundary_r_sin`, `boundary_z_cos` for non-stellarator-symmetric instances),
`requirement_metrics` `[4]` (order: min coil-coil dist, min coil-plasma dist,
max coil curvature, max field error).
**Outputs**: `coil_x_n`, `coil_y_n`, `coil_z_n` `[n_unique_coils, n_modes]`,
`currents` `[n_unique_coils]`. Use dynamic axes for shapes that vary per instance.
`n_field_periods` and stellarator symmetry are taken from the instance boundary.
"""

EVALUATION_QUEUE_TEXT = """
## Submitting a model
1. Push a public HF model repo containing `model.onnx` that conforms to the ONNX contract
   in the About tab.
2. Enter the repo id (`org/model`), revision, your name/contact, and whether the model is
   stochastic (and K samples).
3. On submit we verify the repo is public and that `model.onnx`'s input/output tensor names
   match the contract. Physics evaluation runs on the backend; watch the queue below.
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r""""""
