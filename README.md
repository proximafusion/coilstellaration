<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/978b76bc-cd9b-4af8-b1f3-18efde7c079f">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/ec4e391a-9044-44ae-93f0-9dd8bed70001">
  <img alt="A dark Proxima logo in light color mode and a light one in dark color mode." src="https://github.com/user-attachments/assets/ec4e391a-9044-44ae-93f0-9dd8bed70001" width=400px>
</picture>

# coilstellaration

Code for analyzing and evaluating stellarator boundaries and coilsets, built on top of [`constellaration`](https://github.com/proximafusion/constellaration). This README has two reading paths:

- **[Use the library](#use-the-library)** — run the bundled examples (REGCOIL, DESC-based optimization, training, prediction).
- **[Run the benchmark](#run-the-benchmark)** — tracks, strata, scoring, and an end-to-end submission walkthrough.

## Install

Three install paths, in order of decreasing convenience.

### Devcontainer (recommended)

Open the repo in VS Code and run **Dev Containers: Reopen in Container**. The first launch installs `hatch`, creates the `default`, `test`, and `lint` Hatch environments, and installs the `pre-commit` git hooks. Artifacts written by the examples land in `/home/vscode/tmp/outputs/coilstellaration/` inside the container, which is bind-mounted to `~/tmp/outputs/coilstellaration/` on the host.

### Without the devcontainer (Docker)

The `.devcontainer/Dockerfile` is self-contained — it installs the system build dependencies for `constellaration` (NetCDF, BLAS/LAPACK, gfortran, CMake) and builds the `regcoil` binary into `/usr/local/bin`. Use it directly with `docker` if you'd rather not open the devcontainer:

```bash
# Build the image
docker build --platform=linux/amd64 -f .devcontainer/Dockerfile -t coilstellaration .

# Make sure the host outputs directory exists before the bind mount.
mkdir -p ~/tmp/outputs

# Run an example. PYTHONPATH=src is normally set by devcontainer.json, so we
# pass it explicitly here. Hatch envs aren't pre-built outside the
# devcontainer lifecycle, so create the default env on first run. The
# second bind mount surfaces `paths.OUTPUTS_PATH` (`Path.home() / "tmp" /
# "outputs" / "coilstellaration"`, which resolves to /root/... when the
# container runs as root) at ~/tmp/outputs/coilstellaration/ on the host.
docker run --rm -it --platform=linux/amd64 --shm-size=16g \
    -v "$PWD":/workspaces/coilstellaration \
    -v ~/tmp/outputs:/root/tmp/outputs \
    -w /workspaces/coilstellaration \
    -e PYTHONPATH=/workspaces/coilstellaration/src \
    coilstellaration \
    bash -c "pipx install hatch && hatch env create && hatch run python examples/run_regcoil.py"
```

For repeated runs, mount a persistent volume at `./venv` (or run an editable install with `pip install -e .`) so you don't recreate the Hatch env on every invocation.

### Native (no devcontainer, no Docker)

You'll need a system Python ≥ 3.12, plus the system build dependencies for `constellaration`'s scientific extensions (notably `booz-xform`, a CMake/Fortran build):

```bash
# Debian/Ubuntu
sudo apt install cmake gfortran libnetcdf-dev ninja-build pkg-config

# Then in the repo root:
pipx install hatch
hatch env create
hatch run python examples/run_regcoil.py
```

`run_regcoil.py` also needs the `regcoil` executable on your `PATH`. Build it from [landreman/regcoil](https://github.com/landreman/regcoil) with the Fortran NetCDF bindings and BLAS/LAPACK (`libnetcdff-dev`, `libopenblas-dev`). `.devcontainer/Dockerfile` has the pinned commit and the `make` flags that work with gfortran ≥ 10.

If a fresh install fails, suspect a missing system build dep before suspecting Python.

## Use the library

Five runnable examples sit under `examples/`: the four below, plus `run_benchmark_scoring.py`, which is covered in [Run the benchmark](#run-the-benchmark). Each is a thin script that exercises a real piece of the pipeline; treat them as starting points for your own work, not as production runners.

### Where outputs go

Every example writes artifacts to `paths.OUTPUTS_PATH`, defined in `src/coilstellaration/paths.py` as `pathlib.Path.home() / "tmp" / "outputs" / "coilstellaration"`. Inside the devcontainer this is bind-mounted to `~/tmp/outputs/coilstellaration/` on the host; outside, the directory is created on first run. The filenames listed below are relative to that directory.

Run any example with:

```bash
hatch run python examples/<file>.py
```

### `run_regcoil.py` — REGCOIL coilset from an equilibrium

Loads a fixed pair of IDs (`requirements_id="DegvKVVq5bPPwbVTAiEhPfR"`, `vmecpp_wout_id="DHTyQRcz3UAR3upPKKvCMfo"`) from the published Hugging Face datasets, builds a REGCOIL coilset, scales the coil currents so that the on-axis field $B_0 = 1\,\mathrm{T}$, and evaluates the DESC metrics on the result.

**Outputs**
- `regcoil_coilset.json` — the REGCOIL coilset as a `Coilset` JSON dump.
- `regcoil_coilset_and_equilibrium.html` — interactive Plotly view of the coilset around the equilibrium.
- `regcoil_coilset_metrics.json` — the achieved `Metrics` for the coilset.

The first run pulls real data from `proxima-fusion/constellaration` and `proxima-fusion/coilstellaration` on Hugging Face, so expect internet access and a few-hundred-MB cache hit.

### `run_desc.py` — REGCOIL warm-start + DESC optimization

Uses the same requirements and equilibrium as `run_regcoil.py`, but loads a precomputed REGCOIL coilset (`DPT3MjzMbmJw5VV2mUrWrmQ`) from `proxima-fusion/coilstellaration` instead of running REGCOIL. It then runs DESC's augmented-Lagrangian coilset optimizer (`data_generation_tasks.optimize_coilset_using_desc`) from that warm start. The example uses `maxiter=10`, which is illustrative rather than production; bump it for real runs.

**Outputs**
- `desc_coilset.json` — the post-DESC coilset.
- `desc_coilset_and_equilibrium.html` — Plotly view (REGCOIL in red, DESC-optimized in blue).
- `desc_coilset_metrics.json` — achieved metrics for the optimized coilset.

### `run_model_training.py` — train an MLP `CoilPredictor`

Trains a 4000-step MLP `CoilPredictor` with `TrainConfig` overrides (`batch_size=64`, `eval_every=10`, `max_wall_time_s=3600`). The script is CPU-friendly: `JAX_PLATFORMS=cpu` is set at the top and `jax_enable_x64=False`. The script will try to log to Weights & Biases — set `WANDB_MODE=offline` first if you don't want that.

**Outputs**
- `<unique_id>.json` — a `CoilPredictorCheckpoint` JSON written via `model_dump_json`. The ID is generated by `data_utils.get_unique_id()` and looks like `D…`.

A full 4000-step run on a laptop CPU takes well under the 1-hour wall-time cap; expect it to complete on its own without hanging.

`TrainConfig.model_kind` selects one of eight model types. `mlp`, `res_mlp`, `mlp_ensemble` and `res_mlp_ensemble` train on the `fixed_shape` track. `per_coil`, `masked`, `per_coil_ensemble` and `masked_ensemble` train on the `variable_shape` track, where the number of field periods and coils varies. Set `validation="train_holdout"` to validate on plasma boundaries held out of the train split instead of on the eval split. `eval_batch_size` must not exceed the number of holdout rows.

### `run_model_prediction.py` — run a saved checkpoint on one eval row

Loads the bundled `mlp` checkpoint (see [Bundled checkpoint](#bundled-checkpoint)), runs it on one row of the `track="fixed_shape", stratum="tight"` eval split, and overlays the predicted coilset on the true coilset.

**Outputs**
- `predicted_coilset.json` — the predicted `Coilset`.
- `predicted_coilset_and_equilibrium.html` — Plotly overlay (predicted in blue, true in green).
- `predicted_coilset_metrics.json` — achieved `Metrics` for the predicted coilset, evaluated by `metrics_utils_v2.evaluate_coilset_metrics_from_boundary`.

### Bundled checkpoint

`src/coilstellaration/data/models/` ships one trained checkpoint, `mlp_DHmCARHZ4ay9rsMDPdjVCu3.json`: an `mlp` model for the `fixed_shape` track (NFP 3, 5 coils per half period). `paths.model_path("DHmCARHZ4ay9rsMDPdjVCu3")` resolves it from its ID alone, and `machine_learning.utils.read_checkpoint_from_id("mlp", "DHmCARHZ4ay9rsMDPdjVCu3")` reads it with the right type. For the other model types, train your own with `run_model_training.py`.

The file format is documented in [`src/coilstellaration/data/models/README.md`](src/coilstellaration/data/models/README.md). Models you train yourself follow the same shape — that's what `run_model_training.py` writes.

## Run the benchmark

The benchmark scores ML models that propose coilsets satisfying a given plasma boundary's requirements. This section walks through what a submission looks like, the two tracks, the three strata, the soft-feasibility scoring math, and the end-to-end code path you run to produce a leaderboard number.

### What you're benchmarking

A submission is a model

$$
f \colon (\text{boundary},\ \text{target metrics}) \longrightarrow \text{predicted coilset}
$$

where:

- `boundary` is a `surface_rz_fourier.SurfaceRZFourier` (the plasma surface in Fourier representation).
- `target metrics` is a `RequirementMetrics` (`src/coilstellaration/types.py`) — four normalized scalars: `min_normalized_coil_to_coil_distance`, `min_normalized_coil_to_plasma_distance`, `max_normalized_coil_curvature`, `max_normalized_field_error`, plus the integer `n_coils_per_half_period` (the target number of unique coils). A predicted coilset with a different coil count scores 0. `max_normalized_coil_curvature` is the largest curvature magnitude over all coils: DESC signs curvature by convexity, and a tight concave bend counts as much as a convex one.
- `predicted coilset` is a `Coilset` (`src/coilstellaration/types.py`) in DESC's `FourierXYZCoil` representation, storing only unique pre-symmetry coils.

The eval set is a fixed list of `(boundary, target)` rows published in `proxima-fusion/coilstellaration` on Hugging Face. Scoring re-evaluates the **achieved** metrics on your predicted coilset with `metrics_utils_v2.evaluate_coilset_metrics`. It does this on the row's DESC equilibrium (`EvalData.desc_equilibrium`), the equilibrium the targets were computed on, and compares the result to the target with the soft-feasibility pipeline below. You never get to see the held-out targets at training time — they live only in the `eval` split.

### Tracks: `fixed_shape` and `variable_shape`

Two independent tracks, selected by the `track=` argument to `load_benchmark_dataset` (`src/coilstellaration/data_utils.py`):

- **`fixed_shape`** — every train and eval row uses NFP 3 and 5 coils per half period, across many plasma boundaries. This is the easier of the two and a natural starting point. Filtered by the `benchmark/fixed_shape_track` boolean column on the dataset.
- **`variable_shape`** — the number of field periods and coils per half period vary across rows; the model must generalize over boundary geometry, coil count and target metrics. Filtered by the `benchmark/variable_shape_track` boolean column.

A model trained for one track is not expected to transfer to the other, and the two are scored independently on the leaderboard.

```python
from coilstellaration import data_utils

eval_fixed = data_utils.load_benchmark_dataset(
    track="fixed_shape", stratum="tight", split="eval", n=0
)
eval_variable = data_utils.load_benchmark_dataset(
    track="variable_shape", stratum="tight", split="eval", n=0
)
```

For training data, `data_utils.load_ml_baseline_dataframe(track=..., split=...)` selects the ML-baseline rows of a track (default `fixed_shape`): `fixed_shape` uses `baseline_labels/used_for_ml_baseline`; `variable_shape` uses rows in `benchmark/variable_shape_track` that also satisfy `baseline_labels/meets_relative_error_thresholds`. Loaded rows are checked for consistent coil counts and field periods.

`n=0` means "load the whole filtered split"; pass `n=4` for a quick smoke test (this is what `examples/run_benchmark_scoring.py` does by default with `--n 4`).

### Strata: `loose`, `medium`, `tight`

Each row in the dataset belongs to exactly one stratum, recorded in the `benchmark/stratification` column (the constant `StratificationColumn` in `src/coilstellaration/data_utils.py`). Strata are **tertiles of the column `desc_metrics/normalized_field_error/mean`** (the constant `FIELD_ERROR_MEAN_COLUMN` in the same file) — the mean normalized field error achieved by the data-generation pipeline that produced that row's coilset. In other words, strata partition the dataset by how cleanly the upstream pipeline could solve the problem:

- **`tight`** — bottom tertile. Lowest achievable field error in the source pipeline, i.e. the most demanding requirements. **This is the headline stratum reported on the leaderboard, and what the rest of this section walks through.**
- **`medium`** — middle tertile.
- **`loose`** — top tertile. Highest field error, i.e. the most permissive requirements; easiest.

Switch strata by passing `stratum="loose"` or `"medium"` to `load_benchmark_dataset`. The default in `data_utils.load_benchmark_dataset` is `"tight"`.

### Scoring (soft feasibility)

Scoring is implemented in `src/coilstellaration/benchmark/scoring.py` and parameterized by `ScoringSettings` in `src/coilstellaration/benchmark/scoring_types.py`. The pipeline has four stages: per-metric normalized violation, saturating soft-score map, metric aggregation (per instance), instance aggregation (across the eval set). Each stage below states the formula, the symbol meanings, and the default values pulled from `ScoringSettings`.

#### 1. Per-metric normalized violation

For each of the four `RequirementMetrics` fields, compute a normalized violation $v_i \geq 0$. $v_i = 0$ means the constraint is satisfied on that metric.

For an **upper-bound** metric (`max_normalized_coil_curvature`, `max_normalized_field_error` — lower is better):

$$
v_i = \max\left(0,\ \frac{g_i - \tau_i}{\max(|\tau_i|,\ \mathrm{floor})}\right)
$$

For a **lower-bound** metric (`min_normalized_coil_to_coil_distance`, `min_normalized_coil_to_plasma_distance` — higher is better):

$$
v_i = \max\left(0,\ \frac{\tau_i - g_i}{\max(|\tau_i|,\ \mathrm{floor})}\right)
$$

- $g_i$ — the achieved value evaluated on the predicted coilset.
- $\tau_i$ — the target value from the eval row's `RequirementMetrics`.
- $\mathrm{floor}$ — `ScoringSettings.target_floor`, default `1e-6`. It clamps $|\tau_i|$ from below so that targets at or near zero don't blow $v_i$ up to absurd values.

A non-finite $g_i$ or $\tau_i$ (e.g. from a failed evaluation) gives $v_i = \infty$, as does every metric of a coilset with the wrong number of coils per half period.

The direction (upper/lower bound) for each field is fixed in `REQUIREMENT_METRIC_DIRECTIONS` (`src/coilstellaration/benchmark/scoring_types.py`).

#### 2. Soft-score map

Each $v_i \geq 0$ maps to a soft score $s_i \in [0, 1]$ by a saturating function. Two options, controlled by `ScoringSettings.soft_score`:

Default (`soft_score = "exponential"`):

$$
s_i = \exp(-v_i / \alpha_i)
$$

Alternative (`soft_score = "linear_ramp"`):

$$
s_i = \max(0,\ 1 - v_i / \alpha_i)
$$

Per-metric tolerances $\alpha_i$ come from `ScoringSettings.tolerances`. The defaults:

| Metric                                        | $\alpha_i$ |
| --------------------------------------------- | ---------- |
| `min_normalized_coil_to_coil_distance`        | 0.10       |
| `min_normalized_coil_to_plasma_distance`      | 0.10       |
| `max_normalized_coil_curvature`               | 0.10       |
| `max_normalized_field_error`                  | 0.25       |

Interpretation: a 10% relative violation of a distance or curvature constraint cuts the score to $1/e$ under the default exponential map; field error tolerates a 25% relative violation before the same cut. These values are physics judgment, not tuned.

#### 3. Metric aggregation (per instance)

Per-metric soft scores collapse to one scalar per instance, $S_j$. The headline aggregation is the **geometric mean**:

$$
S_j = \exp\left(\frac{1}{n} \sum_{i=1}^{n} \log s_i\right)
$$

A single $s_i = 0$ collapses $S_j$ to $0$ — that matches feasibility semantics (one hard violation kills the instance).

Three other aggregations are reported as diagnostics, controlled by `ScoringSettings.metric_aggregation`:

- **`min`** — strict worst-case across metrics.
- **`arithmetic_mean`** — more lenient than geometric; useful for spotting when geometric is being collapsed by a single bad metric.
- **`weighted_geometric_mean`** — geometric mean with per-metric weights summing to 1; only computed when `ScoringSettings.metric_weights` is provided.

The headline column on the leaderboard is the geometric mean.

#### 4. Instance aggregation (across the eval set)

Per-instance scalars $S_j$ collapse across the eval set into named diagnostics, controlled by `ScoringSettings.instance_aggregations` (`src/coilstellaration/benchmark/scoring_types.py`). The defaults compute all five:

| Diagnostic               | What it tells you                                                  |
| ------------------------ | ------------------------------------------------------------------ |
| `mean`                   | **Headline score.** Larger is better.                              |
| `median`                 | Robustness check against tail outliers in `mean`.                  |
| `p10`                    | 10th percentile of $S_j$. Tail diagnostic — how bad are the worst? |
| `feasibility_rate_0p9`   | Fraction of instances with $S_j \geq 0.9$ (soft-feasible).         |
| `strict_feasibility`     | Fraction with $v_i \leq$ `strict_feasibility_tolerance` (default `1e-3`) for every metric. |

The leaderboard ranks by `strict_feasibility`, then by `mean`. The other three are reported alongside them for context.

### End-to-end walkthrough

Six steps to score a model on the public eval set. They score the model's prediction as is, as the Hugging Face Space does. `examples/run_benchmark_scoring.py` runs the same pipeline from the command line, with an optional DESC refinement on top; this section explains the *why* of each step.

> **Quick check only.** `examples/run_benchmark_scoring.py` is a local quick check on a small subset of the eval split. Official benchmark numbers come from the Hugging Face Space.

```bash
python examples/run_benchmark_scoring.py --checkpoint-id <id> --model-kind mlp \
    --track fixed_shape --stratum tight --n 4
```

Flags:

- `--checkpoint-id` (required) — the ID of a checkpoint in `src/coilstellaration/data/models/`, e.g. the bundled `DHmCARHZ4ay9rsMDPdjVCu3`. To score a model you trained, copy its JSON there. The ID also names the output files.
- `--model-kind` — any `AnyModelLiteral`; default `mlp`.
- `--track` — `fixed_shape` (default) or `variable_shape`.
- `--stratum` — `loose`, `medium`, `tight` (default) or `all`; `all` scores each stratum.
- `--n` — instances per stratum, default 4; `0` loads all of them.
- `--stratify-nfp-nc` — sample `--n` instances per NFP × coils-per-half-period cell instead.
- `--seed` — sampling seed, used only with `--stratify-nfp-nc`.
- `--desc-refine-from` — the starting coilset: `model` (default, the prediction) or `regcoil` (the dataset's REGCOIL coilset of each row).
- `--desc-maxiters` — iterations of DESC's coilset optimization to refine the starting coilset with, under each instance's own requirements, before scoring. The default, `0`, scores the starting coilset as is, like the Space. Repeat the flag to score several values in one worker per instance, which then compiles once. Scanning it from both starting points on the same rows measures how much the model's prediction speeds up the optimization.
- `--instance-timeout-s` — default 1800. Each instance runs in a fresh subprocess, and the timeout covers all of its `--desc-maxiters`. A value that times out or fails scores zero.

Outputs are written to `paths.OUTPUTS_PATH`, one set per `--desc-maxiters` value, with `<run>` = `<checkpoint-id>_desc_from_<start>_maxiter<n>`:

- `quickcheck_scores_<run>.csv` — one row per instance; see [What to submit](#what-to-submit).
- `quickcheck_summary_<run>.csv` — the five diagnostics, overall and per stratum.
- `quickcheck_summary_nfp_nc_<run>.csv` — mean score, strict feasibility and count per NFP × coils-per-half-period cell.
- `quickcheck_soft_feasibility_<run>.png` — histograms of the four per-metric soft scores and the aggregate.

Run with `--help` for the full list of flags and defaults.

**1. Load the eval split.** `n=0` means full split; bump it down for a smoke test. Only the `eval` split is scored — `train` is for fitting only.

```python
from coilstellaration import data_utils

# n=4 is a quick check; n=0 loads the full eval split (slow). The CLI's `--n` does the same.
eval_dataset = data_utils.load_benchmark_dataset(
    track="fixed_shape", stratum="tight", split="eval", n=4
)
```

**2. Load your model.** The bundled checkpoint follows the `FlaxNnxCheckpoint` JSON format (see [`src/coilstellaration/data/models/README.md`](src/coilstellaration/data/models/README.md)). Models you train with `run_model_training.py` follow the same shape. `read_checkpoint_from_id` reads `data/models/<model_kind>_<id>.json`; pass `model_data_path=` to read a checkpoint stored somewhere else, such as one you trained.

```python
from coilstellaration.machine_learning import model_definition
from coilstellaration.machine_learning import utils as ml_utils

checkpoint = ml_utils.read_checkpoint_from_id("mlp", "DHmCARHZ4ay9rsMDPdjVCu3")
model = model_definition.read_model_from_checkpoint(checkpoint)  # in eval mode
```

**3. Predict coilsets.** `predict_coilsets` returns a list of `EvalData` with `predicted_coilset` populated.

```python
predictions = model_definition.predict_coilsets(model, eval_dataset)
```

**4. Evaluate achieved metrics.** This is the costly step — DESC re-evaluates each predicted coilset on the row's DESC equilibrium, the one the targets were computed on. An equilibrium rebuilt from the boundary alone can have a different minor radius and axis, so even the reference coilset may miss its targets on it. `metrics_to_requirement_metrics` projects the full `Metrics` down to the four scalars the benchmark scores against, plus the coil count.

```python
import warnings

from coilstellaration import coilset_utils, data_utils, desc_utils, metrics_utils_v2

achieved_metrics_per_instance = []
for eval_data in predictions:
    assert eval_data.predicted_coilset is not None
    assert eval_data.desc_equilibrium is not None
    with warnings.catch_warnings(action="ignore"):
        achieved_metrics = metrics_utils_v2.evaluate_coilset_metrics(
            eq=desc_utils.desc_equilibrium_to_desc_object(eval_data.desc_equilibrium),
            coilset=coilset_utils.coilstellaration_to_desc(
                eval_data.predicted_coilset
            ),
        )
    achieved_metrics_per_instance.append(
        data_utils.metrics_to_requirement_metrics(achieved_metrics)
    )
```

**5. Score.** Per-instance, then aggregate across the eval set.

```python
from coilstellaration.benchmark import scoring, scoring_types

settings = scoring_types.ScoringSettings()  # all defaults
instance_scores = [
    scoring.score_eval_data(eval_data, achieved, settings)
    for eval_data, achieved in zip(predictions, achieved_metrics_per_instance)
]
benchmark = scoring.score_benchmark(instance_scores, settings)
print(benchmark.summaries)
```

**6. Inspect outputs.** Two tabular helpers and a histogram.

```python
import pandas as pd

# Per-instance, one row per boundary, with all violation/soft_score columns:
scores_df = pd.DataFrame(
    [
        {"boundary_id": s.boundary_id, **scoring.instance_score_to_row(s)}
        for s in instance_scores
    ]
)

# Per-aggregation summary table (geometric_mean, min, arithmetic_mean):
summary_df = scoring.summarize_scores(
    scoring.instance_scores_to_dataframe(instance_scores),
    feasibility_threshold=0.9,
)
```

`benchmark.summaries["mean"]` and `benchmark.summaries["strict_feasibility"]` under the default `ScoringSettings` are the numbers the leaderboard ranks by. `examples/run_benchmark_scoring.py` additionally writes a PNG of soft-score histograms to `paths.OUTPUTS_PATH`.

### What to submit

Official scores come from the coilstellaration benchmark Space on Hugging Face, which scores submitted models itself. Submit a public Hugging Face model repo containing a `model.onnx` that follows the ONNX contract in the Space's About tab:

- **Inputs** (unscaled): the boundary's `boundary_r_cos` and `boundary_z_sin`, `[n_pol, n_tor]`, plus `boundary_r_sin` and `boundary_z_cos` for non-stellarator-symmetric instances; and `requirement_metrics`, `[4]`, in `RequirementMetrics` field order.
- **Outputs**: `coil_x_n`, `coil_y_n`, `coil_z_n`, `[n_unique_coils, n_modes]`, and `currents`, `[n_unique_coils]`. Use dynamic axes for shapes that vary per instance. The number of field periods and stellarator symmetry come from the instance.

The Space evaluates each model on every track it supports, over all three strata, with the default `ScoringSettings` listed below. Its leaderboard shows one track at a time, either over all strata pooled or for a single stratum, and ranks by `strict_feasibility`, then `mean`.

For your own diagnostics, `examples/run_benchmark_scoring.py` writes a per-instance CSV, `quickcheck_scores_<run>.csv`, to `paths.OUTPUTS_PATH`:

| Column                        | Meaning                                                              |
| ----------------------------- | -------------------------------------------------------------------- |
| `boundary_id`                 | Identifier of the eval row's boundary.                               |
| `stratum`                     | Stratum the row was loaded from (a boundary can appear in several).  |
| `boundary.n_field_periods`    | Number of field periods of the boundary.                             |
| `reqs/n_coils_per_half_period`| Required number of unique coils.                                     |
| `desc_coilset_id`             | Identifier of the reference coilset of the row.                      |
| `score`                       | Per-instance scalar $S_j$ under the default geometric-mean aggregation. |
| `strictly_feasible`           | `True` iff $v_i \leq$ `strict_feasibility_tolerance` for every metric on this instance. |
| `violation/<field>` (×4)      | Per-metric normalized violation $v_i$.                               |
| `soft_score/<field>` (×4)     | Per-metric soft score $s_i$ under the default exponential map.       |

The per-instance breakdown is enough to recompute every aggregation, including tail diagnostics like `p10` and `strict_feasibility`.

The default `ScoringSettings`, which the Space uses:
- `soft_score = "exponential"`
- `metric_aggregation = "geometric_mean"`
- `tolerances` as listed in the [Soft-score map](#2-soft-score-map) table
- `target_floor = 1e-6`
- `strict_feasibility_tolerance = 1e-3`
- `instance_aggregations = ["mean", "median", "p10", "feasibility_rate_0p9", "strict_feasibility"]`

When you quote numbers outside the leaderboard, report `fixed_shape` and `variable_shape` separately and label the stratum (e.g. `fixed_shape / tight / mean = 0.84`). Reach out via the [issue tracker](https://github.com/proximafusion/coilstellaration/issues) if you need a link to the Space.

## Datasets

The benchmark pulls from two public Hugging Face datasets, joined at load time by `data_utils.load_dataframes` (`src/coilstellaration/data_utils.py`).

- **[`proxima-fusion/coilstellaration`](https://huggingface.co/datasets/proxima-fusion/coilstellaration)** — the `results` table (the `desc_metrics/...`, `reqs/...`, `benchmark/...` and `baseline_labels/...` columns, split into `train` and `eval`), plus the `coilsets` and `requirements` tables it references via `desc_coilset_id` and `requirements_id`. The strata tertiles are computed from its column `desc_metrics/normalized_field_error/mean`.
- **[`proxima-fusion/constellaration`](https://huggingface.co/datasets/proxima-fusion/constellaration)** — plasma boundaries, their VMEC++ and DESC equilibria, and the ConStellaration metrics (loaded as `constellaration_metrics/...`). Joined to each `results` row via `constellaration_boundary_id`.

The columns relevant to selecting a track or stratum:

| Column                              | Type   | Meaning                                                                  |
| ----------------------------------- | ------ | ------------------------------------------------------------------------ |
| `benchmark/fixed_shape_track`       | bool   | Row included in the `fixed_shape` track.                                 |
| `benchmark/variable_shape_track`    | bool   | Row included in the `variable_shape` track.                              |
| `benchmark/stratification`          | str    | Tertile of `desc_metrics/normalized_field_error/mean`: `"loose"` / `"medium"` / `"tight"`. |

`load_benchmark_dataset(track=..., stratum=...)` filters on these columns for you; you only need to look at them directly if you want to slice the dataset some other way.

## Develop

Three Hatch environments are declared in `pyproject.toml`. The devcontainer creates all three automatically; outside the devcontainer, run `hatch env create` for each.

```bash
hatch env create               # default env (lives at ./venv)
hatch env create test          # adds pytest
hatch env create lint          # adds ruff and pre-commit

hatch run test:pytest          # run all tests
hatch run lint:pre-commit run --all-files
```

Pre-commit hooks (`ruff --fix`, `black`, `isort`, `pyright`, plus generic hygiene hooks) are auto-installed by the devcontainer's `onCreateCommand`. If you're not in the devcontainer, install them with `pre-commit install` after `hatch env create lint`.

The `default` env intentionally uses `path = "venv"` (rather than the Hatch default `.hatch/`) so that VS Code and Pyright pick it up via `python.defaultInterpreterPath` and `[tool.pyright].venv`.

## License

MIT — see [`LICENSE`](LICENSE).
