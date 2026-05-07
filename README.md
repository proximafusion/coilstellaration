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
    -v "$PWD":/workspaces/constellaration_update \
    -v ~/tmp/outputs:/root/tmp/outputs \
    -w /workspaces/constellaration_update \
    -e PYTHONPATH=/workspaces/constellaration_update/src \
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

If a fresh install fails, suspect a missing system build dep before suspecting Python.

## Use the library

Four runnable examples sit under `examples/`. Each is a thin script that exercises a real piece of the pipeline; treat them as starting points for your own work, not as production runners.

### Where outputs go

Every example writes artifacts to `paths.OUTPUTS_PATH`, defined in `src/coilstellaration/paths.py:31` as `pathlib.Path.home() / "tmp" / "outputs" / "coilstellaration"`. Inside the devcontainer this is bind-mounted to `~/tmp/outputs/coilstellaration/` on the host; outside, the directory is created on first run. The filenames listed below are relative to that directory.

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

Same setup as `run_regcoil.py`, then runs DESC's augmented-Lagrangian coilset optimizer (`data_generation_tasks_no_proxima.optimize_coilset_using_desc`) on top of the REGCOIL warm-start. The example uses `maxiter=10`, which is illustrative rather than production; bump it for real runs.

**Outputs**
- `desc_coilset.json` — the post-DESC coilset.
- `desc_coilset_and_equilibrium.html` — Plotly view (REGCOIL in red, DESC-optimized in blue).
- `desc_coilset_metrics.json` — achieved metrics for the optimized coilset.

### `run_model_training.py` — train an MLP `CoilPredictor`

Trains a 4000-step MLP `CoilPredictor` with `TrainConfig` overrides (`batch_size=64`, `eval_every=10`, `max_wall_time_s=3600`). The script is CPU-friendly: `JAX_PLATFORMS=cpu` is set at the top and `jax_enable_x64=False`. The script will try to log to Weights & Biases — set `WANDB_MODE=offline` first if you don't want that.

**Outputs**
- `<unique_id>.json` — a `CoilPredictorCheckpoint` JSON written via `model_dump_json`. The ID is generated by `data_utils.get_unique_id()` and looks like `D…`.

A full 4000-step run on a laptop CPU takes well under the 1-hour wall-time cap; expect it to complete on its own without hanging.

### `run_model_prediction.py` — run a saved checkpoint on one eval row

Loads the bundled checkpoint `D2HbzeYjo57Aif48z5T6axt` (one of three bundled MLP checkpoints; `src/coilstellaration/data/models/` ships 12 checkpoints in total — three each across `mlp` / `res_mlp` / `mlp_ensemble` / `res_mlp_ensemble`), runs it on one row of the `track="fixed_shape", stratum="tight"` eval split, and overlays the predicted coilset on the true coilset.

**Outputs**
- `predicted_coilset.json` — the predicted `Coilset`.
- `predicted_coilset_and_equilibrium.html` — Plotly overlay (predicted in blue, true in green).
- `predicted_coilset_metrics.json` — achieved `Metrics` for the predicted coilset, evaluated by `metrics_utils_v2.evaluate_coilset_metrics_from_boundary`.

The bundled checkpoint format is documented in [`src/coilstellaration/data/models/README.md`](src/coilstellaration/data/models/README.md). Models you train yourself follow the same shape — that's what `run_model_training.py` writes.

## Run the benchmark

The benchmark scores ML models that propose coilsets satisfying a given plasma boundary's requirements. This section walks through what a submission looks like, the two tracks, the three strata, the soft-feasibility scoring math, and the end-to-end code path you run to produce a leaderboard number.

### What you're benchmarking

A submission is a model

$$
f \colon (\text{boundary},\ \text{target metrics}) \longrightarrow \text{predicted coilset}
$$

where:

- `boundary` is a `surface_rz_fourier.SurfaceRZFourier` (the plasma surface in Fourier representation).
- `target metrics` is a `RequirementMetrics` (`src/coilstellaration/types.py:404`) — four normalized scalars: `min_normalized_coil_to_coil_distance`, `min_normalized_coil_to_plasma_distance`, `max_normalized_coil_curvature`, `max_normalized_field_error`.
- `predicted coilset` is a `Coilset` (`src/coilstellaration/types.py:355`) in DESC's `FourierXYZCoil` representation, storing only unique pre-symmetry coils.

The eval set is a fixed list of `(boundary, target)` rows published in `proxima-fusion/coilstellaration` on Hugging Face. Scoring re-evaluates the **achieved** metrics on your predicted coilset (via `metrics_utils_v2.evaluate_coilset_metrics_from_boundary`) and compares them to the target with the soft-feasibility pipeline below. You never get to see the held-out targets at training time — they live only in the `eval` split.

### Tracks: `fixed_shape` and `variable_shape`

Two independent tracks, selected by the `track=` argument to `load_benchmark_dataset` (`src/coilstellaration/data_utils.py:280`):

- **`fixed_shape`** — every train and eval row shares the same plasma boundary. The model only generalizes over requirement targets. This is the easier of the two and a natural starting point. Filtered by the `benchmark/fixed_shape_track` boolean column on the dataset.
- **`variable_shape`** — boundaries vary across rows; the model must generalize over both boundary geometry and target metrics. Filtered by the `benchmark/variable_shape_track` boolean column.

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

`n=0` means "load the whole filtered split"; pass `n=4` for a quick smoke test (this is what `examples/run_benchmark_scoring.py` does with `EVAL_N = 4`).

### Strata: `loose`, `medium`, `tight`

Each row in the dataset belongs to exactly one stratum, recorded in the `benchmark/stratification` column (the constant `StratificationColumn` in `src/coilstellaration/data_utils.py:49`). Strata are **tertiles of the column `desc_metrics/normalized_field_error/mean`** (the constant `FIELD_ERROR_MEAN_COLUMN` in `src/coilstellaration/data_utils.py:145`) — the mean normalized field error achieved by the data-generation pipeline that produced that row's coilset. In other words, strata partition the dataset by how cleanly the upstream pipeline could solve the problem:

- **`tight`** — bottom tertile. Lowest achievable field error in the source pipeline, i.e. the most demanding requirements. **This is the headline stratum reported on the leaderboard, and what the rest of this section walks through.**
- **`medium`** — middle tertile.
- **`loose`** — top tertile. Highest field error, i.e. the most permissive requirements; easiest.

Switch strata by passing `stratum="loose"` or `"medium"` to `load_benchmark_dataset`. The default in `data_utils.load_benchmark_dataset` is `"tight"`.

### Scoring (soft feasibility)

Scoring is implemented in `src/coilstellaration/benchmark/scoring.py` and parameterized by `ScoringSettings` in `src/coilstellaration/benchmark/types.py`. The pipeline has four stages: per-metric normalized violation, saturating soft-score map, metric aggregation (per instance), instance aggregation (across the eval set). Each stage below states the formula, the symbol meanings, and the default values pulled from `ScoringSettings`.

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

The direction (upper/lower bound) for each field is fixed in `REQUIREMENT_METRIC_DIRECTIONS` (`src/coilstellaration/benchmark/types.py:20`).

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

Per-instance scalars $S_j$ collapse across the eval set into named diagnostics, controlled by `ScoringSettings.instance_aggregations` (`src/coilstellaration/benchmark/types.py:82`). The defaults compute all five:

| Diagnostic               | What it tells you                                                  |
| ------------------------ | ------------------------------------------------------------------ |
| `mean`                   | **Headline ranking column.** Larger is better.                     |
| `median`                 | Robustness check against tail outliers in `mean`.                  |
| `p10`                    | 10th percentile of $S_j$. Tail diagnostic — how bad are the worst? |
| `feasibility_rate_0p9`   | Fraction of instances with $S_j \geq 0.9$ (soft-feasible).         |
| `strict_feasibility`     | Fraction with $v_i = 0$ for every metric (every hard bound met).   |

`mean` is the leaderboard column under default settings. The other four are reported alongside it for context.

### End-to-end walkthrough

Six steps to score a model on the public eval set. The canonical implementation is `examples/run_benchmark_scoring.py`; this section explains the *why* of each step. To run end-to-end without reading the prose, just run that example.

**1. Load the eval split.** `n=0` means full split; bump it down for a smoke test. Only the `eval` split is scored — `train` is for fitting only.

```python
from coilstellaration import data_utils

EVAL_N = 4  # public-test cap; 0 to score the full eval split (slow)
eval_dataset = data_utils.load_benchmark_dataset(
    track="fixed_shape", stratum="tight", split="eval", n=EVAL_N
)
```

**2. Load your model.** Bundled checkpoints follow the `FlaxNnxCheckpoint` JSON format (see [`src/coilstellaration/data/models/README.md`](src/coilstellaration/data/models/README.md)). Models you train with `run_model_training.py` follow the same shape.

```python
from coilstellaration import flax_nnx_checkpoint_util, paths, types
from coilstellaration.machine_learning import model_definition

MODEL_CHECKPOINT_ID = "D2HbzeYjo57Aif48z5T6axt"
model_path = paths.model_path(MODEL_CHECKPOINT_ID)
checkpoint = types.CoilPredictorCheckpoint.model_validate_json(
    model_path.read_text()
)
model = flax_nnx_checkpoint_util.from_checkpoint(
    checkpoint, module_cls=model_definition.CoilPredictor
)
model.eval()
```

**3. Predict coilsets.** `predict_coilsets` returns a list of `EvalData` with `predicted_coilset` populated.

```python
predictions = model_definition.predict_coilsets(model, eval_dataset)
```

**4. Evaluate achieved metrics.** This is the costly step — DESC re-evaluates each predicted coilset against the boundary. `metrics_to_requirement_metrics` projects the full `Metrics` down to the four scalars the benchmark scores against.

```python
import warnings
from constellaration.geometry import surface_utils_desc

from coilstellaration import coilset_utils, data_utils, metrics_utils_v2

achieved_metrics_per_instance = []
for eval_data in predictions:
    assert eval_data.predicted_coilset is not None
    with warnings.catch_warnings(action="ignore"):
        achieved_metrics = metrics_utils_v2.evaluate_coilset_metrics_from_boundary(
            boundary=surface_utils_desc.to_desc_fourier_rz_toroidal_surface(
                eval_data.boundary
            ),
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
from coilstellaration.benchmark import scoring
from coilstellaration.benchmark import types as benchmark_types

settings = benchmark_types.ScoringSettings()  # all defaults
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

`benchmark.summaries["mean"]` under the default `ScoringSettings` is your headline number. `examples/run_benchmark_scoring.py` additionally writes a 2x3 PNG of per-metric soft-score histograms to `paths.OUTPUTS_PATH`.

### What to submit

Two artifacts.

**1. Per-instance CSV.** This is what `examples/run_benchmark_scoring.py` writes to `paths.OUTPUTS_PATH / f"scores_{MODEL_CHECKPOINT_ID}.csv"`:

| Column                        | Meaning                                                              |
| ----------------------------- | -------------------------------------------------------------------- |
| `boundary_id`                 | Identifier of the eval row.                                          |
| `score`                       | Per-instance scalar $S_j$ under the default geometric-mean aggregation. |
| `strictly_feasible`           | `True` iff $v_i = 0$ for every metric on this instance.              |
| `violation/<field>` (×4)      | Per-metric normalized violation $v_i$.                               |
| `soft_score/<field>` (×4)     | Per-metric soft score $s_i$ under the default exponential map.       |

This file is enough for the leaderboard to recompute every aggregation and reproduce the summary statistics — the per-instance breakdown is what makes diagnostics like `p10` and `strict_feasibility` reproducible.

**2. Headline number.** `summaries["mean"]` from `score_benchmark` under default `ScoringSettings`:
- `soft_score = "exponential"`
- `metric_aggregation = "geometric_mean"`
- `tolerances` as listed in the [Soft-score map](#2-soft-score-map) table
- `target_floor = 1e-6`
- `instance_aggregations = ["mean", "median", "p10", "feasibility_rate_0p9", "strict_feasibility"]`

Report `fixed_shape` and `variable_shape` separately. If you also want to claim a number on a non-`tight` stratum, label it explicitly (e.g. `fixed_shape / loose / mean = 0.84`); the leaderboard only ranks the `tight` stratum by default.

This README intentionally does not specify a submission URL or upload mechanism — that infrastructure lives outside this repo. Reach out via the [issue tracker](https://github.com/proximafusion/coilstellaration/issues) if you need a current submission link.

## Datasets

The benchmark pulls from two public Hugging Face datasets, joined at load time by `data_utils.load_dataframes` (`src/coilstellaration/data_utils.py:148`).

- **[`proxima-fusion/constellaration`](https://huggingface.co/datasets/proxima-fusion/constellaration)** — plasma boundaries, VMECPP equilibria, DESC metrics. Source of `boundary` and the `desc_metrics/...` columns. The strata tertiles are computed from the column `desc_metrics/normalized_field_error/mean` in this dataset.
- **[`proxima-fusion/coilstellaration`](https://huggingface.co/datasets/proxima-fusion/coilstellaration)** — coilsets, requirements, and the benchmark eval split. Joined to the constellaration dataset via `desc_coilset_id`, `requirements_id`, and `constellaration_boundary_id`.

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
