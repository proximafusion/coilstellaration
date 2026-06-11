# Coilstellaration Benchmark Leaderboard — Design

**Date:** 2026-06-11
**Status:** Approved design, pending implementation plan

## 1. Purpose

Adapt the HuggingFace `gradio-templates/leaderboard` fork in this repo into a permanent,
public leaderboard for the **Coilstellaration benchmark**. Users submit a Hugging Face
model repo containing an ONNX coil-prediction model. The model maps a plasma boundary plus
a set of requirement metrics to a coilset. Each submitted model is run over `N` benchmark
instances, the predicted coilsets are evaluated with the `coilstellaration` package physics
pipeline, scored, aggregated, and displayed in a ranked leaderboard.

The model may be deterministic or stochastic.

## 2. Background / existing building blocks

The `coilstellaration` package (git dep `proximafusion/coilstellaration@03d1dc2`, installed
in `.venv`) already provides the full evaluation pipeline. This project orchestrates those
calls; it does not reimplement physics or scoring.

- **Dataset:** `coilstellaration.data_utils.load_benchmark_dataset(track, stratum, split, n)`
  → `list[EvalData]`. `track ∈ {fixed_shape, variable_shape}`, `stratum ∈ {loose, medium, tight}`,
  `split ∈ {train, eval}`, `n` caps the number of instances (`0` = all).
- **EvalData fields:** `boundary: constellaration SurfaceRZFourier`, `boundary_id: str`,
  `true_coilset: Coilset`, `true_metrics`, `requirement_metrics: RequirementMetrics`.
- **Model I/O types:**
  - Input boundary `SurfaceRZFourier`: `r_cos`/`z_sin` `[n_pol, n_tor]` always present;
    `r_sin`/`z_cos` present only when `is_stellarator_symmetric` is `False`. Also
    `n_field_periods: int`, `is_stellarator_symmetric: bool`.
  - Input `RequirementMetrics`: 4 floats — `min_normalized_coil_to_coil_distance`,
    `min_normalized_coil_to_plasma_distance`, `max_normalized_coil_curvature`,
    `max_normalized_field_error`. `RequirementMetrics.to_array()` returns them in that order.
  - Output `Coilset`: `coil_x_n`/`coil_y_n`/`coil_z_n` `[n_unique_coils, n_modes]`,
    `currents` `[n_unique_coils]`, `n_field_periods: int`, `is_stellarator_symmetric: bool`.
- **Eval:** `metrics_utils.evaluate_coilset_metrics_from_boundary(boundary_desc, coilset_desc)`
  → `Metrics`; `data_utils.metrics_to_requirement_metrics(metrics)` → achieved `RequirementMetrics`.
  (DESC physics; expensive — jax, surface grid M=80/N=80, coil_eval_n=200 per instance.)
- **Coilset → DESC:** `coilset_utils.coilstellaration_to_desc(coilset)` → DESC `CoilSet`.
  Boundary → DESC surface is available via the same module / constellaration geometry helpers.
- **Scoring:** `benchmark.scoring.score_eval_data(eval_data, achieved, settings)` → `InstanceScore`;
  `benchmark.scoring.score_benchmark(instances, settings)` → `BenchmarkScore` whose `summaries`
  hold `{mean, median, p10, feasibility_rate_0p9, strict_feasibility}`. The per-instance
  `metric_aggregation` (`geometric_mean`/`min`/`arithmetic_mean`) is a `ScoringSettings` knob.

**Explicitly NOT used:** `machine_learning.utils.forward_boundary_to_flat` and
`backward_flat_to_coilset`. These bake in spectral scaling / normalization that the submitted
model must own internally. The harness passes raw physical Fourier coefficients in and reads
raw coilset arrays out.

## 3. Architecture

Three decoupled pieces communicating only through two HF datasets:

```
┌─ Gradio Space (thin) ──────────┐      ┌─ HF Datasets ─┐      ┌─ Backend worker ─────────┐
│ • Display leaderboard          │◀────▶│ requests      │◀────▶│ poll queue → eval → push │
│ • Submit form + light validate │      │ results       │      │ (coilstellaration pkg)   │
│ • restart/refresh on interval  │      │               │      │ runs on your compute     │
└────────────────────────────────┘      └───────────────┘      └──────────────────────────┘
```

- The **Space** never runs physics. On submit it validates lightly and writes a request JSON
  to the `requests` dataset. It reads scored results from the `results` dataset to render the
  table. This is the template's existing pattern, repurposed.
- The **backend worker** is a standalone script (`src/backend/`) run on suitable compute
  (not the Space). It snapshots the `requests` dataset, picks up `PENDING` requests, runs the
  full evaluation over both tracks, and pushes a result JSON plus the updated request status
  back to the `results`/`requests` datasets.

## 4. Submission contract (raw ONNX, no harness scaling)

A submission is a public Hugging Face model repo containing **`model.onnx`**. No `config.json`
is required; the stochastic flag and sample count come from the submit form.

The model owns all internal scaling/normalization. The harness marshals raw physical
quantities only.

**Inputs (named tensors):**

| Name                  | Shape              | Notes |
|-----------------------|--------------------|-------|
| `boundary_r_cos`      | `[n_pol, n_tor]`   | raw Fourier coefficients |
| `boundary_z_sin`      | `[n_pol, n_tor]`   | raw |
| `boundary_r_sin`      | `[n_pol, n_tor]`   | only for non-stellarator-symmetric instances |
| `boundary_z_cos`      | `[n_pol, n_tor]`   | only for non-stellarator-symmetric instances |
| `requirement_metrics` | `[4]`              | raw, order per `RequirementMetrics.to_array()` |

**Outputs (named tensors):**

| Name        | Shape                       | Notes |
|-------------|-----------------------------|-------|
| `coil_x_n`  | `[n_unique_coils, n_modes]` | raw |
| `coil_y_n`  | `[n_unique_coils, n_modes]` | raw |
| `coil_z_n`  | `[n_unique_coils, n_modes]` | raw |
| `currents`  | `[n_unique_coils]`          | raw |

- The harness sets `n_field_periods` and `is_stellarator_symmetric` on the reconstructed
  `Coilset` from the **instance's boundary** (the coilset must match the plasma); the model
  does not emit them.
- The harness pads/truncates boundary arrays to a documented canonical `[n_pol, n_tor]`
  (the benchmark's fixed boundary spectral resolution). Models may declare ONNX **dynamic axes**
  on any dimension they prefer not to fix.
- **`fixed_shape` track:** output coilset shape `[n_unique_coils, n_modes]` is constant across
  instances. **`variable_shape` track:** output shape varies per instance, so the model must
  declare dynamic output axes and the harness reads the actual returned shape each run.

The exact canonical `n_pol`, `n_tor`, and the `fixed_shape` output dims are constants pinned in
the benchmark config module (Section 8), derived from the chosen dataset rows.

## 5. Evaluation harness (`src/backend/eval.py`, new)

For one submitted model and one track, over instances from
`load_benchmark_dataset(track, stratum="tight", split="eval", n=N)`:

1. Marshal `eval_data.boundary` → raw input tensors; `requirement_metrics.to_array()` → `[4]`.
2. `onnxruntime` `session.run(...)` → raw output tensors.
3. Reconstruct `Coilset(coil_x_n=…, coil_y_n=…, coil_z_n=…, currents=…,`
   `n_field_periods=boundary.n_field_periods, is_stellarator_symmetric=boundary.is_stellarator_symmetric)`.
4. `coilstellaration_to_desc(coilset)` and boundary → DESC; `evaluate_coilset_metrics_from_boundary`
   → `Metrics` → `metrics_to_requirement_metrics` → achieved `RequirementMetrics`.
5. `score_eval_data(eval_data, achieved, settings)` → `InstanceScore`.
6. `score_benchmark(instances, settings)` → `BenchmarkScore`.

Run steps 1–6 for **both** tracks (`fixed_shape`, `variable_shape`). The result record carries
both tracks' summaries plus an overall headline.

**Stochastic models:** single pass now (one `session.run` per instance, fixed seed if the
graph exposes one). The per-instance loop is structured so a future `k_samples` enhancement can
run K passes and keep the **best-scoring** coilset per instance with no change to scoring.

**Failure handling:**
- Per-instance failure (wrong output shape, NaN/inf, DESC physics error) → that instance scores
  `0.0` with a recorded reason; evaluation continues.
- Whole-model failure (ONNX won't load, I/O contract mismatch, all instances error) → the
  request is marked `FAILED` with the error message; nothing is written to `results`.

## 6. Result schema (written to `results` dataset)

One JSON per submission, e.g. `results/<org>/<model>_<revision>.json`:

```json
{
  "model": "org/model",
  "revision": "…",
  "submitter": "…",
  "submitted_time": "2026-06-11T…Z",
  "stochastic": false,
  "k_samples": 1,
  "status": "FINISHED",
  "overall_score": 0.0,
  "tracks": {
    "fixed_shape":    {"mean": 0.0, "median": 0.0, "p10": 0.0,
                       "feasibility_rate_0p9": 0.0, "strict_feasibility": 0.0, "n": 0},
    "variable_shape": {"mean": 0.0, "median": 0.0, "p10": 0.0,
                       "feasibility_rate_0p9": 0.0, "strict_feasibility": 0.0, "n": 0}
  }
}
```

`overall_score = mean(tracks.fixed_shape.mean, tracks.variable_shape.mean)`, where each track's
`mean` is the mean over instances of the per-instance **geometric-mean** soft score.

## 7. Leaderboard display (`src/display/utils.py`, `src/about.py`, `src/leaderboard/read_evals.py`, `src/populate.py`)

Replace the LLM columns (`AutoEvalColumn`, `ModelType`, `Precision`, `WeightType`) with
coilstellaration columns:

| Column | Source | Notes |
|--------|--------|-------|
| **Model** | `model` | clickable HF link (`make_clickable_model`) |
| **Submitter** | `submitter` | |
| **Overall** | `overall_score` | **default sort**, descending |
| **fixed: mean** | `tracks.fixed_shape.mean` | per-track mean geometric-mean score |
| **fixed: feas@0.9** | `tracks.fixed_shape.feasibility_rate_0p9` | |
| **fixed: strict** | `tracks.fixed_shape.strict_feasibility` | |
| **var: mean** | `tracks.variable_shape.mean` | |
| **var: feas@0.9** | `tracks.variable_shape.feasibility_rate_0p9` | |
| **var: strict** | `tracks.variable_shape.strict_feasibility` | |
| **Stochastic (K)** | `stochastic`, `k_samples` | |
| **Submitted** | `submitted_time` | |

- All three headline metrics (mean geometric-mean, `feasibility_rate_0p9`, `strict_feasibility`)
  are shown per track; `median`/`p10` available as hideable columns.
- Default ranking is by **Overall**. Per-track columns are sortable.
- Submit tab keeps the Finished / Running / Pending queue accordions, fed from request statuses.

The `Tasks` enum and LLM-specific text in `src/about.py` are rewritten for the two tracks and the
benchmark description; the About tab documents the ONNX contract and reproduction steps.

## 8. Configuration & "permanent" hosting

- A single benchmark-config module (e.g. `src/about.py` constants or `src/backend/config.py`)
  pins: `TRACKS = ["fixed_shape", "variable_shape"]`, `STRATUM = "tight"`, `SPLIT = "eval"`,
  `N` (small for dev, `0`/full for prod — one-line change), the canonical boundary
  `[n_pol, n_tor]` and `fixed_shape` output dims, and a frozen `ScoringSettings`
  (soft-score map, tolerances, `metric_aggregation="geometric_mean"`) that defines the benchmark.
- `src/envs.py`: set `OWNER` to the Proxima Fusion HF org; create the `requests` and `results`
  datasets there (replacing `demo-leaderboard-backend`).
- "Permanent": the Space's existing 30-minute `restart_space` refresh re-pulls results; results
  persist in the HF `results` dataset independently of the Space lifecycle.

## 9. Submit form & validation (`app.py`, `src/submission/submit.py`, `src/submission/check_validity.py`)

**Form fields:** HF model repo, revision (default `main`), submitter/contact, `stochastic?`
checkbox, `K` samples (default 1, only meaningful if stochastic).

**Validation (light + open ONNX graph, no inference):**
1. Repo + revision exist and are public.
2. Repo contains `model.onnx`.
3. Download and open the ONNX graph with `onnxruntime`; confirm input/output **tensor names and
   shapes** match the contract in Section 4 (dynamic axes permitted).
4. No duplicate of an already-submitted `model@revision`.

On pass, write a `PENDING` request JSON to the `requests` dataset (mirrors the template's
`add_new_eval`, minus the LLM-specific fields). Real inference validation is deferred to the
backend, which sets `FAILED` with a reason on error. Requires `onnxruntime` in the Space's deps.

## 10. Testing

- **Unit:** boundary/coilset marshalling round-trips (raw tensors ↔ `SurfaceRZFourier` /
  `Coilset`); contract-validation accepts a well-formed ONNX repo and rejects bad tensor
  names/shapes/missing file.
- **Integration:** export the package's reference model (`CoilPredictor`) to ONNX, then run the
  harness end-to-end on `N=2` per track and assert a finite, in-range `BenchmarkScore` and a
  well-formed result record.
- Scoring itself is covered by the package's `benchmark/scoring_test.py` and is treated as trusted.

## 11. Out of scope (YAGNI)

- K-sample stochastic evaluation (designed for, not built now).
- `loose`/`medium` strata and `train` split.
- Combining/normalizing scores across strata.
- Auth on submissions beyond HF public-repo checks.
- Embedding encode/decode into the ONNX graph (explicitly rejected — model owns scaling).

## 12. New / changed files (anticipated)

- New: `src/backend/__init__.py`, `src/backend/config.py`, `src/backend/eval.py`,
  `src/backend/runner.py` (queue poll + push), `src/backend/marshal.py` (raw tensor ↔ types).
- Changed: `src/about.py` (tracks, text), `src/display/utils.py` (columns),
  `src/leaderboard/read_evals.py` + `src/populate.py` (parse new result schema),
  `src/submission/submit.py` + `check_validity.py` (ONNX contract validation), `app.py`
  (form fields, tab labels, columns), `src/envs.py` (org/datasets), `requirements.txt` /
  `pyproject.toml` (`onnxruntime`).
- Tests under `tests/`.
