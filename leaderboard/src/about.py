# Markdown tables and BibTeX entries cannot be wrapped without changing how they render.
# ruff: noqa: E501

from coilstellaration.data_utils import COILSTELLARATION_REPO_REVISION
from leaderboard.src.backend.config import BENCH_VERSION

# Your leaderboard name
TITLE = '<h1 id="space-title">Coilstellaration Benchmark Leaderboard</h1>'

PAPER_URL = "https://arxiv.org/abs/2611.12689"
CONSTELLARATION_PAPER_URL = "https://arxiv.org/abs/2506.19583"
CODE_URL = "https://github.com/proximafusion/coilstellaration"
LEADERBOARD_CODE_URL = f"{CODE_URL}/tree/main/leaderboard"
DATASET_URL = "https://huggingface.co/datasets/proxima-fusion/coilstellaration"
CONSTELLARATION_DATASET_URL = (
    "https://huggingface.co/datasets/proxima-fusion/constellaration"
)
REQUESTS_URL = (
    "https://huggingface.co/datasets/proxima-fusion/coilstellaration-bench-requests"
)
RESULTS_URL = (
    "https://huggingface.co/datasets/proxima-fusion/coilstellaration-bench-results"
)
EVALUATOR_URL = (
    "https://huggingface.co/spaces/proxima-fusion/coilstellaration-bench-evaluator"
)

INTRODUCTION_TEXT = f"""
Benchmark for ML models that predict a stellarator **coilset** from a plasma boundary
and a set of engineering requirements. Pick a **track** and a **stratum** below;
rows are ranked by strict feasibility, then mean. See the **About** tab for the
task, the ONNX contract, scoring, links and how to cite.
[Paper (arXiv:2611.12689)]({PAPER_URL})
"""

ABOUT_TEXT = f"""
## Links
- 📄 **Paper:** [CoilStellaration: A Dataset and Benchmark for Engineering-Aware
  Stellarator Coilset Generation (arXiv:2611.12689)]({PAPER_URL})
- 📄 **Upstream paper:** [ConStellaration: A dataset of QI-like stellarator plasma
  boundaries and optimization benchmarks (arXiv:2506.19583)]({CONSTELLARATION_PAPER_URL})
- 💻 **Code:** [proximafusion/coilstellaration]({CODE_URL}) (this Space's source
  lives in [`leaderboard/`]({LEADERBOARD_CODE_URL}))
- 🗂️ **Datasets:** [proxima-fusion/coilstellaration]({DATASET_URL}) (coilsets,
  requirements, train/eval splits) and
  [proxima-fusion/constellaration]({CONSTELLARATION_DATASET_URL}) (plasma boundaries
  and equilibria)
- ⚙️ **Evaluation:** submissions are queued in [`coilstellaration-bench-requests`]({REQUESTS_URL}),
  scored by the [evaluator Space]({EVALUATOR_URL}) and published to
  [`coilstellaration-bench-results`]({RESULTS_URL})

## Task
A submission is an ONNX model

`f: (plasma boundary, requirement metrics) → coilset`

The boundary is a `SurfaceRZFourier`. The requirements are four normalized targets
plus the number of unique coils. The coilset is in DESC's `FourierXYZCoil`
representation and stores only the unique coils, before symmetry is applied.

For each benchmark instance, the harness feeds your model the raw boundary Fourier
coefficients and requirements, reconstructs a coilset from the outputs, re-evaluates
the **achieved** metrics on the instance's DESC equilibrium with the
`coilstellaration` physics pipeline, and scores them against the instance's
requirements.

## Tracks
| Track | Field periods (NFP) | Coils per half period | Notes |
|---|---|---|---|
| `fixed_shape` | 3 | 5 | Same coilset shape on every row; the natural starting point. |
| `variable_shape` | 2–5 | 4–8 | NFP and coil count vary per row; the model must generalize over them. |

The tracks are scored independently. A model declares which tracks it supports in its
ONNX metadata (see below).

## Strata
Each eval row belongs to one stratum. The strata are tertiles of the mean normalized
field error that the data-generation pipeline achieved on that row:

- `tight`: the most demanding requirements.
- `medium`
- `loose`: the most permissive requirements.

Every model is scored on all three strata. **`All`**, the pool of all three strata,
is the headline view (as in the paper). The other options in the stratum selector
show a single stratum.

## ONNX contract
All tensors are **raw**: no scaling or normalization. Your model owns any internal
encoding.

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

`n_unique_coils` must equal `n_coils_per_half_period`. A coilset with a different coil
count, or with NaN values, scores 0 on that instance. On `fixed_shape` this is
always 5. On `variable_shape`, declare the coil axis dynamic. You choose the Fourier
order `N`. The number of field periods and stellarator symmetry are taken from the
instance, not from the model.

**Metadata.** Set the ONNX `metadata_props` key `supported_tracks` to exactly
`fixed_shape` or `fixed_shape,variable_shape`. A model without it is rejected.

**Size.** `model.onnx` must be at most 50 MiB.

**Templates.** [`make_example_onnx.py`]({LEADERBOARD_CODE_URL}/scripts/make_example_onnx.py)
builds a minimal, contract-conforming model (it scores about 0).
[`export_coilstellaration_onnx.py`]({LEADERBOARD_CODE_URL}/scripts/export_coilstellaration_onnx.py)
exports the paper's baseline models. [`examples/README.md`]({LEADERBOARD_CODE_URL}/examples/README.md)
has the same contract with a local validation snippet.

## Scoring
For each instance, each of the four metrics gets a normalized violation, which is
mapped to a soft score in `[0, 1]`. The four soft scores are combined with a
geometric mean into the instance score `S`, so one hard violation sets `S` to 0. The
columns aggregate `S` over the selected track and stratum:

| Column | Meaning |
|---|---|
| `strict` | Fraction of instances that meet every requirement (violation ≤ 1e-3). **Primary ranking key.** |
| `strict SE` | Standard error of `strict`, `sqrt(strict × (1 − strict) / n)`. |
| `mean` | Mean of `S`. **Secondary ranking key.** |
| `feas@0.9` | Fraction of instances with `S ≥ 0.9` (soft-feasible). Hidden by default. |
| `median` | Median of `S`. Hidden by default. |
| `p10` | 10th percentile of `S`, showing how the worst instances do. Hidden by default. |
| `n` | Number of instances scored in the selected track and stratum. |

Each model is run once per instance. The full scoring definition
is in the [paper]({PAPER_URL}) and the
[code README]({CODE_URL}#scoring-soft-feasibility).
"""

# Newest first. Add an entry whenever `BENCH_VERSION` is bumped.
CHANGELOG_TEXT = f"""
## Changelog
The **Version** column shows the benchmark version a result was produced with.
Results with different versions are not directly comparable. Older results
without a version are blank.

- **{BENCH_VERSION}**: First versioned release. Every model is scored once per
  instance on the full eval split of both declared tracks, over all strata, on
  dataset revision `{COILSTELLARATION_REPO_REVISION[:7]}`.
"""

EVALUATION_QUEUE_TEXT = """
## Submitting a model
1. Push a public HF model repo containing a `model.onnx` that follows the ONNX
   contract in the **About** tab, including the `supported_tracks` metadata.
2. Optionally, check the repo locally first:
   `uv run python -m leaderboard.scripts.check_submission <org/model>`.
3. Enter the repo id (`org/model`), revision and your name/contact, and confirm
   that the model was not exposed to the eval split during training, tuning or
   model selection.
4. On submit, we check that the repo is public and that `model.onnx` has the
   required tensor names and metadata and is at most 50 MiB. The physics evaluation
   runs on the backend; watch its progress in the queue below.
"""

CITATION_TEXT = r"""@inproceedings{ewers2026coilstellaration,
  title         = {{CoilStellaration}: A Dataset and Benchmark for Engineering-Aware Stellarator Coilset Generation},
  author        = {Ewers, Jan-Hendrik and Cadena, Santiago A. and Foppiani, Nicol{\`o} and Merlo, Andrea},
  booktitle     = {40th Conference on Neural Information Processing Systems (NeurIPS 2026), Track on Evaluations and Datasets},
  year          = {2026},
  doi           = {10.48550/arXiv.2611.12689},
  eprint        = {2611.12689},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2611.12689}
}

@inproceedings{cadena2025constellaration,
  title         = {{ConStellaration}: A dataset of {QI}-like stellarator plasma boundaries and optimization benchmarks},
  author        = {Cadena, Santiago A. and Merlo, Andrea and Laude, Emanuel and Bauer, Alexander and Agrawal, Atul and Pascu, Maria and Savtchouk, Marija and Bonauer, Lukas and Guiraud, Enrico and Hudson, Stuart R. and Kaiser, Markus},
  booktitle     = {The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year          = {2025},
  eprint        = {2506.19583},
  archivePrefix = {arXiv},
  url           = {https://openreview.net/forum?id=NQSbGKlCpx}
}"""

CITATION_INTRO_TEXT = """
## Citation
If you use this benchmark, please cite the CoilStellaration paper and the
ConStellaration dataset it builds on:
"""
