# Coilstellaration Benchmark Leaderboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn this Gradio-leaderboard fork into a permanent leaderboard for the Coilstellaration benchmark: users submit a HuggingFace repo containing an ONNX coil-prediction model; a backend worker runs it over both benchmark tracks and the leaderboard displays the scores.

**Architecture:** A thin Gradio Space (display + submit + light validation) talks only to two HF datasets (`requests`, `results`). A standalone backend worker polls `requests`, runs each ONNX model through the `coilstellaration` physics + scoring pipeline, and pushes scored results. The Space never runs physics.

**Tech Stack:** Python 3.12, Gradio + `gradio_leaderboard`, `onnxruntime`, the `coilstellaration` package (eval + scoring), `constellaration` (geometry), `huggingface_hub`, `datasets`, `pytest`.

**Design refinement vs spec §4:** The harness passes each instance's **native** boundary array shapes and reads the model's **native** output shapes (models own their shape expectations via dynamic axes). No pinned canonical boundary/coilset dimensions are needed, so contract validation checks tensor **names + rank**, not exact sizes.

---

## File structure

New (`src/backend/` — all physics/ONNX work lives here, imported by the worker and by submission validation):
- `src/backend/__init__.py` — empty package marker.
- `src/backend/config.py` — benchmark constants: tracks, stratum, split, N, `ScoringSettings`, tensor-name contract.
- `src/backend/marshal.py` — pure conversions: `EvalData` boundary/requirements → ONNX input dict; ONNX output dict → `coilstellaration.types.Coilset`.
- `src/backend/onnx_model.py` — `OnnxModel` (onnxruntime wrapper) + `validate_onnx_contract` (name/rank checks).
- `src/backend/eval.py` — `evaluate_track` (one track → `BenchmarkScore`) and `evaluate_model` (both tracks → result record dict).
- `src/backend/runner.py` — queue poll → download → eval → push results/status; CLI entry point.

Modified:
- `src/about.py` — tracks + leaderboard prose; remove LLM `Tasks`.
- `src/display/utils.py` — replace LLM columns with coilstellaration columns; rewrite `EvalQueueColumn`; drop `ModelType`/`Precision`/`WeightType`.
- `src/leaderboard/read_evals.py` — parse the new result schema into display rows.
- `src/populate.py` — sort by `overall`, filter on track-mean columns.
- `src/submission/check_validity.py` — add `validate_onnx_submission`.
- `src/submission/submit.py` — rewrite `add_new_eval` for the new request fields.
- `src/envs.py` — point `OWNER` at the real org/datasets.
- `app.py` — new submit form fields, column wiring, tab labels.
- `requirements.txt`, `pyproject.toml` — add `onnxruntime`; dev: `pytest`, `onnx`.

Tests: `tests/backend/{conftest.py,test_marshal.py,test_onnx_model.py,test_eval.py}`, `tests/leaderboard/test_read_evals.py`, `tests/submission/test_check_validity.py`.

---

## Phase 0: Dependencies & test scaffold

### Task 0: Add deps and pytest config

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`, `tests/backend/__init__.py`, `tests/leaderboard/__init__.py`, `tests/submission/__init__.py`

- [ ] **Step 1: Add runtime + dev deps**

In `requirements.txt` add a line `onnxruntime` (after `numpy`).

In `pyproject.toml`, add `"onnxruntime"` to `[project].dependencies`, and add to `[dependency-groups].dev`: `"pytest"`, `"onnx"`. Add a pytest section at the end:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: end-to-end tests that hit the network and run physics"]
```

- [ ] **Step 2: Install**

Run: `uv sync` (or `.venv/bin/pip install onnxruntime onnx pytest`)
Expected: success; `.venv/bin/python -c "import onnxruntime, onnx, pytest"` exits 0.

- [ ] **Step 3: Create empty test package files**

Create the four `__init__.py` files listed above, each empty.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt pyproject.toml tests/
git commit -m "chore: add onnxruntime + pytest scaffolding"
```

---

## Phase 1: Benchmark config

### Task 1: `src/backend/config.py`

**Files:**
- Create: `src/backend/__init__.py` (empty)
- Create: `src/backend/config.py`

- [ ] **Step 1: Write config module**

```python
"""Static configuration that defines the Coilstellaration benchmark.

The default `ScoringSettings()` from the package *is* the benchmark scoring
(exponential soft-score, geometric-mean per-instance aggregation, and the five
instance summaries incl. `mean`, the headline ranking metric).
"""
import os

from coilstellaration.benchmark import types as benchmark_types

# Both tracks are evaluated for every submission, at the `tight` stratum / `eval` split.
TRACKS: list[str] = ["fixed_shape", "variable_shape"]
STRATUM: str = "tight"
SPLIT: str = "eval"

# Number of instances per track. 0 = full split (production). Override for dev via env.
N_INSTANCES: int = int(os.environ.get("COILBENCH_N", "0"))

# Frozen scoring definition for the benchmark.
SCORING_SETTINGS: benchmark_types.ScoringSettings = benchmark_types.ScoringSettings()

# --- ONNX I/O contract (tensor names; shapes are native/dynamic) ---
REQUIRED_INPUT_NAMES: tuple[str, ...] = (
    "boundary_r_cos",
    "boundary_z_sin",
    "requirement_metrics",
)
# Present only for non-stellarator-symmetric instances.
OPTIONAL_INPUT_NAMES: tuple[str, ...] = ("boundary_r_sin", "boundary_z_cos")
OUTPUT_NAMES: tuple[str, ...] = ("coil_x_n", "coil_y_n", "coil_z_n", "currents")
```

- [ ] **Step 2: Verify it imports**

Run: `.venv/bin/python -c "from src.backend import config; print(config.TRACKS, config.SCORING_SETTINGS.metric_aggregation)"`
Expected: `['fixed_shape', 'variable_shape'] geometric_mean`

- [ ] **Step 3: Commit**

```bash
git add src/backend/__init__.py src/backend/config.py
git commit -m "feat: add benchmark config module"
```

---

## Phase 2: Marshalling (pure functions, TDD)

### Task 2: `src/backend/marshal.py`

**Files:**
- Create: `src/backend/marshal.py`
- Test: `tests/backend/test_marshal.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/backend/test_marshal.py
import numpy as np
import pytest

from constellaration.geometry import surface_rz_fourier
from coilstellaration import types

from src.backend import marshal


def _symmetric_boundary():
    r_cos = np.array([[0.0, 1.0, 0.1], [0.0, 0.05, 0.0]], dtype=float)
    z_sin = np.array([[0.0, 0.0, 0.2], [0.0, 0.03, 0.0]], dtype=float)
    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos, z_sin=z_sin, n_field_periods=3, is_stellarator_symmetric=True
    )


def _requirements():
    return types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=0.1,
        min_normalized_coil_to_plasma_distance=0.2,
        max_normalized_coil_curvature=1.5,
        max_normalized_field_error=0.01,
    )


def test_inputs_contain_required_names_and_shapes():
    boundary = _symmetric_boundary()
    inputs = marshal.eval_data_to_inputs(boundary, _requirements())
    assert set(inputs) == {"boundary_r_cos", "boundary_z_sin", "requirement_metrics"}
    assert inputs["boundary_r_cos"].shape == (2, 3)
    assert inputs["requirement_metrics"].shape == (4,)
    assert inputs["requirement_metrics"].dtype == np.float32
    # order matches RequirementMetrics.to_array()
    np.testing.assert_allclose(inputs["requirement_metrics"], [0.1, 0.2, 1.5, 0.01], rtol=1e-6)


def test_asymmetric_inputs_include_r_sin_z_cos():
    r = np.zeros((2, 3))
    boundary = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r.copy(), z_sin=r.copy(), r_sin=r.copy(), z_cos=r.copy(),
        n_field_periods=1, is_stellarator_symmetric=False,
    )
    inputs = marshal.eval_data_to_inputs(boundary, _requirements())
    assert {"boundary_r_sin", "boundary_z_cos"} <= set(inputs)


def test_outputs_to_coilset_uses_boundary_symmetry_and_nfp():
    boundary = _symmetric_boundary()
    outputs = {
        "coil_x_n": np.zeros((4, 5), dtype=np.float32),
        "coil_y_n": np.zeros((4, 5), dtype=np.float32),
        "coil_z_n": np.zeros((4, 5), dtype=np.float32),
        "currents": np.ones((4,), dtype=np.float32),
    }
    coilset = marshal.outputs_to_coilset(outputs, boundary)
    assert isinstance(coilset, types.Coilset)
    assert coilset.n_unique_coils == 4
    assert int(coilset.n_field_periods) == 3
    assert coilset.is_stellarator_symmetric is True
    np.testing.assert_allclose(np.asarray(coilset.currents), [1, 1, 1, 1])


def test_missing_output_key_raises():
    boundary = _symmetric_boundary()
    with pytest.raises(KeyError):
        marshal.outputs_to_coilset({"coil_x_n": np.zeros((4, 5))}, boundary)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/backend/test_marshal.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backend.marshal'`

- [ ] **Step 3: Implement `marshal.py`**

```python
"""Pure conversions between package types and raw ONNX tensors.

No scaling/normalization: raw Fourier coefficients in, raw coilset arrays out.
The model owns all internal encoding (spec §4).
"""
import numpy as np

from constellaration.geometry import surface_rz_fourier
from coilstellaration import types

from src.backend import config


def eval_data_to_inputs(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    requirement_metrics: types.RequirementMetrics,
) -> dict[str, np.ndarray]:
    """Build the raw ONNX feed dict for one instance."""
    inputs: dict[str, np.ndarray] = {
        "boundary_r_cos": np.asarray(boundary.r_cos, dtype=np.float32),
        "boundary_z_sin": np.asarray(boundary.z_sin, dtype=np.float32),
        "requirement_metrics": np.asarray(
            requirement_metrics.to_array(), dtype=np.float32
        ),
    }
    if not boundary.is_stellarator_symmetric:
        inputs["boundary_r_sin"] = np.asarray(boundary.r_sin, dtype=np.float32)
        inputs["boundary_z_cos"] = np.asarray(boundary.z_cos, dtype=np.float32)
    return inputs


def outputs_to_coilset(
    outputs: dict[str, np.ndarray],
    boundary: surface_rz_fourier.SurfaceRZFourier,
) -> types.Coilset:
    """Reconstruct a Coilset from raw model outputs.

    `n_field_periods` and `is_stellarator_symmetric` come from the instance
    boundary (the coilset must match the plasma), not the model.
    """
    for name in config.OUTPUT_NAMES:
        if name not in outputs:
            raise KeyError(f"Model output missing required tensor: {name!r}")
    return types.Coilset(
        coil_x_n=np.asarray(outputs["coil_x_n"], dtype=float),
        coil_y_n=np.asarray(outputs["coil_y_n"], dtype=float),
        coil_z_n=np.asarray(outputs["coil_z_n"], dtype=float),
        currents=np.asarray(outputs["currents"], dtype=float).reshape(-1),
        n_field_periods=int(boundary.n_field_periods),
        is_stellarator_symmetric=bool(boundary.is_stellarator_symmetric),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/backend/test_marshal.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backend/marshal.py tests/backend/test_marshal.py
git commit -m "feat: raw boundary/coilset ONNX marshalling"
```

---

## Phase 3: ONNX model wrapper + contract validation (TDD)

### Task 3: test fixture builder

**Files:**
- Create: `tests/backend/conftest.py`

- [ ] **Step 1: Write the dummy-ONNX fixture**

```python
# tests/backend/conftest.py
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


def _const(name, shape):
    flat = np.zeros(int(np.prod(shape)), dtype=np.float32)
    val = helper.make_tensor(name + "_v", TensorProto.FLOAT, list(shape), flat)
    return helper.make_node("Constant", [], [name], value=val)


def build_dummy_onnx(
    path, n_pol=2, n_tor=3, n_coils=4, n_modes=5,
    input_names=("boundary_r_cos", "boundary_z_sin", "requirement_metrics"),
    output_names=("coil_x_n", "coil_y_n", "coil_z_n", "currents"),
):
    """A contract-conforming ONNX graph whose outputs are zeros of the right shape.

    Inputs are declared (so onnxruntime accepts the feeds) but unused.
    """
    in_shapes = {
        "boundary_r_cos": [n_pol, n_tor],
        "boundary_z_sin": [n_pol, n_tor],
        "boundary_r_sin": [n_pol, n_tor],
        "boundary_z_cos": [n_pol, n_tor],
        "requirement_metrics": [4],
    }
    out_shapes = {
        "coil_x_n": [n_coils, n_modes], "coil_y_n": [n_coils, n_modes],
        "coil_z_n": [n_coils, n_modes], "currents": [n_coils],
    }
    inputs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, in_shapes[n])
        for n in input_names
    ]
    nodes = [_const(n, out_shapes[n]) for n in output_names]
    outputs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, out_shapes[n])
        for n in output_names
    ]
    graph = helper.make_graph(nodes, "dummy", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def dummy_onnx_path(tmp_path):
    return build_dummy_onnx(tmp_path / "model.onnx")
```

- [ ] **Step 2: Commit**

```bash
git add tests/backend/conftest.py
git commit -m "test: dummy ONNX graph fixture"
```

### Task 4: `src/backend/onnx_model.py`

**Files:**
- Create: `src/backend/onnx_model.py`
- Test: `tests/backend/test_onnx_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/backend/test_onnx_model.py
import numpy as np

from src.backend import onnx_model
from tests.backend.conftest import build_dummy_onnx


def test_validate_accepts_conforming_model(dummy_onnx_path):
    errors = onnx_model.validate_onnx_contract(dummy_onnx_path)
    assert errors == []


def test_validate_rejects_missing_required_input(tmp_path):
    path = build_dummy_onnx(
        tmp_path / "bad.onnx",
        input_names=("boundary_r_cos", "boundary_z_sin"),  # no requirement_metrics
    )
    errors = onnx_model.validate_onnx_contract(path)
    assert any("requirement_metrics" in e for e in errors)


def test_validate_rejects_missing_output(tmp_path):
    path = build_dummy_onnx(
        tmp_path / "bad.onnx", output_names=("coil_x_n", "coil_y_n", "coil_z_n")
    )
    errors = onnx_model.validate_onnx_contract(path)
    assert any("currents" in e for e in errors)


def test_run_returns_named_outputs_with_shapes(dummy_onnx_path):
    model = onnx_model.OnnxModel(dummy_onnx_path)
    out = model.run({
        "boundary_r_cos": np.zeros((2, 3), np.float32),
        "boundary_z_sin": np.zeros((2, 3), np.float32),
        "requirement_metrics": np.zeros((4,), np.float32),
    })
    assert set(out) == {"coil_x_n", "coil_y_n", "coil_z_n", "currents"}
    assert out["coil_x_n"].shape == (4, 5)
    assert out["currents"].shape == (4,)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/backend/test_onnx_model.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backend.onnx_model'`

- [ ] **Step 3: Implement `onnx_model.py`**

```python
"""onnxruntime wrapper + contract validation for submitted models."""
import numpy as np
import onnxruntime as ort

from src.backend import config


class OnnxModel:
    """Thin wrapper around an onnxruntime InferenceSession."""

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Feed only the tensors the graph declares; return all named outputs."""
        feed = {k: v for k, v in inputs.items() if k in self.input_names}
        results = self.session.run(self.output_names, feed)
        return dict(zip(self.output_names, results))


def validate_onnx_contract(model_path: str) -> list[str]:
    """Open the graph and check input/output tensor names against the contract.

    Shapes are intentionally not checked (native/dynamic; the model owns them).
    Returns a list of human-readable error strings; empty means valid.
    """
    errors: list[str] = []
    try:
        session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # noqa: BLE001 - surface any load failure to the user
        return [f"Could not load ONNX model: {exc}"]

    input_names = {i.name for i in session.get_inputs()}
    output_names = {o.name for o in session.get_outputs()}

    for name in config.REQUIRED_INPUT_NAMES:
        if name not in input_names:
            errors.append(f"Missing required input tensor: {name!r}")
    for name in config.OUTPUT_NAMES:
        if name not in output_names:
            errors.append(f"Missing required output tensor: {name!r}")
    return errors
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/backend/test_onnx_model.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backend/onnx_model.py tests/backend/test_onnx_model.py
git commit -m "feat: ONNX model wrapper and contract validation"
```

---

## Phase 4: Evaluation harness (TDD with mocked physics)

### Task 5: `src/backend/eval.py`

**Files:**
- Create: `src/backend/eval.py`
- Test: `tests/backend/test_eval.py`

The physics call (`metrics_utils_v2.evaluate_coilset_metrics_from_boundary`) and `data_utils.metrics_to_requirement_metrics` are trusted package code; unit tests monkeypatch them at the `eval` module boundary so they run fast and hermetically. A real end-to-end run is the manual smoke step in Task 9.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backend/test_eval.py
import numpy as np

from constellaration.geometry import surface_rz_fourier
from coilstellaration import types

from src.backend import eval as beval


class FakeModel:
    def run(self, inputs):
        return {
            "coil_x_n": np.zeros((4, 5), np.float32),
            "coil_y_n": np.zeros((4, 5), np.float32),
            "coil_z_n": np.zeros((4, 5), np.float32),
            "currents": np.ones((4,), np.float32),
        }


def _eval_data(boundary_id="b0"):
    r = np.zeros((2, 3))
    boundary = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r.copy(), z_sin=r.copy(), n_field_periods=3, is_stellarator_symmetric=True
    )
    coilset = types.Coilset(
        coil_x_n=np.zeros((4, 5)), coil_y_n=np.zeros((4, 5)), coil_z_n=np.zeros((4, 5)),
        currents=np.ones((4,)), n_field_periods=3, is_stellarator_symmetric=True,
    )
    req = types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=0.1,
        min_normalized_coil_to_plasma_distance=0.2,
        max_normalized_coil_curvature=2.0,
        max_normalized_field_error=0.1,
    )
    return types.EvalData(
        boundary=boundary, boundary_id=boundary_id, true_coilset=coilset,
        true_metrics="metrics-id", requirement_metrics=req,
    )


def _patch_physics(monkeypatch, achieved):
    # achieved == target -> all violations 0 -> per-instance score 1.0
    monkeypatch.setattr(beval, "_evaluate_achieved", lambda boundary, coilset: achieved)


def test_evaluate_track_perfect_scores(monkeypatch):
    data = [_eval_data("b0"), _eval_data("b1")]
    _patch_physics(monkeypatch, data[0].requirement_metrics)
    score = beval.evaluate_track(FakeModel(), data)
    assert score.summaries["mean"] == 1.0
    assert score.summaries["strict_feasibility"] == 1.0
    assert len(score.per_instance) == 2


def test_failed_instance_scores_zero_and_continues(monkeypatch):
    data = [_eval_data("b0"), _eval_data("b1")]

    def flaky(boundary, coilset):
        if not flaky.called:
            flaky.called = True
            raise RuntimeError("physics blew up")
        return data[0].requirement_metrics
    flaky.called = False
    monkeypatch.setattr(beval, "_evaluate_achieved", flaky)

    score = beval.evaluate_track(FakeModel(), data)
    aggregates = sorted(s.aggregate for s in score.per_instance)
    assert aggregates[0] == 0.0       # the failed one
    assert aggregates[1] == 1.0       # the good one


def test_evaluate_model_builds_record(monkeypatch):
    data = [_eval_data("b0")]
    _patch_physics(monkeypatch, data[0].requirement_metrics)
    monkeypatch.setattr(beval, "_load_track", lambda track: data)

    record = beval.evaluate_model(
        model=FakeModel(), model_name="org/m", revision="abc",
        submitter="me", stochastic=False, k_samples=1,
        submitted_time="2026-06-11T00:00:00Z",
    )
    assert record["status"] == "FINISHED"
    assert record["overall_score"] == 1.0
    assert set(record["tracks"]) == {"fixed_shape", "variable_shape"}
    assert record["tracks"]["fixed_shape"]["mean"] == 1.0
    assert record["tracks"]["fixed_shape"]["n"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/backend/test_eval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backend.eval'`

- [ ] **Step 3: Implement `eval.py`**

```python
"""Run an ONNX model through the coilstellaration eval + scoring pipeline."""
import numpy as np

from coilstellaration import coilset_utils, data_utils, metrics_utils_v2
from coilstellaration import types
from coilstellaration.benchmark import scoring
from coilstellaration.benchmark import types as benchmark_types
from constellaration.geometry import surface_rz_fourier
from constellaration.geometry import surface_utils_desc

from src.backend import config, marshal

SUMMARY_KEYS = ("mean", "median", "p10", "feasibility_rate_0p9", "strict_feasibility")


def _evaluate_achieved(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    coilset: types.Coilset,
) -> types.RequirementMetrics:
    """Predicted coilset -> achieved RequirementMetrics via DESC physics.

    Isolated so tests can monkeypatch the expensive physics.
    """
    desc_surface = surface_utils_desc.to_desc_fourier_rz_toroidal_surface(boundary)
    desc_coilset = coilset_utils.coilstellaration_to_desc(coilset)
    metrics = metrics_utils_v2.evaluate_coilset_metrics_from_boundary(
        boundary=desc_surface, coilset=desc_coilset
    )
    return data_utils.metrics_to_requirement_metrics(metrics)


def _load_track(track: str) -> list[types.EvalData]:
    """Load benchmark instances for a track. Isolated for monkeypatching."""
    return data_utils.load_benchmark_dataset(
        track=track, stratum=config.STRATUM, split=config.SPLIT, n=config.N_INSTANCES
    )


def _failed_instance(boundary_id: str) -> benchmark_types.InstanceScore:
    zeros = {f: 0.0 for f in benchmark_types.REQUIREMENT_METRIC_DIRECTIONS}
    return benchmark_types.InstanceScore(
        boundary_id=boundary_id,
        metric_scores=benchmark_types.MetricScores(
            violations=zeros, soft_scores=zeros, strictly_feasible=False
        ),
        aggregate=0.0,
    )


def evaluate_track(
    model,
    eval_datas: list[types.EvalData],
    settings: benchmark_types.ScoringSettings = config.SCORING_SETTINGS,
) -> benchmark_types.BenchmarkScore:
    """Score one model on one track's instances. Single forward pass per instance."""
    instances: list[benchmark_types.InstanceScore] = []
    for data in eval_datas:
        try:
            inputs = marshal.eval_data_to_inputs(data.boundary, data.requirement_metrics)
            outputs = model.run(inputs)
            coilset = marshal.outputs_to_coilset(outputs, data.boundary)
            achieved = _evaluate_achieved(data.boundary, coilset)
            if not _is_finite(achieved):
                raise ValueError("achieved metrics contain NaN/inf")
            instances.append(scoring.score_eval_data(data, achieved, settings))
        except Exception as exc:  # noqa: BLE001 - one bad instance must not abort the run
            print(f"[eval] instance {data.boundary_id} failed: {exc}")
            instances.append(_failed_instance(data.boundary_id))
    return scoring.score_benchmark(instances, settings)


def _is_finite(achieved: types.RequirementMetrics) -> bool:
    return bool(np.all(np.isfinite(np.asarray(achieved.to_array(), dtype=float))))


def evaluate_model(
    model,
    model_name: str,
    revision: str,
    submitter: str,
    stochastic: bool,
    k_samples: int,
    submitted_time: str,
) -> dict:
    """Evaluate a model on all tracks and assemble the result record (spec §6)."""
    tracks: dict[str, dict] = {}
    for track in config.TRACKS:
        eval_datas = _load_track(track)
        score = evaluate_track(model, eval_datas)
        summary = {k: float(score.summaries[k]) for k in SUMMARY_KEYS}
        summary["n"] = len(score.per_instance)
        tracks[track] = summary

    overall = float(np.mean([tracks[t]["mean"] for t in config.TRACKS]))
    return {
        "model": model_name,
        "revision": revision,
        "submitter": submitter,
        "submitted_time": submitted_time,
        "stochastic": stochastic,
        "k_samples": k_samples,
        "status": "FINISHED",
        "overall_score": overall,
        "tracks": tracks,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/backend/test_eval.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backend/eval.py tests/backend/test_eval.py
git commit -m "feat: evaluation harness over both tracks"
```

---

## Phase 5: Result read path & leaderboard columns

### Task 6: rewrite display columns and queue columns

**Files:**
- Modify: `src/display/utils.py`

- [ ] **Step 1: Replace columns**

Replace the entire body of `src/display/utils.py` (lines 1–111) with:

```python
from dataclasses import dataclass, make_dataclass


def fields(raw_class):
    return [v for k, v in raw_class.__dict__.items() if k[:2] != "__" and k[-2:] != "__"]


@dataclass(frozen=True)
class ColumnContent:
    name: str
    type: str
    displayed_by_default: bool
    hidden: bool = False
    never_hidden: bool = False


## Leaderboard columns
auto_eval_column_dict = []
auto_eval_column_dict.append(["model", ColumnContent, ColumnContent("Model", "markdown", True, never_hidden=True)])
auto_eval_column_dict.append(["submitter", ColumnContent, ColumnContent("Submitter", "str", True)])
auto_eval_column_dict.append(["overall", ColumnContent, ColumnContent("Overall ⬆️", "number", True)])
# fixed_shape track
auto_eval_column_dict.append(["fixed_mean", ColumnContent, ColumnContent("fixed: mean", "number", True)])
auto_eval_column_dict.append(["fixed_feas09", ColumnContent, ColumnContent("fixed: feas@0.9", "number", True)])
auto_eval_column_dict.append(["fixed_strict", ColumnContent, ColumnContent("fixed: strict", "number", True)])
auto_eval_column_dict.append(["fixed_median", ColumnContent, ColumnContent("fixed: median", "number", False)])
auto_eval_column_dict.append(["fixed_p10", ColumnContent, ColumnContent("fixed: p10", "number", False)])
# variable_shape track
auto_eval_column_dict.append(["var_mean", ColumnContent, ColumnContent("var: mean", "number", True)])
auto_eval_column_dict.append(["var_feas09", ColumnContent, ColumnContent("var: feas@0.9", "number", True)])
auto_eval_column_dict.append(["var_strict", ColumnContent, ColumnContent("var: strict", "number", True)])
auto_eval_column_dict.append(["var_median", ColumnContent, ColumnContent("var: median", "number", False)])
auto_eval_column_dict.append(["var_p10", ColumnContent, ColumnContent("var: p10", "number", False)])
# metadata
auto_eval_column_dict.append(["stochastic", ColumnContent, ColumnContent("Stochastic (K)", "str", True)])
auto_eval_column_dict.append(["submitted", ColumnContent, ColumnContent("Submitted", "str", False)])
auto_eval_column_dict.append(["revision", ColumnContent, ColumnContent("Revision", "str", False)])

AutoEvalColumn = make_dataclass("AutoEvalColumn", auto_eval_column_dict, frozen=True)


## Queue columns (submission tab)
@dataclass(frozen=True)
class EvalQueueColumn:
    model = ColumnContent("model", "markdown", True)
    revision = ColumnContent("revision", "str", True)
    submitter = ColumnContent("submitter", "str", True)
    stochastic = ColumnContent("stochastic", "str", True)
    status = ColumnContent("status", "str", True)


# Column selection
COLS = [c.name for c in fields(AutoEvalColumn) if not c.hidden]
EVAL_COLS = [c.name for c in fields(EvalQueueColumn)]
EVAL_TYPES = [c.type for c in fields(EvalQueueColumn)]

# Track-mean columns must be present for a row to display.
BENCHMARK_COLS = [
    AutoEvalColumn.fixed_mean.name,
    AutoEvalColumn.var_mean.name,
]
```

- [ ] **Step 2: Verify import**

Run: `.venv/bin/python -c "from src.display.utils import AutoEvalColumn, COLS, BENCHMARK_COLS; print(COLS); print(BENCHMARK_COLS)"`
Expected: prints the column list and `['fixed_mean', 'var_mean']`. No `ModuleNotFoundError`.

- [ ] **Step 3: Commit**

```bash
git add src/display/utils.py
git commit -m "feat: coilstellaration leaderboard columns"
```

### Task 7: rewrite result parsing (TDD)

**Files:**
- Modify: `src/leaderboard/read_evals.py` (full rewrite)
- Modify: `src/populate.py:8-23` (sort + nan filter)
- Test: `tests/leaderboard/test_read_evals.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/leaderboard/test_read_evals.py
import json

from src.display.utils import AutoEvalColumn, COLS
from src.leaderboard.read_evals import get_raw_eval_results


def _write_result(dirpath, name, overall):
    record = {
        "model": f"org/{name}", "revision": "abc", "submitter": "alice",
        "submitted_time": "2026-06-11T00:00:00Z", "stochastic": True, "k_samples": 4,
        "status": "FINISHED", "overall_score": overall,
        "tracks": {
            "fixed_shape": {"mean": overall, "median": 0.5, "p10": 0.2,
                            "feasibility_rate_0p9": 0.3, "strict_feasibility": 0.1, "n": 2},
            "variable_shape": {"mean": overall, "median": 0.4, "p10": 0.1,
                               "feasibility_rate_0p9": 0.2, "strict_feasibility": 0.0, "n": 2},
        },
    }
    sub = dirpath / "org"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"{name}.json").write_text(json.dumps(record))


def test_parses_records_into_rows(tmp_path):
    _write_result(tmp_path, "m1", 0.8)
    rows = get_raw_eval_results(str(tmp_path))
    assert len(rows) == 1
    row = rows[0]
    assert row[AutoEvalColumn.overall.name] == 0.8
    assert row[AutoEvalColumn.fixed_feas09.name] == 0.3
    assert row[AutoEvalColumn.stochastic.name] == "✓ (K=4)"
    assert "org/m1" in row[AutoEvalColumn.model.name]  # clickable markdown
    # every displayed column is present
    for col in COLS:
        assert col in row


def test_skips_unfinished(tmp_path):
    _write_result(tmp_path, "m1", 0.8)
    bad = tmp_path / "org" / "m2.json"
    bad.write_text(json.dumps({"model": "org/m2", "status": "FAILED"}))
    rows = get_raw_eval_results(str(tmp_path))
    assert [r[AutoEvalColumn.model.name] for r in rows if "m2" in r[AutoEvalColumn.model.name]] == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/leaderboard/test_read_evals.py -q`
Expected: FAIL — `get_raw_eval_results` now takes 2 args / parses old schema (TypeError or KeyError).

- [ ] **Step 3: Rewrite `read_evals.py`**

Replace the entire file with:

```python
import json
import os

from src.display.formatting import make_clickable_model
from src.display.utils import AutoEvalColumn


def _stochastic_label(record: dict) -> str:
    if record.get("stochastic"):
        return f"✓ (K={record.get('k_samples', 1)})"
    return "—"


def result_to_row(record: dict) -> dict:
    """Map one result record (spec §6) to a leaderboard dataframe row."""
    fixed = record["tracks"]["fixed_shape"]
    var = record["tracks"]["variable_shape"]
    return {
        AutoEvalColumn.model.name: make_clickable_model(record["model"]),
        AutoEvalColumn.submitter.name: record.get("submitter", ""),
        AutoEvalColumn.overall.name: record["overall_score"],
        AutoEvalColumn.fixed_mean.name: fixed["mean"],
        AutoEvalColumn.fixed_feas09.name: fixed["feasibility_rate_0p9"],
        AutoEvalColumn.fixed_strict.name: fixed["strict_feasibility"],
        AutoEvalColumn.fixed_median.name: fixed["median"],
        AutoEvalColumn.fixed_p10.name: fixed["p10"],
        AutoEvalColumn.var_mean.name: var["mean"],
        AutoEvalColumn.var_feas09.name: var["feasibility_rate_0p9"],
        AutoEvalColumn.var_strict.name: var["strict_feasibility"],
        AutoEvalColumn.var_median.name: var["median"],
        AutoEvalColumn.var_p10.name: var["p10"],
        AutoEvalColumn.stochastic.name: _stochastic_label(record),
        AutoEvalColumn.submitted.name: record.get("submitted_time", ""),
        AutoEvalColumn.revision.name: record.get("revision", ""),
    }


def get_raw_eval_results(results_path: str) -> list[dict]:
    """Walk the results dir; return one display row per FINISHED result."""
    rows: list[dict] = []
    for root, _, files in os.walk(results_path):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(root, fname)) as fp:
                record = json.load(fp)
            if record.get("status") != "FINISHED":
                continue
            try:
                rows.append(result_to_row(record))
            except KeyError as exc:
                print(f"[read_evals] skipping malformed result {fname}: {exc}")
    return rows
```

- [ ] **Step 4: Update `populate.py`**

In `src/populate.py`, replace `get_leaderboard_df` (lines 10–23) with:

```python
def get_leaderboard_df(results_path: str, requests_path: str, cols: list, benchmark_cols: list) -> pd.DataFrame:
    """Creates a dataframe from all the individual experiment results"""
    all_data_json = get_raw_eval_results(results_path)

    df = pd.DataFrame.from_records(all_data_json)
    if df.empty:
        return df
    df = df.sort_values(by=[AutoEvalColumn.overall.name], ascending=False)
    df = df[cols].round(decimals=3)

    # filter out rows missing a track mean
    df = df[has_no_nan_values(df, benchmark_cols)]
    return df
```

The `requests_path` arg is kept for call-site compatibility but unused (the result record is self-contained).

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/leaderboard/test_read_evals.py -q`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/leaderboard/read_evals.py src/populate.py tests/leaderboard/test_read_evals.py
git commit -m "feat: parse coilstellaration result schema into leaderboard rows"
```

---

## Phase 6: Submission validation & form

### Task 8: ONNX submission validation (TDD) + submit rewrite

**Files:**
- Modify: `src/submission/check_validity.py` (add function)
- Modify: `src/submission/submit.py` (rewrite `add_new_eval`)
- Test: `tests/submission/test_check_validity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/submission/test_check_validity.py
from src.submission.check_validity import validate_local_onnx
from tests.backend.conftest import build_dummy_onnx


def test_valid_model_passes(tmp_path):
    path = build_dummy_onnx(tmp_path / "model.onnx")
    ok, msg = validate_local_onnx(path)
    assert ok, msg


def test_missing_tensor_fails(tmp_path):
    path = build_dummy_onnx(tmp_path / "model.onnx", output_names=("coil_x_n", "coil_y_n", "coil_z_n"))
    ok, msg = validate_local_onnx(path)
    assert not ok
    assert "currents" in msg
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/submission/test_check_validity.py -q`
Expected: FAIL — `ImportError: cannot import name 'validate_local_onnx'`

- [ ] **Step 3: Add `validate_local_onnx` to `check_validity.py`**

Append to `src/submission/check_validity.py`:

```python
def validate_local_onnx(local_path: str) -> tuple[bool, str]:
    """Open a downloaded ONNX file and check it against the I/O contract.

    Returns (ok, message). No inference is run.
    """
    from src.backend.onnx_model import validate_onnx_contract

    errors = validate_onnx_contract(local_path)
    if errors:
        return False, "; ".join(errors)
    return True, "ONNX contract OK"
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/submission/test_check_validity.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Rewrite `add_new_eval` in `submit.py`**

Replace the whole body of `src/submission/submit.py` with:

```python
import json
import os
import tempfile
from datetime import datetime, timezone

from huggingface_hub import hf_hub_download

from src.display.formatting import styled_error, styled_message, styled_warning
from src.envs import API, EVAL_REQUESTS_PATH, QUEUE_REPO, TOKEN
from src.submission.check_validity import is_model_on_hub, validate_local_onnx

ONNX_FILENAME = "model.onnx"


def add_new_eval(model: str, revision: str, submitter: str, stochastic: bool, k_samples: int):
    if not model or "/" not in model:
        return styled_error("Please enter a model id of the form `org/model`.")
    if not submitter:
        return styled_error("Please enter a submitter name or contact.")
    revision = revision or "main"
    k_samples = int(k_samples) if stochastic else 1

    user_name, model_path = model.split("/", 1)

    # 1) Repo is public and loadable.
    on_hub, error, _ = is_model_on_hub(model_name=model, revision=revision, token=TOKEN, test_tokenizer=False)
    if not on_hub:
        return styled_error(f'Model "{model}" {error}')

    # 2) Download model.onnx and validate the I/O contract.
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = hf_hub_download(
                repo_id=model, filename=ONNX_FILENAME, revision=revision,
                local_dir=tmp, token=TOKEN,
            )
            ok, msg = validate_local_onnx(local)
    except Exception as exc:  # noqa: BLE001
        return styled_error(f"Could not download `{ONNX_FILENAME}` from the repo: {exc}")
    if not ok:
        return styled_error(f"ONNX contract check failed: {msg}")

    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    eval_entry = {
        "model": model,
        "revision": revision,
        "submitter": submitter,
        "stochastic": bool(stochastic),
        "k_samples": k_samples,
        "status": "PENDING",
        "submitted_time": current_time,
        "private": False,
    }

    out_dir = f"{EVAL_REQUESTS_PATH}/{user_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{model_path}_eval_request_{revision}.json"
    with open(out_path, "w") as f:
        f.write(json.dumps(eval_entry))

    API.upload_file(
        path_or_fileobj=out_path,
        path_in_repo=out_path.split("eval-queue/")[1],
        repo_id=QUEUE_REPO,
        repo_type="dataset",
        commit_message=f"Add {model} to eval queue",
    )
    os.remove(out_path)
    return styled_message("Your model has been submitted to the evaluation queue!")
```

- [ ] **Step 6: Verify imports (no network)**

Run: `.venv/bin/python -c "import src.submission.submit"`
Expected: exits 0.

- [ ] **Step 7: Commit**

```bash
git add src/submission/check_validity.py src/submission/submit.py tests/submission/test_check_validity.py
git commit -m "feat: ONNX-contract submission validation and request schema"
```

---

## Phase 7: App wiring, about text, envs

### Task 9: backend runner CLI

**Files:**
- Create: `src/backend/runner.py`

- [ ] **Step 1: Write the runner**

```python
"""Backend worker: poll the requests dataset, evaluate PENDING models, push results.

Run on suitable compute (NOT the Space):  python -m src.backend.runner
Requires HF_TOKEN with write access to the requests + results datasets.
"""
import json
import os
import tempfile

from huggingface_hub import hf_hub_download, snapshot_download

from src.backend import eval as beval
from src.backend.onnx_model import OnnxModel
from src.envs import API, EVAL_REQUESTS_PATH, QUEUE_REPO, RESULTS_REPO, TOKEN
from src.submission.submit import ONNX_FILENAME


def _iter_pending(requests_dir: str):
    for root, _, files in os.walk(requests_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            with open(path) as fp:
                req = json.load(fp)
            if req.get("status") == "PENDING":
                yield path, req


def _push_request(req: dict, rel_path: str):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(req, f)
        tmp = f.name
    API.upload_file(path_or_fileobj=tmp, path_in_repo=rel_path, repo_id=QUEUE_REPO,
                    repo_type="dataset", commit_message=f"Update status {req['model']} -> {req['status']}")
    os.remove(tmp)


def _push_result(record: dict):
    org, model = record["model"].split("/", 1)
    rel_path = f"{org}/{model}_{record['revision']}.json"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(record, f)
        tmp = f.name
    API.upload_file(path_or_fileobj=tmp, path_in_repo=rel_path, repo_id=RESULTS_REPO,
                    repo_type="dataset", commit_message=f"Add results for {record['model']}")
    os.remove(tmp)


def run_once():
    snapshot_download(repo_id=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH, repo_type="dataset", token=TOKEN)
    for path, req in _iter_pending(EVAL_REQUESTS_PATH):
        rel_path = path.split("eval-queue/")[1]
        req["status"] = "RUNNING"
        _push_request(req, rel_path)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                local = hf_hub_download(repo_id=req["model"], filename=ONNX_FILENAME,
                                        revision=req["revision"], local_dir=tmp, token=TOKEN)
                model = OnnxModel(local)
                record = beval.evaluate_model(
                    model=model, model_name=req["model"], revision=req["revision"],
                    submitter=req.get("submitter", ""), stochastic=req.get("stochastic", False),
                    k_samples=req.get("k_samples", 1), submitted_time=req.get("submitted_time", ""),
                )
            _push_result(record)
            req["status"] = "FINISHED"
        except Exception as exc:  # noqa: BLE001
            print(f"[runner] {req['model']} FAILED: {exc}")
            req["status"] = "FAILED"
            req["error"] = str(exc)
        _push_request(req, rel_path)


if __name__ == "__main__":
    run_once()
```

- [ ] **Step 2: Verify import**

Run: `.venv/bin/python -c "import src.backend.runner"`
Expected: exits 0.

- [ ] **Step 3: Commit**

```bash
git add src/backend/runner.py
git commit -m "feat: backend queue runner"
```

### Task 10: update `envs.py` and `about.py`

**Files:**
- Modify: `src/envs.py:9`
- Modify: `src/about.py` (full rewrite)

- [ ] **Step 1: Set the org in `envs.py`**

Change line 9 from `OWNER = "demo-leaderboard-backend"` to:

```python
OWNER = os.environ.get("COILBENCH_OWNER", "proximafusion")  # HF org owning the leaderboard/requests/results datasets
```

- [ ] **Step 2: Rewrite `about.py`**

Replace the whole file with (remove the LLM `Tasks`/`Task` — no longer imported anywhere):

```python
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
```

- [ ] **Step 3: Verify imports**

Run: `.venv/bin/python -c "import src.about, src.envs; print(src.envs.OWNER)"`
Expected: prints the org; no import error (confirms nothing still imports `Tasks`).

- [ ] **Step 4: Commit**

```bash
git add src/envs.py src/about.py
git commit -m "feat: coilstellaration about text and org config"
```

### Task 11: rewire `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Fix imports**

In `app.py`, change the `src.display.utils` import block (lines 16–26) to:

```python
from src.display.utils import (
    BENCHMARK_COLS,
    COLS,
    EVAL_COLS,
    EVAL_TYPES,
    AutoEvalColumn,
    fields,
)
```

Change the leaderboard df call (line 52) to keep both args (signature unchanged):

```python
LEADERBOARD_DF = get_leaderboard_df(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH, COLS, BENCHMARK_COLS)
```

- [ ] **Step 2: Replace `init_leaderboard` filters**

Replace the `filter_columns=[...]` list (lines 73–86) and the `search_columns`/`hide_columns` lines with:

```python
        search_columns=[AutoEvalColumn.model.name, AutoEvalColumn.submitter.name],
        hide_columns=[c.name for c in fields(AutoEvalColumn) if c.hidden],
        filter_columns=[
            ColumnFilter(AutoEvalColumn.stochastic.name, type="checkboxgroup", label="Stochastic"),
        ],
```

- [ ] **Step 3: Replace the submit form**

Replace the submit-form block (lines 147–189, from `with gr.Row():` containing the model inputs through the `submit_button.click(...)` call) with:

```python
            with gr.Row():
                with gr.Column():
                    model_name_textbox = gr.Textbox(label="Model repo (org/model)")
                    revision_name_textbox = gr.Textbox(label="Revision", placeholder="main")
                    submitter_textbox = gr.Textbox(label="Submitter / contact")
                with gr.Column():
                    stochastic_checkbox = gr.Checkbox(label="Stochastic model", value=False)
                    k_samples_number = gr.Number(label="K samples (if stochastic)", value=1, precision=0)

            submit_button = gr.Button("Submit Eval")
            submission_result = gr.Markdown()
            submit_button.click(
                add_new_eval,
                [
                    model_name_textbox,
                    revision_name_textbox,
                    submitter_textbox,
                    stochastic_checkbox,
                    k_samples_number,
                ],
                submission_result,
            )
```

- [ ] **Step 4: Update tab label**

Change the leaderboard tab label (line 98) from `"🏅 LLM Benchmark"` to `"🏅 Coilstellaration Benchmark"`.

- [ ] **Step 5: Smoke-check the app builds without launching**

Run:
```bash
.venv/bin/python -c "import os; os.environ['GRADIO_ANALYTICS_ENABLED']='False'; \
import unittest.mock as m; \
demo=__import__('app')" 2>&1 | tail -5 || true
```
Because `app.py` calls `demo.launch()` at import, instead verify syntax + imports with:
```bash
.venv/bin/python -c "import ast; ast.parse(open('app.py').read()); print('app.py parses')"
.venv/bin/python -c "from src.submission.submit import add_new_eval; from src.display.utils import COLS; print('wiring imports OK')"
```
Expected: `app.py parses` and `wiring imports OK`.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: coilstellaration submit form and leaderboard wiring"
```

---

## Phase 8: Full-suite verification & manual smoke

### Task 12: run the whole test suite

- [ ] **Step 1: Run all tests**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (marshal, onnx_model, eval, read_evals, check_validity).

- [ ] **Step 2: Commit any fixes** (only if changes were needed)

```bash
git add -A && git commit -m "test: full suite green"
```

### Task 13: manual end-to-end smoke (documented, not automated)

This exercises the real dataset + physics + ONNX path once. It needs network, `HF_TOKEN`, and access to the benchmark dataset. Record results in the PR description.

- [ ] **Step 1: Export a reference model to ONNX conforming to the contract**

Write a throwaway script `scripts/export_reference_onnx.py` that builds a `CoilPredictor`
(from `coilstellaration.machine_learning.model_definition`), wraps it so its ONNX graph
exposes inputs `boundary_r_cos`/`boundary_z_sin`/`requirement_metrics` and outputs
`coil_x_n`/`coil_y_n`/`coil_z_n`/`currents`, and exports via your preferred jax→ONNX path.
(If a clean jax→ONNX export is not readily available, use the `tests/backend/conftest.py`
`build_dummy_onnx` graph as a stand-in to validate the *pipeline* end-to-end; it produces
zero coilsets, which is fine for a plumbing smoke test.)

- [ ] **Step 2: Run a tiny eval**

Run:
```bash
COILBENCH_N=2 .venv/bin/python -c "
from src.backend.onnx_model import OnnxModel
from src.backend import eval as beval
m = OnnxModel('model.onnx')
rec = beval.evaluate_model(m, 'org/m', 'main', 'me', False, 1, '2026-06-11T00:00:00Z')
print(rec['overall_score'], rec['tracks'])
"
```
Expected: prints a finite `overall_score` in `[0, 1]` and both tracks with `n == 2`.

- [ ] **Step 3: Confirm and document** the values in the PR; no commit needed.

---

## Self-review notes (addressed)

- **Spec coverage:** §3 architecture → Tasks 9/11; §4 raw contract → Tasks 1–4 (refined to native shapes, documented in header); §5 harness → Task 5; §6 result schema → Tasks 5 & 7; §7 columns/display → Tasks 6/7/11; §8 config/hosting → Tasks 1 & 10; §9 submit/validation → Task 8; §10 testing → Tasks 2–8, 12–13. K-sample stochastic is designed-for (single pass now), per §11.
- **Type consistency:** `AutoEvalColumn` field names defined in Task 6 (`fixed_mean`, `var_feas09`, `stochastic`, …) are the exact names used in Task 7's `result_to_row`. Result-record keys produced by `evaluate_model` (Task 5: `tracks.<track>.{mean,median,p10,feasibility_rate_0p9,strict_feasibility,n}`, `overall_score`, `stochastic`, `k_samples`) match what `result_to_row` and the read-path test consume. `validate_onnx_contract` (Task 4) is reused by `validate_local_onnx` (Task 8). `ONNX_FILENAME` defined in `submit.py` (Task 8) and imported by `runner.py` (Task 9).
- **No placeholders:** every code step shows complete code; the only deferred value is the optional real jax→ONNX export in Task 13, which has a documented dummy-graph fallback.
```
