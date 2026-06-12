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

# --- Worker robustness (see src/backend/runner.py) ---
# Wall-clock kill for a single evaluation (download + all tracks). 0 disables the
# subprocess guard and runs the eval inline. Keeps one hung model from wedging the
# poll loop (the scheduler runs at most one run_once at a time).
EVAL_TIMEOUT: int = int(os.environ.get("COILBENCH_EVAL_TIMEOUT", "1800"))
# A request left RUNNING longer than this is presumed orphaned (crashed/killed
# worker) and reaped on the next poll. Keep it comfortably above EVAL_TIMEOUT.
LEASE_TTL: int = int(os.environ.get("COILBENCH_LEASE_TTL", "3600"))
# How many times a request may be (re)claimed before it is failed permanently,
# so a job that reliably kills its worker can't loop forever.
MAX_ATTEMPTS: int = int(os.environ.get("COILBENCH_MAX_ATTEMPTS", "3"))

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
