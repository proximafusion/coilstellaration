import os

from coilstellaration.benchmark import types as benchmark_types

# Both tracks are evaluated for every submission, over every stratum of the
# `eval` split. The paper's headline is the mean over all strata; the leaderboard
# additionally reports the per-stratum breakdown (loose/medium/tight/all) so the
# numbers line up with the paper's stratified figure.
TRACKS: list[str] = ["fixed_shape", "variable_shape"]
STRATA: list[str] = ["loose", "medium", "tight"]
SPLIT: str = "eval"
# The reference stratum used by the patch-equivalence tests (the tightest, where
# the plasma single-period tiling is exact). Not the only stratum scored.
STRATUM: str = "tight"

# Number of instances per track. 0 = full split (production). Override for dev via env.
N_INSTANCES: int = int(os.environ.get("COILBENCH_N", "0"))

# Field periods of plasma surface retained when computing coil-to-plasma min
# distance (see src/backend/plasma_distance_patch.py). 1 (the default) tiles the
# plasma to a single period instead of the full NFP-torus; bit-exact on the
# benchmark instances because each (period-0) coil's nearest plasma point is in
# its own period. Raise toward NFP for a geometric safety margin at less
# speedup; >= NFP reproduces upstream exactly. 0 also means full NFP.
PLASMA_PERIODS: int = int(os.environ.get("COILBENCH_PLASMA_PERIODS", "1"))

# Wall-clock kill for a single evaluation (download + all tracks). 0 disables the
# subprocess guard and runs the eval inline.
EVAL_TIMEOUT: int = int(os.environ.get("COILBENCH_EVAL_TIMEOUT", str(4 * 60 * 60)))
# A request left RUNNING longer than this is presumed orphaned (crashed/killed
# worker) and reaped on the next poll.
LEASE_TTL: int = int(os.environ.get("COILBENCH_LEASE_TTL", str(6 * 60 * 60)))
# How many times a request may be (re)claimed before it is failed permanently,
# so a job that reliably kills its worker can't loop forever.
MAX_ATTEMPTS: int = int(os.environ.get("COILBENCH_MAX_ATTEMPTS", "3"))

# JAX persistent compilation cache
JAX_COMPILE_CACHE_DIR: str = os.environ.get(
    "COILBENCH_JAX_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "coilstellaration-bench", "jax-compile"),
)

# Frozen scoring definition for the benchmark.
SCORING_SETTINGS: benchmark_types.ScoringSettings = benchmark_types.ScoringSettings()

# ONNX I/O contract
REQUIRED_INPUT_NAMES: tuple[str, ...] = (
    "boundary_r_cos",
    "boundary_z_sin",
    "requirement_metrics",
)
# Present only for non-stellarator-symmetric instances.
OPTIONAL_INPUT_NAMES: tuple[str, ...] = ("boundary_r_sin", "boundary_z_cos")
OUTPUT_NAMES: tuple[str, ...] = ("coil_x_n", "coil_y_n", "coil_z_n", "currents")
