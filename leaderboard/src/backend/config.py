import os
from pathlib import Path
from typing import Literal

from coilstellaration.benchmark import scoring_types

# The tracks a submission is scored on are declared by its ONNX metadata (see
# `SUPPORTED_TRACKS_METADATA_KEY`), over every stratum of the `eval` split. The
# paper's headline is the mean over all strata; the leaderboard additionally
# reports the per-stratum breakdown (loose/medium/tight/all) so the numbers line
# up with the paper's stratified figure.

TRACKS = Literal["fixed_shape", "variable_shape"]
STRATA = Literal["loose", "medium", "tight"]
SPLITS = Literal["train", "eval"]

# Number of instances per track. 0 = full split (production). Override for dev via env.
N_INSTANCES: int = int(os.environ.get("COILBENCH_N", "0"))

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
    str(Path.home() / ".cache" / "coilstellaration-bench" / "jax-compile"),
)

# Frozen scoring definition for the benchmark.
SCORING_SETTINGS: scoring_types.ScoringSettings = scoring_types.ScoringSettings()

# ONNX I/O contract
# Reject submissions whose ONNX file exceeds this size (checked from Hub
# metadata before download, so an oversized model is never pulled).
MAX_ONNX_BYTES: int = 50 * 1024 * 1024  # 50 MiB
REQUIRED_INPUT_NAMES: tuple[str, ...] = (
    "boundary_r_cos",
    "boundary_z_sin",
    "requirement_metrics",
    "n_field_periods",
    "n_coils_per_half_period",
)
# ONNX `metadata_props` key declaring which tracks the model supports. Its value
# must be exactly one of `SUPPORTED_TRACKS_VALUES`.
SUPPORTED_TRACKS_METADATA_KEY: str = "supported_tracks"
SUPPORTED_TRACKS_VALUES: tuple[str, ...] = ("fixed_shape", "fixed_shape,variable_shape")
# Present only for non-stellarator-symmetric instances.
OPTIONAL_INPUT_NAMES: tuple[str, ...] = ("boundary_r_sin", "boundary_z_cos")
OUTPUT_NAMES: tuple[str, ...] = ("coil_x_n", "coil_y_n", "coil_z_n", "currents")
