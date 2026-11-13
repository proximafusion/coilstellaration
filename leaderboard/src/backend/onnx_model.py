"""Onnxruntime wrapper + contract validation for submitted models."""

import logging
import math
import os
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import onnx
import onnxruntime as ort

from leaderboard.src.backend import config

logger = logging.getLogger(__name__)

_CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_CPU_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_CPU_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _cgroup_cpu_limit() -> float | None:
    """CPUs the container's cgroup allows (quota / period), or None if unlimited."""
    try:
        quota, period = _CGROUP_V2_CPU_MAX.read_text().split()
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        quota = int(_CGROUP_V1_CPU_QUOTA.read_text())
        period = int(_CGROUP_V1_CPU_PERIOD.read_text())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def available_cpus() -> int:
    """CPUs this process may use: the affinity mask, capped by the cgroup quota.

    onnxruntime sizes its thread pool from the host's cores. In a container with a CPU
    quota, a pool of that size spends the quota on spinning threads and the kernel
    throttles the process for the rest of each period.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not available on every platform
        cpus = os.cpu_count() or 1
    limit = _cgroup_cpu_limit()
    if limit is not None:
        cpus = min(cpus, math.ceil(limit))
    return max(1, cpus)


def session_options() -> ort.SessionOptions:
    """Session options with the intra-op thread pool sized to `available_cpus()`."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = available_cpus()
    return options


class OnnxModel:
    """Thin wrapper around an onnxruntime InferenceSession."""

    def __init__(self, model_path: str):
        logger.debug("Loading ONNX model from %s", model_path)
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options(),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.debug(
            "ONNX session ready: inputs=%s outputs=%s",
            self.input_names,
            self.output_names,
        )

    def run(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Feed only the tensors the graph declares; return all named outputs."""
        feed = {k: v for k, v in inputs.items() if k in self.input_names}
        results = map(np.asarray, self.session.run(self.output_names, feed))
        return dict(zip(self.output_names, results))


def count_parameters(model_path: str) -> int:
    """Number of values stored in the graph: initializers plus Constant tensors.

    This is the usual parameter count (weights, biases and buffers such as
    normalization statistics), read without loading external data or running the
    model.
    """
    graph = onnx.load(model_path, load_external_data=False).graph
    initializers = sum(math.prod(t.dims) for t in graph.initializer)
    constants = sum(
        math.prod(attr.t.dims)
        for node in graph.node
        if node.op_type == "Constant"
        for attr in node.attribute
        if attr.name == "value"
    )
    return initializers + constants


def validate_onnx_contract(model_path: str) -> list[str]:
    """Check a graph against the I/O contract; empty means valid.

    Shapes are intentionally not checked (the model owns them).
    """
    return inspect_onnx(model_path)[0]


def read_supported_tracks(
    session: ort.InferenceSession,
) -> list[Literal["fixed_shape", "variable_shape"]]:
    """Tracks a model declares in its `supported_tracks` ONNX metadata.

    Raises:
        ValueError: If the metadata is missing or not exactly one of
            `config.SUPPORTED_TRACKS_VALUES`. There is no fallback: an undeclared
            or malformed value must not silently pick a track.
    """
    metadata = session.get_modelmeta().custom_metadata_map
    value = metadata.get(config.SUPPORTED_TRACKS_METADATA_KEY)
    if value is None:
        raise ValueError(
            f"Missing ONNX metadata {config.SUPPORTED_TRACKS_METADATA_KEY!r}; "
            f"expected one of {list(config.SUPPORTED_TRACKS_VALUES)}"
        )
    if value not in config.SUPPORTED_TRACKS_VALUES:
        raise ValueError(
            f"Invalid ONNX metadata {config.SUPPORTED_TRACKS_METADATA_KEY}={value!r}; "
            f"expected exactly one of {list(config.SUPPORTED_TRACKS_VALUES)}"
        )
    return value.split(",")


def inspect_onnx(
    model_path: str,
) -> tuple[list[str], list[Literal["fixed_shape", "variable_shape"]]]:
    """Open the graph once; return (contract_errors, supported_tracks).

    `supported_tracks` is empty whenever there are contract errors (including a load
    failure or invalid track metadata): nothing can be scored.
    """
    logger.debug("Inspecting ONNX graph %s", model_path)
    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=session_options(),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:  # noqa: BLE001 - surface any load failure to the user
        logger.warning("Could not load ONNX model %s: %s", model_path, exc)
        return [f"Could not load ONNX model: {exc}"], []

    input_names = {i.name for i in session.get_inputs()}
    output_names = {o.name for o in session.get_outputs()}
    errors: list[str] = []
    for name in config.REQUIRED_INPUT_NAMES:
        if name not in input_names:
            errors.append(f"Missing required input tensor: {name!r}")
    for name in config.OUTPUT_NAMES:
        if name not in output_names:
            errors.append(f"Missing required output tensor: {name!r}")
    tracks: list[Literal["fixed_shape", "variable_shape"]] = []
    try:
        tracks = read_supported_tracks(session)
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        logger.debug("ONNX contract check found %d error(s)", len(errors))
        return errors, []
    return errors, tracks
