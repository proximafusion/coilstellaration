"""Onnxruntime wrapper + contract validation for submitted models."""

import logging
from typing import Literal, Mapping

import numpy as np
import onnxruntime as ort

from leaderboard.src.backend import config

logger = logging.getLogger(__name__)


class OnnxModel:
    """Thin wrapper around an onnxruntime InferenceSession."""

    def __init__(self, model_path: str):
        logger.debug("Loading ONNX model from %s", model_path)
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
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
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
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
