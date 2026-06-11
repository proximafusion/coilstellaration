"""onnxruntime wrapper + contract validation for submitted models."""
import logging

import numpy as np
import onnxruntime as ort

from src.backend import config

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
    logger.debug("Validating ONNX contract for %s", model_path)
    errors: list[str] = []
    try:
        session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # noqa: BLE001 - surface any load failure to the user
        logger.warning("Could not load ONNX model %s: %s", model_path, exc)
        return [f"Could not load ONNX model: {exc}"]

    input_names = {i.name for i in session.get_inputs()}
    output_names = {o.name for o in session.get_outputs()}

    for name in config.REQUIRED_INPUT_NAMES:
        if name not in input_names:
            errors.append(f"Missing required input tensor: {name!r}")
    for name in config.OUTPUT_NAMES:
        if name not in output_names:
            errors.append(f"Missing required output tensor: {name!r}")
    if errors:
        logger.debug("ONNX contract check found %d error(s)", len(errors))
    else:
        logger.debug("ONNX contract check passed")
    return errors
