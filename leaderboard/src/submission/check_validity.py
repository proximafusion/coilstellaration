import logging
from typing import Literal

import huggingface_hub
from huggingface_hub import ModelCard

logger = logging.getLogger(__name__)


def check_model_card(repo_id: str) -> tuple[bool, str]:
    """Checks if the model card and license exist and have been filled."""
    logger.debug("Checking model card for %s", repo_id)
    try:
        card = ModelCard.load(repo_id)
    except huggingface_hub.utils.EntryNotFoundError:
        logger.info("No model card found for %s", repo_id)
        return (
            False,
            "Please add a model card to your model to explain how you trained/fine-tuned it.",
        )

    # Enforce license metadata
    if card.data.license is None:
        if not ("license_name" in card.data and "license_link" in card.data):
            return False, (
                "License not found. Please add a license to your model card using the `license` metadata or a"
                " `license_name`/`license_link` pair."
            )

    # Enforce card content
    if len(card.text) < 200:
        return False, "Please add a description to your model card, it is too short."

    return True, ""


def validate_local_onnx(
    local_path: str,
) -> tuple[bool, str, list[Literal["fixed_shape", "variable_shape"]]]:
    """Open a downloaded ONNX file, check it against the I/O contract, and read which
    tracks it declares in its metadata.

    Returns (ok, message, supported_tracks). No inference is run. `supported_tracks`
    is empty when the contract check fails.
    """
    from leaderboard.src.backend.onnx_model import inspect_onnx

    logger.debug("Validating local ONNX file %s", local_path)
    errors, tracks = inspect_onnx(local_path)
    if errors:
        logger.info("Local ONNX validation failed: %s", "; ".join(errors))
        return False, "; ".join(errors), []
    return True, "ONNX contract OK", tracks
