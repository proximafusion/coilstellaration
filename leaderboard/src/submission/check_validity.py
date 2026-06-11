import json
import logging
import os

import huggingface_hub
from huggingface_hub import ModelCard

logger = logging.getLogger(__name__)


def check_model_card(repo_id: str) -> tuple[bool, str]:
    """Checks if the model card and license exist and have been filled"""
    logger.debug("Checking model card for %s", repo_id)
    try:
        card = ModelCard.load(repo_id)
    except huggingface_hub.utils.EntryNotFoundError:
        logger.info("No model card found for %s", repo_id)
        return False, "Please add a model card to your model to explain how you trained/fine-tuned it."

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

def validate_local_onnx(local_path: str) -> tuple[bool, str]:
    """Open a downloaded ONNX file and check it against the I/O contract.

    Returns (ok, message). No inference is run.
    """
    from src.backend.onnx_model import validate_onnx_contract

    logger.debug("Validating local ONNX file %s", local_path)
    errors = validate_onnx_contract(local_path)
    if errors:
        logger.info("Local ONNX validation failed: %s", "; ".join(errors))
        return False, "; ".join(errors)
    return True, "ONNX contract OK"


def submission_key(model: str, commit_hash: str, onnx_path: str) -> str:
    """Canonical dedup identity: same model + commit + file = same evaluation."""
    return f"{model}@{commit_hash}:{onnx_path}"


def already_submitted_models(requested_models_dir: str) -> set[str]:
    """Set of dedup keys (model@commit_hash:onnx_path) already in the queue."""
    keys: set[str] = set()
    for root, _, files in os.walk(requested_models_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file)) as f:
                info = json.load(f)
            model = info.get("model")
            if not model:
                continue
            keys.add(submission_key(model, info.get("commit_hash", ""), info.get("onnx_path", "model.onnx")))
    return keys
