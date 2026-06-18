import json
import logging
import os
import tempfile
from datetime import datetime, timezone

from huggingface_hub import hf_hub_download

from src.display.formatting import styled_error, styled_message
from src.envs import API, QUEUE_REPO, TOKEN
from src.submission import requests_dataset
from src.submission.check_validity import validate_local_onnx

logger = logging.getLogger(__name__)

ONNX_FILENAME = "model.onnx"


def add_new_eval(model: str, revision: str, submitter: str, stochastic: bool, k_samples: int, n_instances: int = -1, onnx_path: str = "model.onnx"):
    if not model or "/" not in model:
        return styled_error("Please enter a model id of the form `org/model`.")
    if not submitter:
        return styled_error("Please enter a submitter name or contact.")
    revision = revision or "main"
    onnx_path = (onnx_path or ONNX_FILENAME).strip().lstrip("/")
    if not onnx_path.lower().endswith(".onnx"):
        return styled_error("The model file path must end in `.onnx`.")
    k_samples = int(k_samples) if stochastic else 1

    # Instances per track to score: any value < 1 means the full split 
    n_instances = int(n_instances)
    if n_instances < 1:
        n_instances = -1

    logger.info(
        "New submission: model=%s revision=%s submitter=%s n_instances=%s",
        model, revision, submitter, n_instances,
    )

    try:
        info = API.repo_info(repo_id=model, revision=revision, repo_type="model", token=TOKEN)
        commit_hash = info.sha
    except Exception as exc:  # noqa: BLE001
        logger.warning("Repo info check failed for %s: %s", model, exc)
        return styled_error(f'Model "{model}" was not found or is not accessible: {exc}')
    if not commit_hash:
        logger.warning("Could not resolve a commit hash for %s@%s", model, revision)
        return styled_error(f'Could not resolve a commit hash for "{model}" at revision "{revision}".')

    try:
        duplicate = requests_dataset.submission_exists(model, commit_hash, onnx_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not check the queue for duplicates: %s", exc)
        return styled_error("Couldn't verify against the queue — please retry.")
    if duplicate:
        logger.info("Duplicate submission for %s@%s:%s", model, commit_hash, onnx_path)
        return styled_error(
            f"This exact model commit ({commit_hash}) and file ({onnx_path}) has already been submitted."
        )

    try:
        with tempfile.TemporaryDirectory() as tmp:
            logger.debug("Downloading %s from %s for validation", onnx_path, model)
            local = hf_hub_download(
                repo_id=model, filename=onnx_path, revision=commit_hash,
                local_dir=tmp, token=TOKEN,
            )
            ok, msg = validate_local_onnx(local)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not download %s from %s: %s", onnx_path, model, exc)
        return styled_error(f"Could not download `{onnx_path}` from the repo: {exc}")
    if not ok:
        logger.info("ONNX contract check failed for %s: %s", model, msg)
        return styled_error(f"ONNX contract check failed: {msg}")

    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    eval_entry = {
        "model": model,
        "revision": revision,
        "commit_hash": commit_hash,
        "onnx_path": onnx_path,
        "submitter": submitter,
        "stochastic": bool(stochastic),
        "k_samples": k_samples,
        "n_instances": n_instances,
        "status": "PENDING",
        "submitted_time": current_time,
        "private": False,
    }

    rel_path = requests_dataset.request_path(model, commit_hash, onnx_path)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(eval_entry, f, indent=2)
        tmp = f.name
    API.upload_file(
        path_or_fileobj=tmp,
        path_in_repo=rel_path,
        repo_id=QUEUE_REPO,
        repo_type="dataset",
        commit_message=f"Add {model} to eval queue",
    )
    os.remove(tmp)
    logger.info("Submission %s queued (PENDING) at %s", model, current_time)
    return styled_message("Your model has been submitted to the evaluation queue!")
