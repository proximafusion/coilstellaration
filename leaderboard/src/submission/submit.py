import json
import logging
import os
import tempfile
from datetime import datetime, timezone

from huggingface_hub import hf_hub_download

from src.display.formatting import styled_error, styled_message
from src.envs import API, EVAL_REQUESTS_PATH, QUEUE_REPO, TOKEN
from src.submission.check_validity import already_submitted_models, submission_key, validate_local_onnx

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

    user_name, model_path = model.split("/", 1)
    logger.info(
        "New submission: model=%s revision=%s submitter=%s n_instances=%s",
        model, revision, submitter, n_instances,
    )

    # 1) Repo exists and is publicly accessible. Submissions are ONNX-only (no
    #    config.json), so we must NOT use a transformers AutoConfig check here.
    try:
        info = API.repo_info(repo_id=model, revision=revision, repo_type="model", token=TOKEN)
        commit_hash = info.sha
    except Exception as exc:  # noqa: BLE001
        logger.warning("Repo info check failed for %s: %s", model, exc)
        return styled_error(f'Model "{model}" was not found or is not accessible: {exc}')
    if not commit_hash:
        logger.warning("Could not resolve a commit hash for %s@%s", model, revision)
        return styled_error(f'Could not resolve a commit hash for "{model}" at revision "{revision}".')

    # Reject an identical (model, commit, file) submission already in the queue.
    if submission_key(model, commit_hash, onnx_path) in already_submitted_models(EVAL_REQUESTS_PATH):
        logger.info("Duplicate submission for %s@%s:%s", model, commit_hash, onnx_path)
        return styled_error("This exact model commit and file has already been submitted.")

    # 2) Download model.onnx and validate the I/O contract.
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

    out_dir = f"{EVAL_REQUESTS_PATH}/{user_name}"
    os.makedirs(out_dir, exist_ok=True)
    onnx_slug = onnx_path.replace("/", "-").removesuffix(".onnx")
    out_path = f"{out_dir}/{model_path}_eval_request_{revision}_{onnx_slug}.json"
    with open(out_path, "w") as f:
        f.write(json.dumps(eval_entry))

    API.upload_file(
        path_or_fileobj=out_path,
        path_in_repo=os.path.relpath(out_path, EVAL_REQUESTS_PATH),
        repo_id=QUEUE_REPO,
        repo_type="dataset",
        commit_message=f"Add {model} to eval queue",
    )
    os.remove(out_path)
    logger.info("Submission %s queued (PENDING) at %s", model, current_time)
    return styled_message("Your model has been submitted to the evaluation queue!")
