import hashlib
import json
import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

import gradio as gr
from huggingface_hub import hf_hub_download

from leaderboard.src.backend import config
from leaderboard.src.backend.onnx_model import count_parameters
from leaderboard.src.envs import API, QUEUE_REPO
from leaderboard.src.submission import requests_dataset
from leaderboard.src.submission.check_validity import validate_local_onnx

logger = logging.getLogger(__name__)

ONNX_FILENAME = "model.onnx"


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file's contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Submitted repos are read anonymously, never with the Space's token: only public
# models can be submitted, and a private repo that the token could read stays
# invisible to the submission form. It must be `token=False`: `token=None` falls
# back to the client's token (or HF_TOKEN).


def _limit_message(size: int) -> str:
    return (
        f"Model file is {size / 1024 / 1024:.1f} MB, over the "
        f"{config.MAX_ONNX_BYTES / 1024 / 1024:.0f} MB limit."
    )


# A Hugging Face user or organisation name. The leaderboard links it to the
# profile at huggingface.co/<name>.
HF_USERNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# The declaration a submitter must make, shown next to the form's checkbox.
NO_EVAL_TRAINING_DECLARATION = (
    "I declare that this model was not exposed to the eval split of "
    "`proxima-fusion/coilstellaration` during training, tuning or model selection."
)


def form_errors(
    model: str, submitter: str, onnx_path: str, no_eval_training: bool
) -> dict[str, str]:
    """The cheap format checks on the form, keyed by field; empty when it is valid."""
    errors = {}
    if not model or "/" not in model:
        errors["model"] = "Please enter a model id of the form `org/model`."
    if not submitter:
        errors["submitter"] = "Please enter your Hugging Face username."
    elif not HF_USERNAME.fullmatch(submitter.strip()):
        errors["submitter"] = (
            "That isn't a Hugging Face username: use letters, digits, `-`, `_` "
            "and `.` only."
        )
    onnx_path = (onnx_path or ONNX_FILENAME).strip()
    if not onnx_path.lower().endswith(".onnx"):
        errors["onnx_path"] = "The model file path must end in `.onnx`."
    if not no_eval_training:
        errors["no_eval_training"] = (
            "Please confirm that the model was not exposed to the eval split."
        )
    return errors


def validate_submission(
    model: str,
    revision: str,  # noqa: ARG001 (validators get every input)
    submitter: str,
    onnx_path: str,
    no_eval_training: bool,
):
    """Gradio validator for the form: a `gr.validate` per input, in input order."""
    errors = form_errors(model, submitter, onnx_path, no_eval_training)
    fields = ["model", "revision", "submitter", "onnx_path", "no_eval_training"]
    return [gr.validate(f not in errors, errors.get(f, "")) for f in fields]


def add_new_eval(
    model: str,
    revision: str,
    submitter: str,
    onnx_path: str = "model.onnx",
    no_eval_training: bool = False,
    progress=gr.Progress(),  # noqa: B008 (Gradio injects it by this default)
) -> str:
    """Validate a submission and add it to the evaluation queue.

    Returns the confirmation message. Raises `gr.Error` with the reason when the
    submission is rejected. The form validator runs the same format checks first,
    but they are repeated here because API callers skip the validator.
    """
    errors = form_errors(model, submitter, onnx_path, no_eval_training)
    if errors:
        raise gr.Error(" ".join(errors.values()))
    revision = revision or "main"
    submitter = submitter.strip()
    onnx_path = (onnx_path or ONNX_FILENAME).strip().lstrip("/")

    logger.info(
        "New submission: model=%s revision=%s submitter=%s",
        model,
        revision,
        submitter,
    )

    progress(0.1, desc="Resolving the model repo")
    try:
        info = API.repo_info(
            repo_id=model, revision=revision, repo_type="model", token=False
        )
        commit_hash = info.sha
    except Exception as exc:  # noqa: BLE001
        logger.warning("Repo info check failed for %s: %s", model, exc)
        raise gr.Error(
            f'Model "{model}" was not found or is not public: {exc}'
        ) from exc
    if not commit_hash:
        logger.warning("Could not resolve a commit hash for %s@%s", model, revision)
        raise gr.Error(
            f'Could not resolve a commit hash for "{model}" at revision "{revision}".'
        )

    progress(0.25, desc="Checking the queue for duplicates")
    try:
        duplicate = requests_dataset.submission_exists(model, commit_hash, onnx_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not check the queue for duplicates: %s", exc)
        raise gr.Error("Couldn't verify against the queue. Please retry.") from exc
    if duplicate:
        logger.info("Duplicate submission for %s@%s:%s", model, commit_hash, onnx_path)
        raise gr.Error(
            f"This exact model commit ({commit_hash}) and file ({onnx_path}) "
            "has already been submitted."
        )

    progress(0.4, desc=f"Reading `{onnx_path}` metadata")
    try:
        infos = API.get_paths_info(
            repo_id=model,
            paths=[onnx_path],
            revision=commit_hash,
            repo_type="model",
            token=False,
        )
        size = next(
            (
                getattr(i, "size", None)
                for i in infos
                if getattr(i, "path", None) == onnx_path
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not fetch metadata for %s from %s: %s", onnx_path, model, exc
        )
        raise gr.Error(
            f"Could not read `{onnx_path}` metadata from the repo: {exc}"
        ) from exc
    if size is not None and size > config.MAX_ONNX_BYTES:
        logger.info("Rejecting oversized model %s: %d bytes", model, size)
        raise gr.Error(_limit_message(size))

    progress(0.55, desc=f"Downloading `{onnx_path}`")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            logger.debug("Downloading %s from %s for validation", onnx_path, model)
            local = hf_hub_download(
                repo_id=model,
                filename=onnx_path,
                revision=commit_hash,
                local_dir=tmp,
                token=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not download %s from %s: %s", onnx_path, model, exc)
            raise gr.Error(
                f"Could not download `{onnx_path}` from the repo: {exc}"
            ) from exc
        actual = Path(local).stat().st_size
        if size is None and actual > config.MAX_ONNX_BYTES:
            logger.info(
                "Rejecting oversized model %s: %d bytes (post-download)",
                model,
                actual,
            )
            raise gr.Error(_limit_message(actual))
        progress(0.75, desc="Checking the ONNX contract")
        ok, msg, tracks = validate_local_onnx(local)
        onnx_sha256 = file_sha256(local)
        n_parameters = count_parameters(local) if ok else None
    if not ok:
        logger.info("ONNX contract check failed for %s: %s", model, msg)
        raise gr.Error(f"ONNX contract check failed: {msg}")

    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    eval_entry = {
        "model": model,
        "revision": revision,
        "commit_hash": commit_hash,
        "onnx_path": onnx_path,
        # The runner checks the file it evaluates against this hash.
        "onnx_sha256": onnx_sha256,
        "submitter": submitter,
        "n_parameters": n_parameters,
        "no_eval_training": True,
        "tracks": tracks,
        "status": "PENDING",
        "submitted_time": current_time,
        "private": False,
    }

    progress(0.9, desc="Adding the model to the queue")
    rel_path = requests_dataset.request_path(model, commit_hash, onnx_path)
    try:
        API.upload_file(
            path_or_fileobj=json.dumps(eval_entry, indent=2).encode(),
            path_in_repo=rel_path,
            repo_id=QUEUE_REPO,
            repo_type="dataset",
            commit_message=f"Add {model} to eval queue",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not add %s to the queue: %s", model, exc)
        raise gr.Error(
            "The model passed all checks, but adding it to the queue failed. "
            "Please retry."
        ) from exc
    if set(tracks) == set(get_args(config.TRACKS)):
        detail = (
            "It declares `supported_tracks=fixed_shape,variable_shape`, so it will "
            "be scored on both tracks."
        )
    else:
        detail = (
            "It declares `supported_tracks=fixed_shape`, so it will be scored on "
            "the fixed_shape track only."
        )
    logger.info("Submission %s queued (PENDING) at %s", model, current_time)
    message = f"Submitted `{model}` to the evaluation queue. {detail}"
    gr.Info(message)
    return message
