import json
import os
import tempfile
from datetime import datetime, timezone

from huggingface_hub import hf_hub_download

from src.display.formatting import styled_error, styled_message, styled_warning
from src.envs import API, EVAL_REQUESTS_PATH, QUEUE_REPO, TOKEN
from src.submission.check_validity import is_model_on_hub, validate_local_onnx

ONNX_FILENAME = "model.onnx"


def add_new_eval(model: str, revision: str, submitter: str, stochastic: bool, k_samples: int):
    if not model or "/" not in model:
        return styled_error("Please enter a model id of the form `org/model`.")
    if not submitter:
        return styled_error("Please enter a submitter name or contact.")
    revision = revision or "main"
    k_samples = int(k_samples) if stochastic else 1

    user_name, model_path = model.split("/", 1)

    # 1) Repo is public and loadable.
    on_hub, error, _ = is_model_on_hub(model_name=model, revision=revision, token=TOKEN, test_tokenizer=False)
    if not on_hub:
        return styled_error(f'Model "{model}" {error}')

    # 2) Download model.onnx and validate the I/O contract.
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = hf_hub_download(
                repo_id=model, filename=ONNX_FILENAME, revision=revision,
                local_dir=tmp, token=TOKEN,
            )
            ok, msg = validate_local_onnx(local)
    except Exception as exc:  # noqa: BLE001
        return styled_error(f"Could not download `{ONNX_FILENAME}` from the repo: {exc}")
    if not ok:
        return styled_error(f"ONNX contract check failed: {msg}")

    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    eval_entry = {
        "model": model,
        "revision": revision,
        "submitter": submitter,
        "stochastic": bool(stochastic),
        "k_samples": k_samples,
        "status": "PENDING",
        "submitted_time": current_time,
        "private": False,
    }

    out_dir = f"{EVAL_REQUESTS_PATH}/{user_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{model_path}_eval_request_{revision}.json"
    with open(out_path, "w") as f:
        f.write(json.dumps(eval_entry))

    API.upload_file(
        path_or_fileobj=out_path,
        path_in_repo=out_path.split("eval-queue/")[1],
        repo_id=QUEUE_REPO,
        repo_type="dataset",
        commit_message=f"Add {model} to eval queue",
    )
    os.remove(out_path)
    return styled_message("Your model has been submitted to the evaluation queue!")
