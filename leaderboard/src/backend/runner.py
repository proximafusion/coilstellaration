"""Backend worker: poll the requests dataset, evaluate PENDING models, push results.

Run on suitable compute (NOT the Space):  python -m src.backend.runner
Requires HF_TOKEN with write access to the requests + results datasets.
"""
import json
import logging
import os
import tempfile

from huggingface_hub import hf_hub_download, snapshot_download

from src.backend import config
from src.backend import eval as beval
from src.backend.onnx_model import OnnxModel
from src.envs import API, EVAL_REQUESTS_PATH, QUEUE_REPO, RESULTS_REPO, TOKEN
from src.submission.submit import ONNX_FILENAME

logger = logging.getLogger(__name__)


def _iter_pending(requests_dir: str):
    for root, _, files in os.walk(requests_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            with open(path) as fp:
                req = json.load(fp)
            if req.get("status") == "PENDING":
                yield path, req


def _push_request(req: dict, rel_path: str):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(req, f)
        tmp = f.name
    logger.debug("Pushing request %s status -> %s", req["model"], req["status"])
    API.upload_file(path_or_fileobj=tmp, path_in_repo=rel_path, repo_id=QUEUE_REPO,
                    repo_type="dataset", commit_message=f"Update status {req['model']} -> {req['status']}")
    os.remove(tmp)


def _push_result(record: dict):
    org, model = record["model"].split("/", 1)
    rel_path = f"{org}/{model}_{record['revision']}.json"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(record, f)
        tmp = f.name
    logger.debug("Pushing result for %s to %s", record["model"], rel_path)
    API.upload_file(path_or_fileobj=tmp, path_in_repo=rel_path, repo_id=RESULTS_REPO,
                    repo_type="dataset", commit_message=f"Add results for {record['model']}")
    os.remove(tmp)


def run_once():
    logger.info("Polling queue %s for PENDING requests", QUEUE_REPO)
    snapshot_download(repo_id=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH, repo_type="dataset", token=TOKEN)
    for path, req in _iter_pending(EVAL_REQUESTS_PATH):
        rel_path = path.split("eval-queue/")[1]
        logger.info("Processing PENDING request %s (revision %s)", req["model"], req["revision"])
        req["status"] = "RUNNING"
        _push_request(req, rel_path)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                logger.debug("Downloading %s from %s", ONNX_FILENAME, req["model"])
                local = hf_hub_download(repo_id=req["model"], filename=ONNX_FILENAME,
                                        revision=req["revision"], local_dir=tmp, token=TOKEN)
                model = OnnxModel(local)
                record = beval.evaluate_model(
                    model=model, model_name=req["model"], revision=req["revision"],
                    submitter=req.get("submitter", ""), stochastic=req.get("stochastic", False),
                    k_samples=req.get("k_samples", 1), submitted_time=req.get("submitted_time", ""),
                    n_instances=req.get("n_instances", config.N_INSTANCES),
                )
            _push_result(record)
            req["status"] = "FINISHED"
            logger.info("Request %s FINISHED", req["model"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Request %s FAILED: %s", req["model"], exc)
            req["status"] = "FAILED"
            req["error"] = str(exc)
        _push_request(req, rel_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("src").setLevel(logging.DEBUG)
    run_once()
