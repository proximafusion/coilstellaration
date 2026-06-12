"""Backend worker: poll the requests dataset, evaluate PENDING models, push results.

Run on suitable compute (NOT the Space):  python -m src.backend.runner
Requires HF_TOKEN with write access to the requests + results datasets.
"""
import datetime
import json
import logging
import multiprocessing as mp
import os
import socket
import tempfile

from huggingface_hub import hf_hub_download, snapshot_download

from src.backend import config
from src.backend import eval as beval
from src.backend.onnx_model import OnnxModel
from src.envs import API, EVAL_REQUESTS_PATH, QUEUE_REPO, RESULTS_REPO, TOKEN
from src.submission.submit import ONNX_FILENAME

logger = logging.getLogger(__name__)


def _now() -> datetime.datetime:
    """Timezone-aware UTC now; indirected so tests can freeze it."""
    return datetime.datetime.now(datetime.timezone.utc)


def _worker_id() -> str:
    """Identify which worker holds a lease (host:pid, or override via env)."""
    return os.environ.get("COILBENCH_WORKER_ID") or f"{socket.gethostname()}:{os.getpid()}"


def _iter_requests(requests_dir: str):
    for root, _, files in os.walk(requests_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            with open(path) as fp:
                req = json.load(fp)
            yield path, req


def _iter_pending(requests_dir: str):
    for path, req in _iter_requests(requests_dir):
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
    commit = record.get("commit_hash") or record.get("revision", "")
    onnx_slug = record.get("onnx_path", "model.onnx").replace("/", "-").removesuffix(".onnx")
    rel_path = f"{org}/{model}_{commit}_{onnx_slug}.json"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(record, f)
        tmp = f.name
    logger.debug("Pushing result for %s to %s", record["model"], rel_path)
    API.upload_file(path_or_fileobj=tmp, path_in_repo=rel_path, repo_id=RESULTS_REPO,
                    repo_type="dataset", commit_message=f"Add results for {record['model']}")
    os.remove(tmp)


def _run_eval(req: dict) -> dict:
    """Download the submitted ONNX at its pinned ref and score it. May be slow."""
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = req.get("onnx_path", ONNX_FILENAME)
        ref = req.get("commit_hash") or req["revision"]
        logger.debug("Downloading %s from %s at %s", onnx_path, req["model"], ref)
        local = hf_hub_download(repo_id=req["model"], filename=onnx_path,
                                revision=ref, local_dir=tmp, token=TOKEN)
        model = OnnxModel(local)
        return beval.evaluate_model(
            model=model, model_name=req["model"], revision=req["revision"],
            submitter=req.get("submitter", ""), stochastic=req.get("stochastic", False),
            k_samples=req.get("k_samples", 1), submitted_time=req.get("submitted_time", ""),
            n_instances=req.get("n_instances", config.N_INSTANCES),
            commit_hash=req.get("commit_hash", ""), onnx_path=onnx_path,
        )


def _eval_worker(fn, req: dict, conn) -> None:
    """Child-process entrypoint: run ``fn(req)`` and ship the outcome back over a pipe."""
    try:
        conn.send(("ok", fn(req)))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", repr(exc)))
    finally:
        conn.close()


def _evaluate_with_timeout(req: dict, timeout: float | None = None, _fn=None) -> dict:
    """Run the eval under a wall-clock kill so a hung model can't wedge the loop.

    Runs the eval in a ``spawn``ed child (a fresh interpreter — ``fork`` would
    inherit JAX's threads and risk deadlocking the child). If it overruns
    ``timeout`` the child is killed and ``TimeoutError`` is raised. ``timeout <= 0``
    runs inline (no subprocess). ``_fn`` overrides the eval callable for tests.
    """
    fn = _fn or _run_eval
    if timeout is None:
        timeout = config.EVAL_TIMEOUT
    if not timeout or timeout <= 0:
        return fn(req)

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_eval_worker, args=(fn, req, child_conn), daemon=True)
    proc.start()
    child_conn.close()  # parent keeps only its end, so it sees EOF if the child dies
    try:
        if parent_conn.poll(timeout):
            try:
                kind, payload = parent_conn.recv()
            except EOFError:
                raise RuntimeError("Evaluation worker died without returning a result")
            proc.join()
            if kind == "err":
                raise RuntimeError(f"Evaluation raised: {payload}")
            return payload
        raise TimeoutError(f"Evaluation exceeded {timeout}s")
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join()
        parent_conn.close()


def _reap_stale(requests_dir: str) -> None:
    """Recover RUNNING requests whose lease expired (worker presumed dead).

    Under the attempt cap they go back to PENDING for another worker; at the cap
    they fail permanently so a job that keeps killing its worker can't loop forever.
    """
    now = _now()
    for path, req in _iter_requests(requests_dir):
        if req.get("status") != "RUNNING":
            continue
        started = req.get("started_at")
        if not started:
            continue
        age = (now - datetime.datetime.fromisoformat(started)).total_seconds()
        if age < config.LEASE_TTL:
            continue
        rel_path = path.split("eval-queue/")[1]
        if req.get("attempts", 0) >= config.MAX_ATTEMPTS:
            logger.warning("Request %s exceeded %d attempts; marking FAILED",
                           req["model"], config.MAX_ATTEMPTS)
            req["status"] = "FAILED"
            req["error"] = f"Exceeded {config.MAX_ATTEMPTS} attempts; worker presumed dead"
        else:
            logger.warning("Re-queuing stale RUNNING request %s (lease expired after %.0fs)",
                           req["model"], age)
            req["status"] = "PENDING"
        _push_request(req, rel_path)


def run_once():
    logger.info("Polling queue %s for PENDING requests", QUEUE_REPO)
    snapshot_download(repo_id=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH, repo_type="dataset", token=TOKEN)
    _reap_stale(EVAL_REQUESTS_PATH)
    for path, req in list(_iter_pending(EVAL_REQUESTS_PATH)):
        rel_path = path.split("eval-queue/")[1]
        logger.info("Processing PENDING request %s (revision %s)", req["model"], req["revision"])
        req["attempts"] = req.get("attempts", 0) + 1
        req["status"] = "RUNNING"
        req["started_at"] = _now().isoformat()
        req["worker_id"] = _worker_id()
        _push_request(req, rel_path)
        try:
            record = _evaluate_with_timeout(req, config.EVAL_TIMEOUT)
            _push_result(record)
            req["status"] = "FINISHED"
            req.pop("error", None)
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
