import datetime
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time

from huggingface_hub import hf_hub_download, snapshot_download

from src.backend import config, jax_cache
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
            try:
                with open(path) as fp:
                    req = json.load(fp)
            except (json.JSONDecodeError, OSError) as exc:
                # A single empty or corrupt file must not wedge the whole poll
                # loop -- skip it so valid requests still get reaped/processed.
                logger.warning("Skipping unreadable request file %s: %s", path, exc)
                continue
            yield path, req


def _iter_pending(requests_dir: str):
    for path, req in _iter_requests(requests_dir):
        if req.get("status") == "PENDING":
            yield path, req


def _push_request(req: dict, rel_path: str):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(req, f, indent=2)
        tmp = f.name
    logger.debug("Pushing request %s status -> %s", req["model"], req["status"])
    API.upload_file(
        path_or_fileobj=tmp,
        path_in_repo=rel_path,
        repo_id=QUEUE_REPO,
        repo_type="dataset",
        commit_message=f"Update status {req['model']} -> {req['status']}",
    )
    os.remove(tmp)


def _push_result(record: dict):
    org, model = record["model"].split("/", 1)
    commit = record.get("commit_hash") or record.get("revision", "")
    onnx_slug = record.get("onnx_path", "model.onnx").replace("/", "-").removesuffix(".onnx")
    rel_path = f"{org}/{model}_{commit}_{onnx_slug}.json"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(record, f, indent=2)
        tmp = f.name
    logger.debug("Pushing result for %s to %s", record["model"], rel_path)
    API.upload_file(
        path_or_fileobj=tmp,
        path_in_repo=rel_path,
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"Add results for {record['model']}",
    )
    os.remove(tmp)


def _run_eval(req: dict) -> dict:
    """Download the submitted ONNX at its pinned ref and score it. May be slow."""
    jax_cache.enable()
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = req.get("onnx_path", ONNX_FILENAME)
        ref = req.get("commit_hash") or req["revision"]
        logger.info("Downloading %s from %s at %s", onnx_path, req["model"], ref)
        dl_start = time.perf_counter()
        local = hf_hub_download(repo_id=req["model"], filename=onnx_path, revision=ref, local_dir=tmp, token=TOKEN)
        logger.info(
            "Downloaded %s (%.1f MB) in %.1fs",
            onnx_path, os.path.getsize(local) / 1e6, time.perf_counter() - dl_start,
        )
        model = OnnxModel(local)
        return beval.evaluate_model(
            model=model,
            model_name=req["model"],
            revision=req["revision"],
            submitter=req.get("submitter", ""),
            stochastic=req.get("stochastic", False),
            k_samples=req.get("k_samples", 1),
            submitted_time=req.get("submitted_time", ""),
            n_instances=req.get("n_instances", config.N_INSTANCES),
            commit_hash=req.get("commit_hash", ""),
            onnx_path=onnx_path,
        )


_EVAL_CMD = [sys.executable, "-u", "-m", "src.backend.eval_one"]


def _kill_group(proc: "subprocess.Popen") -> None:
    """SIGKILL the child's whole process group, then reap it."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass  # already gone
    proc.wait()


def _evaluate_with_timeout(req: dict, timeout: float | None = None) -> dict:
    """Run the eval in an isolated subprocess under a wall-clock kill.

    The child is its own session leader (``start_new_session=True``), so a timeout
    SIGKILLs the entire process group -- the eval plus any JAX/DESC helpers it
    spawned -- raising ``TimeoutError`` instead of wedging the single-threaded poll
    loop and leaking orphans. A hard crash (e.g. segfault from JAX/LLVM) and any
    non-zero exit surface as ``RuntimeError``. The child inherits the worker's
    stdout/stderr, so the eval's own logging streams straight to the console.
    ``timeout <= 0`` runs the eval inline (no subprocess).
    """
    if timeout is None:
        timeout = config.EVAL_TIMEOUT
    if not timeout or timeout <= 0:
        return _run_eval(req)

    with tempfile.TemporaryDirectory() as tmp:
        req_path = os.path.join(tmp, "request.json")
        out_path = os.path.join(tmp, "result.json")
        with open(req_path, "w") as f:
            json.dump(req, f, indent=2)

        child_env = {**os.environ, beval.DEADLINE_ENV: str(time.time() + timeout)}
        proc = subprocess.Popen(
            [*_EVAL_CMD, req_path, out_path],
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=child_env,
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _kill_group(proc)
            raise TimeoutError(f"Evaluation exceeded {timeout}s") from exc

        if proc.returncode != 0:
            if proc.returncode == -signal.SIGSEGV:
                raise RuntimeError("Evaluation crashed (segmentation fault), likely a JAX/LLVM crash or OOM")
            raise RuntimeError(f"Evaluation subprocess exited {proc.returncode}")
        with open(out_path) as f:
            return json.load(f)


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
            logger.warning("Request %s exceeded %d attempts; marking FAILED", req["model"], config.MAX_ATTEMPTS)
            req["status"] = "FAILED"
            req["error"] = f"Exceeded {config.MAX_ATTEMPTS} attempts; worker presumed dead"
        else:
            logger.warning("Re-queuing stale RUNNING request %s (lease expired after %.0fs)", req["model"], age)
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    logging.getLogger("src").setLevel(logging.DEBUG)
    run_once()
