import datetime as dt
import io
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import CommitOperationAdd, hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from leaderboard.src.backend import config, evaluate, jax_cache, onnx_model
from leaderboard.src.envs import (
    API,
    EVAL_REQUESTS_PATH,
    QUEUE_REPO,
    RESULTS_REPO,
    TOKEN,
)
from leaderboard.src.submission.submit import ONNX_FILENAME, file_sha256

logger = logging.getLogger(__name__)


def _now() -> dt.datetime:
    """Timezone-aware UTC now; indirected so tests can freeze it."""
    return dt.datetime.now(dt.timezone.utc)


def _worker_id() -> str:
    """Identify which worker holds a lease (host:pid, or override via env)."""
    return (
        os.environ.get("COILBENCH_WORKER_ID") or f"{socket.gethostname()}:{os.getpid()}"
    )


def _iter_requests(requests_dir: str):
    for root, _, files in os.walk(requests_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            path = str(Path(root) / fname)
            try:
                req = json.loads(Path(path).read_text())
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


# How many times a conditional queue update re-reads the queue after losing a
# race with another commit before giving up.
_CONDITIONAL_PUSH_ATTEMPTS = 3


def _push_request(req: dict, rel_path: str, expect: dict | None = None) -> bool:
    """Write a request file to the queue; return whether it was written.

    Without `expect` the write is unconditional (for the worker that holds the
    request). With `expect`, the write only happens if the file on the Hub still has
    those field values, e.g. `{"status": "PENDING"}` when claiming. The commit is
    made with `parent_commit` set to the queue head that was read, so if anything
    else commits in between, the Hub rejects it and the check is redone. Two
    workers therefore can't both claim the same request.
    """
    data = json.dumps(req, indent=2).encode()
    message = f"Update status {req['model']} -> {req['status']}"
    logger.debug("Pushing request %s status -> %s", req["model"], req["status"])
    if expect is None:
        API.upload_file(
            path_or_fileobj=data,
            path_in_repo=rel_path,
            repo_id=QUEUE_REPO,
            repo_type="dataset",
            commit_message=message,
        )
        return True

    for _ in range(_CONDITIONAL_PUSH_ATTEMPTS):
        head = API.repo_info(repo_id=QUEUE_REPO, repo_type="dataset").sha
        current = json.loads(
            Path(
                hf_hub_download(
                    repo_id=QUEUE_REPO,
                    filename=rel_path,
                    repo_type="dataset",
                    revision=head,
                    token=TOKEN,
                )
            ).read_text()
        )
        if any(current.get(k) != v for k, v in expect.items()):
            logger.info(
                "Request %s changed on the Hub (status %s); not updating it",
                rel_path,
                current.get("status"),
            )
            return False
        try:
            API.create_commit(
                repo_id=QUEUE_REPO,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(path_in_repo=rel_path, path_or_fileobj=data)
                ],
                commit_message=message,
                parent_commit=head,
            )
            return True
        except HfHubHTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in (409, 412):
                raise
            logger.info("Queue moved while updating %s; re-checking", rel_path)
    logger.warning(
        "Gave up updating %s after %d conflicting commits",
        rel_path,
        _CONDITIONAL_PUSH_ATTEMPTS,
    )
    return False


def _result_rel_path(record: dict, ext: str) -> str:
    """Results-repo path for a record's artifact, disambiguated by model, commit and
    ONNX file so re-submissions and per-file variants don't collide."""
    org, model = record["model"].split("/", 1)
    commit = record.get("commit_hash") or record.get("revision", "")
    onnx_slug = (
        record.get("onnx_path", "model.onnx").replace("/", "-").removesuffix(".onnx")
    )
    return f"{org}/{model}_{commit}_{onnx_slug}.{ext}"


def _push_result(record: dict):
    rel_path = _result_rel_path(record, "json")
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
    Path(tmp).unlink()


def _push_result_parquet(record: dict, parquet_bytes: bytes):
    """Upload the raw per-instance scores parquet alongside the JSON summary."""
    rel_path = _result_rel_path(record, "parquet")
    logger.debug("Pushing raw scores parquet for %s to %s", record["model"], rel_path)
    API.upload_file(
        path_or_fileobj=parquet_bytes,
        path_in_repo=rel_path,
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"Add raw scores for {record['model']}",
    )


def _push_result_model(record: dict, model_bytes: bytes):
    """Upload the ONNX model alongside the JSON summary."""
    rel_path = _result_rel_path(record, "onnx")
    logger.debug("Pushing model ONNX for %s to %s", record["model"], rel_path)
    API.upload_file(
        path_or_fileobj=model_bytes,
        path_in_repo=rel_path,
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"Add ONNX for {record['model']}",
    )


def _check_remote_onnx(
    model: str, commit_hash: str | None, onnx_path: str, expected_sha256: str | None
) -> None:
    """Validate the submitted file on the Hub before downloading it.

    The request must be pinned to a commit SHA (never a branch, which can move after
    validation). At that commit the file must exist and be within the size limit,
    and when the Hub stores it in LFS its SHA-256 must match the one recorded at
    submission. Raises `ValueError` otherwise.
    """
    if not commit_hash:
        raise ValueError("Request has no commit_hash; refusing to evaluate a branch")
    infos = API.get_paths_info(
        repo_id=model,
        paths=[onnx_path],
        revision=commit_hash,
        repo_type="model",
        token=False,
    )
    info = next((i for i in infos if getattr(i, "path", None) == onnx_path), None)
    if info is None:
        raise ValueError(f"{onnx_path} not found in {model} at {commit_hash}")
    size = getattr(info, "size", None)
    if size is not None and size > config.MAX_ONNX_BYTES:
        raise ValueError(
            f"{onnx_path} is {size} bytes, over the {config.MAX_ONNX_BYTES}-byte limit"
        )
    lfs = getattr(info, "lfs", None)
    if expected_sha256 and lfs is not None and lfs.sha256 != expected_sha256:
        raise ValueError(
            f"{onnx_path} at {commit_hash} does not match the SHA-256 recorded at "
            "submission"
        )


def _run_eval(req: dict) -> tuple[dict, list[dict], bytes]:
    """Download the submitted ONNX at its pinned ref and score it.

    May be slow.
    """
    jax_cache.enable()
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = req.get("onnx_path", ONNX_FILENAME)
        ref = req.get("commit_hash")
        expected_sha256 = req.get("onnx_sha256")
        _check_remote_onnx(req["model"], ref, onnx_path, expected_sha256)
        logger.info("Downloading %s from %s at %s", onnx_path, req["model"], ref)
        dl_start = time.perf_counter()
        local_model_path = hf_hub_download(
            repo_id=req["model"],
            filename=onnx_path,
            revision=ref,
            local_dir=tmp,
            # Submissions are public repos; never send the worker's write token.
            token=False,
        )
        logger.info(
            "Downloaded %s (%.1f MB) in %.1fs",
            onnx_path,
            Path(local_model_path).stat().st_size / 1e6,
            time.perf_counter() - dl_start,
        )
        if expected_sha256 and file_sha256(local_model_path) != expected_sha256:
            raise ValueError(
                f"{onnx_path} at {ref} does not match the SHA-256 recorded at "
                "submission"
            )
        model = onnx_model.OnnxModel(local_model_path)
        # The model's own metadata decides the tracks: a queued request's `tracks`
        # may predate a re-exported model and must not override what it declares.
        tracks = onnx_model.read_supported_tracks(model.session)
        record, rows = evaluate.evaluate_model(
            model=model,
            tracks=tracks,
        )
        metadata = dict(
            model=req["model"],
            revision=req["revision"],
            submitter=req.get("submitter", ""),
            commit_hash=req.get("commit_hash", ""),
            onnx_path=onnx_path,
            submitted_time=req.get("submitted_time", ""),
        )
        record |= metadata
        model_bytes = Path(local_model_path).read_bytes()
        return record, rows, model_bytes


_EVAL_CMD = [sys.executable, "-u", "-m", "leaderboard.src.backend.evaluate_one"]
# `-m leaderboard...` needs the repo root importable, whatever the worker's cwd.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])


# Environment variables huggingface_hub reads a token from. The eval child runs
# the submitted (untrusted) model and needs only public repos, so it gets none of
# them and can't use or leak the worker's write token.
_HUB_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN")


def _without_hub_tokens(env) -> dict[str, str]:
    """A copy of `env` without any Hugging Face Hub token."""
    return {k: v for k, v in env.items() if k not in _HUB_TOKEN_ENV_VARS}


def _kill_group(proc: "subprocess.Popen") -> None:
    """SIGKILL the child's whole process group, then reap it."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass  # already gone
    proc.wait()


def _evaluate_with_timeout(
    req: dict, timeout: float | None = None
) -> tuple[dict, bytes | None, bytes | None]:
    """Run the eval in an isolated subprocess under a wall-clock kill.

    Returns `(record, parquet_bytes, model_bytes)` where `parquet_bytes` is the
    raw per-instance scores parquet (or `None` when the eval produced no rows) and
    `model_bytes` is the evaluated ONNX (or `None` if the child did not write it).

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
        record, rows, model_bytes = _run_eval(req)
        parquet_bytes = None
        if rows:
            df = pd.DataFrame.from_records(rows)
            df["results_json"] = None
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            parquet_bytes = buf.getvalue()
        return record, parquet_bytes, model_bytes

    with tempfile.TemporaryDirectory() as tmp:
        req_path = Path(tmp) / "request.json"
        out_path = Path(tmp) / "result.json"
        req_path.write_text(json.dumps(req, indent=2))

        child_env = {
            **_without_hub_tokens(os.environ),
            evaluate.DEADLINE_ENV: str(time.time() + timeout),
            "PYTHONPATH": os.pathsep.join(
                p for p in (_REPO_ROOT, os.environ.get("PYTHONPATH")) if p
            ),
        }
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
                raise RuntimeError(
                    "Evaluation crashed (segmentation fault), "
                    "likely a JAX/LLVM crash or OOM"
                )
            raise RuntimeError(f"Evaluation subprocess exited {proc.returncode}")
        record = json.loads(out_path.read_text())
        # The child writes the raw scores parquet next to the JSON summary; read
        # it back before the temp dir is torn down so the runner can upload it.
        parquet_path = out_path.with_suffix(".parquet")
        parquet_bytes = parquet_path.read_bytes() if parquet_path.exists() else None
        model_path = out_path.with_suffix(".onnx")
        model_bytes = model_path.read_bytes() if model_path.exists() else None
        return record, parquet_bytes, model_bytes


def _reap_stale(requests_dir: str) -> None:
    """Recover RUNNING requests whose lease expired (worker presumed dead).

    Under the attempt cap they go back to PENDING for another worker; at the cap they
    fail permanently so a job that keeps killing its worker can't loop forever.
    """
    now = _now()
    for path, req in _iter_requests(requests_dir):
        if req.get("status") != "RUNNING":
            continue
        started = req.get("started_at")
        if not started:
            continue
        age = (now - dt.datetime.fromisoformat(started)).total_seconds()
        if age < config.LEASE_TTL:
            continue
        rel_path = path.split("eval-queue/")[1]
        if req.get("attempts", 0) >= config.MAX_ATTEMPTS:
            logger.warning(
                "Request %s exceeded %d attempts; marking FAILED",
                req["model"],
                config.MAX_ATTEMPTS,
            )
            req["status"] = "FAILED"
            req["error"] = (
                f"Exceeded {config.MAX_ATTEMPTS} attempts; worker presumed dead"
            )
        else:
            logger.warning(
                "Re-queuing stale RUNNING request %s (lease expired after %.0fs)",
                req["model"],
                age,
            )
            req["status"] = "PENDING"
        # Only if no other worker has reaped or re-claimed it in the meantime.
        _push_request(
            req, rel_path, expect={"status": "RUNNING", "started_at": started}
        )


def run_once() -> tuple[int, int]:
    """Process every PENDING request once; return `(finished, failed)` counts."""
    logger.info("Polling queue %s for PENDING requests", QUEUE_REPO)
    snapshot_download(
        repo_id=QUEUE_REPO,
        local_dir=EVAL_REQUESTS_PATH,
        repo_type="dataset",
        token=TOKEN,
    )
    _reap_stale(EVAL_REQUESTS_PATH)
    finished = 0
    failed = 0
    for path, req in list(_iter_pending(EVAL_REQUESTS_PATH)):
        rel_path = path.split("eval-queue/")[1]
        logger.info(
            "Processing PENDING request %s (revision %s)", req["model"], req["revision"]
        )
        req["attempts"] = req.get("attempts", 0) + 1
        req["status"] = "RUNNING"
        req["started_at"] = _now().isoformat()
        req["worker_id"] = _worker_id()
        if not _push_request(req, rel_path, expect={"status": "PENDING"}):
            logger.info("Request %s was claimed elsewhere; skipping", req["model"])
            continue
        try:
            record, parquet_bytes, model_bytes = _evaluate_with_timeout(
                req, config.EVAL_TIMEOUT
            )
            _push_result(record)
            if parquet_bytes:
                _push_result_parquet(record, parquet_bytes)
            if model_bytes:
                _push_result_model(record, model_bytes)
            req["status"] = "FINISHED"
            req.pop("error", None)
            finished += 1
            logger.info("Request %s FINISHED", req["model"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Request %s FAILED: %s", req["model"], exc)
            req["status"] = "FAILED"
            req["error"] = str(exc)
            failed += 1
        _push_request(req, rel_path)
    return finished, failed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logging.getLogger("leaderboard.src").setLevel(logging.DEBUG)
    run_once()
