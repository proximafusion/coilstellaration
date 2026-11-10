import json
import logging
from collections.abc import Iterator
from pathlib import Path

from huggingface_hub import snapshot_download

from leaderboard.src.envs import API, QUEUE_REPO, TOKEN

logger = logging.getLogger(__name__)


def request_path(model: str, commit_hash: str, onnx_path: str) -> str:
    """Canonical path of a request file within the requests dataset repo.

    The filename encodes the dedup identity, so existence of this path *is* the
    duplicate check.
    """
    user_name, model_path = model.split("/", 1)
    onnx_slug = onnx_path.replace("/", "-").removesuffix(".onnx")
    return f"{user_name}/{model_path}_eval_request_{commit_hash}_{onnx_slug}.json"


def list_request_files() -> list[str]:
    """All request JSON paths currently in the requests dataset."""
    files = API.list_repo_files(repo_id=QUEUE_REPO, repo_type="dataset", token=TOKEN)
    return [f for f in files if f.endswith(".json")]


def submission_exists(model: str, commit_hash: str, onnx_path: str) -> bool:
    """True if an identical (model, commit, file) request is already queued."""
    return request_path(model, commit_hash, onnx_path) in list_request_files()


def iter_requests() -> Iterator[dict]:
    """Yield each request record from one snapshot of the requests dataset.

    The request JSON files are fetched in a single `snapshot_download` (only the
    changed ones are transferred), not one request per file. A malformed or
    partially uploaded file is logged and skipped rather than aborting the whole
    queue view.
    """
    root = Path(
        snapshot_download(
            repo_id=QUEUE_REPO,
            repo_type="dataset",
            allow_patterns="*.json",
            token=TOKEN,
        )
    )
    for path in sorted(root.rglob("*.json")):
        try:
            yield json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable request %s: %s", path, exc)
