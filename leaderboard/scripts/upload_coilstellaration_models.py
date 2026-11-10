"""Publish the exported `coilstellaration` ONNX models and queue them for eval.

Companion to `scripts/export_coilstellaration_onnx.py`, which writes the trained
coil-predictor checkpoints to `examples/<arch>_<id6>.onnx`. Those files are
git-ignored (see `.gitignore`); this script gets them onto the Hub instead:

  1. Upload every `examples/*.onnx` model into one Hub *model* repo
     (`jhewers/coilstellaration-models` by default), each kept at its own
     filename so a single repo holds all architectures.
  2. For each uploaded file, queue an eval *request* in the requests dataset
     (`<owner>/coilstellaration-bench-requests`) with `n_instances=10`, by
     delegating to the leaderboard's own `add_new_eval` -- so the request JSON,
     ONNX-contract validation, and dedup all match exactly what the backend
     produces for a web submission.

Auth: needs a write `HF_TOKEN` for the target org.

Run:  HF_TOKEN=hf_... uv run python -m \
          leaderboard.scripts.upload_coilstellaration_models
      (add --dry-run to validate/list without touching the Hub)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_SKIP = set()


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--owner",
        default="jhewers-pf",
        help="Hub org/user that owns both repos (default: jhewers).",
    )
    p.add_argument(
        "--models-repo",
        default=None,
        help="Model repo id (default: <owner>/coilstellaration-models).",
    )
    p.add_argument(
        "--examples-dir",
        type=Path,
        default=repo_root / "examples",
        help="Dir holding the *.onnx exports.",
    )
    p.add_argument(
        "--submitter",
        default=None,
        help="Submitter recorded in each request (default: <owner>).",
    )
    p.add_argument(
        "--n-instances",
        type=int,
        default=10,
        help="Instances per track to score (default: 10).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would happen; do not upload anything.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    owner = args.owner
    models_repo = args.models_repo or f"{owner}/coilstellaration-models"
    submitter = args.submitter or owner

    onnx_files = sorted(
        p for p in args.examples_dir.glob("*.onnx") if p.name not in _SKIP
    )
    if not onnx_files:
        print(
            f"No coil-predictor .onnx files found in {args.examples_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(onnx_files)} model(s) in {args.examples_dir}:")
    for f in onnx_files:
        print(f"  {f.name}")
    # Point the leaderboard's QUEUE_REPO at <owner>'s requests dataset. envs.py
    # reads this at import time, so set it before importing anything under `src`.
    os.environ.setdefault("COILBENCH_OWNER", owner)
    from leaderboard.src.envs import QUEUE_REPO

    print(f"-> model repo:   {models_repo}")
    print(
        f"-> requests repo: {QUEUE_REPO}  "
        f"(n_instances={args.n_instances}, submitter={submitter})"
    )

    if args.dry_run:
        print("\n[dry-run] nothing uploaded.")
        return 0

    if not os.environ.get("HF_TOKEN"):
        print(
            "HF_TOKEN is not set; a write token for the target org is required.",
            file=sys.stderr,
        )
        return 1

    from leaderboard.src.envs import API, TOKEN  # pyright: ignore[reportMissingImports]
    from leaderboard.src.submission.submit import add_new_eval  # pyright: ignore[reportMissingImports]

    # 1) Ensure the model repo exists, then push the ONNX files + a model card.
    API.create_repo(repo_id=models_repo, repo_type="model", exist_ok=True, token=TOKEN)
    _folders = list(set([f.parent for f in onnx_files]))
    _filenames = list(set([f.name for f in onnx_files]))
    for folder in _folders:
        print(f"Uploading {folder}/*.onnx -> {models_repo}")
        API.upload_folder(
            repo_id=models_repo,
            repo_type="model",
            folder_path=folder,
            allow_patterns=_filenames,
            commit_message=f"Adding {folder}",
        )

    # 2) Queue one eval request per file (validates the ONNX contract + dedups).
    print(f"\nQueueing eval requests in {QUEUE_REPO}")
    failures = 0
    for f in onnx_files:
        result = add_new_eval(
            model=models_repo,
            revision="main",
            submitter=submitter,
            stochastic=False,
            k_samples=1,
            n_instances=args.n_instances,
            onnx_path=f.name,
        )
        ok = (
            "color: red" not in result
        )  # styled_error() is red; styled_message() is green
        # Strip the HTML wrapper for a readable one-line status.
        msg = result.split(">", 1)[-1].rsplit("<", 1)[0]
        print(f"  [{'ok' if ok else 'FAIL'}] {f.name}: {msg}")
        failures += not ok

    if failures:
        print(f"\n{failures} request(s) failed.", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    logging.basicConfig()
    raise SystemExit(main())
