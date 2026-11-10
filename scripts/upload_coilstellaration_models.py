"""Publish the `coilstellaration` JSON models to HG.

Auth: needs a write `HF_TOKEN` for the target org.

Run:  HF_TOKEN=hf_... uv run python scripts/upload_coilstellaration_models.py
      (add --dry-run to validate/list without touching the Hub)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

_SKIP = {"mlp_DHmCARHZ4ay9rsMDPdjVCu3.json"}


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
        "--data-dir",
        type=Path,
        default=repo_root / "src" / "coilstellaration" / "data" / "models",
        help="Dir holding the *.json files.",
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

    json_files = sorted(p for p in args.data_dir.glob("*.json") if p.name not in _SKIP)
    if not json_files:
        print(
            f"No coil-predictor .json files found in {args.data_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(json_files)} model(s) in {args.data_dir}:")
    for f in json_files:
        print(f"  {f.name}")
    print(f"-> model repo:   {models_repo}")

    if args.dry_run:
        print("\n[dry-run] nothing uploaded.")
        return 0

    if not os.environ.get("HF_TOKEN"):
        print(
            "HF_TOKEN is not set; a write token for the target org is required.",
            file=sys.stderr,
        )
        return 1

    TOKEN = os.environ.get("HF_TOKEN")  # A read/write token for your org
    if not TOKEN:
        logger.warning(
            "HF_TOKEN is not set; authenticated Hugging Face Hub calls will fail."
        )

    API = HfApi(token=TOKEN)

    # 1) Ensure the model repo exists, then push the JSON files.

    API.create_repo(
        repo_id=models_repo, private=True, repo_type="model", exist_ok=True, token=TOKEN
    )
    # create_repo only applies `private` on creation; enforce it for pre-existing repos.
    API.update_repo_settings(
        repo_id=models_repo, private=True, repo_type="model", token=TOKEN
    )

    _folders = list({f.parent for f in json_files})
    _filenames = list({f.name for f in json_files})
    for folder in map(Path, _folders):
        print(f"Uploading {folder}/*.json -> {models_repo}")
        API.upload_folder(
            repo_id=models_repo,
            repo_type="model",
            folder_path=folder,
            allow_patterns=_filenames,
            commit_message=f"Adding {len(list(folder.glob('*')))} files.",
        )
    print("\nDone.")
    return 0


if __name__ == "__main__":
    logging.basicConfig()
    raise SystemExit(main())
