"""Check that a Hugging Face repo is a valid Coilstellaration submission.

Runs the same steps the leaderboard does at submit time, plus a dummy forward
pass, against any repo you can authenticate to. Useful for verifying a model
(including a private repo) before submitting.

Auth: uses HF_TOKEN if set, otherwise your cached `huggingface-cli login` token.

Usage:
    python scripts/check_submission.py jhewers-pf/coilstellaration-bench-example-circle-a
    python scripts/check_submission.py <repo> --revision main
"""

import argparse
import os
import sys
import tempfile

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from src.backend.onnx_model import OnnxModel, validate_onnx_contract

ONNX_FILENAME = "model.onnx"


def check(repo: str, revision: str) -> bool:
    token = os.environ.get("HF_TOKEN")  # None -> falls back to cached login token
    api = HfApi(token=token)

    # 1) Repo exists and is accessible with our credentials.
    try:
        api.repo_info(repo_id=repo, revision=revision, repo_type="model", token=token)
    except Exception as exc:  # noqa: BLE001
        print(
            f"FAIL: repo not accessible ({type(exc).__name__}). "
            f"If it is private, set HF_TOKEN or run `huggingface-cli login`.\n  {exc}"
        )
        return False
    print(f"ok: repo accessible: {repo}@{revision}")

    with tempfile.TemporaryDirectory() as tmp:
        # 2) model.onnx is present and downloadable.
        try:
            local = hf_hub_download(
                repo_id=repo,
                filename=ONNX_FILENAME,
                revision=revision,
                local_dir=tmp,
                token=token,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: could not download `{ONNX_FILENAME}`: {exc}")
            return False
        print(f"ok: downloaded {ONNX_FILENAME} ({os.path.getsize(local)} bytes)")

        # 3) I/O contract (tensor names).
        errors = validate_onnx_contract(local)
        if errors:
            print("FAIL: contract violations:\n  - " + "\n  - ".join(errors))
            return False
        print("ok: ONNX I/O contract satisfied")

        # 4) Dummy forward pass at a representative boundary shape.
        model = OnnxModel(local)
        feed = {
            "boundary_r_cos": np.zeros((5, 9), np.float32),
            "boundary_z_sin": np.zeros((5, 9), np.float32),
            "requirement_metrics": np.zeros((4,), np.float32),
            "n_field_periods": np.asarray(3, np.int64),
            "n_coils_per_half_period": np.asarray(5, np.int64),
        }
        if "boundary_r_sin" in model.input_names:
            feed["boundary_r_sin"] = np.zeros((5, 9), np.float32)
            feed["boundary_z_cos"] = np.zeros((5, 9), np.float32)
        try:
            out = model.run(feed)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: forward pass errored: {exc}")
            return False
        shapes = {k: tuple(np.asarray(v).shape) for k, v in out.items()}
        print(f"ok: forward pass produced {shapes}")

    print("\nPASS: this repo is a valid submission.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="HF model repo id, e.g. org/model")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    sys.exit(0 if check(args.repo, args.revision) else 1)


if __name__ == "__main__":
    main()
