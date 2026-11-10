import logging
import os
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

TOKEN = os.environ.get("HF_TOKEN")  # A read/write token for your org
if not TOKEN:
    logger.warning(
        "HF_TOKEN is not set; authenticated Hugging Face Hub calls will fail."
    )
OWNER = os.environ.get("COILBENCH_OWNER", "proxima-fusion")
REPO_NAME = os.environ.get("COILBENCH_REPO", "coilstellaration-bench")

REPO_ID = f"{OWNER}/{REPO_NAME}"
QUEUE_REPO = f"{OWNER}/{REPO_NAME}-requests"
RESULTS_REPO = f"{OWNER}/{REPO_NAME}-results"

# If you setup a cache later, just change HF_HOME
CACHE_PATH = os.getenv("HF_HOME", ".")

# Local caches
EVAL_REQUESTS_PATH = str(Path(CACHE_PATH) / "eval-queue")
EVAL_RESULTS_PATH = str(Path(CACHE_PATH) / "eval-results")
EVAL_REQUESTS_PATH_BACKEND = str(Path(CACHE_PATH) / "eval-queue-bk")
EVAL_RESULTS_PATH_BACKEND = str(Path(CACHE_PATH) / "eval-results-bk")

API = HfApi(token=TOKEN)
