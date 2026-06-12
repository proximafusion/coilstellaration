"""Subprocess entrypoint: evaluate a single request in an isolated interpreter.

Invoked as ``python -m src.backend.eval_one <request_path> <result_path>``: reads the
request JSON from ``<request_path>`` and writes the result record JSON to
``<result_path>``. Using files (not pipes) keeps the parent's wait loop free of any
pipe-buffer deadlock while the long-running eval holds its descriptors open.

Running the eval as a plain freshly-exec'd subprocess (rather than a multiprocessing
fork/spawn child) means it has its own guarded ``__main__`` and a clean interpreter:
no re-import of the worker entrypoint's import-time side effects, and no inherited
JAX threads. The parent (`runner._evaluate_with_timeout`) enforces the wall-clock
kill, so a hung or crashing eval can't wedge or take down the worker.
"""
import json
import logging
import sys

from src.backend import runner

logger = logging.getLogger(__name__)


def main() -> None:
    req_path, out_path = sys.argv[1], sys.argv[2]
    with open(req_path) as f:
        req = json.load(f)
    record = runner._run_eval(req)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
