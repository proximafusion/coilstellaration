"""Wait for a Space to build and start the commit that was just synced to it.

Usage: python wait_for_space.py <owner/space> [timeout-minutes]

Exits 0 once the Space runs its latest commit, and 1 if the build or the app
fails, or the timeout passes, printing the end of the build or run logs.
Reads the token from HF_TOKEN.
"""

import itertools
import sys
import time

from huggingface_hub import HfApi

POLL_SECONDS = 30
# Stages in which the Space will not come up without a new commit or a restart.
ERROR_STAGES = {"BUILD_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "RUNTIME_ERROR"}
LOG_LINES = 100


def print_logs(api: HfApi, space: str, *, build: bool) -> None:
    kind = "build" if build else "run"
    print(f"::group::Last {LOG_LINES} {kind} log lines")
    try:
        lines = list(itertools.islice(api.fetch_space_logs(space, build=build), 5000))
        for line in lines[-LOG_LINES:]:
            print(line.rstrip("\n"))
    except Exception as exc:  # noqa: BLE001
        print(f"Could not fetch the {kind} logs: {exc}")
    print("::endgroup::")


def main() -> int:
    space = sys.argv[1]
    timeout = float(sys.argv[2]) * 60 if len(sys.argv) > 2 else 60 * 60
    api = HfApi()
    head = api.space_info(space).sha
    print(f"Waiting for {space} to run commit {head}")
    deadline = time.monotonic() + timeout
    while True:
        runtime = api.get_space_runtime(space)
        stage = runtime.stage
        on_head = runtime.raw.get("sha") == head
        print(f"stage={stage} commit={runtime.raw.get('sha')}", flush=True)
        if on_head and stage == "RUNNING":
            print(f"{space} is running {head}")
            return 0
        if on_head and stage in ERROR_STAGES:
            print(f"::error::{space} is in {stage} at {head}")
            print_logs(api, space, build=stage != "RUNTIME_ERROR")
            return 1
        if time.monotonic() > deadline:
            print(f"::error::{space} did not reach RUNNING within {timeout:.0f}s")
            print_logs(api, space, build=True)
            return 1
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
