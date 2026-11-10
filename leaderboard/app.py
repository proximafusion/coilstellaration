import logging
import os
import sys
from pathlib import Path

# The Space runs this file as a script (`python leaderboard/app.py`), which puts
# leaderboard/ on sys.path, not the repo root. Everything imports itself as
# `leaderboard.src...`, so add the root (where `python -m ...` would start).
# The Space's requirements.txt has only the dependencies, not this package
# (see the sync workflow), so add src/ too for `import coilstellaration`.
REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logging.getLogger("leaderboard.src").setLevel(logging.DEBUG)

coilbench_role = os.environ.get("COILBENCH_ROLE", "leaderboard").lower()

match coilbench_role:
    case "leaderboard":
        from leaderboard.app_leaderboard import app_leaderboard

        app_leaderboard.launch()
    case "runner":
        from leaderboard.app_runner import app_runner

        app_runner.launch()
    case _:
        raise SystemExit(
            f"Unknown COILBENCH_ROLE={coilbench_role!r}; "
            "expected 'leaderboard' or 'runner'."
        )
