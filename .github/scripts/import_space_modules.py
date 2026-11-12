"""Import every module the Hugging Face Spaces run, to catch missing dependencies.

The Spaces install only the `leaderboard` dependency group (see the sync
workflow), so a module that imports something from `dev` passes the test suite
but crashes the Space. Run this from the repo root in an environment built from
that same requirements.txt.

The app entry points (leaderboard/app*.py) aren't imported: they download from
the Hub and start schedulers at import time. Their own imports are third-party
packages that leaderboard/src also uses, plus leaderboard/src itself.
"""

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path.cwd()
# Same paths leaderboard/app.py adds: the repo root and src/.
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def _modules(package_dir: Path, prefix: str) -> list[str]:
    names = []
    for path in sorted(package_dir.rglob("*.py")):
        if path.name.endswith("_test.py") or "testing_data" in path.parts:
            continue
        parts = path.relative_to(package_dir).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join((prefix, *parts)))
    return names


modules = _modules(ROOT / "leaderboard" / "src", "leaderboard.src") + _modules(
    ROOT / "src" / "coilstellaration", "coilstellaration"
)
failed = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception:
        failed.append(name)
        print(f"FAILED {name}\n{traceback.format_exc()}", file=sys.stderr)

print(f"Imported {len(modules) - len(failed)}/{len(modules)} modules.")
sys.exit(1 if failed else 0)
