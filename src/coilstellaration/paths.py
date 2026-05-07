import pathlib
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent


def data_dir() -> Path:
    """Directory containing package-shipped data assets."""
    return _PACKAGE_ROOT / "data"


def models_dir() -> Path:
    """Directory containing serialized model checkpoints."""
    return data_dir() / "models"


def model_path(name: str) -> Path:
    """Resolve a checkpoint JSON inside `models_dir()`.

    Accepts a bare stem (`"my_model"`) or a basename with extension
    (`"my_model.json"`).
    """
    candidate = models_dir() / name
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".json")
    if not candidate.exists():
        raise FileNotFoundError(f"No checkpoint at {candidate}")
    return candidate


OUTPUTS_PATH = pathlib.Path.home() / "tmp" / "outputs" / "coilstellaration"
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
