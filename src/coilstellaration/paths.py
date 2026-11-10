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

    Accepts a bare stem (`"my_model"`) or a basename with extension (`"my_model.json"`).
    """
    stem = name.removesuffix(".json")
    exact = models_dir() / f"{stem}.json"
    if exact.exists():
        return exact
    candidates = sorted(models_dir().glob(f"*_{stem}.json", case_sensitive=True))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No checkpoint at {name}")
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Ambiguous checkpoint name {name!r}; matches: "
            f"{[c.name for c in candidates]}"
        )
    (candidate,) = candidates
    candidates = list(models_dir().glob(f"*{name}.json", case_sensitive=True))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No checkpoint at {name}")
    (candidate,) = candidates
    if not candidate.exists():
        raise FileNotFoundError(f"No checkpoint at {candidate.resolve()}")
    return candidate


OUTPUTS_PATH = pathlib.Path.home() / "tmp" / "outputs" / "coilstellaration"
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

LOCAL_CONSTELLARATION_PATH = pathlib.Path.home() / "tmp" / "outputs" / "constellaration"
LOCAL_CONSTELLARATION_PATH.mkdir(parents=True, exist_ok=True)
