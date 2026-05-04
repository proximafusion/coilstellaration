"""Local file-based persistence mirroring `dapper.read` / `dapper.write`.

Each `write(data)` returns a fresh 23-character object ID (`D` + 22 hex chars,
matching dapper's data-id shape) and persists `data` to a single ``.json`` file
under the data root. `read(cls, object_id)` validates the JSON back into an
instance of `cls`.

The data root defaults to `./data` (relative to the working directory) and
can be overridden with the ``CONSTELLARATION_UPDATE_DATA_ROOT`` environment
variable — useful for redirecting test artifacts into a tmp dir.

Serialization goes through pydantic's `model_dump_json` /
`model_validate_json`, so `data` must be a `pydantic.BaseModel`. Fields
holding binary blobs should use `pydantic.Base64Bytes` (or a custom
serializer) — plain `bytes` only round-trips through JSON when the contents
are valid UTF-8.
"""

import os
import pathlib
import uuid
from typing import TypeVar

import pydantic

_DATA_ROOT_ENV = "CONSTELLARATION_UPDATE_DATA_ROOT"
_DEFAULT_DATA_ROOT = "data"
_FILE_SUFFIX = ".json"

T = TypeVar("T", bound=pydantic.BaseModel)


def _data_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get(_DATA_ROOT_ENV, _DEFAULT_DATA_ROOT))


def _generate_id() -> str:
    return "D" + uuid.uuid4().hex[:22]


def _path_for(object_id: str) -> pathlib.Path:
    return _data_root() / f"{object_id}{_FILE_SUFFIX}"


def write(data: pydantic.BaseModel) -> str:
    """Persist `data` and return its newly-minted object ID."""
    object_id = _generate_id()
    root = _data_root()
    root.mkdir(parents=True, exist_ok=True)
    _path_for(object_id).write_text(data.model_dump_json())
    return object_id


def read(cls: type[T], object_id: str) -> T:
    """Load the object stored at `object_id` and return it as an instance of `cls`.

    Raises:
        FileNotFoundError: no object exists with that ID.
    """
    path = _path_for(object_id)
    if not path.exists():
        raise FileNotFoundError(f"No object stored at {path}")
    return cls.model_validate_json(path.read_text())
