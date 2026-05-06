"""Save/load helpers for Flax NNX modules via orbax-checkpoint.

The on-disk archive is a tar.gz of an `ocp.StandardCheckpointer` directory,
wrapped in `coilstellaration.blob.Blob`; configs are pure pydantic. See
`coilstellaration.types.FlaxNnxCheckpoint`.

`Blob`'s JSON shape is a superset of the legacy dapper representation, so
JSON dumps produced by `dapper.write` + `model_dump_json()` deserialize
directly into `FlaxNnxCheckpoint[ConfigT]` via `model_validate_json`.
"""

from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path
from typing import Any, cast

import orbax.checkpoint as ocp
import pydantic
from flax import nnx

from coilstellaration import types


def to_checkpoint[CkptT: types.FlaxNnxCheckpoint](
    model: nnx.Module,
    config: pydantic.BaseModel,
    checkpoint_cls: type[CkptT] = types.FlaxNnxCheckpoint,  # type: ignore[assignment]
) -> CkptT:
    _, state = nnx.split(model)
    with tempfile.TemporaryDirectory() as tmp:
        # Why: orbax refuses to write into an existing directory, so use a
        # fresh subpath.
        ckpt_dir = Path(tmp) / "ckpt"
        with ocp.StandardCheckpointer() as ckptr:
            # Why: this orbax version rejects `args=ocp.args.StandardSave(...)`
            # and only accepts the positional `(path, state)` form.
            ckptr.save(ckpt_dir.absolute(), state)
        archive_bytes = _tar_gz_directory(ckpt_dir)
    return checkpoint_cls(
        archive=types.Blob.from_bytes(archive_bytes, file_suffix=".tar.gz"),
        config=config,
    )


def from_checkpoint[M: nnx.Module](
    checkpoint: types.FlaxNnxCheckpoint,
    module_cls: type[M],
) -> M:
    # Why: eval_shape traces __init__ symbolically, so no parameter values are
    # allocated for the throwaway template. rngs is injected with a fixed seed
    # because eval_shape doesn't consume randomness, and any values produced
    # would be overwritten by the restored state anyway.
    # `cast(Any, ...)` sidesteps pyright's view that `nnx.Module.__init__` has
    # no parameters; concrete subclasses define their own `__init__` signatures.
    untyped_cls = cast(Any, module_cls)
    abstract_module = nnx.eval_shape(
        lambda: untyped_cls(checkpoint.config, rngs=nnx.Rngs(0)),
    )
    graphdef, abstract_state = nnx.split(abstract_module)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _untar_gz_into(checkpoint.archive.as_bytes(), tmp_path)
        ckpt_dir = tmp_path / "ckpt"
        with ocp.StandardCheckpointer() as ckptr:
            # Why: this orbax version uses the `target=` kwarg instead of the
            # `args=ocp.args.StandardRestore(...)` form.
            restored_state = ckptr.restore(
                ckpt_dir.absolute(),
                target=abstract_state,
            )
    return cast(M, nnx.merge(graphdef, restored_state))


def _tar_gz_directory(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(directory, arcname=directory.name)
    return buffer.getvalue()


def _untar_gz_into(archive_bytes: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        # Why: filter="data" rejects absolute paths, parent traversal, special
        # files, and unsafe symlinks. Soon to be Python's default; cheap defense
        # in depth against malicious archives.
        tar.extractall(destination, filter="data")
