"""Pydantic model payload for serialized Flax NNX module state."""

import typing

import pydantic

ConfigT = typing.TypeVar("ConfigT", bound=pydantic.BaseModel)


class FlaxNnxCheckpoint(pydantic.BaseModel, typing.Generic[ConfigT]):
    """Serialized state of a Flax NNX module, packaged for pydantic storage.

    Parameterize with a concrete `BaseModel` config type and bind the alias at
    module level (e.g. `MyCheckpoint = FlaxNnxCheckpoint[MyConfig]`) so pydantic
    sees a fully resolved class. The associated module class must accept the
    config as a single positional argument alongside keyword-only `rngs`;
    `from_checkpoint` calls `module_cls(config, rngs=...)` to rebuild the
    abstract state.
    """

    archive: bytes
    """tar.gz archive of the directory written by `ocp.StandardCheckpointer`."""

    config: ConfigT
    """Typed init config used to reconstruct the module's abstract template."""
