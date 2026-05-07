"""Tests for `flax_nnx_checkpoint_util` and the `Blob` payload type."""

import base64
import json
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pydantic
import pytest
from coilstellaration import (
    flax_nnx_checkpoint_util,
    types,
)
from flax import nnx


class _SimpleMlpConfig(pydantic.BaseModel):
    hidden: int


_SimpleMlpCheckpoint = types.FlaxNnxCheckpoint[_SimpleMlpConfig]


class _SimpleMlp(nnx.Module):
    def __init__(self, config: _SimpleMlpConfig, *, rngs: nnx.Rngs) -> None:
        self.linear1 = nnx.Linear(4, config.hidden, rngs=rngs)
        self.linear2 = nnx.Linear(config.hidden, 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear2(jax.nn.relu(self.linear1(x)))


def _make_simple_mlp(hidden: int) -> _SimpleMlp:
    return _SimpleMlp(_SimpleMlpConfig(hidden=hidden), rngs=nnx.Rngs(0))


def test_to_from_checkpoint_roundtrip_preserves_params():
    model = _make_simple_mlp(hidden=8)
    checkpoint = flax_nnx_checkpoint_util.to_checkpoint(
        model, _SimpleMlpConfig(hidden=8), _SimpleMlpCheckpoint
    )
    restored = flax_nnx_checkpoint_util.from_checkpoint(checkpoint, _SimpleMlp)

    _, original_state = nnx.split(model)
    _, restored_state = nnx.split(restored)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=0),
        original_state,
        restored_state,
    )

    x = jnp.ones((1, 4))
    np.testing.assert_allclose(np.asarray(model(x)), np.asarray(restored(x)), atol=0)


class _PlainLinearConfig(pydantic.BaseModel):
    in_features: int
    out_features: int


_PlainLinearCheckpoint = types.FlaxNnxCheckpoint[_PlainLinearConfig]


class _PlainLinear(nnx.Module):
    def __init__(self, config: _PlainLinearConfig, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(config.in_features, config.out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


class _NestedMlp(nnx.Module):
    def __init__(self, config: _SimpleMlpConfig, *, rngs: nnx.Rngs) -> None:
        self.inner = _SimpleMlp(config, rngs=rngs)
        self.head = nnx.Linear(2, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.head(self.inner(x))


@pytest.mark.parametrize(
    ("module_cls", "config", "checkpoint_cls", "input_shape"),
    [
        (
            _PlainLinear,
            _PlainLinearConfig(in_features=3, out_features=5),
            _PlainLinearCheckpoint,
            (1, 3),
        ),
        (
            _SimpleMlp,
            _SimpleMlpConfig(hidden=4),
            _SimpleMlpCheckpoint,
            (1, 4),
        ),
        (
            _NestedMlp,
            _SimpleMlpConfig(hidden=4),
            _SimpleMlpCheckpoint,
            (1, 4),
        ),
    ],
)
def test_from_checkpoint_preserves_concrete_type_and_outputs(
    module_cls: type[nnx.Module],
    config: pydantic.BaseModel,
    checkpoint_cls: type[types.FlaxNnxCheckpoint],
    input_shape: tuple[int, ...],
) -> None:
    model = module_cls(config, rngs=nnx.Rngs(0))  # pyright: ignore[reportCallIssue]
    checkpoint = flax_nnx_checkpoint_util.to_checkpoint(model, config, checkpoint_cls)
    restored = flax_nnx_checkpoint_util.from_checkpoint(checkpoint, module_cls)

    assert type(restored) is module_cls
    x = jnp.ones(input_shape)
    np.testing.assert_allclose(np.asarray(model(x)), np.asarray(restored(x)), atol=0)


#


def test_from_checkpoint_raises_on_corrupt_archive() -> None:
    bogus = _SimpleMlpCheckpoint(
        archive=types.Blob.from_bytes(b"not a tar gz file"),
        config=_SimpleMlpConfig(hidden=8),
    )
    with pytest.raises(tarfile.ReadError):
        flax_nnx_checkpoint_util.from_checkpoint(bogus, _SimpleMlp)


def test_blob_roundtrips_json() -> None:
    payload = b"some tar.gz bytes \x00\x01\x02"
    b = types.Blob.from_bytes(payload, file_suffix=".tar.gz")
    parsed = json.loads(b.model_dump_json())
    assert parsed["content"] == base64.b64encode(payload).decode("ascii")
    assert parsed["file_suffix"] == ".tar.gz"
    assert parsed["content_length"] == len(payload)

    restored = types.Blob.model_validate_json(b.model_dump_json())
    assert restored.as_bytes() == payload
    assert restored.file_suffix == ".tar.gz"
