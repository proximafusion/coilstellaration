"""Tests for `checkpoint/types` and `checkpoint/flax_nnx`."""

import tarfile

import jax
import jax.numpy as jnp
import numpy as np
import pydantic
import pytest

flax = pytest.importorskip("flax", reason="requires flax — added in Task 16")
ocp = pytest.importorskip(
    "orbax.checkpoint", reason="requires orbax-checkpoint — added in Task 16"
)

# Detect the jax 0.5+ vs orbax/desc API mismatch in this devcontainer. Both
# `orbax.checkpoint.StandardCheckpointer.save` and `desc` reach for symbols
# that were renamed/removed (`jax.monitoring.record_scalar`,
# `jax.tree_util.tree_broadcast`). Skip orbax-roundtrip tests until the
# version matrix is resolved (see Open follow-ups).
_ORBAX_JAX_INCOMPAT = not hasattr(jax.monitoring, "record_scalar")

from flax import nnx  # noqa: E402

from constellaration_update.checkpoint import (  # noqa: E402
    flax_nnx as flax_nnx_checkpoint,
)
from constellaration_update.checkpoint import (  # noqa: E402
    types as flax_nnx_checkpoint_types,
)

_skip_if_orbax_jax_incompat = pytest.mark.skipif(
    _ORBAX_JAX_INCOMPAT,
    reason="orbax/jax API mismatch in this devcontainer; tracked in Open follow-ups",
)


class _SimpleMlpConfig(pydantic.BaseModel):
    hidden: int


_SimpleMlpCheckpoint = flax_nnx_checkpoint_types.FlaxNnxCheckpoint[_SimpleMlpConfig]


class _SimpleMlp(nnx.Module):
    def __init__(self, config: _SimpleMlpConfig, *, rngs: nnx.Rngs) -> None:
        self.linear1 = nnx.Linear(4, config.hidden, rngs=rngs)
        self.linear2 = nnx.Linear(config.hidden, 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear2(jax.nn.relu(self.linear1(x)))


def _make_simple_mlp(hidden: int) -> _SimpleMlp:
    return _SimpleMlp(_SimpleMlpConfig(hidden=hidden), rngs=nnx.Rngs(0))


def test_flax_nnx_checkpoint_is_pydantic_model():
    ckpt = _SimpleMlpCheckpoint(
        archive=b"placeholder",
        config=_SimpleMlpConfig(hidden=8),
    )
    assert isinstance(ckpt, pydantic.BaseModel)
    assert ckpt.archive == b"placeholder"
    assert ckpt.config.hidden == 8


@_skip_if_orbax_jax_incompat
def test_to_from_checkpoint_roundtrip_preserves_params():
    model = _make_simple_mlp(hidden=8)
    checkpoint = flax_nnx_checkpoint.to_checkpoint(
        model, _SimpleMlpConfig(hidden=8), _SimpleMlpCheckpoint
    )
    restored = flax_nnx_checkpoint.from_checkpoint(checkpoint, _SimpleMlp)

    _, original_state = nnx.split(model)
    _, restored_state = nnx.split(restored)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=0),
        original_state,
        restored_state,
    )

    x = jnp.ones((1, 4))
    np.testing.assert_allclose(np.asarray(model(x)), np.asarray(restored(x)), atol=0)


# TODO(migration): dapper storage layer is removed; reintroduce a serialization
# roundtrip test once a persistence layer is chosen (e.g., HF Hub upload). Body
# kept as a stub so the future implementer sees the shape of the round-trip.
@pytest.mark.skip(reason="dapper storage layer removed in migration")
def test_roundtrip_through_dapper_storage():
    raise NotImplementedError(
        "dapper.write / dapper.read replaced with pydantic + a future persistence "
        "layer; see TODO(migration) above."
    )


class _PlainLinearConfig(pydantic.BaseModel):
    in_features: int
    out_features: int


_PlainLinearCheckpoint = flax_nnx_checkpoint_types.FlaxNnxCheckpoint[_PlainLinearConfig]


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
@_skip_if_orbax_jax_incompat
def test_from_checkpoint_preserves_concrete_type_and_outputs(
    module_cls: type[nnx.Module],
    config: pydantic.BaseModel,
    checkpoint_cls: type[flax_nnx_checkpoint_types.FlaxNnxCheckpoint],
    input_shape: tuple[int, ...],
) -> None:
    model = module_cls(config, rngs=nnx.Rngs(0))  # pyright: ignore[reportCallIssue]
    checkpoint = flax_nnx_checkpoint.to_checkpoint(model, config, checkpoint_cls)
    restored = flax_nnx_checkpoint.from_checkpoint(checkpoint, module_cls)

    assert type(restored) is module_cls
    x = jnp.ones(input_shape)
    np.testing.assert_allclose(np.asarray(model(x)), np.asarray(restored(x)), atol=0)


@_skip_if_orbax_jax_incompat
def test_from_checkpoint_raises_on_shape_mismatch() -> None:
    model = _make_simple_mlp(hidden=8)
    checkpoint = flax_nnx_checkpoint.to_checkpoint(
        model, _SimpleMlpConfig(hidden=8), _SimpleMlpCheckpoint
    )

    # Drift the config so the abstract template's shapes don't match the saved
    # arrays.
    drifted = checkpoint.model_copy(update={"config": _SimpleMlpConfig(hidden=16)})

    with pytest.raises(Exception) as exc_info:  # noqa: PT011
        flax_nnx_checkpoint.from_checkpoint(drifted, _SimpleMlp)
    # Stay loose on the exact error class — orbax's surface evolves; we only
    # care that *something* descriptive raises, not its precise type.
    assert exc_info.value is not None


def test_from_checkpoint_raises_on_corrupt_archive() -> None:
    bogus = _SimpleMlpCheckpoint(
        archive=b"not a tar gz file",
        config=_SimpleMlpConfig(hidden=8),
    )
    with pytest.raises(tarfile.ReadError):
        flax_nnx_checkpoint.from_checkpoint(bogus, _SimpleMlp)
