from collections.abc import Iterable
from typing import Literal, TypeVar

import jax
import numpy as np
import pydantic

from coilstellaration.types import ScalarInt

Sampled = TypeVar("Sampled", bound=pydantic.BaseModel)
Kinds = Literal["int", "float", "choices", "none", "scalar"]


def sample_requirements(
    bounds: pydantic.BaseModel,
    sampled_cls: type[Sampled],
    seed: ScalarInt,
) -> Sampled:
    """Sample a ``sampled_cls`` instance from ``bounds``.

    Fields on ``bounds`` are interpreted as follows:
      * ``tuple[int, int]``: uniform integer in ``[low, high]``.
      * ``tuple[float, float]``: uniform float in ``[low, high)``.
      * ``frozenset``: uniform categorical pick.
      * ``None``: passed through as ``None``.
      * Any scalar: copied through unchanged.
    """
    classifications = _classify(bounds)
    if not any(kind in {"int", "float", "choices"} for _, kind, _ in classifications):
        raise ValueError(
            f"Bounds {type(bounds).__name__} has no sampleable fields "
            "(no numeric ranges or frozensets)."
        )

    rng = jax.random.PRNGKey(int(seed))
    kwargs: dict = {}
    for field_name, kind, value in classifications:
        if kind in {"int", "float", "choices"}:
            rng, subkey = jax.random.split(rng)
            kwargs[field_name] = _draw(kind, value, subkey)
        else:
            kwargs[field_name] = value

    return sampled_cls(**kwargs)


def _classify(
    bounds: pydantic.BaseModel,
) -> list[tuple[str, Kinds, object]]:
    """Return ``(name, kind, value)`` triples sorted by name.

    ``kind`` is one of ``"int"``, ``"float"``, ``"choices"``, ``"none"``,
    ``"scalar"``. Validation errors are raised here.
    """
    out: list[tuple[str, Kinds, object]] = []
    model_fields = type(bounds).model_fields
    for name in sorted(model_fields):
        if model_fields[name].init is False:
            continue
        value = getattr(bounds, name)
        kind = _kind_of(name, value)
        out.append((name, kind, value))
    return out


def _kind_of(name: str, value: object) -> Kinds:
    if value is None:
        return "none"
    if isinstance(value, frozenset):
        if not value:
            raise ValueError(f"Field {name!r}: empty frozenset is not sampleable.")
        return "choices"
    if isinstance(value, tuple):
        if len(value) != 2:
            raise TypeError(
                f"Field {name!r}: tuple must have length 2, got {len(value)}."
            )
        low, high = value
        if isinstance(low, bool) or isinstance(high, bool):
            raise TypeError(f"Field {name!r}: bool bounds not supported.")
        if isinstance(low, int) and isinstance(high, int):
            if low > high:
                raise ValueError(f"Field {name!r}: low > high ({low} > {high}).")
            return "int"
        if isinstance(low, float) and isinstance(high, float):
            if low > high:
                raise ValueError(f"Field {name!r}: low > high ({low} > {high}).")
            return "float"
        raise TypeError(
            f"Field {name!r}: tuple elements must be same numeric type "
            f"(mixed {type(low).__name__} / {type(high).__name__})."
        )
    return "scalar"


def _draw(kind: Kinds, value: object, key: jax.Array) -> object:
    match kind:
        case "int":
            low, high = value  # type: ignore[misc]
            return int(jax.random.randint(key, (), low, high + 1))
        case "float":
            low, high = value  # type: ignore[misc]
            return float(jax.random.uniform(key, (), minval=low, maxval=high))
        case "choices":
            choices: Iterable = sorted(value, key=lambda x: abs(hash(x)))  # type: ignore[arg-type]
            choices_arr = np.array(list(choices))
            idx = int(jax.random.randint(key, (), 0, len(choices_arr)))
            return choices_arr[idx].item()
        case _:
            raise AssertionError(f"unreachable kind {kind!r}")
