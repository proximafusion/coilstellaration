"""Tests for the generic-template sampling_utils rewrite."""

import pydantic
import pytest

from coilstellaration import sampling_utils


class _ProbeTemplate[Numeric, Categorical](pydantic.BaseModel):
    a: Numeric
    b: Categorical


_ProbeSampled = _ProbeTemplate[int, str]
_ProbeBounds = _ProbeTemplate[tuple[int, int], frozenset[str]]


def test_template_parametrizes_into_distinct_classes() -> None:
    assert _ProbeSampled is not _ProbeBounds
    sampled = _ProbeSampled(a=3, b="x")
    bounds = _ProbeBounds(a=(1, 5), b=frozenset({"x", "y"}))
    assert sampled.a == 3
    assert bounds.a == (1, 5)
    assert bounds.b == frozenset({"x", "y"})


def test_template_parametrized_instances_freeze() -> None:
    bounds = _ProbeBounds(a=(1, 5), b=frozenset({"x"}))
    with pytest.raises(Exception):
        bounds.a = (2, 6)


class _TinyTemplate[Numeric, Categorical](pydantic.BaseModel):
    x: Numeric
    y: Categorical


_TinySampled = _TinyTemplate[int | float | str, int | float | str]
_TinyBounds = _TinyTemplate[
    tuple[int | float, int | float] | int | float,
    frozenset[str] | str,
]


def test_sample_requirements_empty_frozenset_raises() -> None:
    bounds = _TinyBounds(x=(0, 1), y=frozenset())
    with pytest.raises(ValueError, match="empty"):
        sampling_utils.sample_requirements(bounds, _TinySampled, seed=0)


def test_sample_requirements_low_greater_than_high_raises() -> None:
    bounds = _TinyBounds(x=(5, 1), y=frozenset({"a"}))
    with pytest.raises(ValueError, match="low > high|low must be"):
        sampling_utils.sample_requirements(bounds, _TinySampled, seed=0)


def test_sample_requirements_mixed_numeric_tuple_raises() -> None:
    bounds = _TinyBounds(x=(1, 2.0), y=frozenset({"a"}))
    with pytest.raises(TypeError, match="mixed|same type"):
        sampling_utils.sample_requirements(bounds, _TinySampled, seed=0)


def test_sample_requirements_no_sampleable_fields_raises() -> None:
    bounds = _TinyBounds(x=3, y="a")
    with pytest.raises(ValueError, match="no sampleable|no fields"):
        sampling_utils.sample_requirements(bounds, _TinySampled, seed=0)


# ---- Task 4: Int-range endpoints -------------------------------------------


def test_sample_requirements_int_range_hits_both_endpoints() -> None:
    bounds = _TinyBounds(x=(1, 3), y=frozenset({"a"}))
    seen: set[int] = set()
    for seed in range(200):
        sampled = sampling_utils.sample_requirements(bounds, _TinySampled, seed=seed)
        assert isinstance(sampled.x, int)
        assert 1 <= sampled.x <= 3
        seen.add(sampled.x)
    assert seen == {1, 2, 3}, f"expected all of 1,2,3 to appear, got {seen}"


# ---- Task 5: Float / categorical / None / scalar pass-through -------------


@pytest.mark.parametrize("seed", list(range(20)))
def test_sample_requirements_float_in_range(seed: int) -> None:
    bounds = _TinyBounds(x=(0.5, 1.5), y=frozenset({"a"}))
    sampled = sampling_utils.sample_requirements(bounds, _TinySampled, seed=seed)
    assert isinstance(sampled.x, float)
    assert 0.5 <= sampled.x < 1.5


def test_sample_requirements_categorical_in_choice_set() -> None:
    choices = frozenset({"red", "green", "blue"})
    bounds = _TinyBounds(x=(0, 1), y=choices)
    for seed in range(50):
        sampled = sampling_utils.sample_requirements(bounds, _TinySampled, seed=seed)
        assert sampled.y in choices


def test_sample_requirements_none_passes_through() -> None:
    class _OptTemplate[Numeric, Categorical](pydantic.BaseModel):
        x: Numeric
        y: Categorical

    _OptSampled = _OptTemplate[int | float | None, str]
    _OptBounds = _OptTemplate[tuple[int, int] | int | None, frozenset[str] | str]
    bounds = _OptBounds(x=None, y=frozenset({"a"}))
    sampled = sampling_utils.sample_requirements(bounds, _OptSampled, seed=0)
    assert sampled.x is None


def test_sample_requirements_scalar_passes_through() -> None:
    bounds = _TinyBounds(x=42, y=frozenset({"a"}))
    sampled = sampling_utils.sample_requirements(bounds, _TinySampled, seed=0)
    assert sampled.x == 42


# ---- Task 6: Determinism --------------------------------------------------


def test_sample_requirements_same_seed_same_output() -> None:
    bounds = _TinyBounds(x=(0.0, 1.0), y=frozenset({"a", "b", "c"}))
    a = sampling_utils.sample_requirements(bounds, _TinySampled, seed=7)
    b = sampling_utils.sample_requirements(bounds, _TinySampled, seed=7)
    assert a.x == b.x
    assert a.y == b.y


def test_sample_requirements_different_seeds_differ() -> None:
    bounds = _TinyBounds(x=(0.0, 1.0), y=frozenset({"a", "b", "c"}))
    a = sampling_utils.sample_requirements(bounds, _TinySampled, seed=1)
    b = sampling_utils.sample_requirements(bounds, _TinySampled, seed=2)
    assert a.x != b.x or a.y != b.y


# ---- Task 7: Uniformity ---------------------------------------------------

_UNIFORM_CHI_SQ_THRESHOLD = 30.0


def _chi_square(counts: list[int], expected: float) -> float:
    return sum((c - expected) ** 2 / expected for c in counts)


def test_sample_requirements_int_uniform_over_range() -> None:
    bounds = _TinyBounds(x=(1, 4), y=frozenset({"a"}))
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    n = 1_000
    for seed in range(n):
        sampled = sampling_utils.sample_requirements(bounds, _TinySampled, seed=seed)
        counts[int(sampled.x)] += 1
    stat = _chi_square(list(counts.values()), n / 4)
    assert stat < _UNIFORM_CHI_SQ_THRESHOLD, f"int uniformity failed: {counts}"


def test_sample_requirements_categorical_uniform_over_choices() -> None:
    choices = sorted({"a", "b", "c", "d", "e"})
    bounds = _TinyBounds(x=(0, 1), y=frozenset(choices))
    counts = dict.fromkeys(choices, 0)
    n = 1_000
    for seed in range(n):
        sampled = sampling_utils.sample_requirements(bounds, _TinySampled, seed=seed)
        assert isinstance(sampled.y, str)
        counts[sampled.y] += 1
    stat = _chi_square(list(counts.values()), n / 5)
    assert stat < _UNIFORM_CHI_SQ_THRESHOLD, f"categorical uniformity failed: {counts}"


def test_sample_requirements_float_uniform_over_range() -> None:
    bounds = _TinyBounds(x=(0.0, 1.0), y=frozenset({"a"}))
    buckets = [0] * 10
    n = 1_000
    for seed in range(n):
        sampled = sampling_utils.sample_requirements(bounds, _TinySampled, seed=seed)
        idx = min(int(float(sampled.x) * 10), 9)
        buckets[idx] += 1
    stat = _chi_square(buckets, n / 10)
    assert stat < _UNIFORM_CHI_SQ_THRESHOLD, f"float uniformity failed: {buckets}"
