"""Unit tests for `types`."""

import jax
import jax.numpy as jnp

from coilstellaration import types


def _requirements(n_coils: int) -> types.RequirementMetrics:
    return types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=0.4,
        min_normalized_coil_to_plasma_distance=1.2,
        max_normalized_coil_curvature=3.8,
        max_normalized_field_error=0.01,
        n_coils_per_half_period=n_coils,
    )


def test_to_array_excludes_coil_count() -> None:
    assert _requirements(6).to_array().shape == (4,)


def test_requirement_metrics_json_and_pytree_roundtrip() -> None:
    a = _requirements(5)
    restored = types.RequirementMetrics.model_validate_json(a.model_dump_json())
    assert int(restored.n_coils_per_half_period) == 5

    stacked = jax.tree.map(lambda *x: jnp.stack(x), a, _requirements(6))
    assert stacked.n_coils_per_half_period.shape == (2,)
