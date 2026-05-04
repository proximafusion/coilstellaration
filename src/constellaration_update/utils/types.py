"""Type aliases used across constellaration_update.

Re-exports `NpOrJaxArray` and `ScalarFloat` from `constellaration.utils.types`
to keep a single source of truth where one exists. Locally defines `ScalarInt`
and `runtime_check_array_sizes`, which `constellaration.utils.types` does not
expose.
"""

import jaxtyping as jt
from beartype import beartype
from constellaration.utils.types import NpOrJaxArray, ScalarFloat

__all__ = [
    "NpOrJaxArray",
    "ScalarFloat",
    "ScalarInt",
    "runtime_check_array_sizes",
]

ScalarInt = int | jt.Int[NpOrJaxArray, " "]
"""Either a normal Python int or a scalar NpOrJaxArray."""


def runtime_check_array_sizes(f):
    """Decorator to enforce jaxtyping shape annotations at runtime."""
    return jt.jaxtyped(typechecker=beartype)(f)
