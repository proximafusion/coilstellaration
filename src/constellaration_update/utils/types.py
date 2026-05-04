"""Type aliases used across constellaration_update.

Re-exports `NpOrJaxArray` and `ScalarFloat` from `constellaration.utils.types`
to keep a single source of truth where one exists. Locally defines `ScalarInt`,
`Blob`, and `runtime_check_array_sizes`, which `constellaration.utils.types`
does not expose.
"""

import base64
from typing import Annotated

import jaxtyping as jt
import pydantic
from beartype import beartype
from constellaration.utils.types import NpOrJaxArray, ScalarFloat

__all__ = [
    "Blob",
    "NpOrJaxArray",
    "ScalarFloat",
    "ScalarInt",
    "runtime_check_array_sizes",
]

ScalarInt = int | jt.Int[NpOrJaxArray, " "]
"""Either a normal Python int or a scalar NpOrJaxArray."""


def _decode_blob(v: object) -> bytes:
    if isinstance(v, bytes):
        return v
    if isinstance(v, str):
        return base64.b64decode(v)
    raise TypeError(f"Cannot decode blob from {type(v).__name__}")


def _encode_blob(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


Blob = Annotated[
    bytes,
    pydantic.BeforeValidator(_decode_blob),
    pydantic.PlainSerializer(_encode_blob, return_type=str, when_used="json"),
]
"""Raw bytes that round-trip through pydantic JSON via base64.

In Python the field is plain `bytes`; in JSON it's a base64-encoded string.
Constructors accept raw bytes (untouched) or a base64 string (decoded).
"""


def runtime_check_array_sizes(f):
    """Decorator to enforce jaxtyping shape annotations at runtime."""
    return jt.jaxtyped(typechecker=beartype)(f)
