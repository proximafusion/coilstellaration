"""Integration test: the lean loader must match the upstream loader exactly.

`src.backend.data.load_benchmark_dataset` reorders the upstream load (filter +
cap before fetching coilset/boundary JSON) to avoid the eager full-corpus merge
that OOMs the 16Gi HF Space. It must return byte-for-byte equivalent
`EvalData`. This hits the network (public HF datasets), so it is `slow`:

    pytest -m slow tests/backend/test_data_integration.py     # run it
    pytest -m "not slow"                                      # skip it
"""
import pytest
from coilstellaration import data_utils

from src.backend import data as lean

# Cap instances per track: enough to exercise filtering/order, cheap to load.
N = 5


def _fingerprint(ed):
    """Exact, comparable signature of an EvalData (pydantic JSON is
    deterministic for equal values, so equal floats compare equal)."""
    return (
        ed.boundary_id,
        ed.true_metrics,
        ed.boundary.model_dump_json(),
        ed.true_coilset.model_dump_json(),
        ed.requirement_metrics.model_dump_json(),
    )


@pytest.mark.slow
@pytest.mark.parametrize("track", ["fixed_shape", "variable_shape"])
def test_lean_loader_matches_upstream(track):
    kwargs = dict(track=track, stratum="tight", split="eval", n=N)
    mine = lean.load_benchmark_dataset(**kwargs)
    ref = data_utils.load_benchmark_dataset(**kwargs)

    assert len(mine) == len(ref) == N
    # Same instances, same order, same parsed boundary/coilset/target metrics.
    assert [_fingerprint(e) for e in mine] == [_fingerprint(e) for e in ref]
