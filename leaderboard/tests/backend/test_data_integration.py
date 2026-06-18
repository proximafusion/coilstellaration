import pytest
from coilstellaration import data_utils

from src.backend import data as lean


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
@pytest.mark.parametrize("N", [5, 0])
@pytest.mark.parametrize("track", ["fixed_shape", "variable_shape"])
@pytest.mark.parametrize("split", ["eval"]) # train is too large
@pytest.mark.parametrize("stratum", ["loose", "medium", "tight"])
def test_lean_loader_matches_upstream(track, N, split, stratum):
    kwargs = dict(track=track, stratum=stratum, split=split, n=N)
    mine = lean.load_benchmark_dataset(**kwargs)
    ref = data_utils.load_benchmark_dataset(**kwargs)

    if N > 0:
        assert len(mine) == N
    assert len(mine) == len(ref)
    assert [_fingerprint(e) for e in mine] == [_fingerprint(e) for e in ref]
