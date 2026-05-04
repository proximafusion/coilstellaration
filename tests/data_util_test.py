"""Tests for `constellaration_update.data_util`."""

import pydantic
import pytest

from constellaration_update import data_util
from constellaration_update.utils.types import Blob


@pytest.fixture(autouse=True)
def _redirect_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv(data_util._DATA_ROOT_ENV, str(tmp_path))


class _Inner(pydantic.BaseModel):
    label: str


class _Sample(pydantic.BaseModel):
    name: str
    values: list[float]
    inner: _Inner


class _WithBinary(pydantic.BaseModel):
    archive: Blob


def test_write_returns_dapper_shaped_id():
    object_id = data_util.write(
        _Sample(name="x", values=[1.0], inner=_Inner(label="a"))
    )

    assert object_id.startswith("D")
    assert len(object_id) == 23


def test_write_then_read_roundtrips_pydantic_with_nested_model():
    original = _Sample(name="x", values=[1.0, 2.0, 3.0], inner=_Inner(label="leaf"))

    object_id = data_util.write(original)
    reloaded = data_util.read(_Sample, object_id)

    assert reloaded == original


def test_write_then_read_roundtrips_binary_via_base64_bytes():
    original = _WithBinary(archive=b"\x00\x01\x02 not utf-8 \xff")

    object_id = data_util.write(original)
    reloaded = data_util.read(_WithBinary, object_id)

    assert reloaded.archive == original.archive


def test_write_assigns_distinct_ids():
    a = data_util.write(_Sample(name="a", values=[], inner=_Inner(label="")))
    b = data_util.write(_Sample(name="b", values=[], inner=_Inner(label="")))

    assert a != b


def test_write_persists_human_readable_json():
    object_id = data_util.write(
        _Sample(name="hello", values=[1.5], inner=_Inner(label="leaf"))
    )

    on_disk = (data_util._data_root() / f"{object_id}.json").read_text()
    assert '"name":"hello"' in on_disk
    assert '"label":"leaf"' in on_disk


def test_read_missing_id_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        data_util.read(_Sample, "Dnonexistent000000000000")


def test_data_root_env_var_redirects_storage(monkeypatch: pytest.MonkeyPatch, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv(data_util._DATA_ROOT_ENV, str(elsewhere))

    object_id = data_util.write(_Sample(name="x", values=[], inner=_Inner(label="")))

    assert (elsewhere / f"{object_id}.json").exists()
