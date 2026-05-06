from __future__ import annotations

import functools
import logging
import typing
from collections.abc import Mapping
from typing import Any, Literal

import datasets
import huggingface_hub
import pandas as pd
import pydantic
from constellaration.mhd import vmec_utils

from coilstellaration import (
    types,
)

logger = logging.getLogger(__name__)


T = typing.TypeVar("T", bound=pydantic.BaseModel)


def load_object_by_id(
    object_id: str,
    cls: type[T],
    *,
    repo: str = "proxima-fusion/constellaration",
    base_path: str = "vmecpp_wout",
    id_to_file_map_filename: str = "id_to_file_map.parquet",
) -> T:
    fs = huggingface_hub.HfFileSystem()
    with fs.open(
        f"hf://datasets/{repo}/{base_path}/{id_to_file_map_filename}", "rb"
    ) as f:
        index_df = pd.read_parquet(
            f,  # pyright: ignore[reportArgumentType]
            filters=[("id", "=", object_id)],
            columns=["id", "path"],
        )
        index_matches = index_df.loc[index_df["id"] == object_id]
        if index_matches.empty:
            raise KeyError(f"id {object_id!r} not found in ID to path mapping.")

        relative_path = str(index_matches.iloc[0]["path"])
    with fs.open(f"hf://datasets/{repo}/{relative_path}", "rb") as f:
        record_df = pd.read_parquet(
            f,  # pyright: ignore[reportArgumentType]
            filters=[("id", "=", object_id)],
            columns=["id", "json"],
        )

        record_matches = record_df.loc[record_df["id"] == object_id]
        if record_matches.empty:
            raise KeyError(
                f"id {object_id!r} listed in index but missing from {relative_path}"
            )
        return cls.model_validate_json(record_matches.iloc[0]["json"])


load_coilset_by_id = functools.partial(
    load_object_by_id,
    cls=types.Coilset,
    base_path="coilsets",
    repo="proxima-fusion/coilstellaration",
)
load_requirements_by_id = functools.partial(
    load_object_by_id,
    cls=types.Requirements,
    base_path="requirements",
    repo="proxima-fusion/coilstellaration",
)
load_metrics_by_id = functools.partial(
    load_object_by_id,
    cls=types.Metrics,
    base_path="metrics",
    repo="proxima-fusion/coilstellaration",
)


def load_vmecpp_wout_by_id(object_id: str) -> vmec_utils.VmecppWOut:
    wout = load_object_by_id(
        object_id,
        cls=vmec_utils.VmecppWOut,
        base_path="vmecpp_wout",
        repo="proxima-fusion/constellaration",
    )
    # Drop pydantic extras (e.g. `curlabel: []`) carried over from older
    # constellaration schemas. They aren't declared on vmecpp.VmecWOut, so
    # vmecpp.save() falls back to a default `dim_00000` of size 0 — which
    # NETCDF3_CLASSIC interprets as a second UNLIMITED dimension and rejects.
    if wout.__pydantic_extra__:
        wout.__pydantic_extra__.clear()
    return wout


JSON_COLUMNS = (
    "json_desc_coilset",
    "json_constellaration_boundary",
)

REQUIREMENT_METRICS_COLUMNS = (
    "desc_metrics/normalized_coil_to_coil_min_distances/min",
    "desc_metrics/normalized_coil_to_plasma_min_distances/min",
    "desc_metrics/normalized_coil_curvatures/max",
    "desc_metrics/normalized_field_error/max",
)
FIELD_ERROR_MEAN_COLUMN = "desc_metrics/normalized_field_error/mean"


def load_dataframes(
    split: Literal["train", "eval"] | None = None,
    filtered: bool = False,
) -> pd.DataFrame:
    constellaration_repo = "proxima-fusion/constellaration"
    coilstellaration_repo = "proxima-fusion/coilstellaration"

    constellaration_dataset = datasets.load_dataset(
        constellaration_repo, name="default", split="train"
    )
    assert type(constellaration_dataset) is datasets.Dataset
    _constellaration_update_dataset = datasets.load_dataset(
        coilstellaration_repo, split=split
    )
    if isinstance(_constellaration_update_dataset, datasets.DatasetDict):
        constellaration_update_dataset = datasets.concatenate_datasets(
            list(_constellaration_update_dataset.values())
        )
    else:
        assert type(_constellaration_update_dataset) is datasets.Dataset
        constellaration_update_dataset = _constellaration_update_dataset
    _constellaration_update_coilsets = datasets.load_dataset(
        coilstellaration_repo, name="coilsets", split="train"
    )
    assert type(_constellaration_update_coilsets) is datasets.Dataset
    constellaration_update_coilsets = _constellaration_update_coilsets
    _constellaration_update_requirements = datasets.load_dataset(
        coilstellaration_repo, name="requirements", split="train"
    )
    assert type(_constellaration_update_requirements) is datasets.Dataset
    constellaration_update_requirements = _constellaration_update_requirements

    _results_df = constellaration_update_dataset.to_pandas(batched=False)
    assert type(_results_df) is pd.DataFrame
    results_df = _results_df.rename(
        columns={"plasma_configuration_id": "constellaration_boundary_id"}
    )
    _coilsets_df = constellaration_update_coilsets.to_pandas(batched=False)
    assert type(_coilsets_df) is pd.DataFrame
    coilsets_df = _coilsets_df.set_index("id").rename(
        columns={"json": "json_desc_coilset"}
    )
    _requirements_df = constellaration_update_requirements.to_pandas(batched=False)
    assert type(_requirements_df) is pd.DataFrame
    requirements_df = _requirements_df.set_index("id").rename(
        columns={"json": "json_requirements"}
    )
    _constellaration_df = constellaration_dataset.to_pandas(batched=False)
    assert type(_constellaration_df) is pd.DataFrame
    constellaration_df = _constellaration_df.set_index("plasma_config_id").rename(
        columns={"boundary.json": "json_constellaration_boundary"}
    )
    constellaration_df = constellaration_df[
        ["json_constellaration_boundary", "boundary.n_field_periods", "vmecpp_wout_id"]
        + [
            col
            for col in constellaration_df.columns
            if col.startswith("metric") and col not in {"metrics.id", "metrics.json"}
        ]
    ].rename(columns=lambda x: x.replace("metrics.", "constellaration_metrics/"))
    merged_df = (
        results_df.join(coilsets_df, on="desc_coilset_id")
        .join(requirements_df, on="requirements_id")
        .join(constellaration_df, on="constellaration_boundary_id")
    )

    if not filtered:
        return merged_df

    filtered_df = merged_df.loc[
        merged_df["additional_label/is_suitable_for_ml_baseline"].fillna(False)
    ].dropna(axis=0, how="any", subset=list(JSON_COLUMNS + REQUIREMENT_METRICS_COLUMNS))

    return filtered_df


def row_to_requirement_metrics(
    row: Mapping[str, Any],
) -> types.RequirementMetrics:
    """Build a `RequirementMetrics` from `results.parquet`'s flattened columns."""
    values = [float(row[c]) for c in REQUIREMENT_METRICS_COLUMNS]
    return types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=values[0],
        min_normalized_coil_to_plasma_distance=values[1],
        max_normalized_coil_curvature=values[2],
        max_normalized_field_error=values[3],
    )
