from __future__ import annotations

import functools
import itertools
import logging
import queue
import threading
import time
import typing
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Hashable, Literal

import datasets
import huggingface_hub
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import optax
import orjson
import pandas as pd
import pydantic
import shortuuid
from beartype.typing import Any
from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import vmec_utils
from flax import nnx

import wandb
from coilstellaration import (
    coilset_utils,
    data_utils,
    flax_nnx_checkpoint_util,
    types,
)
from coilstellaration.machine_learning import (
    model_definition,
)
from coilstellaration.types import NpOrJaxArray

logger = logging.getLogger(__name__)


T = typing.TypeVar("T", bound=pydantic.BaseModel)


Strata = Literal["loose", "medium", "tight"]
StratificationColumn = "benchmark/stratification"
StaticShapeTrackColumn = "benchmark/fixed_shape_track"
VariableShapeTrackColumn = "benchmark/variable_shape_track"


def get_unique_id() -> str:
    """Generate a unique ID string."""

    uuid_str = shortuuid.ShortUUID().uuid()
    return f"D{uuid_str}"


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
        ["json_constellaration_boundary"]
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

    merged_df = merged_df.dropna(
        axis=0, how="any", subset=list(JSON_COLUMNS + REQUIREMENT_METRICS_COLUMNS)
    )

    return merged_df


def load_benchmark_dataframe(
    track: Literal["fixed_shape", "variable_shape"],
    stratum: Literal["loose", "medium", "tight"] = "tight",
    split: Literal["train", "eval"] = "eval",
) -> pd.DataFrame:
    df = load_dataframes(split=split, filtered=False)

    track_column = (
        StaticShapeTrackColumn if track == "fixed_shape" else VariableShapeTrackColumn
    )

    filtered_df = df.loc[df[track_column] & (df[StratificationColumn] == stratum)]

    return filtered_df


def load_ml_baseline_dataframe(
    split: Literal["train", "eval"] = "eval",
) -> pd.DataFrame:
    df = load_dataframes(split=split, filtered=False)
    filtered_df = df.loc[df["baseline_labels/used_for_ml_baseline"].fillna(False)]
    return filtered_df


def load_dataset(
    df: pd.DataFrame,
    n: int = 0,
) -> list[types.EvalData]:
    """Materialize the deterministic eval split as per-sample lists."""
    eval_datas = []
    logger.info("Processing eval rows to build dataset artifacts...")
    for i, (_, row) in enumerate(df.iterrows()):
        if n > 0 and i >= n:
            break
        eval_data = types.EvalData(
            boundary=surface_rz_fourier.SurfaceRZFourier.model_validate(
                orjson.loads(row["json_constellaration_boundary"])
            ),
            boundary_id=row["constellaration_boundary_id"],
            true_coilset=types.Coilset.model_validate(
                orjson.loads(row["json_desc_coilset"])
            ),
            true_metrics=row["desc_metrics_id"],
            requirement_metrics=data_utils.row_to_requirement_metrics(row.to_dict()),
        )
        eval_datas.append(eval_data)
    logger.info("Assembled evaluation dataset with %d samples.", len(eval_datas))
    return eval_datas


def load_ml_baseline_dataset(
    split: Literal["train", "eval"] = "eval",
    n: int = 0,
) -> list[types.EvalData]:
    df = data_utils.load_ml_baseline_dataframe(split=split)
    eval_datas = load_dataset(df, n=n)
    return eval_datas


def load_benchmark_dataset(
    track: Literal["fixed_shape", "variable_shape"],
    stratum: Literal["loose", "medium", "tight"] = "tight",
    split: Literal["train", "eval"] = "eval",
    n: int = 0,
) -> list[types.EvalData]:
    df = data_utils.load_benchmark_dataframe(track=track, stratum=stratum, split=split)
    eval_datas = load_dataset(df, n=n)
    return eval_datas


def metrics_to_requirement_metrics(metrics: types.Metrics) -> types.RequirementMetrics:
    return types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=float(
            np.min(np.asarray(metrics.normalized_coil_to_coil_min_distances))
        ),
        min_normalized_coil_to_plasma_distance=float(
            np.min(np.asarray(metrics.normalized_coil_to_plasma_min_distances))
        ),
        max_normalized_coil_curvature=float(
            np.max(np.asarray(metrics.normalized_coil_curvatures))
        ),
        max_normalized_field_error=float(
            np.max(np.asarray(metrics.normalized_field_error))
        ),
    )


def row_to_requirement_metrics(
    row: Mapping[Hashable, float],
) -> types.RequirementMetrics:
    """Build a `RequirementMetrics` from `results.parquet`'s flattened columns."""
    values = [float(row[c]) for c in REQUIREMENT_METRICS_COLUMNS]
    return types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=values[0],
        min_normalized_coil_to_plasma_distance=values[1],
        max_normalized_coil_curvature=values[2],
        max_normalized_field_error=values[3],
    )
