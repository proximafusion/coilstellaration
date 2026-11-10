from __future__ import annotations

import concurrent.futures
import functools
import logging
import pathlib
import time
from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import BinaryIO, Literal

import datasets
import huggingface_hub
import numpy as np
import orjson
import pandas as pd
import pydantic
import shortuuid
from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import vmec_utils

from coilstellaration import (
    data_utils,
    types,
)

logger = logging.getLogger(__name__)


Strata = Literal["loose", "medium", "tight"]
StratificationColumn = "benchmark/stratification"
StaticShapeTrackColumn = "benchmark/fixed_shape_track"
VariableShapeTrackColumn = "benchmark/variable_shape_track"

COILSTELLARATION_REPO = "proxima-fusion/coilstellaration"
COILSTELLARATION_REPO_REVISION = "f505cc890ff2f5cf24a6d68f4b5156d0a3fecb5c"
CONSTELLARATION_REPO = "proxima-fusion/constellaration"
CONSTELLARATION_REPO_REVISION = "3346beff09e2e57f2b568d49e08e899c82900367"


JSON_COLUMNS = (
    "json_desc_coilset",
    "json_constellaration_boundary",
)

REQUIREMENT_METRICS_COLUMNS = (
    "desc_metrics/normalized_coil_to_coil_min_distances/min",
    "desc_metrics/normalized_coil_to_plasma_min_distances/min",
    "desc_metrics/normalized_coil_curvatures/min",
    "desc_metrics/normalized_coil_curvatures/max",
    "desc_metrics/normalized_field_error/max",
)
"""Columns `row_to_requirement_metrics` reads.

The stored coil curvatures are signed (see `types.Metrics.coil_curvatures`), so the
curvature metric needs both their min and their max.
"""
N_COILS_COLUMN = "reqs/n_coils_per_half_period"
VMECPP_WOUT_ID_COLUMN = "vmecpp_wout_id"
DESC_EQUILIBRIUM_ID_COLUMN = "desc_equilibrium_id"
FIELD_ERROR_MEAN_COLUMN = "desc_metrics/normalized_field_error/mean"

_CONSTELLARATION_COLUMNS = (
    "plasma_config_id",
    "boundary.json",
    "misc.vmecpp_wout_id",
    "misc.desc_equilibrium_id",
    "metrics.aspect_ratio",
    "metrics.aspect_ratio_over_edge_rotational_transform",
    "metrics.average_triangularity",
    "metrics.axis_magnetic_mirror_ratio",
    "metrics.axis_rotational_transform_over_n_field_periods",
    "metrics.edge_magnetic_mirror_ratio",
    "metrics.edge_rotational_transform_over_n_field_periods",
    "metrics.flux_compression_in_regions_of_bad_curvature",
    "metrics.max_elongation",
    "metrics.minimum_normalized_magnetic_gradient_scale_length",
    "metrics.qi",
    "metrics.vacuum_well",
)
"""The ConStellaration `default` columns `load_dataframes` joins onto each result."""


def get_unique_id() -> str:
    """Generate a unique ID string."""

    uuid_str = shortuuid.ShortUUID().uuid()
    return f"D{uuid_str}"


def _read_shard_records(
    relative_path: str,
    ids: list[str],
    repo: str,
    revision: str,
    local_root: pathlib.Path | None,
    download: bool,
) -> pd.DataFrame:
    filters = [("id", "in", ids)]
    if download and local_root is None:
        path = huggingface_hub.hf_hub_download(
            repo, relative_path, repo_type="dataset", revision=revision
        )
        return pd.read_parquet(path, filters=filters, columns=["id", "json"])
    with _open_repo_file(relative_path, repo, revision, local_root) as f:
        return pd.read_parquet(f, filters=filters, columns=["id", "json"])


def _open_repo_file(
    relative_path: str, repo: str, revision: str, local_root: pathlib.Path | None
) -> BinaryIO:
    if local_root is not None:
        return (local_root / relative_path).open("rb")
    return huggingface_hub.HfFileSystem().open(  # pyright: ignore[reportReturnType]
        f"hf://datasets/{repo}/{relative_path}", "rb", revision=revision
    )


def load_objects_by_ids[T: pydantic.BaseModel](
    object_ids: Collection[str],
    cls: type[T],
    *,
    repo: str,
    revision: str,
    base_path: str = "vmecpp_wout",
    id_to_file_map_filename: str = "id_to_file_map.parquet",
    local_root: pathlib.Path | None = None,
    max_workers: int = 4,
    download_shards: bool = False,
) -> dict[str, T]:
    """Load objects from a Hugging Face dataset repo by ID, reading each shard once.

    Args:
        local_root: Read from this local copy of the repo instead of Hugging Face.
        download_shards: Download each shard into the Hugging Face cache, where later
            loads find it, instead of reading it remotely. Only worth it for small
            shards: a `vmecpp_wout` shard is 118 to 684 MB.

    Raises:
        KeyError: If an ID is not in the repo.
    """
    ids = sorted(set(object_ids))
    # `pyarrow` can't infer the type of an empty `in` filter.
    if not ids:
        return {}
    with _open_repo_file(
        f"{base_path}/{id_to_file_map_filename}", repo, revision, local_root
    ) as f:
        index_df = pd.read_parquet(
            f, filters=[("id", "in", ids)], columns=["id", "path"]
        )
    shard_paths, id_lists = zip(
        *((str(path), group.tolist()) for path, group in index_df.groupby("path")["id"])
    )
    read_shard = functools.partial(
        _read_shard_records,
        repo=repo,
        revision=revision,
        local_root=local_root,
        download=download_shards,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers) as pool:
        records = pd.concat(pool.map(read_shard, shard_paths, id_lists))
    missing = sorted(set(ids) - set(records["id"]))
    if missing:
        raise KeyError(f"{len(missing)} ids not found in {base_path}: {missing[:10]}")
    return {
        object_id: cls.model_validate_json(json_str)
        for object_id, json_str in zip(records["id"], records["json"], strict=True)
    }


def load_object_by_id[T: pydantic.BaseModel](
    object_id: str,
    cls: type[T],
    *,
    repo: str,
    revision: str,
    base_path: str = "vmecpp_wout",
    id_to_file_map_filename: str = "id_to_file_map.parquet",
    local_root: pathlib.Path | None = None,
) -> T:
    """Load an object from a Hugging Face dataset repo by its ID.

    Args:
        local_root: Read from this local copy of the repo instead of Hugging Face.
    """
    with _open_repo_file(
        f"{base_path}/{id_to_file_map_filename}", repo, revision, local_root
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
    with _open_repo_file(relative_path, repo, revision, local_root) as f:
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


def load_coilset_by_id(object_id: str) -> types.Coilset:
    return load_object_by_id(
        object_id,
        cls=types.Coilset,
        base_path="coilsets",
        repo=COILSTELLARATION_REPO,
        revision=COILSTELLARATION_REPO_REVISION,
    )


def load_coilsets_by_ids(object_ids: Collection[str]) -> dict[str, types.Coilset]:
    return load_objects_by_ids(
        object_ids,
        cls=types.Coilset,
        base_path="coilsets",
        repo=COILSTELLARATION_REPO,
        revision=COILSTELLARATION_REPO_REVISION,
    )


def load_requirements_by_id(object_id: str) -> types.Requirements:
    return load_object_by_id(
        object_id,
        cls=types.Requirements,
        base_path="requirements",
        repo=COILSTELLARATION_REPO,
        revision=COILSTELLARATION_REPO_REVISION,
    )


def load_requirements_by_ids(
    object_ids: Collection[str],
) -> dict[str, types.Requirements]:
    return load_objects_by_ids(
        object_ids,
        cls=types.Requirements,
        base_path="requirements",
        repo=COILSTELLARATION_REPO,
        revision=COILSTELLARATION_REPO_REVISION,
    )


def load_metrics_by_id(object_id: str) -> types.Metrics:
    return load_object_by_id(
        object_id,
        cls=types.Metrics,
        base_path="metrics",
        repo=COILSTELLARATION_REPO,
        revision=COILSTELLARATION_REPO_REVISION,
    )


def load_metrics_by_ids(object_ids: Collection[str]) -> dict[str, types.Metrics]:
    return load_objects_by_ids(
        object_ids,
        cls=types.Metrics,
        base_path="metrics",
        repo=COILSTELLARATION_REPO,
        revision=COILSTELLARATION_REPO_REVISION,
    )


def load_desc_equilibrium_by_id(object_id: str) -> types.DescEquilibrium:
    return load_object_by_id(
        object_id,
        cls=types.DescEquilibrium,
        base_path="desc_equilibrium",
        repo=CONSTELLARATION_REPO,
        revision=CONSTELLARATION_REPO_REVISION,
    )


def load_desc_equilibria_by_ids(
    object_ids: Collection[str],
) -> dict[str, types.DescEquilibrium]:
    return load_objects_by_ids(
        object_ids,
        cls=types.DescEquilibrium,
        base_path="desc_equilibrium",
        repo=CONSTELLARATION_REPO,
        revision=CONSTELLARATION_REPO_REVISION,
        # The 79 shards are 7 to 40 MB, and a benchmark stratum needs most of them.
        download_shards=True,
    )


def load_vmecpp_wout_by_id(object_id: str) -> vmec_utils.VmecppWOut:
    wout = load_object_by_id(
        object_id,
        cls=vmec_utils.VmecppWOut,
        base_path="vmecpp_wout",
        repo=CONSTELLARATION_REPO,
        revision=CONSTELLARATION_REPO_REVISION,
    )
    # Drop pydantic extras (e.g. `curlabel: []`) carried over from older
    # constellaration schemas. They aren't declared on vmecpp.VmecWOut, so
    # vmecpp.save() falls back to a default `dim_00000` of size 0 — which
    # NETCDF3_CLASSIC interprets as a second UNLIMITED dimension and rejects.
    if wout.__pydantic_extra__:
        wout.__pydantic_extra__.clear()
    return wout


def load_vmecpp_wouts_by_ids(
    object_ids: Collection[str],
) -> dict[str, vmec_utils.VmecppWOut]:
    wouts = load_objects_by_ids(
        object_ids,
        cls=vmec_utils.VmecppWOut,
        base_path="vmecpp_wout",
        repo=CONSTELLARATION_REPO,
        revision=CONSTELLARATION_REPO_REVISION,
    )
    # Drop pydantic extras (e.g. `curlabel: []`) carried over from older
    # constellaration schemas. They aren't declared on vmecpp.VmecWOut, so
    # vmecpp.save() falls back to a default `dim_00000` of size 0 — which
    # NETCDF3_CLASSIC interprets as a second UNLIMITED dimension and rejects.
    for wout in wouts.values():
        if wout.__pydantic_extra__:
            wout.__pydantic_extra__.clear()
    return wouts


def _load_hf_table(
    repo: str,
    revision: str,
    name: str,
    split: Literal["train", "eval"] | None = "train",
    filters: Sequence[tuple[str, str, object]] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    # Pass `filters` only when set: any explicit config kwarg, even `None`, moves the
    # unfiltered table to a new cache directory and rebuilds it.
    if filters is None:
        dataset = datasets.load_dataset(repo, name=name, split=split, revision=revision)
    else:
        dataset = datasets.load_dataset(
            repo,
            name=name,
            split=split,
            revision=revision,
            filters=list(filters),
            columns=None if columns is None else list(columns),
            verification_mode="no_checks",
        )
    if isinstance(dataset, datasets.DatasetDict):
        dataset = datasets.concatenate_datasets(list(dataset.values()))
    assert type(dataset) is datasets.Dataset
    df = dataset.to_pandas(batched=False)
    assert type(df) is pd.DataFrame
    return df


def _load_hf_rows_by_id(
    repo: str,
    revision: str,
    name: str,
    id_column: str,
    ids: pd.Series,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    # Sorted so the same rows always hash to the same cached filtered table.
    wanted = sorted(ids.dropna().unique())
    return _load_hf_table(
        repo, revision, name, filters=[(id_column, "in", wanted)], columns=columns
    )


def load_dataframes(
    split: Literal["train", "eval"] | None = None,
    filters: Sequence[tuple[str, str, object]] | None = None,
) -> pd.DataFrame:
    """The `results` rows of `split`, joined with the coilset, requirements and
    ConStellaration boundary each row references.

    Only the referenced coilsets, requirements and boundaries are loaded, so narrowing
    the rows with `filters` also shrinks what is read from those tables.

    Args:
        filters: Predicates on `results` columns in `pyarrow` filter form, e.g.
            `[(StratificationColumn, "==", "tight")]`.
    """
    results_df = _load_hf_table(
        COILSTELLARATION_REPO,
        COILSTELLARATION_REPO_REVISION,
        "results",
        split=split,
        filters=filters,
    ).rename(columns={"plasma_configuration_id": "constellaration_boundary_id"})
    coilsets_df = (
        _load_hf_rows_by_id(
            COILSTELLARATION_REPO,
            COILSTELLARATION_REPO_REVISION,
            "coilsets",
            id_column="id",
            ids=results_df["desc_coilset_id"],
        )
        .set_index("id")
        .rename(columns={"json": "json_desc_coilset"})
    )
    requirements_df = (
        _load_hf_rows_by_id(
            COILSTELLARATION_REPO,
            COILSTELLARATION_REPO_REVISION,
            "requirements",
            id_column="id",
            ids=results_df["requirements_id"],
        )
        .set_index("id")
        .rename(columns={"json": "json_requirements"})
    )
    constellaration_df = (
        _load_hf_rows_by_id(
            CONSTELLARATION_REPO,
            CONSTELLARATION_REPO_REVISION,
            "default",
            id_column="plasma_config_id",
            ids=results_df["constellaration_boundary_id"],
            columns=_CONSTELLARATION_COLUMNS,
        )
        .set_index("plasma_config_id")
        .rename(
            columns={
                "boundary.json": "json_constellaration_boundary",
                "misc.vmecpp_wout_id": VMECPP_WOUT_ID_COLUMN,
                "misc.desc_equilibrium_id": DESC_EQUILIBRIUM_ID_COLUMN,
            }
        )
        .rename(columns=lambda x: x.replace("metrics.", "constellaration_metrics/"))
    )
    merged_df = (
        results_df.join(coilsets_df, on="desc_coilset_id")
        .join(requirements_df, on="requirements_id")
        .join(constellaration_df, on="constellaration_boundary_id")
    )

    merged_df = merged_df.dropna(
        axis=0,
        how="any",
        subset=list(JSON_COLUMNS + REQUIREMENT_METRICS_COLUMNS + (N_COILS_COLUMN,)),
    ).sort_values(by=["constellaration_boundary_id"], kind="stable")

    return merged_df


def load_benchmark_dataframe(
    track: Literal["fixed_shape", "variable_shape"],
    stratum: Literal["loose", "medium", "tight"] = "tight",
    split: Literal["train", "eval"] = "eval",
) -> pd.DataFrame:
    match track:
        case "fixed_shape":
            track_column = StaticShapeTrackColumn
        case "variable_shape":
            track_column = VariableShapeTrackColumn
    return load_dataframes(
        split=split,
        filters=[(track_column, "==", True), (StratificationColumn, "==", stratum)],
    )


def select_ml_baseline_rows(
    df: pd.DataFrame,
    track: Literal["fixed_shape", "variable_shape"],
) -> pd.DataFrame:
    """Select the rows used to train ML baselines for the given track.

    Missing (`None`) label values count as `False`.
    """
    match track:
        case "fixed_shape":
            mask = _as_bool(df["baseline_labels/used_for_ml_baseline"])
        case "variable_shape":
            mask = (
                _as_bool(df[VariableShapeTrackColumn])
                & _as_bool(df["baseline_labels/meets_relative_error_thresholds"])
                & df["json_desc_coilset"].notna()
            )
    return df.loc[mask]


def validate_coil_counts(df: pd.DataFrame) -> None:
    """Check that decoded coilsets agree with the requirement and boundary columns.

    Raises:
        ValueError: If a row's number of unique coils differs from
            `reqs/n_coils_per_half_period`, or its coilset and boundary disagree on
            the number of field periods. The message lists the offending
            `desc_coilset_id`s.
    """
    bad_ids: list[str] = []
    for _, row in df.iterrows():
        coilset = orjson.loads(row["json_desc_coilset"])
        boundary = orjson.loads(row["json_constellaration_boundary"])
        n_coils = np.asarray(coilset["coil_x_n"]).shape[0]
        consistent = (
            n_coils == int(row[N_COILS_COLUMN])
            and coilset["n_field_periods"] == boundary["n_field_periods"]
        )
        if not consistent:
            bad_ids.append(str(row["desc_coilset_id"]))
    if bad_ids:
        raise ValueError(
            f"Inconsistent coil count or n_field_periods for {len(bad_ids)} rows; "
            f"desc_coilset_id: {bad_ids[:10]}"
        )


def load_ml_baseline_dataframe(
    track: Literal["fixed_shape", "variable_shape"] = "fixed_shape",
    split: Literal["train", "eval"] = "eval",
) -> pd.DataFrame:
    df = load_dataframes(
        split=split,
    )
    filtered_df = select_ml_baseline_rows(df, track=track)
    validate_coil_counts(filtered_df)
    return filtered_df


def sample_stratified_rows(
    df: pd.DataFrame, n_per_cell: int, seed: int
) -> pd.DataFrame:
    """Draw up to `n_per_cell` rows from each (NFP, coils-per-half-period) cell."""
    cell_columns = ["boundary.n_field_periods", N_COILS_COLUMN]
    sampled = df.groupby(cell_columns, group_keys=False, sort=True).apply(
        functools.partial(_sample_cell, n_per_cell=n_per_cell, seed=seed),
        include_groups=False,
    )
    return df.loc[df.index.isin(sampled.index)]


def _as_bool(column: pd.Series) -> pd.Series:
    return column.eq(True)


def load_dataset(
    df: pd.DataFrame,
    n: int = 0,
) -> list[types.EvalData]:
    """Materialize the deterministic eval split as per-sample lists."""
    if n > 0 and n < len(df):
        df = df.iloc[:n]

    eval_datas: list[types.EvalData] = []
    logger.info("Processing eval rows to build dataset artifacts...")
    load_start = time.perf_counter()

    equilibria = load_desc_equilibria_by_ids(
        df[DESC_EQUILIBRIUM_ID_COLUMN].dropna().unique()
    )

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
            desc_equilibrium=equilibria.get(row[DESC_EQUILIBRIUM_ID_COLUMN]),
        )

        eval_datas.append(eval_data)
    logger.info(
        "Assembled evaluation dataset with %d samples in %.1fs.",
        len(eval_datas),
        time.perf_counter() - load_start,
    )
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
        # The curvatures are signed by convexity: a plain max would miss the tightest
        # concave bend.
        max_normalized_coil_curvature=float(
            np.max(np.abs(np.asarray(metrics.normalized_coil_curvatures)))
        ),
        max_normalized_field_error=float(
            np.max(np.asarray(metrics.normalized_field_error))
        ),
        n_coils_per_half_period=int(np.asarray(metrics.coil_currents).size),
    )


def row_to_requirement_metrics(
    row: Mapping[Hashable, float],
) -> types.RequirementMetrics:
    """Build a `RequirementMetrics` from `results.parquet`'s flattened columns."""
    coil_to_coil, coil_to_plasma, curvature_min, curvature_max, field_error = (
        float(row[c]) for c in REQUIREMENT_METRICS_COLUMNS
    )
    return types.RequirementMetrics(
        min_normalized_coil_to_coil_distance=coil_to_coil,
        min_normalized_coil_to_plasma_distance=coil_to_plasma,
        # The stored curvatures are signed by convexity: the tightest bend may be the
        # most negative one.
        max_normalized_coil_curvature=max(curvature_max, -curvature_min),
        max_normalized_field_error=field_error,
        n_coils_per_half_period=int(row[N_COILS_COLUMN]),
    )


def _sample_cell(cell: pd.DataFrame, n_per_cell: int, seed: int) -> pd.DataFrame:
    return cell.sample(n=min(n_per_cell, len(cell)), random_state=seed)
