# pyright: reportAttributeAccessIssue=false
import hashlib
import logging
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pf
import pyarrow.parquet as pq
from datasets import Dataset as HfDataset
from datasets import load_dataset
from etils import epath
from huggingface_hub import HfApi, login
from huggingface_hub.errors import RepositoryNotFoundError

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

base_paths = list(
    map(
        epath.Path,
        (
            # Combined results from multiple runs:
            # - constellaration_update_dataproc:260423_163621
            # - constellaration_update_dataproc:260425_105615
            # - constellaration_update_dataproc:260425_121756
            # - constellaration_update_dataproc:260428_153454
            # - constellaration_update_dataproc:260430_191516
            "gs://proxima-data-repository/dataproc/constellaration_update/dumps/PROD/1777711892-jhewers/",
        ),
    )
)


sidecar_id_columns_in_results: dict[str, list[str]] = {
    # "metrics": ["regcoil_metrics_id", "desc_metrics_id"],
    # "requirements": ["requirements_id"],
    # "coilsets": ["regcoil_coilset_id", "desc_coilset_id"],
}

repo_id = "proxima-fusion/coilstellaration"
staging_dir = Path("/home/devuser/tmp/outputs/constellaration_update_staging/upload")
shutil.rmtree(staging_dir, ignore_errors=True)
staging_dir.mkdir(parents=True, exist_ok=True)
gcs = pf.GcsFileSystem()

try:
    api = HfApi()
    api.repo_info(repo_id, repo_type="dataset")
except RepositoryNotFoundError:
    login()
    api = HfApi()
    api.repo_info(repo_id, repo_type="dataset")

# for path in ["results"] + list(sidecar_id_columns_in_results.keys()):
#     logger.info(f"Deleting {path} from the HF repo...")
#     api.delete_folder(
#         repo_id=repo_id,
#         repo_type="dataset",
#         path_in_repo=path,
#         commit_message=f"Delete {path} for re-upload",
#     )


def _read_dataset(name: str) -> ds.Dataset:
    src_uris = [str((p / name).with_suffix(".parquet")) for p in base_paths]
    shards = []
    for uri in src_uris:
        logger.info(f"  source {uri}")
        shards.append(ds.dataset(uri, format="parquet"))
    return ds.dataset(shards)


BATCH_SIZE = 50_000
MAX_ROWS_PER_FILE = 100_000
MAX_SIZE_PER_FILE = 1000 * 1024 * 1024  # 1 GiB
EVAL_FRACTION = 0.05
SPLIT_KEY_COLUMN = "constellaration_boundary_id"


def _hash_to_eval_mask(keys: pa.Array, eval_fraction: float) -> pa.Array:
    threshold = int(eval_fraction * 2**32)
    return pa.array(
        [
            int.from_bytes(hashlib.md5(k.encode()).digest()[:4], "big") < threshold
            for k in keys.to_pylist()
        ],
        type=pa.bool_(),
    )


def stage_results() -> dict[str, pa.Array]:
    logger.info("Staging results...")
    results_table = (
        _read_dataset("results")
        .filter(~ds.field("regcoil_coilset_id").is_null())
        .to_table()
    )

    constellaration_dataset = load_dataset(
        "proxima-fusion/constellaration",
        "default",
        split="train",
    )
    assert isinstance(
        constellaration_dataset, HfDataset
    ), "Expected a single split in the constellaration dataset"

    constellaration_table = constellaration_dataset.data.table.select(
        ["plasma_config_id", "boundary.n_field_periods"]
    )

    joined = results_table.join(
        constellaration_table,
        keys="constellaration_boundary_id",
        right_keys="plasma_config_id",
        join_type="left outer",
    )

    is_correct_shape_for_ml_baseline = pc.and_(
        pc.equal(joined.column("boundary.n_field_periods"), 3),
        pc.equal(joined.column("reqs/n_coils_per_half_period"), 5),
    )
    rel_normalized_coil_to_coil_min_distances_error = pc.divide(
        pc.subtract(
            joined.column("desc_metrics/normalized_coil_to_coil_min_distances/min"),
            joined.column("reqs/normalized_min_coil_to_coil_distance"),
        ),
        joined.column("reqs/normalized_min_coil_to_coil_distance"),
    )

    rel_normalized_min_coil_to_plasma_distance_error = pc.divide(
        pc.subtract(
            joined.column("desc_metrics/normalized_coil_to_plasma_min_distances/min"),
            joined.column("reqs/normalized_min_coil_plasma_distance"),
        ),
        joined.column("reqs/normalized_min_coil_plasma_distance"),
    )

    rel_normalized_max_coil_curvature_error = pc.divide(
        pc.subtract(
            joined.column("desc_metrics/normalized_coil_curvatures/max"),
            joined.column("reqs/normalized_max_coil_curvature"),
        ),
        joined.column("reqs/normalized_max_coil_curvature"),
    )
    meets_relative_error_thresholds = pc.and_(
        pc.and_(
            pc.less_equal(
                -0.5,
                rel_normalized_coil_to_coil_min_distances_error,
            ),
            pc.less_equal(
                -0.5,
                rel_normalized_min_coil_to_plasma_distance_error,
            ),
        ),
        pc.less_equal(rel_normalized_max_coil_curvature_error, 2.5),
    )
    is_suitable_for_ml_baseline = pc.and_(
        is_correct_shape_for_ml_baseline, meets_relative_error_thresholds
    )

    for key, value in [
        (
            "is_suitable_for_ml_baseline",
            is_suitable_for_ml_baseline,
        ),
        ("meets_relative_error_thresholds", meets_relative_error_thresholds),
        ("is_correct_shape_for_ml_baseline", is_correct_shape_for_ml_baseline),
        (
            "rel_normalized_coil_to_coil_min_distances_error",
            rel_normalized_coil_to_coil_min_distances_error,
        ),
        (
            "rel_normalized_min_coil_to_plasma_distance_error",
            rel_normalized_min_coil_to_plasma_distance_error,
        ),
        (
            "rel_normalized_max_coil_curvature_error",
            rel_normalized_max_coil_curvature_error,
        ),
    ]:
        results_table = results_table.append_column(f"additional_label/{key}", value)

    keys = results_table.column(SPLIT_KEY_COLUMN)
    eval_mask = _hash_to_eval_mask(keys, EVAL_FRACTION)
    eval_keys = pc.filter(keys, eval_mask)
    eval_filter = ds.field(SPLIT_KEY_COLUMN).isin(eval_keys)
    ds.write_dataset(
        results_table.filter(eval_filter),
        base_dir=staging_dir / "results",
        format="parquet",
        max_rows_per_file=MAX_ROWS_PER_FILE,
        max_rows_per_group=MAX_ROWS_PER_FILE,
        basename_template="eval-part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )
    ds.write_dataset(
        results_table.filter(~eval_filter),
        base_dir=staging_dir / "results",
        format="parquet",
        max_rows_per_file=MAX_ROWS_PER_FILE,
        max_rows_per_group=MAX_ROWS_PER_FILE,
        basename_template="train-part-{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )

    allowed_ids_by_sidecar = {}
    for sidecar, columns in sidecar_id_columns_in_results.items():
        results = []
        for col_name in columns:
            col = results_table.to_table(columns=[col_name]).column(0)
            if pa.types.is_list(col.type):
                col = col.flatten()
            results.append(pc.unique(pc.drop_null(col)))
        allowed_ids_by_sidecar[sidecar] = pa.concat_arrays(results)
    return allowed_ids_by_sidecar


def stage_sidecar(name: str, allowed_ids: pa.Array) -> None:
    logger.info(f"Staging {name}...")
    allowed_ids = pc.unique(allowed_ids)
    logger.info(f"  {len(allowed_ids):,} unique ids referenced by results")

    dataset = _read_dataset(name)
    keep_cols = [c for c in dataset.schema.names if c != "error"]
    out_dir = staging_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    current_file: Path | None = None
    file_idx = 0
    rows_in_file = 0
    total_kept = 0

    scanner = dataset.scanner(
        columns=keep_cols,
        batch_size=BATCH_SIZE,
        filter=ds.field("id").isin(allowed_ids),
        batch_readahead=2,
        fragment_readahead=8,
    )
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue

        if writer is None:
            current_file = out_dir / f"part-{file_idx}.parquet"
            logger.info(f"  writing to {current_file}...")
            writer = pq.ParquetWriter(
                current_file,
                batch.schema,
            )

        writer.write_batch(batch)
        rows_in_file += batch.num_rows
        total_kept += batch.num_rows

        if rows_in_file >= MAX_ROWS_PER_FILE or (
            current_file is not None
            and current_file.stat().st_size >= MAX_SIZE_PER_FILE
        ):
            writer.close()
            writer = None
            file_idx += 1
            rows_in_file = 0

    if writer is not None:
        writer.close()

    logger.info(f"  wrote {total_kept:,} rows to {file_idx + (rows_in_file > 0)} files")


allowed_ids_by_sidecar = stage_results()
for sidecar, allowed_ids in allowed_ids_by_sidecar.items():
    stage_sidecar(sidecar, allowed_ids)

readme_yaml = """---
configs:
- config_name: results
  default: true
  data_files:
  - split: train
    path: results/train-part-*.parquet
  - split: eval
    path: results/eval-part-*.parquet
- config_name: metrics
  data_files: metrics/part-*.parquet
- config_name: requirements
  data_files: requirements/part-*.parquet
- config_name: coilsets
  data_files: coilsets/part-*.parquet
---
"""
(staging_dir / "README.md").write_text(readme_yaml)


logger.info(f"Uploading staged folder {staging_dir} to {repo_id}...")
api.upload_large_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=str(staging_dir),
    print_report_every=10,
    print_report=True,
)
