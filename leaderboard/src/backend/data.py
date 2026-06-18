"""Memory-lean benchmark loader (spec: avoid the eager full-corpus merge).

`coilstellaration.data_utils.load_benchmark_dataset` loads four full HF tables
(results 253k rows, coilsets 443k JSON blobs, requirements 266k, boundaries
182k) into pandas and *eagerly joins* them, broadcasting big coilset/boundary
JSON blobs across every result row, and only filters to the benchmark subset
(track + stratum + eval split) and applies the `n` cap afterwards. On a 16Gi HF
Space that peaks ~13Gi+ before the ONNX model and JAX/DESC even load.

This loader reorders the work: load only the scalar `results` table, filter and
cap *first*, then pull coilset + boundary JSON for just the selected ids via
Arrow row-selection. Peak stays at a few hundred MB regardless of `n`.

Output is identical to `data_utils.load_dataset`: a list of `types.EvalData`.
"""
import logging

import datasets
import orjson
from coilstellaration import data_utils, types
from constellaration.geometry import surface_rz_fourier

logger = logging.getLogger(__name__)

_RESULTS_REPO = "proxima-fusion/coilstellaration"
_BOUNDARY_REPO = "proxima-fusion/constellaration"

# Scalar columns we keep from the wide (179-col) results table.
_ID_COLUMNS = ("constellaration_boundary_id", "desc_coilset_id", "desc_metrics_id")
_RESULTS_KEEP = [
    *_ID_COLUMNS,
    data_utils.StratificationColumn,
    data_utils.StaticShapeTrackColumn,
    data_utils.VariableShapeTrackColumn,
    *data_utils.REQUIREMENT_METRICS_COLUMNS,
]


def _id_to_json(repo: str, name: str, id_column: str, json_column: str, ids: set[str]) -> dict[str, str]:
    """Map `id_column -> json_column` for `ids` only, without materializing the
    full table. Reads just the id column to locate rows, then gathers those rows.
    """
    ds = datasets.load_dataset(repo, name=name, split="train")
    id_col = ds.select_columns([id_column]).to_pandas()[id_column]
    idx = id_col.index[id_col.isin(ids)].tolist()
    rows = ds.select(idx).select_columns([id_column, json_column]).to_pandas()
    return dict(zip(rows[id_column], rows[json_column]))


def load_benchmark_dataset(
    track: str,
    stratum: str = "tight",
    split: str = "eval",
    n: int = 0,
) -> list[types.EvalData]:
    """Drop-in replacement for `data_utils.load_benchmark_dataset`."""
    track_column = (
        data_utils.StaticShapeTrackColumn
        if track == "fixed_shape"
        else data_utils.VariableShapeTrackColumn
    )

    results = datasets.load_dataset(_RESULTS_REPO, split=split).select_columns(_RESULTS_KEEP).to_pandas()
    selected = results[
        results[track_column].fillna(False)
        & (results[data_utils.StratificationColumn] == stratum)
    ]
    # Match upstream: rows missing a required id or target metric are unusable.
    selected = selected.dropna(
        subset=list(_ID_COLUMNS) + list(data_utils.REQUIREMENT_METRICS_COLUMNS)
    )
    if n > 0:
        selected = selected.head(n)
    logger.info("Selected %d benchmark rows for track=%s stratum=%s", len(selected), track, stratum)

    coilset_json = _id_to_json(_RESULTS_REPO, "coilsets", "id", "json", set(selected["desc_coilset_id"]))
    boundary_json = _id_to_json(
        _BOUNDARY_REPO, "default", "plasma_config_id", "boundary.json", set(selected["constellaration_boundary_id"])
    )

    eval_datas: list[types.EvalData] = []
    for _, row in selected.iterrows():
        boundary_id = row["constellaration_boundary_id"]
        boundary_raw = boundary_json.get(boundary_id)
        coilset_raw = coilset_json.get(row["desc_coilset_id"])
        if boundary_raw is None or coilset_raw is None:
            logger.warning("Missing JSON for boundary_id=%s; skipping", boundary_id)
            continue
        eval_datas.append(
            types.EvalData(
                boundary=surface_rz_fourier.SurfaceRZFourier.model_validate(orjson.loads(boundary_raw)),
                boundary_id=boundary_id,
                true_coilset=types.Coilset.model_validate(orjson.loads(coilset_raw)),
                true_metrics=row["desc_metrics_id"],
                requirement_metrics=data_utils.row_to_requirement_metrics(row.to_dict()),
            )
        )
    logger.info("Assembled evaluation dataset with %d samples.", len(eval_datas))
    return eval_datas
