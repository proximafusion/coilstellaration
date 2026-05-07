"""Utilities for selecting top QI configurations from the constellaration dataset."""

from collections.abc import Sequence

import datasets
import pandas as pd

_NFP_COL = "omnigenous_field_and_targets.omnigenous_field.n_field_periods"
_QI_COL = "metrics.qi"
_IOTA_OVER_NFP_COL = "metrics.edge_rotational_transform_over_n_field_periods"
_VMECPP_WOUT_ID_COL = "misc.vmecpp_wout_id"
_ERROR_COLS = [
    "misc.has_optimize_boundary_omnigenity_vmec_error",
    "misc.has_optimize_boundary_omnigenity_desc_error",
    "misc.has_generate_qp_initialization_from_targets_error",
    "misc.has_generate_nae_initialization_from_targets_error",
    "misc.has_neurips_2025_forward_model_error",
]
_METHOD_ID_COLS = [
    "desc_omnigenous_field_optimization_settings.id",
    "qp_init_omnigenous_field_optimization_settings.id",
    "nae_init_omnigenous_field_optimization_settings.id",
    "vmec_omnigenous_field_optimization_settings.id",
]

_METHODS = ["desc", "vmec"]


def _derive_method(row: pd.Series) -> str:
    for col in _METHOD_ID_COLS:
        if pd.notna(row[col]):
            return col.replace("_omnigenous_field_optimization_settings.id", "")
    return "unknown"


def get_constellaration_vmecpp_wout(
    nfp: int | Sequence[int] | None = 3,
    n: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Return configurations from the constellaration dataset.

    Filters for desc/vmec methods, NFP=3, and positive iota.

    Args:
        nfp: Number of field periods to query for. If int, check equality, if sequence,
            return row that matches any, and if None, don't filter by number of field
            periods.
        n: Number of randomly sampled configurations to return. If None, returns all
            configurations meeting the filter criteria.
        seed: Random seed for reproducibility when sampling configurations.

    Returns:
        DataFrame with columns "plasma_config_id", "vmecpp_wout_id", and
        "qi", sorted by QI residual (ascending).
    """
    df = datasets.load_dataset("proxima-fusion/constellaration", "default")[  # type: ignore[index]
        "train"
    ].to_pandas()  # type: ignore[union-attr]
    assert isinstance(df, pd.DataFrame)

    # Remove rows with errors
    errors_df = df[_ERROR_COLS].fillna(False)
    df = df[~errors_df.any(axis=1)]

    # Derive method column
    df["method"] = df.apply(_derive_method, axis=1)

    # Compute edge iota = iota_over_nfp * nfp
    df["edge_iota"] = df[_IOTA_OVER_NFP_COL] * df[_NFP_COL]

    # Apply filters
    mask = (
        (df["method"].isin(_METHODS))
        & (df["edge_iota"] > 0.0)
        & df[_QI_COL].notna()
        & df[_VMECPP_WOUT_ID_COL].notna()
    )
    match nfp:
        case int():
            mask &= df[_NFP_COL] == nfp
        case Sequence():
            mask &= df[_NFP_COL].isin(nfp)
    df = df[mask]

    df = df.sort_values(_QI_COL, ascending=True)

    result = df[["plasma_config_id", _VMECPP_WOUT_ID_COL, _QI_COL]].rename(
        columns={_VMECPP_WOUT_ID_COL: "vmecpp_wout_id", _QI_COL: "qi"}
    )

    if n is not None:
        result = result.sample(n, random_state=seed)

    return result.reset_index(drop=True)
