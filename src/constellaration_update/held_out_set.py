"""Accessor for the constellaration_update held-out DESC equilibria set.

The held-out set is the union of six ``sampling-omnigenous-targets-...`` runs
that were pushed through the neurips_2025 forward model and then the DESC
chain (``instantiate_desc_equilibrium_from_vmecpp_wout`` ->
``solve_desc_equilibrium`` ->
``extract_desc_equilibrium_from_optimization_output``). All chain objects are
tagged with ``constellaration_update:held_out_set`` plus either
``constellaration_update:in_domain`` or ``constellaration_update:out_of_domain``.

Two splits are exposed:

- ``general``: the four-tag sweep that covers the usual ``nfp ∈ {2, 3, 4, 5}``
  range (`260421-...`, `260423-...`).
- ``nfp3``: the two-tag sweep restricted to ``nfp = 3`` only (`260424-...`).

The ID triples (``plasma_configuration_id``, ``vmecpp_wout_id``,
``desc_equilibrium_solved_id``) come from a small (~23 KB) parquet on GCS,
produced by ``dump_desc_from_neurips_pipeline``. To regenerate, run the
``launch_dump_desc_from_neurips`` launcher (next to this module) and update
``HELD_OUT_PARQUET_URI`` below to the resulting GCS path.
"""

from __future__ import annotations

import functools
from typing import Literal

import pandas as pd

# Imported explicitly so packagers bundle it; pandas uses pyarrow as the
# parquet engine but only references it lazily.
import pyarrow  # noqa: F401, ICN001

HELD_OUT_PARQUET_URI: str = (
    "gs://proxima-data-repository/dataproc/desc_from_neurips_2025/dumps/PROD/"
    "1777113738-scadena/results.parquet/"
)
"""GCS path to the parquet table emitted by ``dump_desc_from_neurips_pipeline``.

The columns we care about: ``source_run_tag``, ``plasma_configuration_id``,
``vmecpp_wout_id``, ``desc_equilibrium_solved_id``.
"""

DistributionShift = Literal["in_domain", "out_of_domain"]
Category = Literal["general", "nfp3"]

_RUN_TAGS_BY_CATEGORY_AND_SHIFT: dict[
    tuple[Category, DistributionShift], tuple[str, ...]
] = {
    ("general", "in_domain"): (
        "sampling-omnigenous-targets-260421-092425",
        "sampling-omnigenous-targets-260423-192629",
    ),
    ("general", "out_of_domain"): (
        "sampling-omnigenous-targets-260421-092756",
        "sampling-omnigenous-targets-260423-192900",
    ),
    ("nfp3", "in_domain"): ("sampling-omnigenous-targets-260424-203111",),
    ("nfp3", "out_of_domain"): ("sampling-omnigenous-targets-260424-203256",),
}

_OUTPUT_COLUMNS: tuple[str, ...] = (
    "plasma_configuration_id",
    "vmecpp_wout_id",
    "desc_equilibrium_solved_id",
)


def get_held_out_set_ids(
    distribution_shift: DistributionShift,
    category: Category,
) -> pd.DataFrame:
    """Return the held-out DESC chain ID triples for the requested split.

    Args:
        distribution_shift: ``in_domain`` or ``out_of_domain``.
        category: ``general`` (covers the standard ``nfp`` range) or ``nfp3``
            (the ``nfp = 3``-only sweep).

    Returns:
        DataFrame with one row per held-out equilibrium and columns
        ``plasma_configuration_id``, ``vmecpp_wout_id``,
        ``desc_equilibrium_solved_id``.
    """
    run_tags = _RUN_TAGS_BY_CATEGORY_AND_SHIFT[(category, distribution_shift)]
    df = _load_held_out_dataframe()  # noqa: PD901
    return (
        df[df["source_run_tag"].isin(run_tags)][list(_OUTPUT_COLUMNS)]
        .reset_index(drop=True)
        .copy()
    )


@functools.lru_cache(maxsize=1)
def _load_held_out_dataframe() -> pd.DataFrame:
    """Load and cache the held-out parquet from GCS.

    Wrapped behind a private helper so tests can monkeypatch the loader without touching
    the network.
    """
    return pd.read_parquet(HELD_OUT_PARQUET_URI)
