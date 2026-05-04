"""Helpers for loading VMEC++ wout records from the constellaration HuggingFace dataset.

The ``proxima-fusion/constellaration`` dataset exposes a ``vmecpp_wout`` config
with columns ``id``, ``plasma_config_id``, and ``json`` (a JSON-serialized
``VmecppWOut``). The full table is large (one parquet file per record), so we
stream and filter by id rather than materializing the whole thing.
"""

from __future__ import annotations

import datasets
from constellaration.mhd import vmec_utils

DATASET_REPO = "proxima-fusion/constellaration"
WOUT_CONFIG = "vmecpp_wout"


def load_vmecpp_wout_by_id(
    vmecpp_wout_id: str,
    *,
    repo: str = DATASET_REPO,
    config: str = WOUT_CONFIG,
    split: str = "train",
) -> vmec_utils.VmecppWOut:
    """Fetch a single ``VmecppWOut`` record from the HuggingFace dataset.

    Args:
        vmecpp_wout_id: Value of the ``misc.vmecpp_wout_id`` column from the
            ``default`` config (or the ``id`` column of the ``vmecpp_wout``
            config).
        repo: HuggingFace dataset repo. Override for forks/mirrors.
        config: Dataset config name; defaults to ``"vmecpp_wout"``.
        split: Dataset split; defaults to ``"train"``.

    Raises:
        KeyError: If no row matches ``vmecpp_wout_id``.
    """
    stream = datasets.load_dataset(repo, config, split=split, streaming=True)
    for row in stream:
        if row["id"] == vmecpp_wout_id:
            return vmec_utils.VmecppWOut.model_validate_json(row["json"])
    raise KeyError(
        f"vmecpp_wout id {vmecpp_wout_id!r} not found in {repo}:{config}/{split}"
    )
