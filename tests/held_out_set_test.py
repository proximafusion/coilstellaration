"""Tests for ``held_out_set.get_held_out_set_ids``."""

from __future__ import annotations

import unittest.mock

import pandas as pd
import pytest

from constellaration_update import held_out_set


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    held_out_set._load_held_out_dataframe.cache_clear()


@pytest.fixture
def fake_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_run_tag": [
                "sampling-omnigenous-targets-260421-092425",  # general in_domain
                "sampling-omnigenous-targets-260423-192629",  # general in_domain
                "sampling-omnigenous-targets-260421-092756",  # general out_of_domain
                "sampling-omnigenous-targets-260423-192900",  # general out_of_domain
                "sampling-omnigenous-targets-260424-203111",  # nfp3 in_domain
                "sampling-omnigenous-targets-260424-203256",  # nfp3 out_of_domain
            ],
            "plasma_configuration_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "vmecpp_wout_id": ["V1", "V2", "V3", "V4", "V5", "V6"],
            "desc_equilibrium_solved_id": ["D1", "D2", "D3", "D4", "D5", "D6"],
            # Extra columns that are present in the real parquet — should be
            # dropped from the function's output.
            "omnigenous_field_and_targets_id": ["O1", "O2", "O3", "O4", "O5", "O6"],
            "optimizer_family": ["vmec"] * 6,
        }
    )


@pytest.mark.parametrize(
    ("distribution_shift", "category", "expected_solved_ids"),
    [
        ("in_domain", "general", {"D1", "D2"}),
        ("out_of_domain", "general", {"D3", "D4"}),
        ("in_domain", "nfp3", {"D5"}),
        ("out_of_domain", "nfp3", {"D6"}),
    ],
)
def test_get_held_out_set_ids_filters_correctly(
    fake_dataframe: pd.DataFrame,
    distribution_shift: held_out_set.DistributionShift,
    category: held_out_set.Category,
    expected_solved_ids: set[str],
) -> None:
    with unittest.mock.patch.object(
        held_out_set, "_load_held_out_dataframe", return_value=fake_dataframe
    ):
        result = held_out_set.get_held_out_set_ids(distribution_shift, category)

    assert list(result.columns) == [
        "plasma_configuration_id",
        "vmecpp_wout_id",
        "desc_equilibrium_solved_id",
    ]
    assert set(result["desc_equilibrium_solved_id"]) == expected_solved_ids


def test_run_tag_mapping_partition_is_complete_and_disjoint() -> None:
    """All six run tags map to exactly one (category, shift) bucket."""
    all_tags: list[str] = []
    for tags in held_out_set._RUN_TAGS_BY_CATEGORY_AND_SHIFT.values():
        all_tags.extend(tags)

    assert len(all_tags) == 6, "Expected six sampling-omnigenous-targets run tags."
    assert len(set(all_tags)) == 6, "Run tags should not appear in multiple buckets."
    assert all(
        tag.startswith("sampling-omnigenous-targets-") for tag in all_tags
    ), "Run tags should be sampling-omnigenous-targets-*."
