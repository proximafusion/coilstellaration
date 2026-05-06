"""Tests for data_generation_tasks module."""

from typing import Literal, cast
from unittest import mock

import pytest
from physics_models.regcoil import regcoil_types

from coilstellaration import (
    data_generation_tasks,
    types,
)


def _make_requirements(
    regcoil_maximum_normalized_field_error: float | None = None,
    regcoil_target_option: Literal[
        "normalized_coil_to_coil_distance", "normalized_field_error"
    ] = "normalized_coil_to_coil_distance",
) -> types.ConStellarationUpdateRequirements:
    return types.ConStellarationUpdateRequirements(
        normalized_min_coil_to_coil_distance=0.25,
        normalized_min_coil_plasma_distance=0.5,
        regcoil_winding_surface_plasma_distance=0.5,
        normalized_max_coil_curvature=1.5,
        n_coils_per_half_period=4,
        coil_fourier_order=5,
        desc_objective_lambda_log10=-1.0,
        desc_x_scale_mode="ess_2",
        regcoil_target_option=regcoil_target_option,
        regcoil_maximum_normalized_field_error=regcoil_maximum_normalized_field_error,
    )


@pytest.mark.parametrize(
    (
        "target_option",
        "field_error_target",
        "expected_maximum_normalized_field_error",
    ),
    [
        ("normalized_coil_to_coil_distance", None, None),
        ("normalized_field_error", 1.25e-2, 1.25e-2),
    ],
)
def test_generate_regcoil_coilset_from_equilibrium_passes_target_configuration(
    monkeypatch: pytest.MonkeyPatch,
    target_option: Literal[
        "normalized_coil_to_coil_distance",
        "normalized_field_error",
    ],
    field_error_target: float | None,
    expected_maximum_normalized_field_error: float | None,
) -> None:
    eq = mock.MagicMock()
    eq.n_field_periods = 5
    generated_coilset = mock.MagicMock()
    captured: dict[str, object] = {}

    def fake_generate_coilset_from_desc_equilibrium(
        eq_arg: object,
        settings_arg: object,
    ) -> object:
        captured["eq"] = eq_arg
        captured["settings"] = settings_arg
        return generated_coilset

    monkeypatch.setattr(
        data_generation_tasks.regcoil_tasks,
        "generate_coilset_from_desc_equilibrium",
        fake_generate_coilset_from_desc_equilibrium,
    )
    requirements = _make_requirements(
        field_error_target,
        regcoil_target_option=target_option,
    )
    settings = types.ConStellarationUpdateRegcoilSettings()

    result = data_generation_tasks.generate_regcoil_coilset_from_equilibrium(
        eq=eq,
        requirements=requirements,
        settings=settings,
    )

    assert result is generated_coilset
    assert captured["eq"] is eq

    regcoil_settings = cast(regcoil_types.RegcoilSettings, captured["settings"])
    assert regcoil_settings.target_option == target_option
    assert regcoil_settings.target_option == requirements.regcoil_target_option
    assert (
        regcoil_settings.maximum_normalized_field_error
        == expected_maximum_normalized_field_error
    )
    assert regcoil_settings.normalized_coil_to_coil_distance == pytest.approx(0.25)
    assert regcoil_settings.n_coils_per_half_period == 4


def test_generate_regcoil_coilset_requires_field_error_bound() -> None:
    eq = mock.MagicMock()
    eq.n_field_periods = 5
    requirements = _make_requirements(regcoil_target_option="normalized_field_error")
    settings = types.ConStellarationUpdateRegcoilSettings()

    with pytest.raises(ValueError, match="regcoil_maximum_normalized_field_error"):
        data_generation_tasks.generate_regcoil_coilset_from_equilibrium(
            eq=eq,
            requirements=requirements,
            settings=settings,
        )
