"""Tests for data_generation/tasks module."""

from unittest import mock

import numpy as np
import pytest

_SKIP_REASON = (
    "requires desc env with compatible jax (tree_broadcast missing in current env)"
)

from constellaration_update import types as constellaration_update_types  # noqa: E402

data_generation_tasks_no_proxima = pytest.importorskip(
    "constellaration_update.data_generation.tasks",
    reason=_SKIP_REASON,
    exc_type=ImportError,
)


def test_optimize_coilset_using_desc_converts_log10_objective_lambda() -> None:
    eq = mock.MagicMock()
    coilset = mock.MagicMock()

    optimized_coilset = mock.MagicMock()
    solve_results = mock.MagicMock()

    requirements = constellaration_update_types.ConStellarationUpdateRequirements(
        normalized_min_coil_plasma_distance=0.2,
        regcoil_winding_surface_plasma_distance=0.2,
        normalized_min_coil_to_coil_distance=0.1,
        normalized_max_coil_curvature=0.3,
        n_coils_per_half_period=4,
        coil_fourier_order=5,
        desc_x_scale_mode="ess_2",
        regcoil_target_option="normalized_coil_to_coil_distance",
        desc_objective_lambda_log10=float(np.log10(0.5)),
    )
    settings = constellaration_update_types.ConStellarationUpdateDescOptimizerSettings()

    with mock.patch.object(
        data_generation_tasks_no_proxima.coilset_optimization_utils,
        "optimize_filamentary_coilset",
        return_value=(optimized_coilset, solve_results),
    ) as optimize_mock:
        result = data_generation_tasks_no_proxima.optimize_coilset_using_desc(
            eq=eq,
            coilset=coilset,
            requirements=requirements,
            settings=settings,
        )

    optimize_mock.assert_called_once()
    assert optimize_mock.call_args.kwargs["coilset"] is coilset
    assert optimize_mock.call_args.kwargs["eq"] is eq
    assert optimize_mock.call_args.kwargs["objective_lambda"] == pytest.approx(0.5)
    assert result == (optimized_coilset, solve_results)
