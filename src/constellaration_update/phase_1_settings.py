"""Shared settings for constellaration_update phase 1 runs.

Single source of truth for requirement bounds and optimizer settings
used across phase 1 runners.
"""

from constellaration_update import types as constellaration_update_types

REQUIREMENTS_BOUNDS_GENERAL = (
    constellaration_update_types.ConStellarationUpdateRequirementsBounds(
        normalized_min_coil_plasma_distance=(0.5, 1.0),
        regcoil_winding_surface_plasma_distance=(0.5, 1.0),
        normalized_min_coil_to_coil_distance=(0.3, 0.7),
        normalized_max_coil_curvature=(1.5, 5.0),
        n_coils_per_half_period=(4, 8),
        desc_objective_lambda_log10=(-3.0, -1.0),
        coil_fourier_order=(7, 7),
        desc_x_scale_mode=frozenset({"ess_1", "ess_2", "ess_inf", "auto"}),
        regcoil_target_option=frozenset(
            {"normalized_coil_to_coil_distance", "normalized_field_error"}
        ),
        regcoil_maximum_normalized_field_error=(1.0e-3, 1.0e-1),
    )
)
REQUIREMENTS_BOUNDS_ML = REQUIREMENTS_BOUNDS_GENERAL.model_copy(
    update=dict(n_coils_per_half_period=(5, 5))
)

REGCOIL_SETTINGS = constellaration_update_types.ConStellarationUpdateRegcoilSettings()

DESC_OPTIMIZER_SETTINGS = (
    constellaration_update_types.ConStellarationUpdateDescOptimizerSettings(
        eval_grid_m=31,
        eval_grid_n=31,
        coil_grid_n=67,
    )
)
