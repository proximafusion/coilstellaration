"""Plain Python entry points for data-generation tasks.

No dapper decorators, no Proxima-internal dependencies.
"""

import logging
from typing import cast

from desc.coils import CoilSet as DescCoilSet
from desc.equilibrium import Equilibrium as DescEquilibrium

from constellaration_update import types as constellaration_update_types
from constellaration_update.coilset import optimization as coilset_optimization_utils
from constellaration_update.utils.types import ScalarFloat, ScalarInt

logger = logging.getLogger(__name__)


def optimize_coilset_using_desc(
    eq: DescEquilibrium,
    coilset: DescCoilSet,
    requirements: constellaration_update_types.ConStellarationUpdateRequirements,
    settings: constellaration_update_types.ConStellarationUpdateDescOptimizerSettings,
) -> tuple[
    DescCoilSet, constellaration_update_types.ConStellarationUpdateDescSolveResults
]:
    lambda_log10 = cast(ScalarFloat, requirements.desc_objective_lambda_log10)

    match requirements.desc_x_scale_mode:
        case "ess_1":
            x_scale_mode = "ess"
            ess_order = 1
        case "ess_2":
            x_scale_mode = "ess"
            ess_order = 2
        case "ess_inf":
            x_scale_mode = "ess"
            ess_order = float("inf")
        case "auto":
            x_scale_mode = "auto"
            ess_order = None
        case _:
            raise ValueError(
                f"Invalid desc_x_scale_mode: {requirements.desc_x_scale_mode}"
            )

    (
        optimized_coilset,
        solve_results,
    ) = coilset_optimization_utils.optimize_filamentary_coilset(
        coilset=coilset,
        eq=eq,
        normalized_coil_coil_min_distance=cast(
            ScalarFloat, requirements.normalized_min_coil_to_coil_distance
        ),
        normalized_coil_plasma_min_distance=cast(
            ScalarFloat, requirements.normalized_min_coil_plasma_distance
        ),
        normalized_coil_max_curvature=cast(
            ScalarFloat, requirements.normalized_max_coil_curvature
        ),
        objective_lambda=10.0**lambda_log10,
        coil_fourier_order=cast(ScalarInt, requirements.coil_fourier_order),
        x_scale_mode=x_scale_mode,
        ess_order=ess_order,
        settings=settings,
    )
    return optimized_coilset, solve_results
