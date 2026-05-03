import logging
from typing import Literal

import numpy as np
from desc.coils import CoilSet
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.objectives import (
    CoilLength,
    CoilSetMinDistance,
    FixSumCoilCurrent,
    ObjectiveFunction,
    PlasmaCoilSetMinDistance,
    QuadraticFlux,
)
from desc.optimize import Optimizer

from constellaration_update import types as constellaration_update_types
from constellaration_update.coilset import utils as coilset_utils
from constellaration_update.utils.types import ScalarFloat, ScalarInt

logger = logging.getLogger(__name__)


def optimize_filamentary_coilset(
    coilset: CoilSet,
    eq: Equilibrium,
    normalized_coil_coil_min_distance: ScalarFloat,
    normalized_coil_plasma_min_distance: ScalarFloat,
    normalized_coil_max_curvature: ScalarFloat,
    coil_fourier_order: ScalarInt,
    objective_lambda: ScalarFloat,
    x_scale_mode: Literal["ess", "auto"],
    ess_order: None | ScalarFloat | ScalarInt,
    settings: constellaration_update_types.ConStellarationUpdateDescOptimizerSettings,
) -> tuple[CoilSet, constellaration_update_types.ConStellarationUpdateDescSolveResults]:
    coilset = coilset.to_FourierXYZ(N=int(coil_fourier_order))
    data = eq.compute(["a", "R0"])
    minor_radius = data["a"]
    major_radius = data["R0"]
    aspect_ratio = major_radius / minor_radius
    logger.info(
        f"Equilibrium has {minor_radius=:.3f}m, {major_radius=:.3f}m, "  # noqa: G004
        f"and {aspect_ratio=:.3f}"
    )

    coil_plasma_min_distance = normalized_coil_plasma_min_distance * minor_radius
    coil_coil_min_distance = normalized_coil_coil_min_distance * minor_radius
    coil_max_curvature = normalized_coil_max_curvature / minor_radius

    logger.info(
        f"Using {coil_coil_min_distance=:.3f}m, "  # noqa: G004
        f"{coil_plasma_min_distance=:.3f}m, and "
        f"{coil_max_curvature:.3f}m^-1"
    )

    if not (0.0 <= objective_lambda <= 1.0):
        raise ValueError(
            f"objective_lambda must be between 0 and 1. Got {objective_lambda=}"
        )

    coil_grid = LinearGrid(N=settings.coil_grid_n)
    surf_grid = LinearGrid(
        M=settings.eval_grid_m,
        N=settings.eval_grid_n,
        NFP=eq.NFP,
        sym=eq.sym,
    )

    # Objectives
    objective_scaling_factor = 1e-1
    obj_length = CoilLength(
        coilset,
        target=0,
        grid=coil_grid,
        weight=objective_lambda
        * objective_scaling_factor,  # pyright: ignore[reportArgumentType]
        normalize=True,
    )
    obj_quadratic_flux = QuadraticFlux(
        eq=eq,
        field=coilset,
        target=0,
        eval_grid=surf_grid,
        field_grid=coil_grid,
        vacuum=True,
        weight=(1.0 - objective_lambda)
        * objective_scaling_factor,  # pyright: ignore[reportArgumentType]
        normalize=True,
    )

    objective = ObjectiveFunction([obj_length, obj_quadratic_flux])

    # Constraints
    constr_cc_dist = CoilSetMinDistance(
        coilset,
        bounds=(coil_coil_min_distance, np.inf),
        grid=coil_grid,
        normalize=False,
        normalize_target=False,
    )

    constr_cp_dist = PlasmaCoilSetMinDistance(
        eq=eq,
        coil=coilset,
        bounds=(coil_plasma_min_distance, np.inf),
        plasma_grid=surf_grid,
        coil_grid=coil_grid,
        eq_fixed=True,
        normalize=False,
        normalize_target=False,
    )

    constr_kappa = coilset_utils.CoilCurvatureResidual(
        coilset,
        curvature_threshold=coil_max_curvature,
        grid=coil_grid,
        normalize=False,
        normalize_target=False,
    )

    constr_fix_current = FixSumCoilCurrent(
        coilset,
        normalize=True,
        normalize_target=True,
    )

    constraints = (
        constr_cc_dist,
        constr_cp_dist,
        constr_kappa,
        constr_fix_current,
    )

    # Run optimization
    optimizer = Optimizer(settings.optimizer)

    logger.info(
        f"Running optimization ({settings.optimizer}, maxiter={settings.maxiter})"  # noqa: G004
    )
    optimizer_options = settings.optimizer_options or {}
    if x_scale_mode == "ess":
        optimizer_options["options"] = optimizer_options.get("options", {}) | dict(
            ess_order=ess_order,
            ess_alpha=settings.desc_ess_alpha,
        )

    (coilset_optimized,), result = optimizer.optimize(
        things=coilset,
        objective=objective,
        constraints=constraints,
        maxiter=settings.maxiter,
        verbose=3 if settings.verbose else 0,
        ftol=settings.ftol,
        xtol=settings.xtol,
        ctol=settings.ctol,
        x_scale=x_scale_mode,
        **optimizer_options,
    )

    solve_results = constellaration_update_types.ConStellarationUpdateDescSolveResults(
        success=result["success"],
        message=result["message"],
        n_iterations=int(result["nit"]),
        n_function_evals=int(result["nfev"]),
        objective_value=float(result["fun"]),
        optimality=float(result["optimality"]) if "optimality" in result else None,
        constraint_violation=(
            float(result["constr_violation"]) if "constr_violation" in result else None
        ),
    )

    return coilset_optimized, solve_results
