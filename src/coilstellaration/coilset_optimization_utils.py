import logging
from typing import Any, Literal, cast

import jax.numpy as jnp
import numpy as np
from desc.backend import tree_leaves
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
from desc.objectives._coils import _CoilObjective
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.optimize import Optimizer

from coilstellaration import types

logger = logging.getLogger(__name__)


class CoilCurvatureResidual(_CoilObjective):
    """Coil curvature objective using a per-coil integrated exceedance penalty."""

    _coilset_tree: dict[str, Any]
    _static_attrs = _CoilObjective._static_attrs + ["_curvature_threshold"]

    __doc__ = (__doc__ or "").rstrip() + collect_docs(
        target_default="``0``.",
        bounds_default="``None``.",
        coil=True,
    )

    _scalar = False
    _units = "(m^-1)"
    _print_value_fmt = "Coil curvature: "
    _broadcast_input = "Coil"

    def __init__(
        self,
        coil,
        target: Any = None,
        bounds: Any = None,
        curvature_threshold: float | None = None,
        weight: float = 1.0,
        normalize: bool = True,
        normalize_target: bool = True,
        loss_function: str | None = None,
        deriv_mode: str = "auto",
        grid=None,
        name: str = "coil curvature",
        jac_chunk_size: int | None = None,
    ) -> None:
        assert bounds is None, "CoilCurvatureExcessIntegral does not support bounds."
        assert not normalize, "CoilCurvatureExcessIntegral requires normalize=False."
        if target is not None and target != 0.0:
            raise ValueError(
                "CoilCurvatureExcessIntegral only supports target=0.0 because the "
                "objective computes a residual directly."
            )
        if curvature_threshold is None or not np.isfinite(curvature_threshold):
            raise ValueError(
                "CoilCurvatureExcessIntegral requires a finite curvature_threshold."
            )
        self._curvature_threshold = curvature_threshold

        super().__init__(
            coil,
            ["curvature"],
            target=0.0,
            bounds=None,
            weight=cast(Any, weight),
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        super().build(use_jit=use_jit, verbose=verbose)

        self._constants["coil_quad_weights"] = self._constants["quad_weights"]
        self._constants["quad_weights"] = jnp.ones(self.dim_f)
        self._curvature_threshold = self._coilset_broadcast(self._curvature_threshold)

        if self._normalize:
            self._normalization = 1 / np.mean([scale["a"] for scale in self._scales])

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(
        self, params, constants=None
    ):  # pyright: ignore[reportIncompatibleMethodOverride]
        data = cast(
            Any,
            tree_leaves(
                cast(Any, super().compute(params, constants=constants)),
                is_leaf=lambda x: isinstance(x, dict),
            ),
        )
        quad_weights = cast(Any, constants or self._constants)["coil_quad_weights"]
        thresholds = jnp.asarray(self._curvature_threshold)
        residuals = []
        weight_start = 0
        for i, dat in enumerate(data):
            curvature = jnp.abs(dat["curvature"])
            weight_stop = weight_start + curvature.shape[0]
            coil_quad_weights = quad_weights[weight_start:weight_stop]
            weight_start = weight_stop
            coil_length = jnp.sum(coil_quad_weights)
            residuals.append(
                jnp.sum(
                    coil_quad_weights * jnp.maximum(curvature - thresholds[i], 0.0) ** 2
                )
                / coil_length
            )
        return jnp.asarray(residuals)[self._coilset_tree["coilset_mask"]]


def optimize_filamentary_coilset(
    coilset: CoilSet,
    eq: Equilibrium,
    normalized_coil_coil_min_distance: types.ScalarFloat,
    normalized_coil_plasma_min_distance: types.ScalarFloat,
    normalized_coil_max_curvature: types.ScalarFloat,
    coil_fourier_order: types.ScalarFloat,
    objective_lambda: types.ScalarFloat,
    x_scale_mode: Literal["ess", "auto"],
    ess_order: None | types.ScalarFloat | types.ScalarInt,
    settings: types.DescOptimizerSettings,
) -> tuple[CoilSet, types.DescSolveResults]:
    coilset = coilset.to_FourierXYZ(N=int(coil_fourier_order))
    data = eq.compute(["a", "R0"])
    minor_radius = data["a"]
    major_radius = data["R0"]
    aspect_ratio = major_radius / minor_radius
    logger.info(
        f"Equilibrium has {minor_radius=:.3f}m, {major_radius=:.3f}m, "
        f"and {aspect_ratio=:.3f}"
    )

    coil_plasma_min_distance = normalized_coil_plasma_min_distance * minor_radius
    coil_coil_min_distance = normalized_coil_coil_min_distance * minor_radius
    coil_max_curvature = normalized_coil_max_curvature / minor_radius

    logger.info(
        f"Using {coil_coil_min_distance=:.3f}m, "
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

    constr_kappa = CoilCurvatureResidual(
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
        f"Running optimization ({settings.optimizer}, maxiter={settings.maxiter})"
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

    solve_results = types.DescSolveResults(
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
