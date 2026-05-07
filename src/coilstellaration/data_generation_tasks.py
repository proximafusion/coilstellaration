import logging
from typing import cast

from constellaration.mhd import vmec_utils
from desc.coils import CoilSet as DescCoilSet
from desc.equilibrium import Equilibrium as DescEquilibrium

from coilstellaration import (
    coilset_optimization_utils,
    coilset_utils,
    desc_utils,
    metrics_utils,
    metrics_utils_v2,
    regcoil_types,
    regcoil_utils,
    sampling_utils,
    types,
)
from coilstellaration.types import ScalarFloat, ScalarInt

logger = logging.getLogger(__name__)


def sample_requirements(
    bounds: types.RequirementsBounds,
    settings: types.ConfigSamplingSettings,
) -> types.Requirements:
    int_hash = abs(hash(settings.seed))
    sampled_requirements = sampling_utils.sample_requirements(
        bounds=bounds,
        sampled_cls=types.Requirements,
        seed=int_hash,
    )

    return sampled_requirements


def scale_coil_currents_to_B_0_equals_1_T(
    eq: DescEquilibrium,
    coilset: types.Coilset,
) -> types.Coilset:
    desc_coilset = coilset_utils.coilstellaration_to_desc(coilset)

    desc_coilset_updated_current = coilset_utils.scale_coil_currents_to_B_0_equals_1_T(
        desc_coilset, eq
    )

    coilset_updated_current = coilset_utils.coilstellaration_from_desc(
        desc_coilset_updated_current,
        n_field_periods=int(coilset.n_field_periods),
        is_stellarator_symmetric=coilset.is_stellarator_symmetric,
    )

    return coilset_updated_current


def extract_coilset_from_desc_output(
    desc_output: types.DescOutput,
) -> types.Coilset:
    return desc_output.coilset


def optimize_coilset_using_desc(
    eq: DescEquilibrium,
    coilset: types.Coilset,
    requirements: types.Requirements,
    settings: types.DescOptimizerSettings,
) -> types.DescOutput:
    lambda_log10 = cast(ScalarFloat, requirements.desc_objective_lambda_log10)

    desc_coilset: DescCoilSet = coilset_utils.coilstellaration_to_desc(coilset)

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

    desc_optimized_coilset, solve_results = (
        coilset_optimization_utils.optimize_filamentary_coilset(
            coilset=desc_coilset,
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
    )
    optimized_coilset = coilset_utils.coilstellaration_from_desc(
        desc_optimized_coilset,
        n_field_periods=eq.NFP,
        is_stellarator_symmetric=eq.sym,
    )

    return types.DescOutput(
        coilset=optimized_coilset,
        desc_solve_results=solve_results,
    )


def evaluate_coilset_metrics(
    eq: DescEquilibrium,
    coilset: types.Coilset,
    surf_eval_m: int = 80,
    surf_eval_n: int = 80,
    coil_eval_n: int = 200,
) -> types.Metrics:
    desc_coilset = coilset_utils.coilstellaration_to_desc(coilset)

    metrics = metrics_utils_v2.evaluate_coilset_metrics(
        eq=eq,
        coilset=desc_coilset,
        surf_eval_m=surf_eval_m,
        surf_eval_n=surf_eval_n,
        coil_eval_n=coil_eval_n,
    )
    return metrics


def generate_regcoil_coilset_from_equilibrium(
    eq: vmec_utils.VmecppWOut,
    requirements: types.Requirements,
    settings: types.RegcoilSettings,
) -> types.Coilset:
    target_option = cast(
        regcoil_types.RegcoilTargetOption, requirements.regcoil_target_option
    )
    maximum_normalized_field_error = None
    if target_option == "normalized_field_error":
        if requirements.regcoil_maximum_normalized_field_error is None:
            raise ValueError(
                "CoilStellarationRequirements."
                "regcoil_maximum_normalized_field_error must be set when "
                "CoilStellarationRequirements.regcoil_target_option is "
                "'normalized_field_error'."
            )
        maximum_normalized_field_error = float(
            requirements.regcoil_maximum_normalized_field_error
        )

    regcoil_settings = regcoil_types.RegcoilSettings(
        coils_surface_distance_over_minor_radius=float(
            requirements.regcoil_winding_surface_plasma_distance,
        ),
        normalized_coil_to_coil_distance=float(
            requirements.normalized_min_coil_to_coil_distance
        ),
        n_coils_per_half_period=int(requirements.n_coils_per_half_period),
        maximum_normalized_field_error=maximum_normalized_field_error,
        target_option=target_option,
        **settings.model_dump(exclude={"target_option"}),
    )

    return regcoil_utils.generate_regcoil_coilset_from_equilibrium(eq, regcoil_settings)


def generate_coilset_from_equilibrium_pipeline(
    eq: vmec_utils.VmecppWOut,
    bounds: types.RequirementsBounds,
    sampling_settings: types.ConfigSamplingSettings,
    regcoil_settings: types.RegcoilSettings,
    desc_optimizer_settings: types.DescOptimizerSettings,  # noqa: E501
    optimize_coilset_with_desc: bool = True,
) -> types.Coilset:
    requirements = sample_requirements(
        bounds=bounds,
        settings=sampling_settings,
    )
    regcoil_coilset = generate_regcoil_coilset_from_equilibrium(
        eq=eq, requirements=requirements, settings=regcoil_settings
    )
    desc_eq = desc_utils.desc_equilibrium_from_vmecpp_wout(eq)
    coilset_with_B_1_T = scale_coil_currents_to_B_0_equals_1_T(desc_eq, regcoil_coilset)
    _ = evaluate_coilset_metrics(eq=desc_eq, coilset=coilset_with_B_1_T)
    if not optimize_coilset_with_desc:
        return coilset_with_B_1_T
    desc_output = optimize_coilset_using_desc(
        eq=desc_eq,
        coilset=coilset_with_B_1_T,
        requirements=requirements,
        settings=desc_optimizer_settings,
    )
    desc_output_coilset = extract_coilset_from_desc_output(desc_output)
    desc_output_coilset_with_B_1_T = scale_coil_currents_to_B_0_equals_1_T(
        desc_eq, desc_output_coilset
    )
    _ = evaluate_coilset_metrics(eq=desc_eq, coilset=desc_output_coilset_with_B_1_T)
    return desc_output_coilset_with_B_1_T
