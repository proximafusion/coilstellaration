import logging
from typing import cast

import dapper
import numpy as np
from dapper import common_types
from desc.coils import FourierXYZCoil
from desc.equilibrium import Equilibrium as DescEquilibrium
from geometry.curve import curve_types
from physics_models.field_lines import field_lines_tasks, field_lines_types
from physics_models.ideal_mhd import ideal_mhd_desc
from physics_models.ideal_mhd.desc import desc_types, desc_utils
from physics_models.makegrid import makegrid_tasks, makegrid_types
from physics_models.poincare_plot import poincare_plot_tasks, poincare_plot_types
from physics_models.regcoil import regcoil_tasks, regcoil_types
from reactor_design.analysis.single_stage import single_stage_field_lines
from reactor_design.components.coil import coil_types
from reactor_design.components.coilset import coilset_types

from coilstellaration import (
    coilset_utils,
    coilstellaration_plot,
    data_generation_tasks_no_proxima,
    metrics_utils,
    types,
)

logger = logging.getLogger(__name__)

####################
# PROPRIETARY CODE #
####################


@dapper.task
def create_proxima_poincare_plot(
    desc_equilibrium: desc_types.DescEquilibrium,
    coilset: coilset_types.Coilset,
) -> poincare_plot_types.PoincarePlots:
    plasma_configuration = ideal_mhd_desc.plasma_configuration_from_desc_equilibrium(
        desc_utils.desc_equilibrium_to_desc_object(desc_equilibrium)
    )

    makegrid_settings = makegrid_types.MakegridSettings(
        n_radial_grid_points=101,
        n_vertical_grid_points=101,
        n_toroidal_cutplanes=int(
            360 / plasma_configuration.boundary.n_field_periods / 2
        ),
    )
    makegrid = makegrid_tasks.run_makegrid_on_plasma_configuration(
        plasma_configuration=plasma_configuration,
        coilset=coilset,
        makegrid_settings=makegrid_settings,
    )
    n_field_lines = 51
    n_toroidal_transits = 250

    field_lines_settings = field_lines_types.FieldLinesSettings(
        n_field_lines=n_field_lines,
        n_toroidal_transits=n_toroidal_transits,
        toroidal_normalized_coordinate_to_start_tracing_from=1.0,
        currents=makegrid.raw_coil_currents,
        n_toroidal_points_per_period=int(
            360 / plasma_configuration.boundary.n_field_periods
        ),
        start_field_lines_offset=0.0,
        end_field_lines_offset=1.2,
    )
    r_start = single_stage_field_lines._get_r_start(
        boundary_initial_guess=plasma_configuration.boundary,
        field_lines_settings=field_lines_settings,
    )
    field_lines_settings.r_start = r_start
    field_lines = field_lines_tasks.run_fieldlines(
        magnetic_field_on_grid=makegrid,
        settings=field_lines_settings,
    )

    poincare_plots = poincare_plot_tasks.create_poincare_plots(
        toroidal_field_lines=field_lines,
        settings=poincare_plot_types.PoincarePlotSettings(hide_axes_and_labels=True),
        equilibrium_1=None,
        equilibrium_2=None,
        equilibrium_3=None,
        surface_1=plasma_configuration.boundary,
    )
    return poincare_plots


@dapper.task
def generate_regcoil_coilset_from_equilibrium(
    eq: desc_types.DescEquilibrium,
    requirements: types.Requirements,
    settings: types.RegcoilSettings,
) -> coilset_types.Coilset:
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
        **settings.model_dump(exclude={"dapper_type", "target_option"}),
    )

    coilset = regcoil_tasks.generate_coilset_from_desc_equilibrium(eq, regcoil_settings)

    return coilset


@dapper.task
def coilstellaration_to_proxima(
    coilset_desc_data: types.Coilset,
) -> coilset_types.Coilset:
    """Convert a CoilsetDESC to a Proxima Coilset."""
    symmetry_type = (
        coil_types.CoilSymmetryType.STELLARATOR_SYMMETRIC
        if coilset_desc_data.is_stellarator_symmetric
        else coil_types.CoilSymmetryType.NONE
    )

    unique_coils: list[coil_types.AnyCoil] = []
    for i in range(coilset_desc_data.n_unique_coils):
        desc_coil = FourierXYZCoil(
            X_n=np.asarray(coilset_desc_data.coil_x_n[i]),
            Y_n=np.asarray(coilset_desc_data.coil_y_n[i]),
            Z_n=np.asarray(coilset_desc_data.coil_z_n[i]),
            current=float(coilset_desc_data.currents[i]),  # type: ignore[arg-type]
        )

        order = desc_coil.N
        x_n = np.asarray(desc_coil.X_n)
        y_n = np.asarray(desc_coil.Y_n)
        z_n = np.asarray(desc_coil.Z_n)

        x_cos = x_n[order:]
        x_sin = x_n[order - 1 :: -1] if order > 0 else np.array([])
        y_cos = y_n[order:]
        y_sin = y_n[order - 1 :: -1] if order > 0 else np.array([])
        z_cos = z_n[order:]
        z_sin = z_n[order - 1 :: -1] if order > 0 else np.array([])

        central_filament_curve = curve_types.CurveXYZFourier(
            x_cos=x_cos,
            x_sin=x_sin,
            y_cos=y_cos,
            y_sin=y_sin,
            z_cos=z_cos,
            z_sin=z_sin,
        )

        unique_coils.append(
            coil_types.FilamentaryCoil(
                central_filament_curve=central_filament_curve,
                n_windings=1.0,
            )
        )

    return coilset_types.Coilset(
        unique_coils=unique_coils,
        unique_coil_currents=np.asarray(coilset_desc_data.currents),
        unique_coil_symmetry_infos=[
            coil_types.CoilSymmetryInfo(
                n_field_periods=int(coilset_desc_data.n_field_periods),
                symmetry_type=symmetry_type,
            )
            for _ in range(coilset_desc_data.n_unique_coils)
        ],
    )


@dapper.task
def proxima_to_coilstellaration(
    coilset: coilset_types.Coilset,
) -> types.Coilset:
    """Convert a Proxima Coilset to a CoilStellarationCoilset."""
    symmetry_info = coilset.unique_coil_symmetry_infos[0]
    is_stellarator_symmetric = (
        symmetry_info.symmetry_type == coil_types.CoilSymmetryType.STELLARATOR_SYMMETRIC
    )

    coil_x_n_list: list[np.ndarray] = []
    coil_y_n_list: list[np.ndarray] = []
    coil_z_n_list: list[np.ndarray] = []

    for coil in coilset.unique_coils:
        assert isinstance(coil, coil_types.FilamentaryCoil)
        curve = coil.central_filament_curve
        assert isinstance(curve, curve_types.CurveXYZFourier)

        x_n = np.concatenate([np.asarray(curve.x_sin)[::-1], np.asarray(curve.x_cos)])
        y_n = np.concatenate([np.asarray(curve.y_sin)[::-1], np.asarray(curve.y_cos)])
        z_n = np.concatenate([np.asarray(curve.z_sin)[::-1], np.asarray(curve.z_cos)])

        coil_x_n_list.append(x_n)
        coil_y_n_list.append(y_n)
        coil_z_n_list.append(z_n)

    return types.Coilset(
        coil_x_n=np.stack(coil_x_n_list),
        coil_y_n=np.stack(coil_y_n_list),
        coil_z_n=np.stack(coil_z_n_list),
        currents=np.asarray(coilset.unique_coil_currents),
        n_field_periods=symmetry_info.n_field_periods,
        is_stellarator_symmetric=is_stellarator_symmetric,
    )


#############################
# THIN DAPPER TASK WRAPPERS #
#############################


@dapper.task
def optimize_coilset_using_desc(
    eq: desc_types.DescEquilibrium,
    coilset: types.Coilset,
    requirements: types.Requirements,
    settings: types.DescOptimizerSettings,
) -> types.DescOutput:
    desc_eq: DescEquilibrium = desc_utils.desc_equilibrium_to_desc_object(eq)
    return data_generation_tasks_no_proxima.optimize_coilset_using_desc(
        eq=desc_eq,
        coilset=coilset,
        requirements=requirements,
        settings=settings,
    )


sample_requirements = dapper.task(data_generation_tasks_no_proxima.sample_requirements)


@dapper.task
def scale_coil_currents_to_B_0_equals_1_T(
    eq: desc_types.DescEquilibrium,
    coilset: types.Coilset,
) -> types.Coilset:
    return data_generation_tasks_no_proxima.scale_coil_currents_to_B_0_equals_1_T(
        coilset=coilset,
        eq=desc_utils.desc_equilibrium_to_desc_object(eq),
    )


@dapper.task
def evaluate_coilset_metrics(
    eq: desc_types.DescEquilibrium,
    coilset: types.Coilset,
) -> types.Metrics:
    desc_eq: DescEquilibrium = desc_utils.desc_equilibrium_to_desc_object(eq)
    desc_coilset = coilset_utils.coilstellaration_to_desc(coilset)

    metrics = metrics_utils.evaluate_coilset_metrics(
        eq=desc_eq,
        coilset=desc_coilset,
    )
    return metrics


extract_coilset_from_desc_output = dapper.task(
    data_generation_tasks_no_proxima.extract_coilset_from_desc_output
)


@dapper.task
def create_desc_poincare_plot(
    eq: desc_types.DescEquilibrium,
    coilset: types.Coilset,
    settings: types.PoincarePlotSettings,
) -> common_types.MatplotlibFigure:
    """Create a Poincaré plot for a coilset using DESC field line tracing."""
    desc_eq: DescEquilibrium = desc_utils.desc_equilibrium_to_desc_object(eq)
    fig = coilstellaration_plot.create_poincare_matplotlib_figure(
        coilset=coilset,
        desc_equilibrium=desc_eq,
        settings=settings,
    )
    return common_types.MatplotlibFigure.from_matplotlib(fig, dpi=1000)


def generate_coilset_from_equilibrium_pipeline(
    eq: desc_types.DescEquilibrium,
    bounds: types.RequirementsBounds,
    sampling_settings: types.ConfigSamplingSettings,
    regcoil_settings: types.RegcoilSettings,
    desc_optimizer_settings: types.DescOptimizerSettings,  # noqa: E501
    debug: bool = False,
    optimize_coilset_with_desc: bool = True,
) -> coilset_types.Coilset:
    requirements = sample_requirements.stored(
        bounds=bounds,
        settings=sampling_settings,
    )
    regcoil_proxima_coilset = generate_regcoil_coilset_from_equilibrium.stored(
        eq=eq, requirements=requirements, settings=regcoil_settings
    )
    if debug:
        _ = create_proxima_poincare_plot.stored(eq, regcoil_proxima_coilset)
    regcoil_coilset = proxima_to_coilstellaration.stored(regcoil_proxima_coilset)
    coilset_with_B_1_T = scale_coil_currents_to_B_0_equals_1_T.stored(
        eq, regcoil_coilset
    )
    _ = evaluate_coilset_metrics.stored(eq=eq, coilset=coilset_with_B_1_T)
    if not optimize_coilset_with_desc:
        return regcoil_proxima_coilset
    desc_output = optimize_coilset_using_desc.stored(
        eq=eq,
        coilset=coilset_with_B_1_T,
        requirements=requirements,
        settings=desc_optimizer_settings,
    )
    desc_output_coilset = extract_coilset_from_desc_output.stored(desc_output)
    desc_output_coilset_with_B_1_T = scale_coil_currents_to_B_0_equals_1_T.stored(
        eq, desc_output_coilset
    )
    _ = evaluate_coilset_metrics.stored(eq=eq, coilset=desc_output_coilset_with_B_1_T)
    proxima_desc_output_coilset = coilstellaration_to_proxima.stored(
        coilset_desc_data=desc_output_coilset_with_B_1_T
    )
    if debug:
        _ = create_proxima_poincare_plot.stored(eq, proxima_desc_output_coilset)
    return proxima_desc_output_coilset
