import logging
import pathlib

from coilstellaration import (
    data_generation_tasks_no_proxima,
    data_utils,
    desc_utils,
    plot_utils,
    types,
)

logging.basicConfig(level=logging.INFO)

OUTPUTS_PATH = pathlib.Path("/home/vscode/tmp/outputs/coilstellaration")
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    logger = logging.getLogger(__file__)

    regcoil_coilset_id = "DPT3MjzMbmJw5VV2mUrWrmQ"
    regcoil_coilset = data_utils.load_coilset_by_id(regcoil_coilset_id)

    requirements_id = "DegvKVVq5bPPwbVTAiEhPfR"
    requirements = data_utils.load_requirements_by_id(requirements_id)

    vmecpp_wout_id = "DHTyQRcz3UAR3upPKKvCMfo"
    vmecpp_wout = data_utils.load_vmecpp_wout_by_id(vmecpp_wout_id)
    desc_eq = desc_utils.desc_equilibrium_from_vmecpp_wout(vmecpp_wout)

    desc_optimizer_settings = types.DescOptimizerSettings(
        eval_grid_m=31,
        eval_grid_n=31,
        coil_grid_n=67,
        maxiter=10,
    )

    logger.info("Scaling coil currents such that B_0 = 1T...")
    regcoil_coilset_with_B_1_T = (
        data_generation_tasks_no_proxima.scale_coil_currents_to_B_0_equals_1_T(
            desc_eq, regcoil_coilset
        )
    )

    logger.info("Running DESC-based coilset optimization...")
    desc_output = data_generation_tasks_no_proxima.optimize_coilset_using_desc(
        eq=desc_eq,
        coilset=regcoil_coilset_with_B_1_T,
        requirements=requirements,
        settings=desc_optimizer_settings,
    )
    desc_output_coilset = (
        data_generation_tasks_no_proxima.extract_coilset_from_desc_output(desc_output)
    )
    logger.info("Scaling coil currents such that B_0 = 1T...")
    desc_output_coilset_with_B_1_T = (
        data_generation_tasks_no_proxima.scale_coil_currents_to_B_0_equals_1_T(
            desc_eq, desc_output_coilset
        )
    )
    logger.info("Writing optimized coilset to desc_coilset.json")
    with pathlib.Path(OUTPUTS_PATH / "desc_coilset.json").open("w") as f:
        f.write(desc_output_coilset_with_B_1_T.model_dump_json())

    logger.info("Plotting initial coilset and equilibrium...")

    figure = plot_utils.plot_coilset_and_equilibrium(
        desc_eq, regcoil_coilset_with_B_1_T, coilset_color="red"
    )
    figure_1 = plot_utils.plot_coilset_and_equilibrium(
        desc_eq, desc_output_coilset_with_B_1_T, figure=figure, coilset_color="blue"
    )

    figure.write_html(OUTPUTS_PATH / "desc_coilset_and_equilibrium.html")

    metrics = data_generation_tasks_no_proxima.evaluate_coilset_metrics(
        eq=desc_eq,
        coilset=desc_output_coilset_with_B_1_T,
        surf_eval_m=24,
        surf_eval_n=24,
        coil_eval_n=100,
    )
    logger.info("Writing metrics to desc_coilset_metrics.json")
    with pathlib.Path(OUTPUTS_PATH / "desc_coilset_metrics.json").open("w") as f:
        f.write(metrics.model_dump_json())
