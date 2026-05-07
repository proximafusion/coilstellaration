import logging

from coilstellaration import (
    data_generation_tasks,
    data_utils,
    desc_utils,
    paths,
    plot_utils,
    types,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("geometry.surface.surface_types").setLevel(logging.ERROR)
logging.getLogger("storage.google_cloud").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    requirements_id = "DegvKVVq5bPPwbVTAiEhPfR"
    requirements = data_utils.load_requirements_by_id(requirements_id)

    vmecpp_wout_id = "DHTyQRcz3UAR3upPKKvCMfo"
    vmecpp_wout = data_utils.load_vmecpp_wout_by_id(vmecpp_wout_id)
    desc_eq = desc_utils.desc_equilibrium_from_vmecpp_wout(vmecpp_wout)

    regcoil_settings = types.RegcoilSettings()

    desc_optimizer_settings = types.DescOptimizerSettings(
        eval_grid_m=31,
        eval_grid_n=31,
        coil_grid_n=67,
    )

    logger.info("Creating REGCOIL coilset...")
    regcoil_coilset = data_generation_tasks.generate_regcoil_coilset_from_equilibrium(
        eq=vmecpp_wout, requirements=requirements, settings=regcoil_settings
    )
    logger.info("Converting VmecppWOut to DESC Equilibrium...")
    desc_eq = desc_utils.desc_equilibrium_from_vmecpp_wout(vmecpp_wout)
    # logger.info("Solving DESC equilibrium...")
    # desc_eq.solve()
    logger.info("Scaling coil currents such that B_0 = 1T...")
    coilset_with_B_1_T = data_generation_tasks.scale_coil_currents_to_B_0_equals_1_T(
        desc_eq, regcoil_coilset
    )

    regcoil_coilset_path = paths.OUTPUTS_PATH / "regcoil_coilset.json"
    logger.info("Writing coilset to %s", regcoil_coilset_path)
    with regcoil_coilset_path.open("w") as f:
        f.write(coilset_with_B_1_T.model_dump_json())

    logger.info("Plotting initial coilset and equilibrium...")
    figure = plot_utils.plot_coilset_and_equilibrium(desc_eq, coilset_with_B_1_T)
    regcoil_figure_path = paths.OUTPUTS_PATH / "regcoil_coilset_and_equilibrium.html"
    logger.info("Writing figure to %s", regcoil_figure_path)
    figure.write_html(regcoil_figure_path)

    logger.info("Evaluating coilset metrics...")
    metrics = data_generation_tasks.evaluate_coilset_metrics(
        eq=desc_eq,
        coilset=coilset_with_B_1_T,
        surf_eval_m=24,
        surf_eval_n=24,
        coil_eval_n=100,
    )

    regcoil_coilset_metrics_path = paths.OUTPUTS_PATH / "regcoil_coilset_metrics.json"
    logger.info("Writing metrics to %s", regcoil_coilset_metrics_path)
    with regcoil_coilset_metrics_path.open("w") as f:
        f.write(metrics.model_dump_json())
