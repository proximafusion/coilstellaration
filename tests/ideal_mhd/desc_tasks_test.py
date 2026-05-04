"""Tests for the DESC-from-VMEC pipeline."""

from pathlib import Path

import pytest
from constellaration.mhd import vmec_utils

from coilstellaration.ideal_mhd import desc_tasks, desc_types

WOUT_FIXTURE = Path("tests/ideal_mhd/huggingface_dataset_wout.json")


@pytest.fixture
def vmec_equilibrium() -> vmec_utils.VmecppWOut:
    return vmec_utils.VmecppWOut.model_validate_json(WOUT_FIXTURE.read_text())


def test_instantiate_desc_equilibrium_from_vmecpp_wout(
    vmec_equilibrium: vmec_utils.VmecppWOut,
) -> None:
    import desc.equilibrium

    settings = desc_types.DescFromVmecSettings(
        M_grid=8,
        N_grid=8,
        verbose=0,
    )

    eq = desc_tasks.instantiate_desc_equilibrium_from_vmecpp_wout(
        vmec_equilibrium, settings
    )

    assert isinstance(eq, desc.equilibrium.Equilibrium)
    assert eq.NFP == vmec_equilibrium.nfp
    assert eq.M_grid == settings.M_grid
    assert eq.N_grid == settings.N_grid


def test_solve_desc_equilibrium(
    vmec_equilibrium: vmec_utils.VmecppWOut,
) -> None:
    import desc.equilibrium

    settings = desc_types.DescFromVmecSettings(
        M_grid=8,
        N_grid=8,
        verbose=0,
        solve_kwargs={"maxiter": 1},
    )

    eq = desc_tasks.instantiate_desc_equilibrium_from_vmecpp_wout(
        vmec_equilibrium, settings
    )
    solved = desc_tasks.solve_desc_equilibrium(eq, settings)

    assert isinstance(solved, desc.equilibrium.Equilibrium)
    assert solved.NFP == vmec_equilibrium.nfp


def test_instantiate_and_solve_desc_equilibrium_from_vmecpp_wout(
    vmec_equilibrium: vmec_utils.VmecppWOut,
) -> None:
    import desc.equilibrium

    settings = desc_types.DescFromVmecSettings(
        M_grid=8,
        N_grid=8,
        verbose=0,
        solve_kwargs={"maxiter": 1},
    )

    solved = desc_tasks.instantiate_and_solve_desc_equilibrium_from_vmecpp_wout(
        vmec_equilibrium, settings
    )

    assert isinstance(solved, desc.equilibrium.Equilibrium)
    assert solved.NFP == vmec_equilibrium.nfp
