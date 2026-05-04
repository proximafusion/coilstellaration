"""Instantiate a DESC equilibrium from a VMEC++ wout and re-solve it.

This is a slimmed-down, dependency-light port of the
``instantiate_and_solve_desc_equilibrium_from_vmecpp_wout`` Dagster graph from
the original (closed-source) pipeline. It drops the dapper/dagster wrappers,
the serialization layer, and the comparison plot, keeping just the two steps
that matter: VMEC -> DESC, and DESC.solve.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from constellaration.mhd import vmec_utils

from constellaration_update.settings import DescFromVmecSettings

# DESC is imported lazily because importing the package is slow and its plotting
# submodule has side effects (creates a blank matplotlib canvas).
if TYPE_CHECKING:
    import desc.equilibrium

logger = logging.getLogger(__name__)


def instantiate_desc_equilibrium_from_vmecpp_wout(
    wout: vmec_utils.VmecppWOut,
    settings: DescFromVmecSettings,
) -> "desc.equilibrium.Equilibrium":
    """Build a DESC ``Equilibrium`` initialized from a VMEC++ wout.

    Round-trips the wout through a temporary NetCDF file so DESC's VMEC loader
    can ingest it, then applies any grid-resolution overrides from ``settings``.
    """
    import desc.vmec

    with tempfile.TemporaryDirectory() as tmp:
        wout_path = Path(tmp) / "wout.nc"
        wout.save(wout_path)
        eq = desc.vmec.VMECIO.load(str(wout_path), profile=settings.profile)

    if settings.L_grid is not None:
        eq.L_grid = settings.L_grid
    if settings.M_grid is not None:
        eq.M_grid = settings.M_grid
    if settings.N_grid is not None:
        eq.N_grid = settings.N_grid

    return eq


def solve_desc_equilibrium(
    eq: "desc.equilibrium.Equilibrium",
    settings: DescFromVmecSettings,
) -> "desc.equilibrium.Equilibrium":
    """Run ``Equilibrium.solve`` to drive the equilibrium to force balance."""
    solved, _result = eq.solve(
        objective=settings.objective,
        verbose=settings.verbose,
        **settings.solve_kwargs,
    )
    return solved


def instantiate_and_solve_desc_equilibrium_from_vmecpp_wout(
    wout: vmec_utils.VmecppWOut,
    settings: DescFromVmecSettings,
) -> "desc.equilibrium.Equilibrium":
    """Convenience wrapper: VMEC++ wout -> initial DESC equilibrium -> solved."""
    eq = instantiate_desc_equilibrium_from_vmecpp_wout(wout, settings)
    logger.info("Solving DESC equilibrium with objective=%r", settings.objective)
    return solve_desc_equilibrium(eq, settings)
