"""Plotly visualizations for ConStellaration Update."""

import logging

import matplotlib.figure
import numpy as np
from desc import equilibrium as desc_equilibrium_module

from constellaration_update import types as constellaration_update_types

logger = logging.getLogger(__name__)


def create_poincare_matplotlib_figure(
    coilset: constellaration_update_types.ConStellarationUpdateCoilset,
    desc_equilibrium: desc_equilibrium_module.Equilibrium,
    settings: constellaration_update_types.ConStellarationUpdatePoincarePlotSettings,
) -> matplotlib.figure.Figure:
    """Create a Poincaré plot using DESC's built-in field line tracing and plotting.

    Follows the DESC poincare_plot example: compute starting R0, Z0 from the
    equilibrium flux surfaces, then call ``desc.plotting.poincare_plot``.

    Args:
        coilset: Coilset in the internal Fourier representation.
        desc_equilibrium: DESC Equilibrium used to compute starting points.
        settings: Plot generation settings (number of field lines, transits, etc.).

    Returns:
        Matplotlib figure with one subplot per toroidal cross-section.
    """
    from desc import grid as desc_grid
    from desc import plotting as desc_plotting

    from constellaration_update.coilset import utils as coilset_utils

    desc_coilset = coilset_utils.constellaration_update_to_desc(coilset)

    # Build an explicit Grid object for the Biot-Savart source discretization.
    # Passing a plain int would be traced by diffrax's shape-checking into a
    # DynamicJaxprTracer, which fails DESC's isinstance(grid, _Grid) check.
    source_grid = desc_grid.LinearGrid(N=settings.source_grid_points)

    grid_trace = desc_grid.LinearGrid(
        rho=np.linspace(settings.rho_min, settings.rho_max, settings.n_field_lines),
        NFP=desc_equilibrium.NFP,
    )
    r0 = desc_equilibrium.compute("R", grid=grid_trace)["R"]
    z0 = desc_equilibrium.compute("Z", grid=grid_trace)["Z"]

    fli_kwargs: dict = dict(
        rtol=settings.rtol,
        atol=settings.atol,
    )
    if settings.max_steps is not None:
        fli_kwargs["max_steps"] = settings.max_steps

    result = desc_plotting.poincare_plot(
        desc_coilset,
        r0,
        z0,
        NFP=desc_equilibrium.NFP,
        ntransit=settings.n_toroidal_transits,
        phi=settings.toroidal_angles,
        grid=source_grid,
        color="k",
        size=0.5,
        return_data=False,
        **fli_kwargs,
    )
    fig = result[0]
    ax = result[1]

    desc_plotting.plot_surfaces(desc_equilibrium, phi=settings.toroidal_angles, ax=ax)

    # Crop each subplot to its own equilibrium boundary extent
    nfp = desc_equilibrium.NFP
    phi_angles = np.linspace(
        0, 2 * np.pi / nfp, settings.toroidal_angles, endpoint=False
    )
    n_theta = 128
    boundary_grid = desc_grid.LinearGrid(
        rho=np.array([1.0]),
        theta=np.linspace(0, 2 * np.pi, n_theta, endpoint=True),
        zeta=phi_angles,
        NFP=desc_equilibrium.NFP,
    )
    boundary = desc_equilibrium.compute(["R", "Z"], grid=boundary_grid)
    r_all = np.asarray(boundary["R"]).reshape(n_theta, settings.toroidal_angles)
    z_all = np.asarray(boundary["Z"]).reshape(n_theta, settings.toroidal_angles)

    for i, single_ax in enumerate(np.atleast_1d(ax).flat):
        r_sec = r_all[:, i]
        z_sec = z_all[:, i]
        r_mid = (r_sec.min() + r_sec.max()) / 2
        z_mid = (z_sec.min() + z_sec.max()) / 2
        # Pad by 50% beyond the boundary extent
        half_span = max(r_sec.max() - r_sec.min(), z_sec.max() - z_sec.min()) / 2 * 1.5
        single_ax.set_xlim(r_mid - half_span, r_mid + half_span)
        single_ax.set_ylim(z_mid - half_span, z_mid + half_span)

    return fig
