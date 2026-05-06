"""Plotly visualizations for ConStellaration Update."""

import logging

import matplotlib.axes
import matplotlib.figure
import numpy as np
import plotly.graph_objects as go
from constellaration.geometry import (
    surface_rz_fourier,
    surface_utils,
    surface_utils_desc,
)
from desc import equilibrium as desc_equilibrium_module
from desc import grid as desc_grid
from desc import plotting as desc_plotting
from desc.coils import CoilSet

from coilstellaration import coilset_utils, types

logger = logging.getLogger(__name__)


def create_poincare_matplotlib_figure(
    coilset: types.Coilset,
    desc_equilibrium: desc_equilibrium_module.Equilibrium,
    settings: types.PoincarePlotSettings,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Create a Poincaré plot using DESC's built-in field line tracing and plotting.

    Follows the DESC poincare_plot example: compute starting R0, Z0 from the
    equilibrium flux surfaces, then call ``desc.plotting.poincare_plot``.

    Args:
        coilset: Coilset in the internal Fourier representation.
        desc_equilibrium: DESC Equilibrium used to compute starting points.
        settings: Plot generation settings (number of field lines, transits, etc.).
        ax: Optional matplotlib axes to draw on. If None, a new figure is
            created. Only supported when a single toroidal angle is requested.

    Returns:
        Matplotlib figure with one subplot per toroidal cross-section.
    """

    desc_coilset = coilset_utils.coilstellaration_to_desc(coilset)

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

    # Convert normalized toroidal angles to radians
    nfp = desc_equilibrium.NFP
    # assumes all boundaries have stellarator symmetry
    period = np.pi / nfp
    nta = settings.normalized_toroidal_angles
    if isinstance(nta, int):
        phi_angles = np.linspace(0, period, nta, endpoint=False)
        n_sections = nta
    else:
        phi_angles = np.asarray(nta) * period
        n_sections = len(phi_angles)
    phi_rad: list[float] = phi_angles.tolist()

    result = desc_plotting.poincare_plot(
        desc_coilset,
        r0,
        z0,
        NFP=nfp,
        ntransit=settings.n_toroidal_transits,
        phi=phi_rad,
        grid=source_grid,
        ax=np.asarray([ax]) if ax is not None else None,  # type: ignore[arg-type]
        color="k",
        size=0.5,
        return_data=False,
        **fli_kwargs,
    )
    fig: matplotlib.figure.Figure = result[0]
    result_axes = np.atleast_1d(result[1]).flat

    # Plot only the boundary (rho=1), not interior flux surfaces
    desc_plotting.plot_surfaces(  # type: ignore[call-arg]
        desc_equilibrium,
        rho=1,
        theta=0,
        phi=phi_rad,
        ax=result[1],
    )

    # Crop each subplot to its own equilibrium boundary extent
    n_theta = 128
    boundary_grid = desc_grid.LinearGrid(
        rho=[1.0],
        theta=np.linspace(0, 2 * np.pi, n_theta, endpoint=True),
        zeta=phi_angles,
        NFP=desc_equilibrium.NFP,
    )
    boundary = desc_equilibrium.compute(["R", "Z"], grid=boundary_grid)
    r_all = np.asarray(boundary["R"]).reshape(n_theta, n_sections)
    z_all = np.asarray(boundary["Z"]).reshape(n_theta, n_sections)

    for i, single_ax in enumerate(result_axes):
        r_sec = r_all[:, i]
        z_sec = z_all[:, i]
        r_mid = (r_sec.min() + r_sec.max()) / 2
        z_mid = (z_sec.min() + z_sec.max()) / 2
        half_span = max(r_sec.max() - r_sec.min(), z_sec.max() - z_sec.min()) / 2 * 1.15
        single_ax.set_xlim(r_mid - half_span, r_mid + half_span)
        single_ax.set_ylim(z_mid - half_span, z_mid + half_span)

    return fig
