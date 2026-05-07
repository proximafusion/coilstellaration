import plotly.graph_objects as go
from constellaration.geometry import surface_rz_fourier, surface_utils_desc
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid

from coilstellaration import coilset_utils, types


def plot_coilset(
    coilset: types.Coilset, figure: go.Figure | None = None, color: str | None = None
) -> go.Figure:
    from desc.plotting import plot_coils

    if figure is None:
        figure = go.Figure()

    desc_coilset = coilset_utils.coilstellaration_to_desc(coilset)
    plot_coils(desc_coilset, fig=figure, color=color)
    return figure


def plot_equilibrium(
    eq: Equilibrium,
    coilset: types.Coilset | None = None,
    figure: go.Figure | None = None,
) -> go.Figure:
    from desc.plotting import plot_3d

    if figure is None:
        figure = go.Figure()

    desc_coilset = None
    if coilset is not None:
        desc_coilset = coilset_utils.coilstellaration_to_desc(coilset)
    plot_3d(
        eq,
        name="B*n",
        grid=LinearGrid(M=24, N=24, endpoint=True),
        fig=figure,
        field=desc_coilset,
    )
    return figure


def plot_coilset_and_equilibrium(
    eq: Equilibrium,
    coilset: types.Coilset,
    figure: go.Figure | None = None,
    coilset_color: str | None = None,
) -> go.Figure:
    figure = plot_equilibrium(eq, coilset=coilset, figure=figure)
    figure = plot_coilset(coilset, figure=figure, color=coilset_color)
    return figure


def plot_surface(
    surface: surface_rz_fourier.SurfaceRZFourier,
    figure: go.Figure | None = None,
) -> go.Figure:
    from desc.plotting import plot_3d

    if figure is None:
        figure = go.Figure()
    desc_surface = surface_utils_desc.to_desc_fourier_rz_toroidal_surface(surface)
    plot_3d(desc_surface, "R", fig=figure)
    return figure
