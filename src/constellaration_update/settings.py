"""Settings for re-solving DESC equilibria from VMEC++ wout files."""

from __future__ import annotations

import pydantic


class DescFromVmecSettings(pydantic.BaseModel):
    """Settings for instantiating and solving a DESC equilibrium from a VMEC++ wout."""

    profile: str = "current"
    """Which profile to keep fixed when loading the VMEC equilibrium into DESC.

    Either ``"current"`` (constrains net toroidal current) or ``"iota"``
    (constrains rotational transform). The constellaration pipeline uses
    ``"current"``.
    """

    L_grid: int | None = None
    """Radial collocation grid resolution used during the DESC solve.

    ``None`` falls back to DESC's default (``2 * L``).
    """

    M_grid: int | None = None
    """Poloidal collocation grid resolution used during the DESC solve.

    ``None`` falls back to DESC's default (``2 * M``).
    """

    N_grid: int | None = None
    """Toroidal collocation grid resolution used during the DESC solve.

    ``None`` falls back to DESC's default (``2 * N``).
    """

    objective: str = "force"
    """Objective passed to ``Equilibrium.solve`` (e.g. ``"force"``, ``"energy"``)."""

    verbose: int = 3
    """Verbosity level forwarded to ``Equilibrium.solve``."""

    solve_kwargs: dict = pydantic.Field(default_factory=dict)
    """Additional keyword arguments forwarded to ``Equilibrium.solve``.

    Useful keys: ``optimizer``, ``ftol``, ``xtol``, ``gtol``, ``maxiter``,
    ``x_scale``, ``options``. See ``desc.equilibrium.Equilibrium.solve``.
    """
