"""Re-solve a DESC equilibrium from a constellaration HuggingFace VMEC++ wout.

End-to-end workflow:

    1. Look up a record in the ``vmecpp_wout`` config of
       ``proxima-fusion/constellaration`` by id.
    2. Convert the VMEC++ wout into a DESC ``Equilibrium`` (current-profile
       constraint, matching the original pipeline).
    3. Optionally override collocation-grid resolutions.
    4. Drive the equilibrium to force balance with ``Equilibrium.solve``.

Usage:

    python examples/solve_from_id.py <vmecpp_wout_id>

The id is the value found in the ``misc.vmecpp_wout_id`` column of the
``default`` config, or the ``id`` column of the ``vmecpp_wout`` config.
"""

from __future__ import annotations

import argparse
import logging

from coilstellaration.dataset_utils import load_vmecpp_wout_by_id
from coilstellaration.ideal_mhd.desc_tasks import (
    instantiate_and_solve_desc_equilibrium_from_vmecpp_wout,
)
from coilstellaration.ideal_mhd.desc_types import DescFromVmecSettings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vmecpp_wout_id", help="HuggingFace vmecpp_wout id to re-solve")
    parser.add_argument("--L-grid", type=int, default=None)
    parser.add_argument("--M-grid", type=int, default=None)
    parser.add_argument("--N-grid", type=int, default=None)
    parser.add_argument("--objective", default="force")
    parser.add_argument("--verbose", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    settings = DescFromVmecSettings(
        L_grid=args.L_grid,
        M_grid=args.M_grid,
        N_grid=args.N_grid,
        objective=args.objective,
        verbose=args.verbose,
    )

    wout = load_vmecpp_wout_by_id(args.vmecpp_wout_id)
    solved = instantiate_and_solve_desc_equilibrium_from_vmecpp_wout(wout, settings)

    print(
        f"Solved DESC equilibrium for id={args.vmecpp_wout_id}: "
        f"L={solved.L} M={solved.M} N={solved.N} NFP={solved.NFP} Psi={solved.Psi:.4g}"
    )


if __name__ == "__main__":
    main()
