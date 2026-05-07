import pathlib
import tempfile

import desc.vmec
from constellaration.mhd import vmec_utils
from desc.equilibrium import Equilibrium


def desc_equilibrium_from_vmecpp_wout(
    vmec_equilibrium: vmec_utils.VmecppWOut,
) -> Equilibrium:

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_directory = pathlib.Path(temp_dir_name)
        vmec_input_path = temp_directory / "vmec_input.nc"
        vmec_equilibrium.save(vmec_input_path)

        return desc.vmec.VMECIO.load(vmec_input_path, profile="current")
