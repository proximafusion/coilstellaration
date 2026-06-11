"""Pure conversions between package types and raw ONNX tensors.

No scaling/normalization: raw Fourier coefficients in, raw coilset arrays out.
The model owns all internal encoding (spec §4).
"""
import logging

import numpy as np

from constellaration.geometry import surface_rz_fourier
from coilstellaration import types

from src.backend import config

logger = logging.getLogger(__name__)


def eval_data_to_inputs(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    requirement_metrics: types.RequirementMetrics,
) -> dict[str, np.ndarray]:
    """Build the raw ONNX feed dict for one instance."""
    inputs: dict[str, np.ndarray] = {
        "boundary_r_cos": np.asarray(boundary.r_cos, dtype=np.float32),
        "boundary_z_sin": np.asarray(boundary.z_sin, dtype=np.float32),
        "requirement_metrics": np.asarray(
            requirement_metrics.to_array(), dtype=np.float32
        ),
    }
    if not boundary.is_stellarator_symmetric:
        inputs["boundary_r_sin"] = np.asarray(boundary.r_sin, dtype=np.float32)
        inputs["boundary_z_cos"] = np.asarray(boundary.z_cos, dtype=np.float32)
    logger.debug("Built ONNX feed dict with tensors: %s", list(inputs))
    return inputs


def outputs_to_coilset(
    outputs: dict[str, np.ndarray],
    boundary: surface_rz_fourier.SurfaceRZFourier,
) -> types.Coilset:
    """Reconstruct a Coilset from raw model outputs.

    `n_field_periods` and `is_stellarator_symmetric` come from the instance
    boundary (the coilset must match the plasma), not the model.
    """
    for name in config.OUTPUT_NAMES:
        if name not in outputs:
            raise KeyError(f"Model output missing required tensor: {name!r}")
    logger.debug("Reconstructing coilset from model outputs: %s", list(outputs))
    return types.Coilset(
        coil_x_n=np.asarray(outputs["coil_x_n"], dtype=float),
        coil_y_n=np.asarray(outputs["coil_y_n"], dtype=float),
        coil_z_n=np.asarray(outputs["coil_z_n"], dtype=float),
        currents=np.asarray(outputs["currents"], dtype=float).reshape(-1),
        n_field_periods=int(boundary.n_field_periods),
        is_stellarator_symmetric=bool(boundary.is_stellarator_symmetric),
    )
