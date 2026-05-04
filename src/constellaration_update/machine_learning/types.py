import time
from typing import Literal, Optional, Self

import pydantic
from constellaration.geometry import surface_rz_fourier
from constellaration.utils.pytree import register_pydantic_data
from jaxtyping import Float

from constellaration_update import types as constellaration_update_types
from constellaration_update.checkpoint import types as flax_nnx_checkpoint_types
from constellaration_update.utils.types import NpOrJaxArray


class CoilPredictorConfig(pydantic.BaseModel):
    """Init config for `CoilPredictor`, stored alongside its checkpoint."""

    is_stellarator_symmetric: bool
    n_field_periods: int
    n_poloidal_modes: int
    n_toroidal_modes: int
    n_unique_coils: int
    n_modes_coils_max: int
    """Global max fourier order over the training dataset.

    Inferred via `eval_data_utils.infer_max_fourier_order(train_df)` and frozen
    into the resolved config. The MLP readout dimensions and the spectral-scaling
    helper signatures all use this value.
    """
    expected_currents: list[float]
    hidden_dim: int
    num_layers: int
    requirement_metrics_means: list[float]
    """Per-feature train-set means used to standardize `RequirementMetrics` inputs.

    Populated from the train split when the resolved config is built. Length is always
    4, matching `RequirementMetrics.to_array()`.
    """
    requirement_metrics_stds: list[float]
    """Per-feature train-set stds (clamped at `1e-8`) used for standardization."""
    spectral_scaling_alpha: float = 1.2
    spectral_scaling_min: float = 1e-7
    spectral_scaling_order: float = float("inf")


CoilPredictorCheckpoint = flax_nnx_checkpoint_types.FlaxNnxCheckpoint[
    CoilPredictorConfig
]


class AttentionCoilPredictorConfig(pydantic.BaseModel):
    """Init config for `AttentionCoilPredictor`, stored alongside its checkpoint.

    Sibling of `CoilPredictorConfig`. Dataset/scaling fields are duplicated
    because pydantic does not support subclassing in this context.
    """

    is_stellarator_symmetric: bool
    n_field_periods: int
    n_poloidal_modes: int
    n_toroidal_modes: int
    n_unique_coils: int
    n_modes_coils_max: int
    """Global max fourier order over the training dataset.

    Inferred via `eval_data_utils.infer_max_fourier_order(train_df)` and frozen
    into the resolved config. The MLP readout dimensions and the spectral-scaling
    helper signatures all use this value.
    """
    expected_currents: list[float]
    requirement_metrics_means: list[float]
    """Per-feature train-set means used to standardize `RequirementMetrics` inputs."""
    requirement_metrics_stds: list[float]
    """Per-feature train-set stds (clamped at `1e-8`) used for standardization."""
    spectral_scaling_alpha: float = 1.2
    spectral_scaling_min: float = 1e-7
    spectral_scaling_order: float = float("inf")

    hidden_dim: int = 128
    """Channel width carried through the backbone (post conv stem, MHSA, pre conv
    head)."""
    conv_stem_channels: list[int] = pydantic.Field(default_factory=lambda: [32, 64])
    """Per-layer output channels of the conv stem.

    A final 1x1 conv lifts to `hidden_dim`.
    """
    conv_stem_kernel: int = 3
    num_attention_blocks: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    """FFN expansion factor inside each MHSA block."""
    num_decoder_blocks: int = 2
    """Cross-attention decoder blocks over the post-MHSA token grid."""
    decoder_num_heads: int = 4
    decoder_mlp_ratio: float = 4.0
    mode_pos_encoding: Literal["sincos", "learned"] = "sincos"
    """Encoding for the *signed* mode index in `[-N_max, N_max]`.

    `sincos`
    generalises smoothly to unseen N within the trained band; `learned` uses a
    table of size `(2*N_max + 1, hidden_dim)`.
    """
    dropout: float = 0.0
    pos_encoding: Literal["learned", "sincos"] = "sincos"


AttentionCoilPredictorCheckpoint = flax_nnx_checkpoint_types.FlaxNnxCheckpoint[
    AttentionCoilPredictorConfig
]


@register_pydantic_data
class Batch(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    boundaries: surface_rz_fourier.SurfaceRZFourier
    """Batched surface; leaves have leading dim ``batch_size``."""
    coilsets: constellaration_update_types.ConStellarationUpdateCoilset
    """Batched coilset (padded to ``2*batch_max_fourier_order + 1`` modes); leaves have
    leading dim ``batch_size``."""
    requirement_metrics: constellaration_update_types.RequirementMetrics
    """Batched conditioning; each scalar field has leading dim ``batch_size``."""
    fourier_order_mask: Float[NpOrJaxArray, "batch_size n_modes"]
    """1.0 at real coefficient indices, 0.0 at zero-padded indices.

    `n_modes = 2 * batch_max_fourier_order + 1`. Created at collate time so
    that loss can mask per-sample padded entries.
    """
    batch_max_fourier_order: int
    """Per-batch fourier order N.

    Every coilset in the batch is padded to `(2*N + 1)` modes.
    """


class TrainConfig(pydantic.BaseModel):
    """Everything that controls a training run."""

    seed: int = pydantic.Field(default_factory=lambda: int(time.time_ns() % (2**32)))
    learning_rate: float = 1e-3
    end_lr: float = 1e-5
    warmup_steps: int = 50
    steps: int = 1_000
    batch_size: int = 512
    eval_batch_size: int = 1024
    eval_every: int = 50
    early_stopping_patience: int | None = None
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    relative_min_coil_to_plasma_distance_error_threshold: float = -0.4

    model_type: Literal["mlp", "attention"] = "mlp"
    """Which model architecture to instantiate.

    Drives `train()` dispatch and the default config type used when
    `model_config_overrides` is None.
    """

    model_config_overrides: (
        CoilPredictorConfig | AttentionCoilPredictorConfig | None
    ) = None
    """Architecture knobs only. Concrete type must agree with ``model_type``.

    After the dataset scan, a fully-resolved
    config (with ``n_field_periods``, ``n_unique_coils``, ``n_modes_coils_max``,
    ``expected_currents``, ``is_stellarator_symmetric``, ``n_poloidal_modes``,
    ``n_toroidal_modes`` filled in) is re-emitted and stored next to the
    checkpoint. Named ``model_config_overrides`` to avoid shadowing
    pydantic's reserved ``model_config`` attribute on ``BaseModel``.
    """

    @pydantic.model_validator(mode="after")
    def _check_overrides_match_model_type(self) -> Self:
        if self.model_config_overrides is None:
            return self
        if self.model_type == "mlp" and not isinstance(
            self.model_config_overrides, CoilPredictorConfig
        ):
            raise ValueError(
                "model_type='mlp' requires CoilPredictorConfig overrides; got "
                f"{type(self.model_config_overrides).__name__}."
            )
        if self.model_type == "attention" and not isinstance(
            self.model_config_overrides, AttentionCoilPredictorConfig
        ):
            raise ValueError(
                "model_type='attention' requires AttentionCoilPredictorConfig "
                f"overrides; got {type(self.model_config_overrides).__name__}."
            )
        return self


class EvalSettings(pydantic.BaseModel):
    """Inputs that fully determine an evaluation run."""

    relative_min_coil_to_plasma_distance_error_threshold: float = -0.4
    n_eval: int = 128


class PlotSettings(pydantic.BaseModel):
    """Styling/rendering knobs for `render_eval_plots`.

    Decoupled from
    `EvalSettings` so re-rendering does not invalidate the eval cache.
    """

    poincare_settings: (
        constellaration_update_types.ConStellarationUpdatePoincarePlotSettings
    )


class EvalData(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    boundary: surface_rz_fourier.SurfaceRZFourier
    boundary_id: str
    true_coilset: constellaration_update_types.ConStellarationUpdateCoilset
    true_metrics: str
    requirement_metrics: constellaration_update_types.RequirementMetrics
    predicted_coilset: Optional[
        constellaration_update_types.ConStellarationUpdateCoilset
    ] = None


class CompareSettings(pydantic.BaseModel):
    """Inputs that fully determine a `compare_runs` invocation."""

    intersect_on_plasma_id: bool = True
    """Restrict all runs to the intersection of plasma_configuration_ids — fair pairwise
    comparison."""

    summary_quantiles: list[float] = pydantic.Field(default_factory=lambda: [0.5, 0.95])
    per_coil_aggregator: Literal["mean", "max"] = "mean"

    bootstrap_resamples: int = 10_000
    """Number of bootstrap resamples for error-summary 95% CIs.

    Set to 0 to skip.
    """

    bootstrap_seed: int = 0
    """RNG seed for bootstrap resampling so CIs are reproducible across reruns."""
