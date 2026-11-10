from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import jaxtyping as jt
from constellaration.geometry import surface_rz_fourier
from flax import nnx

from coilstellaration import (
    coilset_utils,
    flax_nnx_checkpoint_util,
    types,
)
from coilstellaration.machine_learning import (
    utils,
)
from coilstellaration.types import NpOrJaxArray

AnyVariableShapeConfig = (
    types.PerCoilCoilPredictorConfig
    | types.MaskedCoilPredictorConfig
    | types.PerCoilEnsembleCoilPredictorConfig
    | types.MaskedEnsembleCoilPredictorConfig
)


class CoilTransform(Protocol):
    """Maps boundaries and coilsets to and from the flat vectors the networks use.

    Lengths are in units of the boundary's major radius (see `utils.major_radius`), and
    so are currents: scaling a device scales the currents that produce the same field.
    The networks therefore see every boundary at unit major radius. Every Fourier
    coefficient is also divided by its exponential spectral scaling, and currents are
    standardized with the configured mean and std. `backward_coilset` is the inverse
    of `forward_coilset`. The training loss compares coilsets in physical space, not
    in this representation.
    """

    @types.runtime_check_array_sizes
    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]: ...

    @types.runtime_check_array_sizes
    def forward_coilset(
        self,
        coilset: types.Coilset,
        major_radius: types.ScalarFloat,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]: ...

    @types.runtime_check_array_sizes
    def backward_coilset(
        self,
        flat: jt.Float[jt.Array, " n_flat_coilset"],
        n_field_periods: types.ScalarInt,
        major_radius: types.ScalarFloat,
    ) -> types.Coilset: ...


class CoilPredictor(nnx.Module):
    """Basic MLP predictor: input → `num_layers` GELU + linear layers → flat readout.

    Single readout layer produces the concatenated `[coils, currents]` flat
    vector that `backward_coilset` decodes. Treated as a baseline against
    `ResMlpCoilPredictor`, which adds pre-norm residual blocks, dropout, and
    a separate currents head on top of the same I/O contract.
    """

    config: types.CoilPredictorConfig

    def __init__(self, config: types.CoilPredictorConfig, *, rngs: nnx.Rngs):
        self.config = config
        n_arrays = 2 if config.is_stellarator_symmetric else 4
        input_size = config.n_poloidal_modes * config.n_toroidal_modes * n_arrays + len(
            config.requirement_metrics_means
        )
        flat_size = (
            config.n_unique_coils * config.n_modes_coils_max * 3 + config.n_unique_coils
        )
        self.requirement_metrics_means = nnx.Variable(
            jnp.asarray(config.requirement_metrics_means)
        )
        self.requirement_metrics_stds = nnx.Variable(
            jnp.asarray(config.requirement_metrics_stds)
        )
        self.input_layer = nnx.Linear(input_size, config.hidden_dim, rngs=rngs)
        self.block_linears = nnx.List(
            [
                nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
                for _ in range(config.num_layers - 1)
            ]
        )
        self.output_layer = nnx.Linear(config.hidden_dim, flat_size, rngs=rngs)

    @types.runtime_check_array_sizes
    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]:
        return utils.forward_boundary_to_flat(
            boundary,
            alpha=self.config.spectral_scaling_alpha,
            order=self.config.spectral_scaling_order,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def forward_coilset(
        self,
        coilset: types.Coilset,
        major_radius: types.ScalarFloat,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            major_radius=major_radius,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def backward_coilset(
        self,
        flat: jt.Float[jt.Array, " n_flat_coilset"],
        n_field_periods: types.ScalarInt,
        major_radius: types.ScalarFloat,
    ) -> types.Coilset:
        return utils.backward_flat_to_coilset(
            flat,
            major_radius=major_radius,
            n_unique_coils=self.config.n_unique_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            n_field_periods=n_field_periods,
            is_stellarator_symmetric=self.config.is_stellarator_symmetric,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def __call__(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> types.Coilset:
        flat_boundary = self.forward_boundary(boundary)
        standardized = utils.z_score_normalization(
            requirement_metrics_vector(requirement_metrics),
            self.requirement_metrics_means[...],
            self.requirement_metrics_stds[...],
        )
        x = jnp.concatenate([flat_boundary, standardized])
        x = self.input_layer(x)
        for lin in self.block_linears:
            x = nnx.gelu(x)
            x = lin(x)
        x = nnx.gelu(x)
        x = self.output_layer(x)
        full = self.backward_coilset(
            x, boundary.n_field_periods, utils.major_radius(boundary)
        )
        return utils.truncate_to_fourier_order(
            full,
            n_modes_coils_max=self.config.n_modes_coils_max,
            fourier_order=fourier_order,
        )


class _CurrentsHead(nnx.Module):
    """Bottleneck MLP that predicts per-coil currents from coil shapes + metrics.

    Every dataset coilset is scaled to the same nominal field strength, so the
    currents are largely set by the coil geometry: this head sees the predicted
    `flat_coils` plus the standardized metrics. The output layer starts at zero, so
    the initial prediction is the mean current (`backward_coilset` maps
    `flat_currents=0` to `coil_currents_mean`); any movement must be earned from
    gradient signal. The narrow `hidden_dim` caps the capacity dedicated to
    currents; the currents loss still trains the trunk through `flat_coils`.
    """

    def __init__(
        self,
        in_features: int,
        n_unique_coils: int,
        *,
        hidden_dim: int = 64,
        rngs: nnx.Rngs,
    ):
        self.norm = nnx.LayerNorm(in_features, rngs=rngs)
        self.fc1 = nnx.Linear(in_features, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(
            hidden_dim,
            n_unique_coils,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    @types.runtime_check_array_sizes
    def __call__(
        self,
        flat_coils: jt.Float[NpOrJaxArray, " n_flat_coils"],
        standardized_metrics: jt.Float[NpOrJaxArray, " n_metrics"],
    ) -> jt.Float[jt.Array, " n_unique_coils"]:
        h = jnp.concatenate([flat_coils, standardized_metrics])
        h = self.norm(h)
        h = self.fc1(h)
        h = nnx.gelu(h)
        return self.fc2(h)


class ResMlpCoilPredictor(nnx.Module):
    """MLP predictor with pre-norm residual blocks, dropout, and split readout heads.

    Backbone is `num_layers - 1` pre-norm residual blocks (LayerNorm → GELU →
    Linear → Dropout, then `+residual`), followed by a final GELU + linear coil
    head. A separate `_CurrentsHead` predicts currents from the predicted coil
    geometry plus standardized metrics; see its docstring for why currents are
    inverted from coils-and-metrics rather than predicted directly from the
    boundary trunk.
    """

    config: types.ResMlpCoilPredictorConfig

    def __init__(self, config: types.ResMlpCoilPredictorConfig, *, rngs: nnx.Rngs):
        self.config = config
        n_arrays = 2 if config.is_stellarator_symmetric else 4
        input_size = config.n_poloidal_modes * config.n_toroidal_modes * n_arrays + len(
            config.requirement_metrics_means
        )
        coil_flat_size = config.n_unique_coils * config.n_modes_coils_max * 3
        n_cond = len(config.requirement_metrics_means)
        self.requirement_metrics_means = nnx.Variable(
            jnp.asarray(config.requirement_metrics_means)
        )
        self.requirement_metrics_stds = nnx.Variable(
            jnp.asarray(config.requirement_metrics_stds)
        )
        self.input_layer = nnx.Linear(input_size, config.hidden_dim, rngs=rngs)
        self.block_norms = nnx.List(
            [
                nnx.LayerNorm(config.hidden_dim, rngs=rngs)
                for _ in range(config.num_layers - 1)
            ]
        )
        self.block_linears = nnx.List(
            [
                nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
                for _ in range(config.num_layers - 1)
            ]
        )
        self.dropout = nnx.Dropout(config.dropout, rngs=rngs)
        # The residual stream is not normalized before this head, so at the default
        # (LeCun normal) scale the initial loss is ~10x that of `CoilPredictor` and
        # gradient clipping shrinks the first updates to a few percent.
        self.coil_head = nnx.Linear(
            config.hidden_dim,
            coil_flat_size,
            kernel_init=nnx.initializers.variance_scaling(
                0.01, "fan_in", "truncated_normal"
            ),
            rngs=rngs,
        )
        self.currents_head = _CurrentsHead(
            in_features=coil_flat_size + n_cond,
            n_unique_coils=config.n_unique_coils,
            rngs=rngs,
        )

    @types.runtime_check_array_sizes
    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]:
        return utils.forward_boundary_to_flat(
            boundary,
            alpha=self.config.spectral_scaling_alpha,
            order=self.config.spectral_scaling_order,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def forward_coilset(
        self,
        coilset: types.Coilset,
        major_radius: types.ScalarFloat,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            major_radius=major_radius,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def backward_coilset(
        self,
        flat: jt.Float[jt.Array, " n_flat_coilset"],
        n_field_periods: types.ScalarInt,
        major_radius: types.ScalarFloat,
    ) -> types.Coilset:
        return utils.backward_flat_to_coilset(
            flat,
            major_radius=major_radius,
            n_unique_coils=self.config.n_unique_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            n_field_periods=n_field_periods,
            is_stellarator_symmetric=self.config.is_stellarator_symmetric,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def __call__(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> types.Coilset:
        flat_boundary = self.forward_boundary(boundary)
        standardized = utils.z_score_normalization(
            requirement_metrics_vector(requirement_metrics),
            self.requirement_metrics_means[...],
            self.requirement_metrics_stds[...],
        )
        x = jnp.concatenate([flat_boundary, standardized])
        x = self.input_layer(x)
        for norm, lin in zip(self.block_norms, self.block_linears):
            residual = x
            x = norm(x)
            x = nnx.gelu(x)
            x = lin(x)
            x = self.dropout(x)
            x = x + residual
        x = nnx.gelu(x)
        flat_coils = self.coil_head(x)
        flat_currents = self.currents_head(flat_coils, standardized)
        full = self.backward_coilset(
            jnp.concatenate([flat_coils, flat_currents]),
            boundary.n_field_periods,
            utils.major_radius(boundary),
        )
        return utils.truncate_to_fourier_order(
            full,
            n_modes_coils_max=self.config.n_modes_coils_max,
            fourier_order=fourier_order,
        )


class MlpEnsembleCoilPredictor(nnx.Module):
    """Ensemble of independent basic-MLP `CoilPredictor`s averaged in coilset space.

    Each member is an independent `CoilPredictor` sharing the same architecture
    config but with distinct random init drawn from the shared `rngs` stream.
    Predictions are averaged elementwise across the ensemble for each Fourier
    coefficient and current; this is equivalent to averaging per-member flat
    outputs because `backward_coilset` is affine in the flat representation, with
    the same offset for every member, and `utils.truncate_to_fourier_order` is a
    slice. The ensemble therefore inherits the `CoilTransform` unchanged from any
    single member.
    """

    config: types.MlpEnsembleCoilPredictorConfig

    def __init__(self, config: types.MlpEnsembleCoilPredictorConfig, *, rngs: nnx.Rngs):
        self.config = config
        member_config = types.CoilPredictorConfig(
            is_stellarator_symmetric=config.is_stellarator_symmetric,
            n_field_periods=config.n_field_periods,
            n_poloidal_modes=config.n_poloidal_modes,
            n_toroidal_modes=config.n_toroidal_modes,
            n_unique_coils=config.n_unique_coils,
            n_modes_coils_max=config.n_modes_coils_max,
            coil_currents_mean=config.coil_currents_mean,
            coil_currents_std=config.coil_currents_std,
            requirement_metrics_means=config.requirement_metrics_means,
            requirement_metrics_stds=config.requirement_metrics_stds,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            spectral_scaling_alpha=config.spectral_scaling_alpha,
            spectral_scaling_min=config.spectral_scaling_min,
            spectral_scaling_order=config.spectral_scaling_order,
        )
        self.members = nnx.List(
            [
                CoilPredictor(member_config, rngs=rngs)
                for _ in range(config.n_ensemble_members)
            ]
        )

    @types.runtime_check_array_sizes
    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]:
        return utils.forward_boundary_to_flat(
            boundary,
            alpha=self.config.spectral_scaling_alpha,
            order=self.config.spectral_scaling_order,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def forward_coilset(
        self,
        coilset: types.Coilset,
        major_radius: types.ScalarFloat,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            major_radius=major_radius,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def backward_coilset(
        self,
        flat: jt.Float[jt.Array, " n_flat_coilset"],
        n_field_periods: types.ScalarInt,
        major_radius: types.ScalarFloat,
    ) -> types.Coilset:
        return utils.backward_flat_to_coilset(
            flat,
            major_radius=major_radius,
            n_unique_coils=self.config.n_unique_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            n_field_periods=n_field_periods,
            is_stellarator_symmetric=self.config.is_stellarator_symmetric,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def predict_per_member(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> types.Coilset:
        """Per-member predictions stacked along a leading `n_members` axis.

        Used at training time so the loss can grade each member against the
        target independently. Loss-of-mean (the formulation in `__call__`)
        only pressures the *averaged* prediction to fit, which lets members
        co-adapt and erode the variance-reduction benefit of an ensemble;
        per-member loss decomposes as `(1/M) Σ_m MSE(p_m, target)` with no
        cross-member gradient coupling, so members train as independent
        models that share a single forward sweep over the batch.
        """
        predictions = [
            member(boundary, requirement_metrics, fourier_order=fourier_order)
            for member in self.members
        ]
        return predictions[0].model_copy(
            update={
                "coil_x_n": jnp.stack([p.coil_x_n for p in predictions]),
                "coil_y_n": jnp.stack([p.coil_y_n for p in predictions]),
                "coil_z_n": jnp.stack([p.coil_z_n for p in predictions]),
                "currents": jnp.stack([p.currents for p in predictions]),
            }
        )

    def __call__(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> types.Coilset:
        per_member = self.predict_per_member(
            boundary, requirement_metrics, fourier_order=fourier_order
        )
        return per_member.model_copy(
            update={
                "coil_x_n": jnp.mean(per_member.coil_x_n, axis=0),
                "coil_y_n": jnp.mean(per_member.coil_y_n, axis=0),
                "coil_z_n": jnp.mean(per_member.coil_z_n, axis=0),
                "currents": jnp.mean(per_member.currents, axis=0),
            }
        )


class ResMlpEnsembleCoilPredictor(nnx.Module):
    """Ensemble of independent `ResMlpCoilPredictor`s averaged in coilset space.

    Sibling of `MlpEnsembleCoilPredictor` whose members are advanced
    pre-norm/residual/dropout MLPs with split readout heads. Mirror the
    averaging argument of `MlpEnsembleCoilPredictor` for justification of
    coilset-space averaging.
    """

    config: types.ResMlpEnsembleCoilPredictorConfig

    def __init__(
        self, config: types.ResMlpEnsembleCoilPredictorConfig, *, rngs: nnx.Rngs
    ):
        self.config = config
        member_config = types.ResMlpCoilPredictorConfig(
            is_stellarator_symmetric=config.is_stellarator_symmetric,
            n_field_periods=config.n_field_periods,
            n_poloidal_modes=config.n_poloidal_modes,
            n_toroidal_modes=config.n_toroidal_modes,
            n_unique_coils=config.n_unique_coils,
            n_modes_coils_max=config.n_modes_coils_max,
            coil_currents_mean=config.coil_currents_mean,
            coil_currents_std=config.coil_currents_std,
            requirement_metrics_means=config.requirement_metrics_means,
            requirement_metrics_stds=config.requirement_metrics_stds,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            spectral_scaling_alpha=config.spectral_scaling_alpha,
            spectral_scaling_min=config.spectral_scaling_min,
            spectral_scaling_order=config.spectral_scaling_order,
        )
        self.members = nnx.List(
            [
                ResMlpCoilPredictor(member_config, rngs=rngs)
                for _ in range(config.n_ensemble_members)
            ]
        )

    @types.runtime_check_array_sizes
    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]:
        return utils.forward_boundary_to_flat(
            boundary,
            alpha=self.config.spectral_scaling_alpha,
            order=self.config.spectral_scaling_order,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def forward_coilset(
        self,
        coilset: types.Coilset,
        major_radius: types.ScalarFloat,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            major_radius=major_radius,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def backward_coilset(
        self,
        flat: jt.Float[jt.Array, " n_flat_coilset"],
        n_field_periods: types.ScalarInt,
        major_radius: types.ScalarFloat,
    ) -> types.Coilset:
        return utils.backward_flat_to_coilset(
            flat,
            major_radius=major_radius,
            n_unique_coils=self.config.n_unique_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            n_field_periods=n_field_periods,
            is_stellarator_symmetric=self.config.is_stellarator_symmetric,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def predict_per_member(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> types.Coilset:
        """Per-member predictions stacked along a leading `n_members` axis.

        See `MlpEnsembleCoilPredictor.predict_per_member` for the rationale of
        per-member loss vs loss-of-mean.
        """
        predictions = [
            member(boundary, requirement_metrics, fourier_order=fourier_order)
            for member in self.members
        ]
        return predictions[0].model_copy(
            update={
                "coil_x_n": jnp.stack([p.coil_x_n for p in predictions]),
                "coil_y_n": jnp.stack([p.coil_y_n for p in predictions]),
                "coil_z_n": jnp.stack([p.coil_z_n for p in predictions]),
                "currents": jnp.stack([p.currents for p in predictions]),
            }
        )

    def __call__(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> types.Coilset:
        per_member = self.predict_per_member(
            boundary, requirement_metrics, fourier_order=fourier_order
        )
        return per_member.model_copy(
            update={
                "coil_x_n": jnp.mean(per_member.coil_x_n, axis=0),
                "coil_y_n": jnp.mean(per_member.coil_y_n, axis=0),
                "coil_z_n": jnp.mean(per_member.coil_z_n, axis=0),
                "currents": jnp.mean(per_member.currents, axis=0),
            }
        )


class PerCoilCoilPredictor(nnx.Module):
    """Shared MLP applied to every coil slot with slot-position features.

    Each slot sees the design context plus its one-hot slot index and its
    normalized position `(slot + 0.5) / n_coils`; the network weights are shared
    across slots. Returns `max_coils` rows; rows beyond the requested number of
    coils are meaningless and are expected to be masked by the caller.
    """

    config: types.PerCoilCoilPredictorConfig

    def __init__(self, config: types.PerCoilCoilPredictorConfig, *, rngs: nnx.Rngs):
        self.config = config
        n_arrays = 2 if config.is_stellarator_symmetric else 4
        n_context = (
            config.n_poloidal_modes * config.n_toroidal_modes * n_arrays
            + len(config.requirement_metrics_means)
            + len(config.n_field_periods_values)
            + len(config.n_coils_values)
        )
        input_size = n_context + config.max_coils + 1
        slot_size = config.n_modes_coils_max * 3 + 1
        self.requirement_metrics_means = nnx.Variable(
            jnp.asarray(config.requirement_metrics_means)
        )
        self.requirement_metrics_stds = nnx.Variable(
            jnp.asarray(config.requirement_metrics_stds)
        )
        self.input_layer = nnx.Linear(input_size, config.hidden_dim, rngs=rngs)
        self.block_linears = nnx.List(
            [
                nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
                for _ in range(config.num_layers - 1)
            ]
        )
        self.output_layer = nnx.Linear(config.hidden_dim, slot_size, rngs=rngs)

    @types.runtime_check_array_sizes
    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]:
        return utils.forward_boundary_to_flat(
            boundary,
            alpha=self.config.spectral_scaling_alpha,
            order=self.config.spectral_scaling_order,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def forward_coilset(
        self,
        coilset: types.Coilset,
        major_radius: types.ScalarFloat,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            major_radius=major_radius,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def backward_coilset(
        self,
        flat: jt.Float[jt.Array, " n_flat_coilset"],
        n_field_periods: types.ScalarInt,
        major_radius: types.ScalarFloat,
    ) -> types.Coilset:
        return utils.backward_flat_to_coilset(
            flat,
            major_radius=major_radius,
            n_unique_coils=self.config.max_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            n_field_periods=n_field_periods,
            is_stellarator_symmetric=self.config.is_stellarator_symmetric,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def __call__(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> types.Coilset:
        flat_boundary = self.forward_boundary(boundary)
        standardized = utils.z_score_normalization(
            requirement_metrics_vector(requirement_metrics),
            self.requirement_metrics_means[...],
            self.requirement_metrics_stds[...],
        )
        context = jnp.concatenate(
            [
                flat_boundary,
                standardized,
                (
                    jnp.asarray(self.config.n_field_periods_values)
                    == boundary.n_field_periods
                ).astype(flat_boundary.dtype),
                (
                    jnp.asarray(self.config.n_coils_values)
                    == requirement_metrics.n_coils_per_half_period
                ).astype(flat_boundary.dtype),
            ]
        )
        max_coils = self.config.max_coils
        n_coils = jnp.asarray(
            requirement_metrics.n_coils_per_half_period, dtype=context.dtype
        )
        slots = jnp.arange(max_coils)
        position = jnp.concatenate(
            [
                jnp.eye(max_coils, dtype=context.dtype),
                ((slots + 0.5) / n_coils)[:, None].astype(context.dtype),
            ],
            axis=1,
        )
        x = jnp.concatenate(
            [jnp.broadcast_to(context, (max_coils, context.shape[0])), position],
            axis=1,
        )
        x = self.input_layer(x)
        for lin in self.block_linears:
            x = nnx.gelu(x)
            x = lin(x)
        x = nnx.gelu(x)
        x = self.output_layer(x)
        # Each slot row is `[coil_x_n | coil_y_n | coil_z_n | current]`; regroup into
        # the coefficient-major layout that `backward_coilset` expects.
        n_modes = self.config.n_modes_coils_max
        coeffs = x[:, : 3 * n_modes].reshape(max_coils, 3, n_modes)
        flat = jnp.concatenate([jnp.swapaxes(coeffs, 0, 1).ravel(), x[:, 3 * n_modes]])
        full = self.backward_coilset(
            flat, boundary.n_field_periods, utils.major_radius(boundary)
        )
        return utils.truncate_to_fourier_order(
            full,
            n_modes_coils_max=self.config.n_modes_coils_max,
            fourier_order=fourier_order,
        )


class MaskedCoilPredictor(nnx.Module):
    """MLP that predicts all `max_coils` slots at once from the design context.

    Rows beyond the requested number of coils are meaningless and are expected to be
    masked by the caller.
    """

    config: types.MaskedCoilPredictorConfig

    def __init__(self, config: types.MaskedCoilPredictorConfig, *, rngs: nnx.Rngs):
        self.config = config
        n_arrays = 2 if config.is_stellarator_symmetric else 4
        input_size = (
            config.n_poloidal_modes * config.n_toroidal_modes * n_arrays
            + len(config.requirement_metrics_means)
            + len(config.n_field_periods_values)
            + len(config.n_coils_values)
        )
        flat_size = config.max_coils * config.n_modes_coils_max * 3 + config.max_coils
        self.requirement_metrics_means = nnx.Variable(
            jnp.asarray(config.requirement_metrics_means)
        )
        self.requirement_metrics_stds = nnx.Variable(
            jnp.asarray(config.requirement_metrics_stds)
        )
        self.input_layer = nnx.Linear(input_size, config.hidden_dim, rngs=rngs)
        self.block_linears = nnx.List(
            [
                nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
                for _ in range(config.num_layers - 1)
            ]
        )
        self.output_layer = nnx.Linear(config.hidden_dim, flat_size, rngs=rngs)

    @types.runtime_check_array_sizes
    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]:
        return utils.forward_boundary_to_flat(
            boundary,
            alpha=self.config.spectral_scaling_alpha,
            order=self.config.spectral_scaling_order,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def forward_coilset(
        self,
        coilset: types.Coilset,
        major_radius: types.ScalarFloat,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            major_radius=major_radius,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def backward_coilset(
        self,
        flat: jt.Float[jt.Array, " n_flat_coilset"],
        n_field_periods: types.ScalarInt,
        major_radius: types.ScalarFloat,
    ) -> types.Coilset:
        return utils.backward_flat_to_coilset(
            flat,
            major_radius=major_radius,
            n_unique_coils=self.config.max_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            n_field_periods=n_field_periods,
            is_stellarator_symmetric=self.config.is_stellarator_symmetric,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def __call__(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> types.Coilset:
        flat_boundary = self.forward_boundary(boundary)
        standardized = utils.z_score_normalization(
            requirement_metrics_vector(requirement_metrics),
            self.requirement_metrics_means[...],
            self.requirement_metrics_stds[...],
        )
        x = jnp.concatenate(
            [
                flat_boundary,
                standardized,
                (
                    jnp.asarray(self.config.n_field_periods_values)
                    == boundary.n_field_periods
                ).astype(flat_boundary.dtype),
                (
                    jnp.asarray(self.config.n_coils_values)
                    == requirement_metrics.n_coils_per_half_period
                ).astype(flat_boundary.dtype),
            ]
        )
        x = self.input_layer(x)
        for lin in self.block_linears:
            x = nnx.gelu(x)
            x = lin(x)
        x = nnx.gelu(x)
        x = self.output_layer(x)
        full = self.backward_coilset(
            x, boundary.n_field_periods, utils.major_radius(boundary)
        )
        return utils.truncate_to_fourier_order(
            full,
            n_modes_coils_max=self.config.n_modes_coils_max,
            fourier_order=fourier_order,
        )


class PerCoilEnsembleCoilPredictor(nnx.Module):
    """Ensemble of independent `PerCoilCoilPredictor`s averaged in coilset space.

    Members share the architecture and differ in random init. Averaging is elementwise
    over every coefficient and current; see `MlpEnsembleCoilPredictor` for why this
    matches averaging in the flat network-IO space.
    """

    config: types.PerCoilEnsembleCoilPredictorConfig

    def __init__(
        self, config: types.PerCoilEnsembleCoilPredictorConfig, *, rngs: nnx.Rngs
    ):
        self.config = config
        member_config = types.PerCoilCoilPredictorConfig(
            is_stellarator_symmetric=config.is_stellarator_symmetric,
            n_poloidal_modes=config.n_poloidal_modes,
            n_toroidal_modes=config.n_toroidal_modes,
            n_modes_coils_max=config.n_modes_coils_max,
            coil_currents_mean=config.coil_currents_mean,
            coil_currents_std=config.coil_currents_std,
            n_field_periods_values=config.n_field_periods_values,
            n_coils_values=config.n_coils_values,
            max_coils=config.max_coils,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            requirement_metrics_means=config.requirement_metrics_means,
            requirement_metrics_stds=config.requirement_metrics_stds,
            spectral_scaling_alpha=config.spectral_scaling_alpha,
            spectral_scaling_min=config.spectral_scaling_min,
            spectral_scaling_order=config.spectral_scaling_order,
        )
        self.members = nnx.List(
            [
                PerCoilCoilPredictor(member_config, rngs=rngs)
                for _ in range(config.n_ensemble_members)
            ]
        )

    @types.runtime_check_array_sizes
    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]:
        return utils.forward_boundary_to_flat(
            boundary,
            alpha=self.config.spectral_scaling_alpha,
            order=self.config.spectral_scaling_order,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def forward_coilset(
        self,
        coilset: types.Coilset,
        major_radius: types.ScalarFloat,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            major_radius=major_radius,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def backward_coilset(
        self,
        flat: jt.Float[jt.Array, " n_flat_coilset"],
        n_field_periods: types.ScalarInt,
        major_radius: types.ScalarFloat,
    ) -> types.Coilset:
        return utils.backward_flat_to_coilset(
            flat,
            major_radius=major_radius,
            n_unique_coils=self.config.max_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            n_field_periods=n_field_periods,
            is_stellarator_symmetric=self.config.is_stellarator_symmetric,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def predict_per_member(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> types.Coilset:
        """Per-member predictions stacked along a leading `n_members` axis.

        See `MlpEnsembleCoilPredictor.predict_per_member` for the rationale of
        per-member loss vs loss-of-mean.
        """
        predictions = [
            member(boundary, requirement_metrics, fourier_order=fourier_order)
            for member in self.members
        ]
        return predictions[0].model_copy(
            update={
                "coil_x_n": jnp.stack([p.coil_x_n for p in predictions]),
                "coil_y_n": jnp.stack([p.coil_y_n for p in predictions]),
                "coil_z_n": jnp.stack([p.coil_z_n for p in predictions]),
                "currents": jnp.stack([p.currents for p in predictions]),
            }
        )

    def __call__(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> types.Coilset:
        per_member = self.predict_per_member(
            boundary, requirement_metrics, fourier_order=fourier_order
        )
        return per_member.model_copy(
            update={
                "coil_x_n": jnp.mean(per_member.coil_x_n, axis=0),
                "coil_y_n": jnp.mean(per_member.coil_y_n, axis=0),
                "coil_z_n": jnp.mean(per_member.coil_z_n, axis=0),
                "currents": jnp.mean(per_member.currents, axis=0),
            }
        )


class MaskedEnsembleCoilPredictor(nnx.Module):
    """Ensemble of independent `MaskedCoilPredictor`s averaged in coilset space.

    Members share the architecture and differ in random init. Averaging is elementwise
    over every coefficient and current; see `MlpEnsembleCoilPredictor` for why this
    matches averaging in the flat network-IO space.
    """

    config: types.MaskedEnsembleCoilPredictorConfig

    def __init__(
        self, config: types.MaskedEnsembleCoilPredictorConfig, *, rngs: nnx.Rngs
    ):
        self.config = config
        member_config = types.MaskedCoilPredictorConfig(
            is_stellarator_symmetric=config.is_stellarator_symmetric,
            n_poloidal_modes=config.n_poloidal_modes,
            n_toroidal_modes=config.n_toroidal_modes,
            n_modes_coils_max=config.n_modes_coils_max,
            coil_currents_mean=config.coil_currents_mean,
            coil_currents_std=config.coil_currents_std,
            n_field_periods_values=config.n_field_periods_values,
            n_coils_values=config.n_coils_values,
            max_coils=config.max_coils,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            requirement_metrics_means=config.requirement_metrics_means,
            requirement_metrics_stds=config.requirement_metrics_stds,
            spectral_scaling_alpha=config.spectral_scaling_alpha,
            spectral_scaling_min=config.spectral_scaling_min,
            spectral_scaling_order=config.spectral_scaling_order,
        )
        self.members = nnx.List(
            [
                MaskedCoilPredictor(member_config, rngs=rngs)
                for _ in range(config.n_ensemble_members)
            ]
        )

    @types.runtime_check_array_sizes
    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]:
        return utils.forward_boundary_to_flat(
            boundary,
            alpha=self.config.spectral_scaling_alpha,
            order=self.config.spectral_scaling_order,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def forward_coilset(
        self,
        coilset: types.Coilset,
        major_radius: types.ScalarFloat,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            major_radius=major_radius,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    @types.runtime_check_array_sizes
    def backward_coilset(
        self,
        flat: jt.Float[jt.Array, " n_flat_coilset"],
        n_field_periods: types.ScalarInt,
        major_radius: types.ScalarFloat,
    ) -> types.Coilset:
        return utils.backward_flat_to_coilset(
            flat,
            major_radius=major_radius,
            n_unique_coils=self.config.max_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            n_field_periods=n_field_periods,
            is_stellarator_symmetric=self.config.is_stellarator_symmetric,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def predict_per_member(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> types.Coilset:
        """Per-member predictions stacked along a leading `n_members` axis.

        See `MlpEnsembleCoilPredictor.predict_per_member` for the rationale of
        per-member loss vs loss-of-mean.
        """
        predictions = [
            member(boundary, requirement_metrics, fourier_order=fourier_order)
            for member in self.members
        ]
        return predictions[0].model_copy(
            update={
                "coil_x_n": jnp.stack([p.coil_x_n for p in predictions]),
                "coil_y_n": jnp.stack([p.coil_y_n for p in predictions]),
                "coil_z_n": jnp.stack([p.coil_z_n for p in predictions]),
                "currents": jnp.stack([p.currents for p in predictions]),
            }
        )

    def __call__(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> types.Coilset:
        per_member = self.predict_per_member(
            boundary, requirement_metrics, fourier_order=fourier_order
        )
        return per_member.model_copy(
            update={
                "coil_x_n": jnp.mean(per_member.coil_x_n, axis=0),
                "coil_y_n": jnp.mean(per_member.coil_y_n, axis=0),
                "coil_z_n": jnp.mean(per_member.coil_z_n, axis=0),
                "currents": jnp.mean(per_member.currents, axis=0),
            }
        )


AnyEnsembleModel = (
    MlpEnsembleCoilPredictor
    | ResMlpEnsembleCoilPredictor
    | PerCoilEnsembleCoilPredictor
    | MaskedEnsembleCoilPredictor
)
AnyModel = (
    CoilPredictor
    | ResMlpCoilPredictor
    | PerCoilCoilPredictor
    | MaskedCoilPredictor
    | AnyEnsembleModel
)

if TYPE_CHECKING:

    def _(x: AnyModel) -> CoilTransform:
        return x


def read_model_from_checkpoint(
    checkpoint: types.AnyModelCheckpoint,
) -> AnyModel:
    if isinstance(checkpoint.config, types.CoilPredictorConfig):
        model = flax_nnx_checkpoint_util.from_checkpoint(checkpoint, CoilPredictor)
    elif isinstance(checkpoint.config, types.MlpEnsembleCoilPredictorConfig):
        model = flax_nnx_checkpoint_util.from_checkpoint(
            checkpoint, MlpEnsembleCoilPredictor
        )
    elif isinstance(checkpoint.config, types.ResMlpCoilPredictorConfig):
        model = flax_nnx_checkpoint_util.from_checkpoint(
            checkpoint, ResMlpCoilPredictor
        )
    elif isinstance(checkpoint.config, types.ResMlpEnsembleCoilPredictorConfig):
        model = flax_nnx_checkpoint_util.from_checkpoint(
            checkpoint, ResMlpEnsembleCoilPredictor
        )
    elif isinstance(checkpoint.config, types.PerCoilCoilPredictorConfig):
        model = flax_nnx_checkpoint_util.from_checkpoint(
            checkpoint, PerCoilCoilPredictor
        )
    elif isinstance(checkpoint.config, types.MaskedCoilPredictorConfig):
        model = flax_nnx_checkpoint_util.from_checkpoint(
            checkpoint, MaskedCoilPredictor
        )
    elif isinstance(checkpoint.config, types.PerCoilEnsembleCoilPredictorConfig):
        model = flax_nnx_checkpoint_util.from_checkpoint(
            checkpoint, PerCoilEnsembleCoilPredictor
        )
    elif isinstance(checkpoint.config, types.MaskedEnsembleCoilPredictorConfig):
        model = flax_nnx_checkpoint_util.from_checkpoint(
            checkpoint, MaskedEnsembleCoilPredictor
        )
    else:
        raise TypeError(
            f"Unsupported checkpoint config type: {type(checkpoint.config).__name__}"
        )
    model.eval()

    return model


@types.runtime_check_array_sizes
def predict_coilsets(
    model: AnyModel,
    eval_dataset: list[types.EvalData],
) -> list[types.EvalData]:
    """Runs the model on every eval boundary in one batched call.

    Raises:
        ValueError: If an instance's number of field periods or coil count is
            not supported by the model, before any inference is run.
    """
    _validate_eval_dataset(model, eval_dataset)
    n_modes_coils_max = model.config.n_modes_coils_max
    n_max_fourier_order = (n_modes_coils_max - 1) // 2
    batched_call = jax.jit(
        jax.vmap(jax.tree_util.Partial(model, fourier_order=n_max_fourier_order))
    )
    boundaries = [eval_data.boundary for eval_data in eval_dataset]
    batched_boundaries = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *boundaries)
    requirement_metrics = [eval_data.requirement_metrics for eval_data in eval_dataset]
    batched_requirement_metrics = jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0), *requirement_metrics
    )
    predicted_batched = batched_call(batched_boundaries, batched_requirement_metrics)

    leaves, treedef = jax.tree.flatten(predicted_batched)

    leaves = [jnp.moveaxis(x, 0, 0) for x in leaves]
    stack_length = leaves[0].shape[0]
    predicted_list = [
        jax.tree.unflatten(treedef, [x[i] for x in leaves]) for i in range(stack_length)
    ]

    updated_eval_dataset = []
    for eval_data, predicted in zip(eval_dataset, predicted_list, strict=True):
        if _is_variable_shape(model):
            predicted = coilset_utils.trim_to_n_coils(
                predicted, int(eval_data.requirement_metrics.n_coils_per_half_period)
            )
        updated_eval_data = eval_data.model_copy(
            update={"predicted_coilset": predicted}
        )
        updated_eval_dataset.append(updated_eval_data)
    return updated_eval_dataset


@types.runtime_check_array_sizes
def requirement_metrics_vector(
    requirement_metrics: types.RequirementMetrics,
) -> jt.Float[jt.Array, " 4"]:
    """The metrics the models are conditioned on, in the order of their means and
    stds."""
    return jnp.stack(
        [
            requirement_metrics.min_normalized_coil_to_coil_distance,
            requirement_metrics.min_normalized_coil_to_plasma_distance,
            requirement_metrics.max_normalized_coil_curvature,
            requirement_metrics.max_normalized_field_error,
        ]
    )


def _is_variable_shape(model: AnyModel) -> bool:
    return hasattr(model.config, "n_coils_values")


def _validate_eval_dataset(model: AnyModel, eval_dataset: list[types.EvalData]) -> None:
    config = model.config
    if _is_variable_shape(model):
        allowed_nfp = list(config.n_field_periods_values)  # type: ignore[union-attr]
        allowed_nc = list(config.n_coils_values)  # type: ignore[union-attr]
    else:
        allowed_nfp = [int(config.n_field_periods)]  # type: ignore[union-attr]
        allowed_nc = [int(config.n_unique_coils)]  # type: ignore[union-attr]
    for index, eval_data in enumerate(eval_dataset):
        nfp = int(eval_data.boundary.n_field_periods)
        n_coils = int(eval_data.requirement_metrics.n_coils_per_half_period)
        if nfp not in allowed_nfp:
            raise ValueError(
                f"Eval instance {index} has {nfp} field periods; "
                f"the model supports {allowed_nfp}."
            )
        if n_coils not in allowed_nc:
            raise ValueError(
                f"Eval instance {index} has {n_coils} coils; "
                f"the model supports {allowed_nc}."
            )
