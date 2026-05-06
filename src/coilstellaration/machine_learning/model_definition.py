from typing import Protocol

import jax.numpy as jnp
import jaxtyping as jt
from constellaration.geometry import surface_rz_fourier
from flax import nnx

from coilstellaration import (
    flax_nnx_checkpoint_util,
    types,
)
from coilstellaration.machine_learning import utils
from coilstellaration.types import NpOrJaxArray


class CoilTransform(Protocol):
    """Bidirectional spectral scaling between physical and network-IO space.

    Implementations define a consistent scaling pair so that loss functions
    and dataloaders can compare predicted and ground-truth coilsets in a
    single representation, independent of which concrete model produced
    the prediction. `forward_*` map physical-space inputs/outputs to the
    flat scaled vector the network sees; `backward_coilset` is the inverse
    of `forward_coilset`.
    """

    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]: ...

    def forward_coilset(
        self,
        coilset: types.Coilset,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]: ...

    def backward_coilset(
        self, flat: jt.Float[jt.Array, " n_flat_coilset"]
    ) -> types.Coilset: ...


def _truncate_to_fourier_order(
    full: types.Coilset,
    *,
    n_modes_coils_max: int,
    fourier_order: int | None,
) -> types.Coilset:
    """Centrally slice a full-band coilset to the requested fourier order."""
    max_fourier_order = (n_modes_coils_max - 1) // 2
    if fourier_order is None or fourier_order == max_fourier_order:
        return full
    if fourier_order > max_fourier_order:
        raise ValueError(
            f"fourier_order={fourier_order} exceeds n_modes_coils_max="
            f"{n_modes_coils_max} (max supported fourier_order={max_fourier_order})"
        )
    width = 2 * fourier_order + 1
    pad = (n_modes_coils_max - width) // 2
    return full.model_copy(
        update={
            "coil_x_n": full.coil_x_n[:, pad : pad + width],
            "coil_y_n": full.coil_y_n[:, pad : pad + width],
            "coil_z_n": full.coil_z_n[:, pad : pad + width],
        }
    )


class CoilPredictor(nnx.Module):
    """Basic MLP predictor: input → ``num_layers`` GELU + linear layers → flat readout.

    Single readout layer produces the concatenated `[coils, currents]` flat
    vector that `backward_coilset` decodes. Treated as a baseline against
    `ResMlpCoilPredictor`, which adds pre-norm residual blocks, dropout, and
    a separate currents head on top of the same I/O contract.
    """

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

    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]:
        return utils.forward_boundary_to_flat(
            boundary,
            alpha=self.config.spectral_scaling_alpha,
            order=self.config.spectral_scaling_order,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def forward_coilset(
        self,
        coilset: types.Coilset,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def backward_coilset(
        self, flat: jt.Float[jt.Array, " n_flat_coilset"]
    ) -> types.Coilset:
        return utils.backward_flat_to_coilset(
            flat,
            n_unique_coils=self.config.n_unique_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            n_field_periods=self.config.n_field_periods,
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
        standardized = (
            requirement_metrics.to_array() - self.requirement_metrics_means[...]
        ) / self.requirement_metrics_stds[...]
        x = jnp.concatenate([flat_boundary, standardized])
        x = self.input_layer(x)
        for lin in self.block_linears:
            x = nnx.gelu(x)
            x = lin(x)
        x = nnx.gelu(x)
        x = self.output_layer(x)
        full = self.backward_coilset(x)
        return _truncate_to_fourier_order(
            full,
            n_modes_coils_max=self.config.n_modes_coils_max,
            fourier_order=fourier_order,
        )


_: type[CoilTransform] = CoilPredictor


class _CurrentsHead(nnx.Module):
    """Bottleneck MLP that predicts per-coil currents from coil shapes + metrics.

    Currents are physically set by the target field (B₀ etc., in
    `requirement_metrics`) given the chosen coil geometry, so this head sees
    the predicted `flat_coils` plus the standardized metrics and approximates
    the geometry→currents inversion. The output is zero-initialized so the
    network starts predicting exactly `coil_currents_mean` (after
    `backward_flat_to_coilset` maps `flat_currents=0` to `mean`); any movement
    must be earned from gradient signal. The narrow `hidden_dim` caps the
    capacity dedicated to currents memorization without affecting the trunk.
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
        self.fc2 = nnx.Linear(hidden_dim, n_unique_coils, rngs=rngs)
        self.fc2.kernel.value = jnp.zeros_like(self.fc2.kernel.value)
        if self.fc2.bias is not None:
            self.fc2.bias.value = jnp.zeros_like(self.fc2.bias.value)

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
        self.coil_head = nnx.Linear(config.hidden_dim, coil_flat_size, rngs=rngs)
        self.currents_head = _CurrentsHead(
            in_features=coil_flat_size + n_cond,
            n_unique_coils=config.n_unique_coils,
            rngs=rngs,
        )

    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]:
        return utils.forward_boundary_to_flat(
            boundary,
            alpha=self.config.spectral_scaling_alpha,
            order=self.config.spectral_scaling_order,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def forward_coilset(
        self,
        coilset: types.Coilset,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def backward_coilset(
        self, flat: jt.Float[jt.Array, " n_flat_coilset"]
    ) -> types.Coilset:
        return utils.backward_flat_to_coilset(
            flat,
            n_unique_coils=self.config.n_unique_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            n_field_periods=self.config.n_field_periods,
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
        standardized = (
            requirement_metrics.to_array() - self.requirement_metrics_means[...]
        ) / self.requirement_metrics_stds[...]
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
        full = self.backward_coilset(jnp.concatenate([flat_coils, flat_currents]))
        return _truncate_to_fourier_order(
            full,
            n_modes_coils_max=self.config.n_modes_coils_max,
            fourier_order=fourier_order,
        )


_: type[CoilTransform] = ResMlpCoilPredictor


class MlpEnsembleCoilPredictor(nnx.Module):
    """Ensemble of independent basic-MLP `CoilPredictor`s averaged in coilset space.

    Each member is an independent `CoilPredictor` sharing the same architecture
    config but with distinct random init drawn from the shared `rngs` stream.
    Predictions are averaged elementwise across the ensemble for each Fourier
    coefficient and current; this is equivalent to averaging per-member flat
    outputs because both `backward_coilset` and `_truncate_to_fourier_order`
    are linear in the flat representation, so the ensemble inherits the
    `CoilTransform` scaling unchanged from any single member.
    """

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

    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]:
        return utils.forward_boundary_to_flat(
            boundary,
            alpha=self.config.spectral_scaling_alpha,
            order=self.config.spectral_scaling_order,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def forward_coilset(
        self,
        coilset: types.Coilset,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def backward_coilset(
        self, flat: jt.Float[jt.Array, " n_flat_coilset"]
    ) -> types.Coilset:
        return utils.backward_flat_to_coilset(
            flat,
            n_unique_coils=self.config.n_unique_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            n_field_periods=self.config.n_field_periods,
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


_: type[CoilTransform] = MlpEnsembleCoilPredictor


class ResMlpEnsembleCoilPredictor(nnx.Module):
    """Ensemble of independent `ResMlpCoilPredictor`s averaged in coilset space.

    Sibling of `MlpEnsembleCoilPredictor` whose members are advanced
    pre-norm/residual/dropout MLPs with split readout heads. Mirror the
    averaging argument of `MlpEnsembleCoilPredictor` for justification of
    coilset-space averaging.
    """

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

    def forward_boundary(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, " n_flat_boundary"]:
        return utils.forward_boundary_to_flat(
            boundary,
            alpha=self.config.spectral_scaling_alpha,
            order=self.config.spectral_scaling_order,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def forward_coilset(
        self,
        coilset: types.Coilset,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def backward_coilset(
        self, flat: jt.Float[jt.Array, " n_flat_coilset"]
    ) -> types.Coilset:
        return utils.backward_flat_to_coilset(
            flat,
            n_unique_coils=self.config.n_unique_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            coil_currents_mean=self.config.coil_currents_mean,
            coil_currents_std=self.config.coil_currents_std,
            n_field_periods=self.config.n_field_periods,
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
        per-member loss-of-mean vs mean-of-loss.
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


_: type[CoilTransform] = ResMlpEnsembleCoilPredictor


AnyEnsembleModel = MlpEnsembleCoilPredictor | ResMlpEnsembleCoilPredictor
AnyModel = CoilPredictor | ResMlpCoilPredictor | AnyEnsembleModel


def read_model_from_checkpoint(
    checkpoint: types.AnyModelCheckpoint,
) -> AnyModel:
    """Deserialize a model from a dapper checkpoint and set it to eval mode."""
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
    else:
        raise TypeError(
            f"Unsupported checkpoint config type: {type(checkpoint.config).__name__}"
        )
    model.eval()

    return model
