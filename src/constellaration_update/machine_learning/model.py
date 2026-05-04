from typing import Protocol

import jax.numpy as jnp
import jaxtyping as jt
from constellaration.geometry import surface_rz_fourier
from flax import nnx

from constellaration_update import types as constellaration_update_types
from constellaration_update.machine_learning import types, utils


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
    ) -> jt.Float[jt.Array, " n_flat_boundary"]:
        ...

    def forward_coilset(
        self,
        coilset: constellaration_update_types.ConStellarationUpdateCoilset,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        ...

    def backward_coilset(
        self, flat: jt.Float[jt.Array, " n_flat_coilset"]
    ) -> constellaration_update_types.ConStellarationUpdateCoilset:
        ...


class CoilPredictor(nnx.Module):
    """Predicts coil parameters from a boundary shape (single example)."""

    def __init__(self, config: types.CoilPredictorConfig, *, rngs: nnx.Rngs):
        self.config = config
        n_arrays = 2 if config.is_stellarator_symmetric else 4
        input_size = config.n_poloidal_modes * config.n_toroidal_modes * n_arrays + len(
            config.requirement_metrics_means
        )
        flat_size = (
            config.n_unique_coils * config.n_modes_coils_max * 3 + config.n_unique_coils
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
        coilset: constellaration_update_types.ConStellarationUpdateCoilset,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            n_modes_coils_max=self.config.n_modes_coils_max,
            expected_currents=self.config.expected_currents,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def backward_coilset(
        self, flat: jt.Float[jt.Array, " n_flat_coilset"]
    ) -> constellaration_update_types.ConStellarationUpdateCoilset:
        return utils.backward_flat_to_coilset(
            flat,
            n_unique_coils=self.config.n_unique_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            expected_currents=self.config.expected_currents,
            n_field_periods=self.config.n_field_periods,
            is_stellarator_symmetric=self.config.is_stellarator_symmetric,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def __call__(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: constellaration_update_types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> constellaration_update_types.ConStellarationUpdateCoilset:
        flat_boundary = self.forward_boundary(boundary)
        means = jnp.asarray(self.config.requirement_metrics_means)
        stds = jnp.asarray(self.config.requirement_metrics_stds)
        standardized = (requirement_metrics.to_array() - means) / stds
        x = jnp.concatenate([flat_boundary, standardized])
        x = self.input_layer(x)
        for lin in self.block_linears:
            x = nnx.gelu(x)
            x = lin(x)
        x = nnx.gelu(x)
        x = self.output_layer(x)
        full = self.backward_coilset(x)
        n_max = self.config.n_modes_coils_max
        max_fourier_order = (n_max - 1) // 2
        if fourier_order is None or fourier_order == max_fourier_order:
            return full
        if fourier_order > max_fourier_order:
            raise ValueError(
                f"fourier_order={fourier_order} exceeds n_modes_coils_max="
                f"{n_max} (max supported fourier_order={max_fourier_order})"
            )
        width = 2 * fourier_order + 1
        pad = (n_max - width) // 2
        return full.model_copy(
            update={
                "coil_x_n": full.coil_x_n[:, pad : pad + width],
                "coil_y_n": full.coil_y_n[:, pad : pad + width],
                "coil_z_n": full.coil_z_n[:, pad : pad + width],
            }
        )


_: type[CoilTransform] = CoilPredictor


def _build_2d_sincos_positions(
    n_poloidal: int, n_toroidal: int, hidden_dim: int
) -> jt.Float[jt.Array, "{n_poloidal} {n_toroidal} {hidden_dim}"]:
    """Standard 2D sin/cos positional embedding over a spectral grid.

    Splits `hidden_dim` evenly between the two axes; if `hidden_dim` is not
    divisible by 4 the remainder is left as zeros (the model can absorb the
    asymmetry through learned projections).
    """
    quarter = hidden_dim // 4
    if quarter == 0:
        return jnp.zeros((n_poloidal, n_toroidal, hidden_dim))
    omega = 1.0 / (10000.0 ** (jnp.arange(quarter) / quarter))
    pos_p = jnp.arange(n_poloidal)[:, None] * omega[None, :]
    pos_t = jnp.arange(n_toroidal)[:, None] * omega[None, :]
    enc_p = jnp.concatenate([jnp.sin(pos_p), jnp.cos(pos_p)], axis=-1)  # (n_pol, 2q)
    enc_t = jnp.concatenate([jnp.sin(pos_t), jnp.cos(pos_t)], axis=-1)  # (n_tor, 2q)
    grid = jnp.concatenate(
        [
            jnp.broadcast_to(enc_p[:, None, :], (n_poloidal, n_toroidal, 2 * quarter)),
            jnp.broadcast_to(enc_t[None, :, :], (n_poloidal, n_toroidal, 2 * quarter)),
        ],
        axis=-1,
    )
    pad = hidden_dim - 4 * quarter
    if pad:
        grid = jnp.concatenate(
            [grid, jnp.zeros((n_poloidal, n_toroidal, pad))], axis=-1
        )
    return grid


def _build_signed_mode_sincos(
    fourier_order: int,
    n_modes_coils_max: int,
    hidden_dim: int,
) -> jt.Float[jt.Array, "n_modes hidden_dim"]:
    """Sin-cos positional embedding over the *signed* mode index.

    Index range is `{-fourier_order, ..., -1, 0, 1, ..., fourier_order}` of
    length `2*fourier_order + 1`. The frequency band is sized by
    `n_modes_coils_max` (interpreted as fourier order, NOT mode count) so
    that the embedding produced at any `fourier_order <= n_modes_coils_max`
    is the central slice of the embedding produced at `n_modes_coils_max` —
    enabling consistent inference at unseen `N`.
    """
    if fourier_order > n_modes_coils_max:
        raise ValueError(
            f"fourier_order={fourier_order} > n_modes_coils_max={n_modes_coils_max}"
        )
    half = hidden_dim // 2
    if half == 0:
        return jnp.zeros((2 * fourier_order + 1, hidden_dim))
    omega = 1.0 / (10000.0 ** (jnp.arange(half) / half))
    signed_n = jnp.arange(-fourier_order, fourier_order + 1)
    angles = signed_n[:, None] * omega[None, :]  # (2N+1, half)
    enc = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    pad = hidden_dim - 2 * half
    if pad:
        enc = jnp.concatenate([enc, jnp.zeros((2 * fourier_order + 1, pad))], axis=-1)
    return enc


class _AttentionBlock(nnx.Module):
    """Pre-norm MHSA + FFN block with FiLM conditioning on the FFN output."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm_attn = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            dropout_rate=dropout,
            decode=False,
            rngs=rngs,
        )
        self.norm_ffn = nnx.LayerNorm(hidden_dim, rngs=rngs)
        ffn_hidden = int(hidden_dim * mlp_ratio)
        self.ffn_in = nnx.Linear(hidden_dim, ffn_hidden, rngs=rngs)
        self.ffn_out = nnx.Linear(ffn_hidden, hidden_dim, rngs=rngs)
        self.film = nnx.Linear(hidden_dim, 2 * hidden_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        x: jt.Float[jt.Array, "tokens features"],
        cond_embedding: jt.Float[jt.Array, " features"],
    ) -> jt.Float[jt.Array, "tokens features"]:
        h = self.norm_attn(x)
        h = self.attn(h)
        h = self.dropout(h)
        x = x + h

        h = self.norm_ffn(x)
        h = self.ffn_in(h)
        h = nnx.gelu(h)
        h = self.ffn_out(h)
        scale_shift = self.film(cond_embedding)
        scale, shift = jnp.split(scale_shift, 2, axis=-1)
        h = h * (1.0 + scale) + shift
        h = self.dropout(h)
        return x + h


class _DecoderBlock(nnx.Module):
    """Cross-attention + FFN + FiLM.

    No query self-attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm_q = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.norm_kv = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            dropout_rate=dropout,
            decode=False,
            rngs=rngs,
        )
        self.norm_ffn = nnx.LayerNorm(hidden_dim, rngs=rngs)
        ffn_hidden = int(hidden_dim * mlp_ratio)
        self.ffn_in = nnx.Linear(hidden_dim, ffn_hidden, rngs=rngs)
        self.ffn_out = nnx.Linear(ffn_hidden, hidden_dim, rngs=rngs)
        self.film = nnx.Linear(hidden_dim, 2 * hidden_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        queries: jt.Float[jt.Array, "n_queries features"],
        memory: jt.Float[jt.Array, "n_memory features"],
        cond_embedding: jt.Float[jt.Array, " features"],
    ) -> jt.Float[jt.Array, "n_queries features"]:
        q = self.norm_q(queries)
        kv = self.norm_kv(memory)
        h = self.cross_attn(q, kv)
        h = self.dropout(h)
        queries = queries + h

        h = self.norm_ffn(queries)
        h = self.ffn_in(h)
        h = nnx.gelu(h)
        h = self.ffn_out(h)
        scale_shift = self.film(cond_embedding)
        scale, shift = jnp.split(scale_shift, 2, axis=-1)
        h = h * (1.0 + scale) + shift
        h = self.dropout(h)
        return queries + h


class AttentionCoilPredictor(nnx.Module):
    """Conv-attention `CoilPredictor` variant with mode-indexed decoder readout.

    Treats the boundary as a 2D `(n_poloidal, n_toroidal, n_arrays)` image:
    a conv stem extracts local mode-pair features and multi-head self-attention
    blocks mix globally with a CLS conditioning token. The post-attention token
    grid is consumed as memory by a cross-attention decoder whose queries are
    indexed by `(coil_idx, signed_mode_idx)`. Each query reads out a 3-vector
    coefficient via a small `Linear`. Per-coil current queries read out a
    scalar current. Conditioning (`RequirementMetrics`) is injected as the CLS
    token, as per-block FiLM modulation in both the encoder and decoder, and
    flows through the conditioning embedding.

    Forward-pass argument `fourier_order: int | None` selects the per-call
    output fourier order. `None` defaults to the maximum supported by the
    trained model (derived from `config.n_modes_coils_max`).
    """

    def __init__(self, config: types.AttentionCoilPredictorConfig, *, rngs: nnx.Rngs):
        self.config = config
        n_arrays = 2 if config.is_stellarator_symmetric else 4

        # Conditioning embedding: 4-D requirement metrics -> hidden_dim.
        n_cond = len(config.requirement_metrics_means)
        self.cond_in = nnx.Linear(n_cond, config.hidden_dim, rngs=rngs)
        self.cond_out = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)

        # Conv stem.
        stem_layers: list[nnx.Conv] = []
        prev_c = n_arrays
        for out_c in config.conv_stem_channels:
            stem_layers.append(
                nnx.Conv(
                    in_features=prev_c,
                    out_features=out_c,
                    kernel_size=(config.conv_stem_kernel, config.conv_stem_kernel),
                    padding="SAME",
                    rngs=rngs,
                )
            )
            prev_c = out_c
        self.stem_convs = nnx.List(stem_layers)
        self.stem_proj = nnx.Conv(
            in_features=prev_c,
            out_features=config.hidden_dim,
            kernel_size=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        # Positional encoding.
        if config.pos_encoding == "learned":
            self.pos_embed = nnx.Param(
                jnp.zeros(
                    (
                        config.n_poloidal_modes,
                        config.n_toroidal_modes,
                        config.hidden_dim,
                    )
                )
            )
        else:
            self.pos_embed = nnx.Variable(
                _build_2d_sincos_positions(
                    config.n_poloidal_modes,
                    config.n_toroidal_modes,
                    config.hidden_dim,
                )
            )

        # MHSA blocks.
        self.attention_blocks = nnx.List(
            [
                _AttentionBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    rngs=rngs,
                )
                for _ in range(config.num_attention_blocks)
            ]
        )
        self.post_attn_norm = nnx.LayerNorm(config.hidden_dim, rngs=rngs)

        # Coil identity embedding (learned).
        self.coil_embed = nnx.Embed(
            num_embeddings=config.n_unique_coils,
            features=config.hidden_dim,
            rngs=rngs,
        )

        # Mode positional embedding for the signed mode index. The learned table
        # is indexed by an unsigned offset over the full (2*N_max+1) band; we
        # slice the central 2N+1 rows at call time when fourier_order < N_max.
        # The sin-cos branch is built at call time via `_build_signed_mode_sincos`.
        if config.mode_pos_encoding == "learned":
            n_modes_full = config.n_modes_coils_max
            self.mode_pos_embed_table = nnx.Embed(
                num_embeddings=n_modes_full,
                features=config.hidden_dim,
                rngs=rngs,
            )
        else:
            self.mode_pos_embed_table = None

        # Per-coil current query tokens (learned).
        self.current_query = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(
                rngs.params(), (config.n_unique_coils, config.hidden_dim)
            )
        )

        # Decoder blocks.
        self.decoder_blocks = nnx.List(
            [
                _DecoderBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.decoder_num_heads,
                    mlp_ratio=config.decoder_mlp_ratio,
                    dropout=config.dropout,
                    rngs=rngs,
                )
                for _ in range(config.num_decoder_blocks)
            ]
        )
        self.decoder_post_norm = nnx.LayerNorm(config.hidden_dim, rngs=rngs)

        # Read-out projections.
        self.coef_readout = nnx.Linear(config.hidden_dim, 3, rngs=rngs)
        self.current_readout = nnx.Linear(config.hidden_dim, 1, rngs=rngs)

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
        coilset: constellaration_update_types.ConStellarationUpdateCoilset,
    ) -> jt.Float[jt.Array, " n_flat_coilset"]:
        return utils.forward_coilset_to_flat(
            coilset,
            n_modes_coils_max=self.config.n_modes_coils_max,
            expected_currents=self.config.expected_currents,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def backward_coilset(
        self, flat: jt.Float[jt.Array, " n_flat_coilset"]
    ) -> constellaration_update_types.ConStellarationUpdateCoilset:
        return utils.backward_flat_to_coilset(
            flat,
            n_unique_coils=self.config.n_unique_coils,
            n_modes_coils_max=self.config.n_modes_coils_max,
            expected_currents=self.config.expected_currents,
            n_field_periods=self.config.n_field_periods,
            is_stellarator_symmetric=self.config.is_stellarator_symmetric,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )

    def _boundary_image(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> jt.Float[jt.Array, "n_poloidal n_toroidal n_arrays"]:
        scaled = utils.calculate_exponential_spectral_scaling_surface(
            boundary,
            alpha=self.config.spectral_scaling_alpha,
            order=self.config.spectral_scaling_order,
            ess_min_value=self.config.spectral_scaling_min,
        )
        channels = [boundary.r_cos / scaled.r_cos, boundary.z_sin / scaled.z_sin]
        if boundary.r_sin is not None and scaled.r_sin is not None:
            channels.append(boundary.r_sin / scaled.r_sin)
        if boundary.z_cos is not None and scaled.z_cos is not None:
            channels.append(boundary.z_cos / scaled.z_cos)
        return jnp.stack(channels, axis=-1)

    def __call__(
        self,
        boundary: surface_rz_fourier.SurfaceRZFourier,
        requirement_metrics: constellaration_update_types.RequirementMetrics,
        *,
        fourier_order: int | None = None,
    ) -> constellaration_update_types.ConStellarationUpdateCoilset:
        """Predict a coilset, optionally truncated to the requested fourier order.

        Args:
            boundary: Plasma boundary surface.
            requirement_metrics: Conditioning vector.
            fourier_order: Per-call output fourier order N. Defaults to the
                maximum supported by the trained model. Must satisfy
                `2*fourier_order + 1 <= config.n_modes_coils_max`.

        Note:
            `fourier_order` must be a static Python int. To JIT this method,
            pass `static_argnames=("fourier_order",)` to `nnx.jit`.
        """
        n_max_fourier_order = (self.config.n_modes_coils_max - 1) // 2
        requested_n = (
            n_max_fourier_order if fourier_order is None else int(fourier_order)
        )
        if requested_n > n_max_fourier_order:
            raise ValueError(
                f"fourier_order={requested_n} exceeds n_modes_coils_max="
                f"{self.config.n_modes_coils_max} "
                f"(max fourier order={n_max_fourier_order})"
            )
        n_unique = self.config.n_unique_coils
        hidden = self.config.hidden_dim
        n_modes_out = 2 * requested_n + 1

        # --- Conditioning embedding ---
        means = jnp.asarray(self.config.requirement_metrics_means)
        stds = jnp.asarray(self.config.requirement_metrics_stds)
        standardized = jnp.asarray((requirement_metrics.to_array() - means) / stds)
        cond = self.cond_in(standardized)
        cond = nnx.gelu(cond)
        cond = self.cond_out(cond)  # (hidden,)

        # --- Encoder: conv stem -> MHSA token grid ---
        x = self._boundary_image(boundary)
        for conv in self.stem_convs:
            x = conv(x)
            x = nnx.gelu(x)
        x = self.stem_proj(x)
        x = x + jnp.asarray(self.pos_embed[...])
        n_pol, n_tor, _ = x.shape
        tokens = x.reshape(n_pol * n_tor, hidden)
        tokens = jnp.concatenate([cond[None, :], tokens], axis=0)
        for block in self.attention_blocks:
            tokens = block(tokens, cond)
        memory = self.post_attn_norm(tokens)  # (n_tokens+1, hidden) decoder K/V

        # --- Decoder queries ---
        coil_idx = jnp.arange(n_unique)
        coil_e = self.coil_embed(coil_idx)  # (n_unique, hidden)

        if self.mode_pos_embed_table is None:
            # Sin-cos: produces exactly (2N+1, hidden) tokens, by construction
            # equal to the central slice of the N_max embedding.
            mode_e = _build_signed_mode_sincos(
                fourier_order=requested_n,
                n_modes_coils_max=n_max_fourier_order,
                hidden_dim=hidden,
            )
        else:
            # Learned: index into the central (2*requested_n+1) entries of the table.
            pad = n_max_fourier_order - requested_n
            mode_idx = jnp.arange(pad, pad + n_modes_out)
            mode_e = self.mode_pos_embed_table(mode_idx)

        # Coefficient queries: (n_unique, 2N+1, hidden) -> flatten.
        coef_queries = coil_e[:, None, :] + mode_e[None, :, :]
        coef_queries = coef_queries.reshape(n_unique * n_modes_out, hidden)

        # Current queries: (n_unique, hidden).
        current_queries = jnp.asarray(self.current_query[...])

        queries = jnp.concatenate([coef_queries, current_queries], axis=0)

        # --- Decoder ---
        for block in self.decoder_blocks:
            queries = block(queries, memory, cond)
        queries = self.decoder_post_norm(queries)

        # --- Read-out ---
        n_coef = n_unique * n_modes_out
        coef_out = self.coef_readout(queries[:n_coef])  # (n_coef, 3)
        current_out = self.current_readout(queries[n_coef:]).squeeze(-1)
        coil_xyz_n = coef_out.reshape(n_unique, n_modes_out, 3)

        return utils.assemble_coilset_from_arrays(
            coil_xyz_n=coil_xyz_n,
            currents_normalized=current_out,
            n_field_periods=self.config.n_field_periods,
            is_stellarator_symmetric=self.config.is_stellarator_symmetric,
            expected_currents=self.config.expected_currents,
            alpha=self.config.spectral_scaling_alpha,
            ess_min_value=self.config.spectral_scaling_min,
        )


_: type[CoilTransform] = AttentionCoilPredictor
