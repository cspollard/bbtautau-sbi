"""PAIRS neural architectures: masked Set Transformer embedder and RQS normalizing flow."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block without BatchNorm (LayerNorm + SiLU)."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SetTransformerJetBlock(nn.Module):
    """Permutation-equivariant pre-norm Transformer block over jets within one event."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, jet_mask: torch.Tensor) -> torch.Tensor:
        """Apply masked self-attention and FFN.

        Args:
            x: (B_ev, J, D)
            jet_mask: (B_ev, J) bool, True for valid jets.
        """
        # Guard against all-False rows causing NaN in softmax without GPU->CPU host sync
        safe_mask = jet_mask.clone()
        safe_mask[:, 0] = safe_mask[:, 0] | (~jet_mask.any(dim=1))

        key_padding_mask = ~safe_mask
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x * jet_mask.unsqueeze(-1).to(x.dtype)


class PerEventEmbedder(nn.Module):
    """Permutation-invariant per-event embedder t_omega: X -> R^l.

    Encodes variable-length jet sets via masked Set Transformer layers + masked mean
    pooling, combines with event-level physics invariants, and terminates in an
    unconstrained linear projection to R^l (Assumption A3 of PAIRS Theorem 1).
    """

    def __init__(
        self,
        jet_feat_dim: int,
        evt_feat_dim: int,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        d_jet = cfg.jet_hidden_dim
        self.jet_in = nn.Sequential(
            nn.Linear(jet_feat_dim, d_jet),
            nn.LayerNorm(d_jet),
            nn.SiLU(),
        )
        self.jet_blocks = nn.ModuleList(
            [
                SetTransformerJetBlock(d_jet, cfg.jet_num_heads, cfg.dropout)
                for _ in range(cfg.jet_num_layers)
            ]
        )

        d_evt = cfg.event_hidden_dim
        self.evt_in = nn.Sequential(
            nn.Linear(d_jet + evt_feat_dim, d_evt),
            nn.LayerNorm(d_evt),
            nn.SiLU(),
        )
        self.evt_blocks = nn.ModuleList(
            [ResidualMLPBlock(d_evt, dropout=cfg.dropout) for _ in range(cfg.event_num_blocks)]
        )
        self.evt_norm = nn.LayerNorm(d_evt)
        # Unconstrained linear output layer (no activation / no softmax)
        self.out_proj = nn.Linear(d_evt, cfg.embed_dim)

    def forward(
        self,
        jet_feats: torch.Tensor,
        jet_mask: torch.Tensor,
        evt_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a batch of single events.

        Args:
            jet_feats: (B_ev, J, F_jet)
            jet_mask:  (B_ev, J) bool
            evt_feats: (B_ev, F_evt)
        Returns:
            embeddings: (B_ev, embed_dim)
        """
        mask_f = jet_mask.unsqueeze(-1).to(jet_feats.dtype)
        h = self.jet_in(jet_feats) * mask_f
        for block in self.jet_blocks:
            h = block(h, jet_mask)

        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = (h * mask_f).sum(dim=1) / denom

        x = torch.cat([pooled, evt_feats], dim=-1)
        x = self.evt_in(x)
        for block in self.evt_blocks:
            x = block(x)
        x = F.silu(self.evt_norm(x))
        return self.out_proj(x)


def _rational_quadratic_spline(
    inputs: torch.Tensor,
    unnorm_widths: torch.Tensor,
    unnorm_heights: torch.Tensor,
    unnorm_derivatives: torch.Tensor,
    inverse: bool = False,
    tail_bound: float = 3.5,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise monotonic rational-quadratic spline (Durkan et al., 2019).

    Uses linear identity tails outside [-tail_bound, tail_bound] and MPS-safe
    tensor operations (strictly float32, no torch.searchsorted dependency).
    """
    num_bins = unnorm_widths.shape[-1]
    if min_bin_width * num_bins > 1.0:
        raise ValueError("min_bin_width * num_bins must be <= 1.0")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("min_bin_height * num_bins must be <= 1.0")

    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    x_clamped = inputs.clamp(-tail_bound, tail_bound)

    # Compute normalized widths, heights, and positive interior knot derivatives
    widths = F.softmax(unnorm_widths, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    widths = widths * (2.0 * tail_bound)
    cumwidths = F.pad(torch.cumsum(widths, dim=-1), (1, 0), value=0.0) - tail_bound
    cumwidths[..., 0] = -tail_bound
    cumwidths[..., -1] = tail_bound

    heights = F.softmax(unnorm_heights, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    heights = heights * (2.0 * tail_bound)
    cumheights = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0) - tail_bound
    cumheights[..., 0] = -tail_bound
    cumheights[..., -1] = tail_bound

    # Shift softplus so zero raw parameter initializes derivative = 1.0
    deriv_shift = math.log(math.expm1(1.0 - min_derivative))
    interior_d = min_derivative + F.softplus(unnorm_derivatives + deriv_shift)
    ones = torch.ones(*interior_d.shape[:-1], 1, dtype=interior_d.dtype, device=interior_d.device)
    derivatives = torch.cat([ones, interior_d, ones], dim=-1)

    knots = cumheights if inverse else cumwidths
    # Bin lookup via comparison sum (static shapes, no host sync on MPS)
    bin_idx = ((x_clamped.unsqueeze(-1) >= knots[..., :-1]).sum(dim=-1) - 1).clamp(0, num_bins - 1)
    idx_col = bin_idx.unsqueeze(-1)

    input_cumwidths = torch.gather(cumwidths, -1, idx_col).squeeze(-1)
    input_bin_widths = torch.gather(widths, -1, idx_col).squeeze(-1)
    input_cumheights = torch.gather(cumheights, -1, idx_col).squeeze(-1)
    input_bin_heights = torch.gather(heights, -1, idx_col).squeeze(-1)

    input_delta = input_bin_heights / input_bin_widths
    dk = torch.gather(derivatives, -1, idx_col).squeeze(-1)
    dkp1 = torch.gather(derivatives, -1, idx_col + 1).squeeze(-1)
    deriv_sum_m2 = dkp1 + dk - 2.0 * input_delta

    if inverse:
        dy = x_clamped - input_cumheights
        a = input_bin_heights * (input_delta - dk) + dy * deriv_sum_m2
        b = input_bin_heights * dk - dy * deriv_sum_m2
        c = -input_delta * dy
        discriminant = torch.clamp(b * b - 4.0 * a * c, min=0.0)
        sqrt_disc = torch.sqrt(discriminant)

        # Branch on sign of b to eliminate floating-point cancellation
        root_citardauq = (-2.0 * c) / torch.clamp(b + sqrt_disc, min=1e-12)
        root_direct = (-b + sqrt_disc) / torch.where(a.abs() > 1e-12, 2.0 * a, torch.ones_like(a))
        xi = torch.where((b >= 0.0) | (a.abs() <= 1e-12), root_citardauq, root_direct)
        xi = xi.clamp(0.0, 1.0)

        out_in = xi * input_bin_widths + input_cumwidths
        xi_1mxi = xi * (1.0 - xi)
        denom = input_delta + deriv_sum_m2 * xi_1mxi
        deriv_num = (input_delta * input_delta) * (
            dkp1 * (xi * xi) + 2.0 * input_delta * xi_1mxi + dk * ((1.0 - xi) * (1.0 - xi))
        )
        logdet_in = -(torch.log(torch.clamp(deriv_num, min=1e-18)) - 2.0 * torch.log(torch.clamp(denom, min=1e-18)))
    else:
        xi = ((x_clamped - input_cumwidths) / input_bin_widths).clamp(0.0, 1.0)
        xi_1mxi = xi * (1.0 - xi)
        denom = input_delta + deriv_sum_m2 * xi_1mxi
        numer = input_delta * (xi * xi) + dk * xi_1mxi
        out_in = input_cumheights + input_bin_heights * (numer / denom)

        deriv_num = (input_delta * input_delta) * (
            dkp1 * (xi * xi) + 2.0 * input_delta * xi_1mxi + dk * ((1.0 - xi) * (1.0 - xi))
        )
        logdet_in = torch.log(torch.clamp(deriv_num, min=1e-18)) - 2.0 * torch.log(torch.clamp(denom, min=1e-18))

    outputs = torch.where(inside, out_in, inputs)
    logabsdet = torch.where(inside, logdet_in, torch.zeros_like(inputs))
    return outputs, logabsdet


class RationalQuadraticCoupling2D(nn.Module):
    """2D coupling layer transforming one coordinate via RQS conditioned on the other + context."""

    def __init__(
        self,
        transform_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_blocks: int,
        num_bins: int,
        tail_bound: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if transform_dim not in (0, 1):
            raise ValueError("transform_dim must be 0 or 1 for 2D coupling layer")
        self.transform_dim = transform_dim
        self.cond_dim = 1 - transform_dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        out_dim = 3 * num_bins - 1
        self.in_proj = nn.Sequential(
            nn.Linear(1 + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        # Near-identity initialization for stable early flow optimization
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def _spline_params(
        self,
        x_cond: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = torch.cat([x_cond.unsqueeze(-1), context], dim=-1)
        h = self.in_proj(inp)
        for blk in self.blocks:
            h = blk(h)
        raw = self.out_proj(F.silu(self.out_norm(h)))
        k = self.num_bins
        return raw[..., :k], raw[..., k : 2 * k], raw[..., 2 * k :]

    def _transform(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        inverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_cond = x[..., self.cond_dim]
        x_trans = x[..., self.transform_dim]
        w, h, d = self._spline_params(x_cond, context)
        y_trans, logabsdet = _rational_quadratic_spline(
            x_trans,
            w,
            h,
            d,
            inverse=inverse,
            tail_bound=self.tail_bound,
        )
        out = x.clone()
        out[..., self.transform_dim] = y_trans
        return out, logabsdet

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._transform(x, context, inverse=False)

    def inverse(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._transform(z, context, inverse=True)


class ConditionalSplineFlow2D(nn.Module):
    """2D conditional normalizing flow with standard Gaussian base distribution."""

    def __init__(self, cond_in_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.context_net = nn.Sequential(
            nn.Linear(cond_in_dim, cfg.context_hidden_dim),
            nn.LayerNorm(cfg.context_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.context_hidden_dim, cfg.context_hidden_dim),
            nn.LayerNorm(cfg.context_hidden_dim),
            nn.SiLU(),
        )
        self.transforms = nn.ModuleList(
            [
                RationalQuadraticCoupling2D(
                    transform_dim=i % 2,
                    context_dim=cfg.context_hidden_dim,
                    hidden_dim=cfg.flow_hidden_dim,
                    num_blocks=cfg.flow_num_blocks,
                    num_bins=cfg.flow_num_bins,
                    tail_bound=cfg.flow_tail_bound,
                    dropout=cfg.dropout,
                )
                for i in range(cfg.flow_num_transforms)
            ]
        )

    def log_prob(self, theta_std: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Compute exact log q(theta_std | cond), shape (B,)."""
        ctx = self.context_net(cond)
        z = theta_std
        total_logdet = torch.zeros(theta_std.shape[:-1], dtype=theta_std.dtype, device=theta_std.device)
        for transform in self.transforms:
            z, ld = transform.forward(z, ctx)
            total_logdet = total_logdet + ld
        base_logp = -0.5 * (z.pow(2).sum(dim=-1) + 2.0 * math.log(2.0 * math.pi))
        return base_logp + total_logdet

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Draw samples in standardized theta space, shape (B, num_samples, 2)."""
        batch_size = cond.shape[0]
        ctx = self.context_net(cond)
        ctx_rep = ctx.unsqueeze(1).expand(batch_size, num_samples, -1)
        z = torch.randn(batch_size, num_samples, 2, dtype=cond.dtype, device=cond.device)
        x = z
        for transform in reversed(self.transforms):
            x, _ = transform.inverse(x, ctx_rep)
        return x


class PairsModel(nn.Module):
    """Complete PAIRS model combining per-event embedder, mean-pool aggregator, and NF head."""

    def __init__(
        self,
        jet_feat_dim: int,
        evt_feat_dim: int,
        theta_mean: torch.Tensor,
        theta_std: torch.Tensor,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.jet_feat_dim = jet_feat_dim
        self.evt_feat_dim = evt_feat_dim

        self.embedder = PerEventEmbedder(jet_feat_dim, evt_feat_dim, cfg)
        self.flow_head = ConditionalSplineFlow2D(cond_in_dim=cfg.embed_dim + 1, cfg=cfg)

        self.register_buffer("theta_mean", theta_mean.clone().float())
        self.register_buffer("theta_std", theta_std.clone().float())
        log_std_sum = torch.log(theta_std.float()).sum()
        self.register_buffer("log_theta_std_sum", log_std_sum)

        # Conditioning standardization stats (identity for Stage 1; fitted for Stage 3)
        self.register_buffer("tbar_mean", torch.zeros(cfg.embed_dim, dtype=torch.float32))
        self.register_buffer("tbar_std", torch.ones(cfg.embed_dim, dtype=torch.float32))
        self.register_buffer("logn_mean", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("logn_std", torch.ones(1, dtype=torch.float32))
        self.n_encoding_mode: str = "linear"
        self.stage1_n_max: float = 2.0

    def reset_flow_head(self) -> None:
        """Re-initialize the conditional normalizing flow head (used before Stage 3)."""
        device = self.theta_mean.device
        self.flow_head = ConditionalSplineFlow2D(
            cond_in_dim=self.cfg.embed_dim + 1,
            cfg=self.cfg,
        ).to(device)

    def set_theta_stats(self, theta_mean: torch.Tensor, theta_std: torch.Tensor) -> None:
        """Update target standardization mean/std and physical-unit Jacobian constant."""
        self.theta_mean.copy_(theta_mean.to(self.theta_mean.device).float())
        self.theta_std.copy_(theta_std.to(self.theta_std.device).float())
        self.log_theta_std_sum.copy_(torch.log(self.theta_std).sum())

    def configure_stage1(self, n_max: float = 2.0) -> None:
        self.n_encoding_mode = "linear"
        self.stage1_n_max = float(n_max)
        self.tbar_mean.zero_()
        self.tbar_std.fill_(1.0)
        self.logn_mean.zero_()
        self.logn_std.fill_(1.0)

    def configure_stage3(
        self,
        tbar_mean: torch.Tensor,
        tbar_std: torch.Tensor,
        logn_mean: torch.Tensor,
        logn_std: torch.Tensor,
    ) -> None:
        self.n_encoding_mode = "standardized_log"
        self.tbar_mean.copy_(tbar_mean.to(self.tbar_mean.device))
        self.tbar_std.copy_(tbar_std.clamp(min=1e-6).to(self.tbar_std.device))
        self.logn_mean.copy_(logn_mean.reshape(1).to(self.logn_mean.device))
        self.logn_std.copy_(logn_std.clamp(min=1e-6).reshape(1).to(self.logn_std.device))

    def encode_n(self, counts: torch.Tensor) -> torch.Tensor:
        """Non-learnable scalar encoding of set size N, shape (B, 1)."""
        c_f = counts.float().unsqueeze(-1)
        if self.n_encoding_mode == "linear":
            return c_f / self.stage1_n_max
        if self.n_encoding_mode == "standardized_log":
            return (torch.log(c_f + 1.0) - self.logn_mean) / self.logn_std
        raise ValueError(f"Unknown n_encoding_mode: {self.n_encoding_mode}")

    def build_context(self, t_bar: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        t_norm = (t_bar - self.tbar_mean) / self.tbar_std
        n_enc = self.encode_n(counts)
        return torch.cat([t_norm, n_enc], dim=-1)

    def aggregate_sets(
        self,
        jet_feats: torch.Tensor,
        jet_mask: torch.Tensor,
        evt_feats: torch.Tensor,
        evt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute masked mean-pool embedding T_omega(X_N) and event count N.

        Only evaluates the embedder on valid events (evt_mask=True) to avoid wasted
        compute and padded-slot artifacts. Returns (T_bar, N).
        """
        batch_size, max_n = evt_mask.shape
        counts = evt_mask.sum(dim=1)
        valid_flat = evt_mask.reshape(-1)
        t_bar = torch.zeros(batch_size, self.cfg.embed_dim, dtype=jet_feats.dtype, device=jet_feats.device)
        if not bool(valid_flat.any()):
            return t_bar, counts

        jf_valid = jet_feats.reshape(-1, jet_feats.shape[2], jet_feats.shape[3])[valid_flat]
        jm_valid = jet_mask.reshape(-1, jet_mask.shape[2])[valid_flat]
        ef_valid = evt_feats.reshape(-1, evt_feats.shape[2])[valid_flat]
        emb_valid = self.embedder(jf_valid, jm_valid, ef_valid)

        emb_grid = torch.zeros(
            batch_size * max_n,
            self.cfg.embed_dim,
            dtype=emb_valid.dtype,
            device=emb_valid.device,
        )
        emb_grid[valid_flat] = emb_valid
        emb_grid = emb_grid.view(batch_size, max_n, self.cfg.embed_dim)
        t_bar = emb_grid.sum(dim=1) / counts.clamp(min=1).unsqueeze(-1).to(emb_grid.dtype)
        return t_bar, counts

    def standardize_theta(self, theta_phys: torch.Tensor) -> torch.Tensor:
        return (theta_phys - self.theta_mean) / self.theta_std

    def unstandardize_theta(self, theta_std: torch.Tensor) -> torch.Tensor:
        return theta_std * self.theta_std + self.theta_mean

    def log_prob_from_aggregates(
        self,
        theta_phys: torch.Tensor,
        t_bar: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """Exact log q(theta | T_bar, N) in physical parameter units (Jacobian included)."""
        ctx = self.build_context(t_bar, counts)
        theta_std = self.standardize_theta(theta_phys)
        logp_std = self.flow_head.log_prob(theta_std, ctx)
        return logp_std - self.log_theta_std_sum

    def nll_from_aggregates(
        self,
        theta_phys: torch.Tensor,
        t_bar: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        return -self.log_prob_from_aggregates(theta_phys, t_bar, counts)

    def nll_from_sets(
        self,
        theta_phys: torch.Tensor,
        jet_feats: torch.Tensor,
        jet_mask: torch.Tensor,
        evt_feats: torch.Tensor,
        evt_mask: torch.Tensor,
    ) -> torch.Tensor:
        t_bar, counts = self.aggregate_sets(jet_feats, jet_mask, evt_feats, evt_mask)
        return self.nll_from_aggregates(theta_phys, t_bar, counts)

    @torch.no_grad()
    def sample_from_aggregates(
        self,
        t_bar: torch.Tensor,
        counts: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Draw posterior samples in physical units, shape (B, num_samples, 2)."""
        ctx = self.build_context(t_bar, counts)
        samples_std = self.flow_head.sample(num_samples=num_samples, cond=ctx)
        return self.unstandardize_theta(samples_std)
