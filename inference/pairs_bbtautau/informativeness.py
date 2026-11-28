"""Per-event and per-dataset informativeness estimators for PAIRS models.

Provides library utilities for:
1. Per-event posterior sensitivity: Std_theta[log q_phi(theta | t_omega(x), N=1)]
2. Per-event embedding gradient norm: E_theta[||grad_{t_omega} log q_phi(theta | t_omega(x), N=1)||_2]
3. Unbiased proposal weight construction with a configurable nominal floor.
4. Per-dataset informativeness: KL(q_phi(theta | T_bar, N) || p(theta)), sensitivity, and summary grad norm.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import SplitDataset
from .models import PairsModel
from .physics import UniformPrior2D
from .stages import _embed_split


def make_probe_theta_grid(
    prior: UniformPrior2D,
    n_mu: int = 4,
    n_kl: int = 4,
) -> torch.Tensor:
    """Return a compact (K, 2) tensor of probe (mu_HH, kappa_lambda) points across the prior."""
    mu_pts = np.linspace(prior.low[0] + 0.05 * prior.ranges[0], prior.high[0] - 0.05 * prior.ranges[0], n_mu)
    kl_pts = np.linspace(prior.low[1] + 0.05 * prior.ranges[1], prior.high[1] - 0.05 * prior.ranges[1], n_kl)
    mm, kk = np.meshgrid(mu_pts, kl_pts, indexing="ij")
    grid = np.stack([mm.reshape(-1), kk.reshape(-1)], axis=-1).astype(np.float32)
    return torch.from_numpy(grid)


@torch.no_grad()
def event_posterior_sensitivity(
    model: PairsModel,
    split: SplitDataset,
    prior: UniformPrior2D,
    device: torch.device,
    n_mu: int = 4,
    n_kl: int = 4,
    batch_size: int = 2048,
    embeddings: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-event Std_theta[log q_phi(theta | t_omega(x), N=1)] without backprop."""
    model.eval()
    embs = embeddings if embeddings is not None else _embed_split(model, split, batch_size=batch_size, device=device)
    theta_grid = make_probe_theta_grid(prior, n_mu=n_mu, n_kl=n_kl).to(device)  # (K, 2)
    k_pts = theta_grid.shape[0]

    scores: dict[str, np.ndarray] = {}
    for pname, e_np in embs.items():
        n_ev = e_np.shape[0]
        if n_ev == 0:
            scores[pname] = np.empty(0, dtype=np.float32)
            continue
        out_chunks = []
        for s in range(0, n_ev, batch_size):
            e = min(s + batch_size, n_ev)
            m = e - s
            tb = torch.from_numpy(e_np[s:e]).to(device)
            cnt = torch.ones(m, dtype=torch.long, device=device)
            # Evaluate across all K probe points: (m * K, ...)
            tb_rep = tb.unsqueeze(1).expand(m, k_pts, -1).reshape(m * k_pts, -1)
            cnt_rep = cnt.unsqueeze(1).expand(m, k_pts).reshape(m * k_pts)
            th_rep = theta_grid.unsqueeze(0).expand(m, k_pts, 2).reshape(m * k_pts, 2)
            logp = model.log_prob_from_aggregates(th_rep, tb_rep, cnt_rep).reshape(m, k_pts)
            out_chunks.append(logp.std(dim=1).cpu().numpy().astype(np.float32))
        scores[pname] = np.concatenate(out_chunks, axis=0)
    return scores


def event_embedding_gradient_norm(
    model: PairsModel,
    split: SplitDataset,
    prior: UniformPrior2D,
    device: torch.device,
    n_mu: int = 3,
    n_kl: int = 3,
    batch_size: int = 1024,
    embeddings: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-event E_theta[||grad_{t_omega} log q_phi(theta | t_omega(x), N=1)||_2].

    Differentiates only through the lightweight flow head (O(1) w.r.t. embedder depth).
    """
    model.eval()
    embs = embeddings if embeddings is not None else _embed_split(model, split, batch_size=batch_size, device=device)
    theta_grid = make_probe_theta_grid(prior, n_mu=n_mu, n_kl=n_kl).to(device)  # (K, 2)
    k_pts = theta_grid.shape[0]

    scores: dict[str, np.ndarray] = {}
    for pname, e_np in embs.items():
        n_ev = e_np.shape[0]
        if n_ev == 0:
            scores[pname] = np.empty(0, dtype=np.float32)
            continue
        out_chunks = []
        for s in range(0, n_ev, batch_size):
            e = min(s + batch_size, n_ev)
            m = e - s
            tb_leaf = torch.from_numpy(e_np[s:e]).to(device).detach().requires_grad_(True)
            cnt = torch.ones(m, dtype=torch.long, device=device)
            grad_norm_sq = torch.zeros(m, dtype=torch.float32, device=device)

            for k in range(k_pts):
                th_k = theta_grid[k : k + 1].expand(m, 2)
                logp_k = model.log_prob_from_aggregates(th_k, tb_leaf, cnt)
                (g_k,) = torch.autograd.grad(logp_k.sum(), tb_leaf, retain_graph=(k < k_pts - 1))
                grad_norm_sq = grad_norm_sq + g_k.pow(2).sum(dim=-1)

            rms_norm = torch.sqrt(grad_norm_sq / float(k_pts))
            out_chunks.append(rms_norm.detach().cpu().numpy().astype(np.float32))
        scores[pname] = np.concatenate(out_chunks, axis=0)
    return scores


def normalize_proposal_weights(
    scores: np.ndarray,
    base_weights: np.ndarray,
    floor_eps: float = 0.2,
    power: float = 1.0,
) -> np.ndarray:
    """Convert raw event informativeness scores into bounded, normalized proposal probabilities.

    q_0(i) propto base_weights[i] * ((1 - floor_eps) * (s_i^power / mean(s^power)) + floor_eps).
    """
    if not (0.0 <= floor_eps <= 1.0):
        raise ValueError(f"floor_eps must be in [0, 1], got {floor_eps}")
    bw = np.maximum(np.asarray(base_weights, dtype=np.float64), 0.0)
    s = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
    if power != 1.0:
        s = np.power(s, power)
    mean_s = float(np.average(s, weights=bw)) if bw.sum() > 0 else float(s.mean())
    if mean_s <= 1e-12:
        rel = np.ones_like(s)
    else:
        rel = s / mean_s
    q = bw * ((1.0 - floor_eps) * rel + floor_eps)
    q_sum = float(q.sum())
    if q_sum <= 0.0:
        return np.full(len(q), 1.0 / max(len(q), 1), dtype=np.float64)
    return q / q_sum


def dataset_informativeness(
    model: PairsModel,
    t_bar: torch.Tensor,
    counts: torch.Tensor,
    prior: UniformPrior2D,
    num_samples: int = 256,
    n_mu: int = 4,
    n_kl: int = 4,
) -> dict[str, np.ndarray]:
    """Estimate per-dataset informativeness from (T_bar, N) via KL gain, sensitivity, and grad norm."""
    model.eval()
    device = t_bar.device
    batch_size = t_bar.shape[0]

    # 1. Information gain KL(q_phi(theta | T_bar, N) || p(theta)) = E_q[log q] - log p_prior
    with torch.no_grad():
        samples = model.sample_from_aggregates(t_bar, counts, num_samples=num_samples)  # (B, S, 2)
        ctx = model.build_context(t_bar, counts)
        ctx_rep = ctx.unsqueeze(1).expand(-1, num_samples, -1)
        samples_std = model.standardize_theta(samples)
        logq = model.flow_head.log_prob(samples_std, ctx_rep) - model.log_theta_std_sum
        kl_gain = (logq.mean(dim=1) + prior.prior_nll).cpu().numpy().astype(np.float32)

        # 2. Posterior sensitivity across probe grid
        theta_grid = make_probe_theta_grid(prior, n_mu=n_mu, n_kl=n_kl).to(device)
        k_pts = theta_grid.shape[0]
        tb_rep = t_bar.unsqueeze(1).expand(batch_size, k_pts, -1).reshape(batch_size * k_pts, -1)
        cnt_rep = counts.unsqueeze(1).expand(batch_size, k_pts).reshape(batch_size * k_pts)
        th_rep = theta_grid.unsqueeze(0).expand(batch_size, k_pts, 2).reshape(batch_size * k_pts, 2)
        logp_grid = model.log_prob_from_aggregates(th_rep, tb_rep, cnt_rep).reshape(batch_size, k_pts)
        post_std_logp = logp_grid.std(dim=1).cpu().numpy().astype(np.float32)

    # 3. Gradient norm w.r.t. dataset summary T_bar averaged over posterior samples
    tb_leaf = t_bar.detach().requires_grad_(True)
    # Use a subset of posterior samples for fast gradient norm
    s_grad = min(16, num_samples)
    grad_sq = torch.zeros(batch_size, dtype=torch.float32, device=device)
    for r in range(s_grad):
        th_r = samples[:, r, :].detach()
        logp_r = model.log_prob_from_aggregates(th_r, tb_leaf, counts)
        (g_r,) = torch.autograd.grad(logp_r.sum(), tb_leaf, retain_graph=(r < s_grad - 1))
        grad_sq = grad_sq + g_r.pow(2).sum(dim=-1)
    summary_grad_norm = torch.sqrt(grad_sq / float(s_grad)).detach().cpu().numpy().astype(np.float32)

    return {
        "kl_gain": kl_gain,
        "post_std_logp": post_std_logp,
        "summary_grad_norm": summary_grad_norm,
    }
