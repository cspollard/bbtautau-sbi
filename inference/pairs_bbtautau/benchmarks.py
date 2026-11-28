"""Reference and oracle posterior baselines for PAIRS optimality verification.

Provides centralized implementations of:
1. Shared 2D grid posterior normalization & EvaluationResults computation (reusing metrics.py)
2. Rate-only analytical Poisson posterior p(theta | N)
3. Binned 2D Poisson likelihood posterior over (m_bbtt, m_bb)
4. Stat-only unbinned N=1 Neural Ratio Estimation (NRE) oracle p(theta | X_N)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.special import gammaln, logsumexp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .data import FeatureStats, SplitDataset
from .metrics import (
    EvaluationResults,
    compute_empirical_coverage_1d,
    compute_empirical_coverage_2d_hpd,
)
from .models import PerEventEmbedder, ResidualMLPBlock
from .physics import ProcessSampler, UniformPrior2D, compute_hh_weights, hh_reweight_coefficients


def make_theta_grid(
    prior: UniformPrior2D,
    n_mu: int = 81,
    n_kl: int = 91,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mu_grid, kl_grid, theta_mesh) over the 2D uniform prior."""
    mu_grid = np.linspace(prior.low[0], prior.high[0], n_mu, dtype=np.float64)
    kl_grid = np.linspace(prior.low[1], prior.high[1], n_kl, dtype=np.float64)
    mm, kk = np.meshgrid(mu_grid, kl_grid, indexing="ij")
    theta_mesh = np.stack([mm, kk], axis=-1)  # (n_mu, n_kl, 2)
    return mu_grid, kl_grid, theta_mesh


def _bilinear_interp_grid(
    values_grid: np.ndarray,
    theta: np.ndarray,
    mu_grid: np.ndarray,
    kl_grid: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate (B, G_mu, G_kl) grid values at theta (B, 2)."""
    n_mu = len(mu_grid)
    n_kl = len(kl_grid)
    d_mu = mu_grid[1] - mu_grid[0]
    d_kl = kl_grid[1] - kl_grid[0]

    u = np.clip((theta[:, 0] - mu_grid[0]) / d_mu, 0.0, n_mu - 1.0 - 1e-9)
    v = np.clip((theta[:, 1] - kl_grid[0]) / d_kl, 0.0, n_kl - 1.0 - 1e-9)
    i0 = np.floor(u).astype(np.int64)
    j0 = np.floor(v).astype(np.int64)
    du = u - i0
    dv = v - j0

    b_idx = np.arange(theta.shape[0])
    v00 = values_grid[b_idx, i0, j0]
    v10 = values_grid[b_idx, i0 + 1, j0]
    v01 = values_grid[b_idx, i0, j0 + 1]
    v11 = values_grid[b_idx, i0 + 1, j0 + 1]
    return (
        (1.0 - du) * (1.0 - dv) * v00
        + du * (1.0 - dv) * v10
        + (1.0 - du) * dv * v01
        + du * dv * v11
    )


def evaluate_grid_posteriors(
    log_lik_grid: np.ndarray,
    theta_true: np.ndarray,
    n_events: np.ndarray,
    prior: UniformPrior2D,
    mu_grid: np.ndarray,
    kl_grid: np.ndarray,
    num_samples: int = 1000,
    num_acauc_levels: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[EvaluationResults, np.ndarray]:
    """Normalize 2D grid log-likelihoods to exact posteriors and compute EvaluationResults."""
    if rng is None:
        rng = np.random.default_rng(0)

    n_test, n_mu, n_kl = log_lik_grid.shape
    d_mu = float(mu_grid[1] - mu_grid[0])
    d_kl = float(kl_grid[1] - kl_grid[0])
    cell_area = d_mu * d_kl

    # Trapezoid weights for accurate continuous normalization on the grid
    w_mu = np.ones(n_mu, dtype=np.float64)
    w_mu[0] = 0.5
    w_mu[-1] = 0.5
    w_kl = np.ones(n_kl, dtype=np.float64)
    w_kl[0] = 0.5
    w_kl[-1] = 0.5
    log_quad_w = np.log(np.outer(w_mu, w_kl) * cell_area)

    log_unnorm = log_lik_grid.astype(np.float64)
    log_z = logsumexp(log_unnorm + log_quad_w[None, :, :], axis=(1, 2), keepdims=True)
    log_post_grid = log_unnorm - log_z  # continuous log density p(theta | X)
    post_weights = np.exp(log_post_grid + log_quad_w[None, :, :])
    post_weights = post_weights / np.sum(post_weights, axis=(1, 2), keepdims=True)

    # Test NLL at theta_true via bilinear interpolation of log_post_grid
    true_logp = _bilinear_interp_grid(log_post_grid, theta_true, mu_grid, kl_grid)
    test_nll = float(-np.mean(true_logp))

    # Moments from grid
    mm, kk = np.meshgrid(mu_grid, kl_grid, indexing="ij")
    mean_mu = np.sum(post_weights * mm[None, :, :], axis=(1, 2))
    mean_kl = np.sum(post_weights * kk[None, :, :], axis=(1, 2))
    theta_mean = np.stack([mean_mu, mean_kl], axis=-1)

    var_mu = np.sum(post_weights * (mm[None, :, :] - mean_mu[:, None, None]) ** 2, axis=(1, 2))
    var_kl = np.sum(post_weights * (kk[None, :, :] - mean_kl[:, None, None]) ** 2, axis=(1, 2))
    theta_std = np.sqrt(np.maximum(np.stack([var_mu, var_kl], axis=-1), 1e-12))

    # Draw categorical grid samples + sub-cell jitter for ACAUC
    flat_probs = post_weights.reshape(n_test, -1)
    samples = np.empty((n_test, num_samples, 2), dtype=np.float64)
    sample_logp = np.empty((n_test, num_samples), dtype=np.float64)

    for j in range(n_test):
        cdf = np.cumsum(flat_probs[j])
        cdf[-1] = 1.0
        idx = np.searchsorted(cdf, rng.uniform(0.0, 1.0, size=num_samples), side="right")
        idx = np.clip(idx, 0, n_mu * n_kl - 1)
        i_m = idx // n_kl
        i_k = idx % n_kl
        s_mu = np.clip(mu_grid[i_m] + rng.uniform(-0.5 * d_mu, 0.5 * d_mu, size=num_samples), prior.low[0], prior.high[0])
        s_kl = np.clip(kl_grid[i_k] + rng.uniform(-0.5 * d_kl, 0.5 * d_kl, size=num_samples), prior.low[1], prior.high[1])
        samples[j, :, 0] = s_mu
        samples[j, :, 1] = s_kl
        sample_logp[j] = log_post_grid[j, i_m, i_k]

    mae_mu = float(np.mean(np.abs(theta_mean[:, 0] - theta_true[:, 0])))
    mae_kl = float(np.mean(np.abs(theta_mean[:, 1] - theta_true[:, 1])))
    rmae_mu = mae_mu / float(prior.ranges[0])
    rmae_kl = mae_kl / float(prior.ranges[1])

    alpha_grid = np.linspace(0.0, 1.0, num_acauc_levels)
    cov_2d, acauc_2d = compute_empirical_coverage_2d_hpd(sample_logp, true_logp, alpha_grid)
    cov_mu, acauc_mu = compute_empirical_coverage_1d(samples[:, :, 0], theta_true[:, 0], alpha_grid)
    cov_kl, acauc_kl = compute_empirical_coverage_1d(samples[:, :, 1], theta_true[:, 1], alpha_grid)

    results = EvaluationResults(
        test_nll=test_nll,
        prior_nll=prior.prior_nll,
        mae_mu=mae_mu,
        mae_kl=mae_kl,
        rmae_mu=rmae_mu,
        rmae_kl=rmae_kl,
        rmae_mean=0.5 * (rmae_mu + rmae_kl),
        acauc_2d_hpd=acauc_2d,
        acauc_mu_marginal=acauc_mu,
        acauc_kl_marginal=acauc_kl,
        alpha_grid=alpha_grid,
        cov_2d_hpd=cov_2d,
        cov_mu_marginal=cov_mu,
        cov_kl_marginal=cov_kl,
        theta_true=theta_true,
        theta_mean=theta_mean,
        theta_std=theta_std,
        n_events=n_events,
    )
    return results, log_post_grid


def evaluate_rate_only_baseline(
    sampler: ProcessSampler,
    prior: UniformPrior2D,
    theta_true: np.ndarray,
    n_events: np.ndarray,
    n_mu: int = 81,
    n_kl: int = 91,
    num_samples: int = 1000,
    num_acauc_levels: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[EvaluationResults, np.ndarray, np.ndarray, np.ndarray]:
    """Exact analytical Poisson-rate-only posterior p(theta | N)."""
    mu_grid, kl_grid, theta_mesh = make_theta_grid(prior, n_mu=n_mu, n_kl=n_kl)
    y_bkg = sampler.total_bkg_yield(None)
    y_hh = sampler.hh_expected_yield(theta_mesh[..., 0], theta_mesh[..., 1], None)
    lam_grid = np.maximum(y_bkg + y_hh, 1e-12)  # (n_mu, n_kl)

    n_arr = np.asarray(n_events, dtype=np.float64)[:, None, None]
    log_lik_grid = n_arr * np.log(lam_grid[None, :, :]) - lam_grid[None, :, :] - gammaln(n_arr + 1.0)
    results, log_post_grid = evaluate_grid_posteriors(
        log_lik_grid=log_lik_grid,
        theta_true=theta_true,
        n_events=n_events,
        prior=prior,
        mu_grid=mu_grid,
        kl_grid=kl_grid,
        num_samples=num_samples,
        num_acauc_levels=num_acauc_levels,
        rng=rng,
    )
    return results, log_post_grid, mu_grid, kl_grid


@dataclass
class SampledEventDatasetBatch:
    """Collection of sampled full-luminosity datasets for exact likelihood benchmarking."""

    theta: np.ndarray  # (n_sets, 2)
    n_events: np.ndarray  # (n_sets,)
    n_hh: np.ndarray  # (n_sets,)
    m_bbtt: list[np.ndarray]  # length n_sets
    m_bb: list[np.ndarray]  # length n_sets
    proc_indices: list[dict[str, np.ndarray]]  # length n_sets: proc_name -> event indices in split


def _unstd_m_bbtt_m_bb(evt_features: np.ndarray, stats: FeatureStats) -> tuple[np.ndarray, np.ndarray]:
    """Recover physical (m_bbtt, m_bb) in GeV from standardized SplitProcessPool.evt_features."""
    m_bbtt = (evt_features[:, 1].astype(np.float64) * float(stats.evt_std[1]) + float(stats.evt_mean[1])) * 300.0
    m_bb = (evt_features[:, 2].astype(np.float64) * float(stats.evt_std[2]) + float(stats.evt_mean[2])) * 125.0
    return m_bbtt, m_bb


def sample_full_test_sets(
    sampler: ProcessSampler,
    prior: UniformPrior2D,
    feature_stats: FeatureStats,
    n_sets: int,
    rng: np.random.Generator,
    fixed_theta: np.ndarray | None = None,
) -> SampledEventDatasetBatch:
    """Sample full Poisson-count datasets retaining exact event indices and (m_bbtt, m_bb)."""
    if fixed_theta is not None:
        thetas = np.asarray(fixed_theta, dtype=np.float32)
        n_sets = len(thetas)
    else:
        thetas = prior.sample(n_sets, rng=rng)

    n_events = np.zeros(n_sets, dtype=np.int64)
    n_hh_arr = np.zeros(n_sets, dtype=np.int64)
    m_bbtt_list: list[np.ndarray] = []
    m_bb_list: list[np.ndarray] = []
    proc_indices_list: list[dict[str, np.ndarray]] = []

    for j in range(n_sets):
        mu = float(thetas[j, 0])
        kl = float(thetas[j, 1])
        m_bbtt_parts = []
        m_bb_parts = []
        idx_map: dict[str, np.ndarray] = {}

        for b in sampler.backgrounds:
            lam_b = sampler.bkg_yields[b]
            n_b = int(rng.poisson(lam_b)) if lam_b > 0.0 else 0
            idx_b = sampler._draw_process_indices(b, n_b, kl, rng)
            idx_map[b] = idx_b
            if n_b > 0:
                m1, m2 = _unstd_m_bbtt_m_bb(sampler.split.processes[b].evt_features[idx_b], feature_stats)
                m_bbtt_parts.append(m1)
                m_bb_parts.append(m2)

        lam_s = float(sampler.hh_expected_yield(mu, kl, None))
        n_s = int(rng.poisson(max(lam_s, 0.0))) if lam_s > 0.0 else 0
        idx_s = sampler._draw_process_indices(sampler.signal, n_s, kl, rng)
        idx_map[sampler.signal] = idx_s
        if n_s > 0:
            m1, m2 = _unstd_m_bbtt_m_bb(sampler.split.processes[sampler.signal].evt_features[idx_s], feature_stats)
            m_bbtt_parts.append(m1)
            m_bb_parts.append(m2)

        n_tot = sum(len(v) for v in idx_map.values())
        n_events[j] = n_tot
        n_hh_arr[j] = n_s
        proc_indices_list.append(idx_map)
        if n_tot > 0:
            m_bbtt_list.append(np.concatenate(m_bbtt_parts).astype(np.float64))
            m_bb_list.append(np.concatenate(m_bb_parts).astype(np.float64))
        else:
            m_bbtt_list.append(np.empty(0, dtype=np.float64))
            m_bb_list.append(np.empty(0, dtype=np.float64))

    return SampledEventDatasetBatch(
        theta=thetas,
        n_events=n_events,
        n_hh=n_hh_arr,
        m_bbtt=m_bbtt_list,
        m_bb=m_bb_list,
        proc_indices=proc_indices_list,
    )


class BinnedLikelihoodModel:
    """2D binned Poisson likelihood over (m_bbtt, m_bb) built from MC templates."""

    def __init__(
        self,
        sampler: ProcessSampler,
        feature_stats: FeatureStats,
        m_bbtt_edges: np.ndarray | None = None,
        m_bb_edges: np.ndarray | None = None,
        eps: float = 1e-6,
    ) -> None:
        self.m_bbtt_edges = (
            m_bbtt_edges
            if m_bbtt_edges is not None
            else np.array([200.0, 275.0, 350.0, 450.0, 600.0, 850.0, 1e6], dtype=np.float64)
        )
        self.m_bb_edges = (
            m_bb_edges
            if m_bb_edges is not None
            else np.array([75.0, 100.0, 135.0, 150.0], dtype=np.float64)
        )
        self.eps = eps
        n_b1 = len(self.m_bbtt_edges) - 1
        n_b2 = len(self.m_bb_edges) - 1

        self.bkg_hist = np.zeros((n_b1, n_b2), dtype=np.float64)
        for b in sampler.backgrounds:
            pool = sampler.split.processes[b]
            m1, m2 = _unstd_m_bbtt_m_bb(pool.evt_features, feature_stats)
            h, _, _ = np.histogram2d(
                m1, m2, bins=[self.m_bbtt_edges, self.m_bb_edges], weights=pool.weights[:, 0]
            )
            self.bkg_hist += h

        sig = sampler.split.processes[sampler.signal]
        m1_s, m2_s = _unstd_m_bbtt_m_bb(sig.evt_features, feature_stats)
        w_s = sampler.hh_weights
        self.hh_hists = np.stack(
            [
                np.histogram2d(m1_s, m2_s, bins=[self.m_bbtt_edges, self.m_bb_edges], weights=w_s[:, c])[0]
                for c in range(3)
            ],
            axis=0,
        )  # (3, n_b1, n_b2)

    def expected_hists_grid(self, mu_grid: np.ndarray, kl_grid: np.ndarray) -> np.ndarray:
        """Return expected bin yields of shape (n_mu, n_kl, n_bins)."""
        c_nom, c_kl0, c_kl2 = hh_reweight_coefficients(kl_grid)
        hh_kl = (
            c_nom[:, None, None] * self.hh_hists[0][None, :, :]
            + c_kl0[:, None, None] * self.hh_hists[1][None, :, :]
            + c_kl2[:, None, None] * self.hh_hists[2][None, :, :]
        )  # (n_kl, n_b1, n_b2)
        lam = self.bkg_hist[None, None, :, :] + mu_grid[:, None, None, None] * hh_kl[None, :, :, :]
        lam = np.maximum(lam, self.eps)
        return lam.reshape(len(mu_grid), len(kl_grid), -1)

    def evaluate(
        self,
        datasets: SampledEventDatasetBatch,
        prior: UniformPrior2D,
        n_mu: int = 81,
        n_kl: int = 91,
        num_samples: int = 1000,
        num_acauc_levels: int = 100,
        rng: np.random.Generator | None = None,
    ) -> tuple[EvaluationResults, np.ndarray, np.ndarray, np.ndarray]:
        mu_grid, kl_grid, _ = make_theta_grid(prior, n_mu=n_mu, n_kl=n_kl)
        lam_flat = self.expected_hists_grid(mu_grid, kl_grid)  # (n_mu, n_kl, n_bins)
        log_lam = np.log(lam_flat)
        sum_lam = np.sum(lam_flat, axis=-1)  # (n_mu, n_kl)

        n_sets = len(datasets.theta)
        obs_counts = np.zeros((n_sets, lam_flat.shape[-1]), dtype=np.float64)
        for j in range(n_sets):
            if datasets.n_events[j] > 0:
                h, _, _ = np.histogram2d(
                    datasets.m_bbtt[j],
                    datasets.m_bb[j],
                    bins=[self.m_bbtt_edges, self.m_bb_edges],
                )
                obs_counts[j] = h.reshape(-1)

        # (n_sets, n_mu, n_kl)
        log_lik_grid = np.einsum("bk,ijk->bij", obs_counts, log_lam) - sum_lam[None, :, :]
        results, log_post_grid = evaluate_grid_posteriors(
            log_lik_grid=log_lik_grid,
            theta_true=datasets.theta,
            n_events=datasets.n_events,
            prior=prior,
            mu_grid=mu_grid,
            kl_grid=kl_grid,
            num_samples=num_samples,
            num_acauc_levels=num_acauc_levels,
            rng=rng,
        )
        return results, log_post_grid, mu_grid, kl_grid


class SingleEventNREOracle(nn.Module):
    """N=1 Neural Ratio Estimator log r(x | theta) = log p(x | theta) - log p_ref(x).

    Reuses PerEventEmbedder from models.py so feature encoding matches PAIRS.
    """

    def __init__(
        self,
        jet_feat_dim: int,
        evt_feat_dim: int,
        theta_mean: np.ndarray,
        theta_std: np.ndarray,
        cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.embedder = PerEventEmbedder(jet_feat_dim, evt_feat_dim, cfg)
        d = cfg.embed_dim
        self.theta_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, d),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            ResidualMLPBlock(128, dropout=cfg.dropout),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.register_buffer("theta_mean", torch.tensor(theta_mean, dtype=torch.float32))
        self.register_buffer("theta_std", torch.tensor(theta_std, dtype=torch.float32))

    def encode_events(
        self,
        jet_feats: torch.Tensor,
        jet_mask: torch.Tensor,
        evt_feats: torch.Tensor,
    ) -> torch.Tensor:
        return self.embedder(jet_feats, jet_mask, evt_feats)

    def forward_from_embeddings(self, emb: torch.Tensor, theta_phys: torch.Tensor) -> torch.Tensor:
        th_std = (theta_phys - self.theta_mean) / self.theta_std
        th_emb = self.theta_net(th_std)
        return self.head(torch.cat([emb, th_emb], dim=-1)).squeeze(-1)


def _sample_single_events_from_thetas(
    sampler: ProcessSampler,
    thetas: np.ndarray,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one event x_i ~ p(x | theta_i) for each row in thetas (B, 2)."""
    batch_size = len(thetas)
    set_sizes = np.ones(batch_size, dtype=np.int64)
    jf, jm, ef = sampler.sample_small_sets_batch(thetas, set_sizes, rng)
    return jf[:, 0], jm[:, 0], ef[:, 0]


def train_nre_oracle(
    train_split: SplitDataset,
    val_split: SplitDataset,
    prior: UniformPrior2D,
    feature_stats: FeatureStats,
    cfg: ModelConfig,
    device: torch.device,
    hh_k_factor: float = 1.0,
    epochs: int = 15,
    batches_per_epoch: int = 120,
    batch_size: int = 512,
    lr: float = 5e-4,
    seed: int = 123,
    mu_max_train: float = 300.0,
) -> SingleEventNREOracle:
    """Train N=1 NRE oracle distinguishing (x, theta) ~ p(x|theta)p(theta) vs p_ref(x)p(theta)."""
    train_sampler = ProcessSampler(train_split, hh_k_factor=hh_k_factor)
    val_sampler = ProcessSampler(val_split, hh_k_factor=hh_k_factor)

    sig_pool = train_split.processes[train_split.signal]
    model = SingleEventNREOracle(
        jet_feat_dim=sig_pool.jet_features.shape[-1],
        evt_feat_dim=sig_pool.evt_features.shape[-1],
        theta_mean=prior.mean,
        theta_std=prior.std,
        cfg=cfg,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    rng = np.random.default_rng(seed)

    # Reference mixture p_ref(x): uniform marginal over the training prior
    def sample_pair_batch(smp: ProcessSampler, bs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        th_pos = prior.sample(bs, rng=rng, mu_max_override=mu_max_train)
        th_ref = prior.sample(bs, rng=rng, mu_max_override=mu_max_train)
        jf_p, jm_p, ef_p = _sample_single_events_from_thetas(smp, th_pos, rng)
        jf_n, jm_n, ef_n = _sample_single_events_from_thetas(smp, th_ref, rng)
        jf = torch.cat([jf_p, jf_n], dim=0)
        jm = torch.cat([jm_p, jm_n], dim=0)
        ef = torch.cat([ef_p, ef_n], dim=0)
        th = torch.from_numpy(np.concatenate([th_pos, th_pos], axis=0)).float()
        y = torch.cat([torch.ones(bs), torch.zeros(bs)], dim=0)
        return jf, jm, ef, th, y

    val_batch = sample_pair_batch(val_sampler, 4096)
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for _ in range(batches_per_epoch):
            jf, jm, ef, th, y = sample_pair_batch(train_sampler, batch_size)
            jf, jm, ef, th, y = jf.to(device), jm.to(device), ef.to(device), th.to(device), y.to(device)
            emb = model.encode_events(jf, jm, ef)
            logits = model.forward_from_embeddings(emb, th)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            jf, jm, ef, th, y = [t.to(device) for t in val_batch]
            emb = model.encode_events(jf, jm, ef)
            val_loss = float(F.binary_cross_entropy_with_logits(model.forward_from_embeddings(emb, th), y).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[NRE Oracle] Epoch {epoch:02d}/{epochs} | train_bce={tr_loss / batches_per_epoch:.4f} | val_bce={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_nre_oracle_sets(
    oracle: SingleEventNREOracle,
    test_sampler: ProcessSampler,
    datasets: SampledEventDatasetBatch,
    prior: UniformPrior2D,
    device: torch.device,
    n_mu: int = 81,
    n_kl: int = 91,
    num_samples: int = 1000,
    num_acauc_levels: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[EvaluationResults, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate exact unbinned N=1 NRE log-likelihood sum_i log r(x_i | theta) + log Pois(N | Lambda(theta))."""
    oracle.eval()
    mu_grid, kl_grid, theta_mesh = make_theta_grid(prior, n_mu=n_mu, n_kl=n_kl)
    y_bkg = test_sampler.total_bkg_yield(None)
    y_hh = test_sampler.hh_expected_yield(theta_mesh[..., 0], theta_mesh[..., 1], None)
    lam_grid = np.maximum(y_bkg + y_hh, 1e-12)

    # Pre-embed every unique test MC event once
    proc_embs: dict[str, torch.Tensor] = {}
    for pname, pool in test_sampler.split.processes.items():
        embs = []
        n_p = pool.num_events
        for s in range(0, n_p, 2048):
            e = min(s + 2048, n_p)
            jf = torch.from_numpy(pool.jet_features[s:e]).to(device)
            jm = torch.from_numpy(pool.jet_mask[s:e]).to(device)
            ef = torch.from_numpy(pool.evt_features[s:e]).to(device)
            embs.append(oracle.encode_events(jf, jm, ef).cpu())
        proc_embs[pname] = torch.cat(embs, dim=0) if embs else torch.empty(0, oracle.embedder.out_proj.out_features)

    # Precompute theta_net outputs on the 2D grid: (G, d)
    theta_flat = torch.from_numpy(theta_mesh.reshape(-1, 2)).float().to(device)
    th_std = (theta_flat - oracle.theta_mean) / oracle.theta_std
    th_emb = oracle.theta_net(th_std)  # (G, d)
    n_grid = theta_flat.shape[0]

    n_sets = len(datasets.theta)
    log_lik_grid = np.empty((n_sets, n_mu, n_kl), dtype=np.float64)

    for j in range(n_sets):
        n_tot = float(datasets.n_events[j])
        log_pois = n_tot * np.log(lam_grid) - lam_grid - gammaln(n_tot + 1.0)
        parts = [proc_embs[pname][idxs] for pname, idxs in datasets.proc_indices[j].items() if len(idxs) > 0]
        if not parts:
            log_lik_grid[j] = log_pois
            continue
        ev_emb = torch.cat(parts, dim=0).to(device)  # (N_j, d)
        sum_log_r = torch.zeros(n_grid, dtype=torch.float64, device=device)

        # Process events in chunks to keep memory tiny even for N_j ~ 16,500
        for s in range(0, ev_emb.shape[0], 512):
            eb = ev_emb[s : s + 512]  # (m, d)
            m = eb.shape[0]
            eb_exp = eb.unsqueeze(1).expand(m, n_grid, -1)
            th_exp = th_emb.unsqueeze(0).expand(m, n_grid, -1)
            logits = oracle.head(torch.cat([eb_exp, th_exp], dim=-1)).squeeze(-1)  # (m, G)
            sum_log_r += logits.double().sum(dim=0)

        log_lik_grid[j] = log_pois + sum_log_r.cpu().numpy().reshape(n_mu, n_kl)

    results, log_post_grid = evaluate_grid_posteriors(
        log_lik_grid=log_lik_grid,
        theta_true=datasets.theta,
        n_events=datasets.n_events,
        prior=prior,
        mu_grid=mu_grid,
        kl_grid=kl_grid,
        num_samples=num_samples,
        num_acauc_levels=num_acauc_levels,
        rng=rng,
    )
    return results, log_post_grid, mu_grid, kl_grid
