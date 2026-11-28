"""Evaluation metrics for amortized Bayesian inference: NLL, RMAE, and ACAUC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .models import PairsModel
from .physics import UniformPrior2D


@dataclass
class EvaluationResults:
    """Summary metrics and per-test-dataset predictions."""

    test_nll: float
    prior_nll: float
    mae_mu: float
    mae_kl: float
    rmae_mu: float
    rmae_kl: float
    rmae_mean: float
    acauc_2d_hpd: float
    acauc_mu_marginal: float
    acauc_kl_marginal: float
    alpha_grid: np.ndarray
    cov_2d_hpd: np.ndarray
    cov_mu_marginal: np.ndarray
    cov_kl_marginal: np.ndarray
    theta_true: np.ndarray
    theta_mean: np.ndarray
    theta_std: np.ndarray
    n_events: np.ndarray


def compute_empirical_coverage_1d(
    samples_1d: np.ndarray,
    true_1d: np.ndarray,
    alpha_grid: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute central credible interval coverage curve and ACAUC for a 1D marginal.

    Args:
        samples_1d: Posterior samples of shape (n_test, n_samples).
        true_1d: Ground truth values of shape (n_test,).
        alpha_grid: Credible levels in [0, 1].
    """
    pit = np.mean(samples_1d <= true_1d[:, None], axis=1)
    # Minimal central interval level containing the truth: |2 * pit - 1|
    u_central = np.abs(2.0 * pit - 1.0)
    coverage = np.mean(u_central[:, None] <= alpha_grid[None, :], axis=0)
    acauc = float(np.trapezoid(np.abs(coverage - alpha_grid), alpha_grid))
    return coverage, acauc


def compute_empirical_coverage_2d_hpd(
    sample_log_probs: np.ndarray,
    true_log_probs: np.ndarray,
    alpha_grid: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute 2D Highest Posterior Density (HPD) credible region coverage and ACAUC.

    For each test example j, u_j = P_{theta ~ q}[log q(theta|X_j) >= log q(theta_true|X_j)]
    is the smallest HPD credible level containing theta_true.
    """
    u_hpd = np.mean(sample_log_probs >= true_log_probs[:, None], axis=1)
    coverage = np.mean(u_hpd[:, None] <= alpha_grid[None, :], axis=0)
    acauc = float(np.trapezoid(np.abs(coverage - alpha_grid), alpha_grid))
    return coverage, acauc


@torch.no_grad()
def evaluate_stage3_model(
    model: PairsModel,
    prior: UniformPrior2D,
    agg_test: dict[str, np.ndarray],
    device: torch.device,
    num_samples: int = 1000,
    num_acauc_levels: int = 100,
    batch_size: int = 256,
) -> EvaluationResults:
    """Evaluate a trained PAIRS Stage 3 model on cached test aggregates."""
    model.eval()
    thetas = torch.from_numpy(agg_test["theta"]).float()
    t_bars = torch.from_numpy(agg_test["T_bar"]).float()
    counts = torch.from_numpy(agg_test["N"]).long()
    n_test = int(thetas.shape[0])

    all_nll = []
    all_samples = []
    all_sample_logp = []

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        th_b = thetas[start:end].to(device)
        tb_b = t_bars[start:end].to(device)
        n_b = counts[start:end].to(device)

        logp_true = model.log_prob_from_aggregates(th_b, tb_b, n_b)
        nll_b = -logp_true
        samples_b = model.sample_from_aggregates(tb_b, n_b, num_samples=num_samples)

        # Evaluate log_prob of drawn samples for exact 2D HPD ACAUC
        cond = model.build_context(tb_b, n_b)
        cond_rep = cond.unsqueeze(1).expand(-1, num_samples, -1)
        samples_std = model.standardize_theta(samples_b)
        sample_logp_b = model.flow_head.log_prob(samples_std, cond_rep) - model.log_theta_std_sum

        all_nll.append(nll_b.cpu().numpy())
        all_samples.append(samples_b.cpu().numpy())
        all_sample_logp.append(sample_logp_b.cpu().numpy())

    nll_arr = np.concatenate(all_nll, axis=0)
    samples_arr = np.concatenate(all_samples, axis=0)  # (n_test, num_samples, 2)
    sample_logp_arr = np.concatenate(all_sample_logp, axis=0)
    true_logp_arr = -nll_arr

    theta_true = agg_test["theta"]
    theta_mean = samples_arr.mean(axis=1)
    theta_std = samples_arr.std(axis=1)

    abs_err = np.abs(theta_mean - theta_true)
    mae_mu = float(abs_err[:, 0].mean())
    mae_kl = float(abs_err[:, 1].mean())
    r_mu, r_kl = prior.cfg.ranges
    rmae_mu = mae_mu / float(r_mu)
    rmae_kl = mae_kl / float(r_kl)
    rmae_mean = 0.5 * (rmae_mu + rmae_kl)

    alpha_grid = np.linspace(0.0, 1.0, num_acauc_levels)
    cov_2d, acauc_2d = compute_empirical_coverage_2d_hpd(sample_logp_arr, true_logp_arr, alpha_grid)
    cov_mu, acauc_mu = compute_empirical_coverage_1d(samples_arr[:, :, 0], theta_true[:, 0], alpha_grid)
    cov_kl, acauc_kl = compute_empirical_coverage_1d(samples_arr[:, :, 1], theta_true[:, 1], alpha_grid)

    return EvaluationResults(
        test_nll=float(nll_arr.mean()),
        prior_nll=float(prior.prior_nll),
        mae_mu=mae_mu,
        mae_kl=mae_kl,
        rmae_mu=rmae_mu,
        rmae_kl=rmae_kl,
        rmae_mean=rmae_mean,
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
        n_events=agg_test["N"],
    )
