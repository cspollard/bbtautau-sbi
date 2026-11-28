"""Diagnostic visualizations for PAIRS bb-tautau inference."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import torch

from .config import PriorConfig
from .data import ProcessData
from .metrics import EvaluationResults
from .models import PairsModel
from .physics import compute_hh_weights


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_loss_curves(
    hist_s1: dict[str, list[float]] | None,
    hist_s3: dict[str, list[float]] | None,
    prior_nll: float,
    out_dir: Path,
) -> list[Path]:
    """Plot Stage 1 and Stage 3 training & validation NLL curves."""
    paths: list[Path] = []
    if hist_s1 and hist_s1.get("epoch"):
        s1_prior = float(hist_s1.get("prior_nll", prior_nll))
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(hist_s1["epoch"], hist_s1["train_nll"], label="Train NLL", lw=2.0)
        ax.plot(hist_s1["epoch"], hist_s1["val_nll"], label="Val NLL", lw=2.0)
        ax.axhline(s1_prior, color="gray", ls="--", lw=1.2, label=f"Stage 1 Prior ({s1_prior:.3f})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL (nats, physical units)")
        ax.set_title("PAIRS Stage 1: Embedder Pretraining ($N \\in \\{1, 2\\}$)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = out_dir / "stage1_loss.png"
        _ensure_dir(p)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    if hist_s3 and hist_s3.get("epoch"):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(hist_s3["epoch"], hist_s3["train_nll"], label="Train NLL", lw=2.0)
        ax.plot(hist_s3["epoch"], hist_s3["val_nll"], label="Val NLL", lw=2.0)
        ax.axhline(prior_nll, color="gray", ls="--", lw=1.2, label=f"Uniform Prior ({prior_nll:.3f})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL (nats, physical units)")
        ax.set_title("PAIRS Stage 3: Dataset-Wide Flow Finetuning ($N \\sim \\mathrm{Poisson}(\\Lambda)$)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = out_dir / "stage3_loss.png"
        _ensure_dir(p)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    return paths


def plot_yields_vs_kl(
    processes: dict[str, ProcessData],
    prior_cfg: PriorConfig,
    out_dir: Path,
) -> Path:
    """Plot HH expected signal yield vs kappa_lambda and background yields."""
    kl_grid = np.linspace(prior_cfg.kl_min, prior_cfg.kl_max, 200)
    hh = processes["HH"]
    yields_mu1 = np.array([compute_hh_weights(hh.weights, 1.0, kl).sum() for kl in kl_grid])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax1.plot(kl_grid, yields_mu1, lw=2.2, color="#1f77b4", label="$\\mu_{HH} = 1$")
    ax1.plot(kl_grid, 5.0 * yields_mu1, lw=1.8, ls="--", color="#ff7f0e", label="$\\mu_{HH} = 5$")
    ax1.axvline(1.0, color="crimson", ls=":", lw=1.4, label="SM $\\kappa_\\lambda = 1$")
    ax1.set_xlabel("$\\kappa_\\lambda$")
    ax1.set_ylabel("Expected $HH \\to bb\\tau\\tau$ Yield (100 fb$^{-1}$)")
    ax1.set_title("Selected $HH$ Signal Yield vs $\\kappa_\\lambda$")
    ax1.legend()
    ax1.grid(alpha=0.25)

    # Stacked / bar comparison of selected process yields
    names = ["top", "DYbb", "ZH", "higgs", "HH (SM)"]
    vals = [
        processes["top"].weights[:, 0].sum(),
        processes["DYbb"].weights[:, 0].sum(),
        processes["ZH"].weights[:, 0].sum(),
        processes["higgs"].weights[:, 0].sum(),
        processes["HH"].weights[:, 0].sum(),
    ]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#dd8452"]
    bars = ax2.bar(names, vals, color=colors)
    ax2.set_yscale("log")
    ax2.set_ylabel("Expected Yield (100 fb$^{-1}$, log scale)")
    ax2.set_title("Selected Event Yields per Process")
    ax2.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val * 1.15,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    fig.tight_layout()
    p = out_dir / "yields_vs_kl.png"
    _ensure_dir(p)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


@torch.no_grad()
def _posterior_grid_and_samples(
    model: PairsModel,
    t_bar_1: np.ndarray,
    count_1: int,
    prior_cfg: PriorConfig,
    device: torch.device,
    num_samples: int = 4000,
    grid_res: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    tb_t = torch.from_numpy(t_bar_1[None, :]).float().to(device)
    n_t = torch.tensor([count_1], dtype=torch.long, device=device)

    samples = model.sample_from_aggregates(tb_t, n_t, num_samples=num_samples)[0].cpu().numpy()

    mu_g = np.linspace(prior_cfg.mu_min, prior_cfg.mu_max, grid_res)
    kl_g = np.linspace(prior_cfg.kl_min, prior_cfg.kl_max, grid_res)
    mg, kg = np.meshgrid(mu_g, kl_g, indexing="xy")
    grid_pts = np.stack([mg.ravel(), kg.ravel()], axis=-1).astype(np.float32)

    tb_rep = tb_t.expand(grid_pts.shape[0], -1)
    n_rep = n_t.expand(grid_pts.shape[0])
    logp = model.log_prob_from_aggregates(
        torch.from_numpy(grid_pts).to(device), tb_rep, n_rep
    ).cpu().numpy()
    dens = np.exp(logp - logp.max()).reshape(grid_res, grid_res)
    return mg, kg, dens, samples


def _hpd_contour_levels(dens: np.ndarray, probs: tuple[float, ...] = (0.68, 0.95)) -> list[float]:
    flat = np.sort(dens.ravel())[::-1]
    cdf = np.cumsum(flat) / np.sum(flat)
    levels = []
    for p in probs:
        idx = int(np.searchsorted(cdf, p))
        idx = min(idx, len(flat) - 1)
        levels.append(float(flat[idx]))
    # contour() expects strictly increasing levels
    return sorted(levels)


def plot_posterior_contours(
    model: PairsModel,
    bench_agg: dict[str, np.ndarray],
    prior_cfg: PriorConfig,
    device: torch.device,
    out_dir: Path,
) -> list[Path]:
    """Plot 2D posterior contours for SM and benchmark pseudo-experiments."""
    paths: list[Path] = []
    n_bench = int(bench_agg["theta"].shape[0])
    if n_bench == 0:
        return paths

    # Dedicated SM corner-like plot (index 0 if SM is first)
    sm_idx = 0
    for i in range(n_bench):
        if np.allclose(bench_agg["theta"][i], [1.0, 1.0], atol=1e-3):
            sm_idx = i
            break

    mg, kg, dens, samples = _posterior_grid_and_samples(
        model,
        bench_agg["T_bar"][sm_idx],
        int(bench_agg["N"][sm_idx]),
        prior_cfg,
        device=device,
    )
    th_true = bench_agg["theta"][sm_idx]
    n_obs = int(bench_agg["N"][sm_idx])
    n_hh = int(bench_agg["N_hh"][sm_idx])

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1.2), height_ratios=(1.2, 4), wspace=0.06, hspace=0.06)
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    ax_joint.contourf(mg, kg, dens, levels=30, cmap="Blues")
    levels = _hpd_contour_levels(dens, probs=(0.68, 0.95))
    if len(levels) == 2 and levels[1] > levels[0]:
        cs = ax_joint.contour(
            mg, kg, dens, levels=levels, colors=["#08519c", "#08306b"], linestyles=["--", "-"], linewidths=1.8
        )
        ax_joint.clabel(cs, fmt={levels[0]: "95%", levels[1]: "68%"}, fontsize=8)
    ax_joint.scatter([th_true[0]], [th_true[1]], color="crimson", marker="*", s=160, zorder=5, label=f"True ({th_true[0]:.1f}, {th_true[1]:.1f})")
    ax_joint.scatter([samples[:, 0].mean()], [samples[:, 1].mean()], color="darkorange", marker="o", s=50, zorder=5, label="Post. mean")
    ax_joint.set_xlabel("$\\mu_{HH}$")
    ax_joint.set_ylabel("$\\kappa_\\lambda$")
    ax_joint.set_xlim(prior_cfg.mu_min, prior_cfg.mu_max)
    ax_joint.set_ylim(prior_cfg.kl_min, prior_cfg.kl_max)
    ax_joint.legend(loc="upper right", fontsize=8.5)
    ax_joint.grid(alpha=0.2)

    # Marginals
    in_mu = samples[(samples[:, 0] >= prior_cfg.mu_min) & (samples[:, 0] <= prior_cfg.mu_max), 0]
    in_kl = samples[(samples[:, 1] >= prior_cfg.kl_min) & (samples[:, 1] <= prior_cfg.kl_max), 1]
    ax_top.hist(in_mu, bins=35, density=True, color="#4c72b0", alpha=0.65)
    if len(in_mu) > 10:
        kde_mu = gaussian_kde(in_mu)
        xs = np.linspace(prior_cfg.mu_min, prior_cfg.mu_max, 200)
        ax_top.plot(xs, kde_mu(xs), color="#08306b", lw=1.5)
    ax_top.axvline(th_true[0], color="crimson", ls="--", lw=1.4)
    ax_top.set_ylabel("Density")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_title(f"SM Joint Posterior ($N = {n_obs:,}$, $N_{{HH}} = {n_hh}$)")

    ax_right.hist(in_kl, bins=35, density=True, orientation="horizontal", color="#4c72b0", alpha=0.65)
    if len(in_kl) > 10:
        kde_kl = gaussian_kde(in_kl)
        ys = np.linspace(prior_cfg.kl_min, prior_cfg.kl_max, 200)
        ax_right.plot(kde_kl(ys), ys, color="#08306b", lw=1.5)
    ax_right.axhline(th_true[1], color="crimson", ls="--", lw=1.4)
    ax_right.set_xlabel("Density")
    ax_right.tick_params(labelleft=False)

    p_sm = out_dir / "posterior_sm.png"
    _ensure_dir(p_sm)
    fig.savefig(p_sm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p_sm)

    # Grid of all benchmark test datasets
    ncols = min(3, n_bench)
    nrows = int(np.ceil(n_bench / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.1 * nrows), squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i >= n_bench:
            ax.axis("off")
            continue
        mg_i, kg_i, dens_i, samp_i = _posterior_grid_and_samples(
            model,
            bench_agg["T_bar"][i],
            int(bench_agg["N"][i]),
            prior_cfg,
            device=device,
        )
        t_i = bench_agg["theta"][i]
        lv = _hpd_contour_levels(dens_i, probs=(0.68, 0.95))
        ax.contourf(mg_i, kg_i, dens_i, levels=25, cmap="Blues")
        if len(lv) == 2 and lv[1] > lv[0]:
            ax.contour(mg_i, kg_i, dens_i, levels=lv, colors=["#08519c", "#08306b"], linestyles=["--", "-"], linewidths=1.5)
        ax.scatter([t_i[0]], [t_i[1]], color="crimson", marker="*", s=130, zorder=5, label="Truth")
        ax.scatter([samp_i[:, 0].mean()], [samp_i[:, 1].mean()], color="darkorange", marker="o", s=35, zorder=5, label="Mean")
        ax.set_xlim(prior_cfg.mu_min, prior_cfg.mu_max)
        ax.set_ylim(prior_cfg.kl_min, prior_cfg.kl_max)
        ax.set_xlabel("$\\mu_{HH}$")
        ax.set_ylabel("$\\kappa_\\lambda$")
        ax.set_title(
            f"True=({t_i[0]:.1f}, {t_i[1]:.1f}) | $N$={int(bench_agg['N'][i]):,}, $N_{{HH}}$={int(bench_agg['N_hh'][i])}",
            fontsize=9.5,
        )
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("PAIRS 2D Posterior Contours (68% solid, 95% dashed)", fontsize=12, y=1.01)
    fig.tight_layout()
    p_grid = out_dir / "posterior_contours_grid.png"
    _ensure_dir(p_grid)
    fig.savefig(p_grid, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p_grid)
    return paths


def plot_predicted_vs_true(
    eval_res: EvaluationResults,
    prior_cfg: PriorConfig,
    out_dir: Path,
) -> Path:
    """Plot predicted posterior mean vs true and standardized pulls for both parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.8))

    labels = ["$\\mu_{HH}$", "$\\kappa_\\lambda$"]
    bounds = [(prior_cfg.mu_min, prior_cfg.mu_max), (prior_cfg.kl_min, prior_cfg.kl_max)]
    rmaes = [eval_res.rmae_mu, eval_res.rmae_kl]

    for d in range(2):
        true_d = eval_res.theta_true[:, d]
        pred_d = eval_res.theta_mean[:, d]
        std_d = np.maximum(eval_res.theta_std[:, d], 1e-6)
        pull_d = (pred_d - true_d) / std_d

        # Scatter: predicted vs true
        ax = axes[0, d]
        ax.scatter(true_d, pred_d, s=10, alpha=0.35, color="#1f77b4", edgecolors="none")
        lo, hi = bounds[d]
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal $y = x$")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"True {labels[d]}")
        ax.set_ylabel(f"Posterior Mean {labels[d]}")
        ax.set_title(f"{labels[d]} Recovery (RMAE = {rmaes[d]:.4f})")
        ax.legend()
        ax.grid(alpha=0.25)

        # Pull distribution
        ax_p = axes[1, d]
        ax_p.hist(pull_d, bins=35, range=(-4, 4), density=True, alpha=0.7, color="#4c72b0", label="Empirical pulls")
        xs = np.linspace(-4, 4, 200)
        ax_p.plot(xs, np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi), "r--", lw=1.6, label="$\\mathcal{N}(0, 1)$")
        ax_p.set_xlabel(f"Pull $(\\hat{{\\theta}} - \\theta) / \\sigma$ for {labels[d]}")
        ax_p.set_ylabel("Density")
        ax_p.set_title(f"{labels[d]} Pulls (mean={pull_d.mean():.2f}, std={pull_d.std():.2f})")
        ax_p.legend()
        ax_p.grid(alpha=0.25)

    fig.tight_layout()
    p = out_dir / "predicted_vs_true.png"
    _ensure_dir(p)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_calibration_acauc(
    eval_res: EvaluationResults,
    out_dir: Path,
) -> Path:
    """Plot empirical coverage vs nominal credibility alpha and annotate ACAUC."""
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    a = eval_res.alpha_grid
    ax.plot(a, a, "k--", lw=1.4, label="Perfect calibration")
    ax.plot(
        a,
        eval_res.cov_2d_hpd,
        lw=2.2,
        color="#1f77b4",
        label=f"Joint 2D HPD (ACAUC = {eval_res.acauc_2d_hpd:.4f})",
    )
    ax.plot(
        a,
        eval_res.cov_mu_marginal,
        lw=1.8,
        color="#ff7f0e",
        label=f"$\\mu_{{HH}}$ marginal (ACAUC = {eval_res.acauc_mu_marginal:.4f})",
    )
    ax.plot(
        a,
        eval_res.cov_kl_marginal,
        lw=1.8,
        color="#2ca02c",
        label=f"$\\kappa_\\lambda$ marginal (ACAUC = {eval_res.acauc_kl_marginal:.4f})",
    )
    ax.set_xlabel("Nominal Credible Level $\\alpha$")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title(
        f"PAIRS Posterior Calibration (Test NLL = {eval_res.test_nll:.3f} vs Prior {eval_res.prior_nll:.3f})"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = out_dir / "calibration_acauc.png"
    _ensure_dir(p)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p
