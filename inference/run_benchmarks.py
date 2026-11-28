"""Run reference/oracle posterior benchmarks against PAIRS on identical test datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pairs_bbtautau.benchmarks import (
    BinnedLikelihoodModel,
    evaluate_nre_oracle_sets,
    evaluate_rate_only_baseline,
    sample_full_test_sets,
)
from pairs_bbtautau.config import (
    DEFAULT_INFERENCE_DIR,
    DataConfig,
    ModelConfig,
    PairsConfig,
    PriorConfig,
    resolve_device,
    seed_everything,
)
from pairs_bbtautau.data import build_splits, load_or_build_process_cache
from pairs_bbtautau.metrics import evaluate_stage3_model
from pairs_bbtautau.physics import ProcessSampler, UniformPrior2D
from pairs_bbtautau.stages import _embed_split, load_checkpoint, run_stage2, train_stage1, train_stage3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark PAIRS against Rate-only, Binned, and NRE Oracle posteriors")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_INFERENCE_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_INFERENCE_DIR / "cache")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_INFERENCE_DIR / "outputs")
    p.add_argument("--lumi", type=float, default=100_000.0)
    p.add_argument("--k-factor", type=float, default=1.0)
    p.add_argument("--n-test-sets", type=int, default=500)
    p.add_argument("--n-mu-grid", type=int, default=61)
    p.add_argument("--n-kl-grid", type=int, default=71)
    p.add_argument("--with-nre-oracle", action="store_true", help="Also train & evaluate N=1 NRE unbinned oracle.")
    p.add_argument("--nre-epochs", type=int, default=15)
    p.add_argument("--nre-batches", type=int, default=120)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--smoke", action="store_true", help="Fast lightweight smoke test.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_test_sets = 24
        args.n_mu_grid = 21
        args.n_kl_grid = 25
        args.nre_epochs = 2
        args.nre_batches = 4
        args.with_nre_oracle = True

    seed_everything(args.seed)
    device = resolve_device(args.device)
    cfg = PairsConfig(
        prior=PriorConfig(),
        data=DataConfig(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            lumi_pb_inv=args.lumi,
            hh_k_factor=args.k_factor,
        ),
        model=ModelConfig(),
        output_dir=args.output_dir,
    )
    prior = UniformPrior2D(cfg.prior)

    processes = load_or_build_process_cache(cfg.data)
    splits, feature_stats = build_splits(processes, cfg.data)
    train_sampler = ProcessSampler(splits["train_s3"], hh_k_factor=cfg.data.hh_k_factor)
    test_sampler = ProcessSampler(splits["test"], hh_k_factor=cfg.data.hh_k_factor)

    rng = np.random.default_rng(args.seed + 777)
    datasets = sample_full_test_sets(test_sampler, prior, feature_stats, n_sets=args.n_test_sets, rng=rng)

    summary: dict[str, dict[str, float]] = {}

    rate_res, _, _, _ = evaluate_rate_only_baseline(
        sampler=test_sampler,
        prior=prior,
        theta_true=datasets.theta,
        n_events=datasets.n_events,
        n_mu=args.n_mu_grid,
        n_kl=args.n_kl_grid,
        num_samples=500 if args.smoke else 1000,
        rng=np.random.default_rng(args.seed + 1),
    )
    summary["rate_only"] = {
        "test_nll": rate_res.test_nll,
        "rmae_mu": rate_res.rmae_mu,
        "rmae_kl": rate_res.rmae_kl,
        "acauc_2d": rate_res.acauc_2d_hpd,
    }

    binned = BinnedLikelihoodModel(train_sampler, feature_stats)
    binned_res, _, _, _ = binned.evaluate(
        datasets=datasets,
        prior=prior,
        n_mu=args.n_mu_grid,
        n_kl=args.n_kl_grid,
        num_samples=500 if args.smoke else 1000,
        rng=np.random.default_rng(args.seed + 2),
    )
    summary["binned_2d"] = {
        "test_nll": binned_res.test_nll,
        "rmae_mu": binned_res.rmae_mu,
        "rmae_kl": binned_res.rmae_kl,
        "acauc_2d": binned_res.acauc_2d_hpd,
    }

    if args.with_nre_oracle:
        from pairs_bbtautau.benchmarks import train_nre_oracle

        oracle = train_nre_oracle(
            train_split=splits["train_s1"],
            val_split=splits["val"],
            prior=prior,
            feature_stats=feature_stats,
            cfg=cfg.model,
            device=device,
            hh_k_factor=cfg.data.hh_k_factor,
            epochs=args.nre_epochs,
            batches_per_epoch=args.nre_batches,
            seed=args.seed,
        )
        nre_res, _, _, _ = evaluate_nre_oracle_sets(
            oracle=oracle,
            test_sampler=test_sampler,
            datasets=datasets,
            prior=prior,
            device=device,
            n_mu=args.n_mu_grid,
            n_kl=args.n_kl_grid,
            num_samples=500 if args.smoke else 1000,
            rng=np.random.default_rng(args.seed + 3),
        )
        summary["nre_oracle"] = {
            "test_nll": nre_res.test_nll,
            "rmae_mu": nre_res.rmae_mu,
            "rmae_kl": nre_res.rmae_kl,
            "acauc_2d": nre_res.acauc_2d_hpd,
        }

    # Include trained PAIRS model if checkpoint exists
    ckpt_path = args.output_dir / "checkpoints" / "stage3_best.pt"
    if ckpt_path.exists() and not args.smoke:
        model, _ = load_checkpoint(ckpt_path, cfg, prior, device)
        model.eval()
        emb_test = _embed_split(model, splits["test"], device=device, batch_size=2048)
        t_bars = np.zeros((len(datasets.theta), cfg.model.embed_dim), dtype=np.float32)
        for j, idx_map in enumerate(datasets.proc_indices):
            parts = [emb_test[pname][idxs] for pname, idxs in idx_map.items() if len(idxs) > 0]
            if parts:
                t_bars[j] = np.concatenate(parts, axis=0).mean(axis=0)
        pairs_res = evaluate_stage3_model(
            model=model,
            prior=prior,
            agg_test={"theta": datasets.theta, "T_bar": t_bars, "N": datasets.n_events},
            device=device,
            num_samples=1000,
        )
        summary["pairs"] = {
            "test_nll": pairs_res.test_nll,
            "rmae_mu": pairs_res.rmae_mu,
            "rmae_kl": pairs_res.rmae_kl,
            "acauc_2d": pairs_res.acauc_2d_hpd,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = args.output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / ("benchmarks_smoke.json" if args.smoke else "benchmarks.json")
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== Benchmark Comparison ===")
    for name, vals in summary.items():
        print(
            f"  {name:<14} | NLL={vals['test_nll']:.4f} | "
            f"RMAE(mu)={vals['rmae_mu']:.4f} | RMAE(kl)={vals['rmae_kl']:.4f} | "
            f"ACAUC_2D={vals['acauc_2d']:.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = list(summary.keys())
    nlls = [summary[k]["test_nll"] for k in names]
    rmaes = [0.5 * (summary[k]["rmae_mu"] + summary[k]["rmae_kl"]) for k in names]
    axes[0].bar(names, nlls, color="#3b6ea5")
    axes[0].axhline(prior.prior_nll, ls="--", color="gray", label=f"Prior ({prior.prior_nll:.2f})")
    axes[0].set_ylabel("Test NLL (nats)")
    axes[0].set_title("Test NLL by Method")
    axes[0].legend()
    axes[1].bar(names, rmaes, color="#d97724")
    axes[1].set_ylabel("Mean RMAE")
    axes[1].set_title("Mean RMAE by Method")
    fig.tight_layout()
    fig.savefig(figs_dir / ("benchmark_comparison_smoke.png" if args.smoke else "benchmark_comparison.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
