"""Sweep PAIRS embedding dimension l across seeds using the centralized stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pairs_bbtautau.config import (
    DEFAULT_INFERENCE_DIR,
    DataConfig,
    ModelConfig,
    PairsConfig,
    PriorConfig,
    TrainConfig,
    resolve_device,
    seed_everything,
)
from pairs_bbtautau.data import build_splits, load_or_build_process_cache
from pairs_bbtautau.metrics import evaluate_stage3_model
from pairs_bbtautau.models import PairsModel
from pairs_bbtautau.physics import UniformPrior2D
from pairs_bbtautau.stages import load_checkpoint, run_stage2, train_stage1, train_stage3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan PAIRS embedding dimension embed_dim (l)")
    p.add_argument("--embed-dims", type=int, nargs="+", default=[2, 4, 8, 12, 16, 32, 64])
    p.add_argument("--seeds", type=int, nargs="+", default=[123])
    p.add_argument("--data-dir", type=Path, default=DEFAULT_INFERENCE_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_INFERENCE_DIR / "cache")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_INFERENCE_DIR / "outputs" / "embed_scan")
    p.add_argument("--lumi", type=float, default=100_000.0)
    p.add_argument("--k-factor", type=float, default=1.0)
    p.add_argument("--stage1-epochs", type=int, default=30)
    p.add_argument("--stage1-batches", type=int, default=150)
    p.add_argument("--stage2-train-sets", type=int, default=20_000)
    p.add_argument("--stage2-val-sets", type=int, default=4_000)
    p.add_argument("--stage2-test-sets", type=int, default=2_000)
    p.add_argument("--stage3-epochs", type=int, default=80)
    p.add_argument("--eval-samples", type=int, default=1000)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--smoke", action="store_true", help="Fast lightweight smoke test.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.embed_dims = [4, 8]
        args.seeds = [123]
        args.stage1_epochs = 2
        args.stage1_batches = 4
        args.stage2_train_sets = 64
        args.stage2_val_sets = 32
        args.stage2_test_sets = 32
        args.stage3_epochs = 2
        args.eval_samples = 64

    device = resolve_device(args.device)
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        lumi_pb_inv=args.lumi,
        hh_k_factor=args.k_factor,
    )
    processes = load_or_build_process_cache(data_cfg)
    splits, feature_stats = build_splits(processes, data_cfg)
    prior_cfg = PriorConfig()
    prior = UniformPrior2D(prior_cfg)

    results_list: list[dict[str, float | int]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figs").mkdir(parents=True, exist_ok=True)

    for emb_dim in args.embed_dims:
        for seed in args.seeds:
            seed_everything(seed)
            run_dir = args.output_dir / f"l{emb_dim}_s{seed}"
            cfg = PairsConfig(
                prior=prior_cfg,
                data=data_cfg,
                model=ModelConfig(embed_dim=emb_dim),
                train=TrainConfig(
                    seed=seed,
                    device=args.device,
                    stage1_epochs=args.stage1_epochs,
                    stage1_batches_per_epoch=args.stage1_batches,
                    stage1_val_sets=128 if args.smoke else 8192,
                    stage2_n_train_sets=args.stage2_train_sets,
                    stage2_n_val_sets=args.stage2_val_sets,
                    stage2_n_test_sets=args.stage2_test_sets,
                    stage3_epochs=args.stage3_epochs,
                    eval_num_samples=args.eval_samples,
                ),
                output_dir=run_dir,
            )

            import torch
            sig_pool = splits["train_s1"].processes[splits["train_s1"].signal]
            model = PairsModel(
                jet_feat_dim=sig_pool.jet_features.shape[-1],
                evt_feat_dim=sig_pool.evt_features.shape[-1],
                theta_mean=torch.tensor(prior.mean, dtype=torch.float32),
                theta_std=torch.tensor(prior.std, dtype=torch.float32),
                cfg=cfg.model,
            )
            s1_hist = train_stage1(
                model=model,
                train_split=splits["train_s1"],
                val_split=splits["val"],
                prior=prior,
                feature_stats=feature_stats,
                cfg=cfg,
                device=device,
            )
            model_s1, _ = load_checkpoint(run_dir / "checkpoints" / "stage1_best.pt", cfg, prior, device)
            aggs = run_stage2(model=model_s1, splits=splits, prior=prior, cfg=cfg, device=device)
            s3_hist = train_stage3(
                model=model_s1,
                agg_train=aggs["train"],
                agg_val=aggs["val"],
                prior=prior,
                feature_stats=feature_stats,
                cfg=cfg,
                device=device,
            )
            model_s3, _ = load_checkpoint(run_dir / "checkpoints" / "stage3_best.pt", cfg, prior, device)
            eval_res = evaluate_stage3_model(
                model=model_s3,
                prior=prior,
                agg_test=aggs["test"],
                device=device,
                num_samples=args.eval_samples,
            )
            row = {
                "embed_dim": emb_dim,
                "seed": seed,
                "stage1_best_val_nll": float(min(s1_hist["val_nll"])),
                "stage3_best_val_nll": float(min(s3_hist["val_nll"])),
                "test_nll": float(eval_res.test_nll),
                "rmae_mu": float(eval_res.rmae_mu),
                "rmae_kl": float(eval_res.rmae_kl),
                "rmae_mean": float(eval_res.rmae_mean),
                "acauc_2d_hpd": float(eval_res.acauc_2d_hpd),
            }
            results_list.append(row)
            print(f"[l={emb_dim}, seed={seed}] Test NLL={row['test_nll']:.4f} | RMAE={row['rmae_mean']:.4f} | ACAUC={row['acauc_2d_hpd']:.4f}")

    (args.output_dir / "embed_dim_scan.json").write_text(json.dumps(results_list, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    dims = [r["embed_dim"] for r in results_list]
    axes[0].plot(dims, [r["test_nll"] for r in results_list], "o-")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel(r"Embedding dim $\ell$")
    axes[0].set_ylabel("Test NLL (nats)")
    axes[0].set_title(r"Test NLL vs $\ell$")

    axes[1].plot(dims, [r["rmae_mean"] for r in results_list], "s-", color="#d97724")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel(r"Embedding dim $\ell$")
    axes[1].set_ylabel("Mean RMAE")
    axes[1].set_title(r"Mean RMAE vs $\ell$")

    axes[2].plot(dims, [r["acauc_2d_hpd"] for r in results_list], "^-", color="#2ca02c")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel(r"Embedding dim $\ell$")
    axes[2].set_ylabel("ACAUC (2D HPD)")
    axes[2].set_title(r"Calibration vs $\ell$")

    fig.tight_layout()
    fig.savefig(args.output_dir / "figs" / "embed_dim_scan.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
