"""CLI entrypoint for joint (mu_HH, kappa_lambda) PAIRS inference on bb-tautau."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

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
from pairs_bbtautau.physics import StatOnlyNuisanceModel, UniformPrior2D, compute_hh_weights
from pairs_bbtautau.plotting import (
    plot_calibration_acauc,
    plot_loss_curves,
    plot_posterior_contours,
    plot_predicted_vs_true,
    plot_yields_vs_kl,
)
from pairs_bbtautau.stages import load_checkpoint, run_stage2, train_stage1, train_stage3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PAIRS joint (mu_HH, kappa_lambda) inference for bb tautau")
    p.add_argument(
        "--stage",
        choices=["all", "prepare", "stage1", "stage2", "stage3", "eval"],
        default="all",
        help="Pipeline stage(s) to execute.",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_INFERENCE_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_INFERENCE_DIR / "cache")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_INFERENCE_DIR / "outputs")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=123)

    # Physics & Prior bounds
    p.add_argument("--lumi", type=float, default=100_000.0, help="Integrated luminosity in pb^-1.")
    p.add_argument("--k-factor", type=float, default=1.0, help="NNLO/LO K-factor multiplier on HH signal.")
    p.add_argument("--max-jets", type=int, default=12)
    p.add_argument("--mu-min", type=float, default=0.0)
    p.add_argument("--mu-max", type=float, default=10.0)
    p.add_argument("--kl-min", type=float, default=-5.0)
    p.add_argument("--kl-max", type=float, default=10.0)

    # Model architecture
    p.add_argument("--embed-dim", type=int, default=12)
    p.add_argument("--flow-transforms", type=int, default=4)
    p.add_argument("--flow-hidden-dim", type=int, default=64)
    p.add_argument("--flow-bins", type=int, default=10)

    # Stage 1 hyperparameters
    p.add_argument("--stage1-epochs", type=int, default=30)
    p.add_argument("--stage1-batches", type=int, default=150)
    p.add_argument("--stage1-batch-size", type=int, default=512)
    p.add_argument("--stage1-val-sets", type=int, default=8192)
    p.add_argument("--stage1-lr", type=float, default=5e-4)
    p.add_argument("--stage1-patience", type=int, default=10)
    p.add_argument(
        "--stage1-mu-max",
        type=float,
        default=300.0,
        help="Upper mu_HH bound for Stage 1 N in {1,2} pretraining; pass <=0 to match target prior.",
    )

    # Stage 2 caching
    p.add_argument("--stage2-train-sets", type=int, default=20_000)
    p.add_argument("--stage2-val-sets", type=int, default=4_000)
    p.add_argument("--stage2-test-sets", type=int, default=2_000)

    # Stage 3 hyperparameters
    p.add_argument("--stage3-epochs", type=int, default=80)
    p.add_argument("--stage3-batch-size", type=int, default=256)
    p.add_argument("--stage3-lr", type=float, default=4e-4)
    p.add_argument("--stage3-patience", type=int, default=15)
    p.add_argument("--no-stage3-reinit-head", action="store_true")
    p.add_argument("--no-stage3-mc-aug", action="store_true")

    # Evaluation
    p.add_argument("--eval-samples", type=int, default=1000)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> PairsConfig:
    s1_mu_max = None if args.stage1_mu_max <= 0 else float(args.stage1_mu_max)
    cfg = PairsConfig(
        prior=PriorConfig(
            mu_min=args.mu_min,
            mu_max=args.mu_max,
            kl_min=args.kl_min,
            kl_max=args.kl_max,
        ),
        data=DataConfig(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            lumi_pb_inv=args.lumi,
            hh_k_factor=args.k_factor,
            max_jets=args.max_jets,
        ),
        model=ModelConfig(
            embed_dim=args.embed_dim,
            flow_num_transforms=args.flow_transforms,
            flow_hidden_dim=args.flow_hidden_dim,
            flow_num_bins=args.flow_bins,
            context_hidden_dim=args.flow_hidden_dim,
        ),
        train=TrainConfig(
            seed=args.seed,
            device=args.device,
            stage1_epochs=args.stage1_epochs,
            stage1_batches_per_epoch=args.stage1_batches,
            stage1_batch_size=args.stage1_batch_size,
            stage1_val_sets=args.stage1_val_sets,
            stage1_lr=args.stage1_lr,
            stage1_patience=args.stage1_patience,
            stage1_mu_max=s1_mu_max,
            stage2_n_train_sets=args.stage2_train_sets,
            stage2_n_val_sets=args.stage2_val_sets,
            stage2_n_test_sets=args.stage2_test_sets,
            stage3_epochs=args.stage3_epochs,
            stage3_batch_size=args.stage3_batch_size,
            stage3_lr=args.stage3_lr,
            stage3_patience=args.stage3_patience,
            stage3_reinit_head=not args.no_stage3_reinit_head,
            stage3_mc_aug=not args.no_stage3_mc_aug,
            eval_num_samples=args.eval_samples,
        ),
        output_dir=args.output_dir,
    )
    cfg.validate()
    return cfg


def print_process_summary(processes: dict, splits: dict) -> None:
    print("\n=== Dataset & Yield Summary (stat-only, 100 fb^-1 default) ===")
    total_bkg = 0.0
    for name, proc in processes.items():
        y_nom = float(proc.weights[:, 0].sum())
        if name != "HH":
            total_bkg += y_nom
        s1_n = splits["train_s1"].processes[name].num_events
        s3_n = splits["train_s3"].processes[name].num_events
        va_n = splits["val"].processes[name].num_events
        te_n = splits["test"].processes[name].num_events
        print(
            f"  {name:5s} | n_gen={proc.n_gen:7,d} | selected={proc.num_events:6,d} "
            f"(s1/s3/va/te={s1_n}/{s3_n}/{va_n}/{te_n}) | yield_nom={y_nom:9.3f}"
        )
    print(f"  Total SM background expected yield: {total_bkg:.3f}")
    hh = processes["HH"]
    for kl in (-5.0, -4.0, 0.0, 1.0, 2.0, 5.0, 10.0):
        y_kl = float(compute_hh_weights(hh.weights, 1.0, kl).sum())
        print(f"  HH expected yield (mu_HH=1, kl={kl:5.1f}): {y_kl:8.3f}")
    print()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    print(f"Using device: {device}")

    t0 = time.time()
    processes = load_or_build_process_cache(cfg.data)
    splits, feature_stats = build_splits(processes, cfg.data)
    print(f"Loaded & split selected events in {time.time() - t0:.2f}s")
    print_process_summary(processes, splits)

    prior = UniformPrior2D(cfg.prior)
    nuisance_model = StatOnlyNuisanceModel()
    fig_dir = cfg.output_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_yields_vs_kl(processes, cfg.prior, fig_dir)

    if args.stage == "prepare":
        return

    sample_sig = splits["train_s1"].processes[cfg.data.signal]
    jet_feat_dim = int(sample_sig.jet_features.shape[2])
    evt_feat_dim = int(sample_sig.evt_features.shape[1])

    hist_s1 = None
    hist_s3 = None
    ckpt_s1 = cfg.output_dir / "checkpoints" / "stage1_best.pt"
    ckpt_s3 = cfg.output_dir / "checkpoints" / "stage3_best.pt"
    cache_dir = cfg.output_dir / "stage2_cache"

    if args.stage in ("all", "stage1"):
        theta_mean = torch.tensor(prior.mean, dtype=torch.float32, device=device)
        theta_std = torch.tensor(prior.std, dtype=torch.float32, device=device)
        model = PairsModel(
            jet_feat_dim=jet_feat_dim,
            evt_feat_dim=evt_feat_dim,
            theta_mean=theta_mean,
            theta_std=theta_std,
            cfg=cfg.model,
        ).to(device)
        t_s1 = time.time()
        hist_s1 = train_stage1(
            model=model,
            train_split=splits["train_s1"],
            val_split=splits["val"],
            prior=prior,
            feature_stats=feature_stats,
            cfg=cfg,
            device=device,
            nuisance_model=nuisance_model,
        )
        print(f"Stage 1 finished in {time.time() - t_s1:.1f}s")
    else:
        ckpt_to_load = ckpt_s3 if args.stage == "eval" and ckpt_s3.exists() else ckpt_s1
        model, extra = load_checkpoint(ckpt_to_load, cfg, prior, device)
        if extra.get("stage") == 1:
            hist_s1 = extra.get("history")

    aggregates = None
    if args.stage in ("all", "stage2"):
        t_s2 = time.time()
        aggregates = run_stage2(
            model=model,
            splits=splits,
            prior=prior,
            cfg=cfg,
            device=device,
            nuisance_model=nuisance_model,
        )
        print(f"Stage 2 finished in {time.time() - t_s2:.1f}s")

    if args.stage in ("all", "stage3"):
        if aggregates is None:
            agg_train = torch.load(cache_dir / "agg_train.pt", map_location="cpu", weights_only=False)
            agg_val = torch.load(cache_dir / "agg_val.pt", map_location="cpu", weights_only=False)
        else:
            agg_train = aggregates["train"]
            agg_val = aggregates["val"]

        t_s3 = time.time()
        hist_s3 = train_stage3(
            model=model,
            agg_train=agg_train,
            agg_val=agg_val,
            prior=prior,
            feature_stats=feature_stats,
            cfg=cfg,
            device=device,
        )
        print(f"Stage 3 finished in {time.time() - t_s3:.1f}s")
    elif args.stage == "eval":
        model, extra = load_checkpoint(ckpt_s3, cfg, prior, device)
        hist_s3 = extra.get("history")
        if ckpt_s1.exists():
            _, extra_s1 = load_checkpoint(ckpt_s1, cfg, prior, device)
            hist_s1 = extra_s1.get("history")

    if args.stage in ("all", "eval"):
        if aggregates is None:
            agg_test = torch.load(cache_dir / "agg_test.pt", map_location="cpu", weights_only=False)
            bench_path = cache_dir / "agg_benchmarks.pt"
            bench_agg = torch.load(bench_path, map_location="cpu", weights_only=False) if bench_path.exists() else None
        else:
            agg_test = aggregates["test"]
            bench_agg = aggregates.get("benchmarks")

        t_ev = time.time()
        eval_res = evaluate_stage3_model(
            model=model,
            prior=prior,
            agg_test=agg_test,
            device=device,
            num_samples=cfg.train.eval_num_samples,
            num_acauc_levels=cfg.train.eval_acauc_levels,
        )
        print(f"\n=== Evaluation Results (elapsed {time.time() - t_ev:.1f}s) ===")
        print(f"  Test NLL (phys):   {eval_res.test_nll:.4f} (Prior baseline: {eval_res.prior_nll:.4f})")
        print(f"  MAE (mu_HH):       {eval_res.mae_mu:.4f}  | RMAE (mu_HH): {eval_res.rmae_mu:.4f}")
        print(f"  MAE (kappa_lam):   {eval_res.mae_kl:.4f}  | RMAE (kl):    {eval_res.rmae_kl:.4f}")
        print(f"  Mean RMAE:         {eval_res.rmae_mean:.4f}")
        print(f"  ACAUC (2D HPD):    {eval_res.acauc_2d_hpd:.4f}")
        print(f"  ACAUC (mu marg):   {eval_res.acauc_mu_marginal:.4f}")
        print(f"  ACAUC (kl marg):   {eval_res.acauc_kl_marginal:.4f}")

        metrics_summary = {
            "test_nll": eval_res.test_nll,
            "prior_nll": eval_res.prior_nll,
            "mae_mu": eval_res.mae_mu,
            "mae_kl": eval_res.mae_kl,
            "rmae_mu": eval_res.rmae_mu,
            "rmae_kl": eval_res.rmae_kl,
            "rmae_mean": eval_res.rmae_mean,
            "acauc_2d_hpd": eval_res.acauc_2d_hpd,
            "acauc_mu_marginal": eval_res.acauc_mu_marginal,
            "acauc_kl_marginal": eval_res.acauc_kl_marginal,
        }
        (cfg.output_dir / "metrics.json").write_text(json.dumps(metrics_summary, indent=2))

        plot_loss_curves(hist_s1, hist_s3, prior.prior_nll, fig_dir)
        plot_predicted_vs_true(eval_res, cfg.prior, fig_dir)
        plot_calibration_acauc(eval_res, fig_dir)
        if bench_agg is not None:
            plot_posterior_contours(model, bench_agg, cfg.prior, device, fig_dir)
        print(f"Saved diagnostic plots to {fig_dir}")


if __name__ == "__main__":
    main()
