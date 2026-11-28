"""PAIRS Three-Stage Pipeline: Stage 1 (N in {1,2}), Stage 2 (Caching), Stage 3 (Dataset Head)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .config import PairsConfig, PriorConfig
from .data import FeatureStats, SplitDataset
from .models import PairsModel
from .physics import NuisanceModel, ProcessSampler, StatOnlyNuisanceModel, UniformPrior2D


def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_frac: float,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(round(warmup_frac * total_steps)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    path: Path,
    model: PairsModel,
    cfg: PairsConfig,
    feature_stats: FeatureStats,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "n_encoding_mode": model.n_encoding_mode,
        "stage1_n_max": model.stage1_n_max,
        "jet_feat_dim": model.jet_feat_dim,
        "evt_feat_dim": model.evt_feat_dim,
        "feature_stats": feature_stats.to_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    cfg.save_json(path.parent / "config.json")


def load_checkpoint(
    path: Path,
    cfg: PairsConfig,
    prior: UniformPrior2D,
    device: torch.device,
) -> tuple[PairsModel, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    theta_mean = torch.tensor(prior.mean, dtype=torch.float32, device=device)
    theta_std = torch.tensor(prior.std, dtype=torch.float32, device=device)
    model = PairsModel(
        jet_feat_dim=int(ckpt["jet_feat_dim"]),
        evt_feat_dim=int(ckpt["evt_feat_dim"]),
        theta_mean=theta_mean,
        theta_std=theta_std,
        cfg=cfg.model,
    ).to(device)
    model.n_encoding_mode = ckpt["n_encoding_mode"]
    model.stage1_n_max = float(ckpt["stage1_n_max"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt.get("extra", {})


@torch.no_grad()
def _eval_stage1_val(
    model: PairsModel,
    val_theta: torch.Tensor,
    val_jets: torch.Tensor,
    val_jmask: torch.Tensor,
    val_evts: torch.Tensor,
    val_emask: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> float:
    model.eval()
    total_nll = 0.0
    n_total = int(val_theta.shape[0])
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        th = val_theta[start:end].to(device)
        jf = val_jets[start:end].to(device)
        jm = val_jmask[start:end].to(device)
        ef = val_evts[start:end].to(device)
        em = val_emask[start:end].to(device)
        nll = model.nll_from_sets(th, jf, jm, ef, em)
        total_nll += float(nll.sum().item())
    return total_nll / float(n_total)


def train_stage1(
    model: PairsModel,
    train_split: SplitDataset,
    val_split: SplitDataset,
    prior: UniformPrior2D,
    feature_stats: FeatureStats,
    cfg: PairsConfig,
    device: torch.device,
    nuisance_model: NuisanceModel | None = None,
) -> dict[str, Any]:
    """Stage 1: Jointly pretrain per-event embedder t_omega and flow head at N in {1, 2}."""
    nuis = nuisance_model or StatOnlyNuisanceModel()
    train_sampler = ProcessSampler(train_split, nuis, hh_k_factor=cfg.data.hh_k_factor)
    val_sampler = ProcessSampler(val_split, nuis, hh_k_factor=cfg.data.hh_k_factor)

    model.to(device)
    model.configure_stage1(n_max=2.0)

    tcfg = cfg.train
    rng_train = np.random.default_rng(tcfg.seed + 101)
    rng_val = np.random.default_rng(tcfg.seed + 202)
    mu_max_s1 = tcfg.stage1_mu_max
    prior_s1 = prior
    if mu_max_s1 is not None:
        prior_s1 = UniformPrior2D(
            PriorConfig(
                mu_min=prior.cfg.mu_min,
                mu_max=float(mu_max_s1),
                kl_min=prior.cfg.kl_min,
                kl_max=prior.cfg.kl_max,
            )
        )
    model.set_theta_stats(
        torch.tensor(prior_s1.mean, dtype=torch.float32),
        torch.tensor(prior_s1.std, dtype=torch.float32),
    )

    # Pre-generate fixed validation sets at N in {1, 2} for deterministic early stopping
    val_n_sets = tcfg.stage1_val_sets
    val_theta_np = prior_s1.sample(val_n_sets, rng_val)
    val_sizes = rng_val.integers(1, 3, size=val_n_sets)
    val_psi = nuis.sample_prior(val_n_sets, rng_val)
    val_jf, val_jm, val_ef = val_sampler.sample_small_sets_batch(
        val_theta_np, val_sizes, rng_val, psi=val_psi
    )
    val_emask = torch.arange(2)[None, :] < torch.from_numpy(val_sizes)[:, None]
    val_theta = torch.from_numpy(val_theta_np)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.stage1_lr,
        weight_decay=tcfg.stage1_weight_decay,
    )
    total_steps = tcfg.stage1_epochs * tcfg.stage1_batches_per_epoch
    scheduler = _build_warmup_cosine_scheduler(optimizer, total_steps, tcfg.stage1_warmup_frac)

    history: dict[str, Any] = {
        "epoch": [],
        "train_nll": [],
        "val_nll": [],
        "lr": [],
        "prior_nll": float(prior_s1.prior_nll),
    }
    best_val_nll = float("inf")
    best_epoch = -1
    no_improve = 0
    ckpt_path = cfg.output_dir / "checkpoints" / "stage1_best.pt"

    for epoch in range(1, tcfg.stage1_epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(
            range(tcfg.stage1_batches_per_epoch),
            desc=f"Stage 1 Epoch {epoch:02d}/{tcfg.stage1_epochs}",
            leave=False,
        )
        for _ in pbar:
            theta_np = prior_s1.sample(tcfg.stage1_batch_size, rng_train)
            sizes = rng_train.integers(1, 3, size=tcfg.stage1_batch_size)
            psi = nuis.sample_prior(tcfg.stage1_batch_size, rng_train)
            jf, jm, ef = train_sampler.sample_small_sets_batch(theta_np, sizes, rng_train, psi=psi)
            em = torch.arange(2)[None, :] < torch.from_numpy(sizes)[:, None]

            th_t = torch.from_numpy(theta_np).to(device)
            jf = jf.to(device)
            jm = jm.to(device)
            ef = ef.to(device)
            em = em.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model.nll_from_sets(th_t, jf, jm, ef, em).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tcfg.stage1_grad_clip)
            optimizer.step()
            scheduler.step()

            loss_val = float(loss.item())
            running_loss += loss_val
            pbar.set_postfix(nll=f"{loss_val:.4f}")

        train_nll = running_loss / float(tcfg.stage1_batches_per_epoch)
        val_nll = _eval_stage1_val(
            model, val_theta, val_jf, val_jm, val_ef, val_emask, device=device
        )
        cur_lr = float(optimizer.param_groups[0]["lr"])

        history["epoch"].append(float(epoch))
        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        history["lr"].append(cur_lr)

        print(
            f"[Stage 1] Epoch {epoch:02d}/{tcfg.stage1_epochs:02d} | "
            f"train_nll={train_nll:.4f} | val_nll={val_nll:.4f} "
            f"(prior={prior_s1.prior_nll:.4f}) | lr={cur_lr:.2e}"
        )

        if val_nll < best_val_nll - 1e-5:
            best_val_nll = val_nll
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                ckpt_path,
                model,
                cfg,
                feature_stats,
                extra={"stage": 1, "best_epoch": best_epoch, "best_val_nll": best_val_nll, "history": history},
            )
        else:
            no_improve += 1
            if no_improve >= tcfg.stage1_patience:
                print(
                    f"[Stage 1] Early stopping at epoch {epoch} "
                    f"(best epoch {best_epoch}, val_nll={best_val_nll:.4f})"
                )
                break

    best_model, _ = load_checkpoint(ckpt_path, cfg, prior_s1, device)
    model.load_state_dict(best_model.state_dict())
    return history


@torch.no_grad()
def _embed_split(
    model: PairsModel,
    split: SplitDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Compute frozen per-event embeddings e_i = t_omega(x_i) once per MC event."""
    model.eval()
    embeddings: dict[str, np.ndarray] = {}
    for proc_name in split.process_names():
        pool = split.processes[proc_name]
        n = pool.num_events
        out = np.empty((n, model.cfg.embed_dim), dtype=np.float32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            jf = torch.from_numpy(pool.jet_features[start:end]).to(device)
            jm = torch.from_numpy(pool.jet_mask[start:end]).to(device)
            ef = torch.from_numpy(pool.evt_features[start:end]).to(device)
            emb = model.embedder(jf, jm, ef)
            out[start:end] = emb.cpu().numpy()
        embeddings[proc_name] = out
    return embeddings


def _estimate_finite_mc_covariance(
    train_s3_split: SplitDataset,
    val_split: SplitDataset,
    embeddings_train_s3: dict[str, np.ndarray],
) -> np.ndarray:
    """Estimate finite-MC pool shift covariance purely from Stage 3 training embeddings.

    Accounts for independent MC pool statistical fluctuations between train_s3 and
    held-out splits (val/test) without touching held-out embeddings.
    """
    embed_dim = int(next(iter(embeddings_train_s3.values())).shape[1])
    bkg_yields = {
        b: float(train_s3_split.processes[b].weights[:, 0].sum())
        for b in train_s3_split.backgrounds
    }
    tot_bkg = max(sum(bkg_yields.values()), 1e-12)
    cov_mc = np.zeros((embed_dim, embed_dim), dtype=np.float64)

    for b in train_s3_split.backgrounds:
        f_b = bkg_yields[b] / tot_bkg
        e_b = embeddings_train_s3[b].astype(np.float64)
        if e_b.shape[0] < 2:
            continue
        cov_b = np.cov(e_b, rowvar=False)
        m_tr = float(train_s3_split.processes[b].num_events)
        m_val = float(val_split.processes[b].num_events)
        cov_mc += (f_b * f_b) * cov_b * (1.0 / m_tr + 1.0 / m_val)

    return cov_mc.astype(np.float32)


def run_stage2(
    model: PairsModel,
    splits: dict[str, SplitDataset],
    prior: UniformPrior2D,
    cfg: PairsConfig,
    device: torch.device,
    nuisance_model: NuisanceModel | None = None,
) -> dict[str, dict[str, Any]]:
    """Stage 2: Freeze embedder, compute per-event embeddings, and cache dataset aggregates.

    Uses `train_s3` (disjoint from Stage 1's `train_s1`) to generate `agg_train` so that
    Stage 3 receives strictly out-of-sample embeddings matching `val` and `test`.
    """
    nuis = nuisance_model or StatOnlyNuisanceModel()
    model.eval()
    for p in model.embedder.parameters():
        p.requires_grad_(False)

    tcfg = cfg.train
    cache_dir = cfg.output_dir / "stage2_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_embeddings: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("train_s1", "train_s3", "val", "test"):
        if split_name not in splits:
            continue
        all_embeddings[split_name] = _embed_split(
            model,
            splits[split_name],
            batch_size=tcfg.stage2_embed_batch_size,
            device=device,
        )
    torch.save(all_embeddings, cache_dir / "event_embeddings.pt")

    cov_mc = _estimate_finite_mc_covariance(
        splits["train_s3"],
        splits["val"],
        all_embeddings["train_s3"],
    )

    split_map = {
        "train": ("train_s3", tcfg.stage2_n_train_sets),
        "val": ("val", tcfg.stage2_n_val_sets),
        "test": ("test", tcfg.stage2_n_test_sets),
    }
    aggregates: dict[str, dict[str, Any]] = {}

    for idx, (agg_key, (pool_key, n_sets)) in enumerate(split_map.items()):
        rng = np.random.default_rng(tcfg.seed + 1000 + idx)
        sampler = ProcessSampler(splits[pool_key], nuis, hh_k_factor=cfg.data.hh_k_factor)
        agg = sampler.sample_aggregated_datasets(
            embeddings=all_embeddings[pool_key],
            prior=prior,
            n_datasets=n_sets,
            rng=rng,
        )
        if agg_key == "train":
            agg["cov_mc"] = cov_mc
        aggregates[agg_key] = agg
        torch.save(agg, cache_dir / f"agg_{agg_key}.pt")
        print(
            f"[Stage 2] Cached {agg_key} (from {pool_key}) aggregates: n_sets={agg['N'].shape[0]}, "
            f"N range=[{agg['N'].min()}, {agg['N'].max()}], mean N={agg['N'].mean():.1f}, "
            f"mean N_HH={agg['N_hh'].mean():.1f}"
        )

    benchmark_points = np.array(
        [
            [1.0, 1.0],   # Standard Model
            [5.0, -4.0],  # Enhanced negative trilinear
            [3.0, 1.0],   # Scaled SM kinematics
            [2.0, 5.0],   # Soft threshold enhanced positive trilinear
            [6.0, 2.0],   # High mu_HH at destructive interference
            [0.2, 2.0],   # Near background-only (low identifiability of kl)
        ],
        dtype=np.float32,
    )
    in_prior = prior.in_bounds(benchmark_points)
    benchmark_points = benchmark_points[in_prior]
    if len(benchmark_points) > 0:
        test_sampler = ProcessSampler(splits["test"], nuis, hh_k_factor=cfg.data.hh_k_factor)
        rng_bench = np.random.default_rng(tcfg.seed + 9999)
        bench_agg = test_sampler.sample_aggregated_datasets(
            embeddings=all_embeddings["test"],
            prior=prior,
            n_datasets=len(benchmark_points),
            rng=rng_bench,
            fixed_thetas=benchmark_points,
        )
        aggregates["benchmarks"] = bench_agg
        torch.save(bench_agg, cache_dir / "agg_benchmarks.pt")

    return aggregates


@torch.no_grad()
def _eval_stage3_val(
    model: PairsModel,
    val_theta: torch.Tensor,
    val_tbar: torch.Tensor,
    val_counts: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> float:
    model.eval()
    total_nll = 0.0
    n_total = int(val_theta.shape[0])
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        th = val_theta[start:end].to(device)
        tb = val_tbar[start:end].to(device)
        nc = val_counts[start:end].to(device)
        total_nll += float(model.nll_from_aggregates(th, tb, nc).sum().item())
    return total_nll / float(n_total)


def train_stage3(
    model: PairsModel,
    agg_train: dict[str, Any],
    agg_val: dict[str, Any],
    prior: UniformPrior2D,
    feature_stats: FeatureStats,
    cfg: PairsConfig,
    device: torch.device,
) -> dict[str, list[float]]:
    """Stage 3: Freeze t_omega and train dataset-wide conditional NF head on cached aggregates."""
    tcfg = cfg.train
    model.to(device)
    model.embedder.eval()
    for p in model.embedder.parameters():
        p.requires_grad_(False)

    if tcfg.stage3_reinit_head:
        model.reset_flow_head()

    # Restore exact target prior standardization for Stage 3 inference
    model.set_theta_stats(
        torch.tensor(prior.mean, dtype=torch.float32),
        torch.tensor(prior.std, dtype=torch.float32),
    )

    tbar_train_np = agg_train["T_bar"]
    cov_mc = agg_train.get("cov_mc")
    use_mc_aug = bool(tcfg.stage3_mc_aug and cov_mc is not None)

    tbar_mean = torch.from_numpy(tbar_train_np.mean(axis=0)).float()
    tbar_var = tbar_train_np.var(axis=0).astype(np.float64)
    L_mc_t = None
    if use_mc_aug:
        cov_mc64 = np.asarray(cov_mc, dtype=np.float64)
        tbar_var = tbar_var + np.diag(cov_mc64)
        embed_dim = cov_mc64.shape[0]
        L_mc = np.linalg.cholesky(cov_mc64 + 1e-10 * np.eye(embed_dim))
        L_mc_t = torch.from_numpy(L_mc).float().to(device)

    tbar_std = torch.from_numpy(np.sqrt(np.maximum(tbar_var, 1e-10))).float()

    logn_train_np = np.log(agg_train["N"].astype(np.float64) + 1.0)
    logn_mean = torch.tensor([logn_train_np.mean()], dtype=torch.float32)
    logn_std = torch.tensor([max(float(logn_train_np.std()), 1e-6)], dtype=torch.float32)
    model.configure_stage3(tbar_mean, tbar_std, logn_mean, logn_std)

    train_theta = torch.from_numpy(agg_train["theta"]).float()
    train_tbar = torch.from_numpy(agg_train["T_bar"]).float()
    train_counts = torch.from_numpy(agg_train["N"]).long()

    val_theta = torch.from_numpy(agg_val["theta"]).float()
    val_tbar = torch.from_numpy(agg_val["T_bar"]).float()
    val_counts = torch.from_numpy(agg_val["N"]).long()

    n_train = int(train_theta.shape[0])
    steps_per_epoch = max(1, math.ceil(n_train / tcfg.stage3_batch_size))
    total_steps = tcfg.stage3_epochs * steps_per_epoch

    optimizer = torch.optim.AdamW(
        model.flow_head.parameters(),
        lr=tcfg.stage3_lr,
        weight_decay=tcfg.stage3_weight_decay,
    )
    scheduler = _build_warmup_cosine_scheduler(optimizer, total_steps, tcfg.stage3_warmup_frac)

    rng = np.random.default_rng(tcfg.seed + 303)
    history: dict[str, list[float]] = {"epoch": [], "train_nll": [], "val_nll": [], "lr": []}
    best_val_nll = float("inf")
    best_epoch = -1
    no_improve = 0
    ckpt_path = cfg.output_dir / "checkpoints" / "stage3_best.pt"

    for epoch in range(1, tcfg.stage3_epochs + 1):
        model.flow_head.train()
        perm = torch.from_numpy(rng.permutation(n_train))
        running_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, tcfg.stage3_batch_size):
            idx = perm[start : start + tcfg.stage3_batch_size]
            th = train_theta[idx].to(device)
            tb = train_tbar[idx].to(device)
            if L_mc_t is not None:
                eps = torch.randn(tb.shape[0], tb.shape[1], device=device, dtype=tb.dtype) @ L_mc_t.T
                tb = tb + eps
            nc = train_counts[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model.nll_from_aggregates(th, tb, nc).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.flow_head.parameters(), max_norm=tcfg.stage3_grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            n_batches += 1

        train_nll = running_loss / float(max(1, n_batches))
        val_nll = _eval_stage3_val(model, val_theta, val_tbar, val_counts, device=device)
        cur_lr = float(optimizer.param_groups[0]["lr"])

        history["epoch"].append(float(epoch))
        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        history["lr"].append(cur_lr)

        if epoch == 1 or epoch % 5 == 0 or epoch == tcfg.stage3_epochs:
            print(
                f"[Stage 3] Epoch {epoch:02d}/{tcfg.stage3_epochs:02d} | "
                f"train_nll={train_nll:.4f} | val_nll={val_nll:.4f} | "
                f"(prior={prior.prior_nll:.4f}) | lr={cur_lr:.2e}"
            )

        if val_nll < best_val_nll - 1e-5:
            best_val_nll = val_nll
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                ckpt_path,
                model,
                cfg,
                feature_stats,
                extra={"stage": 3, "best_epoch": best_epoch, "best_val_nll": best_val_nll, "history": history},
            )
        else:
            no_improve += 1
            if no_improve >= tcfg.stage3_patience:
                print(
                    f"[Stage 3] Early stopping at epoch {epoch} "
                    f"(best epoch {best_epoch}, val_nll={best_val_nll:.4f})"
                )
                break

    best_model, _ = load_checkpoint(ckpt_path, cfg, prior, device)
    model.load_state_dict(best_model.state_dict())
    return history
