"""Compare raw-event oracle log(p_s / p_b) against classifiers trained on frozen PAIRS embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pairs_bbtautau.config import DEFAULT_INFERENCE_DIR, DataConfig, ModelConfig, PairsConfig, PriorConfig, resolve_device, seed_everything
from pairs_bbtautau.data import build_splits, load_or_build_process_cache
from pairs_bbtautau.models import PerEventEmbedder, ResidualMLPBlock
from pairs_bbtautau.physics import ProcessSampler, UniformPrior2D
from pairs_bbtautau.stages import _embed_split, load_checkpoint


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    y = labels[order]
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y) + 1, dtype=np.float64)
    return float((ranks[y == 1].sum() - 0.5 * n_pos * (n_pos + 1.0)) / (n_pos * n_neg))


def _sample_balanced_raw(
    sampler: ProcessSampler,
    n_per_class: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    # Background: weighted by nominal background mixture
    bkg_probs = np.array([sampler.bkg_yields[b] for b in sampler.backgrounds], dtype=np.float64)
    bkg_probs /= bkg_probs.sum()
    bkg_counts = rng.multinomial(n_per_class, bkg_probs)

    jf_list, jm_list, ef_list = [], [], []
    for b, cnt in zip(sampler.backgrounds, bkg_counts):
        if cnt <= 0:
            continue
        idx = sampler._draw_process_indices(b, int(cnt), 1.0, rng)
        pool = sampler.split.processes[b]
        jf_list.append(pool.jet_features[idx])
        jm_list.append(pool.jet_mask[idx])
        ef_list.append(pool.evt_features[idx])

    sig_idx = sampler._draw_process_indices(sampler.signal, n_per_class, 1.0, rng)
    sig_pool = sampler.split.processes[sampler.signal]
    jf_list.append(sig_pool.jet_features[sig_idx])
    jm_list.append(sig_pool.jet_mask[sig_idx])
    ef_list.append(sig_pool.evt_features[sig_idx])

    jf = torch.from_numpy(np.concatenate(jf_list, axis=0))
    jm = torch.from_numpy(np.concatenate(jm_list, axis=0))
    ef = torch.from_numpy(np.concatenate(ef_list, axis=0))
    y = np.concatenate([np.zeros(n_per_class, dtype=np.float32), np.ones(n_per_class, dtype=np.float32)])
    return jf, jm, ef, y


def _sample_balanced_emb(
    sampler: ProcessSampler,
    embs: dict[str, np.ndarray],
    n_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    bkg_probs = np.array([sampler.bkg_yields[b] for b in sampler.backgrounds], dtype=np.float64)
    bkg_probs /= bkg_probs.sum()
    bkg_counts = rng.multinomial(n_per_class, bkg_probs)
    parts = []
    for b, cnt in zip(sampler.backgrounds, bkg_counts):
        if cnt > 0:
            idx = sampler._draw_process_indices(b, int(cnt), 1.0, rng)
            parts.append(embs[b][idx])
    sig_idx = sampler._draw_process_indices(sampler.signal, n_per_class, 1.0, rng)
    parts.append(embs[sampler.signal][sig_idx])
    x = np.concatenate(parts, axis=0)
    y = np.concatenate([np.zeros(n_per_class, dtype=np.float32), np.ones(n_per_class, dtype=np.float32)])
    return x, y


def main() -> None:
    p = argparse.ArgumentParser(description="Compare raw oracle ln(p_s/p_b) vs PAIRS-embedding classifier")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_INFERENCE_DIR / "outputs/cmp_m2_iw/checkpoints/stage1_best.pt")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_INFERENCE_DIR / "outputs/cmp_m2_iw")
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    cfg = PairsConfig(data=DataConfig(hh_k_factor=3.0), model=ModelConfig(embed_dim=args.embed_dim))
    prior = UniformPrior2D(cfg.prior)

    processes = load_or_build_process_cache(cfg.data)
    splits, _ = build_splits(processes, cfg.data)
    train_sampler = ProcessSampler(splits["train_s3"], hh_k_factor=cfg.data.hh_k_factor)
    val_sampler = ProcessSampler(splits["val"], hh_k_factor=cfg.data.hh_k_factor)
    test_sampler = ProcessSampler(splits["test"], hh_k_factor=cfg.data.hh_k_factor)

    # 1. Load frozen PAIRS Stage 1 embedder and embed train_s3 / val / test
    pairs_model, _ = load_checkpoint(args.checkpoint, cfg, prior, device)
    pairs_model.eval()
    embs = {k: _embed_split(pairs_model, splits[k], batch_size=2048, device=device) for k in ("train_s3", "val", "test")}

    # 2. Train raw-event oracle classifier d_raw(x) -> logit = ln(p_s / p_b)
    sig_pool = splits["train_s3"].processes["HH"]
    oracle_cfg = ModelConfig(embed_dim=1)
    oracle = PerEventEmbedder(sig_pool.jet_features.shape[-1], sig_pool.evt_features.shape[-1], oracle_cfg).to(device)
    opt_raw = torch.optim.AdamW(oracle.parameters(), lr=5e-4, weight_decay=5e-4)
    rng = np.random.default_rng(args.seed)

    val_jf, val_jm, val_ef, val_y_np = _sample_balanced_raw(val_sampler, 4096, rng)
    val_y = torch.from_numpy(val_y_np).to(device)
    best_val, best_oracle = float("inf"), None

    for ep in range(1, args.epochs + 1):
        oracle.train()
        for _ in range(60):
            jf, jm, ef, y_np = _sample_balanced_raw(train_sampler, 256, rng)
            logits = oracle(jf.to(device), jm.to(device), ef.to(device)).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(y_np).to(device))
            opt_raw.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
            opt_raw.step()
        oracle.eval()
        with torch.no_grad():
            v_logits = oracle(val_jf.to(device), val_jm.to(device), val_ef.to(device)).squeeze(-1)
            v_loss = float(F.binary_cross_entropy_with_logits(v_logits, val_y).item())
        if v_loss < best_val:
            best_val = v_loss
            best_oracle = {k: v.detach().cpu().clone() for k, v in oracle.state_dict().items()}
    oracle.load_state_dict(best_oracle)
    oracle.eval()

    # 3. Train linear and MLP classifiers purely on frozen PAIRS embeddings t_omega(x)
    d_emb = args.embed_dim
    lin_clf = nn.Linear(d_emb, 1).to(device)
    mlp_clf = nn.Sequential(
        nn.Linear(d_emb, 64), nn.LayerNorm(64), nn.SiLU(),
        ResidualMLPBlock(64, dropout=0.1), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64, 1)
    ).to(device)

    for clf, epochs_c in [(lin_clf, 200), (mlp_clf, 200)]:
        opt_c = torch.optim.AdamW(clf.parameters(), lr=3e-3, weight_decay=1e-4)
        vx, vy = _sample_balanced_emb(val_sampler, embs["val"], 4096, rng)
        vx_t, vy_t = torch.from_numpy(vx).to(device), torch.from_numpy(vy).to(device)
        best_c, best_st = float("inf"), None
        for _ in range(epochs_c):
            clf.train()
            tx, ty = _sample_balanced_emb(train_sampler, embs["train_s3"], 1024, rng)
            loss = F.binary_cross_entropy_with_logits(clf(torch.from_numpy(tx).to(device)).squeeze(-1), torch.from_numpy(ty).to(device))
            opt_c.zero_grad(set_to_none=True)
            loss.backward()
            opt_c.step()
            clf.eval()
            with torch.no_grad():
                vl = float(F.binary_cross_entropy_with_logits(clf(vx_t).squeeze(-1), vy_t).item())
            if vl < best_c:
                best_c = vl
                best_st = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
        clf.load_state_dict(best_st)
        clf.eval()

    # 4. Evaluate all three on identical held-out test sample
    test_jf, test_jm, test_ef, test_y_np = _sample_balanced_raw(test_sampler, 5000, np.random.default_rng(args.seed + 99))
    with torch.no_grad():
        llr_raw = oracle(test_jf.to(device), test_jm.to(device), test_ef.to(device)).squeeze(-1).cpu().numpy()
        test_emb = pairs_model.embedder(test_jf.to(device), test_jm.to(device), test_ef.to(device))
        llr_lin = lin_clf(test_emb).squeeze(-1).cpu().numpy()
        llr_mlp = mlp_clf(test_emb).squeeze(-1).cpu().numpy()

    metrics = {
        "auc_raw_oracle": _roc_auc(llr_raw, test_y_np),
        "auc_emb_linear": _roc_auc(llr_lin, test_y_np),
        "auc_emb_mlp": _roc_auc(llr_mlp, test_y_np),
        "pearson_raw_vs_lin": float(np.corrcoef(llr_raw, llr_lin)[0, 1]),
        "pearson_raw_vs_mlp": float(np.corrcoef(llr_raw, llr_mlp)[0, 1]),
    }
    print(json.dumps(metrics, indent=2))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    bkg_m = test_y_np == 0
    sig_m = test_y_np == 1
    for ax, pred, title, r in [
        (axes[0], llr_lin, "Linear on PAIRS $t_\\omega(x)$", metrics["pearson_raw_vs_lin"]),
        (axes[1], llr_mlp, "MLP on PAIRS $t_\\omega(x)$", metrics["pearson_raw_vs_mlp"]),
    ]:
        ax.scatter(llr_raw[bkg_m], pred[bkg_m], s=5, alpha=0.25, label="Background", color="#3b6ea5")
        ax.scatter(llr_raw[sig_m], pred[sig_m], s=5, alpha=0.25, label="HH (SM)", color="#d95f02")
        lims = [min(llr_raw.min(), pred.min()), max(llr_raw.max(), pred.max())]
        ax.plot(lims, lims, "k--", lw=1.2, label="$y = x$")
        ax.set_xlabel(r"Raw-Event Oracle $\ln(p_s / p_b)$")
        ax.set_ylabel(r"Embedding Classifier $\ln(p_s / p_b)$")
        ax.set_title(f"{title} ($r = {r:.3f}$)")
        ax.legend(markerscale=3)
    fig.suptitle(
        f"Test AUC: Raw Oracle = {metrics['auc_raw_oracle']:.3f} | "
        f"Emb Linear = {metrics['auc_emb_linear']:.3f} | Emb MLP = {metrics['auc_emb_mlp']:.3f}"
    )
    fig.tight_layout()
    figs_dir = args.output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    out_fig = figs_dir / "llr_oracle_vs_embedding.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
