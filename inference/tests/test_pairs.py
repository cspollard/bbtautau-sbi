"""Unit and integration tests for PAIRS bb-tautau inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pairs_bbtautau.config import DataConfig, ModelConfig, PairsConfig, PriorConfig, TrainConfig
from pairs_bbtautau.data import ProcessData, build_splits, parse_and_select_process
from pairs_bbtautau.metrics import (
    compute_empirical_coverage_1d,
    compute_empirical_coverage_2d_hpd,
    evaluate_stage3_model,
)
from pairs_bbtautau.models import ConditionalSplineFlow2D, PairsModel, _rational_quadratic_spline
from pairs_bbtautau.physics import ProcessSampler, UniformPrior2D, compute_hh_weights
from pairs_bbtautau.stages import run_stage2, train_stage1, train_stage3


def test_quadratic_kappa_lambda_reweighting_exact_points():
    rng = np.random.default_rng(7)
    # Physical matrix-element-squared parameterization: w(kl) = |a + b * kl|^2 >= 0
    a = rng.normal(0.0, 0.05, size=50) + 1j * rng.normal(0.0, 0.05, size=50)
    b = rng.normal(0.0, 0.02, size=50) + 1j * rng.normal(0.0, 0.02, size=50)
    w_kl0 = np.abs(a) ** 2
    w_nom = np.abs(a + b) ** 2
    w_kl2 = np.abs(a + 2.0 * b) ** 2
    w = np.stack([w_nom, w_kl0, w_kl2], axis=-1)

    # Exact values at kappa_lambda in {0, 1, 2} for mu_HH = 1
    np.testing.assert_allclose(compute_hh_weights(w, 1.0, 0.0), w[:, 1], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(compute_hh_weights(w, 1.0, 1.0), w[:, 0], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(compute_hh_weights(w, 1.0, 2.0), w[:, 2], rtol=1e-12, atol=1e-15)

    # Exact quadratic recovery and mu_HH linearity across arbitrary kappa_lambda
    for kl in (-5.0, -2.5, 0.4, 3.7, 5.3, 10.0):
        expected = 2.5 * (np.abs(a + kl * b) ** 2)
        out = compute_hh_weights(w, 2.5, kl, clamp_negative=False)
        assert np.all(out >= -1e-16)
        np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(compute_hh_weights(w, 2.5, kl, clamp_negative=True), expected, rtol=1e-10, atol=1e-14)


def test_csv_selection_and_split_yield_preservation(tmp_path: Path):
    # Create tiny synthetic CSV with 3 events: 1 passes selection, 1 fails m_bb, 1 fails 2b+2tau
    csv_path = tmp_path / "toy.csv"
    rows = [
        "eventnumber,btagged,ctagged,tautagged,px,py,pz,weight_nom",
        # Event 0: 2 b-jets, 2 taus, m_bb ~ 113, m_tt ~ 89, m_bbtt ~ 300 -> PASSES
        "0,1,1,0,60.0,0.0,0.0,0.5",
        "0,1,1,0,-40.0,30.0,20.0,0.5",
        "0,0,0,1,0.0,50.0,10.0,0.5",
        "0,0,0,1,0.0,-45.0,-10.0,0.5",
        # Event 2 (note event 1 skipped as 0-jet): 2 b-jets, 2 taus, m_bb too small (<75) -> FAILS
        "2,1,1,0,50.0,0.0,0.0,0.5",
        "2,1,1,0,50.0,10.0,0.0,0.5",
        "2,0,0,1,-50.0,50.0,0.0,0.5",
        "2,0,0,1,-50.0,-50.0,0.0,0.5",
        # Event 3: only 1 b-jet -> FAILS
        "3,1,1,0,50.0,0.0,0.0,0.5",
        "3,0,0,1,-50.0,50.0,0.0,0.5",
        "3,0,0,1,-50.0,-50.0,0.0,0.5",
    ]
    csv_path.write_text("\n".join(rows) + "\n")

    proc = parse_and_select_process(csv_path, name="toy", max_jets=8, lumi_pb_inv=100.0)
    assert proc.n_gen == 4  # event numbers 0..3
    assert proc.n_csv_events == 3
    assert proc.num_events == 1
    np.testing.assert_allclose(proc.weights[0, 0], 100.0 * 0.5 / 4.0)

    # Check yield preservation across splits
    rng = np.random.default_rng(11)
    n_syn = 100
    fake_proc = ProcessData(
        name="HH",
        raw_jets=np.tile(proc.raw_jets, (n_syn, 1, 1)),
        jet_mask=np.tile(proc.jet_mask, (n_syn, 1)),
        evt_invariants=np.tile(proc.evt_invariants, (n_syn, 1)),
        weights=rng.uniform(0.1, 1.0, size=(n_syn, 3)),
        n_gen=n_syn,
        n_csv_events=n_syn,
    )
    fake_bkg = ProcessData(
        name="top",
        raw_jets=np.tile(proc.raw_jets, (n_syn, 1, 1)),
        jet_mask=np.tile(proc.jet_mask, (n_syn, 1)),
        evt_invariants=np.tile(proc.evt_invariants, (n_syn, 1)),
        weights=np.full((n_syn, 1), 0.25),
        n_gen=n_syn,
        n_csv_events=n_syn,
    )
    cfg = DataConfig(
        backgrounds=("top",),
        signal="HH",
        stage1_train_frac=0.35,
        stage3_train_frac=0.25,
        val_frac=0.2,
        test_frac=0.2,
    )
    splits, _ = build_splits({"top": fake_bkg, "HH": fake_proc}, cfg)

    expected_bkg_yield = fake_bkg.weights[:, 0].sum()
    for s_name in ("train_s1", "train_s3", "val", "test"):
        np.testing.assert_allclose(
            splits[s_name].processes["top"].weights[:, 0].sum(),
            expected_bkg_yield,
            rtol=1e-6,
        )


def test_rqs_invertibility_and_jacobian_accuracy():
    torch.manual_seed(0)
    b, k = 32, 8
    x = torch.cat(
        [
            torch.linspace(-3.3, 3.3, b - 4),
            torch.tensor([-4.2, -3.5, 3.5, 4.8]),
        ]
    )
    w = torch.randn(b, k)
    h = torch.randn(b, k)
    d = torch.randn(b, k - 1)

    y, ld_fwd = _rational_quadratic_spline(x, w, h, d, inverse=False, tail_bound=3.5)
    x_rec, ld_inv = _rational_quadratic_spline(y, w, h, d, inverse=True, tail_bound=3.5)

    torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(ld_fwd + ld_inv, torch.zeros_like(ld_fwd), atol=1e-4, rtol=1e-4)

    # Check finite-difference Jacobian determinant on ConditionalSplineFlow2D
    cfg = ModelConfig(embed_dim=8, flow_num_transforms=4, flow_hidden_dim=32, flow_num_bins=8, context_hidden_dim=32)
    flow = ConditionalSplineFlow2D(cond_in_dim=9, cfg=cfg)
    flow.eval()
    cond = torch.randn(6, 9)
    theta = torch.randn(6, 2) * 1.2

    ctx = flow.context_net(cond)
    z, logabsdet = theta.clone(), torch.zeros(6)
    for t in flow.transforms:
        z, ld = t.forward(z, ctx)
        logabsdet = logabsdet + ld

    # Finite-difference 2x2 Jacobian per sample
    eps = 1e-4
    for i in range(6):
        th_i = theta[i : i + 1]
        c_i = ctx[i : i + 1]

        def fwd_fn(inp: torch.Tensor) -> torch.Tensor:
            out = inp
            for tr in flow.transforms:
                out, _ = tr.forward(out, c_i)
            return out

        j0 = (fwd_fn(th_i + torch.tensor([[eps, 0.0]])) - fwd_fn(th_i - torch.tensor([[eps, 0.0]]))) / (2 * eps)
        j1 = (fwd_fn(th_i + torch.tensor([[0.0, eps]])) - fwd_fn(th_i - torch.tensor([[0.0, eps]]))) / (2 * eps)
        jac = torch.cat([j0, j1], dim=0)
        num_ld = torch.log(torch.abs(torch.linalg.det(jac)))
        assert torch.isclose(logabsdet[i], num_ld, atol=2e-3)

    samples = flow.sample(num_samples=25, cond=cond)
    assert samples.shape == (6, 25, 2)
    lp = flow.log_prob(theta, cond)
    assert lp.shape == (6,)
    assert torch.isfinite(lp).all()


def test_pairs_model_invariances_and_n0_edge_case():
    torch.manual_seed(42)
    cfg = ModelConfig(embed_dim=16, jet_hidden_dim=16, jet_num_heads=2, jet_num_layers=1, event_hidden_dim=32, event_num_blocks=1)
    model = PairsModel(
        jet_feat_dim=11,
        evt_feat_dim=16,
        theta_mean=torch.tensor([5.0, 2.5]),
        theta_std=torch.tensor([2.88675, 4.3301]),
        cfg=cfg,
    )
    model.eval()

    b, max_n, j_max = 4, 3, 6
    jf = torch.randn(b, max_n, j_max, 11)
    jm = torch.zeros(b, max_n, j_max, dtype=torch.bool)
    jm[:, :, :4] = True
    # Corrupt padded jet slots with extreme values -> must not affect embeddings
    jf[~jm] = 9999.0
    ef = torch.randn(b, max_n, 16)
    em = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [True, True, True],
            [False, False, False],  # N=0 edge case!
        ],
        dtype=torch.bool,
    )

    t_bar, counts = model.aggregate_sets(jf, jm, ef, em)
    assert torch.equal(counts, torch.tensor([2, 1, 3, 0]))
    assert torch.isfinite(t_bar).all()
    assert torch.allclose(t_bar[3], torch.zeros(16))

    # Jet permutation invariance
    perm_j = torch.randperm(j_max)
    t_bar_jperm, _ = model.aggregate_sets(jf[:, :, perm_j], jm[:, :, perm_j], ef, em)
    torch.testing.assert_close(t_bar, t_bar_jperm, atol=1e-5, rtol=1e-5)

    # Set element permutation invariance
    perm_n = torch.tensor([2, 0, 1])
    t_bar_nperm, _ = model.aggregate_sets(jf[:, perm_n], jm[:, perm_n], ef[:, perm_n], em[:, perm_n])
    torch.testing.assert_close(t_bar, t_bar_nperm, atol=1e-5, rtol=1e-5)


def test_end_to_end_pairs_smoke_pipeline(tmp_path: Path):
    # Synthetic miniature 2-process dataset to test Stage 1 -> Stage 2 -> Stage 3 -> Eval
    rng = np.random.default_rng(99)

    def _make_proc(name: str, n_ev: int, base_shift: float, w_cols: int, total_yield: float) -> ProcessData:
        raw_jets = rng.normal(loc=base_shift, scale=1.0, size=(n_ev, 6, 6)).astype(np.float32)
        raw_jets[:, :, 0:3] = 0.0
        raw_jets[:, 0:2, 0] = 1.0
        raw_jets[:, 2:4, 2] = 1.0
        jmask = np.zeros((n_ev, 6), dtype=bool)
        jmask[:, :4] = True
        invs = rng.normal(loc=base_shift, scale=0.5, size=(n_ev, 16)).astype(np.float32)
        weights = np.full((n_ev, w_cols), total_yield / n_ev, dtype=np.float64)
        if w_cols == 3:
            weights[:, 1] *= 1.5
            weights[:, 2] *= 0.7
        return ProcessData(
            name=name,
            raw_jets=raw_jets,
            jet_mask=jmask,
            evt_invariants=invs,
            weights=weights,
            n_gen=n_ev,
            n_csv_events=n_ev,
        )

    procs = {
        "top": _make_proc("top", 200, 0.0, 1, 80.0),
        "HH": _make_proc("HH", 200, 2.0, 3, 10.0),
    }

    cfg = PairsConfig(
        prior=PriorConfig(mu_min=0.0, mu_max=5.0, kl_min=-2.0, kl_max=4.0),
        data=DataConfig(
            data_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            max_jets=6,
            backgrounds=("top",),
            signal="HH",
        ),
        model=ModelConfig(
            embed_dim=16,
            jet_hidden_dim=16,
            jet_num_heads=2,
            jet_num_layers=1,
            event_hidden_dim=32,
            event_num_blocks=1,
            flow_num_transforms=2,
            flow_hidden_dim=32,
            flow_num_blocks=1,
            flow_num_bins=6,
            context_hidden_dim=32,
        ),
        train=TrainConfig(
            seed=7,
            device="cpu",
            stage1_epochs=2,
            stage1_batches_per_epoch=5,
            stage1_batch_size=32,
            stage1_val_sets=64,
            stage2_embed_batch_size=64,
            stage2_n_train_sets=120,
            stage2_n_val_sets=40,
            stage2_n_test_sets=40,
            stage3_epochs=3,
            stage3_batch_size=32,
            eval_num_samples=50,
            eval_acauc_levels=25,
        ),
        output_dir=tmp_path / "out",
    )

    splits, stats = build_splits(procs, cfg.data)
    prior = UniformPrior2D(cfg.prior)
    device = torch.device("cpu")
    model = PairsModel(
        jet_feat_dim=11,
        evt_feat_dim=16,
        theta_mean=torch.tensor(prior.mean, dtype=torch.float32),
        theta_std=torch.tensor(prior.std, dtype=torch.float32),
        cfg=cfg.model,
    )

    h1 = train_stage1(model, splits["train_s1"], splits["val"], prior, stats, cfg, device)
    assert len(h1["val_nll"]) == 2 and np.isfinite(h1["val_nll"]).all()

    aggs = run_stage2(model, splits, prior, cfg, device)
    assert aggs["train"]["T_bar"].shape == (120, 16)
    assert aggs["test"]["T_bar"].shape == (40, 16)

    h3 = train_stage3(model, aggs["train"], aggs["val"], prior, stats, cfg, device)
    assert len(h3["val_nll"]) == 3 and np.isfinite(h3["val_nll"]).all()

    res = evaluate_stage3_model(
        model, prior, aggs["test"], device=device, num_samples=50, num_acauc_levels=25
    )
    assert np.isfinite(res.test_nll)
    assert 0.0 <= res.acauc_2d_hpd <= 0.5
    assert 0.0 <= res.acauc_mu_marginal <= 0.5
    assert 0.0 <= res.acauc_kl_marginal <= 0.5
    assert res.rmae_mu >= 0.0 and res.rmae_kl >= 0.0


def test_benchmarks_rate_binned_and_nre_oracle() -> None:
    """Verify Rate-only, Binned 2D, and N=1 NRE Oracle on synthetic splits."""
    from pairs_bbtautau.benchmarks import (
        BinnedLikelihoodModel,
        evaluate_nre_oracle_sets,
        evaluate_rate_only_baseline,
        sample_full_test_sets,
        train_nre_oracle,
    )

    rng = np.random.default_rng(99)

    def _make_proc(name: str, n_ev: int, base_shift: float, w_cols: int, total_yield: float) -> ProcessData:
        raw_jets = np.zeros((n_ev, 6, 6), dtype=np.float32)
        raw_jets[:, :4, 3:6] = rng.normal(loc=base_shift * 20.0, scale=30.0, size=(n_ev, 4, 3)).astype(np.float32)
        raw_jets[:, :2, 0] = 1.0
        raw_jets[:, 2:4, 2] = 1.0
        jmask = np.zeros((n_ev, 6), dtype=bool)
        jmask[:, :4] = True
        invs = rng.normal(loc=1.0 + base_shift * 0.2, scale=0.2, size=(n_ev, 16)).astype(np.float32)
        weights = np.full((n_ev, w_cols), total_yield / n_ev, dtype=np.float64)
        if w_cols == 3:
            weights[:, 1] *= 1.5
            weights[:, 2] *= 0.7
        return ProcessData(
            name=name,
            raw_jets=raw_jets,
            jet_mask=jmask,
            evt_invariants=invs,
            weights=weights,
            n_gen=n_ev,
            n_csv_events=n_ev,
        )

    procs = {
        "top": _make_proc("top", 120, 0.0, 1, 40.0),
        "HH": _make_proc("HH", 120, 1.0, 3, 5.0),
    }
    data_cfg = DataConfig(
        backgrounds=("top",),
        signal="HH",
        lumi_pb_inv=1.0,
        max_jets=6,
    )
    splits, stats = build_splits(procs, data_cfg)
    prior = UniformPrior2D(PriorConfig(mu_min=0.0, mu_max=5.0, kl_min=-2.0, kl_max=5.0))
    rng = np.random.default_rng(99)
    train_sampler = ProcessSampler(splits["train_s3"])
    test_sampler = ProcessSampler(splits["test"])

    datasets = sample_full_test_sets(test_sampler, prior, stats, n_sets=8, rng=rng)
    assert datasets.theta.shape == (8, 2)
    assert len(datasets.m_bbtt) == 8

    rate_res, rate_post, _, _ = evaluate_rate_only_baseline(
        test_sampler, prior, datasets.theta, datasets.n_events, n_mu=11, n_kl=13, num_samples=50, num_acauc_levels=11, rng=rng
    )
    assert np.isfinite(rate_res.test_nll) and rate_post.shape == (8, 11, 13)

    binned = BinnedLikelihoodModel(train_sampler, stats)
    bin_res, bin_post, _, _ = binned.evaluate(
        datasets, prior, n_mu=11, n_kl=13, num_samples=50, num_acauc_levels=11, rng=rng
    )
    assert np.isfinite(bin_res.test_nll) and bin_post.shape == (8, 11, 13)

    mcfg = ModelConfig(embed_dim=8, jet_hidden_dim=16, jet_num_heads=2, jet_num_layers=1, event_hidden_dim=16, event_num_blocks=1)
    oracle = train_nre_oracle(
        splits["train_s1"], splits["val"], prior, stats, mcfg, torch.device("cpu"),
        epochs=1, batches_per_epoch=2, batch_size=16, mu_max_train=10.0,
    )
    nre_res, nre_post, _, _ = evaluate_nre_oracle_sets(
        oracle, test_sampler, datasets, prior, torch.device("cpu"),
        n_mu=11, n_kl=13, num_samples=50, num_acauc_levels=11, rng=rng,
    )
    assert np.isfinite(nre_res.test_nll) and nre_post.shape == (8, 11, 13)


def test_stage1_importance_weighted_and_m_max(tmp_path: Path) -> None:
    """Verify importance-weighted Stage 1 unbiasedness and Corollary 3 M>2 training."""
    rng = np.random.default_rng(77)

    def _make_proc(name: str, n_ev: int, w_cols: int, total_yield: float) -> ProcessData:
        raw_jets = np.zeros((n_ev, 6, 6), dtype=np.float32)
        raw_jets[:, :4, 3:6] = rng.normal(0.0, 20.0, size=(n_ev, 4, 3)).astype(np.float32)
        raw_jets[:, :2, 0] = 1.0
        raw_jets[:, 2:4, 2] = 1.0
        jmask = np.zeros((n_ev, 6), dtype=bool)
        jmask[:, :4] = True
        invs = rng.normal(0.0, 1.0, size=(n_ev, 16)).astype(np.float32)
        weights = np.full((n_ev, w_cols), total_yield / n_ev, dtype=np.float64)
        if w_cols == 3:
            weights[:, 1] *= 1.4
            weights[:, 2] *= 0.8
        return ProcessData(
            name=name,
            raw_jets=raw_jets,
            jet_mask=jmask,
            evt_invariants=invs,
            weights=weights,
            n_gen=n_ev,
            n_csv_events=n_ev,
        )

    procs = {"top": _make_proc("top", 100, 1, 50.0), "HH": _make_proc("HH", 100, 3, 10.0)}
    data_cfg = DataConfig(backgrounds=("top",), signal="HH", lumi_pb_inv=1.0, max_jets=6)
    splits, stats = build_splits(procs, data_cfg)
    prior = UniformPrior2D(PriorConfig(mu_min=0.0, mu_max=5.0, kl_min=-2.0, kl_max=4.0))

    sampler = ProcessSampler(splits["train_s1"])
    thetas = prior.sample(4096, rng)
    sizes = np.where(rng.integers(0, 2, size=4096) == 0, 1, 4)
    _, _, _, iw = sampler.sample_small_sets_importance_weighted(thetas, sizes, rng)
    assert iw.shape == (4096,)
    assert abs(float(iw.mean()) - 1.0) < 0.1

    cfg = PairsConfig(
        prior=prior.cfg,
        data=data_cfg,
        model=ModelConfig(
            embed_dim=8, jet_hidden_dim=16, jet_num_heads=2, jet_num_layers=1,
            event_hidden_dim=16, event_num_blocks=1, flow_num_transforms=2,
            flow_hidden_dim=16, flow_num_blocks=1, flow_num_bins=6, context_hidden_dim=16,
        ),
        train=TrainConfig(
            seed=11, device="cpu", stage1_epochs=2, stage1_batches_per_epoch=3,
            stage1_batch_size=16, stage1_val_sets=32, stage1_m_max=4,
            stage1_importance_weighted=True,
        ),
        output_dir=tmp_path / "iw_out",
    )
    model = PairsModel(
        11, 16, torch.tensor(prior.mean, dtype=torch.float32), torch.tensor(prior.std, dtype=torch.float32), cfg.model
    )
    hist = train_stage1(model, splits["train_s1"], splits["val"], prior, stats, cfg, torch.device("cpu"))
    assert len(hist["val_nll"]) == 2 and np.isfinite(hist["val_nll"]).all()


def test_mc_bootstrap_exact_variance() -> None:
    """Verify Poisson MC-weight bootstrap reproduces Var(N) = Lambda + sum w_i^2."""
    rng = np.random.default_rng(123)
    n_mc = 50
    w_val = 2.0  # Lambda = 100, sum w_i^2 = 50 * 4 = 200 -> Var(N) = 300
    raw_jets = np.zeros((n_mc, 6, 6), dtype=np.float32)
    jmask = np.ones((n_mc, 6), dtype=bool)
    invs = np.zeros((n_mc, 16), dtype=np.float32)
    procs = {
        "top": ProcessData("top", raw_jets, jmask, invs, np.full((n_mc, 1), w_val), n_mc, n_mc),
        "HH": ProcessData("HH", raw_jets, jmask, invs, np.zeros((n_mc, 3)), n_mc, n_mc),
    }
    data_cfg = DataConfig(
        backgrounds=("top",), signal="HH", lumi_pb_inv=1.0, max_jets=6,
        stage1_train_frac=0.25, stage3_train_frac=0.25, val_frac=0.25, test_frac=0.25,
    )
    splits, _ = build_splits(procs, data_cfg)
    # Use test split where compensated w_val = 2.0 * 4 = 8.0, n_split = 12 or 13 -> check exact formula
    pool = splits["train_s3"].processes["top"]
    w_arr = pool.weights[:, 0]
    lam = float(w_arr.sum())
    expected_var = lam + float(np.sum(w_arr ** 2))

    sampler = ProcessSampler(splits["train_s3"])
    prior = UniformPrior2D(PriorConfig(mu_min=0.0, mu_max=1.0, kl_min=0.0, kl_max=2.0))
    embs = {
        "top": rng.normal(size=(pool.num_events, 4)).astype(np.float32),
        "HH": np.zeros((splits["train_s3"].processes["HH"].num_events, 4), dtype=np.float32),
    }
    fixed = np.zeros((4000, 2), dtype=np.float32)
    agg = sampler.sample_aggregated_datasets(embs, prior, n_datasets=4000, rng=rng, fixed_thetas=fixed, mc_bootstrap=True)
    emp_mean = float(agg["N"].mean())
    emp_var = float(agg["N"].var())
    assert abs(emp_mean - lam) / lam < 0.05
    assert abs(emp_var - expected_var) / expected_var < 0.12
