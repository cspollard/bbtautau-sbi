# PAIRS Joint Amortized Inference for $HH \to b\bar{b}\tau^+\tau^-$

This directory implements **stat-only** joint amortized Bayesian inference over the di-Higgs signal strength $\mu_{HH}$ and Higgs self-coupling modifier $\kappa_\lambda$ using the **PAIRS** (*Pretraining Aggregators for Inference at aRbitrary Set-sizes*, [arXiv:2605.07972](https://arxiv.org/abs/2605.07972)) framework in PyTorch.

---

## 1. Big Picture

### Goal
Given a collider dataset $X_N = \{x_1, \dots, x_N\}$ of $N$ selected $b\bar{b}\tau^+\tau^-$ events at integrated luminosity $\mathcal{L} = 100\text{ fb}^{-1}$ ($N \approx 16{,}500$), we want the full 2D Bayesian posterior:
$$p(\mu_{HH}, \kappa_\lambda \mid X_N)$$
across a 2D uniform prior $\mu_{HH} \in [0, 10]$, $\kappa_\lambda \in [-5, 10]$, capturing both total event-count information and $\kappa_\lambda$-dependent event kinematics (e.g., breaking the quadratic yield symmetry around $\kappa_\lambda \approx 2.5$).

### Why PAIRS?
Training a Deep Set or Transformer posterior estimator end-to-end on $N \sim 16{,}500$ events per dataset requires $\mathcal{O}(B \cdot N)$ memory and compute on every gradient step. **PAIRS** decouples representation learning from dataset-scale density estimation in three stages:

1. **Stage 1 — Learn the per-event embedding $t_\omega$ at $N \in \{1, 2\}$:**
   Train a per-event encoder $t_\omega(x_i) \in \mathbb{R}^l$ and a conditional normalizing flow head $q_\phi(\theta \mid T_\omega(X_N), N)$ jointly on tiny sets ($N \in \{1, 2\}$) with masked mean-pool aggregation $T_\omega(X_N) = \frac{1}{N}\sum_{i=1}^N t_\omega(x_i)$. Under Theorem 1 of arXiv:2605.07972, optimality on $N \in \{1, 2\}$ recovers a sufficient statistic up to an affine transformation for *arbitrary* $N$.
2. **Stage 2 — Cache embeddings and dataset aggregates:**
   Freeze $t_\omega$, embed every MC event once, and build Poisson-sampled pseudo-datasets $\mathcal{D}_{\text{agg}} = \{(\theta^{(j)}, \bar{T}^{(j)}, N^{(j)})\}$ at $\mathcal{L} = 100\text{ fb}^{-1}$ from held-out MC pools.
3. **Stage 3 — Dataset-wide inference:**
   Finetune the 2D conditional normalizing flow head $q_\phi(\theta \mid \bar{T}, N)$ on the cached $(\bar{T}, N)$ summaries. Each gradient step is independent of $N$ and takes milliseconds.

### Current Status & Baseline Results
- **Scope:** Stat-only inference (no active nuisance parameters yet). The `NuisanceModel` interface in [physics.py](file:///Users/cspollard/Physics/bbtautau-sbi/inference/pairs_bbtautau/physics.py) is designed so set-level nuisances $\psi$ (e.g., background normalization, JES/JER) can be added next without changing the PAIRS pipeline.
- **Disjoint MC splits:** Each process is split `35% / 25% / 20% / 20%` into `train_s1 / train_s3 / val / test` with `1 / split_frac` weight compensation so Stage 3 never trains on in-sample Stage 1 embeddings.
- **Baseline metrics** (`embed_dim=12`, 2,000 test datasets, 1,000 posterior samples each; see `outputs/metrics.json`):
  - **Test NLL (physical units):** `3.9171` (vs. Uniform Prior `5.0106`)
  - **RMAE:** `0.2102` ($\mu_{HH}$), `0.1821` ($\kappa_\lambda$)
  - **Calibration (ACAUC, 0 = ideal):** `0.0184` (2D HPD), `0.0176` ($\mu_{HH}$ marginal), `0.0110` ($\kappa_\lambda$ marginal)

---

## 2. Physics & Data Pipeline

- **Inputs:** CSV files (`HH.csv`, `top.csv`, `DYbb.csv`, `ZH.csv`, `higgs.csv`) produced by Rivet, cached after selection to `cache/*.npz` for sub-second reloads.
- **Weights:** CSV weights are cross sections in $\text{pb}$. Expected yields at $\mathcal{L} = 100{,}000\text{ pb}^{-1}$ are normalized by $\mathcal{L} / N_{\text{gen}}$ where $N_{\text{gen}} = \max(\text{eventnumber}) + 1$.
- **$\kappa_\lambda$ Reweighting:** Exact quadratic interpolation for `HH` from `(weight_nom [kl=1], weight_kl0 [kl=0], weight_kl2 [kl=2])`.
- **Event Selection** (applied on the full jet list before padding/truncating to `max_jets=12`):
  - Exactly $2$ $b$-tagged jets and $2$ $\tau$-tagged (non-$b$) jets
  - $m_{bb\tau\tau} > 200\text{ GeV}$, $75 < m_{bb} < 150\text{ GeV}$, $50 < m_{\tau\tau} < 125\text{ GeV}$
- **Stage 1 Signal Contrast:** At $100\text{ fb}^{-1}$, SM $HH$ is only ~9.3 events against ~16,429 background events (~1.4% average signal fraction over $\mu_{HH} \in [0, 10]$). Stage 1 pretraining therefore uses `--stage1-mu-max 300.0` by default so $N \in \{1, 2\}$ sets see strong $\kappa_\lambda$-dependent kinematic contrast, while Stages 2 and 3 use the exact $100\text{ fb}^{-1}$ luminosity and target prior.
- **Finite-MC Regularization:** Because `top` and `DYbb` currently have limited effective luminosity ($L_{\text{eff}} \approx 137\text{--}139\text{ fb}^{-1}$ total, $\approx 28\text{--}35\text{ fb}^{-1}$ per split), Stage 3 applies analytical finite-MC covariance augmentation to $\bar{T}$ by default (`--no-stage3-mc-aug` to disable).

---

## 3. Code Structure

- `run_pipeline.py` — CLI entrypoint (`--stage {all,prepare,stage1,stage2,stage3,eval}`)
- `pairs_bbtautau/`
  - `config.py` — `PriorConfig`, `DataConfig`, `ModelConfig`, `TrainConfig`, `PairsConfig`, device/seed helpers
  - `data.py` — CSV parsing, event selection, feature engineering, `.npz` caching, 4-way disjoint splitting
  - `physics.py` — quadratic $\kappa_\lambda$ reweighting, `UniformPrior2D`, `NuisanceModel` / `StatOnlyNuisanceModel`, `ProcessSampler`
  - `models.py` — `PerEventEmbedder`, `ConditionalSplineFlow2D`, `PairsModel`
  - `stages.py` — Stage 1 training, Stage 2 embedding & Poisson aggregate caching, Stage 3 flow finetuning
  - `metrics.py` — physical-unit Test NLL, MAE/RMAE, 2D HPD & 1D marginal ACAUC
  - `plotting.py` — loss curves, yield curve, SM & grid 2D posterior contours, recovery/pulls, calibration plot
- `tests/test_pairs.py` — unit and end-to-end smoke tests

---

## 4. Neural Network Architectures & Features

### Input Representation
- **Per-jet features (`11` dims, up to `max_jets=12` with boolean `jet_mask`):**
  - Tags (3): `btagged`, `ctagged`, `tautagged`
  - Kinematics (8): $\log(p_T / 50)$, $\eta$, $\cos\phi$, $\sin\phi$, $\log(|p| / 50)$, $p_x/100$, $p_y/100$, $p_z/100$
- **Event-level invariants (`16` dims, computed from the unclipped jet collection):**
  - $\log(m_{bb\tau\tau}/300)$, $m_{bb\tau\tau}/300$, $m_{bb}/125$, $m_{\tau\tau}/100$, $\log((p_{T,bb}+1)/100)$, $\log((p_{T,\tau\tau}+1)/100)$, $\log((p_{T,bb\tau\tau}+1)/50)$, $\Delta R_{bb}$, $\Delta R_{\tau\tau}$, $|\Delta\phi_{bb,\tau\tau}|$, $\log(H_T/200)$, $n_{\text{jets}}/5$, and $\log(p_T/50)$ for leading/subleading $b$-jets and $\tau$-jets.

### Per-Event Embedder $t_\omega : \mathcal{X} \to \mathbb{R}^l$ (`PerEventEmbedder`)
Designed without `BatchNorm` (using `LayerNorm` + `SiLU`, `dropout=0.12`) for variable set-size stability:
1. **Jet Projection:** `Linear(11 -> 64) -> LayerNorm -> SiLU` (masked).
2. **Set Transformer over Jets:** `2` pre-norm `SetTransformerJetBlock` layers (`MultiheadAttention` with `embed_dim=64`, `num_heads=4`, key padding mask + `FFN(64 -> 128 -> 64)`), followed by masked mean pooling across valid jets (`-> 64`).
3. **Event Fusion & Residual MLP:** Concatenate pooled jet vector (`64`) + event invariants (`16`) $\to$ `Linear(80 -> 128) -> LayerNorm -> SiLU` $\to$ `2` pre-norm `ResidualMLPBlock(128)` layers $\to$ `LayerNorm -> SiLU`.
4. **Unconstrained Output Projection:** `Linear(128 -> l)` with **no** activation or softmax (`embed_dim=12` by default), satisfying Assumption A3 of PAIRS Theorem 1.

### Set Aggregator $T_\omega(X_N)$
- Masked mean pooling: $T_\omega(X_N) = \frac{1}{N} \sum_{i=1}^N t_\omega(x_i) \in \mathbb{R}^l$ (returns $\mathbf{0}$ if $N=0$).
- Conditioning vector passed to the flow head: `[T_bar, N_enc]`, where `N_enc = N / 2` in Stage 1 and standardized $\log(N + 1)$ in Stage 3 (with standardized $\bar{T}$ in Stage 3).

### Conditional Normalizing Flow Head $q_\phi(\theta \mid \bar{T}, N)$ (`ConditionalSplineFlow2D`)
- **Targets:** $\theta = (\mu_{HH}, \kappa_\lambda)$ standardized to zero mean and unit variance using prior moments, with $-\sum_j \log \sigma_j$ added to `log_prob` for exact NLL in physical units.
- **Context Network:** `Linear(l + 1 -> 64) -> LayerNorm -> SiLU -> Linear(64 -> 64) -> LayerNorm -> SiLU`.
- **Flow Transforms:** `4` alternating `RationalQuadraticCoupling2D` layers (transforming $\mu_{HH}$ then $\kappa_\lambda$, etc.) over a standard 2D Gaussian base $\mathcal{N}(0, I_2)$:
  - Conditioner per coupling layer: `Linear(1 + 64 -> 64) -> LayerNorm -> SiLU -> 1 ResidualMLPBlock(64) -> LayerNorm -> SiLU -> Linear(64 -> 3K - 1)` (near-identity initialized).
  - Spline: `K = 10` rational-quadratic bins on `[-3.5, 3.5]` with linear identity tails outside.

---

## 5. Usage

```bash
cd /Users/cspollard/Physics/bbtautau-sbi/inference

# Run unit and smoke tests
.venv/bin/pytest -v

# Run full Stage 1 -> Stage 2 -> Stage 3 -> Evaluation pipeline (default config)
.venv/bin/python run_pipeline.py --stage all

# Re-run only Stage 2 -> Stage 3 -> Evaluation (e.g. after changing Stage 3 settings)
.venv/bin/python run_pipeline.py --stage stage2 --stage3-epochs 120

# Custom production run
.venv/bin/python run_pipeline.py --stage all \
  --embed-dim 16 \
  --stage1-epochs 60 --stage1-batches 300 \
  --stage2-train-sets 50000 --stage2-val-sets 8000 --stage2-test-sets 5000 \
  --stage3-epochs 120 --eval-samples 2000
```

Outputs are written to `outputs/` (`stage1_best.pt`, `stage2_aggregates.pt`, `stage3_best.pt`, `metrics.json`, and diagnostic figures in `outputs/figs/`).

---

## 6. Planned Next Steps

1. **Verify Optimality Against Reference Posteriors (Beyond Coverage)**
   - **Rate-only analytical baseline:** Compute the exact Poisson-count posterior $p(\theta \mid N)$ on the test set to quantify the pure kinematic gain from $\bar{T}$.
   - **Binned likelihood reference:** Evaluate exact binned Poisson posteriors (e.g. in $(m_{bb\tau\tau}, m_{bb})$ or a learned classifier score) on benchmark test points.
   - **Stat-only unbinned NRE / likelihood-ratio oracle:** Because events are conditionally independent given $\theta$ in the stat-only setting, an $N=1$ classifier/likelihood-ratio oracle gives a direct grid/MCMC reference for $\log p(X_N \mid \theta) = \log \mathrm{Pois}(N \mid \Lambda(\theta)) + \sum_{i=1}^N \log r(x_i \mid \theta)$ to test whether PAIRS converges to *the* true posterior and separate any PAIRS approximation gap from physics/MC limits.
2. **Embedding Dimension $\ell$ Scan**
   - Sweep `--embed-dim` $\ell \in \{2, 4, 8, 12, 16, 32, 64, 128\}$ across seeds and compare Stage 1 Val NLL, Stage 3 Test NLL, RMAE, and ACAUC to locate the sufficiency saturation threshold ($\ell \ge k$ in Theorem 1 / Corollary 2).
3. **MC Statistics, Signal Normalization, and Preselection**
   - **Expanded MC:** Rebuild the cache with the 2M+ `top` run (and expanded `DYbb` / `ZH` samples) to remove the finite-MC bottleneck ($L_{\text{eff, split}} \approx 28\text{--}35\text{ fb}^{-1}$), which currently causes early Stage 1 overfitting and forces Stage 3 MC-covariance broadening.
   - **NNLO+NNLL $HH$ $K$-factor & experimental benchmark:** Current `HH.csv` uses LO `loop_sm_hh` ($\sigma_{\text{SM}} \approx 0.76\text{ fb}$ vs. $\sim 2.27\text{ fb}$ at NNLO+NNLL), yielding $\mu_{HH}^{95\%} = 9.35$ at $\kappa_\lambda = 1$ vs. ATLAS `3.9` ($139\text{ fb}^{-1}$) and CMS `5.2` ($138\text{ fb}^{-1}$). Apply/compare with the NNLO+NNLL $K$-factor (~$3\times$) to benchmark truth-level stat-only sensitivity directly against Run 2 limits.
   - **Preselection:** Require $\ge 4$ jets with $\ge 2$ $b$-tagged and $\ge 2$ $\tau$-tagged jets as a preselection (currently enforced offline as exact $2b+2\tau$; moving a $\ge 4$-jet, $\ge 2b$, $\ge 2\tau$ preselection upstream into Rivet/CSV generation will keep CSVs compact where $S/B$ is negligible) and explore tighter kinematic/mass-window cuts to suppress the `top`/`DYbb` background (`16,429` events at $100\text{ fb}^{-1}$).
4. **Nuisance Parameters**
   - Enable set-level nuisance parameters $\psi$ (background normalizations, JES/JER) via `NuisanceModel` in [physics.py](file:///Users/cspollard/Physics/bbtautau-sbi/inference/pairs_bbtautau/physics.py).

