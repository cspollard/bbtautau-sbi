# PAIRS Joint Amortized Bayesian Inference for $bb\tau\tau$

Modular PyTorch implementation of **PAIRS** (*Pretraining Aggregators for Inference at aRbitrary Set-sizes*, [arXiv:2605.07972v2](https://arxiv.org/abs/2605.07972)) for joint 2D posterior inference over $(\mu_{HH}, \kappa_\lambda)$ in the $HH \to b\bar{b}\tau^+\tau^-$ analysis.

## Architecture & Pipeline

1. **Data & Physics (`pairs_bbtautau/data.py`, `pairs_bbtautau/physics.py`)**
   - Parses `HH.csv`, `top.csv`, `ZH.csv`, `higgs.csv`, `DYbb.csv` and caches selected arrays to `.npz` (`cache/`).
   - Selection (on full jet list before padding/truncating to `max_jets=12`): exactly 2 $b$-jets, 2 $\tau$ jets, $m_{bb\tau\tau} > 200$ GeV, $75 < m_{bb} < 150$ GeV, $50 < m_{\tau\tau} < 125$ GeV.
   - Weights normalized by `lumi / n_gen` (`n_gen = max(eventnumber) + 1`, `lumi = 100e3` pb$^{-1}$).
   - Exact quadratic $\kappa_\lambda$ reweighting for `HH` from `(weight_nom [kl=1], weight_kl0 [kl=0], weight_kl2 [kl=2])`.
   - Disjoint 4-way `35% / 25% / 20% / 20%` (`train_s1 / train_s3 / val / test`) MC pools per process with exact `N_proc / N_split` weight compensation, ensuring Stage 3 consumes strictly out-of-sample embeddings from `t_omega`.
   - Clean `NuisanceModel` abstraction (`StatOnlyNuisanceModel` active by default) so set-level nuisances $\psi$ (background normalization, JES/JER) can be added without modifying the PAIRS training/inference pipeline.

2. **PAIRS Model (`pairs_bbtautau/models.py`)**
   - **Per-event embedder** $t_\omega : \mathcal{X} \to \mathbb{R}^l$: masked Set Transformer over jets + masked mean pooling + event-level physics invariants + residual MLP ending in an unconstrained linear layer (`LayerNorm` + `SiLU`, no `BatchNorm`, no output softmax).
   - **Aggregator**: masked mean pooling $T_\omega(X_N) = \frac{1}{N} \sum_{i=1}^N t_\omega(x_i)$.
   - **Conditional Normalizing Flow head** $q_\phi(\theta \mid T_\omega(X_N), N_{\text{enc}})$: 2D Rational-Quadratic Spline coupling flow over standardized $(\mu_{HH}, \kappa_\lambda)$ with exact physical-unit Jacobian correction.

3. **Three-Stage Training (`pairs_bbtautau/stages.py`)**
   - **Stage 1**: Jointly optimize $(\omega, \phi)$ on $N \in \{1, 2\}$ sets from `train_s1` with AdamW, warmup + cosine annealing, gradient clipping (`1.0`), and early stopping on `val` NLL (`--stage1-mu-max` defaults to `300.0` so $N \in \{1, 2\}$ sets observe strong signal contrast across $\kappa_\lambda$ while preserving exact likelihood-ratio sufficiency; pass `--stage1-mu-max 0` to match the target prior directly).
   - **Stage 2**: Freeze $t_\omega$, compute per-event embeddings once, and cache dataset-level aggregates $\mathcal{D}_{\text{agg}} = \{(\theta^{(j)}, \bar{T}^{(j)}, N^{(j)})\}$ with $N^{(j)} \sim \mathrm{Poisson}(\Lambda(\theta^{(j)}))$ from `train_s3` (train), `val`, and `test` at exact physical `lumi = 100e3` and target prior bounds.
   - **Stage 3**: Finetune $q_\phi$ on cached $\mathcal{D}_{\text{agg}}$ conditioned on standardized $\bar{T}$ and standardized $\log(N + 1)$, with finite-MC background covariance augmentation (`--no-stage3-mc-aug` to disable).

## Usage

```bash
cd /Users/cspollard/Physics/bbtautau-sbi/inference

# Run unit tests
.venv/bin/pytest -v

# Run full Stage 1 -> Stage 2 -> Stage 3 -> Evaluation pipeline (default config)
.venv/bin/python run_pipeline.py --stage all

# Longer production run (e.g. on MPS/CUDA)
.venv/bin/python run_pipeline.py --stage all \
  --embed-dim 16 \
  --stage1-epochs 60 --stage1-batches 300 \
  --stage2-train-sets 50000 --stage2-val-sets 8000 --stage2-test-sets 5000 \
  --stage3-epochs 120 --eval-samples 2000
```
