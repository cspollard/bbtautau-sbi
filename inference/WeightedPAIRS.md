# Weighted PAIRS: Exact MC-Stat Marginalization & Weighted Stage 1 Shortcut

This note sketches two theoretical results for extending **PAIRS** ([arXiv:2605.07972](https://arxiv.org/abs/2605.07972)) to weighted collider Monte Carlo samples:
1. **Exact MC-stat uncertainty marginalization** via Poisson bootstrapping of MC event weights.
2. **Generalization of PAIRS Stage 1 to weighted (and signed-weight) events**, yielding an importance-weighted Stage 1 shortcut that eliminates per-$\theta$ CDF resampling.

---

## 1. MC-Stat Uncertainties via Poisson-Sampled Event Weights

### Setup
Let a physics process $p$ (or combined mixture) be represented by $M$ generated MC events $\{x_i\}_{i=1}^M$ with expected yields $w_i(\theta, \psi) \ge 0$ normalized to integrated luminosity $\mathcal{L}$, so the nominal expected count is $\Lambda(\theta, \psi) = \sum_{i=1}^M w_i(\theta, \psi)$.

In standard data generation, drawing $N \sim \mathrm{Poisson}(\Lambda)$ followed by $N$ multinomial draws with probabilities $p_i = w_i / \Lambda$ is equivalent to drawing independent data counts
$$k_i \sim \mathrm{Poisson}(w_i), \qquad N = \sum_{i=1}^M k_i.$$
This models **data** shot noise assuming the MC template $\{w_i\}$ is exact, but ignores **finite-MC template uncertainty**.

### Poisson-Bootstrapped MC Template (Unbinned Barlow–Beeston)
Partition $\mathcal{X}$ into arbitrarily fine disjoint cells $\{B_c\}$. If the true differential yield in cell $B_c$ is $\lambda_c^{\text{true}} = \alpha_c \mu_c^{\text{MC}}$ (with per-MC-event weight $\alpha_c$), then the number of generated MC events landing in $B_c$ is $n_c^{\text{MC}} \sim \mathrm{Poisson}(\mu_c^{\text{MC}})$. Under the Poisson bootstrap for the unknown template rate, we draw
$$\tilde{n}_c^{\text{MC}} \sim \mathrm{Poisson}(n_c^{\text{MC}}).$$
By **additivity of independent Poisson random variables**, if we assign each individual MC event $i$ an independent bootstrap multiplicity
$$m_i \stackrel{\text{i.i.d.}}{\sim} \mathrm{Poisson}(1), \qquad \tilde{w}_i := m_i\, w_i,$$
then $\sum_{i \in B_c} m_i \sim \mathrm{Poisson}(n_c^{\text{MC}})$ for **every** partition $\{B_c\}$. Thus the construction is completely **binning-free** and **partition-invariant** (taking $|B_c| \to 0$ gives $n_c^{\text{MC}} \in \{0, 1\}$).

### Exact Moments & Rate–Shape Coupling
Treating $\psi_{\text{MC}} = \{m_i\}_{i=1}^M$ as a set-level nuisance drawn **once per pseudo-dataset** and then drawing data counts $k_i \mid m_i \sim \mathrm{Poisson}(m_i w_i)$:

1. **Unbiased expected yield:**
   $$\mathbb{E}_m[\tilde{\Lambda}] = \sum_{i=1}^M \mathbb{E}[m_i]\, w_i = \Lambda.$$
2. **Exact finite-MC template variance:**
   $$\mathrm{Var}_m(\tilde{\Lambda}) = \sum_{i=1}^M w_i^2\, \mathrm{Var}(m_i) = \sum_{i=1}^M w_i^2,$$
   matching the Kish effective sample size $N_{\text{eff}} = \Lambda^2 / \sum_i w_i^2$.
3. **Total marginal variance of $N$ (Law of Total Variance):**
   $$\mathrm{Var}(N) = \mathbb{E}_m[\mathrm{Var}(N \mid m)] + \mathrm{Var}_m(\mathbb{E}[N \mid m]) = \Lambda + \sum_{i=1}^M w_i^2.$$
4. **Exact marginal covariance of the unnormalized embedding sum $S_\omega = \sum_{i=1}^M k_i\, t_\omega(x_i)$:**
   $$\mathrm{Cov}(S_\omega) = \sum_{i=1}^M (w_i + w_i^2)\, t_\omega(x_i) t_\omega(x_i)^\top, \qquad \mathrm{Cov}(N, S_\omega) = \sum_{i=1}^M (w_i + w_i^2)\, t_\omega(x_i).$$

Unlike a post-hoc Gaussian covariance perturbation on $\bar{T}$ alone, Poisson-sampling the MC weights propagates finite-MC uncertainty simultaneously into **both** the total count $N$ and the kinematic summary $\bar{T}$, preserving their exact correlation.

### Practical Note on Train vs. Test Splits
- Real collider data has **zero** MC-stat uncertainty; only the MC templates carry $\sum_{i \in \text{train}} w_i^2$.
- Applying $m_i \sim \mathrm{Poisson}(1)$ during Stage 2/3 training on `train_s3` yields posteriors calibrated against `train_s3` MC-template uncertainty when evaluated on real data (or a high-$L_{\text{eff}}$ test pool).
- If `test` is itself a small finite MC split ($L_{\text{eff, test}} \lesssim L$), `train_s3` vs. `test` has total variance $\sigma_{\text{train}}^2 + \sigma_{\text{test}}^2$, so expanding low-$L_{\text{eff}}$ test pools (`DYbb`, `top`, `ZH`) remains important.

---

## 2. Generalization of PAIRS Stage 1 to Weighted Events (and Shortcut)

### Population Objective & Theorem 1 Invariance
Suppose the physical single-event distribution at $(\theta, \psi)$ is defined by weighted MC events $\{(x_i, w_i(\theta, \psi))\}_{i=1}^M$ with normalized density
$$p(x_i \mid \theta, \psi) = \frac{w_i(\theta, \psi)}{\Lambda(\theta, \psi)}, \qquad \Lambda(\theta, \psi) = \sum_{j=1}^M w_j(\theta, \psi).$$
For $N \in \{1, 2\}$, the PAIRS Stage 1 objective is:
$$\mathcal{L}_{\text{S1}}(\omega, \phi) = -\mathbb{E}_{N \sim \mathrm{Unif}\{1,2\}}\,\mathbb{E}_{\theta, \psi \sim p(\theta,\psi)}\,\mathbb{E}_{X_N \stackrel{\text{i.i.d.}}{\sim} p(\cdot \mid \theta, \psi)} \!\left[\log q_\phi\!\left(\theta \,\middle|\, \frac{1}{N}\sum_{r=1}^N t_\omega(x_r),\, N\right)\right].$$
PAIRS Theorem 1 depends **only** on the population expectation $\mathbb{E}_{p(\theta) p(X_N \mid \theta)}[\log q_\phi(\theta \mid T_\omega(X_N), N)]$ achieving its global optimum at $N \in \{1, 2\}$. Any unbiased estimator of this expectation has the exact same population minimizer $(\omega^\star, \phi^\star)$, and therefore recovers the same affine equivalence class $t^\star(x) = A t_\omega(x) + b$ for all $N \ge 1$.

### Importance-Weighted Stage 1 Shortcut (No Per-$\theta$ Resampling)
Let $q_0(x)$ be any fixed proposal distribution over the MC pool with $\mathrm{supp}(p(\cdot \mid \theta, \psi)) \subseteq \mathrm{supp}(q_0)$ (for example, uniform over MC events, or stratified 50% `HH` + 50% `bkg`). For any test function $g(x_1, \dots, x_N, \theta)$ and i.i.d. draws $x_r \sim p(\cdot \mid \theta, \psi)$:
$$\mathbb{E}_{x_1,\dots,x_N \stackrel{\text{i.i.d.}}{\sim} p(\cdot \mid \theta, \psi)}[g(x_1, \dots, x_N, \theta)] = \mathbb{E}_{x_1,\dots,x_N \stackrel{\text{i.i.d.}}{\sim} q_0} \!\left[\left(\prod_{r=1}^N \frac{p(x_r \mid \theta, \psi)}{q_0(x_r)}\right) g(x_1, \dots, x_N, \theta)\right].$$
Setting $g = -\log q_\phi(\theta \mid T_\omega(X_N), N)$ gives the **importance-weighted Stage 1 loss** over a minibatch $\{(\theta^{(b)}, \psi^{(b)}, X_{N_b}^{(b)})\}_{b=1}^B$ with $X_{N_b}^{(b)} \stackrel{\text{i.i.d.}}{\sim} q_0$:
$$\widehat{\mathcal{L}}_{\text{S1}}^{\text{IW}}(\omega, \phi) = -\frac{1}{B} \sum_{b=1}^B \left(\prod_{r=1}^{N_b} \frac{p(x_{b,r} \mid \theta^{(b)}, \psi^{(b)})}{q_0(x_{b,r})}\right) \log q_\phi\!\left(\theta^{(b)} \,\middle|\, \frac{1}{N_b}\sum_{r=1}^{N_b} t_\omega(x_{b,r}),\, N_b\right).$$

### Consequences & Practical Advantages
1. **Eliminates per-$\theta$ CDF resampling:** Each batch only draws $N_b \in \{1, 2\}$ indices from the fixed proposal $q_0$ and evaluates $w(x_{b,r} \mid \theta^{(b)}, \psi^{(b)}) / \Lambda(\theta^{(b)}, \psi^{(b)})$ on the sampled $1$ or $2$ events—no $O(M)$ CDF construction per $\theta^{(b)}$.
2. **Natural signal-vs-background variance reduction:** Choosing $q_0$ to oversample rare `HH` events (e.g., 50% `HH`, 50% `bkg`) gives low-variance gradients for $\kappa_\lambda$-sensitive kinematics at physical $\mu_{HH} \in [0, 10]$ without needing an artificial `--stage1-mu-max` prior override.
3. **Handles signed (negative) NLO weights:** Resampling cannot draw events with $w_i < 0$, whereas $\widehat{\mathcal{L}}_{\text{S1}}^{\text{IW}}$ remains an **exact unbiased** estimator of the physical cross-section objective even for signed weights, provided $q_\phi(\theta \mid T, N)$ is a properly normalized density ($\int q_\phi(\theta \mid T, N)\,\mathrm{d}\theta = 1$, guaranteed for normalizing flows).
   - *Variance caveat:* While unbiased, negative weights or extreme importance ratios $p / q_0$ increase gradient variance. Crucially, the set weight $\prod_{r=1}^N p(x_r \mid \theta)/q_0(x_r)$ scales exponentially in $N$: empirically on $bb\tau\tau$, importance-weighted $N \in \{1, 2\}$ achieved `Test NLL = 2.62, ACAUC = 0.016`, whereas $N \in \{1, 16\}$ suffered high weight variance and degraded to `Test NLL = 3.67, ACAUC = 0.188`. Importance weighting should therefore be used strictly at $N \in \{1, 2\}$.
