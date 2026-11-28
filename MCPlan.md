# MC Generation Plan for `bbtautau-sbi`

## Context & Statistical Targets

- **Target luminosity:** $L = 100\text{ fb}^{-1}$ (with planned checks at $L = 300\text{ fb}^{-1}$).
- **Split structure:** Each MC sample is split 4 ways (`35% / 25% / 20% / 20%` for `train_s1 / train_s3 / val / test`), so each `20%` evaluation split has $0.20 \times L_{\text{eff, total}}$.
- **Target effective luminosity:** $L_{\text{eff, split}} \ge 5 \times L$ (minimum $2 \times L$ for $300\text{ fb}^{-1}$ checks), corresponding to:
  - $L = 100\text{ fb}^{-1}$: $L_{\text{eff, total}} \ge 2{,}500\text{ fb}^{-1}$
  - $L = 300\text{ fb}^{-1}$: $L_{\text{eff, total}} \ge 7{,}500\text{ fb}^{-1}$ (or $\ge 3{,}000\text{ fb}^{-1}$ at $2\times L$)

### Current Status (after offline `2b + 2tau` + mass-window selection)

| Sample | Generated | Selected | Yield ($100\text{ fb}^{-1}$) | $L_{\text{eff, total}}$ | Target ($100\text{ fb}^{-1}$, $5\times L$) | Target ($300\text{ fb}^{-1}$, $5\times L$ / $2\times L$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `top` | 2,500,000 | 72,653 | 10,571.8 | $687.2\text{ fb}^{-1}$ | ~9–10M total | ~27M / ~11M |
| `DYbb` | 500,000 | 8,395 | 6,057.0 | $138.6\text{ fb}^{-1}$ | ~9M total | ~27M / ~11M |
| `ZH` | 4,659 | 385 | 97.5 | $395.0\text{ fb}^{-1}$ | ~30k total | ~90k–100k |
| `higgs` | 500,000 | 9,329 | 6.06 | $154{,}024\text{ fb}^{-1}$ | Sufficient | Sufficient |
| `HH-rw` ($\kappa_\lambda=1$) | 500,000 | 61,406 | 9.33 (LO) / ~28 ($K=3$) | $658{,}020\text{ fb}^{-1}$ (SM) | Sufficient at SM | Add $\kappa_\lambda=5$ anchor for tail |

> **Note on `HH-rw` high-$\kappa_\lambda$ tail:** Reweighting from $\kappa_\lambda=1$ to $\kappa_\lambda=+5, +10$ gives $w_{\text{CV}} = 6.5, 4.7$, dropping $L_{\text{eff, total}}$ to $2{,}037\text{ fb}^{-1}$ at $\kappa_\lambda=+10, \mu_{HH}=1$ ($41\text{ fb}^{-1}$ per 20% split at $\mu_{HH}=10$, and $3\times$ lower with $K=3$).

---

## Optimization Strategy (Before Running Large Batches)

### 1. Upstream Preselection in `rivet/tautaujets.cc` (~30× Smaller CSVs)
Without preselection, 27M `top` events produce ~`4.6 GB` of CSVs (and 27M `DYbb` is comparable).
- Require `jets.size() >= 4`, `n_b >= 2`, and `n_tau >= 2` (counting `!btag && tauTagged()`) before writing rows to the CSV.
- **Critical normalization requirement:** `weight_nom` in `tautaujets.cc` is the sample cross section $\sigma$ (in pb), and downstream code normalizes by $N_{\text{gen}}$. If rejected events are skipped in `tautaujets.cc`, `len(unique(eventnumber))` will equal $N_{\text{selected}}$ instead of $N_{\text{gen}}$. Record `N_gen` explicitly (e.g. in a `.meta` file or Rivet log) and use it in `inference/IO.py` for $\mathcal{L} \cdot \sigma / N_{\text{gen}}$.

### 2. Force Hadronic $\tau$ Decays in Pythia for `top` & `DYbb` (2.4× Speedup)
Inclusive $\tau$ decays have $\text{BR}(\tau \to \text{hadrons}) \approx 64.8\%$, so **58%** of $\tau^+\tau^-$ events contain at least one leptonic $\tau \to e/\mu$ and are discarded by `n_tau == 2` downstream.
- Disabling leptonic $\tau$ decays in Pythia (and scaling `weight_nom` by $\text{BR}(\tau \to \text{hadrons})^2 \approx 0.420$) yields **2.4× more selected events per CPU-hour**:
  - $100\text{ fb}^{-1}$ ($5\times L$): **~4M** hadronic-$\tau$ events each for `top` and `DYbb` (instead of ~9–10M).
  - $300\text{ fb}^{-1}$ ($5\times L$): **~11M** hadronic-$\tau$ events each (instead of ~27M).
- *(Skip this only if studying $\tau \to \ell$ fake-tau backgrounds.)*

### 3. Generator-Level Cuts for `DYbb` (2–3× Selection Boost)
`DYbb` currently has only **1.68%** offline selection efficiency (`8,395 / 500,000`) due to soft/collinear $g \to b\bar{b}$ (`ptb 10`, `mmbb 50`, `drbb 0`, `drbl 0`).
- If generating a **fresh** `DYbb` sample (not appending to the existing 500k), tighten `rivet/DYbb.MG` to:
  - `ptb 15` (offline jet $p_T > 20\text{ GeV}$)
  - `mmbb 65` (offline $75 < m_{bb} < 150\text{ GeV}$)
  - `drbb 0.3`, `drbl 0.3` (offline anti-$k_T$ $R = 0.4$ resolved jets)
- If **appending** to the existing `500k` `DYbb.csv`, keep the original `DYbb.MG` cuts so all batches share the exact same fiducial cross section.

### 4. Dedicated `HH-rw` Sample at $\kappa_\lambda = 5$
Instead of generating 2–4× more events at $\kappa_\lambda = 1$, generate a **500k-event `HH-rw` sample anchored at $\kappa_\lambda = 5$** (`set BSMINPUTS 1 5.0` in `loop_sm_hh`) and reweight to $\kappa_\lambda \in \{0, 1, 2, 10\}$.
- Directly populates the high-$m_{HH}$ / high-$\kappa_\lambda$ phase space with much lower weight variance than $\kappa_\lambda=1$ reweighting alone.
- Combine with the existing $\kappa_\lambda=1$ sample using inverse-variance / cross-section-weighted mixing.

---

## Execution Plan (Ordered by Priority)

Split MadGraph runs into batches of $\le 500\text{k}$–$1\text{M}$ events (MG5 caps `nevents` at $1{,}000{,}000$ per `launch`, and batching enables parallel Pythia+Rivet `main144` execution across all available CPU cores).

1. **`ZH` (100k events)** — *Quick win (~10–15 min on 6 cores, faster on beefier hardware)*
   - Run `rivet/ZH.MG` with `nevents 100000` (covers both $100$ and $300\text{ fb}^{-1}$).
2. **`tautaujets.cc` + `IO.py` update**
   - Add upstream `>= 2 b, >= 2 tau` preselection and explicit `N_gen` metadata tracking.
   - Optionally add hadronic-$\tau$ Pythia command card for `top` and `DYbb`.
3. **`DYbb` Phase 1 ($100\text{ fb}^{-1}$)**
   - Generate `+8.5M` inclusive (or `~4M` hadronic-$\tau$) events in batches of `500k`–`1M` with distinct `iseed` values.
   - Run `main144` in parallel across all performance cores, then merge with unique `eventnumber` offsets.
4. **`top` Phase 1 ($100\text{ fb}^{-1}$)**
   - Generate `+7M` inclusive (or `~3M` hadronic-$\tau$) additional events on top of the existing `2.5M` events (`rivet/top.csv`), using `rivet/top_2M.MG` as a template with fresh `iseed` values.
5. **`HH-rw` ($\kappa_\lambda = 5$ anchor, 500k events)**
   - Clone `rivet/HH.MG` with `set BSMINPUTS 1 5.0` and reweight card for `kl0`, `kl1`, `kl2`, `kl10`.
6. **`DYbb` & `top` Phase 2 ($300\text{ fb}^{-1}$)**
   - Extend both to `~11M` (minimum $2\times L$ / or full $5\times L$ if using hadronic-$\tau$ forcing) or `~27M` inclusive.

---

## Reference Benchmarks (on 8-core Apple Silicon, 6 cores used)
- **MadGraph (`top`, 6-body tree-level):** ~13–15 min per `333k` events (`~1.4M events/hr` on 6 cores).
- **MadGraph (`HH-rw`, loop-induced + reweighting + MadSpin):** ~1.5 hr generation + ~30 min reweighting (`kl0`, `kl2`) + ~5 min MadSpin for `500k` events.
- **Pythia 8.318 + Rivet 4.1.5 (`main144`):** single-threaded, `~2,700 events/min` per core (`~1M events/hr` across 6 cores).

---

## MadGraph / Environment Notes for Reproducing `HH-rw`

If setting up `mg5amcnlo-3.6.4` on another machine, the following fixes were required for `HH-rw` generation, reweighting, and MadSpin to run cleanly:

1. **Python environment (`six` + `f2py`):** MadGraph requires `six`, and **reweighting requires `f2py`** (`numpy`) on `PATH` (otherwise `reweight` is marked `Not Avail.`). Run `mg5_aMC` with a Python environment that has both `six` and `f2py` (e.g., Python 3.9 + `numpy 1.26.x`).
2. **`loop_sm_hh` model + Python 3 `write_param_card.py`:** Copy the `loop_sm_hh` UFO model (which exposes $\kappa_\lambda$ as `BSMINPUTS 1`) into `$MGPATH/models/loop_sm_hh`, and replace its `write_param_card.py` with the one from `$MGPATH/models/loop_sm/write_param_card.py` (fixes `ImportError: attempted relative import with no known parent package` in Python 3).
3. **`Template/LO/Source/dsample.f` (MG5 3.6.4 Fortran `FORMAT` bug):** Lines 603–604 are missing commas between `i12` descriptors and `','` literals (`a15,i12,',',i12',',i12...`), which newer `gfortran` rejects at runtime during MadLoop survey. Add the missing commas (`',',i12,',',i12...`).
4. **Coupling-order syntax check (`acknowledged_v3.1_syntax`):** Reweighting `g g > h h QED<=2 [noborn=QCD]` requires `set acknowledged_v3.1_syntax True` (already in `rivet/HH.MG`, or run `set acknowledged_v3.1_syntax True --global` once in `mg5_aMC`).
5. **MadSpin on loop-induced `gg -> hh`:** Requires `set spinmode none` in `madspin_card.dat` (otherwise `decay_events` fails with `InvalidCmd: not supported format standalone_msP`); already configured in `rivet/HH-rw_madspin_card.dat`.
6. **macOS only:** `rivet/setup.sh` exports `SDKROOT`, `CPLUS_INCLUDE_PATH`, and `LIBRARY_PATH` so `gfortran` can resolve `-lc++` when linking MadLoop.

