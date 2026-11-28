"""Physics model, quadratic kappa_lambda reweighting, extensible nuisances, and sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .config import NuisanceConfig, PriorConfig
from .data import FeatureStats, SplitDataset, featurize_raw_jets


def hh_reweight_coefficients(kappa_lambda: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return quadratic Lagrange weights (c_nom [kl=1], c_kl0 [kl=0], c_kl2 [kl=2])."""
    kl = np.asarray(kappa_lambda, dtype=np.float64)
    c_kl0 = 0.5 * (kl - 1.0) * (kl - 2.0)
    c_nom = kl * (2.0 - kl)
    c_kl2 = 0.5 * kl * (kl - 1.0)
    return c_nom, c_kl0, c_kl2


def compute_hh_weights(
    weights_3col: np.ndarray,
    mu_hh: np.ndarray | float,
    kappa_lambda: np.ndarray | float,
    clamp_negative: bool = True,
) -> np.ndarray:
    """Compute exact quadratic HH event weights w_i(mu_HH, kappa_lambda).

    Args:
        weights_3col: Array of shape (..., 3) with columns [weight_nom, weight_kl0, weight_kl2].
        mu_hh: Signal strength multiplier(s), broadcastable against weights_3col[..., 0].
        kappa_lambda: Higgs trilinear coupling modifier(s).
        clamp_negative: Clamp tiny numerical negative values to 0.0 if any arise.
    """
    w = np.asarray(weights_3col, dtype=np.float64)
    mu = np.asarray(mu_hh, dtype=np.float64)
    c_nom, c_kl0, c_kl2 = hh_reweight_coefficients(kappa_lambda)
    combined = mu * (
        w[..., 0] * c_nom
        + w[..., 1] * c_kl0
        + w[..., 2] * c_kl2
    )
    if clamp_negative:
        combined = np.maximum(combined, 0.0)
    return combined


class UniformPrior2D:
    """Uniform 2D prior over theta = (mu_HH, kappa_lambda) with standardization helpers."""

    def __init__(self, cfg: PriorConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.low = np.array([cfg.mu_min, cfg.kl_min], dtype=np.float64)
        self.high = np.array([cfg.mu_max, cfg.kl_max], dtype=np.float64)
        self.ranges = self.high - self.low
        self.mean = 0.5 * (self.low + self.high)
        self.std = self.ranges / np.sqrt(12.0)
        self.log_std_sum = float(np.sum(np.log(self.std)))
        self.prior_nll = float(np.sum(np.log(self.ranges)))

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        mu_max_override: float | None = None,
    ) -> np.ndarray:
        """Draw n samples of (mu_HH, kappa_lambda) in physical units, shape (n, 2)."""
        high = self.high.copy()
        if mu_max_override is not None:
            high[0] = float(mu_max_override)
        u = rng.uniform(0.0, 1.0, size=(n, 2))
        return (self.low + u * (high - self.low)).astype(np.float32)

    def standardize_np(self, theta: np.ndarray) -> np.ndarray:
        return ((theta.astype(np.float64) - self.mean) / self.std).astype(np.float32)

    def unstandardize_np(self, theta_std: np.ndarray) -> np.ndarray:
        return (theta_std.astype(np.float64) * self.std + self.mean).astype(np.float32)

    def in_bounds(self, theta: np.ndarray) -> np.ndarray:
        return np.all((theta >= self.low) & (theta <= self.high), axis=-1)


@dataclass
class NuisanceSample:
    """Container for set-level nuisance parameters psi (empty in stat-only mode)."""

    values: dict[str, np.ndarray]

    def slice(self, idx: int) -> dict[str, float]:
        return {k: float(v[idx]) for k, v in self.values.items()}


class NuisanceModel(ABC):
    """Extensible interface for set-level nuisance parameters psi.

    Future nuisances (e.g. per-background normalization uncertainties, JES/JER energy-scale
    shifts affecting jet momenta and acceptance) can subclass NuisanceModel and plug into
    DatasetSimulator without altering the PAIRS embedder, aggregator, or normalizing flow.
    """

    @abstractmethod
    def sample_prior(self, n_sets: int, rng: np.random.Generator) -> NuisanceSample:
        """Sample n_sets independent nuisance realizations psi."""

    @abstractmethod
    def process_rate_multiplier(
        self,
        process_name: str,
        psi: dict[str, float] | None,
    ) -> float:
        """Return multiplicative rate scaling induced by psi for a given process."""

    @abstractmethod
    def has_shape_nuisances(self) -> bool:
        """Return True if psi modifies per-event features or relative event weights."""


class StatOnlyNuisanceModel(NuisanceModel):
    """Stat-only inference (no active nuisance parameters)."""

    def sample_prior(self, n_sets: int, rng: np.random.Generator) -> NuisanceSample:
        return NuisanceSample(values={})

    def process_rate_multiplier(
        self,
        process_name: str,
        psi: dict[str, float] | None,
    ) -> float:
        return 1.0

    def has_shape_nuisances(self) -> bool:
        return False


class LogNormalNormNuisanceModel(NuisanceModel):
    """Log-normal normalization nuisances psi_p = exp(sigma_p * z_p), z_p ~ N(0, 1)."""

    def __init__(self, log_sigmas: dict[str, float]) -> None:
        self.log_sigmas = {str(k): float(v) for k, v in log_sigmas.items()}

    def sample_prior(self, n_sets: int, rng: np.random.Generator) -> NuisanceSample:
        values: dict[str, np.ndarray] = {}
        for pname, sigma in self.log_sigmas.items():
            if sigma > 0.0:
                z = rng.normal(0.0, 1.0, size=n_sets)
                values[pname] = np.exp(sigma * z).astype(np.float64)
            else:
                values[pname] = np.ones(n_sets, dtype=np.float64)
        return NuisanceSample(values=values)

    def process_rate_multiplier(
        self,
        process_name: str,
        psi: dict[str, float] | None,
    ) -> float:
        if psi is None or process_name not in psi:
            return 1.0
        return float(psi[process_name])

    def has_shape_nuisances(self) -> bool:
        return False


def apply_shape_nuisance_to_raw_jets(
    raw_jets: np.ndarray,
    jet_mask: np.ndarray,
    jes_scale: np.ndarray | float,
    tes_scale: np.ndarray | float,
    jer_sigma: np.ndarray | float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply JES (non-tau jets), TES (tau jets), and optional JER smearing to raw_jets (N, J, 6)."""
    out = raw_jets.copy()
    is_b = jet_mask & (out[..., 0] == 1.0)
    is_tau = jet_mask & (out[..., 2] == 1.0) & (~is_b)

    jes = np.asarray(jes_scale, dtype=np.float32)
    tes = np.asarray(tes_scale, dtype=np.float32)
    if jes.ndim == 1:
        jes = jes[:, None]
    if tes.ndim == 1:
        tes = tes[:, None]

    scale = np.where(is_tau, tes, jes)
    jer = np.asarray(jer_sigma, dtype=np.float32)
    if rng is not None and np.any(jer > 0.0):
        if jer.ndim == 1:
            jer = jer[:, None]
        smear = np.maximum(0.0, 1.0 + jer * rng.normal(size=scale.shape).astype(np.float32))
        scale = scale * smear

    scale = scale * jet_mask.astype(np.float32)
    out[..., 3:6] = out[..., 3:6] * scale[..., None]
    return out


class CombinedNormAndShapeNuisanceModel(LogNormalNormNuisanceModel):
    """Tier 1 log-normal normalizations + Tier 2 JES/TES/JER shape nuisances."""

    def __init__(
        self,
        log_sigmas: dict[str, float],
        jes_sigma: float = 0.03,
        tes_sigma: float = 0.03,
        jer_sigma: float = 0.05,
    ) -> None:
        super().__init__(log_sigmas)
        self.jes_sigma = float(jes_sigma)
        self.tes_sigma = float(tes_sigma)
        self.jer_sigma = float(jer_sigma)

    def sample_prior(self, n_sets: int, rng: np.random.Generator) -> NuisanceSample:
        sample = super().sample_prior(n_sets, rng)
        z_jes = rng.normal(0.0, 1.0, size=n_sets)
        z_tes = rng.normal(0.0, 1.0, size=n_sets)
        z_jer = rng.normal(0.0, 1.0, size=n_sets)
        sample.values["jes_scale"] = np.exp(self.jes_sigma * z_jes).astype(np.float64)
        sample.values["tes_scale"] = np.exp(self.tes_sigma * z_tes).astype(np.float64)
        sample.values["jer_scale"] = (self.jer_sigma * np.abs(z_jer)).astype(np.float64)
        return sample

    def has_shape_nuisances(self) -> bool:
        return True


def build_nuisance_model(cfg: NuisanceConfig) -> NuisanceModel:
    """Instantiate the configured NuisanceModel."""
    if cfg.mode == "stat_only":
        return StatOnlyNuisanceModel()
    if cfg.mode == "norm":
        return LogNormalNormNuisanceModel(cfg.log_sigmas)
    if cfg.mode == "norm_shape":
        return CombinedNormAndShapeNuisanceModel(
            cfg.log_sigmas,
            jes_sigma=cfg.jes_sigma,
            tes_sigma=cfg.tes_sigma,
            jer_sigma=cfg.jer_sigma,
        )
    raise ValueError(f"Unknown nuisance mode: {cfg.mode}")


class ProcessSampler:
    """Vectorized event sampler for a single split (train, val, or test)."""

    def __init__(
        self,
        split: SplitDataset,
        nuisance_model: NuisanceModel | None = None,
        hh_k_factor: float = 1.0,
        feature_stats: FeatureStats | None = None,
    ) -> None:
        self.split = split
        self.nuisance_model = nuisance_model or StatOnlyNuisanceModel()
        self.hh_k_factor = float(hh_k_factor)
        self.feature_stats = feature_stats
        self.backgrounds = split.backgrounds
        self.signal = split.signal

        # Precompute background yields and verify constant-weight optimization
        self.bkg_yields: dict[str, float] = {}
        self.bkg_uniform: dict[str, bool] = {}
        self.bkg_cdfs: dict[str, np.ndarray] = {}
        for b in self.backgrounds:
            pool = split.processes[b]
            w = pool.weights[:, 0]
            self.bkg_yields[b] = float(w.sum())
            is_unif = bool(np.allclose(w, w[0], rtol=1e-6, atol=0.0))
            self.bkg_uniform[b] = is_unif
            if not is_unif:
                cdf = np.cumsum(w)
                self.bkg_cdfs[b] = cdf / cdf[-1]

        sig_pool = split.processes[self.signal]
        self.hh_weights = sig_pool.weights * self.hh_k_factor  # (M_hh, 3) float64: nom, kl0, kl2
        self.hh_yield_sums = self.hh_weights.sum(axis=0)  # [sum_nom, sum_kl0, sum_kl2]
        # Fixed envelope proposal for importance-weighted Stage 1
        self.hh_env_weights = np.maximum(np.max(np.abs(self.hh_weights), axis=1), 1e-12)
        self.hh_env_sum = float(self.hh_env_weights.sum())
        self.hh_env_cdf = np.cumsum(self.hh_env_weights) / self.hh_env_sum

    def total_bkg_yield(self, psi: dict[str, float] | None = None) -> float:
        return sum(
            self.bkg_yields[b] * self.nuisance_model.process_rate_multiplier(b, psi)
            for b in self.backgrounds
        )

    def hh_expected_yield(
        self,
        mu_hh: np.ndarray | float,
        kappa_lambda: np.ndarray | float,
        psi: dict[str, float] | None = None,
    ) -> np.ndarray:
        c_nom, c_kl0, c_kl2 = hh_reweight_coefficients(kappa_lambda)
        rate = self.nuisance_model.process_rate_multiplier(self.signal, psi)
        return (
            np.asarray(mu_hh, dtype=np.float64)
            * rate
            * (
                self.hh_yield_sums[0] * c_nom
                + self.hh_yield_sums[1] * c_kl0
                + self.hh_yield_sums[2] * c_kl2
            )
        )

    def _draw_process_indices(
        self,
        proc_name: str,
        n_draw: int,
        kappa_lambda: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if n_draw <= 0:
            return np.empty(0, dtype=np.int64)
        pool = self.split.processes[proc_name]
        if proc_name == self.signal:
            w = compute_hh_weights(self.hh_weights, mu_hh=1.0, kappa_lambda=kappa_lambda)
            w_sum = w.sum()
            if w_sum <= 0.0:
                return rng.integers(0, pool.num_events, size=n_draw, dtype=np.int64)
            cdf = np.cumsum(w) / w_sum
            return np.searchsorted(cdf, rng.uniform(0.0, 1.0, size=n_draw), side="right").clip(
                0, pool.num_events - 1
            )
        if self.bkg_uniform[proc_name]:
            return rng.integers(0, pool.num_events, size=n_draw, dtype=np.int64)
        cdf = self.bkg_cdfs[proc_name]
        return np.searchsorted(cdf, rng.uniform(0.0, 1.0, size=n_draw), side="right").clip(
            0, pool.num_events - 1
        )

    def sample_small_sets_batch(
        self,
        theta: np.ndarray,
        set_sizes: np.ndarray,
        rng: np.random.Generator,
        psi: NuisanceSample | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of sets of small cardinality N in {1, 2} for Stage 1.

        Returns:
            jet_feats: (B, N_pad, max_jets, F_jet) float32
            jet_mask:  (B, N_pad, max_jets) bool
            evt_feats: (B, N_pad, F_evt) float32
        """
        batch_size = int(theta.shape[0])
        max_n = int(set_sizes.max())

        # Compute per-process probabilities for each set in the batch
        proc_names = list(self.backgrounds) + [self.signal]
        n_procs = len(proc_names)
        yields = np.empty((batch_size, n_procs), dtype=np.float64)

        for idx, b in enumerate(self.backgrounds):
            if psi is not None and psi.values:
                mults = np.array(
                    [self.nuisance_model.process_rate_multiplier(b, psi.slice(i)) for i in range(batch_size)],
                    dtype=np.float64,
                )
            else:
                mults = 1.0
            yields[:, idx] = self.bkg_yields[b] * mults

        hh_y = self.hh_expected_yield(theta[:, 0], theta[:, 1])
        if psi is not None and psi.values:
            hh_mults = np.array(
                [self.nuisance_model.process_rate_multiplier(self.signal, psi.slice(i)) for i in range(batch_size)],
                dtype=np.float64,
            )
            hh_y = hh_y * hh_mults
        yields[:, -1] = np.maximum(hh_y, 0.0)

        probs = yields / yields.sum(axis=1, keepdims=True)
        cdfs = np.cumsum(probs, axis=1)
        u = rng.uniform(0.0, 1.0, size=(batch_size, max_n))
        # Which process each slot (b, r) belongs to
        proc_ids = (u[..., None] >= cdfs[:, None, :]).sum(axis=-1).clip(0, n_procs - 1)

        sample_pool = self.split.processes[self.signal]
        max_jets = sample_pool.jet_features.shape[1]
        f_jet = sample_pool.jet_features.shape[2]
        f_evt = sample_pool.evt_features.shape[1]

        out_jets = np.zeros((batch_size, max_n, max_jets, f_jet), dtype=np.float32)
        out_jmask = np.zeros((batch_size, max_n, max_jets), dtype=bool)
        out_evts = np.zeros((batch_size, max_n, f_evt), dtype=np.float32)

        slot_valid = np.arange(max_n)[None, :] < set_sizes[:, None]

        # Sample background slots in bulk per process
        for pid, b in enumerate(self.backgrounds):
            mask_pb = slot_valid & (proc_ids == pid)
            rows, cols = np.nonzero(mask_pb)
            if len(rows) == 0:
                continue
            ev_idx = self._draw_process_indices(b, len(rows), 1.0, rng)
            pool = self.split.processes[b]
            out_jets[rows, cols] = pool.jet_features[ev_idx]
            out_jmask[rows, cols] = pool.jet_mask[ev_idx]
            out_evts[rows, cols] = pool.evt_features[ev_idx]

        # Sample HH slots (weighted by kappa_lambda for each specific set)
        hh_pid = n_procs - 1
        mask_hh = slot_valid & (proc_ids == hh_pid)
        hh_rows, hh_cols = np.nonzero(mask_hh)
        if len(hh_rows) > 0:
            # Vectorize HH weight evaluation over unique batch rows needing HH events
            kls = theta[hh_rows, 1]
            c_nom, c_kl0, c_kl2 = hh_reweight_coefficients(kls)
            hh_pool = self.split.processes[self.signal]
            # Process in chunks to keep temporary memory bounded
            chunk_size = 64
            ev_indices = np.empty(len(hh_rows), dtype=np.int64)
            for start in range(0, len(hh_rows), chunk_size):
                end = min(start + chunk_size, len(hh_rows))
                w_chunk = (
                    c_nom[start:end, None] * self.hh_weights[None, :, 0]
                    + c_kl0[start:end, None] * self.hh_weights[None, :, 1]
                    + c_kl2[start:end, None] * self.hh_weights[None, :, 2]
                )
                w_chunk = np.maximum(w_chunk, 0.0)
                w_sums = w_chunk.sum(axis=1, keepdims=True)
                # Guard against degenerate zero-sum weights
                zero_mask = w_sums[:, 0] <= 0.0
                if np.any(zero_mask):
                    w_chunk[zero_mask] = 1.0
                    w_sums[zero_mask] = float(hh_pool.num_events)
                cdf_chunk = np.cumsum(w_chunk / w_sums, axis=1)
                u_chunk = rng.uniform(0.0, 1.0, size=(end - start, 1))
                ev_indices[start:end] = (u_chunk >= cdf_chunk).sum(axis=1).clip(0, hh_pool.num_events - 1)

            out_jets[hh_rows, hh_cols] = hh_pool.jet_features[ev_indices]
            out_jmask[hh_rows, hh_cols] = hh_pool.jet_mask[ev_indices]
            out_evts[hh_rows, hh_cols] = hh_pool.evt_features[ev_indices]

        return (
            torch.from_numpy(out_jets),
            torch.from_numpy(out_jmask),
            torch.from_numpy(out_evts),
        )

    def sample_small_sets_importance_weighted(
        self,
        theta: np.ndarray,
        set_sizes: np.ndarray,
        rng: np.random.Generator,
        psi: NuisanceSample | None = None,
        sig_proposal_frac: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample Stage 1 sets from a fixed 50/50 bkg/HH proposal q_0 and return exact IW weights.

        Eliminates per-theta CDF resampling and supports signed event weights.
        Returns (jet_feats, jet_mask, evt_feats, iw_weights) with iw_weights shape (B,).
        """
        batch_size = int(theta.shape[0])
        max_n = int(set_sizes.max())

        # Compute total expected yield Lambda(theta, psi) per set
        total_bkg_nom = self.total_bkg_yield(None)
        bkg_yields_mat = np.empty((batch_size, len(self.backgrounds)), dtype=np.float64)
        for idx, b in enumerate(self.backgrounds):
            if psi is not None and psi.values:
                mults = np.array(
                    [self.nuisance_model.process_rate_multiplier(b, psi.slice(i)) for i in range(batch_size)],
                    dtype=np.float64,
                )
            else:
                mults = 1.0
            bkg_yields_mat[:, idx] = self.bkg_yields[b] * mults

        hh_y = self.hh_expected_yield(theta[:, 0], theta[:, 1])
        if psi is not None and psi.values:
            hh_mults = np.array(
                [self.nuisance_model.process_rate_multiplier(self.signal, psi.slice(i)) for i in range(batch_size)],
                dtype=np.float64,
            )
        else:
            hh_mults = np.ones(batch_size, dtype=np.float64)
        hh_y = hh_y * hh_mults
        lam_total = np.maximum(bkg_yields_mat.sum(axis=1) + hh_y, 1e-12)

        sample_pool = self.split.processes[self.signal]
        max_jets = sample_pool.jet_features.shape[1]
        f_jet = sample_pool.jet_features.shape[2]
        f_evt = sample_pool.evt_features.shape[1]

        out_jets = np.zeros((batch_size, max_n, max_jets, f_jet), dtype=np.float32)
        out_jmask = np.zeros((batch_size, max_n, max_jets), dtype=bool)
        out_evts = np.zeros((batch_size, max_n, f_evt), dtype=np.float32)
        use_shape = (
            self.nuisance_model.has_shape_nuisances()
            and self.feature_stats is not None
            and psi is not None
            and "jes_scale" in psi.values
        )
        out_raw = np.zeros((batch_size, max_n, max_jets, 6), dtype=np.float32) if use_shape else None

        slot_valid = np.arange(max_n)[None, :] < set_sizes[:, None]
        is_sig = rng.uniform(0.0, 1.0, size=(batch_size, max_n)) < sig_proposal_frac
        slot_ratios = np.ones((batch_size, max_n), dtype=np.float64)

        # Background slots drawn from fixed nominal background mixture q_0
        bkg_probs = np.array([self.bkg_yields[b] for b in self.backgrounds], dtype=np.float64)
        bkg_probs = bkg_probs / max(bkg_probs.sum(), 1e-12)
        bkg_cdf = np.cumsum(bkg_probs)

        bkg_rows, bkg_cols = np.nonzero(slot_valid & (~is_sig))
        if len(bkg_rows) > 0:
            pids = np.searchsorted(bkg_cdf, rng.uniform(0.0, 1.0, size=len(bkg_rows)), side="right").clip(
                0, len(self.backgrounds) - 1
            )
            for pid, b in enumerate(self.backgrounds):
                sel = pids == pid
                if not np.any(sel):
                    continue
                r_sub = bkg_rows[sel]
                c_sub = bkg_cols[sel]
                ev_idx = self._draw_process_indices(b, len(r_sub), 1.0, rng)
                pool = self.split.processes[b]
                out_jmask[r_sub, c_sub] = pool.jet_mask[ev_idx]
                if use_shape:
                    out_raw[r_sub, c_sub] = pool.raw_jets[ev_idx]
                else:
                    out_jets[r_sub, c_sub] = pool.jet_features[ev_idx]
                    out_evts[r_sub, c_sub] = pool.evt_features[ev_idx]
                mult_b = bkg_yields_mat[r_sub, pid] / max(self.bkg_yields[b], 1e-12)
                slot_ratios[r_sub, c_sub] = (mult_b * total_bkg_nom) / (
                    (1.0 - sig_proposal_frac) * lam_total[r_sub]
                )

        # Signal slots drawn from fixed HH envelope q_0
        sig_rows, sig_cols = np.nonzero(slot_valid & is_sig)
        if len(sig_rows) > 0:
            ev_idx = np.searchsorted(
                self.hh_env_cdf, rng.uniform(0.0, 1.0, size=len(sig_rows)), side="right"
            ).clip(0, sample_pool.num_events - 1)
            out_jmask[sig_rows, sig_cols] = sample_pool.jet_mask[ev_idx]
            if use_shape:
                out_raw[sig_rows, sig_cols] = sample_pool.raw_jets[ev_idx]
            else:
                out_jets[sig_rows, sig_cols] = sample_pool.jet_features[ev_idx]
                out_evts[sig_rows, sig_cols] = sample_pool.evt_features[ev_idx]

            w_target = compute_hh_weights(
                self.hh_weights[ev_idx],
                mu_hh=theta[sig_rows, 0],
                kappa_lambda=theta[sig_rows, 1],
                clamp_negative=False,
            ) * hh_mults[sig_rows]
            slot_ratios[sig_rows, sig_cols] = (w_target * self.hh_env_sum) / (
                sig_proposal_frac * self.hh_env_weights[ev_idx] * lam_total[sig_rows]
            )

        if use_shape and out_raw is not None:
            v_r, v_c = np.nonzero(slot_valid)
            shifted_raw = apply_shape_nuisance_to_raw_jets(
                out_raw[v_r, v_c],
                out_jmask[v_r, v_c],
                jes_scale=psi.values["jes_scale"][v_r],
                tes_scale=psi.values["tes_scale"][v_r],
                jer_sigma=psi.values["jer_scale"][v_r],
                rng=rng,
            )
            jf_s, ef_s, pass_w = featurize_raw_jets(shifted_raw, out_jmask[v_r, v_c], self.feature_stats)
            out_jets[v_r, v_c] = jf_s
            out_evts[v_r, v_c] = ef_s
            # Events migrating outside the mass window under JES/TES/JER have zero acceptance
            slot_ratios[v_r, v_c] = np.where(pass_w, slot_ratios[v_r, v_c], 0.0)

        iw_weights = np.prod(slot_ratios, axis=1).astype(np.float32)
        return (
            torch.from_numpy(out_jets),
            torch.from_numpy(out_jmask),
            torch.from_numpy(out_evts),
            torch.from_numpy(iw_weights),
        )

    def sample_aggregated_datasets(
        self,
        embeddings: dict[str, np.ndarray],
        prior: UniformPrior2D,
        n_datasets: int,
        rng: np.random.Generator,
        fixed_thetas: np.ndarray | None = None,
        mc_bootstrap: bool = False,
    ) -> dict[str, np.ndarray]:
        """Generate Stage 2/3 Poisson-sampled dataset-level mean embeddings T_bar and counts N.

        Uses exact Poisson thinning per process: N_p ~ Poisson(Lambda_p(theta, psi)),
        N = sum_p N_p ~ Poisson(Lambda_tot), and T_bar = (1/N) * sum_{p} sum_{r=1}^{N_p} e_{i_r},
        accumulated in float64 to preserve precision over N ~ 16,500 events.

        When mc_bootstrap=True, draws per-MC-event template multiplicities m_i ~ Poisson(1)
        once per dataset (exact unbinned Barlow-Beeston MC-stat marginalization).
        """
        if fixed_thetas is not None:
            thetas = np.asarray(fixed_thetas, dtype=np.float32)
            n_datasets = int(thetas.shape[0])
        else:
            thetas = prior.sample(n_datasets, rng)

        psi_sample = self.nuisance_model.sample_prior(n_datasets, rng)
        embed_dim = int(next(iter(embeddings.values())).shape[1])

        t_bars = np.zeros((n_datasets, embed_dim), dtype=np.float32)
        counts = np.zeros(n_datasets, dtype=np.int64)
        hh_counts = np.zeros(n_datasets, dtype=np.int64)

        # Convert embeddings once to float64 for precise large-N summation
        emb_f64 = {k: np.asarray(v, dtype=np.float64) for k, v in embeddings.items()}

        # Precompute HH base CDFs? For each dataset we draw N_hh ~ Poisson(lambda_hh)
        # Note: N_hh is small (0 to ~1500), while N_bkg ~ 16,400 is huge and uniform per process.
        for j in range(n_datasets):
            psi_j = psi_sample.slice(j) if psi_sample.values else None
            sum_emb = np.zeros(embed_dim, dtype=np.float64)
            n_tot = 0

            for b in self.backgrounds:
                mult_b = self.nuisance_model.process_rate_multiplier(b, psi_j)
                pool_size = emb_f64[b].shape[0]
                if mc_bootstrap:
                    m_mc = rng.poisson(1.0, size=pool_size).astype(np.float64)
                    w_eff = m_mc * self.split.processes[b].weights[:, 0] * mult_b
                    lam_b = float(w_eff.sum())
                    n_b = int(rng.poisson(max(lam_b, 0.0)))
                    if n_b > 0 and lam_b > 0.0:
                        mult = rng.multinomial(n_b, w_eff / lam_b).astype(np.float64)
                        nz = mult > 0
                        sum_emb += mult[nz] @ emb_f64[b][nz]
                        n_tot += n_b
                else:
                    lam_b = self.bkg_yields[b] * mult_b
                    n_b = int(rng.poisson(max(lam_b, 0.0)))
                    if n_b > 0:
                        idx = self._draw_process_indices(b, n_b, 1.0, rng)
                        # bincount trick avoids huge gather when n_b > pool size
                        mult = np.bincount(idx, minlength=pool_size).astype(np.float64)
                        nz = mult > 0
                        sum_emb += mult[nz] @ emb_f64[b][nz]
                        n_tot += n_b

            mu_j = float(thetas[j, 0])
            kl_j = float(thetas[j, 1])
            sig_pool_size = emb_f64[self.signal].shape[0]
            if mc_bootstrap:
                mult_s = self.nuisance_model.process_rate_multiplier(self.signal, psi_j)
                m_mc = rng.poisson(1.0, size=sig_pool_size).astype(np.float64)
                w_eff = m_mc * compute_hh_weights(self.hh_weights, mu_j, kl_j) * mult_s
                lam_hh = float(w_eff.sum())
                n_hh = int(rng.poisson(max(lam_hh, 0.0)))
                hh_counts[j] = n_hh
                if n_hh > 0 and lam_hh > 0.0:
                    mult = rng.multinomial(n_hh, w_eff / lam_hh).astype(np.float64)
                    nz = mult > 0
                    sum_emb += mult[nz] @ emb_f64[self.signal][nz]
                    n_tot += n_hh
            else:
                lam_hh = float(self.hh_expected_yield(mu_j, kl_j, psi_j))
                n_hh = int(rng.poisson(max(lam_hh, 0.0)))
                hh_counts[j] = n_hh
                if n_hh > 0:
                    idx_hh = self._draw_process_indices(self.signal, n_hh, kl_j, rng)
                    mult = np.bincount(idx_hh, minlength=sig_pool_size).astype(np.float64)
                    nz = mult > 0
                    sum_emb += mult[nz] @ emb_f64[self.signal][nz]
                    n_tot += n_hh

            counts[j] = n_tot
            if n_tot > 0:
                t_bars[j] = (sum_emb / float(n_tot)).astype(np.float32)

        return {
            "theta": thetas.astype(np.float32),
            "T_bar": t_bars,
            "N": counts,
            "N_hh": hh_counts,
        }

    def sample_aggregated_datasets_with_shape(
        self,
        shape_basis: dict[str, dict[str, np.ndarray]],
        prior: UniformPrior2D,
        n_datasets: int,
        rng: np.random.Generator,
        jes_sigma: float,
        tes_sigma: float,
        fixed_thetas: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate Stage 2/3 aggregates with continuous JES/TES shape & acceptance shifts.

        `shape_basis[proc]` holds `emb_0`, `d_jes`, `d_tes` (each `(M_p, D)`) and
        `pass_0`, `pass_jes_p`, `pass_jes_m`, `pass_tes_p`, `pass_tes_m` (`(M_p,)` bool),
        computed at +/-1 sigma of JES and TES.
        """
        if fixed_thetas is not None:
            thetas = np.asarray(fixed_thetas, dtype=np.float32)
            n_datasets = int(thetas.shape[0])
        else:
            thetas = prior.sample(n_datasets, rng)

        psi_sample = self.nuisance_model.sample_prior(n_datasets, rng)
        first_proc = next(iter(shape_basis.values()))
        embed_dim = int(first_proc["emb_0"].shape[1])

        t_bars = np.zeros((n_datasets, embed_dim), dtype=np.float32)
        counts = np.zeros(n_datasets, dtype=np.int64)
        hh_counts = np.zeros(n_datasets, dtype=np.int64)

        for j in range(n_datasets):
            psi_j = psi_sample.slice(j)
            d_jes_norm = (psi_j["jes_scale"] - 1.0) / max(jes_sigma, 1e-6)
            d_tes_norm = (psi_j["tes_scale"] - 1.0) / max(tes_sigma, 1e-6)
            sum_emb = np.zeros(embed_dim, dtype=np.float64)
            n_tot = 0

            def _acceptance_factor(sb: dict[str, np.ndarray]) -> np.ndarray:
                # Smooth linear interpolation of mass-window pass probability around nominal
                p_j = sb["pass_jes_p"] if d_jes_norm >= 0.0 else sb["pass_jes_m"]
                p_t = sb["pass_tes_p"] if d_tes_norm >= 0.0 else sb["pass_tes_m"]
                acc = sb["pass_0"] + abs(d_jes_norm) * (p_j - sb["pass_0"]) + abs(d_tes_norm) * (p_t - sb["pass_0"])
                return np.clip(acc, 0.0, 1.0)

            for b in self.backgrounds:
                sb = shape_basis[b]
                acc = _acceptance_factor(sb)
                w_eff = self.split.processes[b].weights[:, 0] * acc * self.nuisance_model.process_rate_multiplier(b, psi_j)
                lam_b = float(w_eff.sum())
                n_b = int(rng.poisson(max(lam_b, 0.0)))
                if n_b > 0 and lam_b > 0.0:
                    mult = rng.multinomial(n_b, w_eff / lam_b).astype(np.float64)
                    nz = mult > 0
                    emb_shifted = (
                        sb["emb_0"][nz]
                        + d_jes_norm * sb["d_jes"][nz]
                        + d_tes_norm * sb["d_tes"][nz]
                    )
                    sum_emb += mult[nz] @ emb_shifted
                    n_tot += n_b

            mu_j = float(thetas[j, 0])
            kl_j = float(thetas[j, 1])
            sb_s = shape_basis[self.signal]
            acc_s = _acceptance_factor(sb_s)
            w_eff_s = (
                compute_hh_weights(self.hh_weights, mu_j, kl_j)
                * acc_s
                * self.nuisance_model.process_rate_multiplier(self.signal, psi_j)
            )
            lam_hh = float(w_eff_s.sum())
            n_hh = int(rng.poisson(max(lam_hh, 0.0)))
            hh_counts[j] = n_hh
            if n_hh > 0 and lam_hh > 0.0:
                mult = rng.multinomial(n_hh, w_eff_s / lam_hh).astype(np.float64)
                nz = mult > 0
                emb_shifted = (
                    sb_s["emb_0"][nz]
                    + d_jes_norm * sb_s["d_jes"][nz]
                    + d_tes_norm * sb_s["d_tes"][nz]
                )
                sum_emb += mult[nz] @ emb_shifted
                n_tot += n_hh

            counts[j] = n_tot
            if n_tot > 0:
                t_bars[j] = (sum_emb / float(n_tot)).astype(np.float32)

        return {
            "theta": thetas.astype(np.float32),
            "T_bar": t_bars,
            "N": counts,
            "N_hh": hh_counts,
        }
