"""CSV loading, physics event selection, caching, splitting, and feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import DataConfig


JET_FEATURE_NAMES = (
    "btagged",
    "ctagged",
    "tautagged",
    "log_pt",
    "eta",
    "cos_phi",
    "sin_phi",
    "log_p",
    "px_scaled",
    "py_scaled",
    "pz_scaled",
)

EVENT_FEATURE_NAMES = (
    "log_m_bbtt",
    "m_bbtt_scaled",
    "m_bb_scaled",
    "m_tt_scaled",
    "log_pt_bb",
    "log_pt_tt",
    "log_pt_bbtt",
    "dr_bb",
    "dr_tt",
    "dphi_bb_tt",
    "log_ht",
    "n_jets_scaled",
    "log_pt_b1",
    "log_pt_b2",
    "log_pt_tau1",
    "log_pt_tau2",
)


@dataclass
class ProcessData:
    """Selected MC events for a single physics process before split/standardization."""

    name: str
    raw_jets: np.ndarray  # (N_evt, max_jets, 6) float32: btag, ctag, tautag, px, py, pz
    jet_mask: np.ndarray  # (N_evt, max_jets) bool
    evt_invariants: np.ndarray  # (N_evt, F_evt) float32
    weights: np.ndarray  # (N_evt, 1 or 3) float64 scaled to LUMI / N_gen
    n_gen: int
    n_csv_events: int

    @property
    def num_events(self) -> int:
        return int(self.raw_jets.shape[0])


@dataclass
class FeatureStats:
    """Mean and standard deviation computed strictly on the training split."""

    jet_mean: np.ndarray  # (F_jet,) float32 (binary tag columns have mean=0, std=1)
    jet_std: np.ndarray   # (F_jet,) float32
    evt_mean: np.ndarray  # (F_evt,) float32
    evt_std: np.ndarray   # (F_evt,) float32

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "jet_mean": self.jet_mean,
            "jet_std": self.jet_std,
            "evt_mean": self.evt_mean,
            "evt_std": self.evt_std,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureStats":
        return cls(
            jet_mean=np.asarray(d["jet_mean"], dtype=np.float32),
            jet_std=np.asarray(d["jet_std"], dtype=np.float32),
            evt_mean=np.asarray(d["evt_mean"], dtype=np.float32),
            evt_std=np.asarray(d["evt_std"], dtype=np.float32),
        )


@dataclass
class SplitProcessPool:
    """One split (train/val/test) for a single physics process."""

    name: str
    jet_features: np.ndarray  # (N_split, max_jets, F_jet) float32 (standardized)
    jet_mask: np.ndarray      # (N_split, max_jets) bool
    evt_features: np.ndarray  # (N_split, F_evt) float32 (standardized)
    raw_jets: np.ndarray      # (N_split, max_jets, 6) float32 (for future JES/JER nuisances)
    weights: np.ndarray       # (N_split, 1 or 3) float64, compensated for split fraction

    @property
    def num_events(self) -> int:
        return int(self.jet_features.shape[0])


@dataclass
class SplitDataset:
    """All processes in a single split (train, val, or test)."""

    split_name: str
    processes: dict[str, SplitProcessPool]
    backgrounds: tuple[str, ...]
    signal: str

    def process_names(self) -> list[str]:
        return list(self.backgrounds) + [self.signal]


def _massless_invariant_mass(p3: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute invariant mass assuming massless jets, matching Rivet/old analysis."""
    masked = p3 * mask[..., None]
    energy = np.linalg.norm(masked, axis=-1)
    etot = energy.sum(axis=-1)
    ptot = masked.sum(axis=-2)
    m2 = np.maximum(etot * etot - np.sum(ptot * ptot, axis=-1), 0.0)
    return np.sqrt(m2)


def _compute_jet_kinematic_features(raw_jets: np.ndarray, jet_mask: np.ndarray) -> np.ndarray:
    """Convert raw (btag, ctag, tautag, px, py, pz) into 11 per-jet features."""
    tags = raw_jets[..., 0:3].astype(np.float32)
    px = raw_jets[..., 3].astype(np.float64)
    py = raw_jets[..., 4].astype(np.float64)
    pz = raw_jets[..., 5].astype(np.float64)

    pt = np.sqrt(px * px + py * py)
    p_mag = np.sqrt(pt * pt + pz * pz)
    pt_safe = np.maximum(pt, 1e-3)
    p_safe = np.maximum(p_mag, 1e-3)

    eta = np.arcsinh(pz / pt_safe)
    cos_phi = px / pt_safe
    sin_phi = py / pt_safe
    log_pt = np.log(pt_safe / 50.0)
    log_p = np.log(p_safe / 50.0)

    cont = np.stack(
        [
            log_pt,
            eta,
            cos_phi,
            sin_phi,
            log_p,
            px / 100.0,
            py / 100.0,
            pz / 100.0,
        ],
        axis=-1,
    ).astype(np.float32)
    feats = np.concatenate([tags, cont], axis=-1)
    feats = feats * jet_mask[..., None].astype(np.float32)
    return feats


def _compute_event_invariants_from_full_jets(
    p3: np.ndarray,
    is_b: np.ndarray,
    is_tau: np.ndarray,
    jet_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact 2b+2tau invariant features from unclipped jets.

    Returns (invariants, m_bbtt, m_bb, m_tt).
    """
    n_evt = p3.shape[0]
    btau = is_b | is_tau
    m_bbtt = _massless_invariant_mass(p3, btau)
    m_bb = _massless_invariant_mass(p3, is_b)
    m_tt = _massless_invariant_mass(p3, is_tau)

    # Extract exact 2 b-jets and 2 tau-jets per event (already sorted by pT descending)
    b_rows, b_cols = np.nonzero(is_b)
    t_rows, t_cols = np.nonzero(is_tau)
    b_p3 = p3[b_rows, b_cols].reshape(n_evt, 2, 3)
    t_p3 = p3[t_rows, t_cols].reshape(n_evt, 2, 3)

    # Sort each pair by pT descending for deterministic leading/subleading assignment
    b_pt = np.hypot(b_p3[:, :, 0], b_p3[:, :, 1])
    b_order = np.argsort(-b_pt, axis=1)
    b_p3 = np.take_along_axis(b_p3, b_order[:, :, None], axis=1)
    b_pt = np.take_along_axis(b_pt, b_order, axis=1)

    t_pt = np.hypot(t_p3[:, :, 0], t_p3[:, :, 1])
    t_order = np.argsort(-t_pt, axis=1)
    t_p3 = np.take_along_axis(t_p3, t_order[:, :, None], axis=1)
    t_pt = np.take_along_axis(t_pt, t_order, axis=1)

    def _eta_phi(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pt = np.maximum(np.hypot(vec[..., 0], vec[..., 1]), 1e-3)
        eta = np.arcsinh(vec[..., 2] / pt)
        phi = np.arctan2(vec[..., 1], vec[..., 0])
        return eta, phi

    def _delta_r(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        e1, p1 = _eta_phi(v1)
        e2, p2 = _eta_phi(v2)
        dphi = np.arctan2(np.sin(p1 - p2), np.cos(p1 - p2))
        return np.hypot(e1 - e2, dphi)

    p_bb = b_p3.sum(axis=1)
    p_tt = t_p3.sum(axis=1)
    p_bbtt = p_bb + p_tt

    pt_bb = np.hypot(p_bb[:, 0], p_bb[:, 1])
    pt_tt = np.hypot(p_tt[:, 0], p_tt[:, 1])
    pt_bbtt = np.hypot(p_bbtt[:, 0], p_bbtt[:, 1])

    dr_bb = _delta_r(b_p3[:, 0], b_p3[:, 1])
    dr_tt = _delta_r(t_p3[:, 0], t_p3[:, 1])

    phi_bb = np.arctan2(p_bb[:, 1], p_bb[:, 0])
    phi_tt = np.arctan2(p_tt[:, 1], p_tt[:, 0])
    dphi_bb_tt = np.abs(np.arctan2(np.sin(phi_bb - phi_tt), np.cos(phi_bb - phi_tt)))

    all_pt = np.hypot(p3[:, :, 0], p3[:, :, 1])
    ht = np.sum(all_pt, axis=1)

    invariants = np.stack(
        [
            np.log(np.maximum(m_bbtt, 1.0) / 300.0),
            m_bbtt / 300.0,
            m_bb / 125.0,
            m_tt / 100.0,
            np.log((pt_bb + 1.0) / 100.0),
            np.log((pt_tt + 1.0) / 100.0),
            np.log((pt_bbtt + 1.0) / 50.0),
            dr_bb,
            dr_tt,
            dphi_bb_tt,
            np.log(np.maximum(ht, 1.0) / 200.0),
            jet_counts.astype(np.float64) / 5.0,
            np.log(np.maximum(b_pt[:, 0], 1.0) / 50.0),
            np.log(np.maximum(b_pt[:, 1], 1.0) / 50.0),
            np.log(np.maximum(t_pt[:, 0], 1.0) / 50.0),
            np.log(np.maximum(t_pt[:, 1], 1.0) / 50.0),
        ],
        axis=-1,
    ).astype(np.float32)

    return invariants, m_bbtt, m_bb, m_tt


def parse_and_select_process(
    csv_path: Path,
    name: str,
    max_jets: int = 12,
    lumi_pb_inv: float = 100_000.0,
) -> ProcessData:
    """Load a Rivet CSV file, apply 2b+2tau mass selections, and normalize weights."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]

    evnum = data[:, 0].astype(np.int64)
    change = np.nonzero(np.diff(evnum) != 0)[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(evnum)]))
    if not np.all(np.diff(evnum[starts]) > 0):
        raise ValueError(f"Event numbers in {csv_path} are not strictly increasing per event block")

    n_csv_events = int(len(starts))
    meta_path = Path(f"{csv_path}.meta")
    if meta_path.exists():
        n_gen = None
        for line in meta_path.read_text().splitlines():
            if line.startswith("n_gen="):
                n_gen = int(line.split("=", 1)[1].strip())
                break
        if n_gen is None:
            raise ValueError(f"Missing n_gen= in {meta_path}")
    else:
        n_gen = int(evnum.max()) + 1
    counts = ends - starts
    max_full = int(counts.max())

    jets_feat = data[:, 1:7]
    weights_raw = data[starts, 7:]  # (n_csv_events, 1 or 3)

    full_jets = np.zeros((n_csv_events, max_full, 6), dtype=np.float64)
    full_mask = np.zeros((n_csv_events, max_full), dtype=bool)
    for j in range(max_full):
        valid = counts > j
        full_jets[valid, j] = jets_feat[starts[valid] + j]
        full_mask[valid, j] = True

    is_b = full_mask & (full_jets[:, :, 0] == 1)
    is_tau = full_mask & (full_jets[:, :, 2] == 1) & (~is_b)
    sel_tags = (is_b.sum(axis=1) == 2) & (is_tau.sum(axis=1) == 2)

    full_jets_tag = full_jets[sel_tags]
    is_b_tag = is_b[sel_tags]
    is_tau_tag = is_tau[sel_tags]
    counts_tag = counts[sel_tags]

    invariants_tag, m_bbtt, m_bb, m_tt = _compute_event_invariants_from_full_jets(
        full_jets_tag[:, :, 3:6],
        is_b_tag,
        is_tau_tag,
        counts_tag,
    )

    sel_kin = (
        (m_bbtt > 200.0)
        & (m_bb > 75.0)
        & (m_bb < 150.0)
        & (m_tt > 50.0)
        & (m_tt < 125.0)
    )

    sel_idx = np.nonzero(sel_tags)[0][sel_kin]
    invariants = invariants_tag[sel_kin]
    weights = (lumi_pb_inv / float(n_gen)) * weights_raw[sel_idx]

    # Sort jets per selected event by pT descending and clip/pad to max_jets
    n_sel = len(sel_idx)
    sel_full_jets = full_jets[sel_idx]
    sel_full_mask = full_mask[sel_idx]
    pt_all = np.where(
        sel_full_mask,
        np.hypot(sel_full_jets[:, :, 3], sel_full_jets[:, :, 4]),
        -1.0,
    )
    order = np.argsort(-pt_all, axis=1)[:, :max_jets]
    row_idx = np.arange(n_sel)[:, None]
    clipped_jets = sel_full_jets[row_idx, order].astype(np.float32)
    clipped_mask = sel_full_mask[row_idx, order]

    if max_full < max_jets:
        pad_n = max_jets - max_full
        clipped_jets = np.pad(clipped_jets, ((0, 0), (0, pad_n), (0, 0)))
        clipped_mask = np.pad(clipped_mask, ((0, 0), (0, pad_n)))

    return ProcessData(
        name=name,
        raw_jets=clipped_jets,
        jet_mask=clipped_mask,
        evt_invariants=invariants,
        weights=weights.astype(np.float64),
        n_gen=n_gen,
        n_csv_events=n_csv_events,
    )


def load_or_build_process_cache(cfg: DataConfig) -> dict[str, ProcessData]:
    """Load selected process arrays from .npz cache or build from CSV files."""
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cfg.cache_dir / f"bbtautau_selected_maxjets{cfg.max_jets}_lumi{int(round(cfg.lumi_pb_inv))}.npz"
    proc_names = list(cfg.backgrounds) + [cfg.signal]

    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=False)
        processes: dict[str, ProcessData] = {}
        for name in proc_names:
            processes[name] = ProcessData(
                name=name,
                raw_jets=loaded[f"{name}_raw_jets"],
                jet_mask=loaded[f"{name}_jet_mask"],
                evt_invariants=loaded[f"{name}_evt_invariants"],
                weights=loaded[f"{name}_weights"],
                n_gen=int(loaded[f"{name}_n_gen"][0]),
                n_csv_events=int(loaded[f"{name}_n_csv_events"][0]),
            )
        return processes

    processes = {}
    payload: dict[str, np.ndarray] = {}
    for name in proc_names:
        csv_path = cfg.data_dir / f"{name}.csv"
        proc = parse_and_select_process(
            csv_path=csv_path,
            name=name,
            max_jets=cfg.max_jets,
            lumi_pb_inv=cfg.lumi_pb_inv,
        )
        processes[name] = proc
        payload[f"{name}_raw_jets"] = proc.raw_jets
        payload[f"{name}_jet_mask"] = proc.jet_mask
        payload[f"{name}_evt_invariants"] = proc.evt_invariants
        payload[f"{name}_weights"] = proc.weights
        payload[f"{name}_n_gen"] = np.array([proc.n_gen], dtype=np.int64)
        payload[f"{name}_n_csv_events"] = np.array([proc.n_csv_events], dtype=np.int64)

    np.savez(cache_path, **payload)
    return processes


def build_splits(
    processes: dict[str, ProcessData],
    cfg: DataConfig,
) -> tuple[dict[str, SplitDataset], FeatureStats]:
    """Split each process into disjoint train_s1/train_s3/val/test pools with compensated weights."""
    cfg.validate()
    rng = np.random.default_rng(cfg.split_seed)

    split_names = ("train_s1", "train_s3", "val", "test")
    split_indices: dict[str, dict[str, np.ndarray]] = {s: {} for s in split_names}
    raw_jet_feats: dict[str, np.ndarray] = {}

    for name, proc in processes.items():
        n = proc.num_events
        perm = rng.permutation(n)
        n_s1 = max(1, int(round(cfg.stage1_train_frac * n)))
        n_s3 = max(1, int(round(cfg.stage3_train_frac * n)))
        n_val = max(1, int(round(cfg.val_frac * n)))
        n_test = n - n_s1 - n_s3 - n_val
        if n_test < 1:
            n_s1 = max(1, n_s1 - 1)
            n_test = n - n_s1 - n_s3 - n_val

        split_indices["train_s1"][name] = perm[:n_s1]
        split_indices["train_s3"][name] = perm[n_s1 : n_s1 + n_s3]
        split_indices["val"][name] = perm[n_s1 + n_s3 : n_s1 + n_s3 + n_val]
        split_indices["test"][name] = perm[n_s1 + n_s3 + n_val :]
        raw_jet_feats[name] = _compute_jet_kinematic_features(proc.raw_jets, proc.jet_mask)

    # Fit feature standardization strictly on training pools (train_s1 + train_s3)
    train_jet_vecs = []
    train_evt_vecs = []
    for name, proc in processes.items():
        idx = np.concatenate([split_indices["train_s1"][name], split_indices["train_s3"][name]])
        mask = proc.jet_mask[idx]
        jf = raw_jet_feats[name][idx][mask]
        train_jet_vecs.append(jf)
        train_evt_vecs.append(proc.evt_invariants[idx])

    all_train_jets = np.concatenate(train_jet_vecs, axis=0)
    all_train_evts = np.concatenate(train_evt_vecs, axis=0)

    jet_mean = all_train_jets.mean(axis=0).astype(np.float32)
    jet_std = np.maximum(all_train_jets.std(axis=0), 1e-5).astype(np.float32)
    # Keep binary tag indicators (btagged, ctagged, tautagged) unshifted as 0/1
    jet_mean[:3] = 0.0
    jet_std[:3] = 1.0

    evt_mean = all_train_evts.mean(axis=0).astype(np.float32)
    evt_std = np.maximum(all_train_evts.std(axis=0), 1e-5).astype(np.float32)
    stats = FeatureStats(jet_mean=jet_mean, jet_std=jet_std, evt_mean=evt_mean, evt_std=evt_std)

    splits: dict[str, SplitDataset] = {}
    for split_name in split_names:
        pools: dict[str, SplitProcessPool] = {}
        for name, proc in processes.items():
            idx = split_indices[split_name][name]
            comp = float(proc.num_events) / float(len(idx))
            jf_std = (raw_jet_feats[name][idx] - stats.jet_mean) / stats.jet_std
            jf_std = jf_std * proc.jet_mask[idx][..., None].astype(np.float32)
            ef_std = (proc.evt_invariants[idx] - stats.evt_mean) / stats.evt_std
            pools[name] = SplitProcessPool(
                name=name,
                jet_features=jf_std.astype(np.float32),
                jet_mask=proc.jet_mask[idx],
                evt_features=ef_std.astype(np.float32),
                raw_jets=proc.raw_jets[idx],
                weights=(proc.weights[idx] * comp).astype(np.float64),
            )
        splits[split_name] = SplitDataset(
            split_name=split_name,
            processes=pools,
            backgrounds=cfg.backgrounds,
            signal=cfg.signal,
        )

    return splits, stats
