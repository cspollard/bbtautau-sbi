"""Configuration dataclasses and runtime utilities for PAIRS bb-tautau inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Literal

import numpy as np
import torch


DEFAULT_INFERENCE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class PriorConfig:
    """2D uniform prior bounds over (mu_HH, kappa_lambda)."""

    mu_min: float = 0.0
    mu_max: float = 10.0
    kl_min: float = -5.0
    kl_max: float = 10.0

    def validate(self) -> None:
        if self.mu_max <= self.mu_min:
            raise ValueError(f"Invalid mu bounds: [{self.mu_min}, {self.mu_max}]")
        if self.kl_max <= self.kl_min:
            raise ValueError(f"Invalid kappa_lambda bounds: [{self.kl_min}, {self.kl_max}]")

    @property
    def low(self) -> tuple[float, float]:
        return (self.mu_min, self.kl_min)

    @property
    def high(self) -> tuple[float, float]:
        return (self.mu_max, self.kl_max)

    @property
    def ranges(self) -> tuple[float, float]:
        return (self.mu_max - self.mu_min, self.kl_max - self.kl_min)


@dataclass
class DataConfig:
    """Dataset loading, event selection, disjoint splitting, and caching configuration."""

    data_dir: Path = DEFAULT_INFERENCE_DIR
    cache_dir: Path = DEFAULT_INFERENCE_DIR / "cache"
    lumi_pb_inv: float = 100_000.0  # 100 fb^-1 = 100e3 pb^-1
    max_jets: int = 12
    # Disjoint MC pools: Stage 1 embedder training, Stage 3 aggregate training (unseen by S1),
    # validation, and held-out test evaluation.
    stage1_train_frac: float = 0.35
    stage3_train_frac: float = 0.25
    val_frac: float = 0.20
    test_frac: float = 0.20
    split_seed: int = 42
    backgrounds: tuple[str, ...] = ("top", "ZH", "higgs", "DYbb")
    signal: str = "HH"

    def validate(self) -> None:
        total = self.stage1_train_frac + self.stage3_train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")
        if self.lumi_pb_inv <= 0:
            raise ValueError(f"lumi_pb_inv must be positive, got {self.lumi_pb_inv}")
        if self.max_jets < 4:
            raise ValueError(f"max_jets must be >= 4 for 2b+2tau selection, got {self.max_jets}")


@dataclass
class ModelConfig:
    """Architecture hyperparameters for PAIRS embedder and conditional spline flow."""

    embed_dim: int = 12
    jet_hidden_dim: int = 64
    jet_num_heads: int = 4
    jet_num_layers: int = 2
    event_hidden_dim: int = 128
    event_num_blocks: int = 2
    dropout: float = 0.12
    flow_num_transforms: int = 4
    flow_hidden_dim: int = 64
    flow_num_blocks: int = 1
    flow_num_bins: int = 10
    flow_tail_bound: float = 3.5
    context_hidden_dim: int = 64


@dataclass
class TrainConfig:
    """Training and evaluation hyperparameters for Stages 1, 2, 3."""

    seed: int = 123
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"

    # Stage 1: PAIRS pretraining on N in {1, 2}
    stage1_epochs: int = 30
    stage1_batches_per_epoch: int = 150
    stage1_batch_size: int = 512
    stage1_val_sets: int = 8192
    stage1_lr: float = 5e-4
    stage1_weight_decay: float = 5e-4
    stage1_warmup_frac: float = 0.05
    stage1_patience: int = 10
    stage1_grad_clip: float = 1.0
    # Wider mu_HH prior range during Stage 1 only so N in {1,2} sees strong signal-to-background
    # contrast across kappa_lambda while preserving the exact per-event likelihood ratio basis.
    stage1_mu_max: float | None = 300.0

    # Stage 2: Embedding & dataset-level aggregate caching
    stage2_embed_batch_size: int = 2048
    stage2_n_train_sets: int = 20_000
    stage2_n_val_sets: int = 4_000
    stage2_n_test_sets: int = 2_000

    # Stage 3: Finetuning conditional normalizing flow head on cached aggregates
    stage3_epochs: int = 80
    stage3_batch_size: int = 256
    stage3_lr: float = 4e-4
    stage3_weight_decay: float = 2e-2
    stage3_warmup_frac: float = 0.05
    stage3_patience: int = 15
    stage3_grad_clip: float = 1.0
    stage3_reinit_head: bool = True
    stage3_mc_aug: bool = True

    # Evaluation
    eval_num_samples: int = 1000
    eval_acauc_levels: int = 100


@dataclass
class PairsConfig:
    """Top-level configuration container."""

    prior: PriorConfig = field(default_factory=PriorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: Path = DEFAULT_INFERENCE_DIR / "outputs"

    def validate(self) -> None:
        self.prior.validate()
        self.data.validate()

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = asdict(self)

        def _convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj

        path.write_text(json.dumps(_convert(raw), indent=2))

    @classmethod
    def load_json(cls, path: Path) -> "PairsConfig":
        raw = json.loads(path.read_text())
        data_dict = raw["data"]
        data_dict["data_dir"] = Path(data_dict["data_dir"])
        data_dict["cache_dir"] = Path(data_dict["cache_dir"])
        data_dict["backgrounds"] = tuple(data_dict["backgrounds"])
        return cls(
            prior=PriorConfig(**raw["prior"]),
            data=DataConfig(**data_dict),
            model=ModelConfig(**raw["model"]),
            train=TrainConfig(**raw["train"]),
            output_dir=Path(raw["output_dir"]),
        )


def resolve_device(requested: str = "auto") -> torch.device:
    """Select CUDA > MPS > CPU when 'auto', or validate explicit choice."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    """Set RNG seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
