"""Research configuration: configs/research.yaml loaded strictly into frozen dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from maintadapt.config import SUBSETS, _build
from maintadapt.research.models.spaces import FAMILY_NAMES

RULES = ("R1", "R2")
TUNINGS = ("validation", "crossfit", "oracle", "nested")


@dataclass(frozen=True)
class FleetDataConfig:
    fleets: list[str]
    raw_dir: str
    cap: int
    n_conditions: dict[str, int]
    max_modal_share: float
    windows: list[int]
    include_cycle: bool

    def __post_init__(self) -> None:
        if not self.fleets or not set(self.fleets) <= set(SUBSETS):
            raise ValueError(
                f"data.fleets must be drawn from {SUBSETS}, got {self.fleets}"
            )
        if len(set(self.fleets)) != len(self.fleets):
            raise ValueError(f"data.fleets lists a fleet twice: {self.fleets}")
        if self.cap <= 0:
            raise ValueError(f"data.cap must be positive, got {self.cap}")
        missing = sorted(set(self.fleets) - set(self.n_conditions))
        if missing or min(self.n_conditions.values()) < 1:
            raise ValueError(
                f"data.n_conditions needs a positive count per fleet, missing {missing}"
            )
        if not 0.5 < self.max_modal_share <= 1.0:
            raise ValueError(
                f"data.max_modal_share must lie in (0.5, 1], got {self.max_modal_share}"
            )
        if not self.windows or min(self.windows) < 2:
            raise ValueError(
                f"data.windows must each span two or more cycles, got {self.windows}"
            )


@dataclass(frozen=True)
class FoldsConfig:
    n_outer: int
    n_repeats: int
    inner_val_fraction: float
    seed: int

    def __post_init__(self) -> None:
        if self.n_outer < 2:
            raise ValueError(f"folds.n_outer must be at least 2, got {self.n_outer}")
        if self.n_repeats < 1:
            raise ValueError(f"folds.n_repeats must be positive, got {self.n_repeats}")
        # the block rule draws one inner_val engine per block of 1 / fraction engines
        size = 1.0 / self.inner_val_fraction if self.inner_val_fraction > 0 else 0.0
        if not 0.0 < self.inner_val_fraction <= 0.5 or abs(size - round(size)) > 1e-9:
            raise ValueError(
                "folds.inner_val_fraction must be 1/m for an integer m >= 2, "
                f"got {self.inner_val_fraction}"
            )

    @property
    def inner_block(self) -> int:
        return round(1.0 / self.inner_val_fraction)


@dataclass(frozen=True)
class PoolConfig:
    seed: int
    n_configs: dict[str, int]

    def __post_init__(self) -> None:
        if set(self.n_configs) != set(FAMILY_NAMES):
            raise ValueError(
                f"pool.n_configs must name exactly the families {list(FAMILY_NAMES)}, "
                f"got {sorted(self.n_configs)}"
            )
        if min(self.n_configs.values()) < 1:
            raise ValueError("pool.n_configs must be positive for every family")

    @property
    def total(self) -> int:
        return sum(self.n_configs.values())


@dataclass(frozen=True)
class ResearchTrainingConfig:
    device: str
    max_epochs: int
    patience: int
    grad_clip: float
    batch_size: int
    # CPU threads of a CPU-stream fit; null means half the logical CPUs
    cpu_jobs: int | None
    # CPU threads of a fit that runs on the GPU
    gpu_threads: int

    def __post_init__(self) -> None:
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(
                f"training.device must be auto, cpu or cuda, got {self.device}"
            )
        counts = {
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "gpu_threads": self.gpu_threads,
        }
        if self.cpu_jobs is not None:
            counts["cpu_jobs"] = self.cpu_jobs
        bad = sorted(k for k, v in counts.items() if v < 1)
        if bad or self.grad_clip <= 0:
            raise ValueError(f"training.{bad or ['grad_clip']} must be positive")

    def resolved_cpu_jobs(self) -> int:
        if self.cpu_jobs is not None:
            return self.cpu_jobs
        # the runs last days next to the GPU stream, so the CPU keeps half its threads idle
        return max(1, (os.cpu_count() or 2) // 2)


@dataclass(frozen=True)
class CrossfitConfig:
    # R and any dropped configurations follow from the dry run by PREREGISTRATION.md 4.4
    repeats: int
    # drops configurations 8-11 of every family with twelve configurations
    drop_late_configs: bool

    def configs_run(self, n_configs: int) -> int:
        """How many of a family's configurations the confirmatory runs fit."""
        return 8 if self.drop_late_configs and n_configs == 12 else n_configs


@dataclass(frozen=True)
class PrimaryConfig:
    cost_ratio: float
    lead_time: int
    rule: str
    tuning: str


@dataclass(frozen=True)
class DecisionGridConfig:
    cost_ratios: list[float]
    lead_times: list[int]
    regret_failure_costs: list[float]
    ema_alphas: list[float]
    threshold_levels: int
    primary: PrimaryConfig

    def __post_init__(self) -> None:
        # a failure must cost more than the preventive action it replaces
        if not self.cost_ratios or min(self.cost_ratios) <= 1.0:
            raise ValueError(
                f"decision.cost_ratios must all exceed 1, got {self.cost_ratios}"
            )
        if not self.lead_times or min(self.lead_times) < 0:
            raise ValueError(
                f"decision.lead_times must be non-negative, got {self.lead_times}"
            )
        if not self.regret_failure_costs or min(self.regret_failure_costs) <= 0:
            raise ValueError("decision.regret_failure_costs must be positive")
        # alpha = 1 makes R2 identical to R1, so the R2 grid must contain it
        alphas = self.ema_alphas
        if 1.0 not in alphas or not all(0.0 < a <= 1.0 for a in alphas):
            raise ValueError(
                f"decision.ema_alphas must lie in (0, 1] and include 1, got {alphas}"
            )
        if self.threshold_levels < 2:
            raise ValueError("decision.threshold_levels needs at least two levels")
        p = self.primary
        if p.cost_ratio not in self.cost_ratios or p.lead_time not in self.lead_times:
            raise ValueError("decision.primary must be a point of the settings grid")
        if p.rule not in RULES or p.tuning not in TUNINGS:
            raise ValueError(
                f"decision.primary.rule must be one of {RULES} and tuning one of "
                f"{TUNINGS}, got {p.rule} and {p.tuning}"
            )


@dataclass(frozen=True)
class StatsConfig:
    n_boot: int
    ci: float
    fdr: float
    seed: int

    def __post_init__(self) -> None:
        if self.n_boot < 1:
            raise ValueError(f"stats.n_boot must be positive, got {self.n_boot}")
        if not 0.0 < self.ci < 1.0 or not 0.0 < self.fdr < 1.0:
            raise ValueError("stats.ci and stats.fdr must lie in (0, 1)")


@dataclass(frozen=True)
class NestedConfig:
    """S4 nested tuning: inner cross-fitting over each outer fold's outer-train engines."""

    fleets: list[str]
    repeat: int
    inner_folds: int
    # share of each inner training set held out for early stopping, as 1/m
    early_stopping_fraction: float

    def __post_init__(self) -> None:
        if not self.fleets or not set(self.fleets) <= set(SUBSETS):
            raise ValueError(
                f"secondary.nested.fleets must be fleets, got {self.fleets}"
            )
        if self.inner_folds < 2:
            raise ValueError("secondary.nested.inner_folds must be at least 2")
        f = self.early_stopping_fraction
        size = 1.0 / f if f > 0 else 0.0
        if not 0.0 < f <= 0.5 or abs(size - round(size)) > 1e-9:
            raise ValueError(
                "secondary.nested.early_stopping_fraction must be 1/m for an integer "
                f"m >= 2, got {f}"
            )

    @property
    def early_stopping_block(self) -> int:
        return round(1.0 / self.early_stopping_fraction)


@dataclass(frozen=True)
class CapsConfig:
    """S5: the reference configurations of some families trained with other caps."""

    # null trains on the uncapped RUL
    values: list[int | None]
    families: list[str]
    repeat: int
    # stands in for the missing cap: the networks' target scale and the upper clip of
    # predictions; it lies above every training RUL, so no target is cut
    uncapped_scale: int

    def __post_init__(self) -> None:
        caps = [v for v in self.values if v is not None]
        if not self.values or min(caps, default=1) < 1:
            raise ValueError(
                f"secondary.caps.values must be positive, got {self.values}"
            )
        if not set(self.families) <= set(FAMILY_NAMES):
            raise ValueError("secondary.caps.families must be pool families")
        if self.uncapped_scale <= max(caps, default=0):
            raise ValueError("secondary.caps.uncapped_scale must exceed every cap")

    @staticmethod
    def name(cap: int | None) -> str:
        """Directory and label of a cap: its value, or none."""
        return "none" if cap is None else str(cap)

    def training_cap(self, cap: int | None) -> int:
        """The cap a fit trains with; without one, the scale above every training RUL."""
        return self.uncapped_scale if cap is None else cap


@dataclass(frozen=True)
class SecondaryConfig:
    nested: NestedConfig
    caps: CapsConfig
    # S9: configurations within this many cycles of the best leaderboard RMSE
    competitive_margin: float


@dataclass(frozen=True)
class ResearchPathsConfig:
    folds: str
    pool: str
    manifest: str
    archive: str
    derived: str
    smoke: str
    dryrun: str
    leaderboard: str
    archive_nested: str
    archive_caps: str
    reports: str
    paper_assets: str


@dataclass(frozen=True)
class ResearchConfig:
    data: FleetDataConfig
    folds: FoldsConfig
    pool: PoolConfig
    training: ResearchTrainingConfig
    crossfit: CrossfitConfig
    decision: DecisionGridConfig
    stats: StatsConfig
    secondary: SecondaryConfig
    paths: ResearchPathsConfig

    def __post_init__(self) -> None:
        if not 1 <= self.crossfit.repeats <= self.folds.n_repeats:
            raise ValueError(
                f"crossfit.repeats must lie in 1..{self.folds.n_repeats} (the repeats in "
                f"the folds file), got {self.crossfit.repeats}"
            )
        for name, repeat in (
            ("nested", self.secondary.nested.repeat),
            ("caps", self.secondary.caps.repeat),
        ):
            if not 0 <= repeat < self.crossfit.repeats:
                raise ValueError(
                    f"secondary.{name}.repeat must be a repeat of the runs, got {repeat}"
                )


def fleet_number(name: str) -> int:
    """1 to 4 for FD001 to FD004; part of every fleet-specific random seed."""
    if name not in SUBSETS:
        raise ValueError(f"unknown fleet {name!r}; expected one of {SUBSETS}")
    return int(name[2:])


def load_research_config(path: str | Path) -> ResearchConfig:
    """Load configs/research.yaml; unknown, missing or mistyped keys raise."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _build(ResearchConfig, raw, "research")
