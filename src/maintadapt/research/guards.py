"""Guards that keep test and decision metrics away from the confirmatory archive until frozen.

CLAUDE.md, rule 8: before the tag analysis-frozen exists, no test or decision metric is computed
from runs/research/archive. The smoke and dry-run archives are not guarded.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

from maintadapt.research.config import ResearchConfig

FROZEN_TAG = "analysis-frozen"
# the code that turns the archive into the confirmatory verdicts; frozen with the tag
FROZEN_FILES = (
    "src/maintadapt/research/analysis.py",
    "scripts/research/run_confirmatory.py",
    "src/maintadapt/research/stats.py",
    "src/maintadapt/research/policies.py",
    "src/maintadapt/research/costs.py",
    "src/maintadapt/research/derived.py",
    "scripts/research/build_outcomes.py",
    "scripts/research/build_metrics.py",
)
# archive names that --root accepts besides a path
ROOT_NAMES = ("archive", "smoke", "dryrun", "leaderboard")


class GuardError(RuntimeError):
    pass


def resolve_root(value: str | None, cfg: ResearchConfig) -> Path:
    """An archive named in configs/research.yaml (archive, smoke, ...) or a path."""
    if value is None or value == "":
        return Path(cfg.paths.archive)
    if value in ROOT_NAMES:
        return Path(getattr(cfg.paths, value))
    return Path(value)


def is_confirmatory(root: str | Path, cfg: ResearchConfig) -> bool:
    return Path(root).resolve() == Path(cfg.paths.archive).resolve()


def tag_exists(tag: str = FROZEN_TAG, cwd: str | Path = ".") -> bool:
    found = subprocess.run(
        ["git", "rev-parse", "--quiet", "--verify", f"refs/tags/{tag}"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return found.returncode == 0


def frozen_files_unchanged(
    files: Iterable[str] = FROZEN_FILES, tag: str = FROZEN_TAG, cwd: str | Path = "."
) -> bool:
    """True when every file matches its version at the tag, committed or not."""
    found = subprocess.run(
        ["git", "diff", "--quiet", tag, "--", *files],
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    return found.returncode == 0


def require_frozen_archive(
    root: str | Path,
    cfg: ResearchConfig,
    fleets: Iterable[str],
    validate: Callable[[str, str | Path, ResearchConfig], tuple[list[str], dict]],
    has_tag: Callable[[], bool] = tag_exists,
    unchanged: Callable[[], bool] = frozen_files_unchanged,
) -> None:
    """Raise unless root is not the confirmatory archive, or it is valid and frozen.

    Frozen means the tag analysis-frozen exists and the code that computes the verdicts
    (FROZEN_FILES) is identical to its tagged version.
    """
    if not is_confirmatory(root, cfg):
        return
    if not has_tag():
        raise GuardError(
            f"{root} is the confirmatory archive and the tag {FROZEN_TAG} does not exist; "
            "no test or decision metric may be computed from it yet (CLAUDE.md, rule 8)"
        )
    if not unchanged():
        raise GuardError(
            f"the analysis code differs from its version at {FROZEN_TAG}: "
            f"git diff {FROZEN_TAG} -- {' '.join(FROZEN_FILES)}"
        )
    for fleet in fleets:
        problems, _ = validate(fleet, root, cfg)
        if problems:
            raise GuardError(
                f"{fleet}: research-validate reports {len(problems)} problems in {root}, "
                f"for example {problems[0]}"
            )
