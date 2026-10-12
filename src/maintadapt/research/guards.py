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


def require_frozen_archive(
    root: str | Path,
    cfg: ResearchConfig,
    fleets: Iterable[str],
    validate: Callable[[str, str | Path, ResearchConfig], tuple[list[str], dict]],
    has_tag: Callable[[], bool] = tag_exists,
) -> None:
    """Raise unless root is not the confirmatory archive, or it is valid and frozen."""
    if not is_confirmatory(root, cfg):
        return
    if not has_tag():
        raise GuardError(
            f"{root} is the confirmatory archive and the tag {FROZEN_TAG} does not exist; "
            "no test or decision metric may be computed from it yet (CLAUDE.md, rule 8)"
        )
    for fleet in fleets:
        problems, _ = validate(fleet, root, cfg)
        if problems:
            raise GuardError(
                f"{fleet}: research-validate reports {len(problems)} problems in {root}, "
                f"for example {problems[0]}"
            )
