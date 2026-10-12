"""Wall-time projection from the dry run, and the number of repeats it implies (section 4.4)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# PREREGISTRATION.md 4.4: hours per stream, with the CPU and GPU streams run concurrently
MAX_HOURS_R3 = 72.0
MAX_HOURS_R2 = 120.0
# fleets the dry run times, and the fleet each projection is scaled from
SOURCES = {"FD001": "FD001", "FD002": "FD004", "FD003": "FD001", "FD004": "FD004"}


@dataclass(frozen=True)
class RepeatDecision:
    repeats: int
    drop_late_configs: bool
    reason: str


def configs_kept(n_configs: int, drop_late: bool) -> int:
    return 8 if drop_late and n_configs == 12 else n_configs


def project_hours(
    dryrun: pd.DataFrame,
    streams: Mapping[str, Sequence[str]],
    n_configs: Mapping[str, int],
    rows: Mapping[str, float],
    measured_rows: Mapping[str, float],
    n_outer: int,
) -> pd.DataFrame:
    """Hours per stream, fleet and family for R = 3, R = 2 and R = 2 with configs 8-11 dropped.

    dryrun holds the sidecars of the dry run (fleet, family, status, wall_seconds). A fleet's
    seconds per fit are the mean over the timed configurations of its source fleet, scaled by
    rows[fleet] / measured_rows[source], the inner_fit rows of the fold that was timed.
    """
    ok = dryrun[dryrun["status"] == "ok"]
    seconds = ok.groupby(["fleet", "family"])["wall_seconds"].mean()
    out = []
    for stream, families in streams.items():
        for fleet, source in SOURCES.items():
            for family in families:
                measured = seconds.get((source, family), np.nan)
                scale = rows[fleet] / measured_rows[source]
                per_fit = measured * scale
                n = n_configs[family]
                out.append(
                    {
                        "stream": stream,
                        "fleet": fleet,
                        "family": family,
                        "source_fleet": source,
                        "measured_seconds": measured,
                        "rows_scale": scale,
                        "seconds_per_fit": per_fit,
                        "configs": n,
                        "hours_r3": per_fit * n * n_outer * 3 / 3600,
                        "hours_r2": per_fit * n * n_outer * 2 / 3600,
                        "hours_r2_dropped": per_fit
                        * configs_kept(n, True)
                        * n_outer
                        * 2
                        / 3600,
                    }
                )
    detail = pd.DataFrame(out)
    if detail["measured_seconds"].isna().any():
        missing = detail.loc[
            detail["measured_seconds"].isna(), ["source_fleet", "family"]
        ]
        raise ValueError(
            "the dry run has no finished fit for "
            + ", ".join(
                sorted({f"{f} {m}" for f, m in missing.itertuples(index=False)})
            )
        )
    hours = ["hours_r3", "hours_r2", "hours_r2_dropped"]
    totals = detail.groupby("stream", sort=False)[hours].sum().reset_index()
    totals[["fleet", "family"]] = "all"
    return pd.concat([detail, totals], ignore_index=True)[detail.columns]


def stream_totals(projection: pd.DataFrame) -> pd.DataFrame:
    total = projection[projection["family"] == "all"]
    return total.set_index("stream")[["hours_r3", "hours_r2", "hours_r2_dropped"]]


def decide_repeats(totals: pd.DataFrame) -> RepeatDecision:
    """R = 3 if every stream fits in 72 hours; else R = 2, dropping configs 8-11 past 120."""
    worst = totals.max()
    if worst["hours_r3"] <= MAX_HOURS_R3:
        return RepeatDecision(
            3, False, f"R = 3 projects at most {worst['hours_r3']:.1f} h per stream"
        )
    if worst["hours_r2"] <= MAX_HOURS_R2:
        return RepeatDecision(
            2,
            False,
            f"R = 3 projects {worst['hours_r3']:.1f} h > {MAX_HOURS_R3:g} h; "
            f"R = 2 projects {worst['hours_r2']:.1f} h",
        )
    return RepeatDecision(
        2,
        True,
        f"R = 2 projects {worst['hours_r2']:.1f} h > {MAX_HOURS_R2:g} h; dropping configs "
        f"8-11 gives {worst['hours_r2_dropped']:.1f} h",
    )
