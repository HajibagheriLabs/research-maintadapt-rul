"""Conventional leaderboard metrics: official test engines at their last observed cycle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from maintadapt.evaluate import nasa_score, rmse
from maintadapt.research.archive import OK, records

METRICS = ["rmse", "nasa_score"]
# families whose reference configuration the engineering project also trained
LEGACY_FAMILIES = ("ridge", "xgboost", "lstm", "tcn")
# the legacy summary tables round to three decimals
LEGACY_ROUNDING = 5e-4


def last_cycle_metrics(predictions: pd.DataFrame, cap: int) -> dict[str, float]:
    """RMSE and NASA score at each official test engine's last cycle, truth capped at cap."""
    test = predictions[predictions["split"].astype(str) == "official_test"]
    last = test.loc[test.groupby("unit_id")["cycle"].idxmax()]
    truth = np.minimum(last["rul"].to_numpy(float), cap)
    pred = last["pred"].to_numpy(float)
    return {"rmse": rmse(truth, pred), "nasa_score": nasa_score(truth, pred)}


def per_seed_metrics(root: str | Path, cap: int) -> pd.DataFrame:
    """Leaderboard metrics of every finished fit under root, stored as seed<k>.parquet."""
    found = records(root)
    rows = []
    for record in (
        [] if found.empty else found[found["status"] == OK].to_dict("records")
    ):
        table = pq.read_table(record["parquet"]).to_pandas()
        rows.append(
            {
                "fleet": record["fleet"],
                "family": record["family"],
                "seed": int(Path(record["parquet"]).stem.removeprefix("seed")),
                **last_cycle_metrics(table, cap),
            }
        )
    columns = ["fleet", "family", "seed", *METRICS]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values(["fleet", "family", "seed"])


def compare_with_legacy(reference: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    """Research minus legacy means, with the standard error of that difference over seeds.

    Both tables hold fleet, family and <metric>_mean, <metric>_std, n_seeds. A difference
    counts as within seed noise when it is at most twice its standard error, allowing for the
    rounding of the legacy tables.
    """
    both = reference.merge(
        legacy, on=["fleet", "family"], suffixes=("_research", "_legacy")
    )
    both = both[both["family"].isin(LEGACY_FAMILIES)]
    out = both[["fleet", "family"]].copy()
    for metric in METRICS:
        r, g = f"{metric}_mean_research", f"{metric}_mean_legacy"
        sd_r, sd_g = f"{metric}_std_research", f"{metric}_std_legacy"
        se = np.sqrt(
            both[sd_r] ** 2 / both["n_seeds_research"]
            + both[sd_g] ** 2 / both["n_seeds_legacy"]
        )
        out[f"{metric}_research"] = both[r]
        out[f"{metric}_legacy"] = both[g]
        out[f"{metric}_diff"] = both[r] - both[g]
        out[f"{metric}_diff_se"] = se
        out[f"{metric}_within_noise"] = (
            out[f"{metric}_diff"].abs() <= 2 * se + LEGACY_ROUNDING
        )
    return out.reset_index(drop=True)
