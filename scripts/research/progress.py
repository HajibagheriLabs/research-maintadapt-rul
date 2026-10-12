"""Finished, failed and pending fits per fleet and family against the expected totals.

Reads sidecar statuses only; it prints no metric of any kind.
"""

import argparse

import pandas as pd

from maintadapt.research.archive import records
from maintadapt.research.config import load_research_config
from maintadapt.research.crossfit import STREAMS, plan, select_configs
from maintadapt.research.guards import resolve_root
from maintadapt.research.models.spaces import FAMILY_NAMES, load_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument(
        "--root", help="archive, smoke, dryrun or a path (default: archive)"
    )
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    root = resolve_root(args.root, cfg)

    pool = load_pool(cfg.paths.pool, cfg.pool.seed, cfg.pool.n_configs)
    configs = select_configs(pool, FAMILY_NAMES, None, cfg.crossfit.configs_run)
    jobs = plan(
        configs,
        cfg.data.fleets,
        range(cfg.crossfit.repeats),
        range(cfg.folds.n_outer),
    )
    expected = pd.DataFrame(
        [
            {
                "fleet": j.fleet,
                "family": j.config.family,
                "config_id": j.config.id,
                "repeat": j.repeat,
                "fold": j.fold,
            }
            for j in jobs
        ]
    )
    keys = ["fleet", "config_id", "repeat", "fold"]
    found = records(root)
    if found.empty:
        found = pd.DataFrame(columns=[*keys, "status"])
    merged = expected.merge(found[[*keys, "status"]], on=keys, how="left")
    merged["status"] = merged["status"].fillna("pending")
    table = (
        merged.groupby(["fleet", "family", "status"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["ok", "failed", "pending"], fill_value=0)
    )
    table["expected"] = table.sum(axis=1)
    stream = {f: s for s in ("cpu", "gpu") for f in STREAMS[s]}
    table["stream"] = [stream[f] for f in table.index.get_level_values("family")]
    print(table.to_string())
    print()
    by_stream = table.groupby("stream")[["ok", "failed", "pending", "expected"]].sum()
    print(by_stream.to_string())
    per_repeat = len(jobs) // cfg.crossfit.repeats
    print(
        f"\n{root}: {int(table['ok'].sum())} finished, {int(table['failed'].sum())} failed, "
        f"{int(table['pending'].sum())} pending of {len(jobs)} expected "
        f"(R = {cfg.crossfit.repeats}, configurations 8-11 dropped: "
        f"{cfg.crossfit.drop_late_configs}); totals by rule: R = 3 {3 * per_repeat}, "
        f"R = 2 {2 * per_repeat}"
    )
    unplanned = len(found) - int((merged["status"] != "pending").sum())
    if unplanned:
        print(
            f"{unplanned} sidecars are not part of the plan (other repeats or configs)"
        )
    failed = merged[merged["status"] == "failed"]
    if not failed.empty:
        print("failed fits:")
        print(failed.to_string(index=False))


if __name__ == "__main__":
    main()
