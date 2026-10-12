"""Write configs/research_folds.json: lifetime-stratified, repeated outer and inner folds."""

import argparse

from maintadapt.data import load_subset
from maintadapt.research.config import load_research_config
from maintadapt.research.folds import (
    SPLITS,
    lifetimes,
    load_folds,
    make_folds,
    save_folds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    cfg = load_research_config(parser.parse_args().config)

    lives = {
        fleet: lifetimes(load_subset(cfg.data.raw_dir, fleet, cfg.data.cap).train)
        for fleet in cfg.data.fleets
    }
    save_folds(make_folds(lives, cfg.folds), cfg.folds, cfg.paths.folds)
    # reading the file back runs every structural check the loaders apply later
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)
    for fleet, repeats in folds.items():
        sizes = {
            name: sorted({len(getattr(f, name)) for r in repeats for f in r})
            for name in SPLITS
        }
        text = ", ".join(f"{name} {'-'.join(map(str, s))}" for name, s in sizes.items())
        print(f"{fleet}: {len(lives[fleet])} engines; per fold {text}")
    print(f"wrote {cfg.paths.folds}")


if __name__ == "__main__":
    main()
