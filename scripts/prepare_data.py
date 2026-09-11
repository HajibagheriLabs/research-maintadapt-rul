"""Check the raw C-MAPSS files and write deterministic unit-level splits."""

import argparse

from maintadapt.config import load_config
from maintadapt.data import load_subset, make_splits, save_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(parser.parse_args().config)

    units = {}
    for name in cfg.data.subsets:
        subset = load_subset(cfg.paths.raw_dir, name, cfg.data.rul_cap)
        units[name] = subset.train["unit_id"].unique()
        n_test = subset.test["unit_id"].nunique()
        print(f"{name}: {len(units[name])} run-to-failure units, {n_test} test units")

    splits = make_splits(units, cfg.data.val_fraction, cfg.data.split_seed)
    save_splits(splits, cfg.paths.splits, cfg.data.split_seed, cfg.data.val_fraction)
    for name, parts in splits.items():
        print(
            f"{name}: {len(parts['train'])} train / {len(parts['val'])} validation units"
        )
    print(f"wrote {cfg.paths.splits}")


if __name__ == "__main__":
    main()
