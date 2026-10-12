"""Write configs/research_pool.yaml: the frozen pool of model configurations."""

import argparse
from collections import Counter

from maintadapt.research.config import load_research_config
from maintadapt.research.models.spaces import load_pool, save_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    cfg = load_research_config(parser.parse_args().config)

    save_pool(cfg.pool.seed, cfg.pool.n_configs, cfg.paths.pool)
    pool = load_pool(cfg.paths.pool, cfg.pool.seed, cfg.pool.n_configs)
    counts = Counter(c.family for c in pool)
    print(", ".join(f"{family} {n}" for family, n in counts.items()))
    print(f"wrote {cfg.paths.pool}: {len(pool)} configurations")


if __name__ == "__main__":
    main()
