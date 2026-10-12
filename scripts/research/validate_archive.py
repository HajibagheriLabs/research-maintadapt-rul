"""Validate a prediction archive: every engine of each split once per fit, cycles 1..T,
no duplicates, the true RUL, finite predictions in [0, cap], and pool hashes.

Reads truth and predictions only to check their structure; it computes no metric.
"""

import argparse

from maintadapt.research.archive import validate
from maintadapt.research.config import load_research_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--root", help="archive directory (default: paths.archive)")
    parser.add_argument("--fleets", nargs="+")
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    root = args.root or cfg.paths.archive

    problems = []
    for fleet in args.fleets or cfg.data.fleets:
        found, counts = validate(fleet, root, cfg)
        print(
            f"{fleet}: {counts['ok']} fits checked, {counts['failed']} failed fits, "
            f"{counts['stray_tmp']} stray .tmp files, {len(found)} problems"
        )
        problems += [f"{fleet} {p}" for p in found]
    if problems:
        raise SystemExit(
            "\n".join(problems[:50]) + f"\n{len(problems)} problems in {root}"
        )
    print(f"{root} is valid")


if __name__ == "__main__":
    main()
