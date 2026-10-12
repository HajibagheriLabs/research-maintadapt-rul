"""Check the raw C-MAPSS files against the data manifest, writing the manifest on the first run."""

import argparse
import json
from pathlib import Path

from maintadapt.research.config import load_research_config
from maintadapt.research.manifest import (
    build_manifest,
    compare_manifests,
    render_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    cfg = load_research_config(parser.parse_args().config)

    found = build_manifest(cfg.data.raw_dir, cfg.data.fleets)
    path = Path(cfg.paths.manifest)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_manifest(found), encoding="utf-8", newline="\n")
        print(f"wrote {path}: {len(found['files'])} files recorded")
        return
    problems = compare_manifests(json.loads(path.read_text(encoding="utf-8")), found)
    if problems:
        raise SystemExit(
            f"the raw data in {cfg.data.raw_dir} differ from {path}:\n  "
            + "\n  ".join(problems)
        )
    print(f"{len(found['files'])} files match {path}")


if __name__ == "__main__":
    main()
