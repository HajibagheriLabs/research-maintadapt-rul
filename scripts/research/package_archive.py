"""Package the prediction archives and the frozen design as one file in dist/ (make research-package).

Writes dist/<name>.tar.gz holding the confirmatory archive, the archives of the secondary fits
(S4 nested, S5 caps, N-CMAPSS), the fold files, the pool, the data manifest, README_ARCHIVE.md,
LICENSE_ARCHIVE.md and SHA256SUMS (one line per file in the package). The tarball's own SHA-256
goes to dist/<name>.tar.gz.sha256 and README_ARCHIVE.md is copied beside it. Entries are sorted
and carry no times or owners, so the same inputs give the same bytes. --smoke packages the smoke
archive instead, to check the packaging in seconds.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from collections.abc import Iterable
from pathlib import Path

from maintadapt.research.config import ResearchConfig, load_research_config
from maintadapt.research.crossfit import git_commit

CHUNK = 1 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def contents(cfg: ResearchConfig, smoke: bool) -> dict[str, Path]:
    """Name in the package -> file on disk, for every file that goes in."""
    if smoke:
        archives = {"archive_smoke": Path(cfg.paths.smoke)}
    else:
        archives = {
            "archive": Path(cfg.paths.archive),
            "archive_nested": Path(cfg.paths.archive_nested),
            "archive_caps": Path(cfg.paths.archive_caps),
            "archive_ncmapss": Path(cfg.ncmapss.archive),
        }
    files: dict[str, Path] = {}
    for name, root in archives.items():
        if not root.is_dir():
            raise SystemExit(f"{root} does not exist")
        stray = sorted(root.rglob("*.tmp"))
        if stray:
            raise SystemExit(f"{root} holds unfinished writes, for example {stray[0]}")
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.suffix == ".parquet" and not path.with_suffix(".json").exists():
                raise SystemExit(f"{path} has no sidecar")
            files[f"{name}/{path.relative_to(root).as_posix()}"] = path
    for path in (
        cfg.paths.folds,
        cfg.ncmapss.folds,
        cfg.paths.pool,
        cfg.paths.manifest,
    ):
        files[f"design/{Path(path).name}"] = Path(path)
    return files


def fit_counts(files: dict[str, Path]) -> dict[str, dict[str, int]]:
    """Sidecars per archive and status."""
    counts: dict[str, dict[str, int]] = {}
    for name, path in files.items():
        if not name.endswith(".json") or name.startswith("design/"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if "status" not in record:
            continue
        archive = name.split("/", 1)[0]
        status = counts.setdefault(archive, {})
        status[record["status"]] = status.get(record["status"], 0) + 1
    return counts


def readme(counts: dict[str, dict[str, int]], commit: str, smoke: bool) -> str:
    rows = "\n".join(
        f"| `{archive}/` | {sum(c.values()):,} | "
        + ", ".join(f"{k} {v:,}" for k, v in sorted(c.items()))
        + " |"
        for archive, c in sorted(counts.items())
    )
    note = (
        "**SMOKE PACKAGE: the smoke archive only, for checking the packaging.**\n\n"
        if smoke
        else ""
    )
    return f"""# Prediction archive of research-maintadapt-rul

{note}Predictions of every fit of the pre-registered study "Do remaining-useful-life
leaderboards predict maintenance value?" (pre-registration: https://osf.io/mh9zy,
DOI 10.17605/OSF.IO/MH9ZY), packaged by `make research-package` at commit {commit} of
https://github.com/HajibagheriLabs/research-maintadapt-rul. Archive DOI: ZENODO-DOI-PLACEHOLDER.
Licence: CC BY 4.0 (LICENSE_ARCHIVE.md). The code that wrote and reads it is MIT-licensed in the
repository.

## Contents

| folder | fits | status |
|---|---|---|
{rows}

- `archive/`: the confirmatory cross-fitting archive: FD001-FD004, 113 configurations,
  3 repeats, 5 outer folds.
- `archive_nested/`: S4 nested tuning, one archive per outer fold (`outer<k>/`) of repeat 0 on
  FD001 and FD003; `f<j>` in a file name is the inner fold, and `nested_folds.json` lists the
  inner folds' engines.
- `archive_caps/`: S5, the reference XGBoost and LSTM trained with caps 100, 150, 200 and none
  (`<cap>/`), repeat 0.
- `archive_ncmapss/`: S8, the pooled N-CMAPSS fleet `NCMAPSS` (99 units, one row per flight).
- `design/`: the fold files, the frozen pool and the data manifest, whose SHA-256 values are
  quoted in PREREGISTRATION.md.
- `SHA256SUMS`: the SHA-256 of every other file, checked with `sha256sum -c SHA256SUMS`.

## Layout and schema

Each fit is one Parquet file (zstd) and one JSON sidecar:
`<archive>/<fleet>/<family>/<config_id>/r<repeat>_f<fold>.parquet` and `.json`. A fit is
finished when its sidecar exists; a failed fit has a sidecar and no Parquet file.

Parquet columns, one row per cycle (per flight on N-CMAPSS) of every engine predicted:

| column | type | meaning |
|---|---|---|
| `unit_id` | int32 | engine id (N-CMAPSS: 100 x file position + unit) |
| `cycle` | int32 | cycle t, 1..T for training engines; 1..last observed for official test engines |
| `rul` | int32 | uncapped true RUL at that cycle, T - t (official test: from the RUL files) |
| `split` | dictionary(int8, string) | `inner_val`, `outer_test` or `official_test` |
| `pred` | float32 | predicted RUL, clipped to [0, cap] |

Engines of `inner_fit` are never predicted. N-CMAPSS has no official test engines.

Sidecar fields: `fleet`, `family`, `config_id`, `config_hash` (of the pool entry), `repeat`,
`fold` (and `outer_fold` in the nested archives), `seed`, `status` (`ok` or `failed`),
`attempts`, `error`, `fit_seconds`, `wall_seconds`, `best_iteration_or_epoch`, `val_rmse`
(inner_val RMSE against the capped truth), `n_inputs`, `n_rows`, `device`, `python`, `packages`,
`hardware`, `git_commit` (the commit the fit ran from) and `started_at` (UTC); the secondary
archives add the fields of their analysis, such as `training_cap`.

## Loading

```python
import pandas as pd

fit = pd.read_parquet("archive/FD001/xgboost/xgboost-00/r0_f0.parquet")
outer = fit[fit["split"] == "outer_test"].sort_values(["unit_id", "cycle"])
```

With the repository installed, `maintadapt.research.archive.load(root, fleets=..., config_ids=...,
splits=...)` reads many fits at once and adds the sidecar keys as columns; copy a folder to
`runs/research/<folder>` to rebuild the derived tables and every report with the make targets of
README.md.
"""


LICENCE = """# Licence of the prediction archive

The predictions, sidecars and design files in this package are licensed under the Creative
Commons Attribution 4.0 International licence (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/

Cite the archive DOI (ZENODO-DOI-PLACEHOLDER) and the repository's CITATION.cff. The C-MAPSS and
N-CMAPSS data the models were trained on are NASA's and are not included.
"""


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), 0o644, 0
    tar.addfile(info, io.BytesIO(data))


def add_file(tar: tarfile.TarFile, name: str, path: Path) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
    with open(path, "rb") as fh:
        tar.addfile(info, fh)


def write_package(
    target: Path, prefix: str, files: dict[str, Path], extra: dict[str, bytes]
) -> None:
    """A gzip tarball whose bytes depend on the contents alone."""
    tmp = target.with_name(target.name + ".tmp")
    with (
        open(tmp, "wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0
        ) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for name in sorted(extra):
            add_bytes(tar, f"{prefix}/{name}", extra[name])
        for name in sorted(files):
            add_file(tar, f"{prefix}/{name}", files[name])
    tmp.replace(target)


def checksums(lines: Iterable[tuple[str, str]]) -> str:
    return "".join(f"{digest}  {name}\n" for name, digest in sorted(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--out", default="dist")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    commit = git_commit()
    name = (
        "research-maintadapt-rul-smoke-archive"
        if args.smoke
        else "research-maintadapt-rul-predictions"
    )
    files = contents(cfg, args.smoke)
    print(f"{len(files):,} files; hashing", flush=True)
    sums = [(n, sha256(p)) for n, p in files.items()]
    text = readme(fit_counts(files), commit, args.smoke)
    extra = {"README_ARCHIVE.md": text.encode(), "LICENSE_ARCHIVE.md": LICENCE.encode()}
    sums += [(n, hashlib.sha256(b).hexdigest()) for n, b in extra.items()]
    extra["SHA256SUMS"] = checksums(sums).encode()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{name}.tar.gz"
    print(f"writing {target}", flush=True)
    write_package(target, name, files, extra)
    digest = sha256(target)
    (out / f"{target.name}.sha256").write_text(
        f"{digest}  {target.name}\n", encoding="utf-8", newline="\n"
    )
    (out / "README_ARCHIVE.md").write_text(text, encoding="utf-8", newline="\n")
    size = target.stat().st_size / 2**20
    print(f"{target}: {size:,.1f} MiB, SHA-256 {digest}")


if __name__ == "__main__":
    main()
