"""Prediction archive: one parquet file and one JSON sidecar per fit, written atomically."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from maintadapt.data import Subset, load_subset
from maintadapt.research.config import ResearchConfig
from maintadapt.research.fold_data import PREDICTED, FoldData
from maintadapt.research.folds import Fold, load_folds
from maintadapt.research.models.spaces import config_hash, load_pool

SCHEMA = pa.schema(
    [
        ("unit_id", pa.int32()),
        ("cycle", pa.int32()),
        # uncapped truth: decision outcomes need the real end of life
        ("rul", pa.int32()),
        ("split", pa.dictionary(pa.int8(), pa.string())),
        ("pred", pa.float32()),
    ]
)
# columns load() adds from the sidecars
KEYS = pa.schema(
    [
        ("fleet", pa.string()),
        ("family", pa.string()),
        ("config_id", pa.string()),
        ("repeat", pa.int32()),
        ("fold", pa.int32()),
    ]
)
OK, FAILED = "ok", "failed"


@dataclass(frozen=True)
class FitKey:
    fleet: str
    family: str
    config_id: str
    # file name without suffix: r<repeat>_f<fold> in the cross-fitting archive
    stem: str

    def directory(self, root: str | Path) -> Path:
        return Path(root) / self.fleet / self.family / self.config_id

    def parquet(self, root: str | Path) -> Path:
        return self.directory(root) / f"{self.stem}.parquet"

    def sidecar(self, root: str | Path) -> Path:
        return self.directory(root) / f"{self.stem}.json"


def crossfit_key(
    fleet: str, family: str, config_id: str, repeat: int, fold: int
) -> FitKey:
    return FitKey(fleet, family, config_id, f"r{repeat}_f{fold}")


def prediction_table(data: FoldData, predictions: dict[str, np.ndarray]) -> pa.Table:
    """Every cycle of every inner_val, outer_test and official_test engine, in frame order."""
    parts = []
    for split in PREDICTED:
        frame, pred = data.frame(split), np.asarray(predictions[split])
        if pred.shape != (len(frame),):
            raise ValueError(f"{split}: {pred.shape} predictions for {len(frame)} rows")
        parts.append(
            pd.DataFrame(
                {
                    "unit_id": frame["unit_id"].to_numpy(np.int32),
                    "cycle": frame["cycle"].to_numpy(np.int32),
                    "rul": frame["rul"].to_numpy(np.int32),
                    "split": split,
                    "pred": pred.astype(np.float32),
                }
            )
        )
    table = pd.concat(parts, ignore_index=True)
    # fixed categories, so every file carries the same dictionary
    table["split"] = pd.Categorical(table["split"], categories=PREDICTED)
    return pa.Table.from_pandas(table, schema=SCHEMA, preserve_index=False)


def _replace_atomically(path: Path, write: Any) -> None:
    # a crash leaves at most a .tmp file behind, never a truncated file under the real name
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def write_fit(
    root: str | Path, key: FitKey, table: pa.Table | None, record: dict[str, Any]
) -> None:
    """Write the predictions, then the sidecar that marks the fit as finished."""
    directory = key.directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    parquet = key.parquet(root)
    if record["status"] == OK:
        if table is None:
            raise ValueError("a successful fit needs its prediction table")
        _replace_atomically(
            parquet, lambda p: pq.write_table(table, p, compression="zstd")
        )
    else:
        parquet.unlink(missing_ok=True)
    text = json.dumps(record, indent=1, sort_keys=False) + "\n"
    _replace_atomically(
        key.sidecar(root),
        lambda p: p.write_text(text, encoding="utf-8", newline="\n"),
    )


def read_record(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fit_status(root: str | Path, key: FitKey) -> str | None:
    """ok or failed once the fit is finished; None while it is pending."""
    path = key.sidecar(root)
    return read_record(path)["status"] if path.exists() else None


def records(root: str | Path, fleets: Sequence[str] | None = None) -> pd.DataFrame:
    """Every sidecar under root, one row each, with the path of its parquet file."""
    rows = []
    for fleet_dir in sorted(Path(root).glob("*")) if Path(root).exists() else []:
        if not fleet_dir.is_dir() or (
            fleets is not None and fleet_dir.name not in fleets
        ):
            continue
        for path in sorted(fleet_dir.glob("*/*/*.json")):
            record = read_record(path)
            record["parquet"] = str(path.with_suffix(".parquet"))
            rows.append(record)
    return pd.DataFrame(rows)


def load(
    root: str | Path,
    *,
    fleets: Iterable[str] | None = None,
    families: Iterable[str] | None = None,
    config_ids: Iterable[str] | None = None,
    repeats: Iterable[int] | None = None,
    folds: Iterable[int] | None = None,
    splits: Iterable[str] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Predictions of the finished fits that match every given filter.

    Fits are chosen by their sidecars; rows by split through a pyarrow filter. The key columns
    fleet, family, config_id, repeat and fold can be selected like the stored ones.
    """
    found = records(root, None if fleets is None else list(fleets))
    if found.empty:
        return pd.DataFrame(columns=list(columns or [*KEYS.names, *SCHEMA.names]))
    keep = found["status"] == OK
    for name, values in (
        ("family", families),
        ("config_id", config_ids),
        ("repeat", repeats),
        ("fold", folds),
    ):
        if values is not None:
            keep &= found[name].isin(list(values))
    chosen = found[keep]
    if chosen.empty:
        return pd.DataFrame(columns=list(columns or [*KEYS.names, *SCHEMA.names]))
    partitions = [
        (ds.field("fleet") == r["fleet"])
        & (ds.field("family") == r["family"])
        & (ds.field("config_id") == r["config_id"])
        & (ds.field("repeat") == int(r["repeat"]))
        & (ds.field("fold") == int(r["fold"]))
        for r in chosen.to_dict("records")
    ]
    dataset = ds.FileSystemDataset.from_paths(
        chosen["parquet"].tolist(),
        schema=pa.schema([*SCHEMA, *KEYS]),
        format=ds.ParquetFileFormat(),
        filesystem=pa.fs.LocalFileSystem(),
        partitions=partitions,
    )
    row_filter = None if splits is None else ds.field("split").isin(list(splits))
    table = dataset.to_table(columns=columns, filter=row_filter)
    return table.to_pandas()


def expected_rows(subset: Subset, fold: Fold) -> pd.DataFrame:
    """The unit_id, cycle, rul and split columns every fit of this fold must hold."""
    members = {"inner_val": fold.inner_val, "outer_test": fold.outer_test}
    parts = []
    for split in PREDICTED:
        if split == "official_test":
            frame = subset.test
        else:
            frame = subset.train[subset.train["unit_id"].isin(members[split])]
        frame = frame.sort_values(["unit_id", "cycle"])
        parts.append(frame[["unit_id", "cycle", "rul"]].assign(split=split))
    return pd.concat(parts, ignore_index=True)


def check_fit(table: pd.DataFrame, expected: pd.DataFrame, cap: float) -> list[str]:
    """Problems of one fit's predictions against the rows it must hold."""
    problems = []
    if list(table.columns) != SCHEMA.names:
        return [f"columns {list(table.columns)}, expected {SCHEMA.names}"]
    split = table["split"].astype(str)
    duplicated = table.duplicated(["split", "unit_id", "cycle"])
    if duplicated.any():
        problems.append(f"{int(duplicated.sum())} duplicated (split, unit, cycle) rows")
    for name in PREDICTED:
        got = table[split == name]
        want = expected[expected["split"] == name]
        missing = sorted(set(want["unit_id"]) - set(got["unit_id"]))
        extra = sorted(set(got["unit_id"]) - set(want["unit_id"]))
        if missing or extra:
            problems.append(f"{name}: engines missing {missing}, unexpected {extra}")
            continue
        # contiguous cycles 1..T in order, with T the engine's last recorded cycle
        same = len(got) == len(want) and all(
            np.array_equal(got[c].to_numpy(np.int64), want[c].to_numpy(np.int64))
            for c in ("unit_id", "cycle", "rul")
        )
        if not same:
            problems.append(
                f"{name}: cycles or truth differ from cycles 1..T per engine"
            )
    pred = table["pred"].to_numpy(float)
    if not np.isfinite(pred).all():
        problems.append(f"{int((~np.isfinite(pred)).sum())} non-finite predictions")
    elif pred.min() < 0 or pred.max() > cap:
        problems.append(f"predictions outside [0, {cap}]")
    return problems


def validate_fleet(
    root: str | Path,
    fleet: str,
    subset: Subset,
    repeats: list[list[Fold]],
    cap: float,
    hashes: dict[str, str],
) -> tuple[list[str], dict[str, int]]:
    """Check every finished fit of a fleet; returns the problems and counts of what was seen.

    hashes maps every configuration id the archive may hold to its content hash.
    """
    found = records(root, [fleet])
    problems, counts = [], {"ok": 0, "failed": 0, "stray_tmp": 0, "orphan_parquet": 0}
    fleet_dir = Path(root) / fleet
    counts["stray_tmp"] = len(list(fleet_dir.glob("*/*/*.tmp")))
    sidecars = {Path(p).with_suffix("") for p in found.get("parquet", [])}
    for parquet in fleet_dir.glob("*/*/*.parquet"):
        if parquet.with_suffix("") not in sidecars:
            counts["orphan_parquet"] += 1
            problems.append(f"{parquet}: predictions without a sidecar")
    expected: dict[tuple[int, int], pd.DataFrame] = {}
    for record in found.to_dict("records"):
        where = f"{record['config_id']} r{record['repeat']} f{record['fold']}"
        if record["status"] != OK:
            counts["failed"] += 1
            continue
        counts["ok"] += 1
        if hashes.get(record["config_id"]) != record["config_hash"]:
            problems.append(f"{where}: not a configuration of the frozen pool")
        key = (int(record["repeat"]), int(record["fold"]))
        if key not in expected:
            expected[key] = expected_rows(subset, repeats[key[0]][key[1]])
        table = pq.read_table(record["parquet"]).to_pandas()
        found_problems = check_fit(table, expected[key], cap)
        if record.get("n_rows") is not None and record["n_rows"] != len(table):
            found_problems.append(
                f"sidecar says {record['n_rows']} rows, file has {len(table)}"
            )
        problems += [f"{where}: {p}" for p in found_problems]
    return problems, counts


def validate(
    fleet: str, root: str | Path, cfg: ResearchConfig
) -> tuple[list[str], dict[str, int]]:
    """validate_fleet against the raw data, the committed folds and the frozen pool."""
    from maintadapt.research.models.registry import CONTROL_FAMILIES

    subset = load_subset(cfg.data.raw_dir, fleet, cfg.data.cap)
    folds = load_folds(cfg.paths.folds, cfg.folds, cfg.data.fleets)[fleet]
    pool = load_pool(cfg.paths.pool, cfg.pool.seed, cfg.pool.n_configs)
    hashes = {c.id: c.hash for c in pool}
    hashes |= {f"{name}-00": config_hash(name, {}) for name in CONTROL_FAMILIES}
    return validate_fleet(root, fleet, subset, folds, cfg.data.cap, hashes)
