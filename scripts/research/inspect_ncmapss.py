"""Inspect the N-CMAPSS files before anything is assumed about them, and compute the cap.

Lists every file's groups with their shapes and types; reads A, Y and the first and last rows of
T one unit at a time (never W, X_s or X_v, never a whole file); and reports per sub-dataset the
development and test units, flights per unit, flight classes, the health parameters that
degrade (the failure modes) and any file that fails to load, which is excluded and listed.
Computes the cap of PREREGISTRATION.md section 8 from the readable units: the median RUL at the
first abnormal health state, rounded to 5 cycles. Writes reports/research/ncmapss/inspection.md
with inspection_files.csv and inspection_units.csv. No model is trained here.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from maintadapt.research.config import load_research_config
from maintadapt.research.ncmapss import (
    CAP_STEP,
    GROUPS,
    INPUT_GROUPS,
    PARTS,
    cap_from_onsets,
    health_changes,
    names,
    open_failure,
    read_labels,
    sub_datasets,
    summarise,
    unit_blocks,
    unit_id,
)


def groups_table(path: Path) -> list[dict]:
    with h5py.File(path, "r") as h:
        return [
            {
                "file": path.name,
                "group": key,
                "shape": "x".join(str(n) for n in h[key].shape),
                "dtype": str(h[key].dtype),
            }
            for key in sorted(h.keys())
        ]


def units_table(sub) -> list[dict]:
    rows = []
    with h5py.File(sub.path, "r") as h:
        for part in PARTS:
            for block in unit_blocks(h, part):
                a, y = read_labels(h, block)
                s = summarise(a, y)
                rows.append(
                    {
                        "sub_dataset": sub.name,
                        "part": part,
                        "unit": block.unit,
                        "unit_id": unit_id(sub, block.unit),
                        "samples": block.stop - block.start,
                        "flights": s.flights,
                        "flight_classes": " ".join(map(str, s.flight_classes)),
                        "rul_is_life_minus_cycle": s.rul_is_life_minus_cycle,
                        "health_state_monotone": s.health_state_monotone,
                        "onset_flight": s.onset_flight,
                        "onset_rul": s.onset_rul,
                        "degraded": " ".join(health_changes(h, block)),
                    }
                )
    return rows


def report(
    files: pd.DataFrame,
    found: pd.DataFrame,
    failed: list[tuple[str, str]],
    columns: dict[str, list[str]],
    cap: int,
) -> str:
    per_sub = found.groupby("sub_dataset", sort=True)
    lines = [
        "# N-CMAPSS inspection",
        "",
        (
            "Written by `make research-ncmapss-inspect` (`scripts/research/inspect_ncmapss.py`) "
            "from the files in `data/raw/NCMAPSS/data_set/`; no model was trained. Groups and "
            "shapes of every file are in `inspection_files.csv`, one row per unit in "
            "`inspection_units.csv`."
        ),
        "",
        "## Files that fail to load (excluded)",
        "",
        *([f"- `{name}`: {why}" for name, why in failed] or ["- none"]),
        "",
        "## Layout, as found",
        "",
        (
            "Every readable file holds the groups "
            + ", ".join(f"`{g}_dev`/`{g}_test`" for g in GROUPS)
            + " and their `_var` names, with the same number of rows within each part. Columns:"
        ),
        "",
        *[f"- `{g}`: {', '.join(columns[g])}" for g in columns],
        "",
        (
            "Model inputs come from "
            + " and ".join(f"`{g}`" for g in INPUT_GROUPS)
            + " only; `X_v`, `T` and the health state `hs` never become inputs."
        ),
        "",
        "## Sub-datasets",
        "",
        (
            "| sub-dataset | dev units | test units | flights per unit (min-max) | flight "
            "classes | degraded health parameters |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for name, g in per_sub:
        dev = g.loc[g["part"] == "dev", "unit"].tolist()
        test = g.loc[g["part"] == "test", "unit"].tolist()
        modes = sorted(set(g["degraded"]))
        lines.append(
            f"| {name} | {len(dev)} ({', '.join(map(str, dev))}) | {len(test)} "
            f"({', '.join(map(str, test))}) | {g['flights'].min()}-{g['flights'].max()} | "
            f"{', '.join(sorted(set(' '.join(g['flight_classes']).split())))} | "
            f"{'; '.join(m or 'none' for m in modes)} |"
        )
    onsets = found["onset_rul"].dropna()
    checks = {
        "units whose rows form one contiguous block, flights 1..T in order": len(found),
        "units with RUL = T - t at every flight, constant within flights": int(
            found["rul_is_life_minus_cycle"].sum()
        ),
        "units whose health state only falls from 1 to 0": int(
            found["health_state_monotone"].sum()
        ),
        "units with one flight class": int(
            (~found["flight_classes"].str.contains(" ")).sum()
        ),
        "units that reach an abnormal health state": len(onsets),
    }
    lines += [
        "",
        f"{len(found)} units in {found['sub_dataset'].nunique()} readable sub-datasets.",
        "",
        "| check | units |",
        "|---|---|",
        *[f"| {k} | {v} of {len(found)} |" for k, v in checks.items()],
        "",
        "## Cap (PREREGISTRATION.md section 8)",
        "",
        (
            f"RUL at the first flight with health state 0, over the {len(onsets)} units: "
            f"median {np.median(onsets):g}, quartiles {np.quantile(onsets, 0.25):g} and "
            f"{np.quantile(onsets, 0.75):g}, range {onsets.min():g}-{onsets.max():g}. Rounded "
            f"to {CAP_STEP} cycles, the cap is **{cap}**."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    args = parser.parse_args()
    cfg = load_research_config(args.config)
    out = Path(cfg.paths.reports) / "ncmapss"
    out.mkdir(parents=True, exist_ok=True)
    files, found, failed, columns = [], [], [], {}
    for sub in sub_datasets(cfg.ncmapss.raw_dir):
        why = open_failure(sub.path)
        if why is not None:
            failed.append((sub.path.name, why))
            print(f"{sub.name}: excluded ({why})", flush=True)
            continue
        files += groups_table(sub.path)
        with h5py.File(sub.path, "r") as h:
            seen = {g: names(h, g) for g in GROUPS if g != "Y"}
        if columns and seen != columns:
            raise SystemExit(f"{sub.name}: columns differ from the other files")
        columns = seen
        found += units_table(sub)
        print(
            f"{sub.name}: {sum(r['sub_dataset'] == sub.name for r in found)} units",
            flush=True,
        )
    found = pd.DataFrame(found)
    cap = cap_from_onsets(found["onset_rul"].dropna().astype(int).tolist())
    pd.DataFrame(files).to_csv(
        out / "inspection_files.csv", index=False, lineterminator="\n"
    )
    found.to_csv(out / "inspection_units.csv", index=False, lineterminator="\n")
    text = report(pd.DataFrame(files), found, failed, columns, cap)
    (out / "inspection.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
