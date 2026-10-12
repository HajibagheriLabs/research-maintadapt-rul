"""Integrity manifest of the raw C-MAPSS files, hashed after parsing rather than as bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from maintadapt.data import read_final_rul, read_trajectories

HASHED = "SHA-256 of the parsed values as a row-major little-endian float64 array"


def array_sha256(values: np.ndarray) -> str:
    data = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(data.tobytes()).hexdigest()


def file_entry(path: Path) -> dict:
    """Rows, units and value hash of one file, parsed with the legacy parsers."""
    if path.name.startswith("RUL_"):
        final = read_final_rul(path)
        return {
            "rows": len(final),
            "units": len(final),
            "sha256": array_sha256(final.to_numpy()),
        }
    frame = read_trajectories(path)
    return {
        "rows": len(frame),
        "units": int(frame["unit_id"].nunique()),
        "sha256": array_sha256(frame.to_numpy(dtype=float)),
    }


def build_manifest(raw_dir: str | Path, fleets: list[str]) -> dict:
    root = Path(raw_dir)
    files = {}
    for fleet in fleets:
        for kind in ("train", "test", "RUL"):
            name = f"{kind}_{fleet}.txt"
            files[name] = file_entry(root / name)
    return {"parser": "maintadapt.data", "hash": HASHED, "files": files}


def render_manifest(manifest: dict) -> str:
    return json.dumps(manifest, indent=2) + "\n"


def compare_manifests(expected: dict, actual: dict) -> list[str]:
    """Human-readable differences; empty when the data match the recorded manifest."""
    problems = []
    if expected.get("hash") != actual["hash"]:
        problems.append("the manifest was hashed another way")
    recorded, found = expected.get("files", {}), actual["files"]
    for name in sorted(set(recorded) | set(found)):
        if name not in found:
            problems.append(f"{name}: recorded but not checked")
        elif name not in recorded:
            problems.append(f"{name}: not in the recorded manifest")
        else:
            problems += [
                f"{name}: {key} is {found[name][key]}, recorded {recorded[name][key]}"
                for key in ("rows", "units", "sha256")
                if found[name][key] != recorded[name][key]
            ]
    return problems
