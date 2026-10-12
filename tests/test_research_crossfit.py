import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from synthetic_fleet import CFG, ROOT, TRAINING, fold_data, repeats, subset

from maintadapt.research.archive import (
    FAILED,
    OK,
    FitKey,
    crossfit_key,
    fit_status,
    load,
)
from maintadapt.research.compute import decide_repeats, project_hours, stream_totals
from maintadapt.research.crossfit import (
    Job,
    control_configs,
    describe,
    fit_seed,
    plan,
    run_fit,
    run_jobs,
    select_configs,
)
from maintadapt.research.leaderboard import (
    compare_with_legacy,
    last_cycle_metrics,
    per_seed_metrics,
)
from maintadapt.research.models.registry import FitOutput, fit
from maintadapt.research.models.spaces import FAMILY_NAMES, load_pool

POOL = load_pool(ROOT / CFG.paths.pool, CFG.pool.seed, CFG.pool.n_configs)
CPU = torch.device("cpu")
ENV = {"python": "3.11", "packages": {}, "hardware": {}, "git_commit": "test"}


def test_seed_is_the_first_31_bits_of_the_sha256_of_the_fit_ids() -> None:
    # sha256("FD001|0|0|ridge-00") starts 46e17c2c; 0x46e17c2c = 1189182508, halved 594591254
    assert fit_seed("FD001", 0, 0, "ridge-00") == 594591254
    seeds = {
        fit_seed("FD004", r, f, c.id) for r in range(3) for f in range(5) for c in POOL
    }
    assert len(seeds) == 15 * len(POOL) and max(seeds) < 2**31


def test_plan_counts_and_orders_every_registered_fit() -> None:
    every = select_configs(POOL, FAMILY_NAMES)
    jobs = plan(every, CFG.data.fleets, range(3), range(5))
    assert len(jobs) == 6780
    assert len(plan(every, CFG.data.fleets, range(2), range(5))) == 4520
    # fleet, then repeat, then fold, then family and configuration in pool order
    assert [j.config.id for j in jobs[:6]] == [f"ridge-0{i}" for i in range(5)] + [
        "knn-00"
    ]
    assert (jobs[112].fleet, jobs[112].fold, jobs[113].fold) == ("FD001", 0, 1)
    assert jobs[113 * 15].fleet == "FD002"
    # dropping configurations 8-11 leaves ridge's five and eight of every other family
    kept = select_configs(POOL, FAMILY_NAMES, None, lambda n: 8 if n == 12 else n)
    assert Counter(c.family for c in kept) == {
        f: 5 if f == "ridge" else 8 for f in FAMILY_NAMES
    }
    smoke = select_configs(POOL, ["ridge", "lstm"], [0])
    assert [c.id for c in smoke] == ["ridge-00", "lstm-00"]


def _oracle(data, seed: int) -> FitOutput:
    return fit("control_oracle", {}, data, seed, CPU, TRAINING)


def _identity(job: Job) -> dict:
    c = job.config
    return {
        "fleet": job.fleet,
        "family": c.family,
        "config_id": c.id,
        "config_hash": c.hash,
        "repeat": job.repeat,
        "fold": job.fold,
    }


def _job(config_index: int = 0, fold: int = 0) -> Job:
    return Job("FD001", 0, fold, control_configs()[config_index])


def test_a_failed_attempt_is_retried_once_with_seed_plus_1000(tmp_path: Path) -> None:
    data, job, seeds = fold_data(subset()), _job(), []

    def flaky(seed: int) -> FitOutput:
        seeds.append(seed)
        if len(seeds) == 1:
            raise RuntimeError("CUDA out of memory")
        return _oracle(data, seed)

    record = run_fit(tmp_path, job.key, _identity(job), data, flaky, 11, ENV, "cpu")
    assert seeds == [11, 1011]
    assert (record["status"], record["seed"], record["attempts"]) == (OK, 1011, 2)
    assert "RuntimeError: CUDA out of memory" in record["error"]
    assert job.key.parquet(tmp_path).exists()
    saved = json.loads(job.key.sidecar(tmp_path).read_text(encoding="utf-8"))
    assert saved["val_rmse"] == 0.0 and saved["git_commit"] == "test"
    assert "attempts 2" in describe(job, record)


def test_non_finite_predictions_twice_record_a_failure(tmp_path: Path) -> None:
    data, job = fold_data(subset()), _job()

    def broken(seed: int) -> FitOutput:
        out = _oracle(data, seed)
        out.predictions["official_test"][0] = np.nan
        return out

    record = run_fit(tmp_path, job.key, _identity(job), data, broken, 5, ENV, "cpu")
    assert (record["status"], record["seed"], record["attempts"]) == (FAILED, 1005, 2)
    assert record["error"].count("non-finite") == 2
    assert fit_status(tmp_path, job.key) == FAILED
    assert not job.key.parquet(tmp_path).exists()


def test_an_interrupt_stops_the_run_and_leaves_the_fit_pending(tmp_path: Path) -> None:
    data, job = fold_data(subset()), _job()

    def interrupted(seed: int) -> FitOutput:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_fit(tmp_path, job.key, _identity(job), data, interrupted, 5, ENV, "cpu")
    assert fit_status(tmp_path, job.key) is None


def test_a_rerun_skips_finished_fits_and_builds_only_folds_with_work_left(
    tmp_path: Path,
) -> None:
    jobs = plan(control_configs(), ["FD001"], [0], [0, 1])
    built, fitted = [], []

    def build(fleet: str, repeat: int, fold: int):
        built.append(fold)
        return fold_data(subset(), repeats()[repeat][fold], repeat, fold)

    def fit_job(job: Job, data, seed: int) -> FitOutput:
        fitted.append(job.key)
        return fit(job.config.family, {}, data, seed, CPU, TRAINING)

    def run(limit=None) -> Counter:
        return run_jobs(
            jobs,
            tmp_path,
            build,
            fit_job,
            ENV,
            lambda f: "cpu",
            limit=limit,
            log=lambda _: None,
        )

    first = run(limit=3)
    assert (first["ran"], first["left"], built) == (3, 1, [0, 1])
    stamps = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*.json")}
    second = run()
    assert (second["ran"], second["skipped"], built) == (1, 3, [0, 1, 1])
    third = run()
    assert (third["ran"], third["skipped"], len(built), len(fitted)) == (0, 4, 3, 4)
    # every fit happened exactly once and the earlier files were never rewritten
    assert len(set(fitted)) == 4
    assert all(p.stat().st_mtime_ns == t for p, t in stamps.items())
    assert len(load(tmp_path)) == 2 * sum(
        len(fold_data(subset(), repeats()[0][f]).frame(s))
        for f in (0, 1)
        for s in ("inner_val", "outer_test", "official_test")
    )
    assert (
        fit_status(
            tmp_path,
            crossfit_key("FD001", "control_constant", "control_constant-00", 0, 1),
        )
        == OK
    )


def test_projection_scales_by_inner_fit_rows_and_multiplies_out_the_fits() -> None:
    dryrun = pd.DataFrame(
        [
            ("FD001", "ridge", 10.0),
            ("FD001", "ridge", 20.0),
            ("FD004", "ridge", 30.0),
            ("FD001", "lstm", 60.0),
            ("FD004", "lstm", 120.0),
        ],
        columns=["fleet", "family", "wall_seconds"],
    ).assign(status=OK)
    projection = project_hours(
        dryrun,
        {"cpu": ["ridge"], "gpu": ["lstm"]},
        {"ridge": 5, "lstm": 12},
        rows={"FD001": 1000, "FD002": 2000, "FD003": 1500, "FD004": 4000},
        measured_rows={"FD001": 1000, "FD004": 2000},
        n_outer=5,
    )
    ridge = projection[projection["family"] == "ridge"].set_index("fleet")
    # FD001 15 s; FD003 15 x 1.5; FD004 30 x 4000 / 2000; FD002 30 x 2000 / 2000
    assert ridge["seconds_per_fit"].to_dict() == {
        "FD001": 15.0,
        "FD002": 30.0,
        "FD003": 22.5,
        "FD004": 60.0,
    }
    totals = stream_totals(projection)
    # ridge: 127.5 s x 5 configs x 5 folds x 3 repeats = 9562.5 s
    assert totals.loc["cpu", "hours_r3"] == pytest.approx(9562.5 / 3600)
    # lstm: 510 s x 12 x 5 x 3 = 25.5 h; x 2 repeats = 17 h; 8 configs x 2 = 40800 s
    assert totals.loc["gpu", "hours_r3"] == pytest.approx(25.5)
    assert totals.loc["gpu", "hours_r2"] == pytest.approx(17.0)
    assert totals.loc["gpu", "hours_r2_dropped"] == pytest.approx(40800 / 3600)
    with pytest.raises(ValueError, match="FD004 lstm"):
        project_hours(
            dryrun[dryrun["wall_seconds"] != 120.0],
            {"gpu": ["lstm"]},
            {"lstm": 12},
            {"FD001": 1, "FD002": 1, "FD003": 1, "FD004": 1},
            {"FD001": 1, "FD004": 1},
            5,
        )


@pytest.mark.parametrize(
    ("cpu", "gpu", "expected"),
    [
        ((72.0, 48.0), (50.0, 33.0), (3, False)),
        ((72.1, 48.1), (50.0, 33.0), (2, False)),
        ((180.0, 120.0), (50.0, 33.0), (2, False)),
        ((180.0, 120.1), (50.0, 33.0), (2, True)),
        ((10.0, 7.0), (181.0, 121.0), (2, True)),
    ],
)
def test_repeats_follow_the_registered_72_and_120_hour_rule(cpu, gpu, expected) -> None:
    totals = pd.DataFrame(
        {
            "hours_r3": [cpu[0], gpu[0]],
            "hours_r2": [cpu[1], gpu[1]],
            "hours_r2_dropped": [1.0, 1.0],
        },
        index=["cpu", "gpu"],
    )
    decision = decide_repeats(totals)
    assert (decision.repeats, decision.drop_late_configs) == expected


def test_leaderboard_scores_the_last_observed_cycle_against_capped_truth() -> None:
    rows = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2, 1],
            "cycle": [1, 2, 3, 1, 2, 1],
            "rul": [12, 11, 10, 201, 200, 50],
            "split": ["official_test"] * 5 + ["inner_val"],
            "pred": [0.0, 0.0, 13.0, 0.0, 120.0, 999.0],
        }
    )
    metrics = last_cycle_metrics(rows, cap=125)
    # unit 1: 13 - 10 = +3 (late); unit 2: 120 - min(200, 125) = -5 (early)
    assert metrics["rmse"] == pytest.approx(np.sqrt((9 + 25) / 2))
    assert metrics["nasa_score"] == pytest.approx(np.expm1(3 / 10) + np.expm1(5 / 13))


def test_comparison_with_legacy_tables_flags_gaps_beyond_seed_noise() -> None:
    def table(rows):
        return pd.DataFrame(
            rows,
            columns=[
                "fleet",
                "family",
                "rmse_mean",
                "rmse_std",
                "nasa_score_mean",
                "nasa_score_std",
                "n_seeds",
            ],
        )

    reference = table(
        [
            ("FD001", "xgboost", 12.2, 0.1, 210.0, 5.0, 5),
            ("FD002", "ridge", 16.461, 0.0, 1180.0, 0.0, 5),
            ("FD001", "knn", 15.0, 0.2, 300.0, 9.0, 5),
        ]
    )
    legacy = table(
        [
            ("FD001", "xgboost", 12.14, 0.087, 213.547, 4.83, 5),
            ("FD002", "ridge", 15.961, 0.0, 1179.308, 0.0, 5),
        ]
    )
    out = compare_with_legacy(reference, legacy).set_index(["fleet", "family"])
    assert len(out) == 2
    xgb = out.loc[("FD001", "xgboost")]
    # se = sqrt(0.1^2 / 5 + 0.087^2 / 5) = 0.05928; 0.06 <= 2 se
    assert xgb["rmse_diff"] == pytest.approx(0.06)
    assert xgb["rmse_diff_se"] == pytest.approx(np.sqrt((0.01 + 0.087**2) / 5))
    assert bool(xgb["rmse_within_noise"])
    assert not bool(out.loc[("FD002", "ridge"), "rmse_within_noise"])


def test_leaderboard_reads_every_finished_seed_back_from_its_files(
    tmp_path: Path,
) -> None:
    data = fold_data(subset())
    for seed in (0, 1):
        job = _job(0)
        key = FitKey("FD001", "control_oracle", "control_oracle-00", f"seed{seed}")
        run_fit(
            tmp_path,
            key,
            _identity(job),
            data,
            lambda s: _oracle(data, s),
            seed,
            ENV,
            "cpu",
        )
    assert per_seed_metrics(tmp_path / "empty", 125).empty
    table = per_seed_metrics(tmp_path, 125)
    assert table["seed"].tolist() == [0, 1]
    # the oracle predicts the capped truth, so it scores perfectly at the last cycle
    assert (table["rmse"] == 0.0).all() and (table["nasa_score"] == 0.0).all()
