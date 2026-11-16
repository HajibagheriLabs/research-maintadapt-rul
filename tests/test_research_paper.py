import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import kendalltau

from maintadapt.research.costs import tune_age
from maintadapt.research.paper import booktabs, number, numbers_tex, tex
from maintadapt.research.policies import tune
from maintadapt.research.recheck import (
    Engines,
    Setting,
    age_policy,
    costs_and_cycles,
    first_alarms,
    share,
    tau_b,
    tune_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"analysis", "stats", "policies", "costs", "derived"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{a.name}" for a in node.names}
    return found


def test_the_recomputation_imports_nothing_from_maintadapt() -> None:
    assert not any(
        m.startswith("maintadapt")
        for m in _imports(ROOT / "src/maintadapt/research/recheck.py")
    )


def test_check_numbers_imports_none_of_the_pipeline_it_checks() -> None:
    found = _imports(ROOT / "scripts/research/check_numbers.py")
    for module in found:
        parts = module.split(".")
        assert not (parts[:2] == ["maintadapt", "research"] and FORBIDDEN & set(parts))
    assert "maintadapt.research.recheck" in found


def test_tau_b_by_hand_with_and_without_ties() -> None:
    # pairs (1,2) and (1,3) agree, (2,3) disagree: (2 - 1) / 3
    assert tau_b(np.array([1, 2, 3]), np.array([1, 3, 2])) == pytest.approx(1 / 3)
    # x ties the first pair: S = 2 over sqrt(2 untied x pairs * 3 untied y pairs)
    x, y = np.array([1, 1, 2]), np.array([1, 2, 3])
    assert tau_b(x, y) == pytest.approx(2 / np.sqrt(6))
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 5, 40), rng.integers(0, 5, 40)
    assert tau_b(a, b) == pytest.approx(kendalltau(a, b, variant="b").statistic)


def test_first_alarm_is_the_first_cycle_at_or_below_theta() -> None:
    thetas = np.array([-np.inf, 3.0, 4.0, 0.5, np.inf])
    alarms = first_alarms([np.array([5.0, 3.0, 4.0, 1.0])], thetas)
    assert alarms[:, 0].tolist() == [0, 2, 2, 0, 1]


def test_costs_and_cycles_by_hand() -> None:
    # no alarm fails; alarm 2 + lead 3 = 5 <= 10 is preventive; 9 + 3 = 12 > 10 fails
    cost, used = costs_and_cycles(np.array([0, 2, 9]), np.array([10, 10, 10]), 3, 10.0)
    assert cost.tolist() == [10.0, 1.0, 10.0]
    assert used.tolist() == [10.0, 5.0, 10.0]


def test_threshold_tuning_by_hand() -> None:
    # candidates -inf, 1, 2, 3, inf give rates 10/3, 1/3, 1/2, 1, 1: theta = 1
    engines = Engines(np.array([1]), [np.array([3.0, 2.0, 1.0])], [np.zeros(3)])
    theta, rate = tune_threshold(engines, Setting(10.0, 0, 3, 125, "official_last"))
    assert theta == 1.0
    assert rate == pytest.approx(1 / 3)


def test_age_policy_and_savings_share_by_hand() -> None:
    # A = 10: 2 / 20; A = 20: 11 / 30; never: 20 / 30
    assert age_policy(np.array([10, 20]), 10.0) == 10.0
    # CR = 0.1, CR_age = 0.15, CR* = 0.05
    assert share(2.0, 20.0, 3.0, 20.0, 0.05) == pytest.approx(0.5)


def test_independent_tuning_agrees_with_the_pipeline_on_random_engines() -> None:
    rng = np.random.default_rng(3)
    for trial in range(20):
        lives = rng.integers(20, 80, size=8)
        pred = [
            np.maximum(0.0, np.arange(n)[::-1] + rng.normal(0, 8, n)).round(1)
            for n in lives
        ]
        lead, k = int(rng.choice([0, 10])), float(rng.choice([2.0, 10.0]))
        engines = Engines(np.arange(8), pred, [np.zeros(n) for n in lives])
        theta, rate = tune_threshold(
            engines, Setting(k, lead, 41, 125, "official_last")
        )
        expected = tune(pred, "R1", [1.0], lead, [k], 41)[k]
        assert (theta, rate) == (expected.theta, expected.cost_rate), trial
        assert age_policy(lives, k) == tune_age(lives, k)


def test_numbers_print_minus_signs_in_math_mode() -> None:
    assert number(0.28634) == "0.286"
    assert number(-0.0621) == r"\ensuremath{-0.062}"
    assert number(-0.0001) == "0.000"
    assert number(float("nan")) == "--"


def test_numbers_tex_defines_val_with_a_visible_fallback() -> None:
    text = numbers_tex({"e1.FD001.tau_b": "0.286", "a.b.c": "x"}, "% header")
    assert r"\ifcsname val@#1\endcsname" in text
    assert r"\textbf{??\texttt{\detokenize{#1}}??}" in text
    assert text.index(r"\defval{a.b.c}") < text.index(r"\defval{e1.FD001.tau_b}")
    with pytest.raises(ValueError):
        numbers_tex({"e1.FD001": "1"}, "")
    with pytest.raises(ValueError):
        numbers_tex({"e1.FD001.bad key": "1"}, "")


def test_tables_escape_text_and_use_booktabs_rules() -> None:
    assert tex("inner_val 25%") == r"inner\_val 25\%"
    body = booktabs(["a", "b"], [["1", "2"], "\\midrule", ["3", "4"]], "lr", "src")
    lines = body.splitlines()
    assert lines[2] == r"\begin{tabular}{lr}"
    assert [line for line in lines if "rule" in line] == [
        r"\toprule",
        r"\midrule",
        r"\midrule",
        r"\bottomrule",
    ]
    assert "1 & 2 \\\\" in lines
