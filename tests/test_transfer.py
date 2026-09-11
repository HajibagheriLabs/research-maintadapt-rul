from itertools import pairwise

from maintadapt.transfer import budget_units


def test_target_budgets_are_nested_subsets_of_the_training_units() -> None:
    train_units = list(range(1, 81))
    budgets = [
        budget_units(train_units, share, seed=3) for share in (0.1, 0.25, 0.5, 1.0)
    ]
    assert [len(units) for units in budgets] == [8, 20, 40, 80]
    # nesting means a larger budget only ever adds engines, so the curve compares like with like
    for smaller, larger in pairwise(budgets):
        assert set(smaller) <= set(larger)
    assert set(budgets[-1]) == set(train_units)
