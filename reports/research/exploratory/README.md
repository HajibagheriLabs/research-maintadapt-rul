# Exploratory analyses

**EXPLORATORY.** None of these analyses is in PREREGISTRATION.md; they are reported as exploration, with percentile 95% intervals from the confirmatory bootstrap draws and no correction for multiplicity. Written by `make research-exploratory` at commit 42e09048808267cf2bf131e8f1b3e2bb90b1cee7 from the confirmatory archive, after the tag `analysis-frozen`.

| file | analysis | content |
|---|---|---|
| `x1_isotonic_late_rmse.csv` | X1 | tau-b(proxy, D) for late-phase RMSE (true RUL <= 50) on the official test engines, raw and after isotonic recalibration on each fit's inner_val engines, beside E1. |
| `x2_ema_configurations.csv` | X2 | per configuration and fixed alpha: leaderboard RMSE of the smoothed predictions and S of rule R1 on them, with the changes from alpha = 1. |
| `x2_ema_summary.csv` | X2 | per fleet and alpha: median changes in leaderboard RMSE and in S, the shares of configurations whose RMSE rises, whose S rises and both, and tau-b between the changes. |
| `x3_legacy_booking.csv` | X3 | per fleet: tau-b among the legacy booking cost on truncated official test engines, the trajectory regret at the same failure cost, leaderboard RMSE and D. |
| `x3_legacy_configurations.csv` | X3 | per configuration: legacy booking cost per engine, the trajectory counterpart (mean regret per engine), leaderboard RMSE and S. |

## Definitions

- X1: for each fit, an isotonic regression of the truth capped at 125 on the inner_val predictions maps the official test predictions; late-phase RMSE pools the cycles with true RUL <= 50 over the official test engines of a fit and is averaged over fits, like leaderboard RMSE. The recalibration never reverses the order of two predictions, but it may tie predictions that differed; it changes the proxy only, never a policy.
- X2: predictions smoothed by the causal EMA at each fixed alpha in {1, 0.5, 0.3, 0.2, 0.1}; rule R1 thresholds tuned on the smoothed inner_val predictions at K = 10 and L = 10; S pooled and averaged as for D. The leaderboard RMSE uses the smoothed prediction at the last observed cycle. alpha = 1 reproduces the derived tables, which the script checks.
- X3: the engineering project's protocol (`scripts/run_decision.py`, `configs/default.yaml`): thresholds 0 to 125 in steps of 1 tuned on every inner_val history cut at every cycle (`decision.replay_snapshots`), applied to the truncated official test engines, which keep their last booking (`decision.replay_alarms`); cost per engine = 100 x failure + cycles thrown away, no lead time. Its trajectory counterpart tunes the threshold on complete inner_val trajectories for the lowest mean regret at the same failure cost and no lead time, and scores the complete outer_test trajectories.
