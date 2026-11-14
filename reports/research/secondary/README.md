# Secondary analyses

Pre-registered secondary analyses S1-S7 and S9 of PREREGISTRATION.md section 7, written by `make research-secondary` at commit 35ab3424b86effeee2cfd1dd155aa1eaa01a33a5. None of them is confirmatory and none changes a verdict. S8 (N-CMAPSS) is not part of this folder.

Every estimate uses the derived tables of `runs/research/derived/archive/<fleet>/` and the confirmatory bootstrap: 2,000 replicates with training engines from `default_rng([2026, fleet number, 0])` and official test engines from `[..., 1]`, shared by every configuration, so each interval comes from the same replicates as E1-E3. Rows are point estimates with percentile 95% intervals (`_lo`, `_hi`). The first line of every CSV states its item.

| file | item | content |
|---|---|---|
| `s1_accuracy_metrics.csv` | S1 | E1 with the NASA score and with RMSE over all observed cycles of the official test engines; the leaderboard_rmse row repeats E1 from the same replicates. |
| `s2_settings_grid.csv` | S2 | E1-E3 (with S by validation cost and by validation RMSE) at every rule, lead time and cost ratio, validation-tuned. |
| `s2_rules.csv` | S2 | the interpretation rules of section 6 read descriptively at every setting. |
| `s2_heatmaps.md` | S2 | E1, E2 and E3 as K by L grids per fleet and rule. |
| `s3_families.csv` | S3 | family representatives: leaderboard RMSE, S and jitter with intervals, ranks by leaderboard RMSE and by S (bump-chart data), failure share and waste. |
| `s3_pairwise.csv` | S3 | every pair of families: differences in leaderboard RMSE and in D with intervals, bootstrap p-values, Benjamini-Hochberg q-values over the whole analysis, inversions. |
| `s3_family_e1.csv` | S3 | tau-b(leaderboard RMSE, D) across the family representatives. |
| `s4_threshold_variants.csv` | S4 | E1-E3 under validation, crossfit and oracle tuning; E2 keeps the validation selection. |
| `s4_nested.csv` | S4 | S of each family representative under validation, crossfit, oracle and nested tuning (nested fleets, repeat 0). |
| `s4_nested_e1.csv` | S4 | tau-b(leaderboard RMSE, D) across the family representatives per tuning (repeat 0). |
| `s5_caps.csv` | S5 | reference XGBoost and LSTM per training cap: leaderboard RMSE (truth capped at 125), RMSE against the uncapped truth, jitter and S (repeat 0). |
| `s6_regret.csv` | S6 | E1-E3 with regret (C_f * failure + waste) in place of the cost rate, per C_f. |
| `s7_matched_accuracy.csv` | S7 | tau-b(RMSE at 20 truncation points per outer_test engine, D) against E1. |
| `s9_competitive.csv` | S9 | E1 among the configurations within the margin of the best leaderboard RMSE. |

## Settings of each item

- S2: the grid of section 5.5, K in {1.5, 2, 3, 5, 10, 20, 50} and L in {0, 10, 20}, for rules R1 and R2, all with validation-tuned thresholds.
- S3: in each (repeat, fold) the family's configuration with the lowest inner_val RMSE represents the family; the representatives' outcomes are pooled like a configuration's. Benjamini-Hochberg at 0.05 runs over every p-value of S3 (both metrics, all pairs and fleets). A pair is inverted when both differences are rejected with opposite signs; `inversion_unadjusted` uses the intervals instead.
- S4: crossfit and oracle thresholds come from the derived tables. Nested thresholds are tuned on 4-fold inner cross-fitted predictions of the outer-train engines (FD001, FD003, repeat 0, family representatives) and applied to the confirmatory outer_test predictions.
- S5: caps 100, 150, 200, none besides the confirmatory 125, families xgboost, lstm, configuration 0, repeat 0, with the confirmatory seeds; metrics on the official test engines use the truth capped at 125 except `rmse_last_uncapped`.
- S6: thresholds and age replacement are tuned to the lowest mean regret, rule R1, lead time 10; the regret share is 1 - regret / age replacement's regret, pooled over the folds of a repeat and averaged over repeats (perfect prognostics has zero regret). E2 selects by in-sample validation regret.
- S7: each configuration's RMSE at the truncation points of its outer_test engines, pooled per repeat and averaged, resampled with the training engines like D.
- S9: margin 2 cycles of leaderboard RMSE (point estimates); the set is held fixed in the bootstrap.
