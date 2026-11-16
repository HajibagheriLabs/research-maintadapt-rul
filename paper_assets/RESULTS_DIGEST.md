# Results digest

Facts only. Every number below is a macro of `paper_assets/numbers.tex` (`\val{<analysis>.<fleet>.<quantity>}`), with its source file, column and the command that wrote the source. Intervals are percentile 95% bootstrap intervals; `_lo`, `_hi` and `_ci` macros exist for every number shown with one. Confirmatory numbers are `e1`, `e2`, `e3` and `verdict` on FD001-FD004; `s*` are pre-registered secondary analyses, NCMAPSS is secondary analysis S8 (descriptive), `x*` are exploratory, and `f1`, `f5`, `data` and `compute` are computed by `make paper-assets`. Written by make paper-assets at commit 47df31a0d0948b79b4818fafa45ec413e2bdd6c8.

Definitions of the numbers computed here:

- F1: xgboost-00 on FD001, repeat 0, with its predictions y replaced by y + 15 and by 1.25 y before the policy code of derived.py (repeat_outcomes, age_table, savings) and the metric code (metric_components, fit_metrics) run; S and leaderboard RMSE are for that repeat alone.
- F5 (descriptive, not pre-registered): the oracle selection takes the configuration with the highest S in hindsight (point estimate, the same in every fold); the gap is its S minus the S of the models chosen by validation cost or by validation RMSE, with the confirmatory bootstrap draws of training engines and every selection held fixed.

## RQ1: agreement between leaderboard rank and maintenance-value rank (H1)

Pre-registered verdict: **supported** (`reports/research/confirmatory/verdicts.md`, make research-confirmatory).

| quantity | fleet | estimate | 95% interval | macro | source | column | command |
|---|---|---|---|---|---|---|---|
| E1 tau-b(leaderboard RMSE, D), confirmatory | FD001 | 0.286 | [-0.182, 0.408] | `\val{e1.FD001.tau_b}` | `reports/research/confirmatory/e1.csv` | tau_b, lo, hi | make research-confirmatory |
| E1 band | FD001 | weak |  | `\val{e1.FD001.band}` | `reports/research/confirmatory/e1.csv` | band | make research-confirmatory |
| configurations | FD001 | 113 |  | `\val{e1.FD001.configs}` | `reports/research/confirmatory/e1.csv` | configs | make research-confirmatory |
| E1 tau-b(leaderboard RMSE, D), confirmatory | FD002 | 0.041 | [-0.281, 0.355] | `\val{e1.FD002.tau_b}` | `reports/research/confirmatory/e1.csv` | tau_b, lo, hi | make research-confirmatory |
| E1 band | FD002 | weak |  | `\val{e1.FD002.band}` | `reports/research/confirmatory/e1.csv` | band | make research-confirmatory |
| configurations | FD002 | 113 |  | `\val{e1.FD002.configs}` | `reports/research/confirmatory/e1.csv` | configs | make research-confirmatory |
| E1 tau-b(leaderboard RMSE, D), confirmatory | FD003 | -0.037 | [-0.295, 0.281] | `\val{e1.FD003.tau_b}` | `reports/research/confirmatory/e1.csv` | tau_b, lo, hi | make research-confirmatory |
| E1 band | FD003 | weak |  | `\val{e1.FD003.band}` | `reports/research/confirmatory/e1.csv` | band | make research-confirmatory |
| configurations | FD003 | 113 |  | `\val{e1.FD003.configs}` | `reports/research/confirmatory/e1.csv` | configs | make research-confirmatory |
| E1 tau-b(leaderboard RMSE, D), confirmatory | FD004 | -0.062 | [-0.285, 0.193] | `\val{e1.FD004.tau_b}` | `reports/research/confirmatory/e1.csv` | tau_b, lo, hi | make research-confirmatory |
| E1 band | FD004 | weak |  | `\val{e1.FD004.band}` | `reports/research/confirmatory/e1.csv` | band | make research-confirmatory |
| configurations | FD004 | 113 |  | `\val{e1.FD004.configs}` | `reports/research/confirmatory/e1.csv` | configs | make research-confirmatory |
| E1 tau-b(leaderboard RMSE, D), secondary S8, descriptive | NCMAPSS | 0.677 | [0.374, 0.724] | `\val{e1.NCMAPSS.tau_b}` | `reports/research/ncmapss/e1.csv` | tau_b, lo, hi | make research-ncmapss-report |
| E1 band | NCMAPSS | moderate |  | `\val{e1.NCMAPSS.band}` | `reports/research/ncmapss/e1.csv` | band | make research-ncmapss-report |
| configurations | NCMAPSS | 113 |  | `\val{e1.NCMAPSS.configs}` | `reports/research/ncmapss/e1.csv` | configs | make research-ncmapss-report |
| pre-registered verdict on H1 | all | supported |  | `\val{verdict.all.H1}` | `reports/research/confirmatory/verdicts.md` | heading '## H1' | make research-confirmatory |
| S1 tau-b(leaderboard_rmse, D) | FD001 | 0.286 | [-0.182, 0.408] | `\val{s1.FD001.leaderboard_rmse_tau_b}` | `reports/research/secondary/s1_accuracy_metrics.csv` | tau_b, tau_b_lo, tau_b_hi (metric = leaderboard_rmse) | make research-secondary |
| S1 tau-b(nasa_score, D) | FD001 | 0.193 | [-0.235, 0.364] | `\val{s1.FD001.nasa_score_tau_b}` | `reports/research/secondary/s1_accuracy_metrics.csv` | tau_b, tau_b_lo, tau_b_hi (metric = nasa_score) | make research-secondary |
| S1 tau-b(rmse_all_cycles, D) | FD001 | 0.325 | [-0.151, 0.441] | `\val{s1.FD001.rmse_all_cycles_tau_b}` | `reports/research/secondary/s1_accuracy_metrics.csv` | tau_b, tau_b_lo, tau_b_hi (metric = rmse_all_cycles) | make research-secondary |
| S1 tau-b(leaderboard_rmse, D) | FD002 | 0.041 | [-0.281, 0.355] | `\val{s1.FD002.leaderboard_rmse_tau_b}` | `reports/research/secondary/s1_accuracy_metrics.csv` | tau_b, tau_b_lo, tau_b_hi (metric = leaderboard_rmse) | make research-secondary |
| S1 tau-b(nasa_score, D) | FD002 | 0.009 | [-0.308, 0.317] | `\val{s1.FD002.nasa_score_tau_b}` | `reports/research/secondary/s1_accuracy_metrics.csv` | tau_b, tau_b_lo, tau_b_hi (metric = nasa_score) | make research-secondary |
| S1 tau-b(rmse_all_cycles, D) | FD002 | 0.047 | [-0.299, 0.367] | `\val{s1.FD002.rmse_all_cycles_tau_b}` | `reports/research/secondary/s1_accuracy_metrics.csv` | tau_b, tau_b_lo, tau_b_hi (metric = rmse_all_cycles) | make research-secondary |
| S1 tau-b(leaderboard_rmse, D) | FD003 | -0.037 | [-0.295, 0.281] | `\val{s1.FD003.leaderboard_rmse_tau_b}` | `reports/research/secondary/s1_accuracy_metrics.csv` | tau_b, tau_b_lo, tau_b_hi (metric = leaderboard_rmse) | make research-secondary |
| S1 tau-b(nasa_score, D) | FD003 | -0.038 | [-0.311, 0.295] | `\val{s1.FD003.nasa_score_tau_b}` | `reports/research/secondary/s1_accuracy_metrics.csv` | tau_b, tau_b_lo, tau_b_hi (metric = nasa_score) | make research-secondary |
| S1 tau-b(rmse_all_cycles, D) | FD003 | -0.056 | [-0.304, 0.256] | `\val{s1.FD003.rmse_all_cycles_tau_b}` | `reports/research/secondary/s1_accuracy_metrics.csv` | tau_b, tau_b_lo, tau_b_hi (metric = rmse_all_cycles) | make research-secondary |
| S1 tau-b(leaderboard_rmse, D) | FD004 | -0.062 | [-0.285, 0.193] | `\val{s1.FD004.leaderboard_rmse_tau_b}` | `reports/research/secondary/s1_accuracy_metrics.csv` | tau_b, tau_b_lo, tau_b_hi (metric = leaderboard_rmse) | make research-secondary |
| S1 tau-b(nasa_score, D) | FD004 | -0.081 | [-0.297, 0.213] | `\val{s1.FD004.nasa_score_tau_b}` | `reports/research/secondary/s1_accuracy_metrics.csv` | tau_b, tau_b_lo, tau_b_hi (metric = nasa_score) | make research-secondary |
| S1 tau-b(rmse_all_cycles, D) | FD004 | -0.027 | [-0.272, 0.230] | `\val{s1.FD004.rmse_all_cycles_tau_b}` | `reports/research/secondary/s1_accuracy_metrics.csv` | tau_b, tau_b_lo, tau_b_hi (metric = rmse_all_cycles) | make research-secondary |
| S2 smallest E1 over the 21 settings of R1 | FD001 | -0.042 |  | `\val{s2.FD001.R1_e1_tau_b_min}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R1) | make research-secondary |
| S2 largest E1 over the 21 settings of R1 | FD001 | 0.583 |  | `\val{s2.FD001.R1_e1_tau_b_max}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R1) | make research-secondary |
| S2 settings of R1 with E1 in the weak band (of 21) | FD001 | 17 |  | `\val{s2.FD001.R1_weak_settings}` | `reports/research/secondary/s2_settings_grid.csv` | e1_band (rule = R1) | make research-secondary |
| S2 smallest E1 over the 21 settings of R2 | FD001 | -0.093 |  | `\val{s2.FD001.R2_e1_tau_b_min}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R2) | make research-secondary |
| S2 largest E1 over the 21 settings of R2 | FD001 | 0.562 |  | `\val{s2.FD001.R2_e1_tau_b_max}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R2) | make research-secondary |
| S2 settings of R2 with E1 in the weak band (of 21) | FD001 | 18 |  | `\val{s2.FD001.R2_weak_settings}` | `reports/research/secondary/s2_settings_grid.csv` | e1_band (rule = R2) | make research-secondary |
| S2 smallest E1 over the 21 settings of R1 | FD002 | -0.081 |  | `\val{s2.FD002.R1_e1_tau_b_min}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R1) | make research-secondary |
| S2 largest E1 over the 21 settings of R1 | FD002 | 0.650 |  | `\val{s2.FD002.R1_e1_tau_b_max}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R1) | make research-secondary |
| S2 settings of R1 with E1 in the weak band (of 21) | FD002 | 13 |  | `\val{s2.FD002.R1_weak_settings}` | `reports/research/secondary/s2_settings_grid.csv` | e1_band (rule = R1) | make research-secondary |
| S2 smallest E1 over the 21 settings of R2 | FD002 | -0.239 |  | `\val{s2.FD002.R2_e1_tau_b_min}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R2) | make research-secondary |
| S2 largest E1 over the 21 settings of R2 | FD002 | 0.716 |  | `\val{s2.FD002.R2_e1_tau_b_max}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R2) | make research-secondary |
| S2 settings of R2 with E1 in the weak band (of 21) | FD002 | 11 |  | `\val{s2.FD002.R2_weak_settings}` | `reports/research/secondary/s2_settings_grid.csv` | e1_band (rule = R2) | make research-secondary |
| S2 smallest E1 over the 21 settings of R1 | FD003 | -0.126 |  | `\val{s2.FD003.R1_e1_tau_b_min}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R1) | make research-secondary |
| S2 largest E1 over the 21 settings of R1 | FD003 | 0.550 |  | `\val{s2.FD003.R1_e1_tau_b_max}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R1) | make research-secondary |
| S2 settings of R1 with E1 in the weak band (of 21) | FD003 | 19 |  | `\val{s2.FD003.R1_weak_settings}` | `reports/research/secondary/s2_settings_grid.csv` | e1_band (rule = R1) | make research-secondary |
| S2 smallest E1 over the 21 settings of R2 | FD003 | -0.132 |  | `\val{s2.FD003.R2_e1_tau_b_min}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R2) | make research-secondary |
| S2 largest E1 over the 21 settings of R2 | FD003 | 0.610 |  | `\val{s2.FD003.R2_e1_tau_b_max}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R2) | make research-secondary |
| S2 settings of R2 with E1 in the weak band (of 21) | FD003 | 19 |  | `\val{s2.FD003.R2_weak_settings}` | `reports/research/secondary/s2_settings_grid.csv` | e1_band (rule = R2) | make research-secondary |
| S2 smallest E1 over the 21 settings of R1 | FD004 | -0.229 |  | `\val{s2.FD004.R1_e1_tau_b_min}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R1) | make research-secondary |
| S2 largest E1 over the 21 settings of R1 | FD004 | 0.478 |  | `\val{s2.FD004.R1_e1_tau_b_max}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R1) | make research-secondary |
| S2 settings of R1 with E1 in the weak band (of 21) | FD004 | 18 |  | `\val{s2.FD004.R1_weak_settings}` | `reports/research/secondary/s2_settings_grid.csv` | e1_band (rule = R1) | make research-secondary |
| S2 smallest E1 over the 21 settings of R2 | FD004 | -0.231 |  | `\val{s2.FD004.R2_e1_tau_b_min}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R2) | make research-secondary |
| S2 largest E1 over the 21 settings of R2 | FD004 | 0.558 |  | `\val{s2.FD004.R2_e1_tau_b_max}` | `reports/research/secondary/s2_settings_grid.csv` | e1_tau_b (rule = R2) | make research-secondary |
| S2 settings of R2 with E1 in the weak band (of 21) | FD004 | 18 |  | `\val{s2.FD004.R2_weak_settings}` | `reports/research/secondary/s2_settings_grid.csv` | e1_band (rule = R2) | make research-secondary |
| S2 settings (of 42) where the H1 rule reads supported | all | 33 |  | `\val{s2.all.H1_supported_settings}` | `reports/research/secondary/s2_rules.csv` | H1_rule | make research-secondary |
| S2 settings (of 42) where the H1 rule reads mixed | all | 9 |  | `\val{s2.all.H1_mixed_settings}` | `reports/research/secondary/s2_rules.csv` | H1_rule | make research-secondary |
| S3 tau-b(leaderboard RMSE, D) across the family representatives | FD001 | 0.244 | [-0.289, 0.600] | `\val{s3.FD001.family_tau_b}` | `reports/research/secondary/s3_family_e1.csv` | tau_b, tau_b_lo, tau_b_hi | make research-secondary |
| S3 tau-b(leaderboard RMSE, D) across the family representatives | FD002 | 0.067 | [-0.289, 0.556] | `\val{s3.FD002.family_tau_b}` | `reports/research/secondary/s3_family_e1.csv` | tau_b, tau_b_lo, tau_b_hi | make research-secondary |
| S3 tau-b(leaderboard RMSE, D) across the family representatives | FD003 | 0.156 | [-0.289, 0.511] | `\val{s3.FD003.family_tau_b}` | `reports/research/secondary/s3_family_e1.csv` | tau_b, tau_b_lo, tau_b_hi | make research-secondary |
| S3 tau-b(leaderboard RMSE, D) across the family representatives | FD004 | -0.333 | [-0.601, 0.244] | `\val{s3.FD004.family_tau_b}` | `reports/research/secondary/s3_family_e1.csv` | tau_b, tau_b_lo, tau_b_hi | make research-secondary |
| S4 e1_tau_b with validation thresholds | FD001 | 0.286 | [-0.182, 0.408] | `\val{s4.FD001.validation_e1_tau_b}` | `reports/research/secondary/s4_threshold_variants.csv` | e1_tau_b, _lo, _hi (tuning = validation) | make research-secondary |
| S4 e1_tau_b with crossfit thresholds | FD001 | 0.290 | [-0.127, 0.384] | `\val{s4.FD001.crossfit_e1_tau_b}` | `reports/research/secondary/s4_threshold_variants.csv` | e1_tau_b, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 e1_tau_b with oracle thresholds | FD001 | 0.464 | [0.298, 0.553] | `\val{s4.FD001.oracle_e1_tau_b}` | `reports/research/secondary/s4_threshold_variants.csv` | e1_tau_b, _lo, _hi (tuning = oracle) | make research-secondary |
| S4 e1_tau_b with validation thresholds | FD002 | 0.041 | [-0.281, 0.355] | `\val{s4.FD002.validation_e1_tau_b}` | `reports/research/secondary/s4_threshold_variants.csv` | e1_tau_b, _lo, _hi (tuning = validation) | make research-secondary |
| S4 e1_tau_b with crossfit thresholds | FD002 | 0.287 | [-0.091, 0.422] | `\val{s4.FD002.crossfit_e1_tau_b}` | `reports/research/secondary/s4_threshold_variants.csv` | e1_tau_b, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 e1_tau_b with oracle thresholds | FD002 | 0.592 | [0.428, 0.614] | `\val{s4.FD002.oracle_e1_tau_b}` | `reports/research/secondary/s4_threshold_variants.csv` | e1_tau_b, _lo, _hi (tuning = oracle) | make research-secondary |
| S4 e1_tau_b with validation thresholds | FD003 | -0.037 | [-0.295, 0.281] | `\val{s4.FD003.validation_e1_tau_b}` | `reports/research/secondary/s4_threshold_variants.csv` | e1_tau_b, _lo, _hi (tuning = validation) | make research-secondary |
| S4 e1_tau_b with crossfit thresholds | FD003 | 0.012 | [-0.268, 0.290] | `\val{s4.FD003.crossfit_e1_tau_b}` | `reports/research/secondary/s4_threshold_variants.csv` | e1_tau_b, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 e1_tau_b with oracle thresholds | FD003 | -0.013 | [-0.121, 0.118] | `\val{s4.FD003.oracle_e1_tau_b}` | `reports/research/secondary/s4_threshold_variants.csv` | e1_tau_b, _lo, _hi (tuning = oracle) | make research-secondary |
| S4 e1_tau_b with validation thresholds | FD004 | -0.062 | [-0.285, 0.193] | `\val{s4.FD004.validation_e1_tau_b}` | `reports/research/secondary/s4_threshold_variants.csv` | e1_tau_b, _lo, _hi (tuning = validation) | make research-secondary |
| S4 e1_tau_b with crossfit thresholds | FD004 | -0.067 | [-0.283, 0.175] | `\val{s4.FD004.crossfit_e1_tau_b}` | `reports/research/secondary/s4_threshold_variants.csv` | e1_tau_b, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 e1_tau_b with oracle thresholds | FD004 | 0.002 | [-0.095, 0.102] | `\val{s4.FD004.oracle_e1_tau_b}` | `reports/research/secondary/s4_threshold_variants.csv` | e1_tau_b, _lo, _hi (tuning = oracle) | make research-secondary |
| S4 tau-b across family representatives, validation (repeat 0) | FD001 | -0.022 | [-0.333, 0.467] | `\val{s4.FD001.validation_family_tau_b}` | `reports/research/secondary/s4_nested_e1.csv` | tau_b, _lo, _hi (tuning = validation) | make research-secondary |
| S4 tau-b across family representatives, crossfit (repeat 0) | FD001 | -0.022 | [-0.378, 0.422] | `\val{s4.FD001.crossfit_family_tau_b}` | `reports/research/secondary/s4_nested_e1.csv` | tau_b, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 tau-b across family representatives, oracle (repeat 0) | FD001 | 0.689 | [0.422, 0.822] | `\val{s4.FD001.oracle_family_tau_b}` | `reports/research/secondary/s4_nested_e1.csv` | tau_b, _lo, _hi (tuning = oracle) | make research-secondary |
| S4 tau-b across family representatives, nested (repeat 0) | FD001 | 0.289 | [-0.111, 0.600] | `\val{s4.FD001.nested_family_tau_b}` | `reports/research/secondary/s4_nested_e1.csv` | tau_b, _lo, _hi (tuning = nested) | make research-secondary |
| S4 tau-b across family representatives, validation (repeat 0) | FD003 | 0.156 | [-0.244, 0.333] | `\val{s4.FD003.validation_family_tau_b}` | `reports/research/secondary/s4_nested_e1.csv` | tau_b, _lo, _hi (tuning = validation) | make research-secondary |
| S4 tau-b across family representatives, crossfit (repeat 0) | FD003 | -0.378 | [-0.556, 0.246] | `\val{s4.FD003.crossfit_family_tau_b}` | `reports/research/secondary/s4_nested_e1.csv` | tau_b, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 tau-b across family representatives, oracle (repeat 0) | FD003 | -0.022 | [-0.244, 0.156] | `\val{s4.FD003.oracle_family_tau_b}` | `reports/research/secondary/s4_nested_e1.csv` | tau_b, _lo, _hi (tuning = oracle) | make research-secondary |
| S4 tau-b across family representatives, nested (repeat 0) | FD003 | -0.067 | [-0.422, 0.201] | `\val{s4.FD003.nested_family_tau_b}` | `reports/research/secondary/s4_nested_e1.csv` | tau_b, _lo, _hi (tuning = nested) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 50 | FD001 | 0.457 | [0.231, 0.551] | `\val{s6.FD001.cf50_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 50) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 100 | FD001 | 0.453 | [0.169, 0.528] | `\val{s6.FD001.cf100_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 100) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 200 | FD001 | 0.394 | [0.015, 0.479] | `\val{s6.FD001.cf200_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 200) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 500 | FD001 | 0.323 | [-0.110, 0.428] | `\val{s6.FD001.cf500_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 500) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 1000 | FD001 | 0.290 | [-0.162, 0.413] | `\val{s6.FD001.cf1000_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 1000) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 2000 | FD001 | 0.274 | [-0.190, 0.401] | `\val{s6.FD001.cf2000_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 2000) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 50 | FD002 | 0.593 | [0.410, 0.620] | `\val{s6.FD002.cf50_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 50) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 100 | FD002 | 0.549 | [0.329, 0.599] | `\val{s6.FD002.cf100_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 100) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 200 | FD002 | 0.495 | [0.214, 0.562] | `\val{s6.FD002.cf200_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 200) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 500 | FD002 | 0.329 | [-0.026, 0.479] | `\val{s6.FD002.cf500_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 500) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 1000 | FD002 | 0.123 | [-0.230, 0.388] | `\val{s6.FD002.cf1000_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 1000) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 2000 | FD002 | 0.007 | [-0.302, 0.333] | `\val{s6.FD002.cf2000_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 2000) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 50 | FD003 | 0.114 | [-0.108, 0.294] | `\val{s6.FD003.cf50_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 50) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 100 | FD003 | 0.016 | [-0.227, 0.263] | `\val{s6.FD003.cf100_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 100) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 200 | FD003 | -0.028 | [-0.273, 0.274] | `\val{s6.FD003.cf200_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 200) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 500 | FD003 | -0.035 | [-0.292, 0.283] | `\val{s6.FD003.cf500_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 500) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 1000 | FD003 | -0.038 | [-0.295, 0.283] | `\val{s6.FD003.cf1000_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 1000) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 2000 | FD003 | -0.034 | [-0.294, 0.284] | `\val{s6.FD003.cf2000_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 2000) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 50 | FD004 | 0.235 | [0.076, 0.362] | `\val{s6.FD004.cf50_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 50) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 100 | FD004 | 0.105 | [-0.057, 0.272] | `\val{s6.FD004.cf100_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 100) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 200 | FD004 | -0.012 | [-0.207, 0.186] | `\val{s6.FD004.cf200_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 200) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 500 | FD004 | -0.038 | [-0.224, 0.197] | `\val{s6.FD004.cf500_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 500) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 1000 | FD004 | -0.015 | [-0.231, 0.221] | `\val{s6.FD004.cf1000_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 1000) | make research-secondary |
| S6 e1_tau_b with regret, C_f = 2000 | FD004 | -0.066 | [-0.290, 0.187] | `\val{s6.FD004.cf2000_e1_tau_b}` | `reports/research/secondary/s6_regret.csv` | e1_tau_b, _lo, _hi (failure_cost = 2000) | make research-secondary |
| S7 tau-b(matched-accuracy RMSE, D) | FD001 | 0.329 | [-0.148, 0.448] | `\val{s7.FD001.tau_matched}` | `reports/research/secondary/s7_matched_accuracy.csv` | tau_matched, _lo, _hi | make research-secondary |
| S7 matched minus leaderboard tau-b | FD001 | 0.043 | [-0.064, 0.177] | `\val{s7.FD001.difference}` | `reports/research/secondary/s7_matched_accuracy.csv` | difference, _lo, _hi | make research-secondary |
| S7 tau-b(matched-accuracy RMSE, D) | FD002 | 0.057 | [-0.287, 0.381] | `\val{s7.FD002.tau_matched}` | `reports/research/secondary/s7_matched_accuracy.csv` | tau_matched, _lo, _hi | make research-secondary |
| S7 matched minus leaderboard tau-b | FD002 | 0.015 | [-0.078, 0.095] | `\val{s7.FD002.difference}` | `reports/research/secondary/s7_matched_accuracy.csv` | difference, _lo, _hi | make research-secondary |
| S7 tau-b(matched-accuracy RMSE, D) | FD003 | -0.079 | [-0.310, 0.269] | `\val{s7.FD003.tau_matched}` | `reports/research/secondary/s7_matched_accuracy.csv` | tau_matched, _lo, _hi | make research-secondary |
| S7 matched minus leaderboard tau-b | FD003 | -0.042 | [-0.141, 0.101] | `\val{s7.FD003.difference}` | `reports/research/secondary/s7_matched_accuracy.csv` | difference, _lo, _hi | make research-secondary |
| S7 tau-b(matched-accuracy RMSE, D) | FD004 | -0.061 | [-0.295, 0.232] | `\val{s7.FD004.tau_matched}` | `reports/research/secondary/s7_matched_accuracy.csv` | tau_matched, _lo, _hi | make research-secondary |
| S7 matched minus leaderboard tau-b | FD004 | 0.002 | [-0.065, 0.095] | `\val{s7.FD004.difference}` | `reports/research/secondary/s7_matched_accuracy.csv` | difference, _lo, _hi | make research-secondary |
| S9 E1 among configurations within 2 cycles of the best | FD001 | 0.082 | [-0.297, 0.356] | `\val{s9.FD001.tau_b}` | `reports/research/secondary/s9_competitive.csv` | tau_b, tau_b_lo, tau_b_hi | make research-secondary |
| S9 configurations within the margin | FD001 | 40 |  | `\val{s9.FD001.configs}` | `reports/research/secondary/s9_competitive.csv` | configs | make research-secondary |
| S9 families within the margin | FD001 | 5 |  | `\val{s9.FD001.families}` | `reports/research/secondary/s9_competitive.csv` | families | make research-secondary |
| S9 best leaderboard RMSE | FD001 | 12.38 |  | `\val{s9.FD001.best_rmse}` | `reports/research/secondary/s9_competitive.csv` | best_rmse | make research-secondary |
| S9 E1 among configurations within 2 cycles of the best | FD002 | -0.174 | [-0.389, 0.175] | `\val{s9.FD002.tau_b}` | `reports/research/secondary/s9_competitive.csv` | tau_b, tau_b_lo, tau_b_hi | make research-secondary |
| S9 configurations within the margin | FD002 | 59 |  | `\val{s9.FD002.configs}` | `reports/research/secondary/s9_competitive.csv` | configs | make research-secondary |
| S9 families within the margin | FD002 | 7 |  | `\val{s9.FD002.families}` | `reports/research/secondary/s9_competitive.csv` | families | make research-secondary |
| S9 best leaderboard RMSE | FD002 | 12.63 |  | `\val{s9.FD002.best_rmse}` | `reports/research/secondary/s9_competitive.csv` | best_rmse | make research-secondary |
| S9 E1 among configurations within 2 cycles of the best | FD003 | 0.236 | [-0.269, 0.442] | `\val{s9.FD003.tau_b}` | `reports/research/secondary/s9_competitive.csv` | tau_b, tau_b_lo, tau_b_hi | make research-secondary |
| S9 configurations within the margin | FD003 | 49 |  | `\val{s9.FD003.configs}` | `reports/research/secondary/s9_competitive.csv` | configs | make research-secondary |
| S9 families within the margin | FD003 | 5 |  | `\val{s9.FD003.families}` | `reports/research/secondary/s9_competitive.csv` | families | make research-secondary |
| S9 best leaderboard RMSE | FD003 | 12.66 |  | `\val{s9.FD003.best_rmse}` | `reports/research/secondary/s9_competitive.csv` | best_rmse | make research-secondary |
| S9 E1 among configurations within 2 cycles of the best | FD004 | 0.128 | [-0.123, 0.267] | `\val{s9.FD004.tau_b}` | `reports/research/secondary/s9_competitive.csv` | tau_b, tau_b_lo, tau_b_hi | make research-secondary |
| S9 configurations within the margin | FD004 | 61 |  | `\val{s9.FD004.configs}` | `reports/research/secondary/s9_competitive.csv` | configs | make research-secondary |
| S9 families within the margin | FD004 | 7 |  | `\val{s9.FD004.families}` | `reports/research/secondary/s9_competitive.csv` | families | make research-secondary |
| S9 best leaderboard RMSE | FD004 | 13.52 |  | `\val{s9.FD004.best_rmse}` | `reports/research/secondary/s9_competitive.csv` | best_rmse | make research-secondary |

## RQ2: selecting by validation RMSE instead of validation cost (H2)

Pre-registered verdict: **inconclusive** (`reports/research/confirmatory/verdicts.md`, make research-confirmatory).

| quantity | fleet | estimate | 95% interval | macro | source | column | command |
|---|---|---|---|---|---|---|---|
| E2 selection loss, confirmatory | FD001 | -0.882 | [-1.710, -0.363] | `\val{e2.FD001.selection_loss}` | `reports/research/confirmatory/e2.csv` | selection_loss, lo, hi | make research-confirmatory |
| S of the models chosen by validation cost | FD001 | -0.304 | [-1.219, 0.283] | `\val{e2.FD001.s_by_cost}` | `reports/research/confirmatory/e2.csv` | s_by_cost, s_by_cost_lo, s_by_cost_hi | make research-confirmatory |
| S of the models chosen by validation RMSE | FD001 | 0.578 | [0.189, 0.839] | `\val{e2.FD001.s_by_rmse}` | `reports/research/confirmatory/e2.csv` | s_by_rmse, s_by_rmse_lo, s_by_rmse_hi | make research-confirmatory |
| E2 selection loss, confirmatory | FD002 | -0.621 | [-0.967, -0.324] | `\val{e2.FD002.selection_loss}` | `reports/research/confirmatory/e2.csv` | selection_loss, lo, hi | make research-confirmatory |
| S of the models chosen by validation cost | FD002 | 0.018 | [-0.359, 0.347] | `\val{e2.FD002.s_by_cost}` | `reports/research/confirmatory/e2.csv` | s_by_cost, s_by_cost_lo, s_by_cost_hi | make research-confirmatory |
| S of the models chosen by validation RMSE | FD002 | 0.639 | [0.454, 0.784] | `\val{e2.FD002.s_by_rmse}` | `reports/research/confirmatory/e2.csv` | s_by_rmse, s_by_rmse_lo, s_by_rmse_hi | make research-confirmatory |
| E2 selection loss, confirmatory | FD003 | -1.136 | [-1.933, -0.537] | `\val{e2.FD003.selection_loss}` | `reports/research/confirmatory/e2.csv` | selection_loss, lo, hi | make research-confirmatory |
| S of the models chosen by validation cost | FD003 | -0.653 | [-1.573, 0.015] | `\val{e2.FD003.s_by_cost}` | `reports/research/confirmatory/e2.csv` | s_by_cost, s_by_cost_lo, s_by_cost_hi | make research-confirmatory |
| S of the models chosen by validation RMSE | FD003 | 0.483 | [0.060, 0.776] | `\val{e2.FD003.s_by_rmse}` | `reports/research/confirmatory/e2.csv` | s_by_rmse, s_by_rmse_lo, s_by_rmse_hi | make research-confirmatory |
| E2 selection loss, confirmatory | FD004 | -0.129 | [-0.297, 0.019] | `\val{e2.FD004.selection_loss}` | `reports/research/confirmatory/e2.csv` | selection_loss, lo, hi | make research-confirmatory |
| S of the models chosen by validation cost | FD004 | 0.601 | [0.417, 0.742] | `\val{e2.FD004.s_by_cost}` | `reports/research/confirmatory/e2.csv` | s_by_cost, s_by_cost_lo, s_by_cost_hi | make research-confirmatory |
| S of the models chosen by validation RMSE | FD004 | 0.729 | [0.575, 0.839] | `\val{e2.FD004.s_by_rmse}` | `reports/research/confirmatory/e2.csv` | s_by_rmse, s_by_rmse_lo, s_by_rmse_hi | make research-confirmatory |
| E2 selection loss, secondary S8, descriptive | NCMAPSS | -0.281 | [-0.660, 0.057] | `\val{e2.NCMAPSS.selection_loss}` | `reports/research/ncmapss/e2.csv` | selection_loss, lo, hi | make research-ncmapss-report |
| S of the models chosen by validation cost | NCMAPSS | -0.450 | [-1.354, 0.146] | `\val{e2.NCMAPSS.s_by_cost}` | `reports/research/ncmapss/e2.csv` | s_by_cost, s_by_cost_lo, s_by_cost_hi | make research-ncmapss-report |
| S of the models chosen by validation RMSE | NCMAPSS | -0.169 | [-1.030, 0.402] | `\val{e2.NCMAPSS.s_by_rmse}` | `reports/research/ncmapss/e2.csv` | s_by_rmse, s_by_rmse_lo, s_by_rmse_hi | make research-ncmapss-report |
| pre-registered verdict on H2 | all | inconclusive |  | `\val{verdict.all.H2}` | `reports/research/confirmatory/verdicts.md` | heading '## H2' | make research-confirmatory |
| S2 smallest E2 over the 21 settings of R1 | FD001 | -2.626 |  | `\val{s2.FD001.R1_e2_selection_loss_min}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R1) | make research-secondary |
| S2 largest E2 over the 21 settings of R1 | FD001 | -0.025 |  | `\val{s2.FD001.R1_e2_selection_loss_max}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R1) | make research-secondary |
| S2 smallest E2 over the 21 settings of R2 | FD001 | -4.637 |  | `\val{s2.FD001.R2_e2_selection_loss_min}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R2) | make research-secondary |
| S2 largest E2 over the 21 settings of R2 | FD001 | -0.054 |  | `\val{s2.FD001.R2_e2_selection_loss_max}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R2) | make research-secondary |
| S2 smallest E2 over the 21 settings of R1 | FD002 | -2.561 |  | `\val{s2.FD002.R1_e2_selection_loss_min}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R1) | make research-secondary |
| S2 largest E2 over the 21 settings of R1 | FD002 | 0.019 |  | `\val{s2.FD002.R1_e2_selection_loss_max}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R1) | make research-secondary |
| S2 smallest E2 over the 21 settings of R2 | FD002 | -2.789 |  | `\val{s2.FD002.R2_e2_selection_loss_min}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R2) | make research-secondary |
| S2 largest E2 over the 21 settings of R2 | FD002 | -0.009 |  | `\val{s2.FD002.R2_e2_selection_loss_max}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R2) | make research-secondary |
| S2 smallest E2 over the 21 settings of R1 | FD003 | -3.512 |  | `\val{s2.FD003.R1_e2_selection_loss_min}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R1) | make research-secondary |
| S2 largest E2 over the 21 settings of R1 | FD003 | -0.024 |  | `\val{s2.FD003.R1_e2_selection_loss_max}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R1) | make research-secondary |
| S2 smallest E2 over the 21 settings of R2 | FD003 | -3.403 |  | `\val{s2.FD003.R2_e2_selection_loss_min}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R2) | make research-secondary |
| S2 largest E2 over the 21 settings of R2 | FD003 | -0.024 |  | `\val{s2.FD003.R2_e2_selection_loss_max}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R2) | make research-secondary |
| S2 smallest E2 over the 21 settings of R1 | FD004 | -1.141 |  | `\val{s2.FD004.R1_e2_selection_loss_min}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R1) | make research-secondary |
| S2 largest E2 over the 21 settings of R1 | FD004 | 0.067 |  | `\val{s2.FD004.R1_e2_selection_loss_max}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R1) | make research-secondary |
| S2 smallest E2 over the 21 settings of R2 | FD004 | -1.518 |  | `\val{s2.FD004.R2_e2_selection_loss_min}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R2) | make research-secondary |
| S2 largest E2 over the 21 settings of R2 | FD004 | 0.030 |  | `\val{s2.FD004.R2_e2_selection_loss_max}` | `reports/research/secondary/s2_settings_grid.csv` | e2_selection_loss (rule = R2) | make research-secondary |
| S2 settings (of 42) where the H2 rule reads inconclusive | all | 42 |  | `\val{s2.all.H2_inconclusive_settings}` | `reports/research/secondary/s2_rules.csv` | H2_rule | make research-secondary |
| S4 e2_selection_loss with validation thresholds | FD001 | -0.882 | [-1.710, -0.363] | `\val{s4.FD001.validation_e2_selection_loss}` | `reports/research/secondary/s4_threshold_variants.csv` | e2_selection_loss, _lo, _hi (tuning = validation) | make research-secondary |
| S4 mean S over configurations, validation | FD001 | 0.443 |  | `\val{s4.FD001.validation_mean_savings_share}` | `reports/research/secondary/s4_threshold_variants.csv` | mean_savings_share (tuning = validation) | make research-secondary |
| S4 e2_selection_loss with crossfit thresholds | FD001 | -0.163 | [-0.468, 0.096] | `\val{s4.FD001.crossfit_e2_selection_loss}` | `reports/research/secondary/s4_threshold_variants.csv` | e2_selection_loss, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 mean S over configurations, crossfit | FD001 | 0.712 |  | `\val{s4.FD001.crossfit_mean_savings_share}` | `reports/research/secondary/s4_threshold_variants.csv` | mean_savings_share (tuning = crossfit) | make research-secondary |
| S4 e2_selection_loss with oracle thresholds | FD001 | 0.005 | [0.002, 0.010] | `\val{s4.FD001.oracle_e2_selection_loss}` | `reports/research/secondary/s4_threshold_variants.csv` | e2_selection_loss, _lo, _hi (tuning = oracle) | make research-secondary |
| S4 mean S over configurations, oracle | FD001 | 0.920 |  | `\val{s4.FD001.oracle_mean_savings_share}` | `reports/research/secondary/s4_threshold_variants.csv` | mean_savings_share (tuning = oracle) | make research-secondary |
| S4 e2_selection_loss with validation thresholds | FD002 | -0.621 | [-0.967, -0.324] | `\val{s4.FD002.validation_e2_selection_loss}` | `reports/research/secondary/s4_threshold_variants.csv` | e2_selection_loss, _lo, _hi (tuning = validation) | make research-secondary |
| S4 mean S over configurations, validation | FD002 | 0.660 |  | `\val{s4.FD002.validation_mean_savings_share}` | `reports/research/secondary/s4_threshold_variants.csv` | mean_savings_share (tuning = validation) | make research-secondary |
| S4 e2_selection_loss with crossfit thresholds | FD002 | -0.079 | [-0.240, 0.072] | `\val{s4.FD002.crossfit_e2_selection_loss}` | `reports/research/secondary/s4_threshold_variants.csv` | e2_selection_loss, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 mean S over configurations, crossfit | FD002 | 0.815 |  | `\val{s4.FD002.crossfit_mean_savings_share}` | `reports/research/secondary/s4_threshold_variants.csv` | mean_savings_share (tuning = crossfit) | make research-secondary |
| S4 e2_selection_loss with oracle thresholds | FD002 | 0.010 | [0.006, 0.015] | `\val{s4.FD002.oracle_e2_selection_loss}` | `reports/research/secondary/s4_threshold_variants.csv` | e2_selection_loss, _lo, _hi (tuning = oracle) | make research-secondary |
| S4 mean S over configurations, oracle | FD002 | 0.892 |  | `\val{s4.FD002.oracle_mean_savings_share}` | `reports/research/secondary/s4_threshold_variants.csv` | mean_savings_share (tuning = oracle) | make research-secondary |
| S4 e2_selection_loss with validation thresholds | FD003 | -1.136 | [-1.933, -0.537] | `\val{s4.FD003.validation_e2_selection_loss}` | `reports/research/secondary/s4_threshold_variants.csv` | e2_selection_loss, _lo, _hi (tuning = validation) | make research-secondary |
| S4 mean S over configurations, validation | FD003 | 0.424 |  | `\val{s4.FD003.validation_mean_savings_share}` | `reports/research/secondary/s4_threshold_variants.csv` | mean_savings_share (tuning = validation) | make research-secondary |
| S4 e2_selection_loss with crossfit thresholds | FD003 | 0.171 | [-0.081, 0.478] | `\val{s4.FD003.crossfit_e2_selection_loss}` | `reports/research/secondary/s4_threshold_variants.csv` | e2_selection_loss, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 mean S over configurations, crossfit | FD003 | 0.779 |  | `\val{s4.FD003.crossfit_mean_savings_share}` | `reports/research/secondary/s4_threshold_variants.csv` | mean_savings_share (tuning = crossfit) | make research-secondary |
| S4 e2_selection_loss with oracle thresholds | FD003 | 0.021 | [0.013, 0.033] | `\val{s4.FD003.oracle_e2_selection_loss}` | `reports/research/secondary/s4_threshold_variants.csv` | e2_selection_loss, _lo, _hi (tuning = oracle) | make research-secondary |
| S4 mean S over configurations, oracle | FD003 | 0.926 |  | `\val{s4.FD003.oracle_mean_savings_share}` | `reports/research/secondary/s4_threshold_variants.csv` | mean_savings_share (tuning = oracle) | make research-secondary |
| S4 e2_selection_loss with validation thresholds | FD004 | -0.129 | [-0.297, 0.019] | `\val{s4.FD004.validation_e2_selection_loss}` | `reports/research/secondary/s4_threshold_variants.csv` | e2_selection_loss, _lo, _hi (tuning = validation) | make research-secondary |
| S4 mean S over configurations, validation | FD004 | 0.787 |  | `\val{s4.FD004.validation_mean_savings_share}` | `reports/research/secondary/s4_threshold_variants.csv` | mean_savings_share (tuning = validation) | make research-secondary |
| S4 e2_selection_loss with crossfit thresholds | FD004 | -0.056 | [-0.138, 0.017] | `\val{s4.FD004.crossfit_e2_selection_loss}` | `reports/research/secondary/s4_threshold_variants.csv` | e2_selection_loss, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 mean S over configurations, crossfit | FD004 | 0.846 |  | `\val{s4.FD004.crossfit_mean_savings_share}` | `reports/research/secondary/s4_threshold_variants.csv` | mean_savings_share (tuning = crossfit) | make research-secondary |
| S4 e2_selection_loss with oracle thresholds | FD004 | 0.008 | [0.005, 0.012] | `\val{s4.FD004.oracle_e2_selection_loss}` | `reports/research/secondary/s4_threshold_variants.csv` | e2_selection_loss, _lo, _hi (tuning = oracle) | make research-secondary |
| S4 mean S over configurations, oracle | FD004 | 0.903 |  | `\val{s4.FD004.oracle_mean_savings_share}` | `reports/research/secondary/s4_threshold_variants.csv` | mean_savings_share (tuning = oracle) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 50 | FD001 | -0.016 | [-0.073, 0.033] | `\val{s6.FD001.cf50_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 50) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 100 | FD001 | -0.055 | [-0.139, 0.016] | `\val{s6.FD001.cf100_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 100) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 200 | FD001 | -0.161 | [-0.303, -0.052] | `\val{s6.FD001.cf200_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 200) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 500 | FD001 | -0.415 | [-0.736, -0.171] | `\val{s6.FD001.cf500_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 500) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 1000 | FD001 | -0.808 | [-1.472, -0.352] | `\val{s6.FD001.cf1000_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 1000) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 2000 | FD001 | -1.480 | [-2.894, -0.611] | `\val{s6.FD001.cf2000_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 2000) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 50 | FD002 | -0.008 | [-0.033, 0.016] | `\val{s6.FD002.cf50_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 50) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 100 | FD002 | -0.025 | [-0.063, 0.011] | `\val{s6.FD002.cf100_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 100) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 200 | FD002 | -0.088 | [-0.151, -0.031] | `\val{s6.FD002.cf200_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 200) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 500 | FD002 | -0.258 | [-0.410, -0.123] | `\val{s6.FD002.cf500_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 500) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 1000 | FD002 | -0.512 | [-0.789, -0.270] | `\val{s6.FD002.cf1000_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 1000) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 2000 | FD002 | -1.056 | [-1.642, -0.559] | `\val{s6.FD002.cf2000_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 2000) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 50 | FD003 | -0.024 | [-0.069, 0.023] | `\val{s6.FD003.cf50_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 50) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 100 | FD003 | -0.072 | [-0.130, -0.018] | `\val{s6.FD003.cf100_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 100) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 200 | FD003 | -0.162 | [-0.269, -0.065] | `\val{s6.FD003.cf200_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 200) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 500 | FD003 | -0.447 | [-0.711, -0.208] | `\val{s6.FD003.cf500_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 500) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 1000 | FD003 | -0.933 | [-1.459, -0.454] | `\val{s6.FD003.cf1000_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 1000) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 2000 | FD003 | -1.759 | [-2.931, -0.856] | `\val{s6.FD003.cf2000_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 2000) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 50 | FD004 | 0.061 | [0.038, 0.084] | `\val{s6.FD004.cf50_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 50) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 100 | FD004 | 0.039 | [0.016, 0.062] | `\val{s6.FD004.cf100_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 100) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 200 | FD004 | 0.022 | [-0.014, 0.054] | `\val{s6.FD004.cf200_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 200) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 500 | FD004 | -0.026 | [-0.101, 0.043] | `\val{s6.FD004.cf500_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 500) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 1000 | FD004 | -0.092 | [-0.231, 0.033] | `\val{s6.FD004.cf1000_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 1000) | make research-secondary |
| S6 e2_selection_loss with regret, C_f = 2000 | FD004 | -0.232 | [-0.517, 0.023] | `\val{s6.FD004.cf2000_e2_selection_loss}` | `reports/research/secondary/s6_regret.csv` | e2_selection_loss, _lo, _hi (failure_cost = 2000) | make research-secondary |
| F5 configuration with the highest S in hindsight | FD001 | transformer-05 |  | `\val{f5.FD001.oracle_config}` | `paper_assets/data/f5_oracle_selection.csv` | oracle_config | make paper-assets |
| F5 its S | FD001 | 0.863 |  | `\val{f5.FD001.oracle_s}` | `paper_assets/data/f5_oracle_selection.csv` | oracle_s | make paper-assets |
| F5 oracle-selection gap (descriptive, not pre-registered): S(best in hindsight) - S(chosen by validation cost) | FD001 | 1.166 | [0.619, 2.032] | `\val{f5.FD001.gap_cost}` | `paper_assets/data/f5_oracle_selection.csv` | gap_cost, _lo, _hi | make paper-assets |
| F5 oracle-selection gap (descriptive, not pre-registered): S(best in hindsight) - S(chosen by validation RMSE) | FD001 | 0.285 | [-0.010, 0.649] | `\val{f5.FD001.gap_rmse}` | `paper_assets/data/f5_oracle_selection.csv` | gap_rmse, _lo, _hi | make paper-assets |
| F5 configuration with the highest S in hindsight | FD002 | transformer-04 |  | `\val{f5.FD002.oracle_config}` | `paper_assets/data/f5_oracle_selection.csv` | oracle_config | make paper-assets |
| F5 its S | FD002 | 0.880 |  | `\val{f5.FD002.oracle_s}` | `paper_assets/data/f5_oracle_selection.csv` | oracle_s | make paper-assets |
| F5 oracle-selection gap (descriptive, not pre-registered): S(best in hindsight) - S(chosen by validation cost) | FD002 | 0.862 | [0.540, 1.225] | `\val{f5.FD002.gap_cost}` | `paper_assets/data/f5_oracle_selection.csv` | gap_cost, _lo, _hi | make paper-assets |
| F5 oracle-selection gap (descriptive, not pre-registered): S(best in hindsight) - S(chosen by validation RMSE) | FD002 | 0.241 | [0.081, 0.430] | `\val{f5.FD002.gap_rmse}` | `paper_assets/data/f5_oracle_selection.csv` | gap_rmse, _lo, _hi | make paper-assets |
| F5 configuration with the highest S in hindsight | FD003 | mlp-06 |  | `\val{f5.FD003.oracle_config}` | `paper_assets/data/f5_oracle_selection.csv` | oracle_config | make paper-assets |
| F5 its S | FD003 | 0.723 |  | `\val{f5.FD003.oracle_s}` | `paper_assets/data/f5_oracle_selection.csv` | oracle_s | make paper-assets |
| F5 oracle-selection gap (descriptive, not pre-registered): S(best in hindsight) - S(chosen by validation cost) | FD003 | 1.376 | [0.809, 2.190] | `\val{f5.FD003.gap_cost}` | `paper_assets/data/f5_oracle_selection.csv` | gap_cost, _lo, _hi | make paper-assets |
| F5 oracle-selection gap (descriptive, not pre-registered): S(best in hindsight) - S(chosen by validation RMSE) | FD003 | 0.240 | [-0.042, 0.603] | `\val{f5.FD003.gap_rmse}` | `paper_assets/data/f5_oracle_selection.csv` | gap_rmse, _lo, _hi | make paper-assets |
| F5 configuration with the highest S in hindsight | FD004 | transformer-11 |  | `\val{f5.FD004.oracle_config}` | `paper_assets/data/f5_oracle_selection.csv` | oracle_config | make paper-assets |
| F5 its S | FD004 | 0.876 |  | `\val{f5.FD004.oracle_s}` | `paper_assets/data/f5_oracle_selection.csv` | oracle_s | make paper-assets |
| F5 oracle-selection gap (descriptive, not pre-registered): S(best in hindsight) - S(chosen by validation cost) | FD004 | 0.275 | [0.123, 0.454] | `\val{f5.FD004.gap_cost}` | `paper_assets/data/f5_oracle_selection.csv` | gap_cost, _lo, _hi | make paper-assets |
| F5 oracle-selection gap (descriptive, not pre-registered): S(best in hindsight) - S(chosen by validation RMSE) | FD004 | 0.146 | [0.024, 0.291] | `\val{f5.FD004.gap_rmse}` | `paper_assets/data/f5_oracle_selection.csv` | gap_rmse, _lo, _hi | make paper-assets |

## RQ3: jitter against leaderboard RMSE (H3)

Pre-registered verdict: **rejected** (`reports/research/confirmatory/verdicts.md`, make research-confirmatory).

| quantity | fleet | estimate | 95% interval | macro | source | column | command |
|---|---|---|---|---|---|---|---|
| E3 delta-tau, confirmatory | FD001 | -0.068 | [-0.208, 0.192] | `\val{e3.FD001.delta_tau}` | `reports/research/confirmatory/e3.csv` | delta_tau, lo, hi | make research-confirmatory |
| tau-b(jitter, D) | FD001 | 0.219 | [-0.104, 0.306] | `\val{e3.FD001.tau_jitter}` | `reports/research/confirmatory/e3.csv` | tau_jitter, tau_jitter_lo, tau_jitter_hi | make research-confirmatory |
| E3 delta-tau, confirmatory | FD002 | -0.050 | [-0.209, 0.117] | `\val{e3.FD002.delta_tau}` | `reports/research/confirmatory/e3.csv` | delta_tau, lo, hi | make research-confirmatory |
| tau-b(jitter, D) | FD002 | -0.008 | [-0.247, 0.222] | `\val{e3.FD002.tau_jitter}` | `reports/research/confirmatory/e3.csv` | tau_jitter, tau_jitter_lo, tau_jitter_hi | make research-confirmatory |
| E3 delta-tau, confirmatory | FD003 | -0.220 | [-0.396, 0.126] | `\val{e3.FD003.delta_tau}` | `reports/research/confirmatory/e3.csv` | delta_tau, lo, hi | make research-confirmatory |
| tau-b(jitter, D) | FD003 | -0.257 | [-0.331, 0.039] | `\val{e3.FD003.tau_jitter}` | `reports/research/confirmatory/e3.csv` | tau_jitter, tau_jitter_lo, tau_jitter_hi | make research-confirmatory |
| E3 delta-tau, confirmatory | FD004 | 0.027 | [-0.154, 0.217] | `\val{e3.FD004.delta_tau}` | `reports/research/confirmatory/e3.csv` | delta_tau, lo, hi | make research-confirmatory |
| tau-b(jitter, D) | FD004 | -0.035 | [-0.235, 0.199] | `\val{e3.FD004.tau_jitter}` | `reports/research/confirmatory/e3.csv` | tau_jitter, tau_jitter_lo, tau_jitter_hi | make research-confirmatory |
| E3 delta-tau, secondary S8, descriptive | NCMAPSS | -0.579 | [-0.702, -0.302] | `\val{e3.NCMAPSS.delta_tau}` | `reports/research/ncmapss/e3.csv` | delta_tau, lo, hi | make research-ncmapss-report |
| tau-b(jitter, D) | NCMAPSS | 0.098 | [-0.109, 0.182] | `\val{e3.NCMAPSS.tau_jitter}` | `reports/research/ncmapss/e3.csv` | tau_jitter, tau_jitter_lo, tau_jitter_hi | make research-ncmapss-report |
| pre-registered verdict on H3 | all | rejected |  | `\val{verdict.all.H3}` | `reports/research/confirmatory/verdicts.md` | heading '## H3' | make research-confirmatory |
| S2 smallest E3 over the 21 settings of R1 | FD001 | -0.316 |  | `\val{s2.FD001.R1_e3_delta_tau_min}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R1) | make research-secondary |
| S2 largest E3 over the 21 settings of R1 | FD001 | 0.049 |  | `\val{s2.FD001.R1_e3_delta_tau_max}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R1) | make research-secondary |
| S2 smallest E3 over the 21 settings of R2 | FD001 | -0.369 |  | `\val{s2.FD001.R2_e3_delta_tau_min}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R2) | make research-secondary |
| S2 largest E3 over the 21 settings of R2 | FD001 | 0.061 |  | `\val{s2.FD001.R2_e3_delta_tau_max}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R2) | make research-secondary |
| S2 smallest E3 over the 21 settings of R1 | FD002 | -0.268 |  | `\val{s2.FD002.R1_e3_delta_tau_min}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R1) | make research-secondary |
| S2 largest E3 over the 21 settings of R1 | FD002 | 0.050 |  | `\val{s2.FD002.R1_e3_delta_tau_max}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R1) | make research-secondary |
| S2 smallest E3 over the 21 settings of R2 | FD002 | -0.337 |  | `\val{s2.FD002.R2_e3_delta_tau_min}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R2) | make research-secondary |
| S2 largest E3 over the 21 settings of R2 | FD002 | 0.096 |  | `\val{s2.FD002.R2_e3_delta_tau_max}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R2) | make research-secondary |
| S2 smallest E3 over the 21 settings of R1 | FD003 | -0.335 |  | `\val{s2.FD003.R1_e3_delta_tau_min}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R1) | make research-secondary |
| S2 largest E3 over the 21 settings of R1 | FD003 | 0.246 |  | `\val{s2.FD003.R1_e3_delta_tau_max}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R1) | make research-secondary |
| S2 smallest E3 over the 21 settings of R2 | FD003 | -0.440 |  | `\val{s2.FD003.R2_e3_delta_tau_min}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R2) | make research-secondary |
| S2 largest E3 over the 21 settings of R2 | FD003 | 0.163 |  | `\val{s2.FD003.R2_e3_delta_tau_max}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R2) | make research-secondary |
| S2 smallest E3 over the 21 settings of R1 | FD004 | -0.286 |  | `\val{s2.FD004.R1_e3_delta_tau_min}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R1) | make research-secondary |
| S2 largest E3 over the 21 settings of R1 | FD004 | 0.344 |  | `\val{s2.FD004.R1_e3_delta_tau_max}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R1) | make research-secondary |
| S2 smallest E3 over the 21 settings of R2 | FD004 | -0.284 |  | `\val{s2.FD004.R2_e3_delta_tau_min}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R2) | make research-secondary |
| S2 largest E3 over the 21 settings of R2 | FD004 | 0.333 |  | `\val{s2.FD004.R2_e3_delta_tau_max}` | `reports/research/secondary/s2_settings_grid.csv` | e3_delta_tau (rule = R2) | make research-secondary |
| S2 settings (of 42) where the H3 rule reads rejected | all | 27 |  | `\val{s2.all.H3_rejected_settings}` | `reports/research/secondary/s2_rules.csv` | H3_rule | make research-secondary |
| S2 settings (of 42) where the H3 rule reads inconclusive | all | 15 |  | `\val{s2.all.H3_inconclusive_settings}` | `reports/research/secondary/s2_rules.csv` | H3_rule | make research-secondary |
| S4 e3_delta_tau with validation thresholds | FD001 | -0.068 | [-0.208, 0.192] | `\val{s4.FD001.validation_e3_delta_tau}` | `reports/research/secondary/s4_threshold_variants.csv` | e3_delta_tau, _lo, _hi (tuning = validation) | make research-secondary |
| S4 e3_delta_tau with crossfit thresholds | FD001 | -0.138 | [-0.226, 0.124] | `\val{s4.FD001.crossfit_e3_delta_tau}` | `reports/research/secondary/s4_threshold_variants.csv` | e3_delta_tau, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 e3_delta_tau with oracle thresholds | FD001 | -0.141 | [-0.230, 0.027] | `\val{s4.FD001.oracle_e3_delta_tau}` | `reports/research/secondary/s4_threshold_variants.csv` | e3_delta_tau, _lo, _hi (tuning = oracle) | make research-secondary |
| S4 e3_delta_tau with validation thresholds | FD002 | -0.050 | [-0.209, 0.117] | `\val{s4.FD002.validation_e3_delta_tau}` | `reports/research/secondary/s4_threshold_variants.csv` | e3_delta_tau, _lo, _hi (tuning = validation) | make research-secondary |
| S4 e3_delta_tau with crossfit thresholds | FD002 | 0.051 | [-0.101, 0.197] | `\val{s4.FD002.crossfit_e3_delta_tau}` | `reports/research/secondary/s4_threshold_variants.csv` | e3_delta_tau, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 e3_delta_tau with oracle thresholds | FD002 | -0.137 | [-0.167, 0.028] | `\val{s4.FD002.oracle_e3_delta_tau}` | `reports/research/secondary/s4_threshold_variants.csv` | e3_delta_tau, _lo, _hi (tuning = oracle) | make research-secondary |
| S4 e3_delta_tau with validation thresholds | FD003 | -0.220 | [-0.396, 0.126] | `\val{s4.FD003.validation_e3_delta_tau}` | `reports/research/secondary/s4_threshold_variants.csv` | e3_delta_tau, _lo, _hi (tuning = validation) | make research-secondary |
| S4 e3_delta_tau with crossfit thresholds | FD003 | 0.018 | [-0.229, 0.279] | `\val{s4.FD003.crossfit_e3_delta_tau}` | `reports/research/secondary/s4_threshold_variants.csv` | e3_delta_tau, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 e3_delta_tau with oracle thresholds | FD003 | 0.101 | [-0.042, 0.193] | `\val{s4.FD003.oracle_e3_delta_tau}` | `reports/research/secondary/s4_threshold_variants.csv` | e3_delta_tau, _lo, _hi (tuning = oracle) | make research-secondary |
| S4 e3_delta_tau with validation thresholds | FD004 | 0.027 | [-0.154, 0.217] | `\val{s4.FD004.validation_e3_delta_tau}` | `reports/research/secondary/s4_threshold_variants.csv` | e3_delta_tau, _lo, _hi (tuning = validation) | make research-secondary |
| S4 e3_delta_tau with crossfit thresholds | FD004 | 0.301 | [0.078, 0.365] | `\val{s4.FD004.crossfit_e3_delta_tau}` | `reports/research/secondary/s4_threshold_variants.csv` | e3_delta_tau, _lo, _hi (tuning = crossfit) | make research-secondary |
| S4 e3_delta_tau with oracle thresholds | FD004 | 0.234 | [0.131, 0.300] | `\val{s4.FD004.oracle_e3_delta_tau}` | `reports/research/secondary/s4_threshold_variants.csv` | e3_delta_tau, _lo, _hi (tuning = oracle) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 50 | FD001 | -0.202 | [-0.303, -0.022] | `\val{s6.FD001.cf50_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 50) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 100 | FD001 | -0.176 | [-0.277, 0.045] | `\val{s6.FD001.cf100_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 100) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 200 | FD001 | -0.113 | [-0.233, 0.132] | `\val{s6.FD001.cf200_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 200) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 500 | FD001 | -0.079 | [-0.216, 0.174] | `\val{s6.FD001.cf500_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 500) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 1000 | FD001 | -0.064 | [-0.209, 0.187] | `\val{s6.FD001.cf1000_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 1000) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 2000 | FD001 | -0.069 | [-0.207, 0.192] | `\val{s6.FD001.cf2000_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 2000) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 50 | FD002 | -0.240 | [-0.286, -0.072] | `\val{s6.FD002.cf50_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 50) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 100 | FD002 | -0.172 | [-0.242, -0.007] | `\val{s6.FD002.cf100_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 100) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 200 | FD002 | -0.119 | [-0.198, 0.070] | `\val{s6.FD002.cf200_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 200) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 500 | FD002 | -0.087 | [-0.196, 0.111] | `\val{s6.FD002.cf500_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 500) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 1000 | FD002 | -0.057 | [-0.211, 0.111] | `\val{s6.FD002.cf1000_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 1000) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 2000 | FD002 | -0.043 | [-0.211, 0.121] | `\val{s6.FD002.cf2000_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 2000) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 50 | FD003 | -0.076 | [-0.243, 0.107] | `\val{s6.FD003.cf50_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 50) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 100 | FD003 | -0.059 | [-0.262, 0.140] | `\val{s6.FD003.cf100_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 100) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 200 | FD003 | -0.116 | [-0.320, 0.145] | `\val{s6.FD003.cf200_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 200) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 500 | FD003 | -0.183 | [-0.369, 0.141] | `\val{s6.FD003.cf500_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 500) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 1000 | FD003 | -0.208 | [-0.389, 0.131] | `\val{s6.FD003.cf1000_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 1000) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 2000 | FD003 | -0.226 | [-0.400, 0.127] | `\val{s6.FD003.cf2000_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 2000) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 50 | FD004 | -0.051 | [-0.173, 0.078] | `\val{s6.FD004.cf50_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 50) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 100 | FD004 | 0.081 | [-0.091, 0.199] | `\val{s6.FD004.cf100_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 100) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 200 | FD004 | 0.146 | [-0.046, 0.264] | `\val{s6.FD004.cf200_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 200) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 500 | FD004 | 0.152 | [-0.058, 0.256] | `\val{s6.FD004.cf500_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 500) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 1000 | FD004 | 0.115 | [-0.092, 0.250] | `\val{s6.FD004.cf1000_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 1000) | make research-secondary |
| S6 e3_delta_tau with regret, C_f = 2000 | FD004 | 0.017 | [-0.159, 0.213] | `\val{s6.FD004.cf2000_e3_delta_tau}` | `reports/research/secondary/s6_regret.csv` | e3_delta_tau, _lo, _hi (failure_cost = 2000) | make research-secondary |

## Family level (S3, secondary)

| quantity | fleet | estimate | 95% interval | macro | source | column | command |
|---|---|---|---|---|---|---|---|
| S3 S of the ridge representative | FD001 | 0.451 | [-0.072, 0.767] | `\val{s3.FD001.ridge_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, ridge | FD001 | 15.19 | [13.25, 17.11] | `\val{s3.FD001.ridge_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, ridge | FD001 | 8 |  | `\val{s3.FD001.ridge_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, ridge | FD001 | 6 |  | `\val{s3.FD001.ridge_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, ridge | FD001 | 0.037 |  | `\val{s3.FD001.ridge_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, ridge | FD001 | 11.7 |  | `\val{s3.FD001.ridge_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the knn representative | FD001 | 0.351 | [-0.291, 0.750] | `\val{s3.FD001.knn_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, knn | FD001 | 16.54 | [13.98, 18.99] | `\val{s3.FD001.knn_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, knn | FD001 | 9 |  | `\val{s3.FD001.knn_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, knn | FD001 | 8 |  | `\val{s3.FD001.knn_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, knn | FD001 | 0.047 |  | `\val{s3.FD001.knn_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, knn | FD001 | 8.8 |  | `\val{s3.FD001.knn_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the forest representative | FD001 | 0.551 | [0.094, 0.842] | `\val{s3.FD001.forest_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, forest | FD001 | 13.04 | [10.98, 15.03] | `\val{s3.FD001.forest_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, forest | FD001 | 3 |  | `\val{s3.FD001.forest_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, forest | FD001 | 3 |  | `\val{s3.FD001.forest_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, forest | FD001 | 0.033 |  | `\val{s3.FD001.forest_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, forest | FD001 | 5.1 |  | `\val{s3.FD001.forest_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the xgboost representative | FD001 | 0.372 | [-0.165, 0.730] | `\val{s3.FD001.xgboost_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, xgboost | FD001 | 12.50 | [10.42, 14.62] | `\val{s3.FD001.xgboost_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, xgboost | FD001 | 1 |  | `\val{s3.FD001.xgboost_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, xgboost | FD001 | 7 |  | `\val{s3.FD001.xgboost_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, xgboost | FD001 | 0.047 |  | `\val{s3.FD001.xgboost_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, xgboost | FD001 | 6.6 |  | `\val{s3.FD001.xgboost_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the xgboost_raw representative | FD001 | 0.174 | [-0.603, 0.677] | `\val{s3.FD001.xgboost_raw_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, xgboost_raw | FD001 | 16.75 | [14.00, 19.27] | `\val{s3.FD001.xgboost_raw_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, xgboost_raw | FD001 | 10 |  | `\val{s3.FD001.xgboost_raw_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, xgboost_raw | FD001 | 9 |  | `\val{s3.FD001.xgboost_raw_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, xgboost_raw | FD001 | 0.060 |  | `\val{s3.FD001.xgboost_raw_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, xgboost_raw | FD001 | 9.7 |  | `\val{s3.FD001.xgboost_raw_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the mlp representative | FD001 | 0.130 | [-0.621, 0.597] | `\val{s3.FD001.mlp_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, mlp | FD001 | 14.90 | [12.52, 17.15] | `\val{s3.FD001.mlp_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, mlp | FD001 | 7 |  | `\val{s3.FD001.mlp_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, mlp | FD001 | 10 |  | `\val{s3.FD001.mlp_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, mlp | FD001 | 0.063 |  | `\val{s3.FD001.mlp_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, mlp | FD001 | 9.8 |  | `\val{s3.FD001.mlp_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the lstm representative | FD001 | 0.615 | [0.224, 0.868] | `\val{s3.FD001.lstm_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, lstm | FD001 | 14.75 | [12.35, 17.15] | `\val{s3.FD001.lstm_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, lstm | FD001 | 6 |  | `\val{s3.FD001.lstm_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, lstm | FD001 | 1 |  | `\val{s3.FD001.lstm_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, lstm | FD001 | 0.027 |  | `\val{s3.FD001.lstm_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, lstm | FD001 | 7.3 |  | `\val{s3.FD001.lstm_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the tcn representative | FD001 | 0.535 | [0.051, 0.832] | `\val{s3.FD001.tcn_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, tcn | FD001 | 14.15 | [11.97, 16.17] | `\val{s3.FD001.tcn_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, tcn | FD001 | 5 |  | `\val{s3.FD001.tcn_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, tcn | FD001 | 4 |  | `\val{s3.FD001.tcn_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, tcn | FD001 | 0.033 |  | `\val{s3.FD001.tcn_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, tcn | FD001 | 7.0 |  | `\val{s3.FD001.tcn_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the cnn representative | FD001 | 0.460 | [0.027, 0.748] | `\val{s3.FD001.cnn_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, cnn | FD001 | 12.58 | [10.17, 14.95] | `\val{s3.FD001.cnn_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, cnn | FD001 | 2 |  | `\val{s3.FD001.cnn_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, cnn | FD001 | 5 |  | `\val{s3.FD001.cnn_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, cnn | FD001 | 0.040 |  | `\val{s3.FD001.cnn_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, cnn | FD001 | 6.0 |  | `\val{s3.FD001.cnn_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the transformer representative | FD001 | 0.581 | [0.147, 0.851] | `\val{s3.FD001.transformer_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, transformer | FD001 | 14.03 | [11.70, 16.28] | `\val{s3.FD001.transformer_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, transformer | FD001 | 4 |  | `\val{s3.FD001.transformer_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, transformer | FD001 | 2 |  | `\val{s3.FD001.transformer_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, transformer | FD001 | 0.030 |  | `\val{s3.FD001.transformer_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, transformer | FD001 | 6.5 |  | `\val{s3.FD001.transformer_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the ridge representative | FD002 | 0.483 | [0.204, 0.705] | `\val{s3.FD002.ridge_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, ridge | FD002 | 16.04 | [14.71, 17.36] | `\val{s3.FD002.ridge_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, ridge | FD002 | 8 |  | `\val{s3.FD002.ridge_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, ridge | FD002 | 10 |  | `\val{s3.FD002.ridge_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, ridge | FD002 | 0.031 |  | `\val{s3.FD002.ridge_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, ridge | FD002 | 11.0 |  | `\val{s3.FD002.ridge_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the knn representative | FD002 | 0.594 | [0.352, 0.784] | `\val{s3.FD002.knn_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, knn | FD002 | 16.29 | [14.82, 17.75] | `\val{s3.FD002.knn_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, knn | FD002 | 9 |  | `\val{s3.FD002.knn_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, knn | FD002 | 8 |  | `\val{s3.FD002.knn_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, knn | FD002 | 0.022 |  | `\val{s3.FD002.knn_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, knn | FD002 | 12.6 |  | `\val{s3.FD002.knn_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the forest representative | FD002 | 0.608 | [0.370, 0.795] | `\val{s3.FD002.forest_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, forest | FD002 | 13.26 | [11.82, 14.55] | `\val{s3.FD002.forest_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, forest | FD002 | 3 |  | `\val{s3.FD002.forest_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, forest | FD002 | 7 |  | `\val{s3.FD002.forest_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, forest | FD002 | 0.024 |  | `\val{s3.FD002.forest_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, forest | FD002 | 7.2 |  | `\val{s3.FD002.forest_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the xgboost representative | FD002 | 0.561 | [0.307, 0.747] | `\val{s3.FD002.xgboost_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, xgboost | FD002 | 12.77 | [11.47, 14.00] | `\val{s3.FD002.xgboost_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, xgboost | FD002 | 1 |  | `\val{s3.FD002.xgboost_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, xgboost | FD002 | 9 |  | `\val{s3.FD002.xgboost_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, xgboost | FD002 | 0.027 |  | `\val{s3.FD002.xgboost_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, xgboost | FD002 | 8.5 |  | `\val{s3.FD002.xgboost_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the xgboost_raw representative | FD002 | 0.646 | [0.416, 0.815] | `\val{s3.FD002.xgboost_raw_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, xgboost_raw | FD002 | 16.39 | [14.78, 17.97] | `\val{s3.FD002.xgboost_raw_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, xgboost_raw | FD002 | 10 |  | `\val{s3.FD002.xgboost_raw_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, xgboost_raw | FD002 | 6 |  | `\val{s3.FD002.xgboost_raw_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, xgboost_raw | FD002 | 0.017 |  | `\val{s3.FD002.xgboost_raw_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, xgboost_raw | FD002 | 14.8 |  | `\val{s3.FD002.xgboost_raw_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the mlp representative | FD002 | 0.717 | [0.564, 0.830] | `\val{s3.FD002.mlp_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, mlp | FD002 | 14.09 | [12.92, 15.26] | `\val{s3.FD002.mlp_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, mlp | FD002 | 7 |  | `\val{s3.FD002.mlp_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, mlp | FD002 | 4 |  | `\val{s3.FD002.mlp_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, mlp | FD002 | 0.014 |  | `\val{s3.FD002.mlp_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, mlp | FD002 | 11.1 |  | `\val{s3.FD002.mlp_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the lstm representative | FD002 | 0.791 | [0.656, 0.892] | `\val{s3.FD002.lstm_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, lstm | FD002 | 13.76 | [12.35, 15.22] | `\val{s3.FD002.lstm_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, lstm | FD002 | 6 |  | `\val{s3.FD002.lstm_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, lstm | FD002 | 1 |  | `\val{s3.FD002.lstm_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, lstm | FD002 | 0.010 |  | `\val{s3.FD002.lstm_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, lstm | FD002 | 8.7 |  | `\val{s3.FD002.lstm_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the tcn representative | FD002 | 0.670 | [0.468, 0.823] | `\val{s3.FD002.tcn_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, tcn | FD002 | 13.69 | [12.37, 15.05] | `\val{s3.FD002.tcn_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, tcn | FD002 | 5 |  | `\val{s3.FD002.tcn_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, tcn | FD002 | 5 |  | `\val{s3.FD002.tcn_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, tcn | FD002 | 0.019 |  | `\val{s3.FD002.tcn_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, tcn | FD002 | 8.2 |  | `\val{s3.FD002.tcn_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the cnn representative | FD002 | 0.726 | [0.580, 0.843] | `\val{s3.FD002.cnn_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, cnn | FD002 | 13.13 | [11.73, 14.52] | `\val{s3.FD002.cnn_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, cnn | FD002 | 2 |  | `\val{s3.FD002.cnn_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, cnn | FD002 | 3 |  | `\val{s3.FD002.cnn_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, cnn | FD002 | 0.014 |  | `\val{s3.FD002.cnn_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, cnn | FD002 | 10.0 |  | `\val{s3.FD002.cnn_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the transformer representative | FD002 | 0.758 | [0.630, 0.857] | `\val{s3.FD002.transformer_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, transformer | FD002 | 13.50 | [12.02, 15.00] | `\val{s3.FD002.transformer_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, transformer | FD002 | 4 |  | `\val{s3.FD002.transformer_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, transformer | FD002 | 2 |  | `\val{s3.FD002.transformer_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, transformer | FD002 | 0.013 |  | `\val{s3.FD002.transformer_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, transformer | FD002 | 8.4 |  | `\val{s3.FD002.transformer_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the ridge representative | FD003 | 0.269 | [-0.385, 0.695] | `\val{s3.FD003.ridge_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, ridge | FD003 | 16.28 | [13.99, 18.39] | `\val{s3.FD003.ridge_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, ridge | FD003 | 8 |  | `\val{s3.FD003.ridge_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, ridge | FD003 | 9 |  | `\val{s3.FD003.ridge_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, ridge | FD003 | 0.060 |  | `\val{s3.FD003.ridge_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, ridge | FD003 | 13.5 |  | `\val{s3.FD003.ridge_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the knn representative | FD003 | 0.459 | [-0.027, 0.803] | `\val{s3.FD003.knn_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, knn | FD003 | 19.23 | [16.04, 22.25] | `\val{s3.FD003.knn_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, knn | FD003 | 10 |  | `\val{s3.FD003.knn_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, knn | FD003 | 6 |  | `\val{s3.FD003.knn_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, knn | FD003 | 0.047 |  | `\val{s3.FD003.knn_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, knn | FD003 | 7.4 |  | `\val{s3.FD003.knn_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the forest representative | FD003 | 0.473 | [-0.060, 0.834] | `\val{s3.FD003.forest_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, forest | FD003 | 13.99 | [11.60, 16.23] | `\val{s3.FD003.forest_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, forest | FD003 | 5 |  | `\val{s3.FD003.forest_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, forest | FD003 | 5 |  | `\val{s3.FD003.forest_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, forest | FD003 | 0.047 |  | `\val{s3.FD003.forest_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, forest | FD003 | 5.4 |  | `\val{s3.FD003.forest_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the xgboost representative | FD003 | 0.456 | [-0.030, 0.778] | `\val{s3.FD003.xgboost_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, xgboost | FD003 | 12.66 | [10.61, 14.63] | `\val{s3.FD003.xgboost_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, xgboost | FD003 | 2 |  | `\val{s3.FD003.xgboost_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, xgboost | FD003 | 7 |  | `\val{s3.FD003.xgboost_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, xgboost | FD003 | 0.047 |  | `\val{s3.FD003.xgboost_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, xgboost | FD003 | 7.9 |  | `\val{s3.FD003.xgboost_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the xgboost_raw representative | FD003 | 0.601 | [0.152, 0.888] | `\val{s3.FD003.xgboost_raw_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, xgboost_raw | FD003 | 17.04 | [14.35, 19.67] | `\val{s3.FD003.xgboost_raw_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, xgboost_raw | FD003 | 9 |  | `\val{s3.FD003.xgboost_raw_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, xgboost_raw | FD003 | 3 |  | `\val{s3.FD003.xgboost_raw_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, xgboost_raw | FD003 | 0.033 |  | `\val{s3.FD003.xgboost_raw_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, xgboost_raw | FD003 | 7.8 |  | `\val{s3.FD003.xgboost_raw_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the mlp representative | FD003 | 0.232 | [-0.336, 0.615] | `\val{s3.FD003.mlp_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, mlp | FD003 | 13.39 | [11.37, 15.37] | `\val{s3.FD003.mlp_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, mlp | FD003 | 4 |  | `\val{s3.FD003.mlp_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, mlp | FD003 | 10 |  | `\val{s3.FD003.mlp_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, mlp | FD003 | 0.067 |  | `\val{s3.FD003.mlp_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, mlp | FD003 | 8.8 |  | `\val{s3.FD003.mlp_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the lstm representative | FD003 | 0.600 | [0.258, 0.833] | `\val{s3.FD003.lstm_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, lstm | FD003 | 14.84 | [12.48, 17.04] | `\val{s3.FD003.lstm_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, lstm | FD003 | 7 |  | `\val{s3.FD003.lstm_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, lstm | FD003 | 4 |  | `\val{s3.FD003.lstm_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, lstm | FD003 | 0.033 |  | `\val{s3.FD003.lstm_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, lstm | FD003 | 7.9 |  | `\val{s3.FD003.lstm_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the tcn representative | FD003 | 0.279 | [-0.208, 0.622] | `\val{s3.FD003.tcn_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, tcn | FD003 | 14.53 | [12.18, 16.71] | `\val{s3.FD003.tcn_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, tcn | FD003 | 6 |  | `\val{s3.FD003.tcn_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, tcn | FD003 | 8 |  | `\val{s3.FD003.tcn_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, tcn | FD003 | 0.063 |  | `\val{s3.FD003.tcn_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, tcn | FD003 | 7.4 |  | `\val{s3.FD003.tcn_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the cnn representative | FD003 | 0.696 | [0.431, 0.873] | `\val{s3.FD003.cnn_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, cnn | FD003 | 12.38 | [10.24, 14.46] | `\val{s3.FD003.cnn_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, cnn | FD003 | 1 |  | `\val{s3.FD003.cnn_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, cnn | FD003 | 1 |  | `\val{s3.FD003.cnn_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, cnn | FD003 | 0.023 |  | `\val{s3.FD003.cnn_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, cnn | FD003 | 9.9 |  | `\val{s3.FD003.cnn_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the transformer representative | FD003 | 0.669 | [0.337, 0.889] | `\val{s3.FD003.transformer_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, transformer | FD003 | 13.33 | [11.21, 15.36] | `\val{s3.FD003.transformer_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, transformer | FD003 | 3 |  | `\val{s3.FD003.transformer_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, transformer | FD003 | 2 |  | `\val{s3.FD003.transformer_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, transformer | FD003 | 0.027 |  | `\val{s3.FD003.transformer_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, transformer | FD003 | 8.5 |  | `\val{s3.FD003.transformer_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the ridge representative | FD004 | 0.782 | [0.632, 0.887] | `\val{s3.FD004.ridge_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, ridge | FD004 | 18.38 | [16.64, 19.98] | `\val{s3.FD004.ridge_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, ridge | FD004 | 10 |  | `\val{s3.FD004.ridge_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, ridge | FD004 | 7 |  | `\val{s3.FD004.ridge_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, ridge | FD004 | 0.016 |  | `\val{s3.FD004.ridge_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, ridge | FD004 | 16.4 |  | `\val{s3.FD004.ridge_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the knn representative | FD004 | 0.813 | [0.691, 0.903] | `\val{s3.FD004.knn_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, knn | FD004 | 18.27 | [16.34, 20.16] | `\val{s3.FD004.knn_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, knn | FD004 | 9 |  | `\val{s3.FD004.knn_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, knn | FD004 | 2 |  | `\val{s3.FD004.knn_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, knn | FD004 | 0.012 |  | `\val{s3.FD004.knn_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, knn | FD004 | 17.6 |  | `\val{s3.FD004.knn_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the forest representative | FD004 | 0.788 | [0.677, 0.874] | `\val{s3.FD004.forest_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, forest | FD004 | 14.66 | [12.97, 16.35] | `\val{s3.FD004.forest_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, forest | FD004 | 6 |  | `\val{s3.FD004.forest_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, forest | FD004 | 5 |  | `\val{s3.FD004.forest_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, forest | FD004 | 0.016 |  | `\val{s3.FD004.forest_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, forest | FD004 | 15.3 |  | `\val{s3.FD004.forest_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the xgboost representative | FD004 | 0.783 | [0.667, 0.872] | `\val{s3.FD004.xgboost_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, xgboost | FD004 | 13.84 | [12.16, 15.53] | `\val{s3.FD004.xgboost_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, xgboost | FD004 | 2 |  | `\val{s3.FD004.xgboost_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, xgboost | FD004 | 6 |  | `\val{s3.FD004.xgboost_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, xgboost | FD004 | 0.015 |  | `\val{s3.FD004.xgboost_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, xgboost | FD004 | 18.6 |  | `\val{s3.FD004.xgboost_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the xgboost_raw representative | FD004 | 0.811 | [0.694, 0.891] | `\val{s3.FD004.xgboost_raw_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, xgboost_raw | FD004 | 17.93 | [15.81, 19.90] | `\val{s3.FD004.xgboost_raw_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, xgboost_raw | FD004 | 8 |  | `\val{s3.FD004.xgboost_raw_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, xgboost_raw | FD004 | 3 |  | `\val{s3.FD004.xgboost_raw_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, xgboost_raw | FD004 | 0.008 |  | `\val{s3.FD004.xgboost_raw_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, xgboost_raw | FD004 | 25.3 |  | `\val{s3.FD004.xgboost_raw_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the mlp representative | FD004 | 0.746 | [0.637, 0.830] | `\val{s3.FD004.mlp_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, mlp | FD004 | 14.49 | [12.81, 16.21] | `\val{s3.FD004.mlp_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, mlp | FD004 | 5 |  | `\val{s3.FD004.mlp_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, mlp | FD004 | 10 |  | `\val{s3.FD004.mlp_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, mlp | FD004 | 0.019 |  | `\val{s3.FD004.mlp_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, mlp | FD004 | 18.6 |  | `\val{s3.FD004.mlp_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the lstm representative | FD004 | 0.773 | [0.623, 0.878] | `\val{s3.FD004.lstm_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, lstm | FD004 | 14.34 | [12.80, 15.99] | `\val{s3.FD004.lstm_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, lstm | FD004 | 4 |  | `\val{s3.FD004.lstm_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, lstm | FD004 | 8 |  | `\val{s3.FD004.lstm_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, lstm | FD004 | 0.016 |  | `\val{s3.FD004.lstm_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, lstm | FD004 | 18.1 |  | `\val{s3.FD004.lstm_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the tcn representative | FD004 | 0.830 | [0.732, 0.902] | `\val{s3.FD004.tcn_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, tcn | FD004 | 15.56 | [13.95, 17.22] | `\val{s3.FD004.tcn_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, tcn | FD004 | 7 |  | `\val{s3.FD004.tcn_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, tcn | FD004 | 1 |  | `\val{s3.FD004.tcn_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, tcn | FD004 | 0.012 |  | `\val{s3.FD004.tcn_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, tcn | FD004 | 14.2 |  | `\val{s3.FD004.tcn_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the cnn representative | FD004 | 0.753 | [0.619, 0.849] | `\val{s3.FD004.cnn_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, cnn | FD004 | 13.60 | [11.83, 15.33] | `\val{s3.FD004.cnn_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, cnn | FD004 | 1 |  | `\val{s3.FD004.cnn_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, cnn | FD004 | 9 |  | `\val{s3.FD004.cnn_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, cnn | FD004 | 0.019 |  | `\val{s3.FD004.cnn_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, cnn | FD004 | 17.4 |  | `\val{s3.FD004.cnn_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 S of the transformer representative | FD004 | 0.797 | [0.695, 0.872] | `\val{s3.FD004.transformer_savings_share}` | `reports/research/secondary/s3_families.csv` | savings_share, _lo, _hi | make research-secondary |
| S3 leaderboard RMSE, transformer | FD004 | 13.97 | [12.18, 15.80] | `\val{s3.FD004.transformer_leaderboard_rmse}` | `reports/research/secondary/s3_families.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S3 rank_rmse, transformer | FD004 | 3 |  | `\val{s3.FD004.transformer_rank_rmse}` | `reports/research/secondary/s3_families.csv` | rank_rmse | make research-secondary |
| S3 rank_value, transformer | FD004 | 4 |  | `\val{s3.FD004.transformer_rank_value}` | `reports/research/secondary/s3_families.csv` | rank_value | make research-secondary |
| S3 failure_rate, transformer | FD004 | 0.015 |  | `\val{s3.FD004.transformer_failure_rate}` | `reports/research/secondary/s3_families.csv` | failure_rate | make research-secondary |
| S3 waste_per_maintenance, transformer | FD004 | 16.0 |  | `\val{s3.FD004.transformer_waste_per_maintenance}` | `reports/research/secondary/s3_families.csv` | waste_per_maintenance | make research-secondary |
| S3 inverted family pairs (Benjamini-Hochberg) | FD001 | 0 |  | `\val{s3.FD001.inversions}` | `reports/research/secondary/s3_pairwise.csv` | inversion | make research-secondary |
| S3 inverted family pairs (intervals) | FD001 | 0 |  | `\val{s3.FD001.inversions_unadjusted}` | `reports/research/secondary/s3_pairwise.csv` | inversion_unadjusted | make research-secondary |
| S3 inverted family pairs (Benjamini-Hochberg) | FD002 | 0 |  | `\val{s3.FD002.inversions}` | `reports/research/secondary/s3_pairwise.csv` | inversion | make research-secondary |
| S3 inverted family pairs (intervals) | FD002 | 0 |  | `\val{s3.FD002.inversions_unadjusted}` | `reports/research/secondary/s3_pairwise.csv` | inversion_unadjusted | make research-secondary |
| S3 inverted family pairs (Benjamini-Hochberg) | FD003 | 0 |  | `\val{s3.FD003.inversions}` | `reports/research/secondary/s3_pairwise.csv` | inversion | make research-secondary |
| S3 inverted family pairs (intervals) | FD003 | 0 |  | `\val{s3.FD003.inversions_unadjusted}` | `reports/research/secondary/s3_pairwise.csv` | inversion_unadjusted | make research-secondary |
| S3 inverted family pairs (Benjamini-Hochberg) | FD004 | 0 |  | `\val{s3.FD004.inversions}` | `reports/research/secondary/s3_pairwise.csv` | inversion | make research-secondary |
| S3 inverted family pairs (intervals) | FD004 | 0 |  | `\val{s3.FD004.inversions_unadjusted}` | `reports/research/secondary/s3_pairwise.csv` | inversion_unadjusted | make research-secondary |

## Training cap (S5, secondary)

| quantity | fleet | estimate | 95% interval | macro | source | column | command |
|---|---|---|---|---|---|---|---|
| S5 leaderboard RMSE, xgboost, training cap 125 | FD001 | 12.65 | [10.56, 14.71] | `\val{s5.FD001.xgboost_cap125_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 125 | FD001 | 0.712 | [0.325, 0.967] | `\val{s5.FD001.xgboost_cap125_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 100 | FD001 | 15.19 | [12.93, 17.14] | `\val{s5.FD001.xgboost_cap100_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 100 | FD001 | 0.469 | [-0.163, 0.876] | `\val{s5.FD001.xgboost_cap100_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 150 | FD001 | 16.10 | [13.73, 18.48] | `\val{s5.FD001.xgboost_cap150_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 150 | FD001 | 0.584 | [0.061, 0.958] | `\val{s5.FD001.xgboost_cap150_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 200 | FD001 | 21.72 | [18.56, 24.85] | `\val{s5.FD001.xgboost_cap200_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 200 | FD001 | 0.575 | [0.047, 0.951] | `\val{s5.FD001.xgboost_cap200_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap none | FD001 | 24.91 | [20.99, 28.62] | `\val{s5.FD001.xgboost_capnone_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap none | FD001 | 0.579 | [0.054, 0.955] | `\val{s5.FD001.xgboost_capnone_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 125 | FD001 | 15.47 | [12.81, 18.15] | `\val{s5.FD001.lstm_cap125_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 125 | FD001 | 0.825 | [0.488, 0.963] | `\val{s5.FD001.lstm_cap125_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 100 | FD001 | 16.03 | [13.76, 18.04] | `\val{s5.FD001.lstm_cap100_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 100 | FD001 | 0.450 | [-0.189, 0.862] | `\val{s5.FD001.lstm_cap100_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 150 | FD001 | 19.92 | [17.12, 22.64] | `\val{s5.FD001.lstm_cap150_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 150 | FD001 | 0.693 | [0.258, 0.953] | `\val{s5.FD001.lstm_cap150_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 200 | FD001 | 26.88 | [22.99, 30.72] | `\val{s5.FD001.lstm_cap200_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 200 | FD001 | 0.808 | [0.467, 0.952] | `\val{s5.FD001.lstm_cap200_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap none | FD001 | 34.26 | [28.78, 39.49] | `\val{s5.FD001.lstm_capnone_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap none | FD001 | 0.669 | [0.142, 0.939] | `\val{s5.FD001.lstm_capnone_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 125 | FD002 | 12.70 | [11.39, 13.96] | `\val{s5.FD002.xgboost_cap125_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 125 | FD002 | 0.409 | [0.037, 0.711] | `\val{s5.FD002.xgboost_cap125_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 100 | FD002 | 17.04 | [15.83, 18.25] | `\val{s5.FD002.xgboost_cap100_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 100 | FD002 | 0.309 | [-0.086, 0.646] | `\val{s5.FD002.xgboost_cap100_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 150 | FD002 | 15.29 | [13.64, 16.99] | `\val{s5.FD002.xgboost_cap150_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 150 | FD002 | 0.465 | [0.090, 0.744] | `\val{s5.FD002.xgboost_cap150_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 200 | FD002 | 21.29 | [19.09, 23.50] | `\val{s5.FD002.xgboost_cap200_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 200 | FD002 | 0.359 | [-0.028, 0.676] | `\val{s5.FD002.xgboost_cap200_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap none | FD002 | 23.89 | [21.28, 26.42] | `\val{s5.FD002.xgboost_capnone_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap none | FD002 | 0.360 | [-0.046, 0.686] | `\val{s5.FD002.xgboost_capnone_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 125 | FD002 | 14.54 | [12.99, 16.13] | `\val{s5.FD002.lstm_cap125_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 125 | FD002 | 0.820 | [0.631, 0.934] | `\val{s5.FD002.lstm_cap125_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 100 | FD002 | 17.92 | [16.58, 19.26] | `\val{s5.FD002.lstm_cap100_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 100 | FD002 | 0.779 | [0.556, 0.938] | `\val{s5.FD002.lstm_cap100_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 150 | FD002 | 17.57 | [15.75, 19.45] | `\val{s5.FD002.lstm_cap150_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 150 | FD002 | 0.805 | [0.626, 0.923] | `\val{s5.FD002.lstm_cap150_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 200 | FD002 | 24.10 | [21.90, 26.44] | `\val{s5.FD002.lstm_cap200_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 200 | FD002 | 0.585 | [0.269, 0.816] | `\val{s5.FD002.lstm_cap200_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap none | FD002 | 28.71 | [25.97, 31.37] | `\val{s5.FD002.lstm_capnone_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap none | FD002 | 0.587 | [0.296, 0.810] | `\val{s5.FD002.lstm_capnone_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 125 | FD003 | 12.92 | [10.52, 15.08] | `\val{s5.FD003.xgboost_cap125_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 125 | FD003 | 0.214 | [-0.531, 0.743] | `\val{s5.FD003.xgboost_cap125_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 100 | FD003 | 15.21 | [13.27, 17.05] | `\val{s5.FD003.xgboost_cap100_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 100 | FD003 | 0.434 | [-0.160, 0.863] | `\val{s5.FD003.xgboost_cap100_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 150 | FD003 | 17.26 | [14.19, 20.27] | `\val{s5.FD003.xgboost_cap150_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 150 | FD003 | 0.212 | [-0.585, 0.718] | `\val{s5.FD003.xgboost_cap150_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 200 | FD003 | 26.06 | [21.46, 30.23] | `\val{s5.FD003.xgboost_cap200_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 200 | FD003 | 0.207 | [-0.509, 0.715] | `\val{s5.FD003.xgboost_cap200_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap none | FD003 | 36.25 | [29.11, 43.18] | `\val{s5.FD003.xgboost_capnone_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap none | FD003 | 0.511 | [-0.041, 0.875] | `\val{s5.FD003.xgboost_capnone_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 125 | FD003 | 15.13 | [12.74, 17.52] | `\val{s5.FD003.lstm_cap125_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 125 | FD003 | -0.010 | [-0.844, 0.579] | `\val{s5.FD003.lstm_cap125_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 100 | FD003 | 16.88 | [14.85, 18.83] | `\val{s5.FD003.lstm_cap100_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 100 | FD003 | 0.426 | [-0.191, 0.856] | `\val{s5.FD003.lstm_cap100_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 150 | FD003 | 21.46 | [18.58, 24.21] | `\val{s5.FD003.lstm_cap150_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 150 | FD003 | 0.090 | [-0.724, 0.653] | `\val{s5.FD003.lstm_cap150_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 200 | FD003 | 33.77 | [28.81, 38.28] | `\val{s5.FD003.lstm_cap200_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 200 | FD003 | 0.614 | [0.121, 0.939] | `\val{s5.FD003.lstm_cap200_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap none | FD003 | 45.01 | [38.02, 51.60] | `\val{s5.FD003.lstm_capnone_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap none | FD003 | 0.021 | [-0.842, 0.603] | `\val{s5.FD003.lstm_capnone_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 125 | FD004 | 14.07 | [12.30, 15.80] | `\val{s5.FD004.xgboost_cap125_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 125 | FD004 | 0.735 | [0.543, 0.883] | `\val{s5.FD004.xgboost_cap125_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 100 | FD004 | 18.46 | [17.01, 19.88] | `\val{s5.FD004.xgboost_cap100_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 100 | FD004 | 0.706 | [0.497, 0.866] | `\val{s5.FD004.xgboost_cap100_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 150 | FD004 | 18.50 | [16.69, 20.30] | `\val{s5.FD004.xgboost_cap150_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 150 | FD004 | 0.692 | [0.480, 0.853] | `\val{s5.FD004.xgboost_cap150_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap 200 | FD004 | 29.74 | [26.98, 32.47] | `\val{s5.FD004.xgboost_cap200_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap 200 | FD004 | 0.831 | [0.693, 0.914] | `\val{s5.FD004.xgboost_cap200_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, xgboost, training cap none | FD004 | 42.21 | [36.90, 47.30] | `\val{s5.FD004.xgboost_capnone_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, xgboost, training cap none | FD004 | 0.812 | [0.648, 0.924] | `\val{s5.FD004.xgboost_capnone_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 125 | FD004 | 15.29 | [13.68, 16.93] | `\val{s5.FD004.lstm_cap125_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 125 | FD004 | 0.817 | [0.675, 0.904] | `\val{s5.FD004.lstm_cap125_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 100 | FD004 | 19.07 | [17.80, 20.39] | `\val{s5.FD004.lstm_cap100_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 100 | FD004 | 0.869 | [0.768, 0.921] | `\val{s5.FD004.lstm_cap100_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 150 | FD004 | 19.80 | [18.04, 21.64] | `\val{s5.FD004.lstm_cap150_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 150 | FD004 | 0.827 | [0.685, 0.912] | `\val{s5.FD004.lstm_cap150_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap 200 | FD004 | 30.51 | [28.10, 33.02] | `\val{s5.FD004.lstm_cap200_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap 200 | FD004 | 0.844 | [0.741, 0.900] | `\val{s5.FD004.lstm_cap200_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |
| S5 leaderboard RMSE, lstm, training cap none | FD004 | 46.34 | [41.54, 50.80] | `\val{s5.FD004.lstm_capnone_leaderboard_rmse}` | `reports/research/secondary/s5_caps.csv` | leaderboard_rmse, _lo, _hi | make research-secondary |
| S5 S, lstm, training cap none | FD004 | 0.758 | [0.564, 0.886] | `\val{s5.FD004.lstm_capnone_savings_share}` | `reports/research/secondary/s5_caps.csv` | savings_share, _lo, _hi | make research-secondary |

## Numbers computed by make paper-assets

| quantity | fleet | estimate | 95% interval | macro | source | column | command |
|---|---|---|---|---|---|---|---|
| F1 leaderboard RMSE of xgboost-00, repeat 0, predictions y | FD001 | 12.65 |  | `\val{f1.FD001.original_leaderboard_rmse}` | `paper_assets/data/f1_invariance.csv` | leaderboard_rmse | make paper-assets |
| F1 S, predictions y | FD001 | 0.712 |  | `\val{f1.FD001.original_savings_share}` | `paper_assets/data/f1_invariance.csv` | savings_share | make paper-assets |
| F1 outcomes (engine x setting x tuning) that differ from the original, R1 and R2 | FD001 | 0 and 0 of 12600 |  | `\val{f1.FD001.original_outcomes_differing}` | `paper_assets/data/f1_invariance.csv` | outcomes_differing_R1, outcomes_differing_R2, outcomes_compared | make paper-assets |
| F1 leaderboard RMSE of xgboost-00, repeat 0, predictions y + 15 | FD001 | 20.86 |  | `\val{f1.FD001.shifted_leaderboard_rmse}` | `paper_assets/data/f1_invariance.csv` | leaderboard_rmse | make paper-assets |
| F1 S, predictions y + 15 | FD001 | 0.712 |  | `\val{f1.FD001.shifted_savings_share}` | `paper_assets/data/f1_invariance.csv` | savings_share | make paper-assets |
| F1 outcomes (engine x setting x tuning) that differ from the original, R1 and R2 | FD001 | 0 and 0 of 12600 |  | `\val{f1.FD001.shifted_outcomes_differing}` | `paper_assets/data/f1_invariance.csv` | outcomes_differing_R1, outcomes_differing_R2, outcomes_compared | make paper-assets |
| F1 leaderboard RMSE of xgboost-00, repeat 0, predictions 1.25 y | FD001 | 26.73 |  | `\val{f1.FD001.scaled_leaderboard_rmse}` | `paper_assets/data/f1_invariance.csv` | leaderboard_rmse | make paper-assets |
| F1 S, predictions 1.25 y | FD001 | 0.712 |  | `\val{f1.FD001.scaled_savings_share}` | `paper_assets/data/f1_invariance.csv` | savings_share | make paper-assets |
| F1 outcomes (engine x setting x tuning) that differ from the original, R1 and R2 | FD001 | 0 and 0 of 12600 |  | `\val{f1.FD001.scaled_outcomes_differing}` | `paper_assets/data/f1_invariance.csv` | outcomes_differing_R1, outcomes_differing_R2, outcomes_compared | make paper-assets |

## Conventional leaderboard (reference configurations, legacy split)

| quantity | fleet | estimate | 95% interval | macro | source | column | command |
|---|---|---|---|---|---|---|---|
| reference cnn: RMSE, mean of 5 seeds | FD001 | 14.16 |  | `\val{leaderboard.FD001.cnn_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD001 | 0.55 |  | `\val{leaderboard.FD001.cnn_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD001 | 390 |  | `\val{leaderboard.FD001.cnn_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference forest: RMSE, mean of 5 seeds | FD001 | 12.80 |  | `\val{leaderboard.FD001.forest_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD001 | 0.13 |  | `\val{leaderboard.FD001.forest_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD001 | 237 |  | `\val{leaderboard.FD001.forest_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference knn: RMSE, mean of 5 seeds | FD001 | 16.40 |  | `\val{leaderboard.FD001.knn_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD001 | 0.00 |  | `\val{leaderboard.FD001.knn_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD001 | 583 |  | `\val{leaderboard.FD001.knn_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference lstm: RMSE, mean of 5 seeds | FD001 | 14.52 |  | `\val{leaderboard.FD001.lstm_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD001 | 0.62 |  | `\val{leaderboard.FD001.lstm_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD001 | 413 |  | `\val{leaderboard.FD001.lstm_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference mlp: RMSE, mean of 5 seeds | FD001 | 14.42 |  | `\val{leaderboard.FD001.mlp_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD001 | 0.34 |  | `\val{leaderboard.FD001.mlp_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD001 | 368 |  | `\val{leaderboard.FD001.mlp_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference ridge: RMSE, mean of 5 seeds | FD001 | 14.99 |  | `\val{leaderboard.FD001.ridge_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD001 | 0.00 |  | `\val{leaderboard.FD001.ridge_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD001 | 347 |  | `\val{leaderboard.FD001.ridge_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference tcn: RMSE, mean of 5 seeds | FD001 | 14.69 |  | `\val{leaderboard.FD001.tcn_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD001 | 0.43 |  | `\val{leaderboard.FD001.tcn_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD001 | 395 |  | `\val{leaderboard.FD001.tcn_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference transformer: RMSE, mean of 5 seeds | FD001 | 14.34 |  | `\val{leaderboard.FD001.transformer_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD001 | 0.51 |  | `\val{leaderboard.FD001.transformer_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD001 | 374 |  | `\val{leaderboard.FD001.transformer_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference xgboost: RMSE, mean of 5 seeds | FD001 | 11.96 |  | `\val{leaderboard.FD001.xgboost_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD001 | 0.14 |  | `\val{leaderboard.FD001.xgboost_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD001 | 208 |  | `\val{leaderboard.FD001.xgboost_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference xgboost_raw: RMSE, mean of 5 seeds | FD001 | 16.84 |  | `\val{leaderboard.FD001.xgboost_raw_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD001 | 0.47 |  | `\val{leaderboard.FD001.xgboost_raw_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD001 | 475 |  | `\val{leaderboard.FD001.xgboost_raw_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference cnn: RMSE, mean of 5 seeds | FD002 | 13.55 |  | `\val{leaderboard.FD002.cnn_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD002 | 0.32 |  | `\val{leaderboard.FD002.cnn_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD002 | 903 |  | `\val{leaderboard.FD002.cnn_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference forest: RMSE, mean of 5 seeds | FD002 | 13.27 |  | `\val{leaderboard.FD002.forest_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD002 | 0.08 |  | `\val{leaderboard.FD002.forest_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD002 | 865 |  | `\val{leaderboard.FD002.forest_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference knn: RMSE, mean of 5 seeds | FD002 | 16.30 |  | `\val{leaderboard.FD002.knn_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD002 | 0.00 |  | `\val{leaderboard.FD002.knn_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD002 | 1439 |  | `\val{leaderboard.FD002.knn_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference lstm: RMSE, mean of 5 seeds | FD002 | 14.68 |  | `\val{leaderboard.FD002.lstm_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD002 | 0.32 |  | `\val{leaderboard.FD002.lstm_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD002 | 1108 |  | `\val{leaderboard.FD002.lstm_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference mlp: RMSE, mean of 5 seeds | FD002 | 13.74 |  | `\val{leaderboard.FD002.mlp_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD002 | 0.37 |  | `\val{leaderboard.FD002.mlp_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD002 | 847 |  | `\val{leaderboard.FD002.mlp_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference ridge: RMSE, mean of 5 seeds | FD002 | 15.96 |  | `\val{leaderboard.FD002.ridge_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD002 | 0.00 |  | `\val{leaderboard.FD002.ridge_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD002 | 1178 |  | `\val{leaderboard.FD002.ridge_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference tcn: RMSE, mean of 5 seeds | FD002 | 14.35 |  | `\val{leaderboard.FD002.tcn_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD002 | 0.30 |  | `\val{leaderboard.FD002.tcn_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD002 | 1059 |  | `\val{leaderboard.FD002.tcn_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference transformer: RMSE, mean of 5 seeds | FD002 | 14.26 |  | `\val{leaderboard.FD002.transformer_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD002 | 0.19 |  | `\val{leaderboard.FD002.transformer_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD002 | 956 |  | `\val{leaderboard.FD002.transformer_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference xgboost: RMSE, mean of 5 seeds | FD002 | 12.67 |  | `\val{leaderboard.FD002.xgboost_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD002 | 0.07 |  | `\val{leaderboard.FD002.xgboost_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD002 | 690 |  | `\val{leaderboard.FD002.xgboost_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference xgboost_raw: RMSE, mean of 5 seeds | FD002 | 16.41 |  | `\val{leaderboard.FD002.xgboost_raw_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD002 | 0.03 |  | `\val{leaderboard.FD002.xgboost_raw_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD002 | 1542 |  | `\val{leaderboard.FD002.xgboost_raw_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference cnn: RMSE, mean of 5 seeds | FD003 | 13.29 |  | `\val{leaderboard.FD003.cnn_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD003 | 0.65 |  | `\val{leaderboard.FD003.cnn_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD003 | 358 |  | `\val{leaderboard.FD003.cnn_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference forest: RMSE, mean of 5 seeds | FD003 | 13.69 |  | `\val{leaderboard.FD003.forest_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD003 | 0.12 |  | `\val{leaderboard.FD003.forest_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD003 | 381 |  | `\val{leaderboard.FD003.forest_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference knn: RMSE, mean of 5 seeds | FD003 | 18.73 |  | `\val{leaderboard.FD003.knn_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD003 | 0.00 |  | `\val{leaderboard.FD003.knn_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD003 | 1305 |  | `\val{leaderboard.FD003.knn_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference lstm: RMSE, mean of 5 seeds | FD003 | 13.66 |  | `\val{leaderboard.FD003.lstm_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD003 | 0.77 |  | `\val{leaderboard.FD003.lstm_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD003 | 436 |  | `\val{leaderboard.FD003.lstm_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference mlp: RMSE, mean of 5 seeds | FD003 | 12.54 |  | `\val{leaderboard.FD003.mlp_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD003 | 0.59 |  | `\val{leaderboard.FD003.mlp_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD003 | 284 |  | `\val{leaderboard.FD003.mlp_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference ridge: RMSE, mean of 5 seeds | FD003 | 16.32 |  | `\val{leaderboard.FD003.ridge_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD003 | 0.00 |  | `\val{leaderboard.FD003.ridge_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD003 | 504 |  | `\val{leaderboard.FD003.ridge_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference tcn: RMSE, mean of 5 seeds | FD003 | 14.42 |  | `\val{leaderboard.FD003.tcn_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD003 | 0.49 |  | `\val{leaderboard.FD003.tcn_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD003 | 553 |  | `\val{leaderboard.FD003.tcn_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference transformer: RMSE, mean of 5 seeds | FD003 | 14.43 |  | `\val{leaderboard.FD003.transformer_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD003 | 0.62 |  | `\val{leaderboard.FD003.transformer_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD003 | 459 |  | `\val{leaderboard.FD003.transformer_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference xgboost: RMSE, mean of 5 seeds | FD003 | 12.46 |  | `\val{leaderboard.FD003.xgboost_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD003 | 0.09 |  | `\val{leaderboard.FD003.xgboost_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD003 | 265 |  | `\val{leaderboard.FD003.xgboost_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference xgboost_raw: RMSE, mean of 5 seeds | FD003 | 16.74 |  | `\val{leaderboard.FD003.xgboost_raw_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD003 | 0.04 |  | `\val{leaderboard.FD003.xgboost_raw_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD003 | 717 |  | `\val{leaderboard.FD003.xgboost_raw_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference cnn: RMSE, mean of 5 seeds | FD004 | 14.31 |  | `\val{leaderboard.FD004.cnn_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD004 | 0.23 |  | `\val{leaderboard.FD004.cnn_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD004 | 1215 |  | `\val{leaderboard.FD004.cnn_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference forest: RMSE, mean of 5 seeds | FD004 | 14.38 |  | `\val{leaderboard.FD004.forest_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD004 | 0.07 |  | `\val{leaderboard.FD004.forest_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD004 | 957 |  | `\val{leaderboard.FD004.forest_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference knn: RMSE, mean of 5 seeds | FD004 | 18.42 |  | `\val{leaderboard.FD004.knn_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD004 | 0.00 |  | `\val{leaderboard.FD004.knn_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD004 | 2708 |  | `\val{leaderboard.FD004.knn_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference lstm: RMSE, mean of 5 seeds | FD004 | 15.25 |  | `\val{leaderboard.FD004.lstm_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD004 | 0.32 |  | `\val{leaderboard.FD004.lstm_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD004 | 1408 |  | `\val{leaderboard.FD004.lstm_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference mlp: RMSE, mean of 5 seeds | FD004 | 14.03 |  | `\val{leaderboard.FD004.mlp_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD004 | 0.25 |  | `\val{leaderboard.FD004.mlp_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD004 | 1104 |  | `\val{leaderboard.FD004.mlp_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference ridge: RMSE, mean of 5 seeds | FD004 | 18.12 |  | `\val{leaderboard.FD004.ridge_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD004 | 0.00 |  | `\val{leaderboard.FD004.ridge_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD004 | 1543 |  | `\val{leaderboard.FD004.ridge_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference tcn: RMSE, mean of 5 seeds | FD004 | 16.15 |  | `\val{leaderboard.FD004.tcn_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD004 | 0.23 |  | `\val{leaderboard.FD004.tcn_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD004 | 1703 |  | `\val{leaderboard.FD004.tcn_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference transformer: RMSE, mean of 5 seeds | FD004 | 14.90 |  | `\val{leaderboard.FD004.transformer_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD004 | 0.45 |  | `\val{leaderboard.FD004.transformer_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD004 | 1331 |  | `\val{leaderboard.FD004.transformer_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference xgboost: RMSE, mean of 5 seeds | FD004 | 13.36 |  | `\val{leaderboard.FD004.xgboost_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD004 | 0.11 |  | `\val{leaderboard.FD004.xgboost_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD004 | 808 |  | `\val{leaderboard.FD004.xgboost_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| reference xgboost_raw: RMSE, mean of 5 seeds | FD004 | 17.38 |  | `\val{leaderboard.FD004.xgboost_raw_rmse}` | `reports/research/leaderboard_reference.csv` | rmse_mean | make research-leaderboard |
| standard deviation over seeds | FD004 | 0.14 |  | `\val{leaderboard.FD004.xgboost_raw_rmse_sd}` | `reports/research/leaderboard_reference.csv` | rmse_std | make research-leaderboard |
| NASA score, mean of 5 seeds | FD004 | 1791 |  | `\val{leaderboard.FD004.xgboost_raw_nasa}` | `reports/research/leaderboard_reference.csv` | nasa_score_mean | make research-leaderboard |
| legacy RMSE, lstm | FD001 | 14.52 |  | `\val{leaderboard.FD001.lstm_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, ridge | FD001 | 14.99 |  | `\val{leaderboard.FD001.ridge_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, tcn | FD001 | 14.69 |  | `\val{leaderboard.FD001.tcn_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, xgboost | FD001 | 12.14 |  | `\val{leaderboard.FD001.xgboost_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, lstm | FD002 | 14.68 |  | `\val{leaderboard.FD002.lstm_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, ridge | FD002 | 15.96 |  | `\val{leaderboard.FD002.ridge_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, tcn | FD002 | 14.35 |  | `\val{leaderboard.FD002.tcn_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, xgboost | FD002 | 12.60 |  | `\val{leaderboard.FD002.xgboost_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, lstm | FD003 | 13.66 |  | `\val{leaderboard.FD003.lstm_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, ridge | FD003 | 16.29 |  | `\val{leaderboard.FD003.ridge_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, tcn | FD003 | 14.42 |  | `\val{leaderboard.FD003.tcn_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, xgboost | FD003 | 12.49 |  | `\val{leaderboard.FD003.xgboost_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, lstm | FD004 | 15.25 |  | `\val{leaderboard.FD004.lstm_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, ridge | FD004 | 18.53 |  | `\val{leaderboard.FD004.ridge_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, tcn | FD004 | 16.15 |  | `\val{leaderboard.FD004.tcn_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |
| legacy RMSE, xgboost | FD004 | 13.49 |  | `\val{leaderboard.FD004.xgboost_rmse_legacy}` | `reports/research/leaderboard_vs_legacy.csv` | rmse_legacy | make research-leaderboard |

## Data, pool and compute

| quantity | fleet | estimate | 95% interval | macro | source | column | command |
|---|---|---|---|---|---|---|---|
| train_units | FD001 | 100 |  | `\val{data.FD001.train_units}` | `paper_assets/data/data_folds.csv` | train_units | make paper-assets |
| test_units | FD001 | 100 |  | `\val{data.FD001.test_units}` | `paper_assets/data/data_folds.csv` | test_units | make paper-assets |
| regimes | FD001 | 1 |  | `\val{data.FD001.regimes}` | `paper_assets/data/data_folds.csv` | regimes | make paper-assets |
| life_min | FD001 | 128 |  | `\val{data.FD001.life_min}` | `paper_assets/data/data_folds.csv` | life_min | make paper-assets |
| life_max | FD001 | 362 |  | `\val{data.FD001.life_max}` | `paper_assets/data/data_folds.csv` | life_max | make paper-assets |
| cap | FD001 | 125 |  | `\val{data.FD001.cap}` | `paper_assets/data/data_folds.csv` | cap | make paper-assets |
| flights | FD001 | 0 |  | `\val{data.FD001.flights}` | `paper_assets/data/data_folds.csv` | flights | make paper-assets |
| repeats | FD001 | 3 |  | `\val{data.FD001.repeats}` | `paper_assets/data/data_folds.csv` | repeats | make paper-assets |
| median life | FD001 | 199.0 |  | `\val{data.FD001.life_median}` | `paper_assets/data/data_folds.csv` | life_median | make paper-assets |
| outer_test engines per fold | FD001 | 20 |  | `\val{data.FD001.outer_test}` | `paper_assets/data/data_folds.csv` | outer_test | make paper-assets |
| inner_val engines per fold | FD001 | 20 |  | `\val{data.FD001.inner_val}` | `paper_assets/data/data_folds.csv` | inner_val | make paper-assets |
| inner_fit engines per fold | FD001 | 60 |  | `\val{data.FD001.inner_fit}` | `paper_assets/data/data_folds.csv` | inner_fit | make paper-assets |
| train_units | FD002 | 260 |  | `\val{data.FD002.train_units}` | `paper_assets/data/data_folds.csv` | train_units | make paper-assets |
| test_units | FD002 | 259 |  | `\val{data.FD002.test_units}` | `paper_assets/data/data_folds.csv` | test_units | make paper-assets |
| regimes | FD002 | 6 |  | `\val{data.FD002.regimes}` | `paper_assets/data/data_folds.csv` | regimes | make paper-assets |
| life_min | FD002 | 128 |  | `\val{data.FD002.life_min}` | `paper_assets/data/data_folds.csv` | life_min | make paper-assets |
| life_max | FD002 | 378 |  | `\val{data.FD002.life_max}` | `paper_assets/data/data_folds.csv` | life_max | make paper-assets |
| cap | FD002 | 125 |  | `\val{data.FD002.cap}` | `paper_assets/data/data_folds.csv` | cap | make paper-assets |
| flights | FD002 | 0 |  | `\val{data.FD002.flights}` | `paper_assets/data/data_folds.csv` | flights | make paper-assets |
| repeats | FD002 | 3 |  | `\val{data.FD002.repeats}` | `paper_assets/data/data_folds.csv` | repeats | make paper-assets |
| median life | FD002 | 199.0 |  | `\val{data.FD002.life_median}` | `paper_assets/data/data_folds.csv` | life_median | make paper-assets |
| outer_test engines per fold | FD002 | 52 |  | `\val{data.FD002.outer_test}` | `paper_assets/data/data_folds.csv` | outer_test | make paper-assets |
| inner_val engines per fold | FD002 | 52 |  | `\val{data.FD002.inner_val}` | `paper_assets/data/data_folds.csv` | inner_val | make paper-assets |
| inner_fit engines per fold | FD002 | 156 |  | `\val{data.FD002.inner_fit}` | `paper_assets/data/data_folds.csv` | inner_fit | make paper-assets |
| train_units | FD003 | 100 |  | `\val{data.FD003.train_units}` | `paper_assets/data/data_folds.csv` | train_units | make paper-assets |
| test_units | FD003 | 100 |  | `\val{data.FD003.test_units}` | `paper_assets/data/data_folds.csv` | test_units | make paper-assets |
| regimes | FD003 | 1 |  | `\val{data.FD003.regimes}` | `paper_assets/data/data_folds.csv` | regimes | make paper-assets |
| life_min | FD003 | 145 |  | `\val{data.FD003.life_min}` | `paper_assets/data/data_folds.csv` | life_min | make paper-assets |
| life_max | FD003 | 525 |  | `\val{data.FD003.life_max}` | `paper_assets/data/data_folds.csv` | life_max | make paper-assets |
| cap | FD003 | 125 |  | `\val{data.FD003.cap}` | `paper_assets/data/data_folds.csv` | cap | make paper-assets |
| flights | FD003 | 0 |  | `\val{data.FD003.flights}` | `paper_assets/data/data_folds.csv` | flights | make paper-assets |
| repeats | FD003 | 3 |  | `\val{data.FD003.repeats}` | `paper_assets/data/data_folds.csv` | repeats | make paper-assets |
| median life | FD003 | 220.5 |  | `\val{data.FD003.life_median}` | `paper_assets/data/data_folds.csv` | life_median | make paper-assets |
| outer_test engines per fold | FD003 | 20 |  | `\val{data.FD003.outer_test}` | `paper_assets/data/data_folds.csv` | outer_test | make paper-assets |
| inner_val engines per fold | FD003 | 20 |  | `\val{data.FD003.inner_val}` | `paper_assets/data/data_folds.csv` | inner_val | make paper-assets |
| inner_fit engines per fold | FD003 | 60 |  | `\val{data.FD003.inner_fit}` | `paper_assets/data/data_folds.csv` | inner_fit | make paper-assets |
| train_units | FD004 | 249 |  | `\val{data.FD004.train_units}` | `paper_assets/data/data_folds.csv` | train_units | make paper-assets |
| test_units | FD004 | 248 |  | `\val{data.FD004.test_units}` | `paper_assets/data/data_folds.csv` | test_units | make paper-assets |
| regimes | FD004 | 6 |  | `\val{data.FD004.regimes}` | `paper_assets/data/data_folds.csv` | regimes | make paper-assets |
| life_min | FD004 | 128 |  | `\val{data.FD004.life_min}` | `paper_assets/data/data_folds.csv` | life_min | make paper-assets |
| life_max | FD004 | 543 |  | `\val{data.FD004.life_max}` | `paper_assets/data/data_folds.csv` | life_max | make paper-assets |
| cap | FD004 | 125 |  | `\val{data.FD004.cap}` | `paper_assets/data/data_folds.csv` | cap | make paper-assets |
| flights | FD004 | 0 |  | `\val{data.FD004.flights}` | `paper_assets/data/data_folds.csv` | flights | make paper-assets |
| repeats | FD004 | 3 |  | `\val{data.FD004.repeats}` | `paper_assets/data/data_folds.csv` | repeats | make paper-assets |
| median life | FD004 | 234.0 |  | `\val{data.FD004.life_median}` | `paper_assets/data/data_folds.csv` | life_median | make paper-assets |
| outer_test engines per fold | FD004 | 49--50 |  | `\val{data.FD004.outer_test}` | `paper_assets/data/data_folds.csv` | outer_test | make paper-assets |
| inner_val engines per fold | FD004 | 49--50 |  | `\val{data.FD004.inner_val}` | `paper_assets/data/data_folds.csv` | inner_val | make paper-assets |
| inner_fit engines per fold | FD004 | 149--150 |  | `\val{data.FD004.inner_fit}` | `paper_assets/data/data_folds.csv` | inner_fit | make paper-assets |
| train_units | NCMAPSS | 99 |  | `\val{data.NCMAPSS.train_units}` | `paper_assets/data/data_folds.csv` | train_units | make paper-assets |
| test_units | NCMAPSS | 0 |  | `\val{data.NCMAPSS.test_units}` | `paper_assets/data/data_folds.csv` | test_units | make paper-assets |
| regimes | NCMAPSS | 3 |  | `\val{data.NCMAPSS.regimes}` | `paper_assets/data/data_folds.csv` | regimes | make paper-assets |
| life_min | NCMAPSS | 48 |  | `\val{data.NCMAPSS.life_min}` | `paper_assets/data/data_folds.csv` | life_min | make paper-assets |
| life_max | NCMAPSS | 100 |  | `\val{data.NCMAPSS.life_max}` | `paper_assets/data/data_folds.csv` | life_max | make paper-assets |
| cap | NCMAPSS | 50 |  | `\val{data.NCMAPSS.cap}` | `paper_assets/data/data_folds.csv` | cap | make paper-assets |
| flights | NCMAPSS | 7473 |  | `\val{data.NCMAPSS.flights}` | `paper_assets/data/data_folds.csv` | flights | make paper-assets |
| repeats | NCMAPSS | 3 |  | `\val{data.NCMAPSS.repeats}` | `paper_assets/data/data_folds.csv` | repeats | make paper-assets |
| median life | NCMAPSS | 75.0 |  | `\val{data.NCMAPSS.life_median}` | `paper_assets/data/data_folds.csv` | life_median | make paper-assets |
| outer_test engines per fold | NCMAPSS | 19--20 |  | `\val{data.NCMAPSS.outer_test}` | `paper_assets/data/data_folds.csv` | outer_test | make paper-assets |
| inner_val engines per fold | NCMAPSS | 19--20 |  | `\val{data.NCMAPSS.inner_val}` | `paper_assets/data/data_folds.csv` | inner_val | make paper-assets |
| inner_fit engines per fold | NCMAPSS | 59--60 |  | `\val{data.NCMAPSS.inner_fit}` | `paper_assets/data/data_folds.csv` | inner_fit | make paper-assets |
| configurations in the pool | all | 113 |  | `\val{data.all.configs}` | `configs/research_pool.yaml` | total | make research-pool |
| families in the pool | all | 10 |  | `\val{data.all.families}` | `configs/research_pool.yaml` | families | make research-pool |
| fits in the confirmatory archive | all | 6780 |  | `\val{compute.all.confirmatory_fits}` | `paper_assets/data/compute.csv` | fits (archive = confirmatory) | make paper-assets |
| failed fits in the confirmatory archive | all | 0 |  | `\val{compute.all.confirmatory_failed}` | `paper_assets/data/compute.csv` | failed (archive = confirmatory) | make paper-assets |
| summed fit hours, confirmatory, cpu stream | all | 5.59 |  | `\val{compute.all.confirmatory_cpu_hours}` | `paper_assets/data/compute.csv` | hours (archive = confirmatory, stream = cpu) | make paper-assets |
| summed fit hours, confirmatory, gpu stream | all | 22.41 |  | `\val{compute.all.confirmatory_gpu_hours}` | `paper_assets/data/compute.csv` | hours (archive = confirmatory, stream = gpu) | make paper-assets |
| fits in the S4 nested archive | all | 400 |  | `\val{compute.all.s4_nested_fits}` | `paper_assets/data/compute.csv` | fits (archive = S4 nested) | make paper-assets |
| failed fits in the S4 nested archive | all | 0 |  | `\val{compute.all.s4_nested_failed}` | `paper_assets/data/compute.csv` | failed (archive = S4 nested) | make paper-assets |
| summed fit hours, S4 nested, cpu stream | all | 0.18 |  | `\val{compute.all.s4_nested_cpu_hours}` | `paper_assets/data/compute.csv` | hours (archive = S4 nested, stream = cpu) | make paper-assets |
| summed fit hours, S4 nested, gpu stream | all | 0.77 |  | `\val{compute.all.s4_nested_gpu_hours}` | `paper_assets/data/compute.csv` | hours (archive = S4 nested, stream = gpu) | make paper-assets |
| fits in the S5 caps archive | all | 160 |  | `\val{compute.all.s5_caps_fits}` | `paper_assets/data/compute.csv` | fits (archive = S5 caps) | make paper-assets |
| failed fits in the S5 caps archive | all | 0 |  | `\val{compute.all.s5_caps_failed}` | `paper_assets/data/compute.csv` | failed (archive = S5 caps) | make paper-assets |
| summed fit hours, S5 caps, gpu stream | all | 0.30 |  | `\val{compute.all.s5_caps_gpu_hours}` | `paper_assets/data/compute.csv` | hours (archive = S5 caps, stream = gpu) | make paper-assets |
| fits in the N-CMAPSS archive | all | 1695 |  | `\val{compute.all.ncmapss_fits}` | `paper_assets/data/compute.csv` | fits (archive = N-CMAPSS) | make paper-assets |
| failed fits in the N-CMAPSS archive | all | 0 |  | `\val{compute.all.ncmapss_failed}` | `paper_assets/data/compute.csv` | failed (archive = N-CMAPSS) | make paper-assets |
| summed fit hours, N-CMAPSS, cpu stream | all | 0.32 |  | `\val{compute.all.ncmapss_cpu_hours}` | `paper_assets/data/compute.csv` | hours (archive = N-CMAPSS, stream = cpu) | make paper-assets |
| summed fit hours, N-CMAPSS, gpu stream | all | 1.95 |  | `\val{compute.all.ncmapss_gpu_hours}` | `paper_assets/data/compute.csv` | hours (archive = N-CMAPSS, stream = gpu) | make paper-assets |

## Exploratory (not pre-registered, uncorrected)

| quantity | fleet | estimate | 95% interval | macro | source | column | command |
|---|---|---|---|---|---|---|---|
| X1 tau-b(late_rmse, D) | FD001 | 0.387 | [-0.121, 0.490] | `\val{x1.FD001.late_rmse_tau_b}` | `reports/research/exploratory/x1_isotonic_late_rmse.csv` | tau_b, _lo, _hi (proxy = late_rmse) | make research-exploratory |
| X1 tau-b(late_rmse_isotonic, D) | FD001 | 0.384 | [-0.135, 0.480] | `\val{x1.FD001.late_rmse_isotonic_tau_b}` | `reports/research/exploratory/x1_isotonic_late_rmse.csv` | tau_b, _lo, _hi (proxy = late_rmse_isotonic) | make research-exploratory |
| X1 tau-b(late_rmse, D) | FD002 | 0.136 | [-0.206, 0.417] | `\val{x1.FD002.late_rmse_tau_b}` | `reports/research/exploratory/x1_isotonic_late_rmse.csv` | tau_b, _lo, _hi (proxy = late_rmse) | make research-exploratory |
| X1 tau-b(late_rmse_isotonic, D) | FD002 | 0.080 | [-0.276, 0.389] | `\val{x1.FD002.late_rmse_isotonic_tau_b}` | `reports/research/exploratory/x1_isotonic_late_rmse.csv` | tau_b, _lo, _hi (proxy = late_rmse_isotonic) | make research-exploratory |
| X1 tau-b(late_rmse, D) | FD003 | -0.002 | [-0.260, 0.284] | `\val{x1.FD003.late_rmse_tau_b}` | `reports/research/exploratory/x1_isotonic_late_rmse.csv` | tau_b, _lo, _hi (proxy = late_rmse) | make research-exploratory |
| X1 tau-b(late_rmse_isotonic, D) | FD003 | -0.057 | [-0.281, 0.263] | `\val{x1.FD003.late_rmse_isotonic_tau_b}` | `reports/research/exploratory/x1_isotonic_late_rmse.csv` | tau_b, _lo, _hi (proxy = late_rmse_isotonic) | make research-exploratory |
| X1 tau-b(late_rmse, D) | FD004 | -0.086 | [-0.303, 0.196] | `\val{x1.FD004.late_rmse_tau_b}` | `reports/research/exploratory/x1_isotonic_late_rmse.csv` | tau_b, _lo, _hi (proxy = late_rmse) | make research-exploratory |
| X1 tau-b(late_rmse_isotonic, D) | FD004 | -0.091 | [-0.313, 0.207] | `\val{x1.FD004.late_rmse_isotonic_tau_b}` | `reports/research/exploratory/x1_isotonic_late_rmse.csv` | tau_b, _lo, _hi (proxy = late_rmse_isotonic) | make research-exploratory |
| X3 tau_legacy_booking_cost_vs_leaderboard_rmse | FD001 | 0.676 | [0.501, 0.745] | `\val{x3.FD001.tau_legacy_booking_cost_vs_leaderboard_rmse}` | `reports/research/exploratory/x3_legacy_booking.csv` | tau_legacy_booking_cost_vs_leaderboard_rmse, _lo, _hi | make research-exploratory |
| X3 tau_legacy_booking_cost_vs_decision_loss | FD001 | 0.269 | [-0.157, 0.387] | `\val{x3.FD001.tau_legacy_booking_cost_vs_decision_loss}` | `reports/research/exploratory/x3_legacy_booking.csv` | tau_legacy_booking_cost_vs_decision_loss, _lo, _hi | make research-exploratory |
| X3 tau_legacy_booking_cost_vs_leaderboard_rmse | FD002 | 0.781 | [0.558, 0.803] | `\val{x3.FD002.tau_legacy_booking_cost_vs_leaderboard_rmse}` | `reports/research/exploratory/x3_legacy_booking.csv` | tau_legacy_booking_cost_vs_leaderboard_rmse, _lo, _hi | make research-exploratory |
| X3 tau_legacy_booking_cost_vs_decision_loss | FD002 | 0.063 | [-0.271, 0.345] | `\val{x3.FD002.tau_legacy_booking_cost_vs_decision_loss}` | `reports/research/exploratory/x3_legacy_booking.csv` | tau_legacy_booking_cost_vs_decision_loss, _lo, _hi | make research-exploratory |
| X3 tau_legacy_booking_cost_vs_leaderboard_rmse | FD003 | 0.724 | [0.498, 0.788] | `\val{x3.FD003.tau_legacy_booking_cost_vs_leaderboard_rmse}` | `reports/research/exploratory/x3_legacy_booking.csv` | tau_legacy_booking_cost_vs_leaderboard_rmse, _lo, _hi | make research-exploratory |
| X3 tau_legacy_booking_cost_vs_decision_loss | FD003 | -0.074 | [-0.309, 0.247] | `\val{x3.FD003.tau_legacy_booking_cost_vs_decision_loss}` | `reports/research/exploratory/x3_legacy_booking.csv` | tau_legacy_booking_cost_vs_decision_loss, _lo, _hi | make research-exploratory |
| X3 tau_legacy_booking_cost_vs_leaderboard_rmse | FD004 | 0.772 | [0.644, 0.834] | `\val{x3.FD004.tau_legacy_booking_cost_vs_leaderboard_rmse}` | `reports/research/exploratory/x3_legacy_booking.csv` | tau_legacy_booking_cost_vs_leaderboard_rmse, _lo, _hi | make research-exploratory |
| X3 tau_legacy_booking_cost_vs_decision_loss | FD004 | -0.034 | [-0.267, 0.222] | `\val{x3.FD004.tau_legacy_booking_cost_vs_decision_loss}` | `reports/research/exploratory/x3_legacy_booking.csv` | tau_legacy_booking_cost_vs_decision_loss, _lo, _hi | make research-exploratory |
