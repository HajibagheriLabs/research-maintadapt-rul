## Test RMSE (mean ± std over seeds)

| model | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Ridge, rolling features | 14.99 ± 0.00 | 15.96 ± 0.00 | 16.29 ± 0.00 | 18.53 ± 0.00 |
| XGBoost, rolling features | 12.14 ± 0.09 | 12.60 ± 0.05 | 12.49 ± 0.18 | 13.49 ± 0.10 |
| TCN, raw windows | 14.69 ± 0.43 | 14.35 ± 0.30 | 14.42 ± 0.48 | 16.15 ± 0.23 |
| LSTM, raw windows | 14.52 ± 0.62 | 14.68 ± 0.32 | 13.66 ± 0.77 | 15.25 ± 0.32 |

## Test NASA score (mean ± std over seeds; lower is better)

| model | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Ridge, rolling features | 347 ± 0 | 1179 ± 0 | 508 ± 0 | 1596 ± 0 |
| XGBoost, rolling features | 214 ± 5 | 690 ± 12 | 262 ± 10 | 819 ± 21 |
| TCN, raw windows | 395 ± 65 | 1059 ± 130 | 553 ± 154 | 1703 ± 213 |
| LSTM, raw windows | 413 ± 52 | 1108 ± 194 | 436 ± 118 | 1408 ± 310 |

## XGBoost normalisation ablation

| subset | normalisation | rmse | nasa_score |
|---|---|---|---|
| FD002 | condition-aware | 12.60 ± 0.05 | 690 ± 12 |
| FD002 | global | 15.91 ± 0.13 | 1257 ± 31 |
| FD004 | condition-aware | 13.49 ± 0.10 | 819 ± 21 |
| FD004 | global | 17.39 ± 0.09 | 1713 ± 58 |

## Deep models against XGBoost (Welch t-test on per-seed test RMSE)

| subset | model | rmse_gap_vs_xgboost | welch_p_value | verdict |
|---|---|---|---|---|
| FD001 | TCN | 2.55 | 0.0001 | XGBoost better |
| FD001 | LSTM | 2.38 | 0.0009 | XGBoost better |
| FD002 | TCN | 1.75 | 0.0001 | XGBoost better |
| FD002 | LSTM | 2.08 | 0.0001 | XGBoost better |
| FD003 | TCN | 1.93 | 0.0004 | XGBoost better |
| FD003 | LSTM | 1.17 | 0.0254 | XGBoost better |
| FD004 | TCN | 2.65 | 0.0 | XGBoost better |
| FD004 | LSTM | 1.76 | 0.0001 | XGBoost better |

## Transfer: test RMSE on the target fleet

| target | target engines used | trained from scratch | fine-tuned from FD001 | gain from fine-tuning (Welch) | FD001 model, zero-shot |
|---|---|---|---|---|---|
| FD003 | none |  |  |  | 24.65 ± 1.54 |
| FD003 | 8 (10%) | 22.15 ± 0.58 | 20.37 ± 1.76 | +1.79 (p = 0.085) |  |
| FD003 | 20 (25%) | 22.06 ± 1.79 | 18.73 ± 2.33 | +3.33 (p = 0.037) |  |
| FD003 | 40 (50%) | 18.26 ± 2.20 | 17.14 ± 0.53 | +1.12 (p = 0.324) |  |
| FD003 | 80 (100%) | 14.88 ± 1.29 | 13.97 ± 0.66 | +0.90 (p = 0.212) |  |
| FD004 | none |  |  |  | 26.60 ± 1.88 |
| FD004 | 20 (10%) | 20.37 ± 1.07 | 19.09 ± 1.43 | +1.28 (p = 0.151) |  |
| FD004 | 50 (25%) | 18.27 ± 1.25 | 17.37 ± 0.55 | +0.90 (p = 0.195) |  |
| FD004 | 100 (50%) | 15.99 ± 0.54 | 15.57 ± 0.82 | +0.41 (p = 0.378) |  |
| FD004 | 199 (100%) | 15.14 ± 0.23 | 14.95 ± 0.39 | +0.20 (p = 0.366) |  |

## Coverage of the nominal 90% interval on test

| subset | interval | coverage | mean width (cycles) | coverage after conformal calibration | calibrated width (cycles) |
|---|---|---|---|---|---|
| FD001 | XGBoost quantile regression | 86% ± 1% | 66.8 ± 0.2 | 86% ± 1% | 66.8 ± 0.2 |
| FD001 | TCN MC dropout | 58% ± 4% | 22.4 ± 1.1 | 91% ± 1% | 44.7 ± 0.7 |
| FD001 | LSTM MC dropout | 49% ± 5% | 15.2 ± 0.9 | 86% ± 2% | 40.7 ± 1.6 |
| FD002 | XGBoost quantile regression | 93% ± 1% | 70.6 ± 0.3 | 93% ± 1% | 70.7 ± 0.3 |
| FD002 | TCN MC dropout | 61% ± 2% | 20.6 ± 0.6 | 94% ± 1% | 54.0 ± 0.8 |
| FD002 | LSTM MC dropout | 44% ± 3% | 13.1 ± 1.2 | 93% ± 1% | 53.4 ± 0.6 |
| FD003 | XGBoost quantile regression | 83% ± 2% | 61.9 ± 0.2 | 83% ± 2% | 61.9 ± 0.2 |
| FD003 | TCN MC dropout | 64% ± 4% | 19.5 ± 0.9 | 81% ± 3% | 31.8 ± 1.8 |
| FD003 | LSTM MC dropout | 46% ± 4% | 13.0 ± 1.0 | 87% ± 2% | 35.6 ± 0.5 |
| FD004 | XGBoost quantile regression | 86% ± 0% | 62.0 ± 0.2 | 86% ± 0% | 62.0 ± 0.2 |
| FD004 | TCN MC dropout | 55% ± 2% | 19.3 ± 0.8 | 85% ± 1% | 43.1 ± 0.4 |
| FD004 | LSTM MC dropout | 45% ± 1% | 13.2 ± 1.6 | 87% ± 1% | 44.4 ± 0.9 |

## Maintenance policy at a failure cost of 100 cycles

| subset | policy | threshold or age | expected cost per engine | failure rate | saving vs run to failure | saving vs fixed schedule |
|---|---|---|---|---|---|---|
| FD001 | run to failure |  | 100.0 ± 0.0 | 100% | 0% ± 0% | -59% ± 0% |
| FD001 | fixed schedule | 153 | 63.1 ± 0.0 | 9% | 37% ± 0% | 0% ± 0% |
| FD001 | alarm on predicted RUL | 12 | 27.3 ± 0.8 | 14% | 73% ± 1% | 57% ± 1% |
| FD001 | alarm on lower interval bound | 2 | 26.1 ± 0.7 | 7% | 74% ± 1% | 59% ± 1% |
| FD002 | run to failure |  | 100.0 ± 0.0 | 100% | 0% ± 0% | -53% ± 0% |
| FD002 | fixed schedule | 162 | 65.4 ± 0.0 | 13% | 35% ± 0% | 0% ± 0% |
| FD002 | alarm on predicted RUL | 21 | 36.6 ± 0.5 | 7% | 63% ± 1% | 44% ± 1% |
| FD002 | alarm on lower interval bound | 4 | 34.8 ± 0.6 | 4% | 65% ± 1% | 47% ± 1% |
| FD003 | run to failure |  | 100.0 ± 0.0 | 100% | 0% ± 0% | -23% ± 0% |
| FD003 | fixed schedule | 189 | 81.3 ± 0.0 | 24% | 19% ± 0% | 0% ± 0% |
| FD003 | alarm on predicted RUL | 11 | 32.3 ± 0.9 | 20% | 68% ± 1% | 60% ± 1% |
| FD003 | alarm on lower interval bound | 3 | 24.6 ± 1.3 | 9% | 75% ± 1% | 70% ± 2% |
| FD004 | run to failure |  | 100.0 ± 0.0 | 100% | 0% ± 0% | -8% ± 0% |
| FD004 | fixed schedule | 225 | 92.3 ± 0.0 | 46% | 8% ± 0% | 0% ± 0% |
| FD004 | alarm on predicted RUL | 14 | 39.4 ± 0.2 | 14% | 61% ± 0% | 57% ± 0% |
| FD004 | alarm on lower interval bound | 2 | 36.1 ± 0.6 | 9% | 64% ± 1% | 61% ± 1% |
