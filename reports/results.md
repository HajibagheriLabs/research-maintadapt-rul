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
