## Test RMSE (mean ± std over seeds)

| model | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Ridge, rolling features | 14.99 ± 0.00 | 15.96 ± 0.00 | 16.29 ± 0.00 | 18.53 ± 0.00 |
| XGBoost, rolling features | 12.14 ± 0.09 | 12.60 ± 0.05 | 12.49 ± 0.18 | 13.49 ± 0.10 |

## Test NASA score (mean ± std over seeds; lower is better)

| model | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Ridge, rolling features | 347 ± 0 | 1179 ± 0 | 508 ± 0 | 1596 ± 0 |
| XGBoost, rolling features | 214 ± 5 | 690 ± 12 | 262 ± 10 | 819 ± 21 |

## XGBoost normalisation ablation

| subset | normalisation | rmse | nasa_score |
|---|---|---|---|
| FD002 | condition-aware | 12.60 ± 0.05 | 690 ± 12 |
| FD002 | global | 15.91 ± 0.13 | 1257 ± 31 |
| FD004 | condition-aware | 13.49 ± 0.10 | 819 ± 21 |
| FD004 | global | 17.39 ± 0.09 | 1713 ± 58 |
