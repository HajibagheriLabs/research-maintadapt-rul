# Sanity of the derived tables of `runs/research/archive`

Checks of the pipeline, not results; nothing here bears on the hypotheses. The two
controls run only in the smoke test (`reports/research/smoke/sanity.md`), so the
checks here are structural.

Sources: `runs/research/derived/archive/<fleet>/` (written by
`make research-outcomes ROOT=archive` and `make research-metrics ROOT=archive`); this file is written
by `make research-sanity ROOT=archive`.
S pools the outcomes of the five folds; CR_age uses A* tuned on each fold's outer-train
lifetimes.

| check | expected | observed | result |
|---|---|---|---|
| FD001: rows in policies | 213570 | 213570 | pass |
| FD001: rows in outcomes | 4271400 | 4271400 | pass |
| FD001: rows in age | 2100 | 2100 | pass |
| FD001: rows in components | 237300 | 237300 | pass |
| FD001: rows in fit_metrics | 1695 | 1695 | pass |
| FD001: missing values in policies | 0 | 0 | pass |
| FD001: missing values in outcomes | 0 | 0 | pass |
| FD001: missing values in age | 0 | 0 | pass |
| FD001: missing values in fit_metrics | 0 | 0 | pass |
| FD001: missing values in the components that apply to each split | 0 | 0 | pass |
| FD001: every engine once per repeat in outer_test, for every policy | 100 engines, each once | 42714 policies checked | pass |
| FD001: truncation points per outer_test engine | 20 | 20 to 20 | pass |
| FD002: rows in policies | 213570 | 213570 | pass |
| FD002: rows in outcomes | 11105640 | 11105640 | pass |
| FD002: rows in age | 5460 | 5460 | pass |
| FD002: rows in components | 615285 | 615285 | pass |
| FD002: rows in fit_metrics | 1695 | 1695 | pass |
| FD002: missing values in policies | 0 | 0 | pass |
| FD002: missing values in outcomes | 0 | 0 | pass |
| FD002: missing values in age | 0 | 0 | pass |
| FD002: missing values in fit_metrics | 0 | 0 | pass |
| FD002: missing values in the components that apply to each split | 0 | 0 | pass |
| FD002: every engine once per repeat in outer_test, for every policy | 260 engines, each once | 42714 policies checked | pass |
| FD002: truncation points per outer_test engine | 20 | 20 to 20 | pass |
| FD003: rows in policies | 213570 | 213570 | pass |
| FD003: rows in outcomes | 4271400 | 4271400 | pass |
| FD003: rows in age | 2100 | 2100 | pass |
| FD003: rows in components | 237300 | 237300 | pass |
| FD003: rows in fit_metrics | 1695 | 1695 | pass |
| FD003: missing values in policies | 0 | 0 | pass |
| FD003: missing values in outcomes | 0 | 0 | pass |
| FD003: missing values in age | 0 | 0 | pass |
| FD003: missing values in fit_metrics | 0 | 0 | pass |
| FD003: missing values in the components that apply to each split | 0 | 0 | pass |
| FD003: every engine once per repeat in outer_test, for every policy | 100 engines, each once | 42714 policies checked | pass |
| FD003: truncation points per outer_test engine | 20 | 20 to 20 | pass |
| FD004: rows in policies | 213570 | 213570 | pass |
| FD004: rows in outcomes | 10635786 | 10635786 | pass |
| FD004: rows in age | 5229 | 5229 | pass |
| FD004: rows in components | 589182 | 589182 | pass |
| FD004: rows in fit_metrics | 1695 | 1695 | pass |
| FD004: missing values in policies | 0 | 0 | pass |
| FD004: missing values in outcomes | 0 | 0 | pass |
| FD004: missing values in age | 0 | 0 | pass |
| FD004: missing values in fit_metrics | 0 | 0 | pass |
| FD004: missing values in the components that apply to each split | 0 | 0 | pass |
| FD004: every engine once per repeat in outer_test, for every policy | 249 engines, each once | 42714 policies checked | pass |
| FD004: truncation points per outer_test engine | 20 | 20 to 20 | pass |

48 of 48 checks pass.
