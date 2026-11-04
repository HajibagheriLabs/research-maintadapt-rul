# Smoke test: sanity of the derived tables

NOT CONFIRMATORY. These checks run on the smoke archive (FD001, repeat 0, all five
folds, configuration 0 of every family with 3 training epochs for the networks, and
two controls). They test the pipeline and say nothing about the study's questions.

Sources: `runs/research/derived/smoke/<fleet>/` (written by
`make research-outcomes ROOT=smoke` and `make research-metrics ROOT=smoke`); this file is written
by `make research-sanity`.
S pools the outcomes of the five folds; CR_age uses A* tuned on each fold's outer-train
lifetimes.

| check | expected | observed | result |
|---|---|---|---|
| FD001: oracle control, validation tuning: S at every rule, lead time and K | close to 1 | 1 to 1 | pass |
| FD001: oracle control, crossfit tuning: S at every rule, lead time and K | close to 1 | 1 to 1 | pass |
| FD001: oracle control, oracle tuning: S at every rule, lead time and K | 1 exactly | 1 to 1 | pass |
| FD001: constant control: S at every tuning, rule, lead time and K | <= 0 | -34.5877 to -0.445812 | pass |
| FD001: rows in policies | 7560 | 7560 | pass |
| FD001: rows in outcomes | 151200 | 151200 | pass |
| FD001: rows in age | 700 | 700 | pass |
| FD001: rows in components | 8400 | 8400 | pass |
| FD001: rows in fit_metrics | 60 | 60 | pass |
| FD001: missing values in policies | 0 | 0 | pass |
| FD001: missing values in outcomes | 0 | 0 | pass |
| FD001: missing values in age | 0 | 0 | pass |
| FD001: missing values in fit_metrics | 0 | 0 | pass |
| FD001: missing values in the components that apply to each split | 0 | 0 | pass |
| FD001: every engine once per repeat in outer_test, for every policy | 100 engines, each once | 1512 policies checked | pass |
| FD001: truncation points per outer_test engine | 20 | 20 to 20 | pass |

16 of 16 checks pass.
