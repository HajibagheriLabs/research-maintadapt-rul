# N-CMAPSS inspection

Written by `make research-ncmapss-inspect` (`scripts/research/inspect_ncmapss.py`) from the files in `data/raw/NCMAPSS/data_set/`; no model was trained. Groups and shapes of every file are in `inspection_files.csv`, one row per unit in `inspection_units.csv`.

## Files that fail to load (excluded)

- `N-CMAPSS_DS08d-010.h5`: OSError: Unable to synchronously open file (truncated file: eof = 2885034848, sblock->base_addr = 0, stored_eof = 2885034880)

## Layout, as found

Every readable file holds the groups `W_dev`/`W_test`, `X_s_dev`/`X_s_test`, `X_v_dev`/`X_v_test`, `T_dev`/`T_test`, `Y_dev`/`Y_test`, `A_dev`/`A_test` and their `_var` names, with the same number of rows within each part. Columns:

- `W`: alt, Mach, TRA, T2
- `X_s`: T24, T30, T48, T50, P15, P2, P21, P24, Ps30, P40, P50, Nf, Nc, Wf
- `X_v`: T40, P30, P45, W21, W22, W25, W31, W32, W48, W50, SmFan, SmLPC, SmHPC, phi
- `T`: fan_eff_mod, fan_flow_mod, LPC_eff_mod, LPC_flow_mod, HPC_eff_mod, HPC_flow_mod, HPT_eff_mod, HPT_flow_mod, LPT_eff_mod, LPT_flow_mod
- `A`: unit, cycle, Fc, hs

Model inputs come from `W` and `X_s` only; `X_v`, `T` and the health state `hs` never become inputs.

## Sub-datasets

| sub-dataset | dev units | test units | flights per unit (min-max) | flight classes | degraded health parameters |
|---|---|---|---|---|---|
| DS01-005 | 6 (1, 2, 3, 4, 5, 6) | 4 (7, 8, 9, 10) | 75-100 | 1, 2, 3 | HPT_eff_mod |
| DS02-006 | 6 (2, 5, 10, 16, 18, 20) | 3 (11, 14, 15) | 59-89 | 1, 2, 3 | HPT_eff_mod; HPT_eff_mod LPT_eff_mod LPT_flow_mod |
| DS03-012 | 9 (1, 2, 3, 4, 5, 6, 7, 8, 9) | 6 (10, 11, 12, 13, 14, 15) | 59-93 | 1, 2, 3 | HPT_eff_mod LPT_eff_mod LPT_flow_mod |
| DS04 | 6 (1, 2, 3, 4, 5, 6) | 4 (7, 8, 9, 10) | 69-100 | 2, 3 | fan_eff_mod fan_flow_mod |
| DS05 | 6 (1, 2, 3, 4, 5, 6) | 4 (7, 8, 9, 10) | 65-100 | 1, 2, 3 | HPC_eff_mod HPC_flow_mod |
| DS06 | 6 (1, 2, 3, 4, 5, 6) | 4 (7, 8, 9, 10) | 65-99 | 1, 2, 3 | LPC_eff_mod LPC_flow_mod HPC_eff_mod HPC_flow_mod |
| DS07 | 6 (1, 2, 3, 4, 5, 6) | 4 (7, 8, 9, 10) | 67-96 | 1, 2, 3 | LPT_eff_mod LPT_flow_mod |
| DS08a-009 | 9 (1, 2, 3, 4, 5, 6, 7, 8, 9) | 6 (10, 11, 12, 13, 14, 15) | 51-95 | 1, 2, 3 | fan_eff_mod fan_flow_mod LPC_eff_mod LPC_flow_mod HPC_eff_mod HPC_flow_mod HPT_eff_mod HPT_flow_mod LPT_eff_mod LPT_flow_mod |
| DS08c-008 | 6 (1, 2, 3, 4, 5, 6) | 4 (7, 8, 9, 10) | 48-62 | 2, 3 | fan_eff_mod fan_flow_mod LPC_eff_mod LPC_flow_mod HPC_eff_mod HPC_flow_mod HPT_eff_mod HPT_flow_mod LPT_eff_mod LPT_flow_mod |

99 units in 9 readable sub-datasets.

| check | units |
|---|---|
| units whose rows form one contiguous block, flights 1..T in order | 99 of 99 |
| units with RUL = T - t at every flight, constant within flights | 99 of 99 |
| units whose health state only falls from 1 to 0 | 99 of 99 |
| units with one flight class | 99 of 99 |
| units that reach an abnormal health state | 99 of 99 |

## Cap (PREREGISTRATION.md section 8)

RUL at the first flight with health state 0, over the 99 units: median 50, quartiles 42 and 58.5, range 31-83. Rounded to 5 cycles, the cap is **50**.
