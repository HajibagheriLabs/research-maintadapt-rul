# Deviations from the pre-registration

Every change to PREREGISTRATION.md after the tag `prereg-v1`, newest last; affected results are labelled in every output.

| date | section | change | reason | expected impact |
|---|---|---|---|---|
| 2026-11-15 | 8 | N-CMAPSS pools 99 units from nine of the ten sub-datasets; `N-CMAPSS_DS08d-010.h5` is left out. | The file in the downloaded NASA archive is truncated (its last 32 bytes are missing) and cannot be opened (`reports/research/ncmapss/inspection.md`). | N-CMAPSS is secondary and descriptive; its results cover the readable units only. No effect on FD001-FD004 or on any verdict. |
| 2026-11-15 | 4.3, 8 | On N-CMAPSS the networks fit the RUL divided by the N-CMAPSS cap of 50, not by 125, and predictions are clipped to [0, 50]. | Section 4.3 scales the target by the C-MAPSS cap; section 8 sets a cap of its own for N-CMAPSS, so the scale and the clip follow that cap. | N-CMAPSS only; the target range of the networks matches the other families'. |
| 2026-11-15 | 5.7, 6, 8 | On N-CMAPSS, which has no official test engines, jitter is measured on the held-out units with the truth cap of 50, like section 8's leaderboard RMSE, and one bootstrap draw of units resamples D, leaderboard RMSE and jitter together, instead of section 6's independent draw of official test engines. | Section 8 puts the leaderboard metric on the held-out units that D is also measured on; independent draws of the same units would break their pairing. | N-CMAPSS only (`reports/research/ncmapss/`); intervals reflect the shared draw. No effect on FD001-FD004. |
| 2026-11-17 | 11 | The CPU is an Intel Core i7-4790K with 4 cores and 8 threads, not an 8-core CPU. | The registration described the processor wrongly; the fits ran on the machine registered, with its RTX 3090. | None on any result. The hours in DECISIONS.md and `paper_assets/computed/compute.csv` were measured on this machine. |

Gaps the pre-registration leaves open, such as tie rules, the inner early-stopping engines of the nested fits (S4) and the target scale of uncapped training (S5), were filled before the affected results existed and are recorded in DECISIONS.md; they change no registered choice.
