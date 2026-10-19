# test-02 — The locked 2019+ check of the cone (coordinator, 2026-10-04 ~01:30)

> **Not financial advice.** This is a research log from building an educational tool. Backtest numbers are hypothetical, carry survivorship bias, and say nothing certain about the future. Nothing here is a recommendation to buy, sell or hold anything.

The second and last TEST open of the night. Every model was frozen before this run, and
the product model (`cone2_f_conf_dev`) was pre-registered in `research/PLAN.md` (commit
170c130) before TEST was opened. Nothing was tuned on these numbers.

**Method.** `research/harness.py --split test --coordinator` from a `git archive` snapshot
of commit 581bbb1 (so the engine edits happening at the same time could not touch it), on
all 159 testable tickers (TRAIN and held-out TEST tickers), as-of dates every 5 trading
days from 2019-01 to 2026-10. About 52,000 forecasts per horizon. Raw output:
`research/results/test2_harness_summary.txt` and `research/results/*_test.json`.

## Results (2019-01 .. 2026-10, 159 tickers)

Pinball skill vs `naive0` (higher is better; 90% interval from resampling tickers), and
how often the real price landed inside the 50% and 80% bands (aim 0.50 / 0.80).

| model | h | skill [90% CI] | inside 50% | inside 80% |
|---|---|---|---|---|
| **v2 (served)** `cone2_f_conf_dev` | 5 | **+0.0088** [+0.0077, +0.0098] | 0.510 | **0.802** |
| | 21 | **+0.0119** [+0.0097, +0.0141] | 0.492 | **0.785** |
| | 63 | **+0.0241** [+0.0188, +0.0297] | 0.475 | **0.782** |
| v0 (old engine) | 5 | +0.0008 [-0.0004, +0.0019] | 0.500 | 0.794 |
| | 21 | -0.0001 [-0.0029, +0.0028] | 0.473 | 0.763 |
| | 63 | +0.0101 [+0.0037, +0.0167] | 0.464 | 0.763 |
| cone2 (own-mean centre, not chosen) | 5 / 21 / 63 | +0.0087 / +0.0105 / +0.0182 | 0.510 / 0.487 / 0.479 | 0.802 / 0.780 / 0.779 |
| vol_har_drift | 5 / 21 / 63 | +0.0076 / +0.0080 / +0.0083 | 0.506 / 0.484 / 0.482 | 0.796 / 0.776 / 0.785 |
| naive_drift | 5 / 21 / 63 | +0.0014 / +0.0025 / +0.0016 | 0.555 / 0.527 / 0.512 | 0.819 / 0.799 / 0.796 |
| event_har (not shipped) | 5 / 21 / 63 | +0.0079 / +0.0085 / +0.0083 | 0.508 / 0.489 / 0.481 | 0.797 / 0.781 / 0.784 |

By volatility tercile (low / mid / high), v2's 80% coverage is 0.798 / 0.803 / 0.804 at
1W, 0.770 / 0.795 / 0.792 at 1M and 0.779 / 0.784 / 0.782 at 3M (v0: 0.787 / 0.808 / 0.788,
0.746 / 0.789 / 0.754, 0.750 / 0.791 / 0.746).

**Direction.** v2's Brier skill against the walk-forward base rate is +0.0016 / +0.0015 /
+0.0137 (v0: -0.0073 / -0.0125 / -0.0066), so its "chance it ends higher" is slightly
better calibrated than the base rate at 3 months. Its median-sign hit rate (0.526 / 0.548 /
0.577) is still below simply guessing each stock's usual direction (0.538 / 0.559 / 0.582).
**No direction edge.**

## What it means

- The pre-registered choice held up: v2 beats the old engine, both simple baselines and
  the own-mean variant at every horizon, with intervals clear of zero.
- The ranges are honest at 1 week (80.2% inside the 80% band) and slightly narrow at
  1 and 3 months (78.5% / 78.2%). 2019+ includes the 2020 crash and the 2022 bear market,
  which no trailing-vol model sees coming. The old engine was at 76.3% / 76.3%.
- Direction: use the cone for how far, not which way.

## Not verified

- Survivorship: every ticker exists today, so 2019+ is a survivor sample (it mostly affects
  centres, less the widths).
- Overlapping windows (5-day steps with 21/63-day horizons) make the 1M/3M intervals
  optimistic.
- pandas printed divide-by-zero warnings in `log` during the run. Their source was not
  traced; every model reported 0 abstentions and finite scores.
