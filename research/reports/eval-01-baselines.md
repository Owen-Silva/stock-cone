# eval-01: the harness, and three baselines on DEV and VAL

> **Not financial advice.** This is a research log from building an educational tool. Backtest numbers are hypothetical, carry survivorship bias, and say nothing certain about the future. Nothing here is a recommendation to buy, sell or hold anything.

Lane `cone-harness`, 2026-10-03. Files: `research/splits.py`, `research/harness.py`,
`research/models_base.py`, outputs in `research/results/`.

## Method

**Splits** (`research/splits.py`, as in PLAN.md). TRAIN ticker = even last hex digit of
md5(upper-cased ticker). Evaluation tickers = the 155-stock scoreboard universe plus the
42-stock analog basket, minus SPY/QQQ: 163 tickers, **71 TRAIN / 92 TEST**. DEV = as-of before
2012-01-01, VAL = 2012-01-01..2018-12-31, TEST = 2019-01-01 on. DEV/VAL use TRAIN tickers only.
IRE and RVI have under a year of history and are skipped. That leaves 40 TRAIN tickers with
DEV rows and 48 with VAL rows (the rest listed after 2018).

**As-of grid.** Every 5 trading days from bar 300 of each ticker. A bar is kept only if the
engine's features are valid there, it is not inside a stale-quote stretch (the engine's own
`pool_ok` rule: more than 20% unchanged closes over 63 days, which hits some 1960s-70s Yahoo
data), and it falls inside the split. `--fast` keeps every 20th bar on every 3rd ticker. Those
rows are a subset of the full grid, so a fast run reuses a full run's cache.

**Data.** The harness runs the real `engine.Engine` in offline mode on the cached
`data/prices/*.csv`, read exactly as `engine.load_prices` reads them. There is zero network
access, so the harness never triggers the engine's daily re-download.

**Models** (`research/models_base.py`):
- `naive0`: r ~ N(0, (sig63·sqrt h)²), where sig63 is the engine's trailing 63-day std of daily log returns. p_up = 0.5.
- `naive_drift`: naive0 plus mu·h, where mu = (ln P[p] − ln P[0]) / p is the stock's mean daily log return over all bars up to the as-of date. p_up = Φ(mu·h / sig63·sqrt h).
- `v0`: the engine at commit 4d9b581, unchanged. It calls the engine's own `_select` (time rule A + h <= as-of, K = 250, de-clustering, Gaussian weights) and `_paths` (vol scaling), then takes all 19 quantiles with `engine.wquantile` of the weighted scaled analog end returns. q10..q90 are therefore the engine's p10..p90 exactly, with no interpolation. p_up is the engine's `odds.p_up`. The model abstains (a NaN row) when fewer than K_MIN = 100 analogs are eligible.

**Metrics** (one scoring function in `harness.metrics`, the same for every model; r = ln(P[p+h]/P[p])):
- Scaled pinball: mean pinball over the 19 quantiles, divided by the naive sigma sig63·sqrt h. Skill = 1 − model / naive0, both measured on the same rows. The **90% CI** comes from resampling whole tickers with replacement (2,000 draws), because rows within one ticker overlap in time.
- Coverage of q25..q75 and q10..q90, with |error| against 0.50 / 0.80. Width = mean(q90 − q10).
- Brier skill of p_up against a walk-forward base rate. The base rate is the share of h-bar up-moves in the stock's own history among moves that **end** at or before the as-of bar.
- Median-sign hit rate against the walk-forward majority guess (base >= 0.5 means "up"). A median of exactly 0 counts as the majority guess, so naive0's hit rate equals the majority's by construction.
- Volatility buckets: terciles of the as-of sig63 over all scored rows of that split and horizon. DEV daily-vol thresholds are 1.6% / 2.5%; VAL are 1.2% / 1.9%.
- All rows carry equal weight, so long-history stocks (KO, IBM, CAT, ...) dominate DEV.

**Speed.** v0 uses the engine's selection code, but its distances come from
`batch_dist2` on a `SortedPool`:
- The pool is sorted by each row's 5-bar end date. Rows that cannot be eligible yet are skipped, which helps DEV most.
- Distances are computed 16 as-of dates at a time, in cache-sized chunks. They use the same float32 operations in the same order as `Pool.dist2`, so every distance is bit-identical.
- The engine's `PRE` pre-selection (a 24,000-row partial sort per date) is replaced by a distance ball: every prefix row within a threshold estimated from a 1-in-16 subsample. `_select` uses the ball only when it holds at least 2,000 eligible rows, which makes it exact; otherwise `_select` runs its own full scan.

The harness proof checks the result against `Engine.walk_forward`'s code path, bit for bit.

## Results

### DEV (40 TRAIN tickers, before 2012, as-of every 5 bars)

| h | model | n | pinball skill vs naive0 [90% CI] | cov50 | cov80 | \|cov80−0.80\| | width80 | Brier skill | median-sign hit | majority hit |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | naive0 | 48,277 | 0 | 0.530 | 0.809 | 0.009 | 0.134 | +0.0018 | 0.510 | 0.510 |
| 5 | naive_drift | 48,277 | +0.0007 [+0.0002, +0.0012] | 0.529 | 0.810 | 0.010 | 0.134 | +0.0020 | 0.517 | 0.510 |
| 5 | v0 | 48,277 | −0.0008 [−0.0019, +0.0004] | 0.507 | 0.808 | 0.008 | 0.132 | −0.0046 | 0.515 | 0.510 |
| 21 | naive0 | 48,277 | 0 | 0.514 | 0.805 | 0.005 | 0.274 | −0.0080 | 0.550 | 0.550 |
| 21 | naive_drift | 48,277 | +0.0040 [+0.0017, +0.0065] | 0.520 | 0.806 | 0.006 | 0.274 | +0.0006 | 0.550 | 0.550 |
| 21 | v0 | 48,254 | −0.0011 [−0.0039, +0.0018] | 0.489 | 0.789 | 0.011 | 0.260 | −0.0112 | 0.545 | 0.550 |
| 63 | naive0 | 48,277 | 0 | 0.502 | 0.791 | 0.009 | 0.474 | −0.0181 | 0.580 | 0.580 |
| 63 | naive_drift | 48,277 | +0.0069 [+0.0001, +0.0140] | 0.514 | 0.797 | 0.003 | 0.474 | −0.0005 | 0.581 | 0.580 |
| 63 | v0 | 48,209 | +0.0065 [−0.0014, +0.0146] | 0.488 | 0.779 | 0.021 | 0.446 | −0.0047 | 0.581 | 0.580 |

v0 abstained on 23 (h = 21) and 68 (h = 63) of the earliest DEV rows, where fewer than 100 analogs were eligible.

### VAL (48 TRAIN tickers, 2012-2018, as-of every 5 bars)

| h | model | n | pinball skill vs naive0 [90% CI] | cov50 | cov80 | \|cov80−0.80\| | width80 | Brier skill | median-sign hit | majority hit |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | naive0 | 14,951 | 0 | 0.541 | 0.808 | 0.008 | 0.105 | −0.0090 | 0.554 | 0.554 |
| 5 | naive_drift | 14,951 | +0.0062 [+0.0047, +0.0079] | 0.550 | 0.812 | 0.012 | 0.105 | +0.0027 | 0.553 | 0.554 |
| 5 | v0 | 14,951 | +0.0051 [+0.0025, +0.0075] | 0.520 | 0.802 | 0.002 | 0.102 | −0.0063 | 0.533 | 0.554 |
| 21 | naive0 | 14,951 | 0 | 0.517 | 0.798 | 0.002 | 0.215 | −0.0351 | 0.591 | 0.591 |
| 21 | naive_drift | 14,951 | +0.0226 [+0.0163, +0.0290] | 0.535 | 0.811 | 0.011 | 0.215 | +0.0024 | 0.588 | 0.591 |
| 21 | v0 | 14,951 | +0.0127 [+0.0053, +0.0201] | 0.496 | 0.789 | 0.011 | 0.205 | −0.0179 | 0.574 | 0.591 |
| 63 | naive0 | 14,951 | 0 | 0.481 | 0.779 | 0.021 | 0.373 | −0.0965 | 0.650 | 0.650 |
| 63 | naive_drift | 14,951 | +0.0634 [+0.0445, +0.0842] | 0.523 | 0.812 | 0.012 | 0.373 | +0.0028 | 0.642 | 0.650 |
| 63 | v0 | 14,951 | +0.0421 [+0.0226, +0.0614] | 0.487 | 0.785 | 0.015 | 0.354 | −0.0279 | 0.628 | 0.650 |

### Volatility terciles (pinball skill / cov80 / Brier skill; low / mid / high)

| split | h | model | skill | cov80 | Brier skill |
|---|---|---|---|---|---|
| DEV | 5 | naive0 | 0 / 0 / 0 | 0.776 / 0.805 / 0.844 | −0.000 / +0.002 / +0.004 |
| DEV | 5 | naive_drift | +0.002 / +0.001 / −0.001 | 0.779 / 0.805 / 0.845 | +0.002 / +0.002 / +0.003 |
| DEV | 5 | v0 | −0.002 / +0.000 / −0.001 | 0.783 / 0.805 / 0.835 | −0.008 / −0.003 / −0.003 |
| DEV | 21 | naive0 | 0 / 0 / 0 | 0.765 / 0.803 / 0.846 | −0.019 / −0.008 / +0.003 |
| DEV | 21 | naive_drift | +0.010 / +0.004 / −0.003 | 0.767 / 0.806 / 0.844 | −0.002 / +0.001 / +0.003 |
| DEV | 21 | v0 | +0.005 / −0.001 / −0.008 | 0.759 / 0.789 / 0.817 | −0.011 / −0.008 / −0.015 |
| DEV | 63 | naive0 | 0 / 0 / 0 | 0.748 / 0.789 / 0.835 | −0.047 / −0.020 / +0.010 |
| DEV | 63 | naive_drift | +0.021 / +0.007 / −0.010 | 0.765 / 0.792 / 0.833 | −0.005 / −0.002 / +0.006 |
| DEV | 63 | v0 | +0.028 / −0.002 / −0.011 | 0.763 / 0.781 / 0.795 | +0.004 / −0.016 / −0.002 |
| VAL | 5 | naive0 | 0 / 0 / 0 | 0.764 / 0.824 / 0.837 | −0.013 / −0.010 / −0.004 |
| VAL | 5 | naive_drift | +0.011 / +0.005 / +0.002 | 0.771 / 0.828 / 0.837 | +0.006 / +0.001 / +0.002 |
| VAL | 5 | v0 | +0.005 / +0.008 / +0.002 | 0.763 / 0.815 / 0.827 | −0.009 / −0.005 / −0.005 |
| VAL | 21 | naive0 | 0 / 0 / 0 | 0.731 / 0.830 / 0.833 | −0.066 / −0.027 / −0.015 |
| VAL | 21 | naive_drift | +0.043 / +0.013 / +0.007 | 0.760 / 0.836 / 0.836 | +0.008 / +0.001 / −0.002 |
| VAL | 21 | v0 | +0.032 / +0.009 / −0.008 | 0.739 / 0.814 / 0.814 | −0.009 / −0.016 / −0.029 |
| VAL | 63 | naive0 | 0 / 0 / 0 | 0.691 / 0.827 / 0.819 | −0.139 / −0.116 / −0.040 |
| VAL | 63 | naive_drift | +0.106 / +0.058 / +0.015 | 0.759 / 0.853 / 0.824 | +0.018 / −0.002 / −0.007 |
| VAL | 63 | v0 | +0.074 / +0.061 / −0.019 | 0.728 / 0.830 / 0.797 | −0.018 / +0.002 / −0.065 |

### Run times (this PC, 2 worker threads, other lanes running at the same time)

| run | wall | v0 compute (thread-seconds) |
|---|---|---|
| VAL, 3 models, 71 tickers (48 with rows), 14,951 as-of dates | 217 s | 411 s (~27 ms/date) |
| DEV, 3 models, 71 tickers (40 with rows), 48,277 as-of dates | 363 s | 696 s (~14 ms/date) |
| either split again from the prediction cache | 8-12 s | 0 |

naive0 and naive_drift take under 1 s per split. The engine's own code path costs about 40-60 ms per
date (it varies with machine load), so the accelerated v0 is about 1.6x faster on VAL and
3-4x faster on DEV, before threading.

## What it means

1. **v0 does not beat the simplest drift baseline anywhere.** On VAL, naive_drift's pinball skill is +0.006 / +0.023 / +0.063 (1W / 1M / 3M) and v0's is +0.005 / +0.013 / +0.042. On DEV the two are tied at 3M (+0.0069 vs +0.0065) and v0 is slightly worse at 1W and 1M (−0.001; its CI includes 0). Whatever v0 gains over a zero-drift cone comes from drift: on DEV (1962-2011, no single bull run) it is no better than naive0 at 1W and 1M. That matches PLAN.md's suspicion that the edge is upward drift. A stock's own long-run drift gives that drift more cheaply and better calibrated than the analog pool does.
2. **v0's cone is too narrow at longer horizons.** Its 80% width is 1.5-3% below naive's at 1W and 5-6% below at 1M and 3M, and coverage falls with horizon: 0.808 / 0.789 / 0.779 on DEV, and 0.802 / 0.789 / 0.785 on VAL. Its central 50% band is also slightly narrow (0.49 against 0.50). naive_drift with the same width covers 0.81. Widening or recalibrating v0 per horizon (idea 3) should help it directly.
3. **The baselines fail in calm regimes.** In the low-volatility tercile, every model under-covers: cov80 is 0.69-0.78, worst at 3M on VAL (naive0 0.691, v0 0.728). Trailing 63-day vol is too low when vol is low, because vol mean-reverts. In the high-vol tercile, naive0 over-covers (0.82-0.85). This is the clearest target for the vol-forecast lane: shrink sig63 toward a long-run level.
4. **v0 is worst on volatile stocks.** In the high-vol tercile, v0's skill is negative at 1M and 3M on both splits (−0.008 to −0.019), and its Brier skill reaches −0.065 at 3M on VAL. The analog basket of large caps, vol-scaled, describes small and volatile names poorly (worst per-ticker skill at 1M on VAL: CVNA, KGC, CCJ, UEC).
5. **Direction: v0 has no real skill.** Its Brier skill is negative in every split and horizon except DEV 3M low-vol (+0.004). Its median-sign hit rate is below the walk-forward majority guess on VAL at every horizon: 0.533 vs 0.554, 0.574 vs 0.591, 0.628 vs 0.650. naive_drift's p_up only about matches the base rate (Brier skill about +0.002). The scoreboard's "worse than the usual direction" finding holds under the fair, walk-forward baseline too.
6. **Noise.** The ticker-bootstrap 90% CIs are about ±0.001 at 1W, ±0.003-0.007 at 1M and ±0.008-0.02 at 3M. A variant that wins on VAL by less than that has not shown it is better. The CIs are not paired between two non-naive models, so they overstate the uncertainty of a head-to-head difference.

## Proofs (all passed: `python research/harness.py --selftest`)

- **(a) No look-ahead.** naive0, naive_drift and v0 were run at 5 (ticker, as-of) pairs: KO 2008-10-10, AAPL 2015-06-12, NVDA 2011-03-04, SMCI 2018-02-02, IBM 1987-10-30. Each run was repeated on a context where every series (the pool, SPY, ^GSPC and the target) was truncated at the as-of date. All 19 quantiles and p_up were identical, bit for bit, for all 3 horizons. Any lane can run the same check on its own model with `harness.py --lookahead --models <name>`.
- **(b) Synthetic.** 40 Gaussian random walks (sigma = 2%/day, 6,000 days each, about 45k rows per horizon) went through the real scoring path. With the true sigma, cov50 / cov80 were 0.499 / 0.801 (5d), 0.495 / 0.799 (21d) and 0.495 / 0.792 (63d). naive0 with its trailing estimate came in slightly lower, as expected: 0.496 / 0.796, 0.492 / 0.795, 0.492 / 0.787. The true-sigma model beat naive0 by +0.002 pinball skill, a sensible size for estimation noise alone.
- **(c) v0 matches the engine.** At 5 (ticker, as-of) pairs × 3 horizons, v0's q10/q25/q50/q75/q90 match `Engine._forecast_at` to within 6e-6 in log terms. The live forecast rounds prices to 6 significant digits, so this is rounding. They match `Engine.walk_forward` exactly (difference 0.0), and p_up matches as well. Across 5 tickers × 48 dates × 3 horizons, the accelerated search and the engine's own walk_forward code path give bit-identical outputs.
- `python engine.py --selfcheck` passed (18/18 OK) at 23:36 and at 23:13. engine.py is unmodified.

## Not verified

- Statistical significance of v0 against naive_drift as a paired test. Only the unpaired ticker-bootstrap CIs against naive0 exist.
- Exactness of the accelerated v0 search on every row. It is proven by argument (bit-identical distances, plus the ball rule) and checked bit for bit on 240 dates × 3 horizons and in the look-ahead proof. It was not compared on all 63k rows. If two analogs tie at exactly the 2,000th distance, the engine's own `argpartition` is arbitrary too.
- TEST was never run. `--split test` refuses without `--coordinator`, and it refused (exit 2) when tried without it.
- Rows are pooled with equal weight, so DEV leans on long-history mega caps. No ticker-equal-weighted view is reported.
