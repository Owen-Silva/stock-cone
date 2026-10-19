# timing-01: Market timing and risk regimes on the S&P 500

> **Not financial advice.** This is a research log from building an educational tool. Backtest numbers are hypothetical, carry survivorship bias, and say nothing certain about the future. Nothing here is a recommendation to buy, sell or hold anything.

Lane: cone-timing, 2026-10-03. Code: `research/timing.py`. Data outputs: `research/results/timing_*`.
Nothing here was computed on 2019+ data. TEST has not been run.

**Bottom line.** None of the classic timing rules adds return. Over the long bull stretches
even the better ones cost 1.4 to 2.3 points a year (1950-99). What they do reliably, on DEV, on DEV after 1950, and on
VAL, is cut the worst drawdown by about a third to two-thirds. One flag is honest and
repeatable: "S&P below its 200-day average" roughly triples the odds of a month worse than
-5% in both periods. But it also raises the odds of a month better than +5%, so it flags
**turbulence**, not direction. The one rule worth keeping is a risk dial, not a return
engine: **S&P 500 vs its 200-day moving average, checked at the month-end close**.

## Method

- **Data.** `data/prices/_GSPC.csv` (date, close, volume) holds daily S&P 500 index closes from
  1927-12-30 (24.8k rows; the 1933-03 bank-holiday gap is the only long one). This is a
  **price index with no dividends**. `SPY.csv` (from 1993) and `QQQ.csv` (from 1999-03) are
  yfinance `auto_adjust=True` closes (see `engine.py` around line 170), so they are
  total-return. They serve as VAL cross-checks.
- **Splits (by time).** DEV covers returns before 2000-01-01; evaluation runs 1929-02-04..1999-12-31
  once every signal is warm. VAL covers 2000-01-03..2018-12-31. For dev and val the loader
  cuts every series at 2018-12-31, so 2019+ cannot be touched. TEST (2019+) only runs with
  `python research/timing.py --split test --coordinator`; without the flag the script
  refuses (checked).
- **Execution.** Each rule decides at the close of a check day (the last trading day of the
  week, or of the month) and trades at the **next** close. The return from close u-1 to u
  therefore uses the weight decided at close u-2. Every rule shares the same evaluation
  window, which starts once every rule at every frequency holds a position.
- **Costs and assumptions.** 5 bps per unit of weight changed, so a full in/out switch costs
  5 bps. Buy-and-hold pays no cost. **Cash earns 0.** There is no leverage, so weights sit
  in [0, 1].
- **Variants: 12 rules x 2 check frequencies = 24, every one reported below.** The
  parameters were fixed before any run.
  - `ma50`, `ma100`, `ma200`: close above its 50/100/200-day SMA.
  - `ma10m`: the Faber rule, close above the mean of the last 9 month-end closes plus today's close.
  - `tsm3`, `tsm6`, `tsm12`: the 3/6/12-month (63/126/252-day) return is positive.
  - `vt15`: weight = min(1, 15% / EWMA vol), with lambda 0.94. 15% is roughly the long-run S&P
    vol, chosen a priori.
  - `dd10`, `dd20`: invested unless the close is more than 10% / 20% below its 252-day high.
  - `vote3`: majority of ma50, ma200 and tsm12.
  - `ma200_vt`: ma200 multiplied by vt15.
- **Selection, pre-registered.** The DEV winner is the variant with the highest DEV Sharpe.
  A rule counts as **real** only if both Sharpe and max drawdown beat buy-and-hold on all
  three of DEV, DEV post-1950 (which drops the Depression) and VAL.
- **Uncertainty.** The bracket is a 90% CI on (Sharpe of the rule minus Sharpe of B&H) from a
  circular block bootstrap with 126-day blocks and 500 reps.
- **Engine sanity checks** (`timing_sanity_dev.json`).
  - A perfect-foresight cheat (daily) scores Sharpe 7.8, so the engine does respond to
    information.
  - 20 coin-flip monthly signals score Sharpe 0.28 ± 0.09, against 0.42 for B&H on the same
    window. That is the expected penalty for random 50% exposure, so randomness does not
    look like edge.

## Results (DEV / VAL tables)

Abbreviations: CAGR = annual return, maxDD = max drawdown, dSharpe = Sharpe minus B&H Sharpe.

### DEV, ^GSPC 1929-02-04..1999-12-31 (price index, cash 0)

| rule | freq | CAGR | vol | Sharpe | dSharpe [90% CI] | maxDD | worst yr | invested | switches/yr |
|---|---|---|---|---|---|---|---|---|---|
| buy_hold | - | 5.9% | 18.8% | 0.40 | +0.00 | -86.2% | -47.1% (1931) | 100% | 0.0 |
| ma50 | weekly | 3.8% | 12.1% | 0.37 | -0.03 [-0.21, +0.14] | -51.1% | -26.1% (1938) | 61% | 7.9 |
| ma100 | weekly | 5.9% | 12.3% | 0.52 | +0.13 [-0.04, +0.30] | -38.5% | -22.7% (1939) | 63% | 5.0 |
| ma200 | weekly | 6.0% | 12.9% | 0.52 | +0.12 [-0.06, +0.30] | -56.8% | -23.0% (1932) | 66% | 2.7 |
| ma10m | weekly | 6.1% | 12.9% | 0.52 | +0.13 [-0.04, +0.31] | -49.9% | -18.4% (1990) | 66% | 2.5 |
| tsm3 | weekly | 4.4% | 12.7% | 0.40 | +0.00 [-0.15, +0.16] | -53.9% | -22.2% (1931) | 63% | 5.4 |
| tsm6 | weekly | 5.3% | 13.2% | 0.46 | +0.06 [-0.10, +0.25] | -57.4% | -26.5% (1932) | 66% | 3.1 |
| tsm12 | weekly | 5.4% | 12.9% | 0.47 | +0.07 [-0.09, +0.24] | -48.8% | -23.2% (1929) | 68% | 2.0 |
| vt15 | weekly | 5.8% | 13.2% | 0.49 | +0.09 [+0.00, +0.17] | -65.7% | -26.4% (1931) | 89% | 19.2 |
| dd10 | weekly | 6.0% | 11.8% | 0.56 | +0.16 [-0.01, +0.32] | -42.1% | -17.6% (1973) | 70% | 2.0 |
| dd20 | weekly | 5.1% | 14.3% | 0.42 | +0.02 [-0.12, +0.16] | -64.7% | -24.8% (1973) | 87% | 1.3 |
| vote3 | weekly | 6.0% | 12.6% | 0.53 | +0.13 [-0.04, +0.31] | -51.1% | -24.3% (1932) | 67% | 3.5 |
| **ma200_vt** | **weekly** | 5.7% | 10.1% | **0.60** | +0.21 [+0.02, +0.39] | -35.2% | -18.3% (1990) | 62% | 12.0 |
| ma50 | monthly | 4.0% | 12.1% | 0.38 | -0.01 [-0.19, +0.15] | -55.0% | -29.5% (1939) | 60% | 4.4 |
| ma100 | monthly | 4.5% | 13.2% | 0.40 | +0.00 [-0.15, +0.18] | -50.2% | -20.3% (1973) | 63% | 2.9 |
| ma200 | monthly | 6.3% | 13.2% | 0.53 | +0.13 [-0.05, +0.31] | -53.9% | -20.6% (1932) | 65% | 1.6 |
| ma10m | monthly | 5.8% | 13.3% | 0.49 | +0.10 [-0.08, +0.28] | -53.9% | -20.6% (1932) | 66% | 1.5 |
| tsm3 | monthly | 4.0% | 13.5% | 0.36 | -0.04 [-0.18, +0.11] | -68.7% | -42.9% (1931) | 63% | 2.7 |
| tsm6 | monthly | 5.2% | 13.0% | 0.45 | +0.05 [-0.12, +0.23] | -60.1% | -26.0% (1939) | 65% | 1.6 |
| tsm12 | monthly | 5.2% | 13.2% | 0.45 | +0.05 [-0.11, +0.21] | -54.4% | -25.5% (1940) | 68% | 1.1 |
| vt15 | monthly | 5.5% | 13.5% | 0.46 | +0.07 [-0.02, +0.14] | -70.8% | -26.7% (1931) | 89% | 5.0 |
| dd10 | monthly | 5.3% | 12.1% | 0.49 | +0.09 [-0.08, +0.27] | -51.0% | -26.9% (1940) | 70% | 1.1 |
| dd20 | monthly | 6.4% | 14.2% | 0.51 | +0.11 [-0.04, +0.25] | -62.5% | -28.5% (1940) | 87% | 0.6 |
| vote3 | monthly | 6.0% | 13.0% | 0.52 | +0.12 [-0.06, +0.30] | -45.6% | -21.4% (1939) | 66% | 2.0 |
| ma200_vt | monthly | 5.7% | 10.5% | 0.58 | +0.19 [+0.00, +0.36] | -35.3% | -17.4% (1990) | 61% | 3.8 |

### DEV after 1950 only (1950-01..1999-12, same rules; `timing_dev_post1950.csv`)

The DEV advantage is almost entirely the Depression. After 1950 no rule beats B&H on Sharpe
by more than 0.04, and every one gives up return.

| series | CAGR | vol | Sharpe | maxDD |
|---|---|---|---|---|
| buy_hold | 9.4% | 13.5% | 0.73 | -48.2% |
| ma200 monthly | 8.0% | 10.8% | 0.77 | -33.2% |
| ma200 weekly | 7.4% | 10.6% | 0.73 | -36.9% |
| ma10m monthly | 7.7% | 10.8% | 0.74 | -33.2% |
| ma200_vt monthly | 7.4% | 10.3% | 0.75 | -31.4% |
| ma200_vt weekly | 7.1% | 9.9% | 0.74 | -32.4% |
| vt15 weekly | 8.6% | 12.1% | 0.74 | -41.2% |
| dd10 monthly | 7.5% | 11.4% | 0.69 | -33.2% |
| tsm12 monthly | 7.4% | 11.4% | 0.68 | -35.2% |
| vote3 monthly | 7.6% | 11.0% | 0.73 | -33.2% |

### VAL, ^GSPC 2000-01-03..2018-12-31 (price index, cash 0)

| rule | freq | CAGR | vol | Sharpe | dSharpe [90% CI] | maxDD | worst yr | invested | switches/yr |
|---|---|---|---|---|---|---|---|---|---|
| buy_hold | - | 2.9% | 19.1% | 0.24 | +0.00 | -56.8% | -38.5% (2008) | 100% | 0.0 |
| ma50 | weekly | 2.0% | 10.8% | 0.24 | -0.01 [-0.26, +0.23] | -26.9% | -13.8% (2000) | 62% | 8.2 |
| ma100 | weekly | 2.6% | 10.9% | 0.29 | +0.04 [-0.29, +0.38] | -39.9% | -17.5% (2000) | 67% | 5.4 |
| ma200 | weekly | 2.7% | 11.0% | 0.30 | +0.06 [-0.28, +0.36] | -22.7% | -13.1% (2000) | 69% | 3.8 |
| ma10m | weekly | 2.4% | 11.1% | 0.27 | +0.02 [-0.29, +0.34] | -25.1% | -13.1% (2000) | 69% | 4.3 |
| tsm3 | weekly | 1.7% | 10.8% | 0.21 | -0.03 [-0.37, +0.28] | -37.3% | -18.5% (2000) | 64% | 5.9 |
| tsm6 | weekly | 3.6% | 11.2% | 0.38 | +0.13 [-0.18, +0.43] | -28.5% | -14.9% (2000) | 69% | 3.3 |
| tsm12 | weekly | 4.4% | 12.2% | 0.42 | +0.17 [-0.14, +0.45] | -20.0% | -12.8% (2011) | 72% | 1.5 |
| vt15 | weekly | 3.1% | 13.4% | 0.30 | +0.05 [-0.08, +0.17] | -44.4% | -21.4% (2008) | 87% | 24.4 |
| dd10 | weekly | 3.9% | 11.9% | 0.38 | +0.14 [-0.18, +0.45] | -26.3% | -19.6% (2011) | 74% | 1.6 |
| dd20 | weekly | 3.8% | 14.7% | 0.33 | +0.09 [-0.11, +0.24] | -37.4% | -23.0% (2008) | 88% | 0.8 |
| vote3 | weekly | 3.1% | 11.2% | 0.33 | +0.09 [-0.24, +0.40] | -23.0% | -16.0% (2011) | 70% | 4.1 |
| **ma200_vt** | **weekly** | 2.7% | 10.1% | 0.31 | +0.07 [-0.27, +0.38] | -19.0% | -12.1% (2015) | 66% | 12.1 |
| ma50 | monthly | 0.7% | 11.9% | 0.12 | -0.12 [-0.35, +0.10] | -33.4% | -17.3% (2008) | 62% | 4.4 |
| ma100 | monthly | 3.1% | 11.3% | 0.32 | +0.08 [-0.21, +0.35] | -27.5% | -16.8% (2000) | 66% | 3.0 |
| ma200 | monthly | 6.4% | 11.7% | 0.59 | +0.34 [+0.05, +0.62] | -17.9% | -9.8% (2015) | 70% | 0.9 |
| ma10m | monthly | 5.5% | 11.9% | 0.51 | +0.27 [-0.03, +0.57] | -21.3% | -9.8% (2015) | 71% | 1.1 |
| tsm3 | monthly | 3.0% | 11.0% | 0.33 | +0.08 [-0.23, +0.36] | -23.5% | -13.7% (2000) | 62% | 2.9 |
| tsm6 | monthly | 4.9% | 11.9% | 0.46 | +0.22 [-0.09, +0.50] | -21.8% | -9.0% (2018) | 69% | 1.6 |
| tsm12 | monthly | 3.8% | 12.4% | 0.37 | +0.12 [-0.20, +0.40] | -23.4% | -10.5% (2000) | 72% | 0.8 |
| vt15 | monthly | 3.3% | 14.0% | 0.30 | +0.06 [-0.06, +0.17] | -43.6% | -22.0% (2008) | 87% | 6.1 |
| dd10 | monthly | 5.5% | 12.6% | 0.49 | +0.25 [-0.05, +0.51] | -19.8% | -10.5% (2000) | 76% | 0.5 |
| dd20 | monthly | 3.2% | 15.2% | 0.28 | +0.04 [-0.17, +0.23] | -49.3% | -23.0% (2002) | 89% | 0.4 |
| vote3 | monthly | 6.1% | 11.9% | 0.56 | +0.32 [+0.02, +0.59] | -18.3% | -9.8% (2015) | 72% | 1.2 |
| ma200_vt | monthly | 5.8% | 10.9% | 0.57 | +0.32 [+0.03, +0.60] | -17.8% | -10.0% (2015) | 68% | 3.2 |

VAL cross-checks on other assets. These are total-return series, and the same 24 variants
appear in `timing_tables_val_SPY.md` and `timing_tables_val_QQQ.md`.

| rule | SPY CAGR | SPY Sharpe | SPY maxDD | QQQ CAGR | QQQ Sharpe | QQQ maxDD |
|---|---|---|---|---|---|---|
| buy_hold | 4.8% | 0.34 | -55.2% | 2.9% | 0.24 | -81.2% |
| ma200 monthly | 7.4% | 0.65 | -21.1% | 5.8% | 0.42 | -38.2% |
| ma10m monthly | 6.4% | 0.57 | -21.1% | 5.9% | 0.43 | -38.2% |
| vote3 monthly | 7.4% | 0.64 | -21.1% | 6.6% | 0.46 | -38.2% |
| ma200_vt monthly | 6.8% | 0.65 | -18.1% | 5.9% | 0.52 | -26.7% |
| ma200_vt weekly | 4.1% | 0.44 | -20.7% | 5.0% | 0.48 | -24.6% |
| vt15 weekly | 4.9% | 0.42 | -42.1% | 6.9% | 0.53 | -40.9% |
| dd10 monthly | 6.1% | 0.53 | -22.5% | 6.3% | 0.48 | -25.3% |
| tsm12 monthly | 6.4% | 0.54 | -19.5% | 5.9% | 0.40 | -41.0% |

The QQQ window is 2000-04-04..2018-12-31, because QQQ starts in 1999-03 and needs a year of
warm-up. Vol targeting is the only rule whose QQQ CI excludes zero: weekly +0.29 [+0.11, +0.41]
and monthly +0.28 [+0.10, +0.41]. It cuts exposure to tech's very high vol.

### Selection and the "real" test (`timing_selection_val.csv`)

- **Pre-registered DEV winner: `ma200_vt` weekly** (DEV Sharpe 0.60 vs 0.40, CI [+0.02, +0.39]).
  - On VAL: Sharpe 0.31 vs 0.24, CI [-0.27, +0.38], so the Sharpe gain is not significant.
    Max drawdown is -19.0% vs -56.8%, CAGR 2.7% vs 2.9%, and it switches 12 times a year.
  - After 1950 on DEV: Sharpe +0.01 and CAGR 7.1% vs 9.4%.
  - **Verdict: it passes the "real" test as a drawdown cutter, but the Sharpe gain did not
    replicate at a meaningful size.**
- **Variants that pass "real"** (dSharpe > 0 and shallower maxDD on DEV, DEV post-1950 and VAL):
  `ma200_vt` weekly, `ma200_vt` monthly, `ma200` monthly, `ma200` weekly, `ma10m` monthly,
  `dd20` monthly and `vt15` weekly. All 24 variants cut max drawdown on all three periods.
  The 17 that fail do so on the Sharpe leg, mostly after 1950. The post-1950 Sharpe gains of
  the passers are tiny, between +0.000 and +0.035.
- **Return** (CAGR minus B&H): positive on both DEV and VAL only for `ma200` monthly
  (+0.4 / +3.5 points), `dd10` weekly (+0.1 / +1.0), `vote3` (+0.1 to +0.2 / +0.3 to +3.3),
  and `dd20` monthly (+0.6 / +0.3). After 1950, **every one of the 24** loses 0.5 to 4.9
  points a year against B&H.
- **Check frequency is a fragility flag.** `ma200` checked monthly makes +6.4% a year on VAL,
  but checked weekly it makes +2.7%. That is the same signal with a different reading day:
  the 2000s were 5.8% vs 0.7% a year. Whipsaw around the average in sideways markets
  (2000-03, 2011, 2015-16) accounts for the gap. On DEV the two frequencies were close.
  Any single headline number here therefore carries several points of path luck.

### By decade (CAGR, `timing_decades_*.csv`)

| decade | B&H | ma200 mo | ma200 wk | ma10m mo | ma200_vt mo | ma200_vt wk |
|---|---|---|---|---|---|---|
| 1920s (1929) | -18.7% | -12.7% | -13.3% | -12.7% | -9.6% | -3.8% |
| 1930s | -5.3% | +3.1% | +2.4% | +2.0% | +2.1% | +1.2% |
| 1940s | +3.1% | +2.8% | +4.5% | +2.5% | +2.5% | +4.4% |
| 1950s | +13.6% | +13.2% | +11.4% | +12.9% | +12.6% | +11.0% |
| 1960s | +4.4% | +3.4% | +3.8% | +3.2% | +3.3% | +3.7% |
| 1970s | +1.6% | +3.5% | +3.0% | +3.5% | +3.2% | +2.8% |
| 1980s | +12.5% | +10.8% | +9.6% | +9.6% | +10.3% | +9.6% |
| 1990s | +15.3% | +9.4% | +9.5% | +9.4% | +8.2% | +8.6% |
| 2000s (VAL) | -2.7% | +5.8% | +0.7% | +5.8% | +4.9% | +0.6% |
| 2010s (VAL) | +9.4% | +7.0% | +5.0% | +5.2% | +6.7% | +5.0% |

The pattern is the same every time. The rules win the bear decades (1930s, 1970s, 2000s) and
lose 0.4 to 6 points a year in the bull decades (1950s, 1980s, 1990s, 2010s). Within-decade
max drawdown improves sharply in the bear decades (1930s: -83% to about -42%; 1970s: -48% to
-16%; 2000s: -57% to -13%), but barely at all in the 1980s (-34% to -33%, because of 1987)
or the 1990s and 2010s.

## Crashes and rebounds

Each cell shows the return from peak to trough, then over the 252 trading days after the
trough (`timing_crashes_*.csv`).

| rule | 1929-32 | 1937-38 | 1973-74 | 1987 | **2000-02** | **2007-09** |
|---|---|---|---|---|---|---|
| buy_hold | -86% / +138% | -54% / +30% | -48% / +35% | -34% / +21% | -49% / +34% | -57% / +69% |
| ma200 monthly | -28% / -17% | -12% / -6% | -8% / +6% | -24% / -0% | -11% / +13% | -6% / +23% |
| ma200 weekly | -31% / -14% | -17% / +6% | -13% / +8% | -33% / -1% | -19% / +14% | -12% / +15% |
| ma10m monthly | -28% / -17% | -12% / -6% | -8% / +6% | -24% / -0% | -11% / +13% | -6% / +23% |
| ma200_vt monthly | -22% / -7% | -11% / -5% | -8% / +3% | -23% / +1% | -10% / +11% | -6% / +17% |
| ma200_vt weekly | -18% / -6% | -16% / -1% | -13% / +5% | -29% / -1% | -16% / +13% | -11% / +11% |
| dd10 monthly | -28% / +7% | -12% / -18% | -25% / -6% | -24% / -3% | -14% / +7% | -11% / +11% |
| tsm12 monthly | -46% / +24% | -27% / -16% | -16% / -10% | -31% / -3% | -19% / +6% | -11% / +9% |
| vt15 weekly | -59% / +30% | -36% / +8% | -41% / +23% | -29% / +10% | -44% / +22% | -34% / +30% |
| vote3 monthly | -28% / -1% | -19% / -10% | -10% / +6% | -24% / -5% | -11% / +13% | -6% / +23% |

Peaks and troughs:

| crash | peak | trough |
|---|---|---|
| 2000-02 | 2000-03-24 | 2002-10-09 |
| 2007-09 | 2007-10-09 | 2009-03-09 |
| 1929-32 | 1929-09-16 | 1932-06-01 |
| 1973-74 | 1973-01-11 | 1974-10-03 |
| 1987 | 1987-08-25 | 1987-12-04 |

`ma200` monthly in the two VAL crashes:
- **2000-02:** it avoided about 78% of the fall (-11% vs -49%) and captured about 38% of the
  first-year rebound (+13% vs +34%).
- **2007-09:** it avoided about 89% of the fall (-6% vs -57%) and captured about 33% of the
  rebound (+23% vs +69%).

The trade-off is structural. The rules re-enter months after the bottom, so they always miss
the sharpest part of a recovery. In 1932-33 they lost money while the market rose 138%.

They also do **not** protect against a fast crash. **In 1987 the trend rules still lost
24-33%**, because the market fell 30% in days from above its average. They then missed the
+21% rebound. A 2020-style crash is likely the same failure mode, but that lies in TEST and
was not computed.

On QQQ, `ma200` monthly took -34% in 2000-02 vs -81% for B&H. Only the vol-scaled versions got
the loss near -12%.

**Dependence on a few crashes.** The DEV Sharpe edge comes from 1929-32. The VAL edge comes
from 2000-02 and 2007-09. Take those three episodes out and every rule has given up
return to B&H in every period tested. That is the honest description of the product: it is
crash insurance with a premium of about 1.5-2 points a year in normal markets, plus
whipsaw losses in sideways ones (`ma200` weekly: 3.8 switches a year on VAL).

## Risk-regime flag

The question here is whether a flag read at a check-day close predicts the **next**
week or month. Outcomes:
- A big drop is a next week worse than -3% or a next month worse than -5%.
- Periods are non-overlapping, running from the flag's close to the next check close.
- The table also shows P(big gain): above +3% for a week, above +5% for a month.

Data: `timing_flags_dev.csv` and `timing_flags_val.csv`.

**Monthly, P(next month < -5%)**

| flag | DEV on (n, drops) | DEV P on / off | VAL on (n, drops) | VAL P on / off | P(next month > +5%) on / off, DEV / VAL |
|---|---|---|---|---|---|
| all months | 851, 92 | 10.8% | 227, 27 | 11.9% | 15.0% / 11.0% |
| below 200-day MA | 294, 53 | **18.0% / 7.0%** | 68, 20 | **29.4% / 4.4%** | 17.7/13.6 ; 19.1/7.5 |
| below 10-month MA | 291, 52 | 17.9% / 7.1% | 66, 19 | 28.8% / 5.0% | 18.2/13.4 ; 19.7/7.5 |
| 12-mo return < 0 | 274, 47 | 17.2% / 7.8% | 63, 15 | 23.8% / 7.3% | 16.8/14.2 ; 20.6/7.3 |
| 3-mo return < 0 | 318, 51 | 16.0% / 7.7% | 86, 20 | 23.3% / 5.0% | 18.6/12.9 ; 17.4/7.1 |
| >10% below 52-wk high | 257, 50 | 19.5% / 7.1% | 55, 17 | 30.9% / 5.8% | 20.6/12.6 ; 23.6/7.0 |
| EWMA vol > 20% | 158, 42 | 26.6% / 7.2% | 57, 12 | 21.1% / 8.8% | 27.2/12.3 ; 26.3/5.9 |
| EWMA vol > 30% | 74, 23 | 31.1% / 8.9% | 17, 6 | 35.3% / 10.0% | 33.8/13.3 ; 35.3/9.0 |
| below 200d AND vol > 20% | 97, 32 | 33.0% / 8.0% | 46, 12 | 26.1% / 8.3% | 28.9/13.3 ; 23.9/7.7 |
| vote3 off | 291, 56 | 19.2% / 6.4% | 64, 20 | 31.2% / 4.3% | 18.2/13.4 ; 20.3/7.4 |

The mean next-month return under "below 200-day" was +0.2% vs +0.9% when above on DEV, and
-0.9% vs +0.9% on VAL. Next-month standard deviation roughly doubles under the flag
(7.5% vs 4.4% on DEV, 6.0% vs 3.0% on VAL).

**Weekly, P(next week < -3%)**

| flag | DEV P on / off | VAL P on / off |
|---|---|---|
| all weeks | 7.5% (3,699 weeks, 278 drops) | 8.2% (991 weeks, 81 drops) |
| below 200-day | 14.0% / 4.3% | 16.3% / 4.4% |
| >10% below high | 16.5% / 3.8% | 16.0% / 5.4% |
| vol > 20% | 21.4% / 4.4% | 14.6% / 6.0% |
| vol > 30% | 25.4% / 5.9% | 27.1% / 6.7% |

The weekly flags also raise P(week > +3%), in about the same proportion. Mean weekly
returns barely differ.

**Reading.** Every flag works in the same direction on DEV and VAL. It raises the chance of a
big down period by 2.5 to 7 times, and the signal is strong in both periods. But it is
mostly a **volatility** flag: the chance of a big up month rises almost as much. The
directional part is small, with mean returns lower under the flag by 0.7-1.8 points a month,
and in VAL it rests on two bear markets. The VAL counts are small: 20 drop months under the
200-day flag, clustered in roughly three episodes (2000-02, 2008-09, 2011/2015-16). Read
those percentages as having wide error bars.

## What it means

- **No rule here makes more money than just holding the S&P over a long run of normal
  years.** In the bull decades (1950s, 80s, 90s, 2010s) these rules earned 0.4-6 points a
  year less than buy-and-hold.
- **What the simple trend rule does is keep you out of the big, slow bear markets.**
  "Hold the S&P when its month-end close is above its 200-day average, otherwise cash."
  - 2000-02 and 2008-09: it lost 11% and 6% where buy-and-hold lost 49% and 57%.
  - Its worst drawdown was about -18% in 2000-2018, against -57%.
  - It switches about once a year.
  - It gets back in late, so expect to miss roughly two-thirds of the first year of a rebound.
- **It will not save you from a sudden crash.** In 1987 it still lost about 24%. Use it as a
  "how much risk am I running" dial, not as a market call.
- **As a warning light it is genuinely useful.** When the S&P is below its 200-day average,
  a month worse than -5% happened 18% of the time (1929-99) and 29% of the time (2000-18),
  against 4-7% when above. Big up months also get more common, so read it as "rough water
  ahead, size down", not "the market is going down".
- **For individual stocks** it describes risk, not an entry signal: the flag marks
  turbulent periods.
- **The single rule worth keeping:** S&P 500 vs its 200-day moving average, checked
  at the month-end close. It is a drawdown/risk filter, not an edge, and it carries about
  1.4 points a year of expected drag in normal markets (post-1950 DEV: 8.0% vs 9.4%).
- **Caveat on that pick.** It ranked 4th on DEV by Sharpe. The pre-registered DEV winner was
  `ma200_vt` weekly, which also passes but trades 12 times a year and replicated more weakly
  on VAL. I prefer the plain 200-day monthly rule for three reasons: it is the simplest
  classic, it trades least, and it passes every check. Choosing it with VAL in view is a
  mild selection effect; TEST should confirm it.

## Not verified

- **TEST (2019+) has not been run.** This includes the 2020 fast crash and the 2022 bear,
  which are the obvious next checks. Coordinator command:
  `python research/timing.py --split test --coordinator`.
- **Dividends and cash yield are not modelled for ^GSPC.** Both sides lose about 2-4% a year
  of dividends while invested, and cash earning 0 understates the timing rules (T-bills paid
  3-5% for long stretches). The SPY total-return cross-check shows the same ranking.
  Taxes are ignored: every exit is a taxable sale in a taxable account.
- **Slippage beyond 5 bps is not modelled,** and neither are intraday or overnight gaps
  between the decision close and the trade close. Trading at the next close is modelled;
  gaps inside that day are not.
- **The bootstrap CIs are approximate.** Blocks are 126 days and there are 500 reps. Crash
  clustering makes them optimistic about how much independent evidence exists.
- **Only one parameter set per rule was tried** (200d, 15% vol target, 10/20% drawdown).
  That is deliberate, to avoid fishing, but it means no sensitivity map. The weekly vs monthly
  gap shows real path sensitivity.
- **The risk-flag outcomes use the flag close to the next check close,** with no one-day
  trading lag. They answer "is risk higher", not "can I trade it".
- **The pre-1957 ^GSPC is the S&P 90-stock composite as reported by Yahoo.** Data quality
  before 1950 was not audited beyond basic checks.
