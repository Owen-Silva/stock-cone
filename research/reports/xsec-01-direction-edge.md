# xsec-01: is there a cross-sectional direction edge?

> **Not financial advice.** This is a research log from building an educational tool. Backtest numbers are hypothetical, carry survivorship bias, and say nothing certain about the future. Nothing here is a recommendation to buy, sell or hold anything.

Lane cone-xsec, 2026-10-03/04. Code: `research/backtest.py`, `research/xsec.py`. Data: `research/results/xsec_*`. TEST (2019+, all tickers) has **not** been opened.

**Short answer.** One family survives the rule: 12-1 month momentum, plus the trend signals and models built from it. Its sign agrees on DEV and VAL, and its VAL IC t is about 3.

It is **not a usable trading signal by itself.** After costs, the top-quintile excess over an equal-weight basket of the same stocks is:
- **DEV:** +2 to +3.5%/yr, not significant (t about 1).
- **VAL:** +11 to +12%/yr, but inflated by survivorship.

A deliberately meaningless "long-run drift" signal, which buys whatever has gone up most since it listed, earns **+13.4%/yr on VAL** (t 2.8) and nothing on DEV. With that drift ranking removed, momentum's VAL excess falls to +5 to +7.5%/yr, not significant.

Realistic estimate: **0 to 4%/yr** over an equal-weight basket of the same stocks, before taxes, with 10-15%/yr tracking error.

## Method

**Universe**
- TRAIN tickers only (md5 rule, `research.splits`): 71 tickers, of which 52 have prices before 2019.
- The panel is cut at 2018-12-31 before anything is computed, and the code asserts this. No TEST ticker is loaded on DEV or VAL.
- Prices come straight from the frozen `data/prices` cache. `engine.load_prices` is never called, so nothing can download.

**Splits**
- DEV uses signal dates before 2012-01-01. VAL uses 2012-01-01 to 2018-12-31.
- Every exit price must also fall inside the split, so DEV never touches 2012 prices and VAL never touches 2019 prices.
- A rebalance date counts only when at least 20 names are eligible.
- In practice DEV runs 1987-06 to 2011-12 with about 30 names (about 6 per quintile). VAL runs 2012-01 to 2018-12 with about 41 names (about 8 per quintile).

**Eligibility on the signal date**
- A real close that day, at least 252 days of history, 63-day median dollar volume of at least $2M, and no more than 10% zero-volume days.
- *Deviation from the brief:* no "$5 price" filter. The cached closes are split- and dividend-adjusted, so a $5 test would drop NVDA for all of 2012-2018 (adjusted about $0.33, real about $13) and most large caps before 2000.
- Dollar volume does not change with splits, so it does the tiny-name job. It removes the illiquid shells CLSK and OLB (74% and 92% zero-volume days).

**Backtest**
- The signal uses data up to the close of day s. The trade happens at the next close, s+1 (lag 1, the honest case). A same-close variant (lag 0) is also reported.
- Rebalancing is every 5 trading days (weekly) or every 21 (monthly).
- Portfolio: top quintile, equal weight, long only.
- Benchmarks:
  - The equal-weight universe: same eligible names, same dates, its own rebalancing costs.
  - SPY buy-and-hold. Before 1993 this is the ^GSPC price index, which has no dividends.
- Long-short spread: top minus bottom quintile, with no financing or borrow costs.
- Costs: 10 bps per side on traded notional. Turnover is measured against drifted weights, and the first buy-in counts.

**Metrics**
- Geometric annual return, volatility, Sharpe (risk-free rate = 0), max drawdown, one-way turnover per year, monthly hit rate against EW, and calendar-year results.
- IC is the Spearman correlation between the signal and the next period's return on each rebalance date. Its t-stat is Newey-West (Bartlett, automatic lag).
- A second t-stat uses daily-sampled overlapping forward returns, with Newey-West lag = horizon + automatic lag.

**Signals** (all use data up to day t only)
- 12-1, 6-1 and 3-1 month momentum.
- 1-week and 1-month reversal.
- Low 63-day volatility.
- Closeness to the 52-week high.
- Price against the 50- and 200-day moving averages, and the 50/200 cross.
- Volume shock (5-day over 63-day volume), raw and signed by the 5-day return.
- Risk-adjusted 12-1 momentum.
- Low beta.
- Beta-neutral versions: residual 12-1 and 6-1 momentum, beta-neutral 1-week and 1-month reversal, low idiosyncratic volatility.
- MAX: the largest daily return in the last 21 days, sign reversed.

Each signal is oriented so that "higher = should beat the others".

**Combinations, fit on DEV and frozen** (`xsec_frozen.json`)
- `combo_dev` averages the rank scores of the signals with DEV |IC t| >= 2.
  - Weekly: mom_12_1, rmom_12_1, resid_mom_12_1, rev_1w, rev_1w_bn.
  - Monthly: no signal qualified, so there is no monthly combo.
- `ridge_dev` and `logit_dev` are numpy ridge and L2-logistic models on all 20 rank scores. Ridge predicts the next period's rank; logistic predicts beat-the-median.
  - Lambda was chosen by 5-fold contiguous-date CV inside DEV with a one-period purge. The smallest alpha in the grid won.
- `combo_all_prior` averages all 20 signals with their prior signs and is fit on nothing.

**Rule for "real":** the IC sign agrees on DEV and VAL, and the VAL Newey-West IC t is above 2. Candidates are ranked on VAL only.

## Sanity checks

| check | weekly DEV | weekly VAL | monthly DEV | monthly VAL |
|---|---|---|---|---|
| (a) random signal, 20 seeds: mean IC | -0.000 | 0.001 | -0.002 | 0.006 |
| random: mean IC t | -0.04 | 0.18 | -0.23 | 0.45 |
| random: excess vs EW, gross | -1.1% | -0.7% | -0.8% | +1.7% |
| random: excess vs EW, net | -10.1% | -9.6% | -3.0% | -0.6% |
| random: one-way turnover/yr | 40.5x | 40.6x | 9.7x | 9.7x |
| (b) look-ahead signal: IC | 1.00 | 1.00 | 1.00 | 1.00 |
| look-ahead: top-quintile return/yr | 2463% | 1041% | 403% | 281% |

- **(a) Random signal.** IC is about 0 and the gross excess is about 0. The small negative weekly gross figure is volatility drag from holding 6-8 names. Net of costs, the excess matches what about 40x/yr turnover costs (about 8%/yr). So a random signal earns the EW return minus costs, as it should.
- **(b) Look-ahead signal.** The cheating signal makes hundreds to thousands of percent a year, so the pipeline transmits information. The best real top quintile makes about 29%/yr on VAL, orders of magnitude lower.
- **(c) Shift and timing.**
  - Making each signal one day staler moves IC by at most 0.02, and that maximum is 1-week reversal, which is expected because short-term reversal decays within days.
  - The momentum family moves by at most 0.003.
  - Trading at the same close instead of the next close moves momentum IC by less than 0.007. No sign of leakage.
- **Is t > 2 strict enough? No.** 30 uninformative but persistent signals (a random score per stock, redrawn yearly) give |t| > 2 in 10-13% of draws on VAL, not the nominal 5%; the maximum was 3.9. The cause is that stocks in a hand-picked survivor universe differ a lot in average return. Ordinary random signals (redrawn daily) behave normally (VAL t-stat sd 0.73-0.77). **On VAL, a t of about 3 is roughly a 1-in-30 placebo outcome.**
- **Survivorship yardstick** (`xsec_diag.json`). A stock's own average return since it listed has no reason to predict anything in an unbiased universe:

| | DEV IC t | DEV excess/yr | VAL IC t | VAL excess/yr |
|---|---|---|---|---|
| drift placebo, weekly | 0.38 | -3.5% | 3.59 | +13.4% |
| drift placebo, monthly | 0.32 | -0.8% | 3.47 | +13.5% |
| mom_12_1 with drift regressed out, weekly | 3.20 | -2.6% | 2.19 | +5.2% |
| mom_12_1 with drift regressed out, monthly | 0.96 | -2.9% | 1.15 | +7.5% |

- The drift placebo beats every real signal on VAL and is zero on DEV.
- Its rank correlation with 12-1 momentum is 0.16 on DEV but 0.44 on VAL.
- So much of momentum's VAL result is "own the stocks we now know became winners". The universe was picked in 2026, and VAL is closer to 2026 than DEV, so the hindsight is stronger there.
- The benchmarks show the same bias. The equal-weight survivor basket made 16.8%/yr on DEV against 7.4% for SPY (price-only before 1993), and 15.9% against 12.1% on VAL. Nobody could have bought this list in 1987.

## Results (trade at the next close, 10 bps per side)

**Columns**
- **IC, t, t(d):** IC mean; its Newey-West t on rebalance dates; its t from daily overlapping returns.
- **top/yr:** top-quintile return per year, net of costs.
- **exc/yr, exc t:** excess over EW per year (geometric, net) and the Newey-West t of the per-period excess.
- **Sh, MDD:** top-quintile Sharpe and max drawdown.
- **TO:** one-way turnover per year.
- **L/S/yr, L/S t:** long-short return per year and its t.

The (EW universe) and (SPY buy-hold) rows show those portfolios. `ridge_dev`, `logit_dev` and `combo_dev` are **in-sample on DEV**, so read only their VAL rows. Lag-0 rows and everything else are in `xsec_all.csv`; calendar years are in `xsec_years.csv`.

### DEV, weekly (1987-06..2011-12, 1068 periods, ~30 names)

| signal | IC | t | t(d) | top/yr | exc/yr | exc t | Sh | MDD | TO | L/S/yr | L/S t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mom_12_1 | 0.029 | 3.45 | 3.58 | 18.7% | +2.0% | 0.90 | 0.72 | -63% | 7.9x | -7.1% | -0.02 |
| mom_6_1 | 0.007 | 0.86 | 1.21 | 12.7% | -4.0% | -0.40 | 0.55 | -65% | 10.5x | -18.5% | -1.97 |
| mom_3_1 | -0.001 | -0.07 | 0.17 | 11.5% | -5.3% | -0.64 | 0.51 | -59% | 15.1x | -17.9% | -2.04 |
| rev_1w | 0.041 | 4.81 | 5.30 | 16.9% | +0.2% | 0.71 | 0.64 | -70% | 40.4x | +2.6% | 1.05 |
| rev_1m | -0.001 | -0.18 | -0.71 | 3.0% | -13.7% | -2.26 | 0.26 | -72% | 20.1x | -13.1% | -1.41 |
| lowvol_63 | 0.000 | 0.03 | 0.26 | 12.5% | -4.3% | -1.55 | 0.86 | -38% | 4.4x | -21.0% | -1.85 |
| high52 | 0.004 | 0.45 | 1.09 | 4.6% | -12.2% | -3.39 | 0.32 | -58% | 17.4x | -20.3% | -2.07 |
| trend_50 | -0.005 | -0.61 | 0.23 | 11.8% | -4.9% | -0.90 | 0.55 | -64% | 16.9x | -10.9% | -0.83 |
| trend_200 | 0.010 | 1.15 | 1.75 | 11.8% | -4.9% | -0.67 | 0.54 | -61% | 9.6x | -15.1% | -1.08 |
| ma_cross | 0.016 | 1.91 | 2.11 | 17.0% | +0.2% | 0.51 | 0.68 | -61% | 4.3x | -9.5% | -0.37 |
| volshock | -0.011 | -1.64 | -1.29 | 4.3% | -12.4% | -2.79 | 0.29 | -63% | 32.6x | -20.2% | -3.30 |
| volshock_signed | -0.013 | -1.82 | -2.04 | 1.2% | -15.5% | -4.33 | 0.18 | -72% | 35.4x | -23.5% | -3.85 |
| rmom_12_1 | 0.029 | 3.63 | 3.94 | 17.6% | +0.8% | 0.43 | 0.76 | -55% | 7.8x | -8.9% | -0.50 |
| lowbeta | -0.016 | -1.71 | -1.55 | 3.8% | -12.9% | -2.89 | 0.29 | -52% | 3.0x | -23.6% | -2.27 |
| resid_mom_12_1 | 0.025 | 3.28 | 3.33 | 20.2% | +3.4% | 1.28 | 0.77 | -60% | 8.3x | -6.4% | -0.12 |
| resid_mom_6_1 | 0.002 | 0.30 | 0.76 | 13.4% | -3.4% | -0.25 | 0.57 | -64% | 10.6x | -19.3% | -2.44 |
| rev_1w_bn | 0.034 | 4.23 | 4.21 | 16.0% | -0.8% | 0.51 | 0.62 | -69% | 40.0x | -0.9% | 0.53 |
| rev_1m_bn | -0.002 | -0.25 | -1.19 | 1.7% | -15.0% | -2.74 | 0.22 | -75% | 20.1x | -15.9% | -2.11 |
| lowivol_63 | -0.000 | -0.02 | 0.32 | 12.5% | -4.2% | -1.52 | 0.81 | -38% | 4.9x | -18.5% | -1.63 |
| max_21 | -0.000 | -0.02 | 0.10 | 12.3% | -4.5% | -1.62 | 0.76 | -49% | 13.4x | -20.1% | -2.43 |
| (EW universe) | | | | 16.8% | | | 0.80 | -57% | | | |
| (SPY buy-hold) | | | | 7.4% | | | 0.48 | | | | |
| combo_dev (in-sample) | 0.044 | 5.94 | 5.67 | 20.6% | +3.9% | 1.28 | 0.80 | -55% | 27.4x | -1.0% | 0.59 |
| ridge_dev (in-sample) | 0.063 | 7.91 | 8.69 | 28.3% | +11.6% | 3.16 | 0.97 | -69% | 35.0x | +15.3% | 2.95 |
| logit_dev (in-sample) | 0.059 | 7.53 | 8.00 | 24.9% | +8.2% | 2.54 | 0.85 | -76% | 32.9x | +12.6% | 2.54 |
| combo_all_prior | 0.019 | 2.30 | 2.75 | 12.3% | -4.5% | -1.11 | 0.61 | -54% | 15.9x | -19.2% | -2.06 |

### VAL, weekly (2012-01..2018-12, 351 periods, ~41 names)

| signal | IC | t | t(d) | top/yr | exc/yr | exc t | Sh | MDD | TO | L/S/yr | L/S t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mom_12_1 | 0.049 | 2.97 | 3.33 | 26.8% | +11.0% | 2.08 | 1.18 | -29% | 6.2x | +7.2% | 0.96 |
| mom_6_1 | 0.034 | 2.56 | 2.81 | 18.7% | +2.8% | 0.70 | 0.90 | -30% | 8.8x | +6.0% | 1.00 |
| mom_3_1 | 0.008 | 0.70 | 0.99 | 14.2% | -1.7% | -0.08 | 0.74 | -39% | 14.1x | -3.3% | -0.02 |
| rev_1w | 0.004 | 0.27 | 0.23 | 0.6% | -15.3% | -2.31 | 0.14 | -45% | 38.2x | -23.8% | -2.91 |
| rev_1m | -0.014 | -1.09 | -1.20 | 1.0% | -14.9% | -2.21 | 0.16 | -44% | 19.0x | -19.3% | -2.19 |
| lowvol_63 | 0.015 | 0.85 | 0.90 | 13.1% | -2.7% | -0.70 | 1.19 | -11% | 5.6x | -6.8% | -0.22 |
| high52 | 0.034 | 2.11 | 2.14 | 16.2% | +0.3% | 0.03 | 1.17 | -12% | 18.8x | +1.4% | 0.47 |
| trend_50 | 0.014 | 1.06 | 1.11 | 19.7% | +3.8% | 0.77 | 1.03 | -34% | 14.8x | +5.5% | 0.87 |
| trend_200 | 0.038 | 2.81 | 2.79 | 27.5% | +11.7% | 2.06 | 1.30 | -26% | 7.7x | +13.7% | 1.65 |
| ma_cross | 0.037 | 2.63 | 2.76 | 29.6% | +13.7% | 2.33 | 1.31 | -26% | 3.7x | +11.8% | 1.48 |
| volshock | -0.002 | -0.21 | -0.35 | 9.7% | -6.2% | -1.04 | 0.56 | -41% | 32.5x | -7.6% | -0.82 |
| volshock_signed | -0.017 | -1.64 | -1.16 | 4.8% | -11.1% | -2.58 | 0.36 | -42% | 35.4x | -19.8% | -3.42 |
| rmom_12_1 | 0.047 | 2.83 | 2.95 | 20.9% | +5.1% | 1.11 | 1.18 | -23% | 6.9x | +5.8% | 0.86 |
| lowbeta | -0.016 | -0.89 | -1.29 | 10.0% | -5.9% | -0.99 | 0.70 | -28% | 2.4x | -13.6% | -1.17 |
| resid_mom_12_1 | 0.042 | 2.66 | 2.93 | 26.5% | +10.6% | 1.94 | 1.19 | -31% | 6.1x | +4.9% | 0.78 |
| resid_mom_6_1 | 0.033 | 2.48 | 2.58 | 22.2% | +6.3% | 1.30 | 1.06 | -29% | 9.2x | +7.3% | 1.13 |
| rev_1w_bn | 0.003 | 0.23 | 0.12 | 1.2% | -14.7% | -2.16 | 0.17 | -48% | 38.0x | -22.4% | -2.55 |
| rev_1m_bn | -0.013 | -1.05 | -1.01 | 4.9% | -11.0% | -1.51 | 0.32 | -41% | 19.1x | -16.3% | -1.87 |
| lowivol_63 | 0.012 | 0.70 | 0.80 | 14.9% | -1.0% | -0.34 | 1.23 | -10% | 5.5x | -4.6% | -0.06 |
| max_21 | 0.006 | 0.44 | 0.64 | 12.4% | -3.5% | -0.95 | 1.05 | -12% | 14.1x | -6.9% | -0.43 |
| (EW universe) | | | | 15.9% | | | 1.08 | -19% | | | |
| (SPY buy-hold) | | | | 12.1% | | | 1.00 | | | | |
| combo_dev | 0.041 | 3.03 | 3.17 | 18.8% | +2.9% | 0.75 | 1.01 | -26% | 26.1x | -2.6% | -0.01 |
| ridge_dev | 0.040 | 3.06 | 3.60 | 20.0% | +4.1% | 1.02 | 1.05 | -23% | 34.3x | -0.2% | 0.26 |
| logit_dev | 0.031 | 2.39 | 3.05 | 19.8% | +4.0% | 1.06 | 1.03 | -21% | 31.9x | -6.9% | -0.57 |
| combo_all_prior | 0.041 | 2.73 | 2.65 | 19.0% | +3.1% | 0.77 | 1.20 | -21% | 14.7x | +9.2% | 1.24 |

### DEV, monthly (1987-06..2011-12, 254 periods)

| signal | IC | t | t(d) | top/yr | exc/yr | exc t | Sh | MDD | TO | L/S/yr | L/S t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mom_12_1 | 0.023 | 1.32 | 2.10 | 20.4% | +3.5% | 1.09 | 0.80 | -52% | 3.5x | -2.8% | 0.45 |
| mom_6_1 | -0.017 | -1.08 | -0.60 | 14.0% | -3.0% | -0.21 | 0.61 | -55% | 4.8x | -14.2% | -1.48 |
| mom_3_1 | -0.020 | -1.30 | -0.57 | 12.8% | -4.2% | -0.47 | 0.57 | -54% | 7.0x | -12.5% | -1.41 |
| rev_1w | -0.008 | -0.45 | 0.35 | 8.1% | -8.9% | -1.27 | 0.41 | -68% | 9.2x | -9.7% | -0.85 |
| rev_1m | -0.015 | -1.17 | -1.13 | 10.7% | -6.3% | -0.84 | 0.48 | -66% | 9.4x | -9.8% | -1.45 |
| lowvol_63 | -0.007 | -0.32 | -0.68 | 11.2% | -5.8% | -1.66 | 0.88 | -31% | 2.3x | -21.4% | -1.75 |
| high52 | 0.002 | 0.13 | 0.39 | 12.5% | -4.4% | -1.39 | 0.72 | -50% | 6.3x | -14.6% | -1.56 |
| trend_50 | 0.003 | 0.24 | 0.22 | 17.3% | +0.3% | 0.47 | 0.77 | -63% | 7.4x | -2.5% | 0.14 |
| trend_200 | 0.006 | 0.41 | 0.90 | 17.8% | +0.8% | 0.56 | 0.76 | -55% | 4.3x | -5.4% | -0.04 |
| ma_cross | -0.004 | -0.26 | 0.47 | 16.6% | -0.4% | 0.35 | 0.69 | -53% | 3.2x | -8.7% | -0.54 |
| volshock | -0.018 | -1.41 | -1.85 | 16.0% | -1.0% | 0.15 | 0.70 | -54% | 9.5x | -9.7% | -1.43 |
| volshock_signed | 0.003 | 0.23 | 0.90 | 14.1% | -2.8% | -0.65 | 0.69 | -54% | 9.5x | -3.6% | -0.20 |
| rmom_12_1 | 0.026 | 1.71 | 2.38 | 21.6% | +4.6% | 1.36 | 0.96 | -42% | 3.5x | +0.3% | 0.75 |
| lowbeta | -0.026 | -1.25 | -1.75 | 6.2% | -10.8% | -2.45 | 0.41 | -47% | 1.5x | -19.2% | -1.79 |
| resid_mom_12_1 | 0.020 | 1.25 | 1.87 | 24.0% | +7.0% | 1.88 | 0.94 | -49% | 3.7x | -0.7% | 0.64 |
| resid_mom_6_1 | -0.028 | -1.90 | -1.18 | 14.4% | -2.6% | -0.12 | 0.62 | -55% | 4.8x | -15.5% | -1.88 |
| rev_1w_bn | -0.007 | -0.41 | -0.11 | 9.9% | -7.0% | -0.84 | 0.46 | -71% | 9.3x | -9.8% | -0.97 |
| rev_1m_bn | -0.017 | -1.33 | -1.25 | 8.7% | -8.2% | -1.27 | 0.42 | -65% | 9.3x | -11.1% | -1.45 |
| lowivol_63 | -0.004 | -0.19 | -0.56 | 14.3% | -2.6% | -0.98 | 1.02 | -28% | 2.4x | -17.0% | -1.30 |
| max_21 | 0.001 | 0.06 | -0.69 | 14.2% | -2.7% | -0.97 | 0.94 | -40% | 6.7x | -12.0% | -1.05 |
| (EW universe) | | | | 17.0% | | | 0.89 | -49% | | | |
| (SPY buy-hold) | | | | 7.8% | | | 0.57 | | | | |
| ridge_dev (in-sample) | 0.075 | 5.93 | 5.78 | 33.5% | +16.5% | 4.67 | 1.19 | -59% | 7.8x | +19.8% | 4.18 |
| logit_dev (in-sample) | 0.071 | 5.16 | 5.17 | 28.9% | +11.9% | 3.42 | 1.03 | -58% | 7.4x | +17.3% | 3.62 |
| combo_all_prior | -0.008 | -0.46 | 0.23 | 15.2% | -1.8% | -0.37 | 0.77 | -48% | 5.6x | -15.9% | -1.65 |

### VAL, monthly (2012-02..2018-12, 83 periods)

| signal | IC | t | t(d) | top/yr | exc/yr | exc t | Sh | MDD | TO | L/S/yr | L/S t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mom_12_1 | 0.064 | 2.39 | 2.92 | 29.1% | +11.7% | 2.75 | 1.29 | -21% | 2.9x | +11.8% | 1.73 |
| mom_6_1 | 0.047 | 2.27 | 2.44 | 25.9% | +8.5% | 1.57 | 1.15 | -22% | 4.3x | +10.8% | 1.74 |
| mom_3_1 | 0.016 | 0.75 | 1.29 | 17.6% | +0.3% | 0.32 | 0.82 | -43% | 6.7x | -0.4% | 0.26 |
| rev_1w | 0.027 | 0.89 | -1.77 | 13.6% | -3.7% | -0.30 | 0.70 | -33% | 9.0x | -4.2% | -0.09 |
| rev_1m | -0.010 | -0.45 | -1.75 | 8.4% | -9.0% | -1.26 | 0.49 | -38% | 9.3x | -11.3% | -1.23 |
| lowvol_63 | 0.026 | 0.77 | 1.42 | 15.2% | -2.1% | -0.52 | 1.60 | -9% | 2.8x | -5.4% | -0.11 |
| high52 | 0.044 | 1.63 | 2.68 | 20.7% | +3.4% | 0.65 | 1.53 | -9% | 6.8x | +3.4% | 0.78 |
| trend_50 | 0.006 | 0.26 | 1.66 | 18.9% | +1.5% | 0.46 | 0.90 | -32% | 7.3x | +2.0% | 0.53 |
| trend_200 | 0.049 | 2.33 | 3.06 | 28.7% | +11.4% | 2.11 | 1.33 | -18% | 3.8x | +12.1% | 1.76 |
| ma_cross | 0.050 | 2.21 | 2.34 | 30.8% | +13.5% | 2.43 | 1.34 | -26% | 2.7x | +13.5% | 1.94 |
| volshock | -0.004 | -0.24 | 0.23 | 15.8% | -1.6% | -0.14 | 0.87 | -23% | 9.0x | -1.4% | -0.00 |
| volshock_signed | 0.001 | 0.04 | 1.03 | 12.6% | -4.8% | -1.00 | 0.83 | -16% | 8.2x | -11.1% | -1.81 |
| rmom_12_1 | 0.056 | 2.20 | 2.73 | 21.0% | +3.7% | 1.05 | 1.25 | -14% | 3.1x | +6.7% | 1.26 |
| lowbeta | -0.030 | -0.85 | -1.05 | 12.9% | -4.4% | -0.73 | 0.90 | -23% | 1.2x | -14.4% | -1.12 |
| resid_mom_12_1 | 0.050 | 1.93 | 2.54 | 25.4% | +8.0% | 1.97 | 1.13 | -19% | 2.9x | +4.3% | 0.94 |
| resid_mom_6_1 | 0.041 | 1.88 | 2.30 | 25.9% | +8.5% | 1.62 | 1.20 | -24% | 4.6x | +8.5% | 1.49 |
| rev_1w_bn | 0.026 | 0.89 | -1.59 | 14.6% | -2.7% | -0.15 | 0.74 | -32% | 9.0x | -3.2% | 0.02 |
| rev_1m_bn | -0.019 | -0.79 | -1.51 | 7.8% | -9.5% | -1.43 | 0.46 | -36% | 9.3x | -13.4% | -1.51 |
| lowivol_63 | 0.022 | 0.68 | 1.43 | 14.0% | -3.3% | -0.92 | 1.35 | -11% | 3.0x | -5.3% | -0.11 |
| max_21 | 0.022 | 0.75 | 1.46 | 18.7% | +1.4% | 0.25 | 1.73 | -9% | 7.1x | -1.1% | 0.19 |
| (EW universe) | | | | 17.3% | | | 1.28 | -14% | | | |
| (SPY buy-hold) | | | | 13.2% | | | 1.25 | | | | |
| ridge_dev | 0.040 | 1.58 | 1.75 | 22.1% | +4.7% | 1.18 | 1.29 | -13% | 8.1x | +5.6% | 0.87 |
| logit_dev | 0.036 | 1.35 | 1.87 | 22.5% | +5.1% | 1.32 | 1.23 | -16% | 7.5x | +9.1% | 1.30 |
| combo_all_prior | 0.055 | 2.52 | 2.75 | 20.8% | +3.5% | 1.01 | 1.32 | -14% | 5.2x | +10.0% | 1.67 |

**What the tables say**
- **12-1 momentum** (and its risk-adjusted and residual versions) is the only family with a significant IC on both splits at weekly frequency: DEV t 3.3-3.6, VAL t 2.7-3.0.
  - Its money result on DEV is weak: +2.0%/yr weekly (t 0.9), +3.5%/yr monthly (t 1.1). The DEV long-short is about zero.
  - It beat EW in 14 of 23 DEV years (weekly) and 12 of 23 (monthly), and in 5 of 7 and 6 of 7 VAL years.
  - Worst DEV years against EW: 2009 (-33 to -38%, the momentum crash), 1992 and 1996.
- **Trend** (price against the 200-day average, the 50/200 cross) is weak on DEV and good on VAL (+11 to 14%/yr). This is the same winners-kept-winning pattern.
- **1-week reversal** was the strongest DEV signal (IC 0.041, t 4.8) and vanished on VAL (t 0.3). At about 38x/yr turnover, costs eat about 8%/yr. This is the classic decayed, untradeable anomaly.
- **Low vol, low beta, low idiosyncratic vol, MAX, 52-week high and volume shocks** show nothing on DEV. Low beta is negative on both splits, because high-beta names won in this survivor universe.
- **DEV-fit models.**
  - The weekly ridge has the best VAL IC (0.040, t 3.06), but turnover is 34x/yr and it keeps only +4.1%/yr after costs (t 1.0).
  - Its largest weights are 1-week reversal and low vol, the two signals that died on VAL.
  - Its VAL excess swings from -7.4% to +4.1%/yr depending on rebalance weekday, and is -4.1% if it trades at the same close. That is fragile.
  - The monthly models miss the VAL bar (t 1.3-1.6).

**Robustness of the VAL candidates** (`xsec_robust.csv`)
- mom_12_1 weekly holds up across all 5 rebalance weekdays: IC t 2.77-3.48, excess +9.8 to +11.8%/yr.
- Removing the 3 names it held most often (MU, NVDA, ADBE: 17% of portfolio slots) cuts the excess to +6.7%/yr (t 1.35) and the IC t to 2.35.
- The ridge and combo results change sign depending on the rebalance weekday.

## Best candidates (chosen on VAL only; frozen for TEST)

15 signal/frequency pairs pass the rule (listed under `candidates_val` in `xsec_frozen.json`); most are momentum variants. A fixed rule (`primary_for_test`) picks three for TEST:

1. **ridge_dev, weekly:** best VAL IC t of anything (3.06). Low net excess, fragile.
2. **mom_12_1, weekly:** best single signal by VAL IC t (2.97). VAL excess +11.0%/yr, t 2.08.
3. **mom_12_1, monthly:** best VAL net-excess t (2.75). VAL excess +11.7%/yr, turnover 2.9x/yr. This is the only one an individual could realistically run.

**How to run TEST.** `python research/xsec.py --split test --coordinator`.
- It loads every ticker, scores 2019+ with the frozen weights, and never refits.
- It prints the three primaries first, then the other candidates, 20 random signals, the look-ahead signal and **the drift placebo**.
- `--split dryrun` rehearses the same code path on VAL; it reproduced the VAL numbers exactly.

**Warning for reading TEST.** The 2019+ universe contains names that are in it because they soared in 2023-25 (the quantum, nuclear and AI-infra names). Momentum and the drift placebo will both look spectacular there. A momentum result on TEST counts only to the extent that it beats the drift placebo on TEST.

## What it means

- **Nothing here is a money machine.** The only signal that survived honest testing is the most famous anomaly in finance: stocks that rose most over the past year (skipping the last month) tend to keep beating the others a little.
- **How big is "a little"?**
  - In the cleanest test (1987-2011, 25 years), holding the top-momentum fifth of these stocks and rebalancing monthly beat simply holding all of them equally by about 3.5% a year after trading costs. That is not distinguishable from luck.
  - The 2012-2018 figure of about +12%/yr is mostly hindsight, because a meaningless "buy what has gone up since it listed" rule earned even more over the same years.
  - Realistic estimate: 0 to 4% a year over an equal-weight basket of the same stocks, before taxes.
- **The ride is rough.**
  - Against the basket, it loses in about 4 years out of 10.
  - In a crash year it can trail the basket by 30-40% (2009).
  - Tracking error is about 10-15%/yr. With an edge of about 3%, telling skill from luck in a real portfolio would take decades.
- **Costs and taxes.**
  - Weekly versions turn the portfolio over 6-35 times a year, which eats most of the edge even at 10 bps.
  - Monthly momentum turns over about 3x a year. In a taxable account those are mostly short-term gains, which can erase a 3% edge.
- **Research reading, if any.**
  - At most it would be a small tilt within a fixed basket, rebalanced monthly, after costs and taxes.
  - Even that reading holds only if TEST shows momentum beating the drift placebo.
  - Do not use it for concentrated bets or day trading.
  - A Picks view in the app should carry its own live track record and this caveat.

## Not verified

- **TEST not opened.** All conclusions are DEV/VAL only.
- **Survivorship cannot be removed** with this data (no delisted names, no point-in-time membership). The drift placebo measures it but does not correct for it.
- **No "$5" filter**; a dollar-volume filter replaces it (reason in Method).
- **Small universe:** about 30 names on DEV and about 41 on VAL, so a quintile is 6-8 stocks. Sector concentration (gold miners, semis) is not neutralised because the cache has no sector data.
- **Costs are a flat 10 bps.** Real costs are higher for small, volatile names, and there are no borrow or financing costs, no market impact and no taxes.
- **SPY before 1993 is the price-only ^GSPC**, which understates DEV SPY by about 2-4%/yr. The EW comparison is unaffected.
- **Not tested:** the engine's analog expected-return signal, intraday or open prices, earnings and fundamentals, and short interest.
- **Ridge/logistic CV chose the smallest alpha in the grid** (edge of the grid). This is moot, because the models failed the robustness checks.
- **Multiple testing:** about 48 signal/frequency pairs were tested, and the persistent placebo shows the VAL t-stat is about twice as permissive as nominal.
