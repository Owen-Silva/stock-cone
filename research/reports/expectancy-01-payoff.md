# Expectancy 01: does payoff asymmetry make money, and where does it come from?

Lane `payoff`, 2026-10-04. Script `research/expectancy.py` (runs in about 10 s), results
`research/results/expectancy_summary.json`.

## The question

The claim being tested: "If we're at about a 50% win rate and our winning trades make way more than the
losing ones, we could still be extremely profitable." That is true as arithmetic:
expectancy = win rate x avg win - loss rate x avg loss - costs. Last night's work judged
direction by hit rate only. This pass measures expectancy, and asks whether a lopsided
payoff comes from the ENTRY (information about the stock) or can be manufactured by the
EXIT (stops and targets).

## Short answer

- Exits cannot manufacture an edge. On simulated random-walk prices every exit rule gives
  zero excess expectancy (all 45 "buy everything" cells fall within +/-0.09% per trade,
  and every 90% CI includes 0). Exits trade win rate against payoff ratio and nothing else.
  On a random walk you can get 41% wins with winners 1.4x losers, or 48% wins with
  winners 1.1x losers, and both are worth exactly nothing.
- On real prices, putting a stop or target on "buy everything" LOSES money against simply
  holding, in every period and horizon (for example -0.22% per trade at 21 days in DEV,
  -0.42% in 2019+, 90% CIs exclude 0). The cause is the same one the drifting random walk
  shows: stocks drift up, so an early exit gives up drift you would have earned by holding.
  Real-data exit penalties are about the size the drifting random walk predicts, so the
  exits carry no information of their own.
- The only rule here with a real edge is the ENTRY E2 (top fifth by 12-1 momentum), held to
  the horizon. It beats "buy everything" in all three periods, with ticker and year CIs
  that exclude 0. Adding any stop, target or trailing stop to it made the per-trade edge
  smaller, never bigger.
- So for the claim: yes, ~50-55% wins with winners bigger than losers can be profitable.
  But in this data the profit comes from what you buy (and from stocks drifting up),
  not from how you exit. Momentum held to 63 days shows exactly that shape: 57-64% wins,
  winners 1.35-2.1x losers, positive excess after cost. The survivorship caveat below
  applies in full.
- E1 ("the cone leans up", using a stand-in model) takes 91-94% of all trades and adds
  nothing: about 0 in DEV and VAL, and NEGATIVE in 2019+ (-0.51% per 21-day trade).

## Method (rules frozen in the script; nothing was tuned)

- Universe: the 155 stocks in `data/accuracy.json`; adjusted closes from `data/prices`,
  loaded with `research/backtest.py` `Panel` (forward-fill up to 5 days for P&L).
  Eligibility at the as-of: `backtest.eligibility` (real close, >= 252 days of history,
  63-day median dollar volume >= $2M, <= 10% zero-volume days), plus finite momentum and
  vol forecasts. As-of dates with fewer than 20 eligible names are skipped, so the first
  as-of is 1982-10-19; the last is 2026-09-24.
- As-of dates every 5 trading days. Signals use closes up to the as-of close s; the buy is at
  the NEXT close (s+1), so nothing trades on the close it was computed from. Long only.
- Entries. E0: every eligible stock (the benchmark). E1: the cone leans up, defined as
  the `vol_har_drift` median at that horizon > 0 (its own walk-forward drift x h + HAR
  sigma x the frozen median z). It is a STAND-IN: the product cone (`cone2_f_conf_dev`)
  has no per-forecast output cached for this universe (the cone2 npz panels hold 48 TRAIN
  tickers at harness dates and stop at 2018). E2: the top fifth by `mom_12_1` (the
  `research/xsec.py` definition, log price 21 days ago minus log price 252 days ago),
  ranked within each as-of.
- sigma_h: the frozen HAR forecast (`volforecast.forecast_sigma_series`, market variant,
  on each stock's own trading rows) made at the as-of, x sqrt(h). Levels sit in log price:
  stop = entry x exp(-k sigma_h), target = entry x exp(+k sigma_h).
- Exits, checked on the daily closes after entry; a close through a level fills at that
  close, so gaps cost what they cost. X0 hold h; X1 stop -1 sigma, target +2 sigma;
  X2 stop -2 sigma, target +1 sigma; X3 stop -1 sigma, no target; X4 trailing stop 1 sigma
  below the running max close (entry included), held at most h days.
- Cost 5 bps per side (10 bps round trip), applied multiplicatively. 0-cost numbers are
  in the JSON.
- Periods by as-of date: DEV before 2012, VAL 2012-2018, 2019+. Rules were fixed before any
  run and nothing was tuned, so all three periods are descriptive. They are not a
  train/test split, and the full universe is used in every period.
- Excess per trade = trade P&L minus the mean E0-X0 P&L of all eligible stocks at the same
  as-of and horizon (the "just buy everything and hold" bar). E0-X0 is therefore 0 by
  construction. 90% CIs: (1) resample tickers (1000 draws), (2) resample calendar years
  (1000 draws; trades on the same dates share shocks). The benchmark is held fixed inside
  the resampling.
- Per-year figure for one capital slot: the asked formula, excess per trade x 252 / avg
  hold days, is in the JSON as `excess_per_year_asked`. It overstates rules that exit
  early, because their excess is measured against a benchmark that held for the full h
  days. The JSON also gives `excess_per_year_rate` = 252 x (expectancy / hold days -
  benchmark / h). Both are rough. Trades overlap (a new trade starts every 5 days for
  horizons of 21 and 63 days) and capital is not tracked.

## Simulator sanity

Same trades and same levels, but the path after entry is replaced by a simulated random
walk. Each stock uses its own HAR daily sigma at that as-of, with Gaussian steps. "rw0" has
zero drift in price terms (E[P_k/P_0] = 1), so optional stopping says EVERY exit rule has
expectancy -10 bps (the cost) and excess 0. "rwdrift" adds the real pooled average drift
of 9.25 bps a day.

Result: rw0 passes. Across all 135 cells (3 entries x 5 exits x 3 h x 3 periods) the
largest |excess| is 0.24% (E2-X0, 63 d, 2019+, CI -0.83..+0.34), and 134 of 135 ticker CIs
include 0. The one exception is E1-X1 at 5 d in 2019+, at -0.05% [-0.10, -0.00], which is
borderline. Entries also show no edge on rw0, as they should: they use real past data but
face a random future.

rw0, entry E0 only (10 bps round trip). Cell = win rate / payoff ratio / excess per trade
[ticker 90% CI], avg hold days:

*DEV*

| rule | h=5 | h=21 | h=63 |
|---|---|---|---|
| E0-X0 | 48% / 1.02 / +0.00% [-0.03,+0.03], 5d | 48% / 1.09 / +0.00% [-0.08,+0.07], 21d | 46% / 1.16 / +0.00% [-0.11,+0.11], 63d |
| E0-X1 | 47% / 1.04 / +0.00% [-0.03,+0.03], 5d | 47% / 1.14 / +0.00% [-0.07,+0.07], 18d | 44% / 1.23 / -0.03% [-0.14,+0.08], 54d |
| E0-X2 | 49% / 0.99 / +0.00% [-0.03,+0.03], 5d | 49% / 1.03 / -0.01% [-0.07,+0.06], 19d | 48% / 1.08 / -0.05% [-0.14,+0.04], 55d |
| E0-X3 | 47% / 1.04 / +0.00% [-0.03,+0.03], 5d | 47% / 1.14 / -0.00% [-0.07,+0.07], 19d | 44% / 1.23 / -0.01% [-0.13,+0.10], 54d |
| E0-X4 | 47% / 1.07 / +0.00% [-0.02,+0.04], 5d | 44% / 1.26 / -0.02% [-0.08,+0.05], 17d | 41% / 1.42 / -0.02% [-0.12,+0.09], 48d |

*VAL*

| rule | h=5 | h=21 | h=63 |
|---|---|---|---|
| E0-X0 | 48% / 1.01 / +0.00% [-0.04,+0.05], 5d | 48% / 1.07 / +0.00% [-0.08,+0.08], 21d | 46% / 1.15 / +0.00% [-0.16,+0.17], 63d |
| E0-X1 | 47% / 1.03 / -0.00% [-0.05,+0.04], 5d | 46% / 1.11 / -0.02% [-0.10,+0.05], 18d | 45% / 1.21 / -0.02% [-0.18,+0.14], 54d |
| E0-X2 | 48% / 0.99 / +0.00% [-0.04,+0.05], 5d | 49% / 1.02 / +0.05% [-0.01,+0.12], 19d | 48% / 1.07 / -0.01% [-0.16,+0.14], 55d |
| E0-X3 | 47% / 1.03 / -0.00% [-0.04,+0.04], 5d | 46% / 1.11 / -0.02% [-0.10,+0.06], 19d | 45% / 1.21 / -0.01% [-0.17,+0.16], 54d |
| E0-X4 | 47% / 1.05 / -0.00% [-0.05,+0.05], 5d | 44% / 1.22 / -0.02% [-0.10,+0.06], 17d | 41% / 1.40 / -0.03% [-0.20,+0.13], 49d |

*2019+*

| rule | h=5 | h=21 | h=63 |
|---|---|---|---|
| E0-X0 | 48% / 1.06 / +0.00% [-0.06,+0.06], 5d | 47% / 1.09 / -0.00% [-0.12,+0.13], 21d | 45% / 1.21 / +0.00% [-0.24,+0.23], 63d |
| E0-X1 | 48% / 1.08 / -0.02% [-0.08,+0.03], 5d | 46% / 1.15 / +0.04% [-0.07,+0.16], 18d | 43% / 1.28 / -0.04% [-0.27,+0.17], 53d |
| E0-X2 | 49% / 1.02 / -0.02% [-0.08,+0.03], 5d | 48% / 1.04 / -0.00% [-0.10,+0.10], 19d | 46% / 1.14 / +0.09% [-0.13,+0.28], 55d |
| E0-X3 | 48% / 1.08 / -0.02% [-0.07,+0.04], 5d | 46% / 1.15 / +0.04% [-0.07,+0.15], 18d | 43% / 1.28 / -0.04% [-0.27,+0.18], 54d |
| E0-X4 | 47% / 1.10 / -0.01% [-0.07,+0.04], 5d | 43% / 1.26 / +0.03% [-0.09,+0.14], 17d | 40% / 1.48 / -0.03% [-0.25,+0.18], 48d |

On the drifting walk (DEV shown; VAL and 2019+ look the same), every early exit has
negative excess per trade. It forfeits drift. Per day held, the expectancy is nearly the
same for every exit (8.6-8.8 bps a day at 5 d and 9.1-9.8 at 63 d, before cost). The small
spread (X2 lowest, X4 highest) is expected and is not an edge. Drift accrues per dollar
per day, and X2 cashes out the paths that just went up while it keeps holding the ones
that went down. X4 does the reverse.

*DEV*

| rule | h=5 | h=21 | h=63 |
|---|---|---|---|
| E0-X0 | 52% / 1.08 / -0.00% [-0.03,+0.03], 5d | 56% / 1.21 / +0.00% [-0.08,+0.07], 21d | 61% / 1.40 / -0.00% [-0.12,+0.12], 63d |
| E0-X1 | 52% / 1.09 / -0.03% [-0.06,+0.00], 5d | 55% / 1.22 / -0.18% [-0.26,-0.11], 19d | 59% / 1.37 / -0.62% [-0.73,-0.50], 56d |
| E0-X2 | 53% / 1.03 / -0.04% [-0.07,-0.01], 5d | 58% / 1.09 / -0.31% [-0.37,-0.24], 18d | 62% / 1.18 / -1.24% [-1.33,-1.16], 52d |
| E0-X3 | 52% / 1.09 / -0.03% [-0.06,+0.01], 5d | 55% / 1.22 / -0.16% [-0.23,-0.08], 19d | 59% / 1.39 / -0.46% [-0.58,-0.34], 57d |
| E0-X4 | 51% / 1.11 / -0.03% [-0.06,+0.00], 5d | 53% / 1.34 / -0.27% [-0.34,-0.20], 18d | 55% / 1.61 / -0.91% [-1.03,-0.80], 52d |

Three trades printed by hand (real prices, h = 21) show the path logic is right:

- AAPL X1 stop: signal 2026-07-29, buy 2026-07-30 at 333.1427, sigma_h 0.0776; stop 308.2563, target 389.105. Closes after entry: 308.64, 303.16. Exit day 2 (2026-08-03) at 303.1586: -9.00% before cost.
- AAPL X1 target: signal 2024-06-07, buy 2024-06-10 at 191.2965, sigma_h 0.0627; stop 179.6657, target 216.8656. Closes after entry: 205.19, 211.06, 212.22, 210.48, 214.62, 212.27, 207.7, 205.53, 206.17, 207.1, 211.24, 212.08, 208.63, 214.7, 218.19. Exit day 15 (2024-07-02) at 218.1902: +14.06% before cost.
- AAPL X4 trail: signal 2026-07-22, buy 2026-07-23 at 321.3828, sigma_h 0.0751; running max 339.7869, trail level 315.2187 (= max x exp(-sigma_h)). Closes after entry: 332.73, 336.62, 339.79, 337.9, 333.14, 308.64. Exit day 6 (2026-07-31) at 308.6438: -3.96% before cost.

## Real data (10 bps round trip)

Cell = win rate / payoff ratio (avg win / avg loss) / excess per trade vs E0-X0
[ticker 90% CI], avg hold days. Year-resampled CIs are in the JSON. They are wider for E2,
for example E2-X0 2019+ 63 d is +4.01% [years +1.04, +8.22].

*DEV*

| rule | h=5 | h=21 | h=63 |
|---|---|---|---|
| E0-X0 | 52% / 1.07 / +0.00% [-0.03,+0.03], 5d | 56% / 1.12 / +0.00% [-0.12,+0.13], 21d | 61% / 1.26 / +0.00% [-0.35,+0.40], 63d |
| E0-X1 | 51% / 1.07 / -0.06% [-0.09,-0.03], 5d | 55% / 1.11 / -0.22% [-0.33,-0.11], 19d | 59% / 1.23 / -0.52% [-0.88,-0.14], 57d |
| E0-X2 | 52% / 1.04 / -0.01% [-0.04,+0.02], 5d | 57% / 1.05 / -0.12% [-0.23,-0.02], 19d | 62% / 1.12 / -0.68% [-0.96,-0.40], 55d |
| E0-X3 | 51% / 1.07 / -0.07% [-0.09,-0.04], 5d | 55% / 1.11 / -0.23% [-0.35,-0.11], 19d | 59% / 1.24 / -0.47% [-0.85,-0.06], 57d |
| E0-X4 | 51% / 1.08 / -0.08% [-0.10,-0.05], 5d | 53% / 1.20 / -0.32% [-0.44,-0.21], 18d | 56% / 1.38 / -0.72% [-1.09,-0.33], 53d |
| E1-X0 | 52% / 1.07 / -0.01% [-0.03,+0.02], 5d | 56% / 1.11 / -0.03% [-0.13,+0.09], 21d | 61% / 1.25 / -0.10% [-0.46,+0.31], 63d |
| E1-X1 | 51% / 1.07 / -0.06% [-0.09,-0.04], 5d | 55% / 1.11 / -0.24% [-0.34,-0.13], 19d | 59% / 1.22 / -0.59% [-0.95,-0.23], 57d |
| E1-X2 | 52% / 1.04 / -0.02% [-0.05,+0.01], 5d | 57% / 1.05 / -0.15% [-0.24,-0.04], 19d | 62% / 1.10 / -0.79% [-1.06,-0.51], 55d |
| E1-X3 | 51% / 1.06 / -0.07% [-0.10,-0.05], 5d | 55% / 1.11 / -0.25% [-0.35,-0.13], 19d | 59% / 1.23 / -0.53% [-0.90,-0.14], 57d |
| E1-X4 | 51% / 1.07 / -0.08% [-0.11,-0.06], 5d | 53% / 1.19 / -0.34% [-0.45,-0.23], 18d | 56% / 1.37 / -0.80% [-1.16,-0.42], 53d |
| E2-X0 | 53% / 1.07 / +0.14% [+0.08,+0.21], 5d | 57% / 1.16 / +0.59% [+0.30,+0.88], 21d | 61% / 1.35 / +1.53% [+0.66,+2.35], 63d |
| E2-X1 | 52% / 1.07 / +0.07% [+0.00,+0.13], 5d | 56% / 1.13 / +0.25% [-0.02,+0.51], 19d | 59% / 1.31 / +0.86% [+0.07,+1.60], 56d |
| E2-X2 | 53% / 1.03 / +0.10% [+0.03,+0.16], 5d | 58% / 1.07 / +0.36% [+0.10,+0.60], 18d | 62% / 1.17 / +0.43% [-0.25,+1.10], 54d |
| E2-X3 | 52% / 1.07 / +0.06% [-0.01,+0.13], 5d | 56% / 1.13 / +0.26% [-0.02,+0.53], 19d | 59% / 1.32 / +0.94% [+0.09,+1.75], 57d |
| E2-X4 | 52% / 1.09 / +0.06% [-0.01,+0.13], 5d | 53% / 1.23 / +0.10% [-0.17,+0.37], 18d | 56% / 1.50 / +0.65% [-0.20,+1.45], 53d |

*VAL*

| rule | h=5 | h=21 | h=63 |
|---|---|---|---|
| E0-X0 | 53% / 1.04 / +0.00% [-0.05,+0.05], 5d | 58% / 1.12 / +0.00% [-0.18,+0.19], 21d | 63% / 1.26 / +0.00% [-0.55,+0.58], 63d |
| E0-X1 | 52% / 1.04 / -0.04% [-0.08,+0.00], 5d | 56% / 1.09 / -0.25% [-0.41,-0.07], 19d | 61% / 1.21 / -0.59% [-1.12,-0.04], 57d |
| E0-X2 | 54% / 1.01 / -0.01% [-0.05,+0.03], 5d | 59% / 1.03 / -0.18% [-0.35,+0.01], 19d | 64% / 1.11 / -0.62% [-1.11,-0.11], 55d |
| E0-X3 | 52% / 1.03 / -0.04% [-0.08,+0.00], 5d | 56% / 1.09 / -0.23% [-0.41,-0.06], 19d | 61% / 1.22 / -0.53% [-1.07,+0.02], 57d |
| E0-X4 | 52% / 1.05 / -0.04% [-0.08,-0.00], 5d | 54% / 1.17 / -0.32% [-0.49,-0.14], 18d | 57% / 1.36 / -0.93% [-1.47,-0.38], 54d |
| E1-X0 | 53% / 1.04 / +0.01% [-0.03,+0.05], 5d | 58% / 1.14 / +0.05% [-0.12,+0.24], 21d | 64% / 1.31 / +0.15% [-0.40,+0.72], 63d |
| E1-X1 | 53% / 1.04 / -0.03% [-0.07,+0.01], 5d | 57% / 1.11 / -0.19% [-0.35,-0.02], 19d | 62% / 1.24 / -0.41% [-0.93,+0.13], 57d |
| E1-X2 | 54% / 1.01 / -0.01% [-0.05,+0.03], 5d | 59% / 1.05 / -0.11% [-0.28,+0.07], 19d | 65% / 1.16 / -0.40% [-0.87,+0.09], 55d |
| E1-X3 | 53% / 1.04 / -0.03% [-0.07,+0.01], 5d | 57% / 1.11 / -0.17% [-0.35,+0.00], 19d | 62% / 1.25 / -0.38% [-0.93,+0.17], 57d |
| E1-X4 | 52% / 1.05 / -0.03% [-0.08,+0.01], 5d | 54% / 1.19 / -0.26% [-0.43,-0.09], 18d | 58% / 1.39 / -0.77% [-1.32,-0.24], 54d |
| E2-X0 | 54% / 1.08 / +0.16% [+0.07,+0.25], 5d | 58% / 1.21 / +0.59% [+0.21,+0.98], 21d | 64% / 1.42 / +1.31% [+0.14,+2.43], 63d |
| E2-X1 | 53% / 1.07 / +0.08% [-0.00,+0.18], 5d | 56% / 1.17 / +0.27% [-0.09,+0.63], 19d | 62% / 1.27 / +0.37% [-0.75,+1.46], 57d |
| E2-X2 | 54% / 1.06 / +0.15% [+0.07,+0.24], 5d | 59% / 1.12 / +0.40% [+0.07,+0.74], 19d | 64% / 1.30 / +0.91% [-0.03,+1.79], 55d |
| E2-X3 | 53% / 1.06 / +0.07% [-0.02,+0.17], 5d | 56% / 1.16 / +0.22% [-0.16,+0.59], 19d | 62% / 1.28 / +0.42% [-0.75,+1.58], 58d |
| E2-X4 | 52% / 1.08 / +0.06% [-0.03,+0.15], 5d | 54% / 1.24 / +0.11% [-0.26,+0.48], 18d | 57% / 1.41 / -0.22% [-1.30,+0.93], 54d |

*2019+*

| rule | h=5 | h=21 | h=63 |
|---|---|---|---|
| E0-X0 | 52% / 1.14 / -0.00% [-0.07,+0.07], 5d | 55% / 1.33 / -0.00% [-0.32,+0.30], 21d | 58% / 1.75 / +0.00% [-1.15,+1.13], 63d |
| E0-X1 | 51% / 1.13 / -0.08% [-0.14,-0.02], 5d | 53% / 1.32 / -0.42% [-0.68,-0.16], 19d | 55% / 1.61 / -1.99% [-2.86,-1.17], 55d |
| E0-X2 | 53% / 1.08 / -0.08% [-0.13,-0.02], 5d | 56% / 1.13 / -0.78% [-0.99,-0.58], 18d | 59% / 1.35 / -2.82% [-3.48,-2.14], 53d |
| E0-X3 | 51% / 1.14 / -0.06% [-0.12,+0.01], 5d | 53% / 1.36 / -0.20% [-0.52,+0.10], 19d | 55% / 1.72 / -1.03% [-2.17,+0.08], 56d |
| E0-X4 | 51% / 1.16 / -0.06% [-0.12,-0.00], 5d | 51% / 1.47 / -0.34% [-0.65,-0.04], 18d | 52% / 1.95 / -1.48% [-2.63,-0.29], 52d |
| E1-X0 | 53% / 1.10 / -0.11% [-0.18,-0.05], 5d | 56% / 1.25 / -0.51% [-0.79,-0.23], 21d | 59% / 1.57 / -1.96% [-2.92,-0.95], 63d |
| E1-X1 | 52% / 1.10 / -0.16% [-0.22,-0.11], 5d | 54% / 1.25 / -0.83% [-1.07,-0.59], 19d | 57% / 1.48 / -3.38% [-4.13,-2.64], 55d |
| E1-X2 | 53% / 1.05 / -0.15% [-0.21,-0.10], 5d | 57% / 1.09 / -1.02% [-1.21,-0.82], 18d | 60% / 1.27 / -3.80% [-4.41,-3.18], 53d |
| E1-X3 | 52% / 1.10 / -0.15% [-0.21,-0.09], 5d | 54% / 1.28 / -0.71% [-0.97,-0.44], 19d | 56% / 1.53 / -2.92% [-3.85,-1.96], 56d |
| E1-X4 | 52% / 1.12 / -0.14% [-0.20,-0.09], 5d | 52% / 1.38 / -0.80% [-1.07,-0.53], 18d | 53% / 1.77 / -3.19% [-4.14,-2.19], 52d |
| E2-X0 | 52% / 1.22 / +0.46% [+0.28,+0.64], 5d | 55% / 1.50 / +1.59% [+0.88,+2.27], 21d | 57% / 2.08 / +4.01% [+1.75,+6.02], 63d |
| E2-X1 | 52% / 1.19 / +0.29% [+0.16,+0.42], 5d | 53% / 1.43 / +0.82% [+0.25,+1.31], 19d | 54% / 1.85 / +1.36% [-0.25,+2.82], 56d |
| E2-X2 | 53% / 1.12 / +0.31% [+0.19,+0.42], 5d | 56% / 1.24 / +0.49% [+0.06,+0.90], 18d | 58% / 1.59 / +0.21% [-1.00,+1.35], 54d |
| E2-X3 | 52% / 1.22 / +0.37% [+0.19,+0.56], 5d | 53% / 1.49 / +1.19% [+0.47,+1.90], 19d | 54% / 1.96 / +2.55% [+0.30,+4.53], 57d |
| E2-X4 | 51% / 1.23 / +0.35% [+0.19,+0.49], 5d | 50% / 1.64 / +0.95% [+0.21,+1.76], 18d | 51% / 2.27 / +2.49% [+0.05,+4.78], 52d |

Absolute E0-X0 expectancy (buy everything, hold, after cost): DEV +0.28 / +1.44 / +4.61%,
VAL +0.23 / +1.31 / +4.20%, 2019+ +0.55 / +2.76 / +9.16% at 5 / 21 / 63 days.

### Reading it

- **Exits on the benchmark.** All 4 exit rules have negative excess per trade in every
  period and horizon. They give up 0.01-0.08% per trade at 5 d, 0.12-0.78% at 21 d and
  0.47-2.82% at 63 d. In DEV and VAL that roughly matches the drifting random walk
  (X1 63 d: real -0.52 / -0.59 vs rwdrift -0.62 / -0.60). In 2019+ it is larger
  (-1.99 vs -0.60) because the drift was larger (+9.2% per 63 d). The exit rules also do
  change the win/payoff mix on real data the way the random walk predicts: X4 lowers
  wins and raises payoff, X2 does the reverse.
- **E2 momentum is the entry with information.** At X0, its excess is +0.14 / +0.59 /
  +1.53% (DEV), +0.16 / +0.59 / +1.31% (VAL) and +0.46 / +1.59 / +4.01% (2019+) per trade,
  and both kinds of CI exclude 0 in every cell. As a rough per-slot figure that is +5-8% a
  year in DEV and VAL and +16-23% in 2019+. Every exit rule shrinks E2's per-trade edge.
  E2-X3 / X4 at 63 d in 2019+ (51-54% wins, payoff 2.0-2.3, excess +2.5%) are the closest
  match to that description, and they are still smaller than E2-X0 (+4.0%).
- **E1 (cone leans up, stand-in)** skips only the 6-9% of trades with a negative
  walk-forward drift forecast. It is within noise in DEV and VAL. In 2019+ it is worse
  than buying everything (X0: -0.11 / -0.51 / -1.96%, CIs exclude 0), because the stocks
  it skipped then did better than average.

## Not verified / caveats

- **Survivorship (the big one).** The universe is today's 155 popular names, so every
  stock in it survived to 2026, and many were picked because they did well. That inflates
  the absolute returns and very likely the momentum edge. Stocks that ran up and then
  died are missing. The E2 numbers, and the 2019+ figure of roughly +16-23% a year
  especially, should be treated as an upper bound until they are rerun on a point-in-time
  universe. Per the standing rule, a result this good counts as a bug until proven
  otherwise. It does agree in sign with last night's xsec TEST (`mom_12_1` weekly lag-1
  top-quintile excess about +32% a year, `research/results/xsec_test.csv`), which used the
  same kind of universe and so has the same bias.
- E1 uses the `vol_har_drift` stand-in, not the served product cone. The product cone
  shrinks the centre toward the basket mean, so its "leans up" share and its result could
  differ.
- The HAR vol model was fitted on DEV, so DEV sigma levels are in-sample. This only scales
  the stop and target distances; no rule was tuned on it.
- Fills at the daily close only. There is no intraday stop fill (an intraday stop would
  usually fill nearer the level, but a close-only check is how the rules were specified),
  and no slippage beyond 5 bps a side. Dividends are inside the adjusted closes.
- Per-year figures ignore overlapping trades and position sizing; they are rough.
- The random walk is Gaussian. It checks the simulator's path logic and the
  no-free-lunch result, not fat tails.
- CIs hold the per-date benchmark fixed. The year bootstrap has only 7-8 years for 2019+
  and VAL.
