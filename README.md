# Stock Cone

Type a ticker, see its chart. Flip **Future** and a cone is drawn forward from the
last close: the typical path plus 50% and 80% bands, built from what actually
happened after the most similar past setups (analog / nearest-neighbour
forecasting). **Replay** picks a past date, hides everything after it, draws the
cone, then reveals what really happened. The page shows the cone's own track record.

It is a local prototype. Free Yahoo data, no accounts, nothing leaves your machine
except the price downloads.

## Run it

Double-click `run.bat`. It starts the server and opens http://127.0.0.1:8642.

Or by hand:

```
"python" server.py
```

The first start downloads the analog pool (44 tickers plus the S&P 500 index,
about 40 seconds, progress in the console) into `data/`. After that, prices are
refreshed at most once a day per ticker. Needs Python 3.13 with yfinance, pandas
and numpy; nothing else.

Other commands:

| command | what it does |
|---|---|
| `python engine.py --selfcheck` | proves there is no look-ahead (see below) |
| `python calibrate.py` | walk-forward report: analog cone vs a naive volatility cone, writes `data/calibration.json` |
| `python calibrate.py --dev-only` | the same, pre-2015 dates only (the only split allowed for tuning) |

API: `GET /api/forecast?ticker=NVDA&horizon=5&asof=2024-06-14` (see `CONTRACT.md`),
plus `GET /api/health` (warm-up progress).

## Accuracy scoreboard

`GET /api/accuracy` (and the Accuracy page) shows the walk-forward track record
for about 160 hand-picked individual stocks at all three horizons: the stocks
on the watchlist, mega caps, growth and semis, volatile small caps, and other
sectors. No ETFs. The list and its groups live in `universe.py`; it does not touch
the analog pool, so forecasts are unchanged.

`python accuracy.py` builds `data/accuracy.json` (about 6 minutes, sequential
downloads, resumes a crashed build; `--fresh` starts over, `--only NVDA,KO` prints
a few without writing). The server serves that file and, when it is missing or
over a day old, rebuilds it in one background thread while returning
`status: "building"` with partial results. Each stock's numbers come from the same
function and cache as the chart's `track`, so they match click-through exactly.
Stocks Yahoo does not know, or with too little history (about a year to warm up
plus ~30 testable dates), are listed in `skipped` with the reason. With ~150 tests
per stock, a single stock's direction rate can be off by 6-8 points from luck
alone; trust the pooled summary more than any one row.

## How the cone is made

1. **Describe the setup** on every trading day with 12 numbers, each computed only
   from bars up to that day: trailing returns over 5/21/63/126 days, distance from
   the 50- and 200-day moving averages, drawdown from the 52-week high (all divided
   by the stock's own 63-day volatility, so a calm stock's +5% month and a wild
   stock's +15% month look alike), volatility regime (21-day vol / 252-day vol),
   a volume ratio (5-day / 63-day), and the market regime from SPY (its 63-day move,
   its volatility level, its drawdown; before SPY existed in 1993 the S&P 500 index
   stands in). Each number is divided by a fixed scale measured once on pre-2010
   data and frozen in `engine.py`, so the scaling never depends on the data being
   forecast.
2. **Find the 250 nearest past setups** (Euclidean distance) across the ticker's own
   history and a fixed basket of 42 large caps across sectors plus SPY and QQQ.
   An analog dated A only counts if A + horizon <= the as-of date. At most one
   analog per ticker per 5-day block, so one episode cannot fill the pool.
   Analogs from stale-quote stretches of old data are excluded.
3. **Take what happened next.** Each analog's forward path over the horizon is
   rescaled by (this stock's 63-day vol now / the analog's vol then), clipped to
   0.25x-4x, so a calm stock does not inherit a wild stock's cone.
4. **Read off the percentiles** p10/p25/p50/p75/p90 per day, weighting nearer
   analogs a little more (the nearest counts about 2.7x the 250th), and apply them
   to the last close.

`width_vs_naive` compares the 80% band with a naive random-walk cone from the
stock's own trailing volatility (zero drift, sqrt-time). The track record walks
forward over about 150 past dates for that ticker, forecasting each from data up
to that date only.

## No look-ahead

`python engine.py --selfcheck` forecasts as of a past date D, then rebuilds the
whole engine from raw data truncated at D (every pool ticker and the market
series) and forecasts again. History, cone, odds, analogs and walk-forward records
must match exactly, for six tickers and dates (1987 to 2024) at all three horizons.

## Limits (read these)

- **Survivorship bias.** Yahoo has no delisted companies, and the basket is today's
  large caps, picked with hindsight. Their past paths lean toward winners, which
  tilts the cone up.
- **Adjusted prices.** Replay shows split/dividend-adjusted closes with today's
  factors, so old price levels differ from what was quoted on the day. Returns
  are right.
- **It is odds, not a promise.** The cone is a summary of what similar setups did.
  It knows nothing about earnings dates, news, or anything outside the price and
  volume history. See `calibrate.py` for how well it actually does; the honest
  result is in "Calibration" below.
- Free Yahoo data can have bad ticks and stale stretches in old history. Features
  are clipped and stale stretches dropped from the pool, but not every glitch is
  caught.
- Today's unfinished bar is dropped during US market hours.

## Calibration

`python calibrate.py`, run 2026-10-03: 18 tickers (SPY, QQQ, AAPL, MSFT, NVDA,
AMZN, GOOGL, META, JPM, XOM, JNJ, KO, WMT, BA, AMD, TSLA, NFLX, PLTR), a forecast
every 10 trading days from 1 year after listing. The parameters (K = 250, the
features, the weights, the 0.25x-4x clip) were chosen before any outcome was
scored and were not tuned afterwards. The split is still reported: dev = as-of
dates before 2015, holdout = 2015 onward.

Holdout (2015 onward). A = analog cone, N = naive vol cone. Pinball skill is
1 - A/N: positive means the analog cone is better. The 90% interval comes from a
bootstrap over ticker-years. "vs N+8%" is the same skill against the naive cone
plus a fixed 8%/yr drift.

| h | n | cov80 A / N | cov50 A / N | width80 A / N | pinball skill | 90% CI | vs N+8% | dir hit A | majority base |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 5139 | 0.809 / 0.817 | 0.501 / 0.542 | 10.8% / 11.3% | +0.5% | +0.1% to +1.0% | +0.2% | 0.525 | 0.563 |
| 21 | 5110 | 0.785 / 0.801 | 0.495 / 0.514 | 22.0% / 23.2% | +1.3% | +0.6% to +1.9% | +0.1% | 0.597 | 0.612 |
| 63 | 5034 | 0.777 / 0.774 | 0.487 / 0.467 | 39.4% / 40.4% | +4.3% | +2.8% to +5.7% | +0.9% | 0.655 | 0.669 |

Dev (before 2015): pinball skill +0.4% / +0.3% / +1.4%. Against N+8% it is
+0.2% / -0.5% / -0.9%.

**Verdict.** The analog cone is honest: its bands cover about what they claim, and
it is slightly narrower than the naive cone. It beats the zero-drift naive cone by
a hair at 1 week and 1 month, and by about 4% at 3 months in the holdout. Most of
that edge is drift, not pattern matching. The analogs come from a survivor basket
and their median path carries roughly 13-20%/yr of upward drift. Give the naive cone a plain 8%/yr drift
and the edge mostly disappears (+0.2% / +0.1% / +0.9% holdout, negative in dev at
1M and 3M). The p50 direction does worse than always guessing each stock's usual
direction at every horizon. Treat the cone as a well-calibrated range, not as a
directional signal. Full numbers, including per-ticker skill, are in
`data/calibration.json`.
