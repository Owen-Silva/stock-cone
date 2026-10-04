# Stock Cone — API contract (shared by the engine and UI lanes)

The product: type a ticker, see its chart. Flip a **Future** toggle and a cone is
drawn forward from today: the typical path plus 50% and 80% bands. Since 2026-10-04
(cone v2, `cone_model.py`) the bands come from a volatility forecast calibrated on past
errors, centred on a cautious drift, with a small blend of what actually happened after
the most similar past setups. **Replay** mode picks a past
date, hides everything after it, draws the cone, then reveals what really happened.

Honesty is a feature: the cone is odds, not a promise, and the page shows its own
track record.

## Layout

```
stock-cone/
  server.py        stdlib http.server (no Flask), 127.0.0.1:8642, serves web/ at / and the API at /api/
  engine.py        data fetch + cache, features, analog search, cone, track record
  cone_model.py    the cone model (v2 = research `cone2_f_conf_dev`, frozen); `--vs-research` proves it matches the research code
  calibrate.py     offline walk-forward report: does the served cone beat a naive volatility cone? (its record cache data/calib_records.json predates v2: delete it before rerunning)
  run.bat          starts the server and opens http://127.0.0.1:8642
  data/            cache (gitignored)
  web/index.html   the whole UI: one file, inline CSS + JS, no build step, no CDN (works offline)
  web/mock-forecast.json   a response matching this contract; index.html uses it ONLY with ?mock=1
```

Python: `python` (3.13, has
yfinance 1.2, pandas 3.0, numpy 2.2; NO flask). In Git Bash, bare `python` is a Store stub, so use the full path.

## GET /api/forecast

Query params:

| param | required | values |
|---|---|---|
| `ticker` | yes | case-insensitive, Yahoo symbols (`NVDA`, `SPY`, `BRK-B`) |
| `horizon` | no | trading days: `5` (1W, default), `21` (1M), `63` (3M) |
| `asof` | no | `YYYY-MM-DD`; snaps to the last trading day on or before it. Omitted = latest bar. |

200 response:

```jsonc
{
  "ticker": "NVDA",
  "name": "NVIDIA Corporation",          // falls back to the ticker
  "asof": "2026-10-02",                  // the last bar the forecast may see
  "horizon": 5,
  "last_close": 187.30,
  "history": [["2021-10-04", 20.71], ...],   // adjusted daily closes, from max(listing, asof - 5 calendar years) to asof INCLUSIVE. Nothing after asof, ever.
  "future": {
    "dates": ["2026-10-05", ...],        // length == horizon. Real trading dates when known (replay), else projected US trading weekdays
    "p10": [...], "p25": [...], "p50": [...], "p75": [...], "p90": [...]   // PRICE levels, length == horizon; index i = close i+1 trading days after asof
    // v2: the model is defined at the horizon end (day h == the model exactly). Day d < h is
    // interpolated in log space as m*d + (Q_h - m*h)*sqrt(d/h), m = mean of the 19 end
    // quantiles / h (a random walk with drift), so bands stay ordered every day.
  },
  "odds": {
    "n_analogs": 250,                    // analogs found by the search the cone blends in (under 100: no blend, see notes)
    "p_up": 0.56,                        // the cone's probability of ending above the start: its quantiles' CDF at 0, linear between quantiles, clipped to 0.05..0.95
    "base_p_up": 0.53,                   // share of this ticker's past windows of the same length that ended up, only windows completed by asof (the "normal" rate)
    "median_ret": 0.006,                 // end-of-horizon simple return at p50
    "p10_ret": -0.058, "p90_ret": 0.069,
    "width_vs_naive": 0.93               // 80% band width / width of a naive Gaussian cone from trailing 63-day volatility, same ticker+horizon
  },
  "actual": [["2026-10-05", 189.10], ...],   // replay only: the real closes for the horizon days (may be shorter than horizon near today). null when asof is the latest bar.
  "track": {                             // walk-forward record for THIS ticker + horizon; null if not computed yet
    "n": 150,                            // past forecasts tested (no look-ahead)
    "inside_80": 0.79,                   // share where the real end price landed inside p10..p90 (ideal 0.80)
    "inside_50": 0.51,                   // inside p25..p75 (ideal 0.50)
    "direction_hit": 0.55,               // p50 direction matched reality
    "direction_base": 0.53               // walk-forward baseline (the bar to beat): on each tested date guess the stock's usual direction so far (up if base_p_up as of that date >= 0.5), hit rate of that guess
  },
  "analogs_sample": [                    // the closest matches, up to 10, nearest first
    {"ticker": "MSFT", "date": "2019-05-03", "ret": 0.021, "ret_scaled": 0.034}   // ret_scaled: after v0-style vol scaling (info only)
  ],
  "notes": ["Free Yahoo data: delisted companies are missing (survivorship bias)."]
}
```

Errors: HTTP 4xx/5xx with `{"error": "human-readable message"}` (unknown ticker, under
~1 year of history, bad horizon, asof before enough history, Yahoo unreachable).

Speed: first request for a new ticker may take a few seconds (download); cached after.
Repeat requests should answer in well under a second. `track` may be computed lazily
and cached.

### Change log

- **2026-10-04, cone v2.** The response shape is unchanged; what fills it changed.
  - The cone is `cone_model.py` (research `cone2_f_conf_dev`, pre-registered in
    `research/PLAN.md`): HAR volatility width with walk-forward conformal factors, a centre
    shrunk to the analog basket's drift, a DEV-fitted shape, and a 20% / 10% / 10% (1W / 1M /
    3M) blend with the deviation-scaled analog cone. `python cone_model.py --vs-research`
    checks it against the research harness model.
  - `p_up`, `median_ret`, `p10_ret`, `p90_ret` and `width_vs_naive` are read off the cone's
    end-of-horizon quantiles (before: off the analogs). `n_analogs` and `analogs_sample` still
    come from the same analog search, which the cone blends in.
  - With fewer than 100 analogs the cone is served without the blend (and a note says so);
    v0 answered 400 there. Under ~1 year of history is still a 400.
  - `track` scores the v2 cone walk-forward (same ~150-date scheme). **`direction_base` is
    now walk-forward**: for each tested date, guess the majority direction of the stock's
    history before that date; `direction_base` is the hit rate of that guess. Before, it was
    the majority share over the tested dates themselves, which used hindsight and inflated
    the bar by 1-2 points (research/reports/diag-01-v0-failure-modes.md, Q6).
  - The track cache key suffix is `engine.TRACK_VERSION` (`c2`); older entries are dropped
    on load and never served.

## Hard rules

- **No look-ahead.** A forecast as of date D uses only bars dated <= D: features, analog
  pool (an analog dated A counts only if A + horizon <= D), volatility scaling, everything.
  `history` never contains a bar after `asof`. Replay is worthless if this leaks.
- **No silent failures.** Every error or fallback is shown in the UI with its reason,
  not just logged. The mock is used only when `?mock=1` is in the URL.

## GET /api/accuracy  (added 2026-10-03: the Accuracy scoreboard)

The walk-forward track record for ~150 individual stocks, all three horizons in one
payload. Per-stock numbers MUST be the exact same `track` the chart page shows for that
ticker + horizon (same function, same cache), so clicking through never disagrees.

```jsonc
{
  "status": "ready",                     // "building" while a (re)build runs; partial results are still returned
  "progress": {"done": 87, "total": 152},
  "generated": "2026-10-03T17:40:00",    // when the last complete build finished (null if never)
  "horizons": [5, 21, 63],
  "groups": [                            // display order; a stock can be in several
    {"id": "watchlist", "label": "Watchlist"},
    {"id": "mega",   "label": "Mega caps"},
    {"id": "growth", "label": "Growth & tech"},
    {"id": "volatile", "label": "Volatile & small"},
    {"id": "other",  "label": "Other sectors"}
  ],
  "summary": {                           // over the stocks done so far, keyed by horizon as a string
    "5": {
      "stocks": 148, "forecasts": 22200,
      "inside_80": 0.79, "inside_50": 0.49,   // pooled over every forecast
      "direction_hit": 0.52, "direction_base": 0.55,
      "beat_base": 41,                   // stocks where direction_hit > direction_base
      "calibrated": 120                  // stocks whose inside_80 is within 0.72..0.88
    },
    "21": {}, "63": {}
  },
  "stocks": [
    {
      "ticker": "NVDA", "name": "NVIDIA Corporation",
      "groups": ["watchlist", "mega", "growth"],
      "first_date": "1999-01-22", "last_close": 233.95,
      "h": {                              // keyed by horizon as a string; same fields as /api/forecast `track` + avg_width_80
        "5":  {"n": 150, "inside_80": 0.767, "inside_50": 0.447, "direction_hit": 0.52, "direction_base": 0.56, "avg_width_80": 0.121},
        "21": {}, "63": {}
      }
    }
  ],
  "skipped": [{"ticker": "CHYM", "reason": "under 1 year of history"}],
  "notes": ["..."]
}
```

- Built by `accuracy.py` (CLI: `python accuracy.py`) into `data/accuracy.json`. The server
  serves that file; if it is missing or older than one day it starts a background rebuild
  and returns `status: "building"` with whatever is done so far. Never block a request.
- The universe lives in `universe.py` (hand-curated, grouped). It does NOT change the
  analog pool: adding the scoreboard must not change any forecast.

## Optional `event` field on /api/forecast (added 2026-10-04 overnight)

```jsonc
"event": {                       // null when no known event falls inside the horizon, or for replays
  "type": "earnings",
  "date": "2026-10-23",          // next scheduled earnings date (Yahoo calendar), live forecasts only
  "in_window": true,
  "widen": 1.18                  // factor applied to the cone's spread at the horizon end because of it
}
```

Applied only to the live (latest-bar) forecast: past earnings dates are not known
point-in-time, so replays and the track record never use it.
