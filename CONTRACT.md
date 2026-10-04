# Stock Cone — API contract (shared by the engine and UI lanes)

The product: type a ticker, see its chart. Flip a **Future** toggle and a cone is
drawn forward from today: the typical path plus 50% and 80% bands, built from what
actually happened after the most similar past setups. **Replay** mode picks a past
date, hides everything after it, draws the cone, then reveals what really happened.

Honesty is a feature: the cone is odds, not a promise, and the page shows its own
track record.

## Layout

```
stock-cone/
  server.py        stdlib http.server (no Flask), 127.0.0.1:8642, serves web/ at / and the API at /api/
  engine.py        data fetch + cache, features, analog search, cone, track record
  calibrate.py     offline walk-forward report: does the analog cone beat a naive volatility cone?
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
  },
  "odds": {
    "n_analogs": 312,
    "p_up": 0.56,                        // share of analogs ending the horizon above the start
    "base_p_up": 0.53,                   // unconditional share for this ticker over its own history, same horizon (the "normal" rate)
    "median_ret": 0.006,                 // end-of-horizon simple return at p50
    "p10_ret": -0.058, "p90_ret": 0.069,
    "width_vs_naive": 0.93               // 80% band width / width of a naive volatility cone for the same ticker+horizon (<1 = analogs agree more than usual)
  },
  "actual": [["2026-10-05", 189.10], ...],   // replay only: the real closes for the horizon days (may be shorter than horizon near today). null when asof is the latest bar.
  "track": {                             // walk-forward record for THIS ticker + horizon; null if not computed yet
    "n": 150,                            // past forecasts tested (no look-ahead)
    "inside_80": 0.79,                   // share where the real end price landed inside p10..p90 (ideal 0.80)
    "inside_50": 0.51,                   // inside p25..p75 (ideal 0.50)
    "direction_hit": 0.55,               // p50 direction matched reality
    "direction_base": 0.53               // hit rate of always guessing the majority direction (the bar to beat)
  },
  "analogs_sample": [                    // the closest matches, up to 10, nearest first
    {"ticker": "MSFT", "date": "2019-05-03", "ret": 0.021}
  ],
  "notes": ["Free Yahoo data: delisted companies are missing (survivorship bias)."]
}
```

Errors: HTTP 4xx/5xx with `{"error": "human-readable message"}` (unknown ticker, under
~1 year of history, bad horizon, asof before enough history, Yahoo unreachable).

Speed: first request for a new ticker may take a few seconds (download); cached after.
Repeat requests should answer in well under a second. `track` may be computed lazily
and cached.

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
