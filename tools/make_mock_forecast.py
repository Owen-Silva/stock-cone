"""Write web/mock-forecast.json from SYNTHETIC prices (no market data).

The page's ?mock=1 mode reads this file so the UI can be developed and tested
without the server or a network. The prices are a seeded random walk
(geometric Brownian motion), so the file is reproducible and contains no
Yahoo or other third-party data. The schema matches the /api/forecast reply.

    python tools/make_mock_forecast.py            # rewrite web/mock-forecast.json
    python tools/make_mock_forecast.py --seed 7   # a different synthetic path
"""

import argparse
import datetime as dt
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "web", "mock-forecast.json")

TICKER, NAME = "DEMO", "Demo Corp (synthetic data)"
ASOF = dt.date(2026, 9, 2)
YEARS, HORIZON = 5, 21
START_PRICE, MU, SIGMA = 40.0, 0.18, 0.45      # annual drift and volatility of the walk
Z = {"p10": -1.2816, "p25": -0.6745, "p50": 0.0, "p75": 0.6745, "p90": 1.2816}


def weekdays(start, end):
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def next_weekdays(after, n):
    d, out = after, []
    while len(out) < n:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


def walk(rng, p0, n):
    dtyr = 1 / 252
    drift = (MU - 0.5 * SIGMA ** 2) * dtyr
    out, p = [], p0
    for _ in range(n):
        p *= math.exp(drift + SIGMA * math.sqrt(dtyr) * rng.gauss(0, 1))
        out.append(round(p, 2))
    return out


def build(seed):
    rng = random.Random(seed)
    days = weekdays(ASOF.replace(year=ASOF.year - YEARS), ASOF)
    closes = walk(rng, START_PRICE, len(days))
    last = closes[-1]
    fut_days = next_weekdays(ASOF, HORIZON)
    actual = walk(rng, last, HORIZON)

    # Cone: a log-normal fan around a small drift, slightly narrower than the walk's own volatility.
    sig_d, mu_d = SIGMA * 0.93 / math.sqrt(252), 0.0005
    future = {"dates": [d.isoformat() for d in fut_days]}
    for q, z in Z.items():
        future[q] = [round(last * math.exp(mu_d * j + z * sig_d * math.sqrt(j)), 2)
                     for j in range(1, HORIZON + 1)]
    end = lambda q: round(future[q][-1] / last - 1, 4)  # noqa: E731

    pool = "AAA BBB CCC DDD EEE FFF GGG HHH".split()
    analogs = [{"ticker": rng.choice(pool + [TICKER]),
                "date": (dt.date(2012, 1, 3) + dt.timedelta(days=rng.randrange(0, 365 * 12))).isoformat(),
                "ret": round(rng.gauss(0.01, 0.08), 3)} for _ in range(10)]

    return {
        "ticker": TICKER,
        "name": NAME,
        "asof": ASOF.isoformat(),
        "horizon": HORIZON,
        "last_close": last,
        "history": [[d.isoformat(), c] for d, c in zip(days, closes)],
        "future": future,
        "odds": {"n_analogs": 300, "p_up": 0.54, "base_p_up": 0.53, "median_ret": end("p50"),
                 "p10_ret": end("p10"), "p90_ret": end("p90"), "width_vs_naive": 0.93},
        "actual": [[d.isoformat(), c] for d, c in zip(fut_days, actual)],
        "track": {"n": 150, "inside_80": 0.79, "inside_50": 0.51,
                  "direction_hit": 0.52, "direction_base": 0.53},
        "analogs_sample": analogs,
        "notes": ["Synthetic demo data: a seeded random walk, not real prices.",
                  "Run the server and drop ?mock=1 from the address to see real forecasts."],
        "event": {"type": "earnings", "date": (ASOF + dt.timedelta(days=51)).isoformat(),
                  "in_window": True, "widen": 1.18},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    data = build(a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {a.out}: {len(data['history'])} synthetic closes, last {data['last_close']}")


if __name__ == "__main__":
    main()
