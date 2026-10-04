"""Walk-forward calibration report: does the analog cone beat a naive volatility cone?

For each ticker, forecast every STEP trading days (from 1 year after listing)
using only data up to that date, for horizons 5/21/63, and score both cones:

  analog cone : engine's cone (K nearest past setups, vol-scaled paths)
  naive cone  : random walk, zero drift, the ticker's own trailing 63-day vol,
                quantiles = last * exp(z_q * vol * sqrt(h))

Metrics (end-of-horizon price only):
  cov80 / cov50 : share of outcomes inside p10..p90 / p25..p75 (ideal 0.80 / 0.50)
  width80       : average (p90 - p10) / last close
  pinball       : average pinball (quantile) loss over q = .1 .25 .5 .75 .9 on the
                  simple return; lower is better. skill = 1 - analog/naive.
  vs N+8%       : the same skill against the naive cone plus an 8%/yr drift
  dir_hit       : p50 direction matched reality; dir_base = always guessing each
                  ticker's majority direction over the same dates (hindsight baseline)

Split: dates before SPLIT are "dev", dates on/after are "holdout". Any parameter
choice may only look at dev (`--dev-only`); the honest numbers are holdout.

    python calibrate.py              # full report, writes data/calibration.json
    python calibrate.py --dev-only   # print dev split only (for tuning)
"""

import json
import os
import sys
import time

import numpy as np

import engine as E

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM",
           "XOM", "JNJ", "KO", "WMT", "BA", "AMD", "TSLA", "NFLX", "PLTR"]
STEP = 10
SPLIT = "2015-01-01"
BOOT = 1000
# Second baseline: the naive cone plus a plain 8%/yr upward drift (a long-run
# equity-return guess fixed in advance, not fitted). It tests whether the analog
# edge is setup-matching or just the upward drift the analogs carry.
DRIFT = 0.08 / 252


def pinball(y, yq):
    """Mean pinball loss over the 5 quantiles. y: (n,), yq: (n, 5) simple returns."""
    d = y[:, None] - yq
    return np.mean(np.maximum(E.QS * d, (E.QS - 1) * d), axis=1)


def per_record(recs, ticker, h):
    r = np.array([x["ret"] for x in recs])
    aq = np.array([x["aq"] for x in recs])
    nq = np.array([x["nq"] for x in recs])
    y, ya, yn = np.expm1(r), np.expm1(aq), np.expm1(nq)
    return {
        "ticker": np.array([ticker] * len(r)),
        "year": np.array([int(x["date"][:4]) for x in recs]),
        "date": np.array([x["date"] for x in recs]),
        "up": r > 0,
        "a80": (r >= aq[:, 0]) & (r <= aq[:, 4]), "a50": (r >= aq[:, 1]) & (r <= aq[:, 3]),
        "n80": (r >= nq[:, 0]) & (r <= nq[:, 4]), "n50": (r >= nq[:, 1]) & (r <= nq[:, 3]),
        "aw": ya[:, 4] - ya[:, 0], "nw": yn[:, 4] - yn[:, 0],
        "apl": pinball(y, ya), "npl": pinball(y, yn),
        "dpl": pinball(y, np.expm1(nq + DRIFT * h)),
        "ahit": (aq[:, 2] > 0) == (r > 0),
    }


def concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def summarise(d, rng):
    n = len(d["up"])
    if n == 0:
        return None
    # majority-direction baseline: per ticker, over these same dates
    base_hits = 0.0
    for t in np.unique(d["ticker"]):
        m = d["ticker"] == t
        u = d["up"][m].mean()
        base_hits += max(u, 1 - u) * m.sum()
    skill = 1 - d["apl"].sum() / d["npl"].sum()
    skill_drift = 1 - d["apl"].sum() / d["dpl"].sum()
    # cluster bootstrap over (ticker, year): outcomes overlap in time, so dates
    # are not independent; resampling whole ticker-years gives a fairer interval
    keys = np.char.add(d["ticker"].astype(str), d["year"].astype(str))
    _, cid = np.unique(keys, return_inverse=True)
    a_sum = np.bincount(cid, d["apl"])
    n_sum = np.bincount(cid, d["npl"])
    c = len(a_sum)
    draws = rng.integers(0, c, size=(BOOT, c))
    boot = 1 - a_sum[draws].sum(1) / n_sum[draws].sum(1)
    years = np.unique(d["year"])
    won = sum(d["apl"][d["year"] == yv].sum() < d["npl"][d["year"] == yv].sum() for yv in years)
    return {
        "n": int(n),
        "analog": {"cov80": float(d["a80"].mean()), "cov50": float(d["a50"].mean()),
                   "width80": float(d["aw"].mean()), "pinball": float(d["apl"].mean()),
                   "dir_hit": float(d["ahit"].mean())},
        "naive": {"cov80": float(d["n80"].mean()), "cov50": float(d["n50"].mean()),
                  "width80": float(d["nw"].mean()), "pinball": float(d["npl"].mean())},
        "dir_base": float(base_hits / n),
        "pinball_skill": float(skill),
        "pinball_skill_90ci": [float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
        "pinball_skill_vs_naive_plus_8pct_drift": float(skill_drift),
        "years_analog_better": [int(won), int(len(years))],
    }


def table(title, res):
    print(f"\n{title}")
    print(f"{'h':>3} {'n':>6} | {'cov80 A/N':>11} {'cov50 A/N':>11} | {'width80 A/N':>13} | "
          f"{'pinball A/N':>15} {'skill':>7} {'90% CI':>16} {'yrs':>6} {'vs N+8%':>8} | {'dir A':>6} {'base':>6}")
    for h, s in res.items():
        if s is None:
            continue
        a, nv = s["analog"], s["naive"]
        lo, hi = s["pinball_skill_90ci"]
        print(f"{h:>3} {s['n']:>6} | {a['cov80']:.3f}/{nv['cov80']:.3f} {a['cov50']:.3f}/{nv['cov50']:.3f} | "
              f"{a['width80']:.4f}/{nv['width80']:.4f} | {a['pinball']:.5f}/{nv['pinball']:.5f} "
              f"{s['pinball_skill']:+7.2%} [{lo:+.2%},{hi:+.2%}] {s['years_analog_better'][0]:>2}/{s['years_analog_better'][1]:<3} "
              f"{s['pinball_skill_vs_naive_plus_8pct_drift']:+8.2%} | "
              f"{a['dir_hit']:.3f}  {s['dir_base']:.3f}")


def main():
    dev_only = "--dev-only" in sys.argv
    eng = E.Engine(quiet=True)
    t0 = time.time()
    eng.warm_up()
    parts = {h: [] for h in E.HORIZONS}
    # walk-forward records are cached per ticker, keyed by every engine setting
    # that affects them, so a dev run followed by the full report is not redone
    sig = json.dumps([E.K, E.BLOCK, E.CANDIDATE_MULT, E.SCALE_CLIP, E.FEATURE_SCALE, E.FEATURE_WEIGHT,
                      E.FEATURE_CLIP, E.POOL, STEP, E.MIN_BARS], sort_keys=True)
    cache_path = os.path.join(E.DATA_DIR, "calib_records.json")
    try:
        with open(cache_path, encoding="utf-8") as fh:
            cache = json.load(fh)
        if cache.get("sig") != sig:
            cache = {"sig": sig, "tickers": {}}
    except (OSError, ValueError):
        cache = {"sig": sig, "tickers": {}}
    for t in TICKERS:
        t1 = time.time()
        td = eng.get_ticker(t)
        ok = np.flatnonzero(td.valid)
        lo = max(int(ok[0]), 252)
        pos = list(range(lo, len(td.day) - 1 - min(E.HORIZONS), STEP))
        ck = f"{t}|{E.day_str(td.day[-1])}"
        if ck in cache["tickers"]:
            recs = {int(h): v for h, v in cache["tickers"][ck].items()}
        else:
            recs = eng.walk_forward(t, list(E.HORIZONS), pos)
            cache["tickers"][ck] = {str(h): v for h, v in recs.items()}
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(cache, fh)
        for h in E.HORIZONS:
            if recs[h]:
                parts[h].append(per_record(recs[h], t, h))
        print(f"  {t:6s} {len(pos):5d} dates from {E.day_str(td.day[lo])}  ({time.time() - t1:.1f}s)", flush=True)

    rng = np.random.default_rng(0)
    out = {"config": {"tickers": TICKERS, "step": STEP, "split": SPLIT, "K": E.K, "block": E.BLOCK,
                      "scale_clip": E.SCALE_CLIP, "features": E.FEATURES, "pool": E.POOL,
                      "naive": "zero-drift lognormal, trailing 63-day vol",
                      "naive_plus_drift": "same plus 8%/yr log drift"},
           "results": {}}
    splits = ["dev"] if dev_only else ["dev", "holdout", "all"]
    per_ticker = {}
    for sp in splits:
        res = {}
        for h in E.HORIZONS:
            d = concat(parts[h])
            m = np.ones(len(d["up"]), bool)
            if sp == "dev":
                m = d["date"] < SPLIT
            elif sp == "holdout":
                m = d["date"] >= SPLIT
            res[h] = summarise({k: v[m] for k, v in d.items()}, rng)
            if sp == "holdout":
                for t in TICKERS:
                    mt = m & (d["ticker"] == t)
                    if mt.sum() >= 20:
                        per_ticker.setdefault(t, {})[h] = float(1 - d["apl"][mt].sum() / d["npl"][mt].sum())
        out["results"][sp] = {str(h): v for h, v in res.items()}
        label = {"dev": f"DEV (as-of dates before {SPLIT})", "holdout": f"HOLDOUT (as-of dates {SPLIT} onward)",
                 "all": "ALL DATES"}[sp]
        table(label + "   A = analog cone, N = naive vol cone", res)
    if per_ticker:
        out["results"]["holdout_pinball_skill_by_ticker"] = per_ticker
        print("\nHoldout pinball skill by ticker (positive = analog better):")
        print("  " + "  ".join(f"{t}:" + "/".join(f"{per_ticker[t].get(h, float('nan')):+.1%}" for h in E.HORIZONS)
                               for t in TICKERS if t in per_ticker))
    if not dev_only:
        path = os.path.join(E.DATA_DIR, "calibration.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {path}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
