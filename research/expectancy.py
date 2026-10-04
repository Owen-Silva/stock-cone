"""Expectancy, not hit rate (lane payoff, report research/reports/expectancy-01-payoff.md).

The point being tested: ~50% wins can still be very profitable if winners are much bigger than
losers. expectancy = win% x avg win - loss% x avg loss - costs. This script measures
expectancy for a few frozen long-only entry x exit rules, and checks the simulator on
random-walk prices, where theory says exits trade win rate against payoff ratio but
leave expectancy at the drift.

Rules are FROZEN in this file (nothing is tuned), so every period is descriptive.

    python research/expectancy.py            # writes research/results/expectancy_summary.json

Timing: signals read at the close of as-of day s (closes <= s only); buy at the NEXT
close e = s+1; the exit is checked on the daily closes e+1 .. e+h; a close through a
level fills at that close (gap risk included). sigma_h = frozen HAR forecast
(research/volforecast.forecast_sigma_series, market variant) made at s, x sqrt(h);
levels sit in log-price space: stop = entry * exp(-k sigma_h), target = entry * exp(+k sigma_h).
P&L is the simple return; cost 5 bps per side.

Entries:  E0 every eligible stock; E1 vol_har_drift median > 0 (stand-in for the served
model, see report); E2 top fifth by mom_12_1 (xsec.compute_signals definition).
Exits:    X0 hold h; X1 stop -1s / target +2s; X2 stop -2s / target +1s;
          X3 stop -1s, no target; X4 trailing stop 1s below running max close, max h.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research import backtest as bt  # noqa: E402
from research import volforecast as vf  # noqa: E402

RESULTS = os.path.join(ROOT, "research", "results")
HS = (5, 21, 63)
STEP = 5
EXITS = ("X0", "X1", "X2", "X3", "X4")
ENTRIES = ("E0", "E1", "E2")
VARIANTS = ("real", "rw0", "rwdrift")
COSTS = (0.0, 5.0)          # bps per side
PERIODS = ("DEV", "VAL", "2019+")
VAL_START = pd.Timestamp("2012-01-01")
TEST_START = pd.Timestamp("2019-01-01")
MIN_NAMES = 20
N_BOOT = 1000
SEED = 20261004


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------------------
# Exit rules on a block of log-price paths L (n x h), L[:, k-1] = ln(P[e+k]/P[e])
# ---------------------------------------------------------------------------

def _first_hit(cond):
    """Index (0-based) of the first True per row, or -1."""
    any_ = cond.any(axis=1)
    idx = cond.argmax(axis=1)
    return np.where(any_, idx, -1)


def apply_exits(L, s):
    """L: (n, h) log paths, s: (n,) sigma_h. Returns {exit: (gross simple return, hold days)}."""
    n, h = L.shape
    rows = np.arange(n)
    out = {}

    def finish(k):          # k = 0-based exit column or -1 (no hit -> hold to h)
        k = np.where(k < 0, h - 1, k)
        return np.expm1(L[rows, k]), (k + 1).astype(np.int16)

    S = s[:, None]
    out["X0"] = finish(np.full(n, -1))
    out["X1"] = finish(_first_hit((L <= -S) | (L >= 2 * S)))
    out["X2"] = finish(_first_hit((L <= -2 * S) | (L >= S)))
    out["X3"] = finish(_first_hit(L <= -S))
    M = np.maximum(np.maximum.accumulate(L, axis=1), 0.0)     # running max incl. entry (L0 = 0)
    out["X4"] = finish(_first_hit(L <= M - S))
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def universe():
    with open(os.path.join(ROOT, "data", "accuracy.json"), encoding="utf-8") as f:
        acc = json.load(f)
    return sorted(s["ticker"] for s in acc["stocks"])


def market_returns_full():
    """SPY daily log returns spliced onto ^GSPC before SPY starts (no cutoff)."""
    spy, gs = bt.read_cached("SPY"), bt.read_cached("^GSPC")
    rs = np.log(spy["close"]).diff().dropna()
    rg = np.log(gs["close"]).diff().dropna()
    return pd.concat([rg[rg.index < rs.index[0]], rs]).sort_index()


def own_forecasts(ticker, mkt):
    """HAR sigma_h (h-day, log units) and vol_har_drift median, on the ticker's own rows."""
    df = bt.read_cached(ticker)
    r = np.concatenate([[np.nan], np.diff(np.log(df["close"].to_numpy(float)))])
    m = mkt.reindex(df.index).to_numpy()
    mu = vf.walkforward_drift(r)
    out = {}
    for h in HS:
        sig = vf.forecast_sigma_series(r, h, m) * np.sqrt(h)
        z50 = vf.z_quantiles(h, "drift", market=True)[9]          # q = 0.50
        out[f"sig{h}"] = sig
        out[f"med{h}"] = mu * h + sig * z50
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# Main build: one row per (as-of s, eligible stock)
# ---------------------------------------------------------------------------

def build():
    tickers = universe()
    log(f"universe {len(tickers)} tickers from data/accuracy.json")
    panel = bt.Panel(tickers)
    log(f"panel {len(panel.tickers)} tickers {panel.cal[0].date()}..{panel.cal[-1].date()} "
        f"missing={panel.missing} rows={len(panel.cal)}")
    elig = bt.eligibility(panel).to_numpy()
    P = panel.pff.to_numpy(float)
    lp = np.log(P)
    N, J = P.shape
    mom = np.full_like(lp, np.nan)          # xsec.compute_signals "mom_12_1" = lp.shift(21) - lp.shift(252)
    mom[252:] = lp[252 - 21:N - 21] - lp[:N - 252]
    # real average daily drift (simple), pooled over eligible stock-days
    dr = P[1:] / P[:-1] - 1.0
    ok = elig[1:] & np.isfinite(dr)
    avg_drift = float(dr[ok].mean())
    del dr, ok
    log(f"pooled avg daily simple drift over eligible stock-days: {avg_drift * 1e4:.2f} bps")

    mkt = market_returns_full()
    grid = np.arange(260, N - 2, STEP)
    asof_dates = panel.cal[grid]

    rows = []
    examples = []
    for j, t in enumerate(panel.tickers):
        fc = own_forecasts(t, mkt).reindex(panel.cal)
        base = elig[grid, j] & np.isfinite(mom[grid, j]) & np.isfinite(P[grid, j])
        for h in HS:
            base &= np.isfinite(fc[f"sig{h}"].to_numpy()[grid]) & np.isfinite(fc[f"med{h}"].to_numpy()[grid])
        g = grid[base]
        if len(g) == 0:
            continue
        rec = {"tid": np.full(len(g), j, np.int16), "s": g.astype(np.int32),
               "mom": mom[g, j].astype(np.float32)}
        rng = np.random.default_rng(SEED + 7919 * j)
        for h in HS:
            e = g + 1
            valid = e + h < N
            idx = np.clip(e[:, None] + np.arange(0, h + 1)[None, :], 0, N - 1)
            path = P[idx, j]
            valid &= np.isfinite(path).all(axis=1) & (path > 0).all(axis=1)
            L = np.log(path[:, 1:] / path[:, :1])
            L = np.where(valid[:, None], L, 0.0)
            s = fc[f"sig{h}"].to_numpy()[g]
            rec[f"valid{h}"] = valid
            rec[f"e1_{h}"] = fc[f"med{h}"].to_numpy()[g] > 0
            rec[f"sig{h}"] = s.astype(np.float32)
            sd = s / np.sqrt(h)                     # daily sigma for the random walk
            Z = rng.standard_normal((len(g), h))
            rw0 = np.cumsum(sd[:, None] * Z - 0.5 * sd[:, None] ** 2, axis=1)   # E[P_k/P_0] = 1
            rwd = rw0 + np.log1p(avg_drift) * np.arange(1, h + 1)[None, :]
            for v, LL in (("real", L), ("rw0", rw0), ("rwdrift", rwd)):
                ex = apply_exits(LL, s)
                for x in EXITS:
                    rec[f"{v}_{h}_{x}_r"] = ex[x][0].astype(np.float32)
                    rec[f"{v}_{h}_{x}_d"] = ex[x][1]
            if t in ("AAPL", "MSFT", "JPM") and h == 21:
                examples.extend(hand_examples(t, h, g, valid, path, s, L, panel.cal, rec))
        rows.append(pd.DataFrame(rec))
        if (j + 1) % 25 == 0:
            log(f"  {j + 1}/{len(panel.tickers)} tickers")
    df = pd.concat(rows, ignore_index=True)
    del rows
    # as-of filter: >= MIN_NAMES eligible stocks
    cnt = df.groupby("s")["tid"].transform("size")
    df = df[cnt >= MIN_NAMES].reset_index(drop=True)
    # E2: top fifth by mom_12_1 within each as-of
    pct = df.groupby("s")["mom"].rank(pct=True, method="first")
    df["e2"] = pct > 0.8
    d = panel.cal[df["s"].to_numpy()]
    df["period"] = np.where(d < VAL_START, 0, np.where(d < TEST_START, 1, 2)).astype(np.int8)
    df["yr"] = (d.year - d.year.min()).astype(np.int16)
    return panel, df, examples, avg_drift


def hand_examples(t, h, g, valid, path, s, L, cal, rec):
    """Pick one X1 stop, one X1 target and one X4 trailing exit for printing."""
    out = []
    want = {"X1 stop": None, "X1 target": None, "X4 trail": None}
    r1, d1 = rec[f"real_{h}_X1_r"], rec[f"real_{h}_X1_d"]
    r4, d4 = rec[f"real_{h}_X4_r"], rec[f"real_{h}_X4_d"]
    for i in range(len(g) - 1, -1, -1):
        if not valid[i]:
            continue
        if want["X1 stop"] is None and d1[i] < h and r1[i] < 0:
            want["X1 stop"] = i
        if want["X1 target"] is None and d1[i] < h and r1[i] > 0:
            want["X1 target"] = i
        if want["X4 trail"] is None and d4[i] < h and d4[i] > 3:
            want["X4 trail"] = i
        if all(v is not None for v in want.values()):
            break
    for kind, i in want.items():
        if i is None:
            continue
        k = int((d1 if kind.startswith("X1") else d4)[i])
        p0 = float(path[i, 0])
        ex = {"ticker": t, "rule": kind, "h": h, "asof": str(cal[g[i]].date()),
              "entry_date": str(cal[g[i] + 1].date()), "entry_price": round(p0, 4),
              "sigma_h": round(float(s[i]), 4),
              "exit_day": k, "exit_date": str(cal[g[i] + 1 + k].date()),
              "exit_price": round(float(path[i, k]), 4),
              "pnl_gross_pct": round(100 * (float(path[i, k]) / p0 - 1), 3)}
        if kind.startswith("X1"):
            ex["stop"] = round(p0 * np.exp(-s[i]), 4)
            ex["target"] = round(p0 * np.exp(2 * s[i]), 4)
            ex["closes_to_exit"] = [round(float(x), 2) for x in path[i, 1:k + 1]]
        else:
            run = np.maximum(np.maximum.accumulate(path[i, :k + 1]), p0)
            ex["running_max_at_exit"] = round(float(run[-1]), 4)
            ex["trail_level_at_exit"] = round(float(run[-2] if k >= 1 else p0) * np.exp(-s[i]), 4)
            ex["trail_level_after_exit_close"] = round(float(run[-1]) * np.exp(-s[i]), 4)
            ex["closes_to_exit"] = [round(float(x), 2) for x in path[i, 1:k + 1]]
        out.append(ex)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def net_of(r, cost_bps):
    c = cost_bps / 1e4
    return (1.0 + r) * (1.0 - c) / (1.0 + c) - 1.0


def summarize(df, n_tickers):
    rng = np.random.default_rng(SEED)
    # ticker bootstrap weights: B x n_tickers multinomial counts
    W = np.stack([np.bincount(rng.integers(0, n_tickers, n_tickers), minlength=n_tickers)
                  for _ in range(N_BOOT)]).astype(np.float64)
    tid = df["tid"].to_numpy()
    s_idx = df["s"].to_numpy()
    period = df["period"].to_numpy()
    yr = df["yr"].to_numpy()
    n_yr = int(yr.max()) + 1
    yrng = np.random.default_rng(SEED + 1)
    out = []
    for v in VARIANTS:
        for h in HS:
            valid = df[f"valid{h}"].to_numpy()      # same trades on real and simulated paths
            entry_mask = {"E0": valid, "E1": valid & df[f"e1_{h}"].to_numpy(),
                          "E2": valid & df["e2"].to_numpy()}
            for cost in COSTS:
                bench = net_of(df[f"{v}_{h}_X0_r"].to_numpy(np.float64), cost)
                # per-as-of benchmark: mean E0-X0 net return at that as-of
                bsum = np.bincount(s_idx[valid], weights=bench[valid], minlength=s_idx.max() + 1)
                bcnt = np.bincount(s_idx[valid], minlength=s_idx.max() + 1)
                B = np.divide(bsum, bcnt, out=np.full(len(bsum), np.nan), where=bcnt > 0)[s_idx]
                for x in EXITS:
                    net = net_of(df[f"{v}_{h}_{x}_r"].to_numpy(np.float64), cost)
                    hold = df[f"{v}_{h}_{x}_d"].to_numpy(np.float64)
                    exc = net - B
                    for E in ENTRIES:
                        for pi, pname in enumerate(PERIODS):
                            m = entry_mask[E] & (period == pi)
                            n = int(m.sum())
                            if n == 0:
                                continue
                            nm, em, hm = net[m], exc[m], hold[m]
                            wins, losses = nm[nm > 0], nm[nm <= 0]
                            aw = float(wins.mean()) if len(wins) else float("nan")
                            al = float(-losses.mean()) if len(losses) else float("nan")
                            S = np.bincount(tid[m], weights=em, minlength=n_tickers)
                            C = np.bincount(tid[m], minlength=n_tickers).astype(np.float64)
                            boot = (W @ S) / np.maximum(W @ C, 1)
                            lo, hi = np.percentile(boot, [5, 95])
                            # second CI: resample calendar YEARS (trades on one date share shocks)
                            Sy = np.bincount(yr[m], weights=em, minlength=n_yr)
                            Cy = np.bincount(yr[m], minlength=n_yr).astype(np.float64)
                            ys = np.flatnonzero(Cy > 0)
                            pick = ys[yrng.integers(0, len(ys), (N_BOOT, len(ys)))]
                            yboot = Sy[pick].sum(axis=1) / Cy[pick].sum(axis=1)
                            ylo, yhi = np.percentile(yboot, [5, 95])
                            avg_hold = float(hm.mean())
                            exm = float(em.mean())
                            bmean = float(B[m].mean())
                            out.append({
                                "variant": v, "h": h, "cost_bps_side": cost, "exit": x,
                                "entry": E, "period": pname, "trades": n,
                                "tickers": int((C > 0).sum()),
                                "win_rate": float(len(wins) / n), "avg_win": aw, "avg_loss": al,
                                "payoff": aw / al if al and al > 0 else float("nan"),
                                "expectancy": float(nm.mean()),
                                "excess": exm, "excess_ci90": [float(lo), float(hi)],
                                "excess_ci90_years": [float(ylo), float(yhi)], "years": int(len(ys)),
                                "bench_expectancy_same_asofs": bmean,
                                "avg_hold_days": avg_hold,
                                "excess_per_year_asked": exm * 252.0 / avg_hold,
                                "excess_per_year_rate": 252.0 * (float(nm.mean()) / avg_hold - bmean / h),
                                "per_day_expectancy": float(nm.mean()) / avg_hold,
                            })
    return out


def fmt_tables(res, variant, cost, entries=ENTRIES, periods=PERIODS):
    """Compact markdown: per period, rows = entry x exit, cols = h, cell = win / payoff / excess."""
    key = {(r["variant"], r["cost_bps_side"], r["period"], r["entry"], r["exit"], r["h"]): r for r in res}
    lines = []
    for p in periods:
        lines.append(f"**{p}** ({variant}, {cost * 2:g} bps round trip). Cell = win rate / payoff ratio / "
                     f"excess per trade vs E0-X0 [90% CI], avg hold days")
        lines.append("")
        lines.append("| rule | h=5 | h=21 | h=63 |")
        lines.append("|---|---|---|---|")
        for E in entries:
            for x in EXITS:
                cells = []
                for h in HS:
                    r = key.get((variant, cost, p, E, x, h))
                    if r is None:
                        cells.append("-")
                        continue
                    lo, hi = r["excess_ci90"]
                    cells.append(f"{100 * r['win_rate']:.0f}% / {r['payoff']:.2f} / "
                                 f"{100 * r['excess']:+.2f}% [{100 * lo:+.2f},{100 * hi:+.2f}], "
                                 f"{r['avg_hold_days']:.0f}d")
                lines.append(f"| {E}-{x} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    t0 = time.time()
    panel, df, examples, avg_drift = build()
    log(f"rows {len(df)}  mem {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    res = summarize(df, len(panel.tickers))
    info = {
        "universe": len(panel.tickers), "rows": int(len(df)),
        "asof_dates": int(df["s"].nunique()),
        "first_asof": str(panel.cal[df["s"].min()].date()), "last_asof": str(panel.cal[df["s"].max()].date()),
        "rows_by_period": {p: int((df["period"] == i).sum()) for i, p in enumerate(PERIODS)},
        "e1_share_of_e0": {str(h): float(df.loc[df[f"valid{h}"], f"e1_{h}"].mean()) for h in HS},
        "avg_daily_drift_simple": avg_drift,
        "sigma_source": "frozen HAR (volforecast.forecast_sigma_series, market variant) x sqrt(h)",
        "e1_model": "vol_har_drift median > 0 (stand-in; the product cone per-forecast output is not cached)",
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(RESULTS, "expectancy_summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"info": info, "examples": examples, "results": res}, f, indent=1)
    log(json.dumps(info))
    for ex in examples:
        print(json.dumps(ex))
    for v in VARIANTS:
        print(fmt_tables(res, v, 5.0))
    print(fmt_tables(res, "real", 0.0, entries=("E0",)))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
