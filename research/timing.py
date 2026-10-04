"""Market-level timing research: when to be in stocks vs cash (research/PLAN.md).

Question: is there an honest market-timing or risk-regime signal on the S&P 500 that
survives out-of-sample testing after costs?

Splits for THIS study (strict, by time; differ from research/splits.py on purpose because
the index history is decades longer than the single-stock panel):
    dev  : returns dated  < 2000-01-01   (index history from 1927; evaluation ~1929..1999)
    val  : returns dated 2000-01-01 .. 2018-12-31
    test : returns dated >= 2019-01-01   -- coordinator only: --split test --coordinator

For dev/val the price series is TRUNCATED at 2018-12-31 at load time, so nothing can be
computed on 2019+ data. Signals at date t use closes up to t only; warm-up history from
before a split's start is allowed (it is older data, never newer).

Assumptions (stated in the report):
  * Decision at the close of a check day t, trade executes at the close of t+1, so the
    return from close u-1 to close u is earned with the weight decided at close u-2.
  * 5 bps cost per unit of weight changed (a full in/out switch = 5 bps), charged on the
    trade day. Buy-and-hold pays nothing. Initial position at the split start is free.
  * Cash earns 0. No leverage (weights in [0, 1]).
  * ^GSPC is a PRICE index (no dividends); SPY (adjusted, total return) is a VAL cross-check.

Usage:
    python research/timing.py --split dev
    python research/timing.py --split val
    python research/timing.py --split test --coordinator     # coordinator only
Outputs go to research/results/timing_*.
"""
import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICES = os.path.join(ROOT, "data", "prices")
RESULTS = os.path.join(ROOT, "research", "results")

DEV_END = pd.Timestamp("2000-01-01")      # dev: date < this
VAL_START = pd.Timestamp("2000-01-01")
VAL_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")

COST = 0.0005          # 5 bps per unit of weight traded
ANN = 252
VOL_TARGET = 0.15      # ~ long-run S&P vol; fixed a priori, not fitted
EWMA_LAMBDA = 0.94     # RiskMetrics
SEED = 12345

# Crash windows: (name, search-start, search-end). Peak = max close in window before the
# trough; trough = min close in window. Rebound = trough -> trough + 252 trading days.
CRASHES = {
    "dev": [("1929-32", "1929-06-01", "1932-12-31"),
            ("1937-38", "1937-01-01", "1938-06-30"),
            ("1973-74", "1972-10-01", "1974-12-31"),
            ("1987", "1987-07-01", "1987-12-31")],
    "val": [("2000-02", "2000-01-01", "2002-12-31"),
            ("2007-09", "2007-07-01", "2009-06-30")],
    "test": [("2020", "2020-01-01", "2020-06-30"),
             ("2022", "2022-01-01", "2022-12-31")],
}


# ----------------------------------------------------------------------------- data
def load_close(name, split):
    path = os.path.join(PRICES, name + ".csv")
    df = pd.read_csv(path, parse_dates=["date"])
    s = df.set_index("date")["close"].astype(float).sort_index()
    s = s[np.isfinite(s) & (s > 0)]
    s = s[~s.index.duplicated(keep="last")]
    if split in ("dev", "val"):
        s = s[s.index <= VAL_END]          # hard guard: never touch 2019+
    return s


def split_bounds(split):
    if split == "dev":
        return pd.Timestamp("1900-01-01"), DEV_END - pd.Timedelta(days=1)
    if split == "val":
        return VAL_START, VAL_END
    if split == "test":
        return TEST_START, pd.Timestamp("2100-01-01")
    raise ValueError(split)


# -------------------------------------------------------------------------- signals
def ewma_vol(close):
    r = np.log(close).diff().fillna(0.0).to_numpy()
    v = np.empty_like(r)
    acc = r[:21].var() if len(r) > 21 else 1e-4
    for i, x in enumerate(r):
        acc = EWMA_LAMBDA * acc + (1 - EWMA_LAMBDA) * x * x
        v[i] = acc
    out = pd.Series(np.sqrt(v * ANN), index=close.index)
    out.iloc[:63] = np.nan                     # warm-up
    return out


def ma_10m(close):
    """Faber 10-month SMA: mean of the last 9 completed month-end closes and today's close."""
    me = close.groupby(close.index.to_period("M")).last()
    # sum of the 9 month-ends before the current month
    prev9 = me.shift(1).rolling(9).sum()
    per = close.index.to_period("M")
    p9 = prev9.reindex(per).to_numpy()
    return pd.Series((p9 + close.to_numpy()) / 10.0, index=close.index)


def build_signals(close):
    """Desired weight in [0,1] at each close, using data up to that close only."""
    sig = {}
    for n in (50, 100, 200):
        ma = close.rolling(n).mean()
        sig[f"ma{n}"] = (close > ma).astype(float).where(ma.notna())
    m10 = ma_10m(close)
    sig["ma10m"] = (close > m10).astype(float).where(m10.notna())
    for m, n in ((3, 63), (6, 126), (12, 252)):
        past = close.shift(n)
        sig[f"tsm{m}"] = (close > past).astype(float).where(past.notna())
    vol = ewma_vol(close)
    sig["vt15"] = (VOL_TARGET / vol).clip(upper=1.0).where(vol.notna())
    hi = close.rolling(252).max()
    dd = close / hi - 1.0
    sig["dd10"] = (dd > -0.10).astype(float).where(hi.notna())
    sig["dd20"] = (dd > -0.20).astype(float).where(hi.notna())
    v3 = pd.concat([sig["ma50"], sig["ma200"], sig["tsm12"]], axis=1)
    sig["vote3"] = (v3.sum(axis=1, min_count=3) >= 2).astype(float).where(v3.notna().all(axis=1))
    sig["ma200_vt"] = sig["ma200"] * sig["vt15"]
    feats = pd.DataFrame({"vol": vol, "dd": dd})
    return pd.DataFrame(sig), feats


def check_mask(index, freq):
    if freq == "daily":
        return np.ones(len(index), bool)
    s = pd.Series(1, index=index)
    key = index.to_period("W-FRI") if freq == "weekly" else index.to_period("M")
    last = s.groupby(key).apply(lambda g: g.index[-1])
    return index.isin(last.values)


# ------------------------------------------------------------------------- backtest
def run_rule(close, desired, freq):
    """Daily net returns and held weights. desired: weight decided at each close."""
    chk = check_mask(close.index, freq)
    d = desired.where(chk).ffill()                 # decision held between checks
    held = d.shift(1)                              # trade executes at next close
    r = close.pct_change()
    gross = held.shift(1) * r                      # return u earned with weight held at u-1
    trade = held.diff().abs()
    net = gross - COST * trade.fillna(0.0)
    return pd.DataFrame({"r": r, "w": held.shift(1), "net": net, "trade": trade})


def metrics(net, w, trade):
    net = net.dropna()
    eq = (1 + net).cumprod()
    yrs = len(net) / ANN
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = net.std() * np.sqrt(ANN)
    sharpe = net.mean() / net.std() * np.sqrt(ANN) if net.std() > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    yr = (1 + net).groupby(net.index.year).prod() - 1
    sw = int((trade.reindex(net.index).fillna(0) > 1e-9).sum())
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, maxdd=mdd, worst_year=yr.min(),
                worst_year_at=int(yr.idxmin()), invested=float(w.reindex(net.index).mean()),
                switches=sw, switches_per_yr=sw / yrs, years=yrs)


def block_boot_sharpe_diff(a, b, reps=500, block=126, seed=SEED):
    """Circular block bootstrap of Sharpe(a) - Sharpe(b) on aligned daily returns."""
    a = np.asarray(a); b = np.asarray(b); n = len(a)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    out = np.empty(reps)
    for k in range(reps):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        x, y = a[idx], b[idx]
        out[k] = (x.mean() / x.std() - y.mean() / y.std()) * np.sqrt(ANN)
    return float(np.quantile(out, 0.05)), float(np.quantile(out, 0.95)), float((out <= 0).mean())


def evaluate(close, split, label, do_boot=True):
    lo, hi = split_bounds(split)
    sig, feats = build_signals(close)
    # evaluation starts once EVERY rule at EVERY frequency holds a position (signals valid,
    # first check day passed, 2-day decision/trade lag), so all rules share one window
    bts = {(f, n): run_rule(close, sig[n], f) for f in ("weekly", "monthly") for n in sig.columns}
    first_live = max(bt["net"].first_valid_index() for bt in bts.values())
    start = max(lo, first_live)
    win = (close.index >= start) & (close.index <= hi)
    r = close.pct_change()
    bh = r[win]
    bh_w = pd.Series(1.0, index=bh.index)
    bh_t = pd.Series(0.0, index=bh.index)
    rows, daily = [], {"bh": bh}
    base = metrics(bh, bh_w, bh_t)
    rows.append(dict(asset=label, split=split, rule="buy_hold", freq="-", **base,
                     d_sharpe=0.0, d_maxdd=0.0, d_cagr=0.0, ci05=np.nan, ci95=np.nan, p_le0=np.nan))
    for freq in ("weekly", "monthly"):
        for name in sig.columns:
            b = bts[(freq, name)][win]
            assert b["net"].notna().all(), (name, freq)
            m = metrics(b["net"], b["w"], b["trade"])
            ci = block_boot_sharpe_diff(b["net"].to_numpy(), bh.to_numpy()) if do_boot else (np.nan,) * 3
            rows.append(dict(asset=label, split=split, rule=name, freq=freq, **m,
                             d_sharpe=m["sharpe"] - base["sharpe"], d_maxdd=m["maxdd"] - base["maxdd"],
                             d_cagr=m["cagr"] - base["cagr"], ci05=ci[0], ci95=ci[1], p_le0=ci[2]))
            daily[f"{name}|{freq}"] = b["net"]
    return pd.DataFrame(rows), pd.DataFrame(daily), sig, feats, win


def decade_table(daily, label, split):
    rows = []
    dec = (daily.index.year // 10) * 10
    for col in daily.columns:
        for d, g in daily[col].groupby(dec):
            g = g.dropna()
            if len(g) < 120:
                continue
            eq = (1 + g).cumprod()
            rows.append(dict(asset=label, split=split, decade=f"{d}s", series=col,
                             cagr=eq.iloc[-1] ** (ANN / len(g)) - 1,
                             maxdd=(eq / eq.cummax() - 1).min(),
                             sharpe=g.mean() / g.std() * np.sqrt(ANN)))
    return pd.DataFrame(rows)


def crash_table(close, daily, split, label):
    rows = []
    for name, a, b in CRASHES.get(split, []):
        seg = close[(close.index >= a) & (close.index <= b)]
        if len(seg) < 20:
            continue
        tr = seg.idxmin()
        pk = seg[seg.index <= tr].idxmax()
        it = close.index.get_loc(tr)
        rb_end = close.index[min(it + 252, len(close) - 1)]
        for col in daily.columns:
            x = daily[col]
            dn = (1 + x[(x.index > pk) & (x.index <= tr)]).prod() - 1
            up = (1 + x[(x.index > tr) & (x.index <= rb_end)]).prod() - 1
            rows.append(dict(asset=label, split=split, crash=name, peak=str(pk.date()), trough=str(tr.date()),
                             rebound_end=str(rb_end.date()), series=col, crash_ret=dn, rebound_ret=up))
    df = pd.DataFrame(rows)
    if len(df):
        bh = df[df.series == "bh"].set_index("crash")
        df["crash_avoided"] = 1 - df["crash_ret"] / df["crash"].map(bh["crash_ret"])
        df["rebound_captured"] = df["rebound_ret"] / df["crash"].map(bh["rebound_ret"])
    return df


# ---------------------------------------------------------------------- risk flags
def flag_table(close, sig, feats, win, split, label):
    """P(big drop next period | flag) vs | no flag). Flag read at close of a check day t;
    outcome = close t -> close of the next check day (non-overlapping periods)."""
    flags = {
        "below_ma200": sig["ma200"] == 0,
        "below_ma10m": sig["ma10m"] == 0,
        "tsm12_neg": sig["tsm12"] == 0,
        "tsm3_neg": sig["tsm3"] == 0,
        "dd_gt10": sig["dd10"] == 0,
        "vol_gt20": feats["vol"] > 0.20,
        "vol_gt30": feats["vol"] > 0.30,
        "below_ma200_and_vol_gt20": (sig["ma200"] == 0) & (feats["vol"] > 0.20),
        "vote3_off": sig["vote3"] == 0,
    }
    rows = []
    for freq, thr in (("weekly", -0.03), ("monthly", -0.05)):
        chk = check_mask(close.index, freq)
        c = close[chk]
        fwd = c.shift(-1) / c - 1
        keep = win[chk] & fwd.notna().to_numpy()
        valid = sig.dropna().index
        keep &= c.index.isin(valid)
        f_ret = fwd[keep]
        big = f_ret < thr
        gain = f_ret > -thr
        for nm, fl in flags.items():
            on = fl.reindex(f_ret.index).fillna(False).astype(bool)
            for state, m in (("flag", on), ("no_flag", ~on)):
                n = int(m.sum())
                rows.append(dict(asset=label, split=split, freq=freq, threshold=thr, flag=nm, state=state,
                                 n=n, n_drop=int(big[m].sum()),
                                 p_drop=float(big[m].mean()) if n else np.nan,
                                 p_bigup=float(gain[m].mean()) if n else np.nan,
                                 mean_ret=float(f_ret[m].mean()) if n else np.nan,
                                 sd_ret=float(f_ret[m].std()) if n > 1 else np.nan))
        rows.append(dict(asset=label, split=split, freq=freq, threshold=thr, flag="ALL", state="all",
                         n=int(len(f_ret)), n_drop=int(big.sum()), p_drop=float(big.mean()),
                         p_bigup=float(gain.mean()), mean_ret=float(f_ret.mean()), sd_ret=float(f_ret.std())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- tables
def fmt_pct(x, d=1):
    return "" if pd.isna(x) else f"{100 * x:.{d}f}%"


def md_variants(v):
    out = ["| rule | freq | CAGR | vol | Sharpe | dSharpe [90% CI] | maxDD | worst yr | invested | switches/yr |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in v.iterrows():
        ci = "" if pd.isna(r.ci05) else f" [{r.ci05:+.2f}, {r.ci95:+.2f}]"
        out.append(f"| {r.rule} | {r.freq} | {fmt_pct(r.cagr)} | {fmt_pct(r.vol)} | {r.sharpe:.2f} | "
                   f"{r.d_sharpe:+.2f}{ci} | {fmt_pct(r.maxdd)} | {fmt_pct(r.worst_year)} ({r.worst_year_at}) | "
                   f"{fmt_pct(r.invested, 0)} | {r.switches_per_yr:.1f} |")
    return "\n".join(out)


def run_asset(name, split, label, do_boot, tag):
    close = load_close(name, split)
    v, daily, sig, feats, win = evaluate(close, split, label, do_boot)
    dec = decade_table(daily, label, split)
    cr = crash_table(close, daily, split, label)
    fl = flag_table(close, sig, feats, win, split, label)
    v.to_csv(os.path.join(RESULTS, f"timing_variants_{tag}.csv"), index=False)
    dec.to_csv(os.path.join(RESULTS, f"timing_decades_{tag}.csv"), index=False)
    cr.to_csv(os.path.join(RESULTS, f"timing_crashes_{tag}.csv"), index=False)
    fl.to_csv(os.path.join(RESULTS, f"timing_flags_{tag}.csv"), index=False)
    md = [f"## {label} {split} ({daily.index[0].date()} .. {daily.index[-1].date()})", "", md_variants(v), ""]
    with open(os.path.join(RESULTS, f"timing_tables_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return v, daily, close


def sanity(close, split):
    """Engine checks on DEV only: (1) a perfect-foresight cheat must look absurd, (2) a
    coin-flip signal must look like nothing, (3) the lag matters (same-close vs next-close)."""
    lo, hi = split_bounds(split)
    r = close.pct_change()
    out = {}
    # cheat: desired weight at close t = 1 if return t+2 > 0 (exactly offsets the 2-day lag)
    cheat = (r.shift(-2) > 0).astype(float)
    bt = run_rule(close, cheat, "daily")
    win = (close.index >= "1930-01-01") & (close.index <= hi)
    out["cheat_daily_sharpe"] = float(metrics(bt["net"][win], bt["w"][win], bt["trade"][win])["sharpe"])
    rng = np.random.default_rng(SEED)
    rs = []
    for k in range(20):
        coin = pd.Series(rng.integers(0, 2, len(close)).astype(float), index=close.index)
        bt = run_rule(close, coin, "monthly")
        rs.append(metrics(bt["net"][win], bt["w"][win], bt["trade"][win])["sharpe"])
    out["coinflip_monthly_sharpe_mean"] = float(np.mean(rs))
    out["coinflip_monthly_sharpe_sd"] = float(np.std(rs))
    out["buyhold_sharpe_same_window"] = float(metrics(r[win], r[win] * 0 + 1, r[win] * 0)["sharpe"])
    with open(os.path.join(RESULTS, f"timing_sanity_{split}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


def select(v_split, split):
    """Pre-registered: the DEV winner is the variant with the highest DEV Sharpe (net).
    A variant counts as REAL only if dSharpe > 0 AND dMaxDD > 0 (shallower) on DEV, on
    DEV post-1950, and on this split."""
    dev = pd.read_csv(os.path.join(RESULTS, "timing_variants_dev.csv"))
    p50 = pd.read_csv(os.path.join(RESULTS, "timing_dev_post1950.csv"))
    bh50 = p50[p50.series == "bh"].iloc[0]
    p50["key"] = p50.series
    p50["p50_dsharpe"] = p50.sharpe - bh50.sharpe
    p50["p50_dmaxdd"] = p50.maxdd - bh50.maxdd
    key = lambda d: d.rule + "|" + d.freq
    dev = dev[dev.rule != "buy_hold"].assign(key=key)
    vs = v_split[v_split.rule != "buy_hold"].assign(key=key)
    m = dev[["key", "sharpe", "d_sharpe", "d_maxdd", "d_cagr"]].rename(
        columns=lambda c: c if c == "key" else "dev_" + c)
    m = m.merge(p50[["key", "p50_dsharpe", "p50_dmaxdd"]], on="key")
    m = m.merge(vs[["key", "sharpe", "d_sharpe", "d_maxdd", "d_cagr", "ci05", "p_le0"]].rename(
        columns=lambda c: c if c == "key" else f"{split}_" + c), on="key")
    m["real"] = ((m.dev_d_sharpe > 0) & (m.dev_d_maxdd > 0) & (m.p50_dsharpe > 0) & (m.p50_dmaxdd > 0)
                 & (m[f"{split}_d_sharpe"] > 0) & (m[f"{split}_d_maxdd"] > 0))
    m["dev_rank"] = m.dev_sharpe.rank(ascending=False).astype(int)
    m = m.sort_values("dev_rank")
    m.to_csv(os.path.join(RESULTS, f"timing_selection_{split}.csv"), index=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "val", "test"), required=True)
    ap.add_argument("--coordinator", action="store_true", help="required for --split test")
    ap.add_argument("--no-boot", action="store_true")
    a = ap.parse_args()
    if a.split == "test" and not a.coordinator:
        sys.exit("refusing: --split test is coordinator-only (add --coordinator)")
    os.makedirs(RESULTS, exist_ok=True)
    boot = not a.no_boot
    summary = {}
    v, daily, close = run_asset("_GSPC", a.split, "GSPC", boot, a.split)
    summary["GSPC"] = v.to_dict(orient="records")
    if a.split == "dev":
        # robustness: post-1950 only (drops the Depression), same rules
        sub = daily[daily.index >= "1950-01-01"]
        rows = []
        for col in sub.columns:
            m = metrics(sub[col], pd.Series(1.0, index=sub.index), pd.Series(0.0, index=sub.index))
            rows.append(dict(series=col, cagr=m["cagr"], vol=m["vol"], sharpe=m["sharpe"], maxdd=m["maxdd"]))
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "timing_dev_post1950.csv"), index=False)
        sanity(close, a.split)
    if a.split in ("val", "test"):
        for nm in ("SPY", "QQQ"):
            run_asset(nm, a.split, nm, boot, f"{a.split}_{nm}")
        select(v, a.split)
    with open(os.path.join(RESULTS, f"timing_summary_{a.split}.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(f"done {a.split}; GSPC window {daily.index[0].date()}..{daily.index[-1].date()}")


if __name__ == "__main__":
    main()
