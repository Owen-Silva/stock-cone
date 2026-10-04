"""Small, deliberately boring cross-sectional backtester (lane cone-xsec).

Used by research/xsec.py. Nothing here is imported by the app.

Timing (the honest default is lag=1):
  * A signal is read at the CLOSE of signal day s (it may use closes <= s only).
  * lag=0: trade at that same close s (optimistic: you cannot really compute a
    signal from the close and trade at that close).
  * lag=1: trade at the NEXT close s+1 (the number to believe).
  * The position is held until the next rebalance trade (s_next + lag).
Rebalance every `step` trading days (5 = weekly, 21 = monthly).

Portfolios at each rebalance, over the eligible names:
  * top    : top quintile by signal, equal weight, long only
  * bottom : bottom quintile, equal weight
  * ew     : every eligible name, equal weight (the honest benchmark: same
             stocks, same dates, same survivorship bias)
  * ls     : top minus bottom (dollar neutral, no financing cost or rebate)
  * spy    : SPY buy-and-hold over the same periods (^GSPC price index before
             SPY's 1993 start, which omits dividends, ~2-4%/yr)
Costs: COST_BPS per side on traded notional. Turnover is measured against the
drifted weights (holdings move with prices between rebalances), and the first
buy-in counts.

Eligibility at signal time: an actual close on day s, >= MIN_HIST days of
history, 63-day median dollar volume >= MIN_DOLLAR_VOL and <= 10% zero-volume
days. The brief asked for "price > $5"; the cached closes are split- AND
dividend-adjusted, so a $5 test on them would drop NVDA for all of 2012-2018
(adjusted ~$0.33, real ~$13) and most large caps before 2000. Dollar volume is
split-invariant (Yahoo's volume is split-adjusted too), so it is used as the
tiny-name filter instead; min_price is still available as an option.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import datetime as dt
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import engine  # noqa: E402  (only _cache_file/_clean: reads the disk cache, no network)
from research import splits  # noqa: E402

COST_BPS = 10.0
MIN_HIST = 252
MIN_DOLLAR_VOL = 2e6
MAX_ZERO_VOL = 0.10
MIN_NAMES = 20
QUANT = 0.2


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_cached(ticker):
    """Cached daily close/volume for a ticker, straight from data/prices (no refresh)."""
    path = engine._cache_file(ticker)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return engine._clean(df["close"], df["volume"], df.index)


class Panel:
    """Wide close/volume frames on one market calendar, plus the market series."""

    def __init__(self, tickers, end=None):
        end = pd.Timestamp(end) if end is not None else None
        spy, gspc = read_cached("SPY"), read_cached("^GSPC")
        if end is not None:
            spy, gspc = spy[spy.index <= end], gspc[gspc.index <= end]
        frames, self.missing = {}, []
        for t in tickers:
            df = read_cached(t)
            if df is None or df.empty:
                self.missing.append(t)
                continue
            if end is not None:
                df = df[df.index <= end]
            if len(df):                     # tickers listed after `end` simply drop out
                frames[t] = df
        first = min(df.index[0] for df in frames.values())
        cal = gspc.index.union(spy.index)
        cal = cal[cal >= first]
        self.off_calendar = sum(int((~df.index.isin(cal)).sum()) for df in frames.values())
        self.cal = cal
        self.tickers = sorted(frames)
        self.close = pd.DataFrame({t: frames[t]["close"].reindex(cal) for t in self.tickers})
        self.volume = pd.DataFrame({t: frames[t]["volume"].reindex(cal) for t in self.tickers})
        self.pff = self.close.ffill(limit=5)      # prices for P&L (bridges short gaps)
        # Market: SPY (total return), spliced onto ^GSPC (price only) before 1993.
        s = spy["close"].reindex(cal).ffill(limit=5)
        g = gspc["close"].reindex(cal).ffill(limit=5)
        s0 = spy.index[0]
        scale = s.loc[s0] / g.loc[s0]
        self.mkt = s.where(cal >= s0, g * scale)
        self.mkt_is_spy = pd.Series(cal >= s0, index=cal)

    def idx_on_or_before(self, day):
        return int(self.cal.searchsorted(pd.Timestamp(day), side="right")) - 1

    def idx_on_or_after(self, day):
        return int(self.cal.searchsorted(pd.Timestamp(day), side="left"))


def eligibility(panel, min_hist=MIN_HIST, min_dv=MIN_DOLLAR_VOL, max_zero=MAX_ZERO_VOL,
                min_price=None):
    c, v = panel.close, panel.volume
    has = c.notna()
    hist = has.rolling(min_hist + 10, min_periods=1).sum() >= min_hist
    dv = (c * v).where(v > 0)
    med_dv = dv.rolling(63, min_periods=40).median()
    zero = (v.fillna(0) <= 0).astype(float).where(has).rolling(63, min_periods=40).mean()
    el = has & hist & (med_dv >= min_dv) & (zero <= max_zero)
    if min_price:
        el &= c >= min_price
    return el


def split_last_index(panel, split):
    """Last calendar index whose price a split may touch."""
    hi = splits.split_day_range(split)[1]
    return panel.idx_on_or_before(pd.Timestamp(dt.date(1970, 1, 1) + dt.timedelta(days=min(hi, 40000))))


def split_window(panel, split, elig, step, lag, min_names=MIN_NAMES, offset=0):
    """Signal indices for a split. Signal dates lie inside the split's as-of range;
    every exit (next signal + lag) must lie on or before the split's last day, so
    DEV never sees 2012 prices and VAL never sees 2019 prices. The schedule starts
    on the first day with >= min_names eligible names."""
    lo, hi = splits.split_day_range(split)
    epoch = dt.date(1970, 1, 1)
    lo_d = pd.Timestamp(epoch + dt.timedelta(days=max(lo, -25000)))
    hi_d = pd.Timestamp(epoch + dt.timedelta(days=min(hi, 40000)))
    n_el = elig.sum(axis=1)
    ok = n_el[(n_el >= min_names) & (panel.cal >= lo_d)]
    if ok.empty:
        return np.array([], int)
    i0 = panel.idx_on_or_after(ok.index[0])
    i_last = panel.idx_on_or_before(hi_d)          # last price index this split may touch
    idx = np.arange(i0 + offset, len(panel.cal), step)
    # run() uses consecutive pairs (s_k, s_k+1): every trade/exit index s+lag must
    # stay inside the split, which also keeps every signal date inside it.
    return idx[idx + lag <= i_last]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def nw_t(x, lags=None):
    """Newey-West (Bartlett) t-stat of the mean; auto lag floor(4 (n/100)^(2/9))."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    u = x - x.mean()
    s = u @ u / n
    for L in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - L / (lags + 1)) * (u[L:] @ u[:-L]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else float("nan")


def perf(r, ppy):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return dict(ann=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan)
    w = np.cumprod(1 + r)
    ann = w[-1] ** (ppy / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(ppy)
    dd = (w / np.maximum.accumulate(np.concatenate([[1.0], w]))[1:] - 1).min()
    return dict(ann=float(ann), vol=float(vol), sharpe=float(r.mean() * ppy / vol) if vol > 0 else np.nan,
                maxdd=float(min(dd, 0.0)))


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run(panel, signal, elig, sidx, step, lag, cost_bps=COST_BPS, quant=QUANT, min_names=MIN_NAMES):
    """Backtest one signal. `signal` is a DataFrame aligned to panel.close.
    Returns a per-period DataFrame indexed by period-end date."""
    S = signal.reindex(index=panel.cal, columns=panel.tickers).to_numpy(float)
    E = elig.reindex(index=panel.cal, columns=panel.tickers).fillna(False).to_numpy(bool)
    P = panel.pff.to_numpy(float)
    M = panel.mkt.to_numpy(float)
    c = cost_bps / 1e4
    nt = P.shape[1]
    rows = []
    top_counts = np.zeros(nt)
    prev = {k: (np.zeros(nt), np.zeros(nt), 0.0) for k in ("top", "bot", "ew")}  # w, r, R
    for k in range(len(sidx) - 1):
        s, e0, e1 = sidx[k], sidx[k] + lag, sidx[k + 1] + lag
        with np.errstate(invalid="ignore", divide="ignore"):
            r = P[e1] / P[e0] - 1
        sv = S[s]
        valid = E[s] & np.isfinite(sv) & np.isfinite(r)
        n = int(valid.sum())
        if n < min_names:
            continue
        vi = np.flatnonzero(valid)
        order = vi[np.argsort(sv[vi], kind="stable")]
        nq = max(1, int(round(n * quant)))
        sel = {"top": order[-nq:], "bot": order[:nq], "ew": vi}
        top_counts[order[-nq:]] += 1
        rec = {"date": panel.cal[e1], "n": n, "nq": nq,
               "ic": spearman(sv[vi], r[vi]), "spy": M[e1] / M[e0] - 1}
        for key, ix in sel.items():
            w = np.zeros(nt)
            w[ix] = 1.0 / len(ix)
            pw, pr, pR = prev[key]
            drift = pw * (1 + np.nan_to_num(pr)) / (1 + pR) if pw.sum() > 0 else pw
            to = float(np.abs(w - drift).sum())
            R = float(r[ix].mean())
            rec[f"{key}_gross"] = R
            rec[f"{key}_net"] = (1 - c * to) * (1 + R) - 1
            rec[f"to_{key}"] = to
            prev[key] = (w, np.where(w > 0, r, 0.0), R)
        rec["ls_net"] = rec["top_gross"] - rec["bot_gross"] - c * (rec["to_top"] + rec["to_bot"])
        rows.append(rec)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.set_index("date")
    df.attrs["top_counts"] = {t: int(c) for t, c in zip(panel.tickers, top_counts) if c}
    return df


def ic_daily(panel, signal, elig, H, lag, i_from, i_to_exit, min_names=MIN_NAMES):
    """Overlapping IC: every day t in [i_from, ...] whose exit t+lag+H <= i_to_exit.
    Returns (mean, Newey-West t with lag H + auto)."""
    P = panel.pff
    fwd = P.shift(-(lag + H)) / P.shift(-lag) - 1
    s = signal.reindex(index=panel.cal, columns=panel.tickers)
    m = elig & s.notna() & fwd.notna()
    last = i_to_exit - lag - H
    if last < i_from:
        return float("nan"), float("nan")
    m = m.iloc[i_from:last + 1]
    a = s.iloc[i_from:last + 1].where(m).rank(axis=1)
    b = fwd.iloc[i_from:last + 1].where(m).rank(axis=1)
    a = a.sub(a.mean(axis=1), axis=0)
    b = b.sub(b.mean(axis=1), axis=0)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    ic = (num / den).where(m.sum(axis=1) >= min_names)
    x = ic.dropna().to_numpy()
    if len(x) < 30:
        return float("nan"), float("nan")
    auto = int(np.floor(4 * (len(x) / 100) ** (2 / 9)))
    return float(x.mean()), nw_t(x, lags=H + auto)


def summarize(bt, step):
    """One row of metrics from a run() frame."""
    ppy = 252.0 / step
    if bt is None or len(bt) < 10:
        return {}
    top, ew, bot = perf(bt.top_net, ppy), perf(bt.ew_net, ppy), perf(bt.bot_net, ppy)
    ls, spy = perf(bt.ls_net, ppy), perf(bt.spy, ppy)
    gross = perf(bt.top_gross, ppy)
    act = (bt.top_net - bt.ew_net).to_numpy()
    mon = (1 + bt[["top_net", "ew_net"]]).groupby(bt.index.to_period("M")).prod() - 1
    to_ex_first = bt.to_top.iloc[1:]
    return {
        "start": str(bt.index[0].date()), "end": str(bt.index[-1].date()), "periods": len(bt),
        "names_avg": float(bt.n.mean()), "nq_avg": float(bt.nq.mean()),
        "ic_mean": float(bt.ic.mean()), "ic_t": nw_t(bt.ic.to_numpy()),
        "top_ann": top["ann"], "top_gross_ann": gross["ann"], "top_vol": top["vol"],
        "top_sharpe": top["sharpe"], "top_maxdd": top["maxdd"],
        "ew_ann": ew["ann"], "ew_sharpe": ew["sharpe"], "ew_maxdd": ew["maxdd"],
        "excess_ann": top["ann"] - ew["ann"],
        "excess_t": nw_t(act), "excess_ir": float(act.mean() / act.std(ddof=1) * np.sqrt(ppy)),
        "hit_month_vs_ew": float((mon.top_net > mon.ew_net).mean()),
        "turnover_1way_ann": float(to_ex_first.mean() / 2 * ppy),
        "bot_ann": bot["ann"], "ls_ann": ls["ann"], "ls_sharpe": ls["sharpe"], "ls_maxdd": ls["maxdd"],
        "ls_t": nw_t(bt.ls_net.to_numpy()),
        "spy_ann": spy["ann"], "spy_sharpe": spy["sharpe"],
    }


def by_year(bt):
    g = (1 + bt[["top_net", "ew_net", "ls_net", "spy"]]).groupby(bt.index.year).prod() - 1
    g["excess"] = g.top_net - g.ew_net
    ic = bt.ic.groupby(bt.index.year).mean()
    g["ic"] = ic
    return g


# ---------------------------------------------------------------------------
# Sanity checks that the pipe is honest
# ---------------------------------------------------------------------------

def forward_return_signal(panel, step, lag):
    """The deliberately cheating signal: the very return the period will earn."""
    P = panel.pff
    return P.shift(-(step + lag)) / P.shift(-lag) - 1


def random_signal(panel, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal(panel.close.shape), index=panel.cal, columns=panel.tickers)
