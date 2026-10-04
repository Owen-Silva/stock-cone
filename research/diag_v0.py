"""Diagnostics of the v0 engine's failure modes (report-only lane, diag-01).

Usage:
    python research/diag_v0.py collect   # walk-forward v0 records -> research/results/diag_v0_parts/
    python research/diag_v0.py analyze   # tables -> research/results/diag_v0_tables.md (+ stdout)

Data rules (research/PLAN.md): TRAIN tickers only (md5 of upper-cased ticker, even
last hex digit) and as-of dates before 2019-01-01. Stricter than PLAN: an outcome is
recorded only if the horizon END is also before 2019-01-01, so no 2019+ price is read
as an outcome. Fully offline: prices come from data/prices/*.csv via Engine(raw=...).
"""
import hashlib
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import engine as E            # noqa: E402
import universe as U          # noqa: E402

RES = os.path.join(HERE, "results")
# Per-ticker parts live OUTSIDE the repo by default: research/results/ was wiped by
# another process once mid-run (2026-10-03 23:25) and took 56 finished parts with it.
PARTS = os.environ.get("DIAG_PARTS") or os.path.join(
    os.environ.get("LOCALAPPDATA", ROOT), "Temp", "stock-cone-diag_v0_parts")
CUTOFF = (pd.Timestamp("2019-01-01") - pd.Timestamp("1970-01-01")).days
VAL_START = (pd.Timestamp("2012-01-01") - pd.Timestamp("1970-01-01")).days
STRIDE = 10
Q19 = np.round(np.arange(0.05, 0.951, 0.05), 2)
Z19 = np.array([-1.6448536, -1.2815516, -1.0364334, -0.8416212, -0.6744898, -0.5244005,
                -0.3853205, -0.2533471, -0.1256613, 0.0, 0.1256613, 0.2533471, 0.3853205,
                0.5244005, 0.6744898, 0.8416212, 1.0364334, 1.2815516, 1.6448536])


def is_train(t):
    return int(hashlib.md5(t.upper().encode()).hexdigest()[-1], 16) % 2 == 0


def load_raw():
    raw = {}
    for fn in os.listdir(E.PRICE_DIR):
        if not fn.endswith(".csv"):
            continue
        t = "^GSPC" if fn == "_GSPC.csv" else fn[:-4]
        df = pd.read_csv(os.path.join(E.PRICE_DIR, fn), index_col=0, parse_dates=True)
        raw[t] = E._clean(df["close"], df["volume"], df.index)
    return raw


def market_series(raw):
    """SPY (spliced with ^GSPC) trailing 63d log return and 21d annualised vol, by date."""
    spy, spl = raw["SPY"], raw["^GSPC"]
    r_spy = np.log(spy["close"]).diff()
    r_old = np.log(spl["close"]).diff()
    first = spy.index[0]
    r = pd.concat([r_old[r_old.index <= first], r_spy[r_spy.index > first]])
    r = r[~r.index.duplicated(keep="first")].fillna(0.0)
    lvl = r.cumsum()
    return pd.DataFrame({"mkt_ret63": lvl - lvl.shift(63),
                         "mkt_vol21": r.rolling(21).std() * math.sqrt(252)})


def targets():
    groups = dict(U.UNIVERSE)
    out = []
    for t, g in U.UNIVERSE:
        if is_train(t):
            prim = [x for x in g if x != "watchlist"]
            out.append((t, prim[0] if prim else "watchlist"))
    return out, groups


_CACHE = {}


def _engine():
    if "eng" not in _CACHE:
        raw = load_raw()
        eng = E.Engine(raw=raw, quiet=True)
        eng.warm_up()
        _CACHE["eng"] = eng
        _CACHE["mk"] = market_series(raw)
    return _CACHE["eng"], _CACHE["mk"]


def collect_one(args):
    t, grp = args
    path = os.path.join(PARTS, f"{t}.csv")
    if os.path.exists(path):
        return t, "cached"
    eng, mk = _engine()
    try:
        td = eng.get_ticker(t)
    except E.ForecastError as e:
        return t, f"skip: {e.msg}"
    pool = eng.pool_for(t)
    self_id = E.POOL.index(t) if t in E.POOL else len(E.POOL)
    n = len(td.day)
    lr = np.r_[np.nan, np.diff(td.logc)]
    mkt_on = mk.reindex(td.dates, method="ffill")
    ok = np.flatnonzero(td.valid)
    if ok.size == 0:
        return t, "skip: no valid rows"
    lo = max(int(ok[0]), 252)
    rows = []
    for p in range(lo, n, STRIDE):
        D = td.day[p]
        if D >= CUTOFF:
            break
        if not td.valid[p]:
            continue
        hs = [h for h in E.HORIZONS if p + h < n and td.day[p + h] < CUTOFF]
        if not hs:
            continue
        q = td.X[p]
        d2 = pool.dist2(q)
        pre = np.argpartition(d2, E.PRE - 1)[:E.PRE] if pool.n > E.PRE else None
        sig_now = td.sig[p]
        logc0 = td.logc[: p + 1]
        hi252 = np.max(td.logc[max(0, p - 251): p + 1])
        for h in hs:
            sel, d, w = eng._select(pool, q, d2, D, h, pre=pre)
            if len(sel) < E.K_MIN:
                continue
            rawp, scaled, ratio = eng._paths(pool, sel, h, sig_now, end_only=True)
            s, rr = scaled[:, 0], rawp[:, 0]
            q19 = E.wquantile(s, w, Q19)
            half = len(sel) // 2
            qh = E.wquantile(s[:half], w[:half], Q19)
            rq50 = E.wquantile(rr, w, np.array([0.5]))[0]
            fwd = lr[p + 1: p + h + 1]
            c = td.close[: p + 1]
            base = float(np.mean(c[h:] > c[:-h])) if len(c) > h else np.nan
            hret = logc0[h:] - logc0[:-h]
            rec = {
                "ticker": t, "group": grp, "in_pool": int(t in E.POOL), "date": E.day_str(D),
                "split": "VAL" if D >= VAL_START else "DEV", "h": h,
                "ret": float(td.logc[p + h] - td.logc[p]),
                "sig": float(sig_now),
                "rv": float(np.sqrt(np.mean(fwd ** 2))),
                "rv_sd": float(np.std(fwd, ddof=1)) if h > 2 else np.nan,
                "p_up": float(np.sum(w * (s > 0)) / np.sum(w)),
                "p_up_half": float(np.sum(w[:half] * (s[:half] > 0)) / np.sum(w[:half])),
                "base_up": base,
                "wf_mu_day": float((td.logc[p] - td.logc[0]) / p),
                "wf_mean_h": float(np.mean(hret)) if hret.size else np.nan,
                "an_mean_scaled": float(np.sum(w * s) / np.sum(w)),
                "an_mean_raw": float(np.sum(w * rr) / np.sum(w)),
                "an_q50_raw": float(rq50),
                "ratio_med": float(np.median(ratio)),
                "ratio_wmean_log": float(np.sum(w * np.log(ratio)) / np.sum(w)),
                "clip_share": float(np.mean((ratio <= E.SCALE_CLIP[0]) | (ratio >= E.SCALE_CLIP[1]))),
                "n_an": int(len(sel)),
                "d_min": float(d[0]), "d_med": float(np.median(d)), "d_k": float(d[-1]),
                "d_half": float(d[half - 1]),
                "same_share": float(np.mean(pool.tid[sel] == self_id)),
                "n_tickers": int(len(np.unique(pool.tid[sel]))),
                "an_age_yrs_med": float(np.median((D - pool.day[sel]) / 365.25)),
                "dd252": float(td.logc[p] - hi252),
                "ret63": float(td.logc[p] - td.logc[p - 63]),
                "mkt_ret63": float(mkt_on["mkt_ret63"].iloc[p]),
                "mkt_vol21": float(mkt_on["mkt_vol21"].iloc[p]),
            }
            for j, f in enumerate(E.FEATURES):
                rec["x_" + f] = float(q[j])
            for j, qq in enumerate(Q19):
                rec[f"q{int(round(qq * 100)):02d}"] = float(q19[j])
                rec[f"h{int(round(qq * 100)):02d}"] = float(qh[j])
            rows.append(rec)
    os.makedirs(PARTS, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path + ".tmp", index=False, float_format="%.7g")
    os.replace(path + ".tmp", path)
    return t, f"{len(df)} records"


def check_replication(t="AAPL"):
    """My loop must reproduce Engine.walk_forward's analog quantiles exactly."""
    raw = load_raw()
    eng = E.Engine(raw=raw, quiet=True)
    eng.warm_up()
    td = eng.get_ticker(t)
    df = pd.read_csv(os.path.join(PARTS, f"{t}.csv"))
    ds = sorted(set(df.date))
    pick = [ds[len(ds) // 10], ds[len(ds) // 2], ds[-30]]
    pos = [int(np.searchsorted(td.dates, pd.Timestamp(x))) for x in pick]
    ref = eng.walk_forward(t, E.HORIZONS, pos)
    worst = 0.0
    for h in E.HORIZONS:
        for r in ref[h]:
            mine = df[(df.date == r["date"]) & (df.h == h)]
            if mine.empty:
                continue
            a = np.array(r["aq"])
            b = mine[["q10", "q25", "q50", "q75", "q90"]].to_numpy()[0]
            worst = max(worst, float(np.max(np.abs(a - b))))
    return worst, sum(len(v) for v in ref.values())


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_all():
    fs = [os.path.join(PARTS, f) for f in os.listdir(PARTS) if f.endswith(".csv")]
    fs = [f for f in fs if os.path.getsize(f) > 10]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df


def pinball(y, Q, qs):
    u = y[:, None] - Q
    return np.mean(np.maximum(qs * u, (qs - 1) * u), axis=1)


def add_metrics(df):
    qc = [f"q{int(round(q * 100)):02d}" for q in Q19]
    hc = [f"h{int(round(q * 100)):02d}" for q in Q19]
    y = df["ret"].to_numpy()
    sh = (df["sig"] * np.sqrt(df["h"])).to_numpy()
    Q = df[qc].to_numpy()
    H = df[hc].to_numpy()
    N0 = Z19[None, :] * sh[:, None]
    ND = N0 + (df["wf_mu_day"] * df["h"]).to_numpy()[:, None]
    df["pb_v0"] = pinball(y, Q, Q19) / sh
    df["pb_half"] = pinball(y, H, Q19) / sh
    df["pb_n0"] = pinball(y, N0, Q19) / sh
    df["pb_nd"] = pinball(y, ND, Q19) / sh
    # v0 with its median shift removed (cone recentred on zero drift)
    df["pb_v0_c0"] = pinball(y, Q - df["q50"].to_numpy()[:, None], Q19) / sh
    df["in80"] = ((y >= df["q10"]) & (y <= df["q90"])).astype(float)
    df["in50"] = ((y >= df["q25"]) & (y <= df["q75"])).astype(float)
    df["in80_half"] = ((y >= df["h10"]) & (y <= df["h90"])).astype(float)
    df["in80_n0"] = (np.abs(y) <= 1.2815516 * sh).astype(float)
    df["below10"] = (y < df["q10"]).astype(float)
    df["above90"] = (y > df["q90"]).astype(float)
    df["below05"] = (y < df["q05"]).astype(float)
    df["above95"] = (y > df["q95"]).astype(float)
    df["w80"] = df["q90"] - df["q10"]
    df["w80_sig"] = df["w80"] / (2 * 1.2815516 * sh)      # cone width vs naive cone
    df["up"] = (y > 0).astype(float)
    df["z"] = y / sh
    # PIT within the cone (linear in the 19 quantiles, flat tails beyond)
    df["pit"] = [np.interp(v, row, Q19, left=0.0, right=1.0) for v, row in zip(y, Q)]
    df["rv_ratio"] = df["rv"] / df["sig"]
    return df


def terc(s):
    return pd.qcut(s, 3, labels=["low", "mid", "high"])


def cover_table(df, col, label):
    g = df.groupby([col, "h"], observed=True)
    t = g.agg(n=("in80", "size"), in80=("in80", "mean"), in50=("in50", "mean"),
              below10=("below10", "mean"), above90=("above90", "mean"),
              w_vs_naive=("w80_sig", "median"),
              skill=("pb_v0", "mean"), naive=("pb_n0", "mean")).reset_index()
    t["skill"] = 1 - t["skill"] / t["naive"]
    t = t.drop(columns="naive")
    t.insert(0, "breakdown", label)
    t = t.rename(columns={col: "bucket"})
    return t


def md(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def logit_fit(X, y, offset, lam=1.0, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = offset + X @ b
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        g = X.T @ (y - p) - lam * b
        Hm = (X * W[:, None]).T @ X + lam * np.eye(X.shape[1])
        step = np.linalg.solve(Hm, g)
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def analyze():
    df = add_metrics(load_all())
    out = []

    def sec(title, table=None, text=None):
        out.append(f"\n## {title}\n")
        if text:
            out.append(text + "\n")
        if table is not None:
            out.append(md(table) + "\n")

    # --- sample
    smp = df.groupby(["split", "h"]).agg(records=("ret", "size"), tickers=("ticker", "nunique"),
                                         first=("date", "min"), last=("date", "max")).reset_index()
    smp["first"] = smp["first"].dt.date.astype(str)
    smp["last"] = smp["last"].dt.date.astype(str)
    sec("Sample", smp)

    # --- headline
    hd = df.groupby(["split", "h"]).agg(in80=("in80", "mean"), in50=("in50", "mean"),
                                        in80_naive=("in80_n0", "mean"),
                                        pb_v0=("pb_v0", "mean"), pb_n0=("pb_n0", "mean"),
                                        pb_nd=("pb_nd", "mean"), pb_v0_c0=("pb_v0_c0", "mean"),
                                        pb_half=("pb_half", "mean")).reset_index()
    for c in ("pb_v0", "pb_nd", "pb_v0_c0", "pb_half"):
        hd["skill_" + c[3:]] = 1 - hd[c] / hd["pb_n0"]
    hd = hd.drop(columns=["pb_v0", "pb_nd", "pb_v0_c0", "pb_half"])
    sec("Headline (scaled-pinball skill vs naive0; v0_c0 = v0 recentred to zero median; half = closest 125 analogs)", hd)

    # --- Q1 drift
    ann = 252 / df["h"]
    df["impl_med_ann"] = df["q50"] * ann
    df["impl_mean_ann"] = df["an_mean_scaled"] * ann
    df["act_ann"] = df["ret"] * ann
    df["own_ann"] = df["wf_mu_day"] * 252
    df["raw_med_ann"] = df["an_q50_raw"] * ann

    def drift(g):
        return pd.Series({"n": len(g), "v0_median": g["impl_med_ann"].mean(),
                          "v0_mean": g["impl_mean_ann"].mean(),
                          "analog_raw_median": g["raw_med_ann"].mean(),
                          "actual_mean": g["act_ann"].mean(),
                          "actual_median": g["act_ann"].median(),
                          "own_wf_mean": g["own_ann"].mean(),
                          "corr_v0med_actual": np.corrcoef(g["q50"], g["ret"])[0, 1]})
    sec("Q1 drift by split x horizon (annualised log return)",
        df.groupby(["split", "h"]).apply(drift, include_groups=False).reset_index())
    sec("Q1 drift by stock group (all splits)",
        df.groupby(["group", "h"]).apply(drift, include_groups=False).reset_index())
    sec("Q1 drift by pool membership (in_pool=1: the target is itself an analog source)",
        df.groupby(["in_pool", "h"]).apply(drift, include_groups=False).reset_index())
    df["era"] = pd.cut(df["year"], [1900, 1990, 1995, 2000, 2003, 2008, 2010, 2013, 2016, 2019],
                       labels=["<1990", "90-94", "95-99", "00-02", "03-07", "08-09", "10-12", "13-15", "16-18"],
                       right=False)
    sec("Q1 drift by era, h=21",
        df[df.h == 21].groupby("era", observed=True).apply(drift, include_groups=False).reset_index())
    # bias vs median: does the median sign/level of v0 explain anything?
    df["bias_med"] = df["q50"] - df["ret"]

    # --- Q2 width
    tabs = []
    df["vol_t"] = df.groupby("h")["sig"].transform(terc)
    df["ret63_t"] = df.groupby("h")["ret63"].transform(terc)
    df["dd_b"] = pd.cut(df["dd252"], [-10, -0.30, -0.15, -0.05, 0.001],
                        labels=["<-30%", "-30..-15%", "-15..-5%", ">-5%"])
    df["mret_t"] = df.groupby("h")["mkt_ret63"].transform(terc)
    df["mvol_t"] = df.groupby("h")["mkt_vol21"].transform(terc)
    df["mkt_regime"] = np.where(df["mkt_ret63"] < 0, "SPY63<0", "SPY63>=0") + "/" + \
        np.where(df["mkt_vol21"] > 0.20, "vol>20%", "vol<=20%")
    for col, lab in (("vol_t", "own trailing vol tercile"), ("dd_b", "drawdown from 52w high"),
                     ("ret63_t", "own trailing 63d return tercile"), ("mret_t", "SPY 63d return tercile"),
                     ("mvol_t", "SPY 21d vol tercile"), ("mkt_regime", "market regime"),
                     ("era", "era"), ("split", "split"), ("group", "stock group")):
        tabs.append(cover_table(df, col, lab))
    sec("Q2 coverage breakdowns (in80 target 0.80, below10/above90 target 0.10 each; "
        "w_vs_naive = median cone width / naive cone width)", pd.concat(tabs, ignore_index=True))

    # --- Q3 vol scaling
    rows = []
    for h, g in df.groupby("h"):
        r = g["rv_ratio"]
        lo_ = g["rv_ratio"] < 1
        rows.append({"h": h, "n": len(g), "p10": r.quantile(.1), "p25": r.quantile(.25),
                     "median": r.median(), "mean": r.mean(), "p75": r.quantile(.75), "p90": r.quantile(.9),
                     "share_rv>1.25x": (r > 1.25).mean(), "share_rv<0.8x": (r < 0.8).mean(),
                     "corr_log": np.corrcoef(np.log(g["rv"].clip(1e-6)), np.log(g["sig"]))[0, 1],
                     "in80|rv<sig": g.loc[lo_, "in80"].mean(), "in80|rv>sig": g.loc[~lo_, "in80"].mean(),
                     "in80|rv>1.5sig": g.loc[r > 1.5, "in80"].mean()})
    sec("Q3 realised forward vol / trailing 63d vol (the vol v0 scales to)", pd.DataFrame(rows))
    df["rvr_t"] = pd.cut(df["rv_ratio"], [0, 0.7, 0.9, 1.1, 1.4, 2.0, 99])
    sec("Q3 coverage by realised/forecast vol bucket",
        cover_table(df, "rvr_t", "rv/sig bucket").drop(columns=["breakdown"]))
    # does an analog-implied vol do better? compare rv with sig and with v0 cone width
    rows = []
    for h, g in df.groupby("h"):
        cone_sig = g["w80"] / (2 * 1.2815516 * np.sqrt(h))
        rows.append({"h": h,
                     "MAE log(rv/sig63)": np.mean(np.abs(np.log(g["rv"].clip(1e-6) / g["sig"]))),
                     "MAE log(rv/cone_sig)": np.mean(np.abs(np.log(g["rv"].clip(1e-6) / cone_sig))),
                     "median rv/cone_sig": np.median(g["rv"] / cone_sig),
                     "median cone width/naive": g["w80_sig"].median(),
                     "ratio_med (vol now/analog vol)": g["ratio_med"].median(),
                     "clip_share mean": g["clip_share"].mean(),
                     "corr(miss, log rv/sig)": np.corrcoef(1 - g["in80"], np.log(g["rv_ratio"].clip(1e-6)))[0, 1]})
    sec("Q3 vol forecast error: trailing vol vs the cone's own implied sigma", pd.DataFrame(rows))

    # --- Q4 tails
    rows = []
    for h, g in df.groupby("h"):
        z = g["z"]
        k = ((z - z.mean()) ** 4).mean() / z.var() ** 2
        pit = g["pit"]
        rows.append({"h": h, "below10": g["below10"].mean(), "above90": g["above90"].mean(),
                     "below05": g["below05"].mean(), "above95": g["above95"].mean(),
                     "pit<=0.01": (pit <= 0.0).mean(), "pit>=0.99": (pit >= 1.0).mean(),
                     "z_kurtosis": k, "z_skew": ((z - z.mean()) ** 3).mean() / z.std() ** 3,
                     "|z|>2 share": (z.abs() > 2).mean(), "normal |z|>2": 0.0455,
                     "|z|>3 share": (z.abs() > 3).mean(), "normal |z|>3": 0.0027})
    sec("Q4 tails (z = ret / (sig63*sqrt h); pit = position of outcome inside the 19-quantile cone)",
        pd.DataFrame(rows))
    pt = df.groupby("h")["pit"].apply(lambda s: pd.Series(np.histogram(s, bins=np.linspace(0, 1, 11))[0] / len(s))).unstack()
    pt.columns = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
    sec("Q4 PIT histogram (uniform = 0.100 per decile)", pt.reset_index())

    # --- Q5 analog quality
    rows = []
    for h, g in df.groupby("h"):
        rows.append({"h": h, "d_min med": g["d_min"].median(), "d_med med": g["d_med"].median(),
                     "d_K med": g["d_k"].median(), "d_K p90": g["d_k"].quantile(.9),
                     "same_ticker share": g["same_share"].mean(),
                     "same share | in_pool": g.loc[g.in_pool == 1, "same_share"].mean(),
                     "same share | not in pool": g.loc[g.in_pool == 0, "same_share"].mean(),
                     "distinct tickers": g["n_tickers"].median(),
                     "analog age yrs (med)": g["an_age_yrs_med"].median()})
    sec("Q5 analog distances (12-dim standardised feature space) and composition", pd.DataFrame(rows))
    df["dk_t"] = df.groupby("h")["d_k"].transform(terc)
    sec("Q5 quality by K-th analog distance tercile (far = poor matches)",
        cover_table(df, "dk_t", "d_K tercile").drop(columns=["breakdown"]))
    rows = []
    for (s, h), g in df.groupby(["split", "h"]):
        rows.append({"split": s, "h": h, "skill_full": 1 - g["pb_v0"].mean() / g["pb_n0"].mean(),
                     "skill_half": 1 - g["pb_half"].mean() / g["pb_n0"].mean(),
                     "in80_full": g["in80"].mean(), "in80_half": g["in80_half"].mean(),
                     "w80_full": g["w80"].mean(), "w80_half": (g["h90"] - g["h10"]).mean()})
    sec("Q5 full K=250 vs closest half (125)", pd.DataFrame(rows))

    # --- Q6 baseline fairness
    rows = []
    per = []
    for (t, h), g in df.groupby(["ticker", "h"]):
        up = g["up"].to_numpy()
        hit = np.mean((g["q50"] > 0).to_numpy() == (up > 0))
        hind = max(up.mean(), 1 - up.mean())
        wf = np.mean((g["base_up"] >= 0.5).to_numpy() == (up > 0))
        per.append({"ticker": t, "h": h, "n": len(g), "v0_hit": hit, "hindsight": hind, "wf_majority": wf})
    per = pd.DataFrame(per)
    per = per[per.n >= 30]
    for h, g in per.groupby("h"):
        rows.append({"h": h, "tickers": len(g), "v0_hit": g.v0_hit.mean(), "base_hindsight": g.hindsight.mean(),
                     "base_walkforward": g.wf_majority.mean(),
                     "beat_hindsight (share tickers)": (g.v0_hit > g.hindsight).mean(),
                     "beat_walkforward (share tickers)": (g.v0_hit > g.wf_majority).mean(),
                     "v0-hind (pts)": (g.v0_hit - g.hindsight).mean() * 100,
                     "v0-wf (pts)": (g.v0_hit - g.wf_majority).mean() * 100})
    sec("Q6 direction: hindsight vs walk-forward majority baseline (tickers with >=30 dates)", pd.DataFrame(rows))
    # Brier of v0 p_up vs walk-forward base rate
    rows = []
    for (s, h), g in df.groupby(["split", "h"]):
        bm = np.mean((g["p_up"] - g["up"]) ** 2)
        bb = np.mean((g["base_up"] - g["up"]) ** 2)
        bh = np.mean((g["p_up_half"] - g["up"]) ** 2)
        rows.append({"split": s, "h": h, "brier_v0": bm, "brier_wfbase": bb, "skill_v0": 1 - bm / bb,
                     "skill_half": 1 - bh / bb, "mean p_up": g["p_up"].mean(), "mean base": g["base_up"].mean(),
                     "actual up": g["up"].mean()})
    sec("Q6 Brier of P(up): v0 vs walk-forward base rate", pd.DataFrame(rows))

    # --- Q7 directional signal
    feats = ["x_" + f for f in E.FEATURES]
    rows = []
    for h in E.HORIZONS:
        g = df[df.h == h]
        dev, val = g[g.split == "DEV"], g[g.split == "VAL"]
        lb = lambda b: np.log(np.clip(b, 0.02, 0.98) / (1 - np.clip(b, 0.02, 0.98)))
        bb_val = np.mean((val["base_up"] - val["up"]) ** 2)
        for f in feats + ["ALL12", "v0_p_up"]:
            if f == "v0_p_up":
                Xd = (lb(dev["p_up"]) - lb(dev["base_up"])).to_numpy()[:, None]
                Xv = (lb(val["p_up"]) - lb(val["base_up"])).to_numpy()[:, None]
            elif f == "ALL12":
                Xd, Xv = dev[feats].to_numpy(), val[feats].to_numpy()
            else:
                Xd, Xv = dev[[f]].to_numpy(), val[[f]].to_numpy()
            b = logit_fit(Xd, dev["up"].to_numpy(), lb(dev["base_up"]).to_numpy(), lam=10.0)
            pv = 1 / (1 + np.exp(-(lb(val["base_up"]).to_numpy() + Xv @ b)))
            bm = np.mean((pv - val["up"]) ** 2)
            d = (pv - val["up"]) ** 2 - (val["base_up"] - val["up"]) ** 2
            dd = pd.Series(d.to_numpy()).groupby(val["date"].to_numpy()).mean()
            se = dd.std(ddof=1) / np.sqrt(len(dd))
            hit = np.mean((pv > 0.5) == (val["up"] > 0))
            hit_b = np.mean((val["base_up"] >= 0.5) == (val["up"] > 0))
            rows.append({"h": h, "feature": f, "coef": b[0] if len(b) == 1 else np.nan,
                         "VAL brier skill": 1 - bm / bb_val, "dBrier/SE (neg=better)": d.mean() / se if se > 0 else np.nan,
                         "VAL hit": hit, "VAL hit wf-majority": hit_b})
    sec("Q7 logistic on DEV (offset = walk-forward base-rate logit, ridge 10), scored on VAL. "
        "SE clusters by as-of date.", pd.DataFrame(rows))

    # --- Fix simulations: recentre and rescale v0's own quantiles (no re-search)
    qc = [f"q{int(round(q * 100)):02d}" for q in Q19]
    rows = []
    for h in E.HORIZONS:
        g = df[df.h == h]
        y = g["ret"].to_numpy()
        sh = (g["sig"] * np.sqrt(h)).to_numpy()
        Q = g[qc].to_numpy()
        med = g["q50"].to_numpy()[:, None]
        dev = (g["split"] == "DEV").to_numpy()
        own = (g["wf_mu_day"] * h).to_numpy()[:, None]
        rawmed = g["an_q50_raw"].to_numpy()[:, None]
        centres = {"v0 (analog median)": med, "zero": 0 * med, "own wf mean": own,
                   "unscaled analog median": rawmed,
                   "half analog median": 0.5 * med, "half own wf mean": 0.5 * own}
        n0 = pinball(y, Z19[None, :] * sh[:, None], Q19) / sh
        for cname, c in centres.items():
            base = c + (Q - med)
            # width factor s fitted on DEV so that the 80% interval covers 80% on DEV
            lo_, hi_ = base[:, 1], base[:, 17]
            cen = c[:, 0]
            u = np.where(y >= cen, (y - cen) / np.maximum(hi_ - cen, 1e-9), (cen - y) / np.maximum(cen - lo_, 1e-9))
            s_fit = float(np.quantile(u[dev], 0.80))
            for sname, s in (("1.00", 1.0), (f"DEV-fit {s_fit:.2f}", s_fit)):
                Qs = c + s * (Q - med)
                pb = pinball(y, Qs, Q19) / sh
                inn = (y >= Qs[:, 1]) & (y <= Qs[:, 17])
                for split, m in (("DEV", dev), ("VAL", ~dev)):
                    rows.append({"h": h, "centre": cname, "width x": sname, "split": split,
                                 "skill": 1 - pb[m].mean() / n0[m].mean(), "in80": inn[m].mean(),
                                 "below10": (y[m] < Qs[m, 1]).mean(), "above90": (y[m] > Qs[m, 17]).mean()})
    sec("Fix simulations on v0's own quantiles: change the centre and/or scale the spread "
        "(width factor fitted on DEV only, then applied to VAL)", pd.DataFrame(rows))

    # --- Fix simulation 2: a better vol forecast + empirical (fat-tailed) z quantiles
    rows = []
    for h in E.HORIZONS:
        g = df[df.h == h]
        y = g["ret"].to_numpy()
        sh = (g["sig"] * np.sqrt(h)).to_numpy()
        dev = (g["split"] == "DEV").to_numpy()
        n0 = pinball(y, Z19[None, :] * sh[:, None], Q19) / sh
        X = np.column_stack([np.ones(len(g)), np.log(g["sig"]), g["x_volreg"] * E.FEATURE_SCALE["volreg"],
                             np.log(g["mkt_vol21"].clip(0.03))])
        ly = np.log(g["rv"].clip(1e-5).to_numpy())
        beta, *_ = np.linalg.lstsq(X[dev], ly[dev], rcond=None)
        sig_hat = np.exp(X @ beta)
        c = (g["wf_mu_day"] * h).to_numpy()
        for vname, sv in (("sig63 (v0)", g["sig"].to_numpy()), ("regression", sig_hat)):
            r2 = 1 - np.mean((ly[~dev] - np.log(sv[~dev])) ** 2) / np.var(ly[~dev])
            zs = (y - c) / (sv * np.sqrt(h))
            for qname in ("normal", "empirical DEV"):
                zq = np.quantile(zs[dev], Q19)
                if qname == "normal":
                    zq = Z19 * (np.quantile(zs[dev], 0.9) - np.quantile(zs[dev], 0.1)) / (2 * 1.2815516)
                Qm = c[:, None] + zq[None, :] * (sv * np.sqrt(h))[:, None]
                pb = pinball(y, Qm, Q19) / sh
                inn = (y >= Qm[:, 1]) & (y <= Qm[:, 17])
                for split, m in (("DEV", dev), ("VAL", ~dev)):
                    rows.append({"h": h, "vol": vname, "z quantiles": qname, "split": split,
                                 "VAL R2 log rv": r2, "skill": 1 - pb[m].mean() / n0[m].mean(),
                                 "in80": inn[m].mean(), "below10": (y[m] < Qm[m, 1]).mean(),
                                 "above90": (y[m] > Qm[m, 17]).mean()})
        rows[-1]["beta"] = " ".join(f"{b:.2f}" for b in beta)
    sec("Fix simulation 2: centre = own walk-forward mean; sigma = trailing sig63 or a DEV-fitted "
        "regression log rv ~ log sig63 + log(s21/s252) + log SPY vol21; z quantiles normal (80% width "
        "matched on DEV) or empirical from DEV. No analogs at all.", pd.DataFrame(rows))

    text = "\n".join(out)
    with open(os.path.join(RES, "diag_v0_tables.md"), "w", encoding="utf-8") as fh:
        fh.write("# diag-01 tables (generated by research/diag_v0.py analyze)\n" + text)
    print(text)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "collect":
        os.makedirs(PARTS, exist_ok=True)
        tl, _ = targets()
        only = sys.argv[2:] or None
        if only:
            tl = [x for x in tl if x[0] in only]
        print(f"{len(tl)} TRAIN targets", flush=True)
        from multiprocessing import Pool as MP
        with MP(2) as mp:
            for t, msg in mp.imap_unordered(collect_one, tl):
                print(t, msg, flush=True)
    elif cmd == "check":
        print("max |aq diff| vs Engine.walk_forward:", check_replication(*(sys.argv[2:3] or ['AAPL'])))
    else:
        analyze()


if __name__ == "__main__":
    main()
