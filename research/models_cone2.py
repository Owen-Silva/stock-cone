"""cone2: a calibrated cone = centre (drift) + HAR width (+ calibration) + empirical shape.

Lane cone-combo (2026-10-03/04). Report: research/reports/cone2-01-calibrated-cone.md.

    Q_h = centre_h + s_h * z_h          (19 quantiles of r = ln(P[p+h]/P[p]), q = .05...95)

centre_h  an h-day drift (own walk-forward mean, market/pool mean, James-Stein shrink,
          analog medians, blends ...), see centre().
s_h       the frozen HAR sigma of research/volforecast.py (har_mx_QLr) x sqrt(h), optionally
          times a calibration multiplier (long-run vol floor, walk-forward conformal
          factors from trailing standardised residuals of the stock or of the basket),
          see width().
z_h       19 standardised quantiles frozen from DEV residuals of THIS centre and width
          (weighted so they minimise the harness's scaled pinball), optionally one set per
          regime bin, see shape_fit(). Optional: quantile-average with an analog cone.

Every input is causal: bars <= p for the ticker; other series (market index, the analog
basket) only up to the as-of date; analog rows only if A + h <= as-of (engine._select).

Usage
    python research/models_cone2.py panel dev|val     # build research/results/cone2_panel_<split>.npz
    python research/models_cone2.py research 1                      # stage 1: 40 centres
    python research/models_cone2.py research 2 cone2_c_jspool_mean0 # stage 2: width/shape/ensemble
    python research/models_cone2.py research 3 cone2_c_jspool_mean0 # stage 3: combinations
    python research/models_cone2.py research 4                      # stage 4: own-mean combinations
    python research/models_cone2.py research 5                      # stage 5: post-blend regime factor
    python research/models_cone2.py freeze stage4 cone2_g_own       # -> FINAL_* literals below
    python research/models_cone2.py export stage3 cone2_f_conf_dev  # -> cone2_params.json (harness)
    (each stage fits settings on DEV, scores DEV+VAL, writes cone2_variants_stage*.csv)
    python research/harness.py --models cone2 --split val --workers 2
    python research/harness.py --lookahead --models cone2
"""
import hashlib
import inspect
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import engine                                   # noqa: E402
from research import harness                    # noqa: E402
from research import splits                     # noqa: E402
from research import volforecast as vf          # noqa: E402
from research.harness import QS19, HORIZONS     # noqa: E402
from research.models_base import V0             # noqa: E402

RESULTS = os.path.join(HERE, "results")
ANALOG_DIR = os.path.join(RESULTS, "cone2_analog")
DAY0 = 30000                     # calendar-day axis offset (day 0 = 1887-11-..)
NDAY = 60000                     # covers to ~2034
DEPS = ("research/volforecast.py", "research/models_base.py", "research/harness.py")

_lock = threading.Lock()
_v0 = V0()


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _cum(x):
    x = np.asarray(x, float)
    f = np.isfinite(x)
    return (np.concatenate([[0.0], np.cumsum(np.where(f, x, 0.0))]),
            np.concatenate([[0], np.cumsum(f)]))


def wquantile(x, w, qs):
    """Weighted quantiles (midpoint rule), x and w 1-D, finite."""
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, c, x)


def p_up(Q):
    """P(r > 0) read off sorted quantile rows, linear between them, clipped to [.05, .95]
    (same rule as volforecast._p_up, vectorised)."""
    Q = np.asarray(Q, float)
    out = np.full(len(Q), np.nan)
    fin = np.isfinite(Q).all(axis=1)
    k = (Q < 0).sum(axis=1)
    lo = np.clip(k - 1, 0, 18)
    hi = np.clip(k, 0, 18)
    a, b = Q[np.arange(len(Q)), lo], Q[np.arange(len(Q)), hi]
    with np.errstate(invalid="ignore", divide="ignore"):
        x = QS19[lo] + np.where(b > a, (0.0 - a) / (b - a), 0.0) * 0.05
    v = np.where(k == 0, 0.05, np.where(k == 19, 0.95, x))   # k==0: q05 >= 0 -> P(r<0) small
    out[fin] = 1.0 - v[fin]
    return out


# ---------------------------------------------------------------------------
# Per-context memo: market index mean, basket-wide drift, basket residual stats
# ---------------------------------------------------------------------------

def _memo(ctx):
    with _lock:
        m = getattr(ctx, "_cone2", None)
        if m is None:
            m = {"series": {}, "pool": None}
            ctx._cone2 = m
        return m


class _Series:
    """Full-bar causal arrays for one ticker (index = bar; value uses bars <= index)."""

    def __init__(self, ctx, t):
        td = ctx.td(t)
        self.td = td
        n = len(td.day)
        r = np.concatenate([[np.nan], np.diff(np.asarray(td.logc, float))])
        m = vf._ctx_market(ctx, td.dates)
        self.r = r
        self.sig = {h: vf.forecast_sigma_series(r, h, m) for h in HORIZONS}    # daily units
        self.mu = vf.walkforward_drift(r)                                       # mean daily log ret
        self.s2 = vf._trail_mean(r * r, n + 1, vf.MIN_HISTORY)                  # mean r^2, all history
        self.nh = np.cumsum(np.isfinite(r)).astype(float)
        self.lr = np.sqrt(vf._trail_mean(r * r, 756, vf.MIN_HISTORY))          # long-run daily vol
        lc = np.asarray(td.logc, float)
        # 252-bar drawdown (log) and 126-bar trend in vol units
        rm = pd.Series(lc).rolling(252, min_periods=126).max().to_numpy()
        self.dd = lc - rm
        tr = np.full(n, np.nan)
        tr[126:] = lc[126:] - lc[:-126]
        with np.errstate(invalid="ignore", divide="ignore"):
            self.trend = tr / (np.asarray(td.sig, float) * math.sqrt(126))
        # own reference residuals |r_h - mu*h| / (sig_h*sqrt h), indexed by START bar
        self.eabs = {}
        for h in HORIZONS:
            e = np.full(n, np.nan)
            if n > h:
                rh = lc[h:] - lc[:-h]
                with np.errstate(invalid="ignore", divide="ignore"):
                    e[:-h] = np.abs(rh - self.mu[:-h] * h) / (self.sig[h][:-h] * math.sqrt(h))
            self.eabs[h] = e

    def own_conf(self, h, W, positions):
        """Mean of the stock's own reference |residual| over windows whose outcome ended in
        the last W bars (start bar in (p-W-h, p-h]); NaN when fewer than W/2 are known."""
        cs, cn = _cum(self.eabs[h])
        p = np.asarray(positions, np.int64)
        hi = np.clip(p - h + 1, 0, len(self.eabs[h]))
        lo = np.clip(p - h + 1 - W, 0, len(self.eabs[h]))
        s, c = cs[hi] - cs[lo], cn[hi] - cn[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c >= W // 2, s / np.maximum(c, 1), np.nan)


def series(ctx, t):
    memo = _memo(ctx)
    s = memo["series"].get(t)
    if s is None:
        s = _Series(ctx, t)
        with _lock:
            memo["series"][t] = s
    return s


def _market_mean(ctx):
    """(days, expanding mean daily log return) of the spliced ^GSPC/SPY index series."""
    spy, gs = ctx.raw.get("SPY"), ctx.raw.get("^GSPC")
    parts = []
    if spy is not None and len(spy) > 1:
        parts.append(np.log(spy["close"]).diff().dropna())
    if gs is not None and len(gs) > 1:
        rg = np.log(gs["close"]).diff().dropna()
        parts.insert(0, rg[rg.index < parts[0].index[0]] if parts else rg)
    s = pd.concat(parts).sort_index()
    days = s.index.values.astype("datetime64[D]").astype(np.int64)
    x = s.to_numpy(float)
    return days, vf._trail_mean(x, len(x) + 1, vf.MIN_HISTORY)


class _Pool:
    """Basket-wide (engine.BASKET) causal statistics on a calendar-day axis.

    mu   pooled mean daily log return of every basket bar dated <= D
    muz  pooled mean of (daily log return / trailing sig63 of the previous bar)
    conf[h] cumulative sums of the basket's reference |residual| keyed by the END day of
         each h-day window, so a window (D-Wd, D] uses only outcomes known by D."""

    def __init__(self, ctx):
        S = np.zeros(NDAY)
        Z = np.zeros(NDAY)
        N = np.zeros(NDAY)
        LS = np.zeros(NDAY)              # basket cross-section: sum / count of log sig63 per day
        LN = np.zeros(NDAY)
        CE = {h: np.zeros(NDAY) for h in HORIZONS}
        CN = {h: np.zeros(NDAY) for h in HORIZONS}
        for b in engine.BASKET:
            if b not in ctx.raw:
                continue
            try:
                sb = series(ctx, b)
            except engine.ForecastError:
                continue
            td = sb.td
            d = td.day + DAY0
            ok = np.zeros(len(d), bool)
            ok[1:] = td.pool_ok[:-1] & np.isfinite(sb.r[1:])
            sig_prev = np.concatenate([[np.nan], np.asarray(td.sig, float)[:-1]])
            np.add.at(S, d[ok], sb.r[ok])
            np.add.at(Z, d[ok], sb.r[ok] / sig_prev[ok])
            np.add.at(N, d[ok], 1.0)
            sg = np.asarray(td.sig, float)
            okv = td.pool_ok & np.isfinite(sg) & (sg > 0)
            np.add.at(LS, d[okv], np.log(sg[okv]))
            np.add.at(LN, d[okv], 1.0)
            for h in HORIZONS:
                e = sb.eabs[h]
                n = len(e)
                if n <= h:
                    continue
                st = np.arange(n - h)
                g = np.isfinite(e[:n - h]) & td.pool_ok[:n - h]
                np.add.at(CE[h], d[st[g] + h], e[st[g]])
                np.add.at(CN[h], d[st[g] + h], 1.0)
        self.cS, self.cZ, self.cN = np.cumsum(S), np.cumsum(Z), np.cumsum(N)
        # basket mean log daily vol on the latest day <= D that has >= 10 basket stocks
        good = LN >= 10
        last = np.where(good, np.arange(NDAY), -1)
        last = np.maximum.accumulate(last)
        with np.errstate(invalid="ignore", divide="ignore"):
            xm = np.where(good, LS / np.maximum(LN, 1), np.nan)
        self.xlv = np.where(last >= 0, xm[np.maximum(last, 0)], np.nan)
        self.cE = {h: np.cumsum(CE[h]) for h in HORIZONS}
        self.cEN = {h: np.cumsum(CN[h]) for h in HORIZONS}
        self.mkt_days, self.mkt_mu = _market_mean(ctx)

    def at(self, day):
        i = np.asarray(day, np.int64) + DAY0
        with np.errstate(invalid="ignore", divide="ignore"):
            n = self.cN[i]
            mu = np.where(n > 1000, self.cS[i] / n, np.nan)
            muz = np.where(n > 1000, self.cZ[i] / n, np.nan)
        j = np.searchsorted(self.mkt_days, np.asarray(day, np.int64), side="right") - 1
        mk = np.where(j >= 0, self.mkt_mu[np.maximum(j, 0)], np.nan)
        return mu, muz, mk

    def conf(self, h, Wd, day):
        """Mean basket reference |residual| over windows that ended in (D - Wd, D]."""
        i = np.asarray(day, np.int64) + DAY0
        s = self.cE[h][i] - self.cE[h][i - Wd]
        c = self.cEN[h][i] - self.cEN[h][i - Wd]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c >= 200, s / np.maximum(c, 1), np.nan)


def pool(ctx):
    memo = _memo(ctx)
    with _lock:
        p = memo["pool"]
    if p is None:
        p = _Pool(ctx)
        with _lock:
            memo["pool"] = p
    return p


# ---------------------------------------------------------------------------
# Analog statistics (v0's own selection), disk-memoised
# ---------------------------------------------------------------------------

def _analog_compute(ctx, t, positions):
    """For each as-of p and h, from exactly v0's analog selection (engine._select, K=250,
    time rule A + h <= as-of, Gaussian weights): v0's 19 quantiles (vol-scaled paths),
    the UNSCALED weighted median and mean of the analog end returns, and the
    deviation-scaled cone med + ratio*(raw - med) (vol scaling applied to the spread only,
    not to the drift). NaN where fewer than K_MIN analogs are eligible."""
    eng = ctx.eng
    td = ctx.td(t)
    pl = eng.pool_for(t)
    pos = np.asarray(positions, dtype=np.int64)
    n = len(pos)
    out = {}
    for h in HORIZONS:
        out[f"qv0_{h}"] = np.full((n, 19), np.nan)
        out[f"qdev_{h}"] = np.full((n, 19), np.nan)
        out[f"amed_{h}"] = np.full(n, np.nan)
        out[f"amean_{h}"] = np.full(n, np.nan)
        out[f"apup_{h}"] = np.full(n, np.nan)
    if n == 0:
        return out
    sp = _v0.sorted_pool(eng, pl)
    from research.models_base import batch_dist2
    for b0 in range(0, n, 16):
        bp = pos[b0:b0 + 16]
        Ms = [sp.prefix(td.day[p]) for p in bp]
        D2 = batch_dist2(sp, td.X[bp], max(max(Ms), 1))
        for j, p in enumerate(bp):
            i, M, d2 = b0 + j, Ms[j], D2[j]
            pre = _v0._pre(d2, M)
            for h in HORIZONS:
                sel, _, w = eng._select(sp, td.X[p], d2, td.day[p], h, pre=pre)
                if len(sel) < engine.K_MIN:
                    continue
                raw, scaled, ratio = eng._paths(sp, sel, h, td.sig[p], end_only=True)
                raw, e = raw[:, 0], scaled[:, 0]
                out[f"qv0_{h}"][i] = engine.wquantile(e, w, QS19)
                out[f"apup_{h}"][i] = float(np.sum(w * (e > 0)) / np.sum(w))
                med = float(engine.wquantile(raw, w, [0.5])[0])
                out[f"amed_{h}"][i] = med
                out[f"amean_{h}"][i] = float(np.sum(w * raw) / np.sum(w))
                out[f"qdev_{h}"][i] = engine.wquantile(med + ratio * (raw - med), w, QS19)
    return out


def _analog_key():
    h = hashlib.md5()
    for f in ("engine.py", "research/models_base.py"):
        with open(os.path.join(ROOT, f), "rb") as fh:
            h.update(fh.read())
    h.update(inspect.getsource(_analog_compute).encode())
    return h.hexdigest()[:10]


_AKEY = None
_tlocks = {}


def analog(ctx, t, positions):
    """_analog_compute with an on-disk memo (full, untruncated contexts only: a truncated
    context, as in the harness look-ahead proof, always recomputes)."""
    global _AKEY
    pos = np.asarray(positions, dtype=np.int64)
    if getattr(ctx, "cutoff", None) is not None:
        return _analog_compute(ctx, t, pos)
    if _AKEY is None:
        _AKEY = _analog_key()
    path = os.path.join(ANALOG_DIR, _AKEY, f"{t}.npz")
    with _lock:
        tl = _tlocks.setdefault(t, threading.Lock())
    with tl:
        have = None
        if os.path.exists(path):
            with np.load(path) as z:          # close it: Windows refuses to replace an open file
                have = {k: z[k] for k in z.files}
        if have is not None:
            miss = pos[~np.isin(pos, have["pos"])]
        else:
            miss = pos
        if len(miss):
            new = _analog_compute(ctx, t, miss)
            new["pos"] = miss
            if have is not None:
                allp = np.concatenate([have["pos"], miss])
                o = np.argsort(allp, kind="stable")
                have = {k: np.concatenate([have[k], new[k]])[o] for k in new}
            else:
                have = new
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".tmp", "wb") as fh:
                np.savez(fh, **have)
            os.replace(path + ".tmp", path)
        idx = np.searchsorted(have["pos"], pos)
        return {k: v[idx] for k, v in have.items() if k != "pos"}


# ---------------------------------------------------------------------------
# One row per (ticker, as-of): every causal input any variant needs
# ---------------------------------------------------------------------------

OWN_W = (252, 756)               # own-conformal windows (bars)
POOL_W = (91, 365)               # basket-conformal windows (calendar days)


def ticker_rows(ctx, t, positions, need_analog=True):
    S = series(ctx, t)
    td = S.td
    p = np.asarray(positions, dtype=np.int64)
    P = pool(ctx)
    mu_pool, muz_pool, mu_mkt = P.at(td.day[p])
    R = {"pos": p, "day": td.day[p], "sig63": np.asarray(td.sig, float)[p], "lr": S.lr[p],
         "mu_own": S.mu[p], "s2_own": S.s2[p], "n_own": S.nh[p], "mu_pool": mu_pool,
         "muz_pool": muz_pool, "mu_mkt": mu_mkt, "dd": S.dd[p], "trend": S.trend[p],
         "xlv": P.xlv[td.day[p] + DAY0]}
    for h in HORIZONS:
        R[f"sh_{h}"] = S.sig[h][p] * math.sqrt(h)
        for W in OWN_W:
            R[f"cown_{h}_{W}"] = S.own_conf(h, W, p)
        for Wd in POOL_W:
            R[f"cpool_{h}_{Wd}"] = P.conf(h, Wd, td.day[p])
    if need_analog:
        R.update(analog(ctx, t, p))
    return R


# ---------------------------------------------------------------------------
# Variant algebra: centre, width, shape, ensemble
# ---------------------------------------------------------------------------

def centre(R, h, c):
    """h-day centre for spec c (dict with 'kind')."""
    k = c["kind"]
    n = len(R["pos"])
    if "mu_pool_f" not in R:          # basket drift undefined (< 1000 basket return-days,
        R["mu_pool_f"] = np.where(np.isfinite(R["mu_pool"]), R["mu_pool"],   # early 1963)
                                  np.where(np.isfinite(R["mu_mkt"]), R["mu_mkt"], R["mu_own"]))
    if k == "zero":
        return np.zeros(n)
    if k == "own":
        return R["mu_own"] * h
    if k == "mkt":
        return R["mu_mkt"] * h
    if k == "pool":
        return R["mu_pool_f"] * h
    if k == "const":
        return np.full(n, c["a"] * h)
    if k == "js":                      # James-Stein: target + w*(own - target), w = tau2/(tau2 + var(own))
        tg = c.get("target", "pool")
        if tg == "amed":               # shrink toward the unscaled analog median (per day)
            tgt = np.where(np.isfinite(R[f"amed_{h}"]), R[f"amed_{h}"] / h, R["mu_pool_f"])
        else:
            tgt = R["mu_pool_f"] if tg == "pool" else R["mu_" + tg]
        var = R["s2_own"] / np.maximum(R["n_own"], 1.0)
        w = c["tau2"] / (c["tau2"] + var)
        return (tgt + w * (R["mu_own"] - tgt)) * h
    if k in ("amed", "amean", "v0med", "v0deb"):
        if k == "amed":                # unscaled analog median
            x = R[f"amed_{h}"]
        elif k == "amean":             # unscaled analog weighted mean
            x = R[f"amean_{h}"]
        elif k == "v0med":             # v0's (vol-scaled) analog median, as is
            x = R[f"qv0_{h}"][:, 9]
        else:                          # v0 median minus the scaling-inflated pool drift
            x = R[f"qv0_{h}"][:, 9] - R["sig63"] * h * R["muz_pool"] + h * R["mu_pool_f"]
        return np.where(np.isfinite(x), x, R["mu_pool_f"] * h)   # < K_MIN analogs: basket drift
    if k == "blend":
        return sum(w * centre(R, h, cc) for w, cc in c["parts"])
    raise ValueError(k)


def width(R, h, wd):
    """h-day scale s for spec wd."""
    base = wd.get("base", "har")
    s = R[f"sh_{h}"] if base == "har" else R["sig63"] * math.sqrt(h)
    b = wd.get("floor_b", 0.0)
    if b:
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.log(R["lr"] * math.sqrt(h) / s)
        s = s * np.exp(b * np.maximum(np.nan_to_num(x, nan=0.0), 0.0))
    bx = wd.get("xfloor_b", 0.0)      # pull calm stocks toward the basket's vol level
    if bx:
        x = R["xlv"] - np.log(s / math.sqrt(h))
        s = s * np.exp(bx * np.maximum(np.nan_to_num(x, nan=0.0), 0.0))
    if wd.get("own"):
        W, lam = wd["own"]
        k = R[f"cown_{h}_{W}"]
        s = s * np.where(np.isfinite(k), k, wd.get("own_fill", 0.8)) ** lam
    if wd.get("pool"):
        Wd, lam = wd["pool"]
        k = R[f"cpool_{h}_{Wd}"]
        s = s * np.where(np.isfinite(k), k, wd.get("pool_fill", 0.8)) ** lam
    return s


def regime_var(R, h, name):
    if name == "relvol":               # forecast vol vs the stock's own long-run vol
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log(R[f"sh_{h}"] / (R["lr"] * math.sqrt(h)))
    if name == "dd":
        return R["dd"]
    if name == "trend":
        return R["trend"]
    if name == "xlevel":               # forecast vol vs the basket's average vol that day
        return np.log(R[f"sh_{h}"] / math.sqrt(h)) - R["xlv"]
    if name == "level":                # forecast daily vol level
        return np.log(R[f"sh_{h}"] / math.sqrt(h))
    raise ValueError(name)


def shape_fit(e, wt, mode):
    """19 z-quantiles from DEV standardised residuals e with pinball weights wt."""
    z = wquantile(e, wt, QS19)
    if mode == "mean0":
        z = z - z.mean()
    elif mode == "med0":
        z = z - z[9]
    return z


def spec_centre(spec, h):
    """The centre spec for horizon h (spec['centre_h'][h] when it differs by horizon)."""
    return spec["centre_h"][str(h)] if "centre_h" in spec else spec["centre"]


def cone_Q(R, h, spec, prm):
    """Quantiles (n,19) for spec with fitted params prm[h] (z, optional regime bins)."""
    c = centre(R, h, spec_centre(spec, h))
    s = width(R, h, spec.get("width", {}))
    ph = prm[str(h)]
    if spec.get("regime"):
        v = regime_var(R, h, spec["regime"])
        b = np.digitize(np.nan_to_num(v, nan=0.0), ph["edges"])
        Z = np.asarray(ph["z"], float)[b]
    else:
        Z = np.asarray(ph["z"], float)[None, :]
    Q = c[:, None] + s[:, None] * Z
    ens = spec.get("ens")
    if ens:
        kind, w = ens["kind"], ph.get("w", ens.get("w", 0.0))
        if kind == "v0":
            A = R[f"qv0_{h}"]
        elif kind == "dev":           # deviation-scaled analog cone, as is
            A = R[f"qdev_{h}"]
        elif kind == "dev_rc":        # its shape only: re-centred on this cone's median
            A = R[f"qdev_{h}"] - R[f"qdev_{h}"][:, 9:10] + Q[:, 9:10]
        elif kind == "v0_rc":
            A = R[f"qv0_{h}"] - R[f"qv0_{h}"][:, 9:10] + Q[:, 9:10]
        else:
            raise ValueError(kind)
        A = np.where(np.isfinite(A), A, Q)       # no analogs -> the plain cone
        Q = (1 - w) * Q + w * A
    if spec.get("post") and "post_k" in ph:   # per-regime spread factor around the median
        v = regime_var(R, h, spec["post"])
        b = np.digitize(np.nan_to_num(v, nan=0.0), ph["post_edges"])
        k = np.asarray(ph["post_k"], float)[b]
        Q = Q[:, 9:10] + k[:, None] * (Q - Q[:, 9:10])
    return Q


def fit_post(P, h, spec, ph, var, grid=np.round(np.arange(0.88, 1.125, 0.01), 2)):
    """DEV fit of the per-regime spread factor applied after the blend: regime bins =
    DEV terciles of regime variable var; factor per bin = argmin of DEV scaled pinball."""
    Q0 = cone_Q(P, h, dict(spec, post=None), {str(h): ph})
    m = np.isfinite(P[f"r_{h}"]) & np.isfinite(Q0).all(axis=1)
    v = np.nan_to_num(regime_var(P, h, var), nan=0.0)
    edges = [float(x) for x in np.quantile(v[m], [1 / 3, 2 / 3])]
    b = np.digitize(v, edges)
    ks = []
    for j in range(3):
        mm = m & (b == j)
        r, sn, q = P[f"r_{h}"][mm], P[f"sn_{h}"][mm], Q0[mm]
        best = min(grid, key=lambda k: harness.pinball_rows(q[:, 9:10] + k * (q - q[:, 9:10]), r, sn).mean())
        ks.append(float(best))
    return dict(ph, post_edges=edges, post_k=ks)


def needs_analog(spec):
    def has(c):
        if c["kind"] in ("amed", "amean", "v0med", "v0deb"):
            return True
        return c["kind"] == "blend" and any(has(cc) for _, cc in c["parts"])
    cs = list(spec["centre_h"].values()) if "centre_h" in spec else [spec["centre"]]
    return any(has(c) for c in cs) or bool(spec.get("ens"))


# ---------------------------------------------------------------------------
# Harness models
# ---------------------------------------------------------------------------

def _apply(spec, g):
    import copy
    spec = copy.deepcopy(spec)
    for path, v in (g or {}).items():
        d = spec
        ks = path.split(".")
        for k in ks[:-1]:
            d = d.setdefault(k, {})
        d[ks[-1]] = v
    return spec


class Cone2Model:
    """Harness model: spec + per-horizon chosen settings (grid values picked on DEV) +
    per-horizon fitted params (z quantiles, regime edges, ensemble weight)."""
    cache_deps = DEPS

    def __init__(self, name, spec, prm, chosen=None):
        self.name = name
        self.chosen = chosen or {str(h): {} for h in HORIZONS}
        self.specs = {str(h): _apply(spec, self.chosen.get(str(h), {})) for h in HORIZONS}
        self.prm = prm
        self.params = (spec, self.chosen, prm)

    def predict(self, ctx, t, positions, horizons):
        need = any(needs_analog(self.specs[str(h)]) for h in horizons)
        R = ticker_rows(ctx, t, positions, need_analog=need)
        out = {}
        for h in horizons:
            Q = cone_Q(R, h, self.specs[str(h)], self.prm)
            Q = np.sort(Q, axis=1)                        # guard against crossing
            out[h] = (Q, p_up(Q))
        return out


# FROZEN FINAL (from DEV fits; variant cone2_g_own of stage stage4; see the report)
FINAL_SPEC = {'centre': {'kind': 'own'},
 'width': {'base': 'har'},
 'shape': 'mean0',
 'ens': {'kind': 'dev', 'wgrid': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}}
FINAL_CHOSEN = {'5': {'width.own': [756, 0.3], 'width.pool': [365, 0.25]},
 '21': {'width.own': [756, 0.15], 'width.pool': [365, 0.0]},
 '63': {'width.own': [756, 0.0], 'width.pool': [365, 0.0]}}
FINAL_PRM = {'5': {'z': [-1.869314, -1.394401, -1.111641, -0.892129, -0.709501, -0.543281, -0.395244, -0.257122,
             -0.126163, 0.002071, 0.133204, 0.268097, 0.401395, 0.553425, 0.710453, 0.895859,
             1.110033, 1.394155, 1.830105],
       'w': 0.3},
 '21': {'z': [-1.732355, -1.289152, -1.015156, -0.813564, -0.641752, -0.485492, -0.346747,
              -0.211098, -0.090811, 0.029565, 0.149286, 0.267694, 0.387827, 0.515541, 0.657157,
              0.812741, 0.993067, 1.229792, 1.583457],
        'w': 0.3},
 '63': {'z': [-1.699876, -1.242319, -0.976213, -0.770565, -0.601913, -0.453022, -0.321725,
              -0.200858, -0.079688, 0.031653, 0.146553, 0.263307, 0.380934, 0.500782, 0.63444,
              0.775894, 0.948871, 1.165925, 1.49782],
        'w': 0.5}}
# END FROZEN FINAL

MODELS = {}
_VPATH = os.path.join(RESULTS, "cone2_params.json")
if os.path.exists(_VPATH):
    with open(_VPATH, encoding="utf-8") as _fh:
        for _name, _v in json.load(_fh).items():
            MODELS[_name] = Cone2Model(_name, _v["spec"], _v["prm"], _v.get("chosen"))
if FINAL_SPEC is not None:
    MODELS["cone2"] = Cone2Model("cone2", FINAL_SPEC, FINAL_PRM, FINAL_CHOSEN)


# ---------------------------------------------------------------------------
# Research: panel of every input + outcome on the harness grid (DEV / VAL only)
# ---------------------------------------------------------------------------

def panel_path(split):
    return os.path.join(RESULTS, f"cone2_panel_{split}.npz")


def build_panel(split, workers=2, only=None, log=print):
    if split not in ("dev", "val"):
        raise SystemExit("cone2 research never touches TEST")
    t0 = time.time()
    ctx = harness.Context()
    ts = splits.tickers_for(split)
    if only:
        ts = [t for t in ts if t in only]
    pool(ctx)

    def one(t):
        try:
            td = ctx.td(t)
        except engine.ForecastError:
            return t, None
        pos = harness.grid_positions(td, split, harness.STEP_FULL, HORIZONS)
        if not len(pos):
            return t, None
        R = ticker_rows(ctx, t, pos)
        n = len(pos)
        for h in HORIZONS:
            ok, r, sn, base, _ = harness.outcome_rows(td, pos, h)
            for k, v in (("r", r), ("sn", sn), ("base", base)):
                a = np.full(n, np.nan)
                a[ok] = v
                R[f"{k}_{h}"] = a
        ctx.release(t)
        return t, R

    parts, names = [], []
    with ThreadPoolExecutor(workers) as ex:
        for k, (t, R) in enumerate(ex.map(one, ts), 1):
            if R is None:
                continue
            R["tid"] = np.full(len(R["pos"]), len(names))
            names.append(t)
            parts.append(R)
            log(f"  [{k}/{len(ts)}] {t:6s} {len(R['pos']):5d} rows  {time.time() - t0:.0f}s", flush=True)
    P = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
    P["tickers"] = np.array(names)
    with open(panel_path(split) + ".tmp", "wb") as fh:
        np.savez(fh, **P)
    os.replace(panel_path(split) + ".tmp", panel_path(split))
    log(f"panel {split}: {len(P['pos'])} rows, {len(names)} tickers, {time.time() - t0:.0f}s")
    return P


def load_panel(split):
    z = np.load(panel_path(split), allow_pickle=False)
    return {k: z[k] for k in z.files}



# ---------------------------------------------------------------------------
# Research: fit every variant on DEV, score on DEV and VAL with harness.summarise
# ---------------------------------------------------------------------------

def _fit_mask(P, h):
    return np.isfinite(P[f"r_{h}"]) & np.isfinite(P[f"sh_{h}"]) & np.isfinite(P["mu_own"])


def fit_prm(P, spec, h):
    """Fit z (and regime edges) for spec at h on panel P (DEV)."""
    m = _fit_mask(P, h)
    c = centre(P, h, spec_centre(spec, h))
    sc = width(P, h, spec.get("width", {}))
    m &= np.isfinite(c) & np.isfinite(sc)
    e = (P[f"r_{h}"][m] - c[m]) / sc[m]
    wt = sc[m] / P[f"sn_{h}"][m]                 # scaled-pinball weights
    mode = spec.get("shape", "fit")
    if mode.startswith("vf_"):
        return {"z": [float(x) for x in vf.z_quantiles(h, mode[3:], market=True)]}
    if spec.get("regime"):
        v = regime_var(P, h, spec["regime"])[m]
        v = np.nan_to_num(v, nan=0.0)
        edges = [float(x) for x in np.quantile(v, [1 / 3, 2 / 3])]
        b = np.digitize(v, edges)
        return {"edges": edges, "z": [[float(x) for x in shape_fit(e[b == k], wt[b == k], mode)]
                                      for k in range(3)]}
    return {"z": [float(x) for x in shape_fit(e, wt, mode)]}


def _pinball(P, h, Q):
    m = np.isfinite(P[f"r_{h}"]) & np.isfinite(Q).all(axis=1)
    return float(harness.pinball_rows(Q[m], P[f"r_{h}"][m], P[f"sn_{h}"][m]).mean())


def _rows(P, h, Q, pu=None):
    ok = np.isfinite(P[f"r_{h}"])
    Q = Q[ok]
    pu = p_up(np.sort(Q, axis=1)) if pu is None else pu[ok]
    tid = P["tid"][ok]
    cols = {"r": P[f"r_{h}"][ok], "sn": P[f"sn_{h}"][ok], "base": P[f"base_{h}"][ok],
            "sig": P["sig63"][ok]}
    rows = []
    for t in np.unique(tid):
        mm = tid == t
        d = {k: v[mm] for k, v in cols.items()}
        d["Q"], d["P"] = Q[mm], pu[mm]
        rows.append(d)
    return rows


def evaluate(P, h, Q, pu=None):
    return harness.summarise(_rows(P, h, Q, pu))


def paired_ci(P, h, QA, QB, n_boot=2000, seed=0):
    """skill(A) - skill(B) and its 90% ticker-bootstrap interval, on rows where both exist."""
    ok = np.isfinite(P[f"r_{h}"]) & np.isfinite(QA).all(axis=1) & np.isfinite(QB).all(axis=1)
    r, sn, tid = P[f"r_{h}"][ok], P[f"sn_{h}"][ok], P["tid"][ok]
    la = harness.pinball_rows(QA[ok], r, sn)
    lb = harness.pinball_rows(QB[ok], r, sn)
    ln = harness.pinball_rows(harness.Z19[None, :] * sn[:, None], r, sn)
    ts = np.unique(tid)
    A = np.array([la[tid == t].sum() for t in ts])
    B = np.array([lb[tid == t].sum() for t in ts])
    N = np.array([ln[tid == t].sum() for t in ts])
    idx = np.random.default_rng(seed).integers(0, len(ts), (n_boot, len(ts)))
    d = (B[idx].sum(1) - A[idx].sum(1)) / N[idx].sum(1)
    return float((B.sum() - A.sum()) / N.sum()), float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))


def baselines(P, h):
    sh = P[f"sh_{h}"]
    sd = P["sig63"] * math.sqrt(h)
    out = {
        "naive0": harness.Z19[None, :] * sd[:, None],
        "naive_drift": (P["mu_own"] * h)[:, None] + harness.Z19[None, :] * sd[:, None],
        "v0": P[f"qv0_{h}"],
        "vol_har": sh[:, None] * vf.z_quantiles(h, "zero")[None, :],
        "vol_har_drift": (P["mu_own"] * h)[:, None] + sh[:, None] * vf.z_quantiles(h, "drift")[None, :],
    }
    pus = {"naive0": np.full(len(sh), 0.5),
           "naive_drift": harness.norm_cdf(P["mu_own"] * h / sd),
           "v0": P[f"apup_{h}"]}
    return out, pus


def _grid_points(grid):
    import itertools
    keys = [k for k, _ in grid]
    return [dict(zip(keys, vals)) for vals in itertools.product(*[v for _, v in grid])]


def fit_variant(Pd, spec, grid=None):
    """Grid values and ensemble weight chosen PER HORIZON by DEV pinball; z fit on DEV.
    Returns (chosen {h: grid point}, prm {h: {...}}, dev pinball {h})."""
    prm, chosen, devpb = {}, {}, {}
    for h in HORIZONS:
        best = None
        for g in (_grid_points(grid) if grid else [{}]):
            sp = _apply(spec, g)
            ph = fit_prm(Pd, sp, h)
            if sp.get("ens"):
                bw = None
                for w in sp["ens"]["wgrid"]:
                    ph2 = dict(ph, w=float(w))
                    pb = _pinball(Pd, h, cone_Q(Pd, h, sp, {str(h): ph2}))
                    if bw is None or pb < bw[0]:
                        bw = (pb, ph2)
                pb, ph = bw
            else:
                pb = _pinball(Pd, h, cone_Q(Pd, h, sp, {str(h): ph}))
            if best is None or pb < best[0]:
                best = (pb, ph, g)
        devpb[h], prm[str(h)], chosen[str(h)] = best
    return chosen, prm, devpb


def variant_Q(P, spec, chosen, prm, h):
    return cone_Q(P, h, _apply(spec, chosen[str(h)]), prm)


ANN = 252.0
TAUS = [(x / ANN) ** 2 for x in (0.0, 0.01, 0.02, 0.04, 0.06, 0.10, 0.2)]


def research_centres():
    """Stage 1: every centre tried (HAR width, DEV empirical shape). Names are cone2_*."""
    V = []
    HAR = {"base": "har"}

    def add(nm, c, mode, grid=None):
        V.append((f"cone2_c_{nm}_{mode}", {"centre": c, "width": HAR, "shape": mode}, grid))

    for mode in ("fit", "mean0", "med0", "vf_centred"):
        add("zero", {"kind": "zero"}, mode)
        add("own", {"kind": "own"}, mode)
        add("amed", {"kind": "amed"}, mode)
    for mode in ("mean0", "med0"):
        add("mkt", {"kind": "mkt"}, mode)
        add("pool", {"kind": "pool"}, mode)
        add("amean", {"kind": "amean"}, mode)
        add("v0med", {"kind": "v0med"}, mode)
        add("v0deb", {"kind": "v0deb"}, mode)
        add("const", {"kind": "const"}, mode,
            [("centre.a", [x / ANN for x in (0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18)])])
        add("jspool", {"kind": "js", "target": "pool"}, mode, [("centre.tau2", TAUS)])
        add("jsmkt", {"kind": "js", "target": "mkt"}, mode, [("centre.tau2", TAUS)])
        for w in (0.25, 0.5, 0.75):
            add(f"own{int(w * 100)}amed",
                {"kind": "blend", "parts": [[w, {"kind": "own"}], [1 - w, {"kind": "amed"}]]}, mode)
            add(f"pool{int(w * 100)}amed",
                {"kind": "blend", "parts": [[w, {"kind": "pool"}], [1 - w, {"kind": "amed"}]]}, mode)
    return V


def research_width(cbest, mode):
    """Stage 2: width / shape / ensemble variants on top of the chosen centre."""
    V = []

    def mk(nm, w, extra=None, grid=None):
        V.append((nm, dict({"centre": cbest, "width": w, "shape": mode}, **(extra or {})), grid))

    mk("cone2_w_sd63", {"base": "sd63"})
    mk("cone2_w_har", {"base": "har"})
    mk("cone2_w_floor", {"base": "har"}, None, [("width.floor_b", [0.0, 0.25, 0.5, 0.75, 1.0])])
    mk("cone2_w_own", {"base": "har"}, None,
       [("width.own", [[W, lam] for W in OWN_W for lam in (0.0, 0.15, 0.3, 0.5, 0.75)])])
    mk("cone2_w_pool", {"base": "har"}, None,
       [("width.pool", [[W, lam] for W in POOL_W for lam in (0.0, 0.25, 0.5, 0.75, 1.0)])])
    mk("cone2_w_relvol", {"base": "har"}, {"regime": "relvol"})
    mk("cone2_w_level", {"base": "har"}, {"regime": "level"})
    mk("cone2_w_xlevel", {"base": "har"}, {"regime": "xlevel"})
    mk("cone2_w_xfloor", {"base": "har"}, None, [("width.xfloor_b", [0.0, 0.1, 0.2, 0.35, 0.5])])
    mk("cone2_s_dd", {"base": "har"}, {"regime": "dd"})
    mk("cone2_s_trend", {"base": "har"}, {"regime": "trend"})
    for kind in ("v0", "v0_rc", "dev", "dev_rc"):
        mk(f"cone2_e_{kind}", {"base": "har"},
           {"ens": {"kind": kind, "wgrid": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]}})
    return V


def _flat(name, split, h, res, chosen):
    a = res["all"]
    ci = a.get("skill_ci90", [np.nan, np.nan])
    d = {"model": name, "split": split, "h": h, "n": a["n"], "skill": a["skill"],
         "ci_lo": ci[0], "ci_hi": ci[1], "cov50": a["cov50"], "cov80": a["cov80"],
         "width80": a["width80"], "brier_skill": a["brier_skill"], "hit": a["hit"],
         "hit_majority": a["hit_majority"], "chosen": json.dumps(chosen) if chosen else ""}
    for b in ("low", "mid", "high"):
        v = res.get("vol", {}).get(b, {})
        d[f"skill_{b}"] = v.get("skill", np.nan)
        d[f"cov50_{b}"] = v.get("cov50", np.nan)
        d[f"cov80_{b}"] = v.get("cov80", np.nan)
    return d


def score_variants(Pd, Pv, V, table, params, log=print):
    for name, spec, grid in V:
        t0 = time.time()
        chosen, prm, _ = fit_variant(Pd, spec, grid)
        params[name] = {"spec": spec, "chosen": chosen, "prm": prm}
        for split, P in (("dev", Pd), ("val", Pv)):
            for h in HORIZONS:
                Q = np.sort(variant_Q(P, spec, chosen, prm, h), axis=1)
                table.append(_flat(name, split, h, evaluate(P, h, Q), chosen[str(h)]))
        a = [x for x in table if x["model"] == name]
        log(f"{name:26s} " + " ".join(f"{x['split'][0]}{x['h']}:{x['skill']:+.4f}/{x['cov80']:.3f}" for x in a)
            + f"  {json.dumps(chosen) if grid else ''} ({time.time() - t0:.0f}s)", flush=True)


def score_baselines(Pd, Pv, table):
    for split, P in (("dev", Pd), ("val", Pv)):
        for h in HORIZONS:
            Qs, pus = baselines(P, h)
            for nm, Q in Qs.items():
                table.append(_flat(nm, split, h, evaluate(P, h, Q, pus.get(nm)), None))


def save_tables(table, params, tag):
    pd.DataFrame(table).to_csv(os.path.join(RESULTS, f"cone2_variants_{tag}.csv"), index=False)
    with open(os.path.join(RESULTS, f"cone2_params_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(params, fh)


def research(stage="1", base=None, log=print):
    Pd, Pv = load_panel("dev"), load_panel("val")
    table, params = [], {}
    if stage == "1":
        score_baselines(Pd, Pv, table)
        for x in table:
            log(f"baseline {x['model']:14s} {x['split']} h{x['h']:<2d} skill {x['skill']:+.4f} "
                f"cov50 {x['cov50']:.3f} cov80 {x['cov80']:.3f} low/mid/high cov80 "
                f"{x['cov80_low']:.3f}/{x['cov80_mid']:.3f}/{x['cov80_high']:.3f}")
        score_variants(Pd, Pv, research_centres(), table, params, log)
    elif stage == "2":                # width / shape / ensemble on a stage-1 centre
        with open(os.path.join(RESULTS, "cone2_params_stage1.json"), encoding="utf-8") as fh:
            v = json.load(fh)[base]
        ch = {h: _apply(v["spec"], v["chosen"][str(h)])["centre"] for h in map(str, HORIZONS)}
        log(f"stage 2 on centre {base}: {json.dumps(ch)}")
        V = research_width(None, v["spec"]["shape"])
        for _, sp, _ in V:
            sp.pop("centre")
            sp["centre_h"] = ch
        score_variants(Pd, Pv, V, table, params, log)
    elif stage == "3":                # combinations of the stage-2 winners, jointly on DEV
        with open(os.path.join(RESULTS, "cone2_params_stage1.json"), encoding="utf-8") as fh:
            v = json.load(fh)[base]
        ch = {h: _apply(v["spec"], v["chosen"][str(h)])["centre"] for h in map(str, HORIZONS)}
        conf = [("width.own", [[756, x] for x in (0.0, 0.15, 0.3, 0.5)]),
                ("width.pool", [[365, x] for x in (0.0, 0.25, 0.5)])]
        WG = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        V = [("cone2_f_conf", {"centre_h": ch, "width": {"base": "har"}, "shape": "mean0"}, conf),
             ("cone2_f_dev", {"centre_h": ch, "width": {"base": "har"}, "shape": "mean0",
                              "ens": {"kind": "dev", "wgrid": WG}}, None),
             ("cone2_f_conf_dev", {"centre_h": ch, "width": {"base": "har"}, "shape": "mean0",
                                   "ens": {"kind": "dev", "wgrid": WG}}, conf),
             ("cone2_f_conf_v0", {"centre_h": ch, "width": {"base": "har"}, "shape": "mean0",
                                  "ens": {"kind": "v0", "wgrid": WG}}, conf)]
        score_baselines(Pd, Pv, table)
        score_variants(Pd, Pv, V, table, params, log)
    elif stage == "4":                # the stage-3 recipe on the VAL-favoured (own-mean) centres
        conf = [("width.own", [[756, x] for x in (0.0, 0.15, 0.3, 0.5)]),
                ("width.pool", [[365, x] for x in (0.0, 0.25, 0.5)])]
        WG = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        own75 = {"kind": "blend", "parts": [[0.75, {"kind": "own"}], [0.25, {"kind": "amed"}]]}
        V = [("cone2_g_own", {"centre": {"kind": "own"}, "width": {"base": "har"}, "shape": "mean0",
                              "ens": {"kind": "dev", "wgrid": WG}}, conf),
             ("cone2_alt", {"centre": own75, "width": {"base": "har"}, "shape": "mean0",
                            "ens": {"kind": "dev", "wgrid": WG}}, conf)]
        score_variants(Pd, Pv, V, table, params, log)
    elif stage == "5":                # per-regime spread factor after the blend, on cone2_g_own
        with open(os.path.join(RESULTS, "cone2_params_stage4.json"), encoding="utf-8") as fh:
            g = json.load(fh)["cone2_g_own"]
        for var in ("xlevel", "level", "relvol"):
            name = f"cone2_p_{var}"
            spec = dict(g["spec"], post=var)
            prm = {}
            for h in HORIZONS:
                sp = _apply(spec, g["chosen"][str(h)])
                prm[str(h)] = fit_post(Pd, h, sp, g["prm"][str(h)], var)
            params[name] = {"spec": spec, "chosen": g["chosen"], "prm": prm}
            for split, P in (("dev", Pd), ("val", Pv)):
                for h in HORIZONS:
                    Q = np.sort(variant_Q(P, spec, g["chosen"], prm, h), axis=1)
                    table.append(_flat(name, split, h, evaluate(P, h, Q), {"k": prm[str(h)]["post_k"]}))
            a = [x for x in table if x["model"] == name]
            log(f"{name:18s} " + " ".join(f"{x['split'][0]}{x['h']}:{x['skill']:+.4f}/{x['cov80']:.3f}"
                                          f"[{x['cov80_low']:.3f},{x['cov80_mid']:.3f},{x['cov80_high']:.3f}]" for x in a))
            log("   k per bin (5/21/63): " + " ".join(str(prm[str(h)]["post_k"]) for h in HORIZONS))
    save_tables(table, params, f"stage{stage}")
    return table, params


def freeze(stage, name, log=print):
    """Write variant `name` from cone2_params_<stage>.json into this file as literals
    (FINAL_SPEC / FINAL_CHOSEN / FINAL_PRM), rounded to 6 decimals."""
    with open(os.path.join(RESULTS, f"cone2_params_{stage}.json"), encoding="utf-8") as fh:
        v = json.load(fh)[name]

    def rnd(x):
        if isinstance(x, float):
            return round(x, 6) if abs(x) >= 1e-4 else float(f"{x:.6g}")
        if isinstance(x, list):
            return [rnd(y) for y in x]
        if isinstance(x, dict):
            return {k: rnd(y) for k, y in x.items()}
        return x

    import pprint
    body = ("# FROZEN FINAL (from DEV fits; variant %s of stage %s; see the report)\n" % (name, stage)
            + "FINAL_SPEC = " + pprint.pformat(rnd(v["spec"]), width=100, sort_dicts=False) + "\n"
            + "FINAL_CHOSEN = " + pprint.pformat(rnd(v["chosen"]), width=100, sort_dicts=False) + "\n"
            + "FINAL_PRM = " + pprint.pformat(rnd(v["prm"]), width=100, compact=True, sort_dicts=False) + "\n")
    path = os.path.abspath(__file__)
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    a = src.index("# FROZEN FINAL")
    b = src.index("# END FROZEN FINAL")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src[:a] + body + src[b:])
    log(f"froze {name} into {path}")


def export_variants(stage, names, log=print):
    """Copy variants into cone2_params.json so research/harness.py can run them by name."""
    with open(os.path.join(RESULTS, f"cone2_params_{stage}.json"), encoding="utf-8") as fh:
        v = json.load(fh)
    path = os.path.join(RESULTS, "cone2_params.json")
    cur = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            cur = json.load(fh)
    for n in names:
        cur[n] = v[n]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cur, fh)
    log(f"exported {names} to {path}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "panel":
        only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
        build_panel(sys.argv[2], only=only)
    elif cmd == "research":
        research(sys.argv[2] if len(sys.argv) > 2 else "1", sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "freeze":                # freeze <stage> <variant>
        freeze(sys.argv[2], sys.argv[3])
    elif cmd == "export":                # export <stage> <variant,variant,...>
        export_variants(sys.argv[2], sys.argv[3].split(","))
