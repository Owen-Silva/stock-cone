"""Stock Cone v2: the product cone model (research name `cone2_f_conf_dev`).

Frozen copy of the pre-registered product cone. Sources of truth:
  - research/models_cone2.py, variant `cone2_f_conf_dev` (spec, chosen settings and fitted
    parameters in research/results/cone2_params.json): centre, width, shape, blend.
  - research/volforecast.py, frozen HAR sigma `har_mx_QLr` (with market) and `har_x_QLr`
    (no market): the width base.
  - Report: research/reports/cone2-01-calibrated-cone.md. Pre-registration:
    research/PLAN.md, "Pre-registered before TEST #2".
Nothing here imports research code: every literal below is copied. `python cone_model.py
--vs-research` rebuilds the research harness model and checks this file against it.

For a stock at as-of bar p and horizon h (5, 21 or 63 trading days) the model gives 19
quantiles (q = 0.05 .. 0.95) of the h-day log return r = ln(close[p+h] / close[p]):

    centre_h = h * (mu_b + w_js * (mu_own - mu_b)),   w_js = tau2_h / (tau2_h + s2_own / n_own)
    s_h      = sigma_HAR,h * sqrt(h) * own_conf_h ** lam_own_h * basket_conf_h ** lam_pool_h
    plain_h  = centre_h + s_h * z_h
    Q_h      = sort((1 - w_h) * plain_h + w_h * Qdev_h)

  mu_own     the stock's walk-forward mean daily log return (all bars <= p, >= 252 returns)
  mu_b       the pooled walk-forward mean daily log return of every analog-basket bar dated
             <= the as-of day (falls back to the index mean, then mu_own, early on)
  sigma_HAR  frozen log-variance HAR forecast (own 5/22/66/252-day and long-run variance,
             market variance, shock, drawdown), daily units
  own_conf   mean standardised |residual| of the stock's own past h-day windows that ended
             in the last 756 bars (0.8 if fewer than half are known)
  basket_conf the same over the analog basket, windows that ended in the last 365 days
             (0.8 if fewer than 200)
  z_h        19 DEV-fitted standardised quantiles, mean removed
  Qdev_h     the deviation-scaled analog cone: the engine's own K = 250 analogs, end
             returns med + ratio * (raw - med) (vol scaling applied to the spread only),
             weighted quantiles. If fewer than K_MIN analogs exist it is the plain cone.

Every input is causal: bars <= p for the stock, basket bars and market bars dated <= the
as-of day, own and basket residuals only from windows whose outcome is known by then.
"""

import math

import numpy as np
import pandas as pd

HORIZONS = (5, 21, 63)
QS19 = np.round(np.arange(1, 20) * 0.05, 2)          # 0.05 .. 0.95
I10, I25, I50, I75, I90 = 1, 4, 9, 14, 17             # QS19 indices of the five served bands
IDX5 = [I10, I25, I50, I75, I90]
MIN_HISTORY = 252                                     # HAR + drift need >= 252 daily returns
VAR_FLOOR = 1e-8
DAY0 = 30000                                          # calendar-day axis offset for the basket
NDAY = 60000                                          # covers to ~2034
FILL = 0.8                                            # conformal factor when too few windows

# --- frozen HAR sigma (research/volforecast.py FROZEN["sigma"], fitted on DEV by QLIKE) ----
HAR = {
    "market": {   # har_mx_QLr
        5: (["c", "l5", "l22", "l66", "l252", "llr", "lm22", "lm252", "shock", "neg"],
            [-1.4596824566, 0.041239699, 0.040543688, 0.3359948564, 0.4330799388, 0.046024543,
             0.2243563697, -0.2947605195, 0.043200362, -1.6331981197]),
        21: (["c", "l5", "l22", "l66", "l252", "llr", "lm22", "lm252", "shock", "neg"],
             [-1.736661993, 0.0479271826, -0.0323826945, 0.3239691346, 0.5348536234, 0.0480770585,
              0.2149108444, -0.3382484037, 0.0167575551, -1.0409228607]),
        63: (["c", "l5", "l22", "l66", "l252", "llr", "lm22", "lm252", "shock", "neg"],
             [-1.9589018833, 0.0346043973, 0.0424721718, 0.3085798358, 0.5233742493, 0.0352268742,
              0.1226283271, -0.2956515427, 0.0026135895, -0.6350193238]),
    },
    "nomarket": {   # har_x_QLr
        5: (["c", "l5", "l22", "l66", "l252", "llr", "shock", "neg"],
            [-1.0960315159, 0.0474558804, 0.1414495805, 0.4030819118, 0.198411777, 0.0715628174,
             0.0446007735, -1.8496133022]),
        21: (["c", "l5", "l22", "l66", "l252", "llr", "shock", "neg"],
             [-1.0149820598, 0.0535489262, 0.0642356634, 0.3876242753, 0.2769470995, 0.0876351528,
              0.0167130088, -1.2177495207]),
        63: (["c", "l5", "l22", "l66", "l252", "llr", "shock", "neg"],
             [-0.8693466107, 0.0374053366, 0.0984603868, 0.3565221305, 0.2916591076, 0.0945607759,
              0.0028143097, -0.6584959709]),
    },
}

# --- frozen cone2_f_conf_dev (research/results/cone2_params.json) ------------------------
#   tau2   James-Stein prior variance of the daily drift (centre_h: kind js, target pool)
#   own    (window in bars, lambda) of the own conformal factor   (chosen "width.own")
#   pool   (window in days, lambda) of the basket conformal factor (chosen "width.pool")
#   z      19 standardised quantiles, mean removed (shape mean0), fitted on DEV
#   w      weight of the deviation-scaled analog cone (ens kind dev)
SPEC = {
    5: {"tau2": 1.5747039556563367e-09, "own": (756, 0.3), "pool": (365, 0.25), "w": 0.2,
        "z": [-1.8671192625572013, -1.3894279969587893, -1.1126693456661156, -0.8926191288984439,
              -0.7060284432952963, -0.5461003649193208, -0.39631875476764233, -0.25649699699637135,
              -0.12800481018691062, -0.0002319248521844789, 0.1296566253127048, 0.2652975129397905,
              0.4005099139818311, 0.5531870649466177, 0.712181606812889, 0.89543861482959,
              1.1116116268772878, 1.3952884278985418, 1.8318456354990231]},
    21: {"tau2": 6.298815822625347e-09, "own": (756, 0.15), "pool": (365, 0.25), "w": 0.1,
         "z": [-1.8576975274190815, -1.3820219119500499, -1.089224188119358, -0.8724608432105048,
               -0.6903103393000398, -0.5247768858825892, -0.37417645179239883, -0.23040738122517573,
               -0.09720408634280414, 0.032422917369442444, 0.1593199687493435, 0.2886059497680972,
               0.4165123296244185, 0.5559609227099778, 0.7102498897424306, 0.8722549320115291,
               1.0623240803549385, 1.3187128755206339, 1.7019157493911896]},
    63: {"tau2": 6.298815822625347e-09, "own": (756, 0.0), "pool": (365, 0.0), "w": 0.1,
         "z": [-1.6686170058432814, -1.225388560637836, -0.9621340472234804, -0.7627854422524651,
               -0.595688033204434, -0.44702726815945176, -0.3148310149685933, -0.19368615840888648,
               -0.07707581637975117, 0.03408458641637066, 0.14682031300879272, 0.26208997282651275,
               0.3778448934059428, 0.4972709955729959, 0.6227927964739384, 0.7654647019590549,
               0.9311254661199094, 1.1426364213329223, 1.4671031999617397]},
}


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _trail_mean(x, k, min_count):
    """NaN-aware trailing mean of x over rows t-k+1..t (needs >= min_count finite)."""
    x = np.asarray(x, float)
    n = len(x)
    f = np.isfinite(x)
    cs = np.concatenate([[0.0], np.cumsum(np.where(f, x, 0.0))])
    cn = np.concatenate([[0], np.cumsum(f)])
    hi = np.arange(1, n + 1)
    lo = np.maximum(hi - k, 0)
    s = cs[hi] - cs[lo]
    c = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c >= min_count, s / np.maximum(c, 1), np.nan)


def _cum(x):
    x = np.asarray(x, float)
    f = np.isfinite(x)
    return (np.concatenate([[0.0], np.cumsum(np.where(f, x, 0.0))]),
            np.concatenate([[0], np.cumsum(f)]))


def wquantile(v, w, qs):
    """Weighted quantiles of 1-D v (midpoint rule, linear interpolation); same as engine.wquantile."""
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    p = (cw - 0.5 * w) / cw[-1]
    return np.interp(qs, p, v)


def p_up(Q):
    """P(r > 0) read off sorted 19-quantile rows (the interpolated CDF at 0, linear between
    quantiles), clipped to [0.05, 0.95] because the cone says nothing beyond its 5/95 tails."""
    Q = np.atleast_2d(np.asarray(Q, float))
    out = np.full(len(Q), np.nan)
    fin = np.isfinite(Q).all(axis=1)
    k = (Q < 0).sum(axis=1)
    lo = np.clip(k - 1, 0, 18)
    hi = np.clip(k, 0, 18)
    a, b = Q[np.arange(len(Q)), lo], Q[np.arange(len(Q)), hi]
    with np.errstate(invalid="ignore", divide="ignore"):
        x = QS19[lo] + np.where(b > a, (0.0 - a) / (b - a), 0.0) * 0.05
    v = np.where(k == 0, 0.05, np.where(k == 19, 0.95, x))
    out[fin] = 1.0 - v[fin]
    return out


# ---------------------------------------------------------------------------
# HAR sigma (frozen copy of volforecast.forecast_sigma_series, only the features it uses)
# ---------------------------------------------------------------------------

def _lg(x):
    return np.log(np.maximum(x, VAR_FLOOR))


def har_sigma_series(log_returns, h, market_log_returns=None):
    """Daily-unit HAR sigma forecast for every row t, using rows <= t (NaN under 252 returns)."""
    r = np.asarray(log_returns, float)
    r2 = r * r
    have_m = market_log_returns is not None
    F = {"n_hist": np.cumsum(np.isfinite(r)).astype(float)}
    for k in (5, 22, 66, 252):
        F[f"rv{k}"] = _trail_mean(r2, k, max(1, int(np.ceil(0.8 * k))))
    F["lr756"] = _trail_mean(r2, 756, MIN_HISTORY)
    with np.errstate(invalid="ignore", divide="ignore"):
        F["shock"] = np.minimum(np.abs(r) / np.sqrt(F["rv66"]), 6.0)
        F["neg22"] = np.minimum(_trail_mean(r, 22, 18) * 22, 0.0)
    F["shock"] = np.where(np.isfinite(F["shock"]), F["shock"], 0.0)
    F["neg22"] = np.where(np.isfinite(F["neg22"]), F["neg22"], 0.0)
    if have_m:   # a missing market window falls back to the stock's own long-run variance
        m = np.asarray(market_log_returns, float)
        F["m_rv22"] = _trail_mean(m * m, 22, 18)
        F["m_rv252"] = _trail_mean(m * m, 252, 200)
        for k in ("m_rv22", "m_rv252"):
            F[k] = np.where(np.isfinite(F[k]), F[k], F["lr756"])
    cols = {
        "c": lambda: np.ones_like(F["rv22"]),
        "l5": lambda: _lg(F["rv5"]),
        "l22": lambda: _lg(F["rv22"]),
        "l66": lambda: _lg(F["rv66"]),
        "l252": lambda: _lg(F["rv252"]),
        "llr": lambda: _lg(F["lr756"]),
        "lm22": lambda: _lg(F["m_rv22"]),
        "lm252": lambda: _lg(F["m_rv252"]),
        "shock": lambda: F["shock"],
        "neg": lambda: F["neg22"],
    }
    names, beta = HAR["market" if have_m else "nomarket"][h]
    # explicit column sum (not X @ beta): BLAS rounding can depend on array length, and the
    # no-look-ahead check demands bit-identical output on truncated data
    logvar = np.zeros(len(F["n_hist"]))
    for c, b in zip(names, beta):
        logvar = logvar + float(b) * cols[c]()
    sig = np.sqrt(np.exp(logvar))
    ok = (F["n_hist"] >= MIN_HISTORY) & np.isfinite(sig)
    return np.where(ok, sig, np.nan)


# ---------------------------------------------------------------------------
# Market series (SPY daily log returns, ^GSPC before SPY starts)
# ---------------------------------------------------------------------------

class Market:
    """Index daily log returns spliced from ^GSPC (before SPY) and SPY, from price frames
    the caller loaded (so a truncated engine sees only data up to its cutoff)."""

    def __init__(self, spy, gspc):
        parts = []
        if spy is not None and len(spy) > 1:
            parts.append(np.log(spy["close"]).diff().dropna())
        if gspc is not None and len(gspc) > 1:
            rg = np.log(gspc["close"]).diff().dropna()
            parts.insert(0, rg[rg.index < parts[0].index[0]] if parts else rg)
        self.ret = pd.concat(parts).sort_index() if parts else None
        if self.ret is not None:
            self.days = self.ret.index.values.astype("datetime64[D]").astype(np.int64)
            x = self.ret.to_numpy(float)
            self.mu = _trail_mean(x, len(x) + 1, MIN_HISTORY)      # expanding mean daily log return
        else:
            self.days = np.zeros(0, np.int64)
            self.mu = np.zeros(0)

    def returns_on(self, dates):
        """Index returns aligned to a stock's dates (NaN where missing), or None."""
        return None if self.ret is None else self.ret.reindex(dates).to_numpy()

    def mean_at(self, day):
        j = np.searchsorted(self.days, np.asarray(day, np.int64), side="right") - 1
        return np.where(j >= 0, self.mu[np.maximum(j, 0)] if len(self.mu) else np.nan, np.nan)


# ---------------------------------------------------------------------------
# Per-stock causal series
# ---------------------------------------------------------------------------

class TickerSeries:
    """Full-length causal arrays for one stock (index = bar; value uses bars <= index).

    td is an engine.TickerData (uses .logc, .day, .sig, .pool_ok); market is a Market."""

    def __init__(self, td, market):
        self.td = td
        lc = np.asarray(td.logc, float)
        n = len(lc)
        r = np.concatenate([[np.nan], np.diff(lc)])
        m = market.returns_on(td.dates)
        self.r = r
        self.sig = {h: har_sigma_series(r, h, m) for h in HORIZONS}      # daily units
        self.mu = _trail_mean(r, n + 1, MIN_HISTORY)                      # walk-forward mean
        self.s2 = _trail_mean(r * r, n + 1, MIN_HISTORY)                  # mean r^2, all history
        self.nh = np.cumsum(np.isfinite(r)).astype(float)
        # reference residuals |r_h - mu*h| / (sigma_h * sqrt h), indexed by START bar
        self.eabs = {}
        for h in HORIZONS:
            e = np.full(n, np.nan)
            if n > h:
                rh = lc[h:] - lc[:-h]
                with np.errstate(invalid="ignore", divide="ignore"):
                    e[:-h] = np.abs(rh - self.mu[:-h] * h) / (self.sig[h][:-h] * math.sqrt(h))
            self.eabs[h] = e

    def own_conf(self, h, W, positions):
        """Mean own |residual| over windows whose outcome ended in the last W bars (start bar
        in (p-W-h, p-h]); NaN when fewer than W/2 are known."""
        cs, cn = _cum(self.eabs[h])
        p = np.asarray(positions, np.int64)
        hi = np.clip(p - h + 1, 0, len(self.eabs[h]))
        lo = np.clip(p - h + 1 - W, 0, len(self.eabs[h]))
        s, c = cs[hi] - cs[lo], cn[hi] - cn[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c >= W // 2, s / np.maximum(c, 1), np.nan)


class Basket:
    """Analog-basket statistics on a calendar-day axis (all causal by day D):
    pooled mean daily log return of every basket bar dated <= D, and cumulative sums of
    the basket's reference |residual| keyed by the END day of each h-day window.

    series: TickerSeries of the basket members, in engine.BASKET order (order matters for
    bit-identical sums)."""

    def __init__(self, series, market):
        S = np.zeros(NDAY)
        N = np.zeros(NDAY)
        CE = {h: np.zeros(NDAY) for h in HORIZONS}
        CN = {h: np.zeros(NDAY) for h in HORIZONS}
        for sb in series:
            td = sb.td
            d = td.day + DAY0
            ok = np.zeros(len(d), bool)
            ok[1:] = td.pool_ok[:-1] & np.isfinite(sb.r[1:])
            np.add.at(S, d[ok], sb.r[ok])
            np.add.at(N, d[ok], 1.0)
            for h in HORIZONS:
                e = sb.eabs[h]
                n = len(e)
                if n <= h:
                    continue
                st = np.arange(n - h)
                g = np.isfinite(e[:n - h]) & td.pool_ok[:n - h]
                np.add.at(CE[h], d[st[g] + h], e[st[g]])
                np.add.at(CN[h], d[st[g] + h], 1.0)
        self.cS, self.cN = np.cumsum(S), np.cumsum(N)
        self.cE = {h: np.cumsum(CE[h]) for h in HORIZONS}
        self.cEN = {h: np.cumsum(CN[h]) for h in HORIZONS}
        self.market = market
        self.members = len(series)

    def drift_at(self, day):
        """(basket mean daily log return, index mean daily log return) as of each day."""
        i = np.asarray(day, np.int64) + DAY0
        with np.errstate(invalid="ignore", divide="ignore"):
            n = self.cN[i]
            mu = np.where(n > 1000, self.cS[i] / n, np.nan)
        return mu, self.market.mean_at(day)

    def conf(self, h, Wd, day):
        """Mean basket |residual| over windows that ended in (D - Wd, D]; NaN under 200."""
        i = np.asarray(day, np.int64) + DAY0
        s = self.cE[h][i] - self.cE[h][i - Wd]
        c = self.cEN[h][i] - self.cEN[h][i - Wd]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c >= 200, s / np.maximum(c, 1), np.nan)


# ---------------------------------------------------------------------------
# The cone
# ---------------------------------------------------------------------------

def dev_quantiles(raw_end, ratio, w):
    """Deviation-scaled analog cone (19 quantiles) from the engine's analog selection:
    raw_end = unscaled analog end log returns, ratio = vol-scaling ratios, w = weights.
    Vol scaling is applied to each analog's deviation from the weighted median only."""
    med = float(wquantile(raw_end, w, [0.5])[0])
    return wquantile(med + ratio * (raw_end - med), w, QS19)


def centre(S, B, positions, h):
    """h-day centre: the stock's own drift shrunk (James-Stein) toward the basket drift."""
    p = np.asarray(positions, np.int64)
    mu_own = S.mu[p]
    mu_pool, mu_mkt = B.drift_at(S.td.day[p])
    tgt = np.where(np.isfinite(mu_pool), mu_pool, np.where(np.isfinite(mu_mkt), mu_mkt, mu_own))
    var = S.s2[p] / np.maximum(S.nh[p], 1.0)
    tau2 = SPEC[h]["tau2"]
    w = tau2 / (tau2 + var)
    return (tgt + w * (mu_own - tgt)) * h


def width(S, B, positions, h):
    """h-day scale: HAR sigma * sqrt(h) times the own and basket conformal factors."""
    p = np.asarray(positions, np.int64)
    s = S.sig[h][p] * math.sqrt(h)
    W, lam = SPEC[h]["own"]
    k = S.own_conf(h, W, p)
    s = s * np.where(np.isfinite(k), k, FILL) ** lam
    Wd, lam = SPEC[h]["pool"]
    k = B.conf(h, Wd, S.td.day[p])
    s = s * np.where(np.isfinite(k), k, FILL) ** lam
    return s


def end_quantiles(S, B, positions, h, qdev):
    """(n, 19) sorted quantiles of the h-day log return at each as-of position.
    qdev: (n, 19) deviation-scaled analog cones; rows of NaN (too few analogs) fall back
    to the plain cone."""
    c = centre(S, B, positions, h)
    s = width(S, B, positions, h)
    Q = c[:, None] + s[:, None] * np.asarray(SPEC[h]["z"], float)[None, :]
    A = np.asarray(qdev, float).reshape(len(c), 19)
    A = np.where(np.isfinite(A), A, Q)
    w = SPEC[h]["w"]
    Q = (1 - w) * Q + w * A
    return np.sort(Q, axis=1)


def daily_paths(Q, h, qs_idx=IDX5):
    """Log-return paths for days 1..h from the end-of-horizon quantiles Q (19,).

    The model is defined at the horizon end only. In between, each band follows a random
    walk with drift: day d = m*d + (Q_h - m*h) * sqrt(d/h), with m = mean(Q_h) / h (the
    cone's mean daily drift; for the plain cone this is exactly the centre, because z has
    mean 0). Bands stay ordered on every day, and day h equals Q_h exactly."""
    Q = np.asarray(Q, float)
    m = float(np.mean(Q)) / h
    d = np.arange(1, h + 1, dtype=float)
    q = Q[list(qs_idx)]
    out = m * d[None, :] + (q[:, None] - m * h) * np.sqrt(d / h)[None, :]
    out[:, -1] = q                                     # exact at the horizon end
    return out


# ---------------------------------------------------------------------------
# Proof: this file == the research harness model cone2_f_conf_dev
# ---------------------------------------------------------------------------

def vs_research():
    """Rebuild research/models_cone2.MODELS['cone2_f_conf_dev'] on a harness Context and
    compare its quantiles with engine.Engine.cone_at (the product path) on many
    (ticker, as-of, h) samples. Read-only: the Context gets a no-op cutoff so the research
    analog memo is neither read nor written."""
    import datetime as dt
    import os
    import sys
    import time
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    import engine as E
    from research import harness
    from research import models_cone2 as mc

    model = mc.MODELS["cone2_f_conf_dev"]
    t0 = time.time()
    ctx = harness.Context(cutoff=pd.Timestamp("2100-01-01"))   # full data; disables the disk memo
    eng = E.Engine(quiet=True)
    eng.warm_up()
    cases = [("NVDA", "2024-06-14"), ("NVDA", None), ("KO", "2008-10-10"), ("SPY", "2001-09-28"),
             ("AAPL", "2015-06-12"), ("IBM", "1987-10-30"), ("XOM", "1975-06-02"),
             ("TSLA", "2019-03-01"), ("TSLA", None), ("PLTR", "2022-12-30"), ("PLTR", None),
             ("SMCI", "2018-02-02"), ("SHOP", "2020-03-20"), ("QQQ", "2008-11-21"),
             ("IBM", "1963-02-01"), ("KO", "1963-06-03"), ("CRWV", None), ("CHYM", None)]
    worst, n = 0.0, 0
    for t, asof in cases:
        td = eng.get_ticker(t)
        if asof is None:
            p = len(td.day) - 1
        else:
            p = td.pos_on_or_before((dt.date.fromisoformat(asof) - dt.date(1970, 1, 1)).days)
        ref = model.predict(ctx, t, np.array([p]), HORIZONS)
        tag = "pool" if t in E.POOL else "outside pool"
        for h in HORIZONS:
            mine = eng.cone_at(td, p, h)
            Qr, Pr = ref[h][0][0], ref[h][1][0]
            dq = float(np.max(np.abs(mine["Q"] - Qr)))
            dp = abs(float(p_up(mine["Q"])[0]) - float(Pr))
            worst = max(worst, dq, dp)
            n += 1
            print(f"{t:5s} {E.day_str(td.day[p])} h={h:2d} ({tag}, {p + 1} bars, n_analogs={mine['n_analogs']:3d}) "
                  f"max|dQ|={dq:.2e} |dp_up|={dp:.2e}  q10/q50/q90={Qr[I10]:+.4f}/{Qr[I50]:+.4f}/{Qr[I90]:+.4f}")
    ok = worst <= 1e-9
    print(f"VS-RESEARCH {'PASSED' if ok else 'FAILED'}: {n} samples, max abs diff {worst:.3e} "
          f"({time.time() - t0:.0f}s)")
    return ok


if __name__ == "__main__":
    import sys
    if "--vs-research" in sys.argv:
        sys.exit(0 if vs_research() else 1)
    print(__doc__)
