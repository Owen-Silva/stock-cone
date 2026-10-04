"""Volatility forecasting for the Stock Cone (research lane `cone-vol`, see research/PLAN.md).

The cone's width is a volatility forecast. This module finds and freezes the best
h-day volatility forecaster (h = 5, 21, 63 trading days) and the standardised
quantiles that turn it into a 19-quantile forecast of the h-day log return.

Public API (import from other research code; everything is causal):

    forecast_sigma(log_returns, t, h, market_log_returns=None, volume=None) -> float
        Daily-unit volatility forecast made at the close of row t for the next h rows.
        The h-day forecast is sigma * sqrt(h). Uses rows 0..t only.

    forecast_sigma_series(log_returns, h, market_log_returns=None, volume=None) -> ndarray
        The same for every t at once (NaN where history < MIN_HISTORY returns).
        forecast_sigma(r, t, h) == forecast_sigma_series(r, h)[t] (checked by `selfcheck`).

    z_quantiles(h, drift=False) -> ndarray of 19
        Empirical quantiles (q = 0.05 .. 0.95) of z = r_h / (sigma * sqrt(h)) fitted on DEV
        and frozen here. drift=True gives the quantiles of (r_h - mu*h) / (sigma*sqrt(h)),
        mu = the stock's walk-forward mean daily log return (see walkforward_drift).

    quantiles(log_returns, t, h, market_log_returns=None, drift=False) -> ndarray of 19
        drift_term + sigma * sqrt(h) * z_q  (zero drift by default).

    MODELS : dict name -> harness model (only filled when research/harness.py exposes a
        clear interface; see the bottom of this file).

Inputs: `log_returns[i] = ln(close[i] / close[i-1])` on the stock's own trading rows
(row 0 may be NaN; NaN anywhere is treated as a missing day). `market_log_returns`
must be aligned to the same rows (SPY daily log returns on the stock's dates, NaN where
missing). Without it the frozen no-market variant is used.

Research pipeline (writes research/results/vol_*; never reads 2019+ data, never reads
TEST tickers):

    python research/volforecast.py research     # fit on DEV, compare on VAL
    python research/volforecast.py selfcheck    # no-look-ahead + frozen-vs-refit checks
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import re
import sys
import time
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "research", "results")

HORIZONS = (5, 21, 63)
QS = np.round(np.arange(1, 20) * 0.05, 2)
Z_GAUSS = np.array([NormalDist().inv_cdf(q) for q in QS])
MIN_HISTORY = 252          # forecasts need >= 252 finite daily returns
VAR_FLOOR = 1e-8           # floor on daily variances before taking logs
EWMA_LAMBDAS = (0.94, 0.97, 0.98, 0.99, 0.995)

# ---------------------------------------------------------------------------
# Causal features. Every array is indexed by t = "as of the close of row t".
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


def features(log_returns, market_log_returns=None, volume=None):
    """All candidate features as a dict of arrays (daily variance units unless noted)."""
    r = np.asarray(log_returns, float)
    n = len(r)
    r2 = r * r
    F = {"n_hist": np.cumsum(np.isfinite(r)).astype(float)}
    F["rv1"] = np.where(np.isfinite(r2), r2, np.nan)
    for k in (5, 22, 66, 126, 252):
        F[f"rv{k}"] = _trail_mean(r2, k, max(1, int(np.ceil(0.8 * k))))
    F["lr756"] = _trail_mean(r2, 756, MIN_HISTORY)          # long-run: up to 3 years
    s = pd.Series(r)
    for k in (21, 63, 126, 252):                              # trailing std (demeaned, ddof 1)
        F[f"sd{k}"] = (s.rolling(k, min_periods=int(0.8 * k)).std() ** 2).to_numpy()
    s2 = pd.Series(r2)
    for lam in EWMA_LAMBDAS:
        F[f"ewma{lam}"] = s2.ewm(alpha=1 - lam, adjust=True, ignore_na=True,
                                 min_periods=21).mean().to_numpy()
    F["mu"] = _trail_mean(r, n + 1, MIN_HISTORY)               # walk-forward mean daily log return
    rv66 = F["rv66"]
    with np.errstate(invalid="ignore", divide="ignore"):
        F["shock"] = np.minimum(np.abs(r) / np.sqrt(rv66), 6.0)
        F["neg22"] = np.minimum(_trail_mean(r, 22, 18) * 22, 0.0)
    F["shock"] = np.where(np.isfinite(F["shock"]), F["shock"], 0.0)
    F["neg22"] = np.where(np.isfinite(F["neg22"]), F["neg22"], 0.0)
    if market_log_returns is not None:
        m = np.asarray(market_log_returns, float)
        F["m_rv22"] = _trail_mean(m * m, 22, 18)
        F["m_rv252"] = _trail_mean(m * m, 252, 200)
    if volume is not None:
        v = np.asarray(volume, float)
        v = np.where(v > 0, v, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            vr = np.log(_trail_mean(v, 5, 3) / _trail_mean(v, 252, 126))
        F["volr"] = np.where(np.isfinite(vr), np.clip(vr, -3, 3), 0.0)
    return F


def _lg(x):
    return np.log(np.maximum(x, VAR_FLOOR))


# Regressor definitions for the log-variance (HAR-family) models.
COLS = {
    "c": lambda F: np.ones_like(F["rv22"]),
    "l5": lambda F: _lg(F["rv5"]),
    "l22": lambda F: _lg(F["rv22"]),
    "l66": lambda F: _lg(F["rv66"]),
    "l252": lambda F: _lg(F["rv252"]),
    "llr": lambda F: _lg(F["lr756"]),
    "lm22": lambda F: _lg(F["m_rv22"]),
    "lm252": lambda F: _lg(F["m_rv252"]),
    "shock": lambda F: F["shock"],
    "neg": lambda F: F["neg22"],
    "volr": lambda F: F["volr"],
}


def design(F, cols):
    return np.column_stack([COLS[c](F) for c in cols])


# ---------------------------------------------------------------------------
# Frozen model. The literal between the BEGIN/END markers is written by
# `python research/volforecast.py freeze` from the DEV fit (research/results/vol_fit.json)
# and is also saved to research/results/vol_params.json; `selfcheck` verifies they match.
#   sigma.market   : har_mx_QLr  (log-variance HAR + long-run + market + shock + drawdown,
#                    coefficients fitted on DEV by QLIKE against r_h^2)
#   sigma.nomarket : har_x_QLr   (same without the two market regressors)
#   z[model][variant][h] : 19 DEV quantiles of the standardised h-day return
# ---------------------------------------------------------------------------

# BEGIN FROZEN
FROZEN = {
 "sigma": {
  "market": {
   "name": "har_mx_QLr",
   "5": {
    "cols": ["c", "l5", "l22", "l66", "l252", "llr", "lm22", "lm252", "shock", "neg"],
    "beta": [-1.4596824566, 0.041239699, 0.040543688, 0.3359948564, 0.4330799388, 0.046024543, 0.2243563697, -0.2947605195, 0.043200362, -1.6331981197]
   },
   "21": {
    "cols": ["c", "l5", "l22", "l66", "l252", "llr", "lm22", "lm252", "shock", "neg"],
    "beta": [-1.736661993, 0.0479271826, -0.0323826945, 0.3239691346, 0.5348536234, 0.0480770585, 0.2149108444, -0.3382484037, 0.0167575551, -1.0409228607]
   },
   "63": {
    "cols": ["c", "l5", "l22", "l66", "l252", "llr", "lm22", "lm252", "shock", "neg"],
    "beta": [-1.9589018833, 0.0346043973, 0.0424721718, 0.3085798358, 0.5233742493, 0.0352268742, 0.1226283271, -0.2956515427, 0.0026135895, -0.6350193238]
   }
  },
  "nomarket": {
   "name": "har_x_QLr",
   "5": {
    "cols": ["c", "l5", "l22", "l66", "l252", "llr", "shock", "neg"],
    "beta": [-1.0960315159, 0.0474558804, 0.1414495805, 0.4030819118, 0.198411777, 0.0715628174, 0.0446007735, -1.8496133022]
   },
   "21": {
    "cols": ["c", "l5", "l22", "l66", "l252", "llr", "shock", "neg"],
    "beta": [-1.0149820598, 0.0535489262, 0.0642356634, 0.3876242753, 0.2769470995, 0.0876351528, 0.0167130088, -1.2177495207]
   },
   "63": {
    "cols": ["c", "l5", "l22", "l66", "l252", "llr", "shock", "neg"],
    "beta": [-0.8693466107, 0.0374053366, 0.0984603868, 0.3565221305, 0.2916591076, 0.0945607759, 0.0028143097, -0.6584959709]
   }
  }
 },
 "z": {
  "har_mx_QLr": {
   "zero": {
    "5": [-1.541338, -1.139691, -0.896503, -0.707847, -0.551702, -0.413395, -0.286623, -0.169296, -0.059457, 0.053204, 0.163861, 0.277584, 0.395336, 0.523321, 0.660496, 0.816408, 1.001666, 1.244849, 1.62333],
    "21": [-1.539933, -1.12103, -0.862745, -0.669805, -0.503998, -0.357885, -0.226101, -0.099758, 0.01808, 0.133581, 0.246303, 0.362643, 0.481567, 0.607224, 0.738569, 0.885758, 1.058929, 1.280755, 1.620927],
    "63": [-1.514771, -1.058588, -0.790635, -0.590071, -0.423595, -0.277775, -0.144463, -0.021825, 0.093334, 0.207754, 0.319599, 0.432742, 0.548639, 0.668017, 0.797861, 0.943341, 1.109333, 1.321882, 1.646649]
   },
   "drift": {
    "5": [-1.595981, -1.193757, -0.949187, -0.76186, -0.604694, -0.465011, -0.33845, -0.220788, -0.111223, 0.000772, 0.112305, 0.226221, 0.342895, 0.470823, 0.608238, 0.763968, 0.950245, 1.194658, 1.573822],
    "21": [-1.658107, -1.234156, -0.973673, -0.777804, -0.61129, -0.465314, -0.332536, -0.20703, -0.08706, 0.027162, 0.140094, 0.255921, 0.374765, 0.498306, 0.631925, 0.780456, 0.955888, 1.18001, 1.51984],
    "63": [-1.722063, -1.256369, -0.985183, -0.778638, -0.610486, -0.46299, -0.329876, -0.204617, -0.084806, 0.026873, 0.141425, 0.256997, 0.373587, 0.495612, 0.627932, 0.773681, 0.944115, 1.16162, 1.493254]
   },
   "centred": {
    "5": [-1.587229, -1.185581, -0.942393, -0.753737, -0.597593, -0.459285, -0.332514, -0.215186, -0.105348, 0.007314, 0.117971, 0.231693, 0.349446, 0.477431, 0.614606, 0.770517, 0.955776, 1.198959, 1.577439],
    "21": [-1.633364, -1.214461, -0.956176, -0.763237, -0.597429, -0.451317, -0.319532, -0.193189, -0.075352, 0.040149, 0.152872, 0.269211, 0.388136, 0.513792, 0.645137, 0.792327, 0.965498, 1.187324, 1.527495],
    "63": [-1.66791, -1.211726, -0.943773, -0.74321, -0.576734, -0.430913, -0.297601, -0.174964, -0.059805, 0.054615, 0.166461, 0.279603, 0.3955, 0.514879, 0.644723, 0.790202, 0.956194, 1.168744, 1.493511]
   }
  },
  "har_x_QLr": {
   "zero": {
    "5": [-1.541134, -1.137159, -0.892878, -0.70408, -0.548028, -0.411142, -0.284555, -0.167836, -0.058886, 0.052828, 0.162096, 0.276122, 0.392776, 0.51956, 0.656746, 0.8126, 0.998411, 1.240235, 1.621964],
    "21": [-1.546598, -1.119284, -0.861915, -0.665776, -0.501966, -0.355273, -0.224249, -0.098624, 0.017732, 0.13194, 0.244243, 0.358742, 0.477635, 0.601107, 0.733706, 0.881124, 1.055177, 1.275421, 1.617983],
    "63": [-1.516733, -1.058927, -0.790977, -0.588516, -0.422212, -0.276979, -0.143702, -0.02168, 0.09223, 0.206076, 0.316183, 0.429871, 0.543693, 0.663394, 0.791764, 0.937437, 1.103731, 1.317879, 1.649031]
   },
   "drift": {
    "5": [-1.596413, -1.188711, -0.944447, -0.756793, -0.600489, -0.463046, -0.33583, -0.219113, -0.110295, 0.000811, 0.111242, 0.224534, 0.341136, 0.467726, 0.605673, 0.761162, 0.947958, 1.189193, 1.573111],
    "21": [-1.661655, -1.231289, -0.970539, -0.774287, -0.60747, -0.462422, -0.329487, -0.204868, -0.086423, 0.027048, 0.138828, 0.253075, 0.371226, 0.494997, 0.629119, 0.776562, 0.952196, 1.17621, 1.518989],
    "63": [-1.731107, -1.25566, -0.984117, -0.778972, -0.60761, -0.461536, -0.327511, -0.203331, -0.08428, 0.02656, 0.139947, 0.25486, 0.371707, 0.491844, 0.622577, 0.767716, 0.938235, 1.159615, 1.498502]
   },
   "centred": {
    "5": [-1.586259, -1.182284, -0.938003, -0.749206, -0.593153, -0.456267, -0.32968, -0.212961, -0.104011, 0.007703, 0.11697, 0.230997, 0.347651, 0.474435, 0.611621, 0.767475, 0.953285, 1.19511, 1.576839],
    "21": [-1.638691, -1.211378, -0.954009, -0.75787, -0.594059, -0.447367, -0.316343, -0.190718, -0.074362, 0.039847, 0.152149, 0.266648, 0.385541, 0.509013, 0.641612, 0.789031, 0.963083, 1.183327, 1.52589],
    "63": [-1.669169, -1.211363, -0.943413, -0.740952, -0.574648, -0.429415, -0.296138, -0.174116, -0.060206, 0.05364, 0.163747, 0.277435, 0.391258, 0.510958, 0.639328, 0.785001, 0.951295, 1.165443, 1.496595]
   }
  }
 }
}
# END FROZEN

Z_VARIANTS = ("zero", "drift", "centred")


def _frozen():
    if FROZEN is None:
        raise RuntimeError("volforecast: no frozen parameters yet (run `research` then `freeze`)")
    return FROZEN


def _model_for(h, have_market):
    return _frozen()["sigma"]["market" if have_market else "nomarket"][str(h)]


def forecast_sigma_series(log_returns, h, market_log_returns=None):
    """Daily-unit sigma forecast for every row t (causal). See module docstring."""
    if h not in HORIZONS:
        raise ValueError(f"h must be one of {HORIZONS}")
    have_m = market_log_returns is not None
    spec = _model_for(h, have_m)
    F = features(log_returns, market_log_returns if have_m else None)
    if have_m:   # a missing market window falls back to the stock's own long-run variance
        for k in ("m_rv22", "m_rv252"):
            F[k] = np.where(np.isfinite(F[k]), F[k], F["lr756"])
    # explicit column sum, not X @ beta: BLAS rounding can depend on the array length,
    # and the harness look-ahead proof demands bit-identical output on truncated data
    logvar = np.zeros(len(F["n_hist"]))
    for c, b in zip(spec["cols"], spec["beta"]):
        logvar = logvar + float(b) * COLS[c](F)
    sig = np.sqrt(np.exp(logvar))
    ok = (F["n_hist"] >= MIN_HISTORY) & np.isfinite(sig)
    return np.where(ok, sig, np.nan)


def forecast_sigma(log_returns, t, h, market_log_returns=None):
    """Daily-unit sigma forecast at row t for the next h rows, using rows 0..t only."""
    r = np.asarray(log_returns, float)[: t + 1]
    m = None if market_log_returns is None else np.asarray(market_log_returns, float)[: t + 1]
    return float(forecast_sigma_series(r, h, m)[-1])


def z_quantiles(h, variant="zero", market=True):
    """19 standardised quantiles (q = .05 .. .95), frozen from DEV.

    variant "zero":    quantiles of r_h / (sigma*sqrt(h)). Use with drift_term = 0; the
                       DEV-pooled average drift (about +0.05 / +0.13 / +0.21 sigma at
                       5 / 21 / 63 days) is baked into the location.
    variant "drift":   quantiles of (r_h - mu*h) / (sigma*sqrt(h)), mu = walkforward_drift.
                       Use with drift_term = mu*h.
    variant "centred": the "zero" quantiles minus their DEV mean (a zero-mean shape). Use
                       with any externally chosen drift_term (e.g. from the drift lane).
    market selects the z of the model forecast_sigma used (with / without market returns).
    """
    if variant not in Z_VARIANTS:
        raise ValueError(f"variant must be one of {Z_VARIANTS}")
    name = _frozen()["sigma"]["market" if market else "nomarket"]["name"]
    return np.asarray(_frozen()["z"][name][variant][str(h)], float)


def walkforward_drift(log_returns):
    """Expanding mean daily log return up to each row (NaN until MIN_HISTORY returns)."""
    return _trail_mean(np.asarray(log_returns, float), len(log_returns) + 1, MIN_HISTORY)


def quantiles(log_returns, t, h, market_log_returns=None, variant="zero", drift_term=None):
    """19 quantiles of the h-day log return made at row t: drift_term + sigma*sqrt(h)*z_q.
    drift_term defaults to 0 for "zero"/"centred" and to the walk-forward mean*h for "drift"."""
    sig = forecast_sigma(log_returns, t, h, market_log_returns)
    if drift_term is None:
        drift_term = (walkforward_drift(np.asarray(log_returns, float)[: t + 1])[-1] * h
                      if variant == "drift" else 0.0)
    return drift_term + sig * np.sqrt(h) * z_quantiles(h, variant, market_log_returns is not None)


# ---------------------------------------------------------------------------
# Research pipeline
# ---------------------------------------------------------------------------

CUTOFF = pd.Timestamp("2018-12-31")      # nothing after this date is ever loaded
DEV_END = pd.Timestamp("2012-01-01")
VAL_START = pd.Timestamp("2012-01-01")


def _read_cache(ticker):
    """engine.load_prices' cached branch, without its Yahoo refresh (no network)."""
    import engine
    path = engine._cache_file(ticker)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return engine._clean(df["close"], df["volume"], df.index)


def _market_returns():
    """SPY daily log returns, extended before SPY's start with ^GSPC (as engine.market_frame)."""
    spy = _read_cache("SPY")
    gs = _read_cache("^GSPC")
    rs = np.log(spy["close"]).diff().dropna()
    rg = np.log(gs["close"]).diff().dropna()
    rg = rg[rg.index < rs.index[0]]
    m = pd.concat([rg, rs]).sort_index()
    return m[m.index <= CUTOFF]


def train_tickers():
    from research import splits
    import engine
    ts = splits.tickers_for("val")
    have = [t for t in ts if os.path.exists(engine._cache_file(t))]
    assert all(splits.is_train_ticker(t) for t in have)
    return have


def build_panel(log=print):
    """One row per (TRAIN ticker, as-of t) with features, targets per horizon and split labels."""
    mkt = _market_returns()
    frames = []
    n_stale = 0
    for tk in train_tickers():
        px = _read_cache(tk)
        px = px[px.index <= CUTOFF]                       # 2019+ never enters memory
        if len(px) < MIN_HISTORY + 70:
            continue
        r = np.log(px["close"]).diff().to_numpy()
        m = mkt.reindex(px.index).to_numpy()
        F = features(r, m, px["volume"].to_numpy())
        d = pd.DataFrame(F, index=px.index)
        n = len(r)
        dates = px.index.to_numpy()
        r0 = np.where(np.isfinite(r), r, 0.0)
        cr = np.concatenate([[0.0], np.cumsum(r0)])
        cr2 = np.concatenate([[0.0], np.cumsum(r0 * r0)])
        for h in HORIZONS:
            idx = np.arange(n)
            end = idx + h
            ok = end < n
            e = np.minimum(end, n - 1)
            d[f"rh{h}"] = np.where(ok, cr[e + 1] - cr[idx + 1], np.nan)
            d[f"rv{h}f"] = np.where(ok, cr2[e + 1] - cr2[idx + 1], np.nan)
            end_date = pd.DatetimeIndex(dates[e])
            asof = px.index
            spl = np.full(n, "", dtype=object)
            spl[(asof < DEV_END) & (end_date < DEV_END) & ok] = "dev"
            spl[(asof >= VAL_START) & (end_date <= CUTOFF) & ok] = "val"
            d[f"split{h}"] = spl
        d["ticker"] = tk
        flat = pd.Series(np.where(np.isfinite(r), (r == 0).astype(float), np.nan))
        stale = (flat.rolling(63).mean() > 0.20).to_numpy()      # engine.py's stale-quote rule
        keep = ((d["n_hist"] >= MIN_HISTORY) & np.isfinite(d["sd63"]) & (d["sd63"] > VAR_FLOOR)
                & np.isfinite(d["lr756"]) & ~stale)
        n_stale += int((stale & (d["n_hist"] >= MIN_HISTORY)).sum())
        frames.append(d[keep])
    P = pd.concat(frames)
    log(f"panel: {len(P):,} rows, {P['ticker'].nunique()} TRAIN tickers, "
        f"{P.index.min().date()} .. {P.index.max().date()}; {n_stale:,} stale-quote rows dropped")
    return P


def fit_ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def fit_qlike(X, y, beta0=None, iters=60):
    """Minimise mean(y*exp(-X b) + X b) (QLIKE, convex in b) by damped Newton."""
    n = len(y)
    if beta0 is None:
        pos = y > 0
        beta0 = fit_ols(X[pos], np.log(y[pos]))
        beta0[0] += np.log(np.mean(y / np.exp(X @ beta0)))
    b = beta0.copy()

    def loss(bb):
        eta = X @ bb
        return np.mean(y * np.exp(-eta) + eta)

    cur = loss(b)
    for _ in range(iters):
        eta = X @ b
        u = y * np.exp(-eta)
        g = X.T @ (1 - u) / n
        H = (X * u[:, None]).T @ X / n
        step = np.linalg.solve(H + 1e-10 * np.eye(len(b)), g)
        t = 1.0
        while t > 1e-6:
            nb = b - t * step
            nl = loss(nb)
            if nl <= cur:
                break
            t *= 0.5
        if cur - nl < 1e-12:
            b, cur = nb, min(cur, nl)
            break
        b, cur = nb, nl
    return b


def metrics(F, rv, rh):
    """F = forecast h-day variance; rv = realised h-day variance; rh = h-day log return."""
    pos = rv > 0
    lr = np.log(F[pos] / rv[pos])
    q = rv[pos] / F[pos]
    return {
        "n": int(len(rv)),
        "qlike": float(np.mean(q - np.log(q) - 1)),
        "logmse": float(np.mean(lr ** 2)),
        "bias_log": float(np.mean(lr)),
        "rv_over_f": float(np.mean(rv / F)),
        "qlike_r": float(np.mean(rh ** 2 / F + np.log(F))),
        "r2_over_f": float(np.mean(rh ** 2 / F)),
    }


def pinball_eval(Q, r, naive_sig):
    """Q: (n,19) quantiles of r; returns coverage, scaled pinball, width."""
    u = r[:, None] - Q
    pin = np.maximum(QS * u, (QS - 1) * u)
    return {
        "cov50": float(np.mean((r >= Q[:, 4]) & (r <= Q[:, 14]))),
        "cov80": float(np.mean((r >= Q[:, 1]) & (r <= Q[:, 17]))),
        "pinball": float(np.mean(pin.mean(axis=1) / naive_sig)),
        "width80": float(np.mean(Q[:, 17] - Q[:, 1])),
    }


SIMPLE = ["sd21", "sd63", "sd126", "sd252", "rv22", "rv66", "lr756"] + [f"ewma{l}" for l in EWMA_LAMBDAS]
HAR_SPECS = {
    "har4": ["c", "l5", "l22", "l66", "l252"],
    "har4lr": ["c", "l5", "l22", "l66", "l252", "llr"],
    "har_m": ["c", "l5", "l22", "l66", "l252", "llr", "lm22", "lm252"],
    "har_mx": ["c", "l5", "l22", "l66", "l252", "llr", "lm22", "lm252", "shock", "neg"],
    "har_mxv": ["c", "l5", "l22", "l66", "l252", "llr", "lm22", "lm252", "shock", "neg", "volr"],
    "har_x": ["c", "l5", "l22", "l66", "l252", "llr", "shock", "neg"],     # no market
}


def _fdict(P):
    return {c: P[c].to_numpy() for c in P.columns if c not in ("ticker",) and not c.startswith("split")}


def research(log=print):
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    P = build_panel(log)
    rows, params = [], {"sigma": {}, "fit_info": {}}
    qrows = []
    zfit = {"zero": {}, "drift": {}, "regime_edges": {}}
    for h in HORIZONS:
        S = P[f"split{h}"].to_numpy()
        D, V = P[S == "dev"], P[S == "val"]
        FD, FV = _fdict(D), _fdict(V)
        yD, yV = FD[f"rv{h}f"], FV[f"rv{h}f"]
        rD, rV = FD[f"rh{h}"], FV[f"rh{h}"]
        log(f"h={h}: DEV {len(D):,} rows / {D['ticker'].nunique()} tickers, "
            f"VAL {len(V):,} rows / {V['ticker'].nunique()} tickers")
        fc = {}   # name -> (F_dev, F_val) as h-day variance

        def add(name, fd, fv, info=None):
            fd, fv = np.maximum(fd, VAR_FLOOR * h), np.maximum(fv, VAR_FLOOR * h)
            fc[name] = (fd, fv)
            for split, F, y, rr in (("dev", fd, yD, rD), ("val", fv, yV, rV)):
                m = metrics(F, y, rr)
                m.update(model=name, h=h, split=split)
                rows.append(m)
            if info is not None:
                params["fit_info"].setdefault(str(h), {})[name] = info

        # 1. simple trailing estimators, raw and with a DEV-fitted scale (QLIKE-optimal)
        for c in SIMPLE:
            fd, fv = np.maximum(FD[c], VAR_FLOOR) * h, np.maximum(FV[c], VAR_FLOOR) * h
            add(c, fd, fv)
            s = float(np.mean(yD / fd))
            add(c + "*s", fd * s, fv * s, {"scale": s})
        # 2. mean reversion: log var = w log ewma0.97 + (1-w) log lr756 + log s, w by DEV QLIKE
        best = None
        for w in np.round(np.arange(0, 1.0001, 0.05), 2):
            f0 = np.exp(w * _lg(FD["ewma0.97"]) + (1 - w) * _lg(FD["lr756"])) * h
            s = np.mean(yD / f0)
            ql = metrics(f0 * s, yD, rD)["qlike"]
            if best is None or ql < best[0]:
                best = (ql, w, s)
        _, w, s = best
        mk = lambda FF: np.exp(w * _lg(FF["ewma0.97"]) + (1 - w) * _lg(FF["lr756"])) * h * s
        add("mr_ewma_lr", mk(FD), mk(FV), {"w": float(w), "scale": float(s)})
        # 3. HAR in variance space (OLS levels)
        XvD = np.column_stack([np.ones(len(D)), FD["rv5"], FD["rv22"], FD["rv66"], FD["rv252"]])
        XvV = np.column_stack([np.ones(len(V)), FV["rv5"], FV["rv22"], FV["rv66"], FV["rv252"]])
        bv = fit_ols(XvD, yD / h)
        fl = lambda X, FF: np.maximum(X @ bv, 0.25 * FF["rv252"]) * h
        add("har4_varOLS", fl(XvD, FD), fl(XvV, FV), {"beta": bv.tolist()})
        # 4. HAR in log space: OLS (+ DEV scale) and direct QLIKE fits on RV and on r_h^2
        for name, cols in HAR_SPECS.items():
            XD, XV = design(FD, cols), design(FV, cols)
            pos = yD > 0
            b = fit_ols(XD[pos], np.log(yD[pos] / h))
            s = float(np.mean(yD / (np.exp(XD @ b) * h)))
            b_ols = b.copy(); b_ols[0] += np.log(s)
            if name == "har4":
                add(name + "_logOLS", np.exp(XD @ b_ols) * h, np.exp(XV @ b_ols) * h,
                    {"cols": cols, "beta": b_ols.tolist()})
            bq = fit_qlike(XD, yD / h, b_ols)
            add(name + "_QL", np.exp(XD @ bq) * h, np.exp(XV @ bq) * h, {"cols": cols, "beta": bq.tolist()})
            br = fit_qlike(XD, rD ** 2 / h, bq)
            add(name + "_QLr", np.exp(XD @ br) * h, np.exp(XV @ br) * h, {"cols": cols, "beta": br.tolist()})
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "vol_losses.csv"), index=False)
        log(f"  sigma models done ({time.time() - t0:.0f}s)")

        # 5. quantiles: z = r_h/(sigma sqrt h) on DEV -> empirical z_q; evaluate on VAL
        naiveD = np.sqrt(FD["sd63"] * h)
        naiveV = np.sqrt(FV["sd63"] * h)
        muD, muV = FD["mu"] * h, FV["mu"] * h
        cand = ["sd63", "ewma0.97*s", "mr_ewma_lr", "har4lr_QL", "har_m_QL", "har_mx_QL",
                "har_mx_QLr", "har_x_QL", "har_x_QLr", "har_mxv_QLr"]
        ratioD = None
        for c in cand:
            sD, sV = np.sqrt(fc[c][0]), np.sqrt(fc[c][1])
            for dname, mD, mV in (("zero", 0.0, 0.0), ("drift", muD, muV)):
                zD = (rD - mD) / sD
                zq = np.quantile(zD, QS)
                variants = {"gauss": (Z_GAUSS, Z_GAUSS), "emp": (zq, zq)}
                # regime-dependent z: terciles of sigma_hat / long-run vol (edges from DEV)
                regD = sD / np.sqrt(FD["lr756"] * h)
                regV = sV / np.sqrt(FV["lr756"] * h)
                edges = np.quantile(regD, [1 / 3, 2 / 3])
                bD, bV = np.digitize(regD, edges), np.digitize(regV, edges)
                zreg = np.vstack([np.quantile(zD[bD == k], QS) for k in range(3)])
                for split, sg, mm, rr, nv, bb in (("dev", sD, mD, rD, naiveD, bD),
                                                  ("val", sV, mV, rV, naiveV, bV)):
                    for vname, (zz, _) in variants.items():
                        Q = mm + np.asarray(sg)[:, None] * np.asarray(zz)[None, :] if np.ndim(mm) == 0 \
                            else mm[:, None] + sg[:, None] * zz[None, :]
                        e = pinball_eval(Q, rr, nv)
                        e.update(model=c, h=h, drift=dname, z=vname, split=split)
                        qrows.append(e)
                    Q = (mm if np.ndim(mm) else np.zeros(len(sg)))[:, None] + sg[:, None] * zreg[bb]
                    e = pinball_eval(Q, rr, nv)
                    e.update(model=c, h=h, drift=dname, z="emp_regime", split=split)
                    qrows.append(e)
                zfit[dname].setdefault(str(h), {})[c] = zq.tolist()
                zfit["regime_edges"].setdefault(str(h), {}).setdefault(c, {})[dname] = {
                    "edges": edges.tolist(), "z": zreg.tolist()}
        pd.DataFrame(qrows).to_csv(os.path.join(RESULTS, "vol_quantiles.csv"), index=False)
        log(f"  quantiles done ({time.time() - t0:.0f}s)")
    params["z"] = zfit
    with open(os.path.join(RESULTS, "vol_fit.json"), "w") as f:
        json.dump(params, f, indent=1)
    log(f"research done in {time.time() - t0:.0f}s")


def summarize():
    L = pd.read_csv(os.path.join(RESULTS, "vol_losses.csv"))
    Q = pd.read_csv(os.path.join(RESULTS, "vol_quantiles.csv"))
    pd.set_option("display.width", 200)
    for h in HORIZONS:
        v = L[(L.h == h) & (L.split == "val")].set_index("model")
        d = L[(L.h == h) & (L.split == "dev")].set_index("model")
        v = v.assign(dev_qlike=d["qlike"]).sort_values("qlike")
        print(f"\n=== h={h} VAL sigma losses ===")
        print(v[["qlike", "dev_qlike", "logmse", "bias_log", "rv_over_f", "qlike_r", "r2_over_f"]].round(4).to_string())
        q = Q[(Q.h == h) & (Q.split == "val")]
        print(f"--- h={h} VAL quantiles ---")
        print(q.sort_values("pinball")[["model", "drift", "z", "cov50", "cov80", "pinball", "width80"]]
              .round(4).head(25).to_string(index=False))


CHOSEN = {"market": "har_mx_QLr", "nomarket": "har_x_QLr"}   # winners on VAL (see report)


def _api_panel(sigma):
    """Panel rows joined with the API's own sigma forecasts (both variants, all horizons)."""
    global FROZEN
    FROZEN = {"sigma": sigma, "z": {}}
    P = build_panel()
    mkt = _market_returns()
    parts = []
    for tk in P["ticker"].unique():
        px = _read_cache(tk)
        px = px[px.index <= CUTOFF]
        r = np.log(px["close"]).diff().to_numpy()
        m = mkt.reindex(px.index).to_numpy()
        d = pd.DataFrame(index=px.index)
        for h in HORIZONS:
            d[f"sigM{h}"] = forecast_sigma_series(r, h, m)
            d[f"sigN{h}"] = forecast_sigma_series(r, h, None)
        d["ticker"] = tk
        parts.append(d)
    S = pd.concat(parts).reset_index()
    P = P.reset_index().merge(S, on=["date", "ticker"], how="left").set_index("date")
    return P


def freeze(log=print):
    """Freeze the chosen sigma models + DEV z-quantiles into this file and evaluate them."""
    fit = json.load(open(os.path.join(RESULTS, "vol_fit.json")))
    sigma = {}
    for key, name in CHOSEN.items():
        sigma[key] = {"name": name}
        for h in HORIZONS:
            fi = fit["fit_info"][str(h)][name]
            sigma[key][str(h)] = {"cols": fi["cols"], "beta": [round(b, 10) for b in fi["beta"]]}
    P = _api_panel(sigma)
    z = {CHOSEN[k]: {v: {} for v in Z_VARIANTS} for k in CHOSEN}
    rows, years = [], []
    for h in HORIZONS:
        S = P[f"split{h}"].to_numpy()
        D, V = P[S == "dev"], P[S == "val"]
        assert np.isfinite(D[f"sigM{h}"]).all() and np.isfinite(V[f"sigM{h}"]).all()
        sq = np.sqrt(h)
        for key, tag in (("market", "M"), ("nomarket", "N")):
            name = CHOSEN[key]
            zz = D[f"rh{h}"] / (D[f"sig{tag}{h}"] * sq)
            zd = (D[f"rh{h}"] - D["mu"] * h) / (D[f"sig{tag}{h}"] * sq)
            z[name]["zero"][str(h)] = np.quantile(zz, QS).round(6).tolist()
            z[name]["drift"][str(h)] = np.quantile(zd, QS).round(6).tolist()
            z[name]["centred"][str(h)] = (np.quantile(zz, QS) - zz.mean()).round(6).tolist()
        for split, X in (("dev", D), ("val", V)):
            rv, rh, mu = X[f"rv{h}f"].to_numpy(), X[f"rh{h}"].to_numpy(), X["mu"].to_numpy() * h
            nsig = np.sqrt(X["sd63"].to_numpy() * h)
            zsd = np.quantile(D[f"rh{h}"] / np.sqrt(D["sd63"] * h), QS)
            cands = [("naive0 (sd63, Gaussian, zero drift)", nsig, 0.0, Z_GAUSS),
                     ("naive_drift (sd63, Gaussian, own drift)", nsig, mu, Z_GAUSS),
                     ("sd63 + DEV empirical z (zero)", nsig, 0.0, zsd)]
            for key, tag in (("market", "M"), ("nomarket", "N")):
                name = CHOSEN[key]
                s = X[f"sig{tag}{h}"].to_numpy() * sq
                cands += [(f"{name} z=zero", s, 0.0, np.array(z[name]["zero"][str(h)])),
                          (f"{name} z=drift (own drift)", s, mu, np.array(z[name]["drift"][str(h)])),
                          (f"{name} z=centred (zero drift)", s, 0.0, np.array(z[name]["centred"][str(h)])),
                          (f"{name} Gaussian (zero drift)", s, 0.0, Z_GAUSS)]
            for label, s, m0, zq in cands:
                Q = (m0[:, None] if np.ndim(m0) else m0) + s[:, None] * zq[None, :]
                e = metrics(s ** 2, rv, rh)
                e.update(pinball_eval(Q, rh, nsig))
                e.update(model=label, h=h, split=split)
                rows.append(e)
                if split == "val" and label in (f"{CHOSEN['market']} z=zero", "naive0 (sd63, Gaussian, zero drift)"):
                    inside = (rh >= Q[:, 1]) & (rh <= Q[:, 17])
                    for y, c in pd.Series(inside, index=X.index.year).groupby(level=0).mean().items():
                        years.append({"h": h, "model": label, "year": y, "cov80": c})
    out = pd.DataFrame(rows)
    base = out[out.model.str.startswith("naive0")].set_index(["h", "split"])["pinball"]
    out["skill"] = 1 - out["pinball"] / [base[(a, b)] for a, b in zip(out.h, out.split)]
    out.to_csv(os.path.join(RESULTS, "vol_final.csv"), index=False)
    pd.DataFrame(years).to_csv(os.path.join(RESULTS, "vol_final_by_year.csv"), index=False)
    params = {"sigma": sigma, "z": z}
    with open(os.path.join(RESULTS, "vol_params.json"), "w") as f:
        json.dump(params, f, indent=1)
    path = os.path.abspath(__file__)
    src = open(path, encoding="utf-8").read()
    a = src.index("# BEGIN FROZEN\n") + len("# BEGIN FROZEN\n")
    b = src.index("# END FROZEN")
    lit = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda mm: "[" + " ".join(mm.group(1).split()) + "]",
                 json.dumps(params, indent=1))
    src = src[:a] + "FROZEN = " + lit + "\n" + src[b:]
    open(path, "w", encoding="utf-8").write(src)
    log("frozen parameters written to volforecast.py and results/vol_params.json")
    pd.set_option("display.width", 220)
    cols = ["qlike", "logmse", "bias_log", "rv_over_f", "r2_over_f", "cov50", "cov80", "pinball", "skill", "width80"]
    for h in HORIZONS:
        print(f"\n=== h={h} VAL ===")
        print(out[(out.h == h) & (out.split == "val")].set_index("model")[cols].round(4).to_string())


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def selfcheck():
    ok = True
    rng = np.random.default_rng(0)
    px = _read_cache("AAPL")
    px = px[px.index <= CUTOFF]
    r = np.log(px["close"]).diff().to_numpy()
    m = _market_returns().reindex(px.index).to_numpy()
    for h in HORIZONS:
        for mm in (m, None):
            full = forecast_sigma_series(r, h, mm)
            ts = rng.choice(np.arange(300, len(r)), 25, replace=False)
            for t in ts:
                one = forecast_sigma(r, t, h, mm)
                # perturbing the future must not change the forecast
                rr = r.copy(); rr[t + 1:] = rng.normal(0, 0.2, len(r) - t - 1)
                m2 = None if mm is None else mm.copy()
                if m2 is not None:
                    m2[t + 1:] = rng.normal(0, 0.2, len(r) - t - 1)
                pert = forecast_sigma_series(rr, h, m2)[t]
                if not (one == full[t] and pert == full[t]):   # bit-identical
                    ok = False
                    print(f"LOOK-AHEAD FAIL h={h} t={t}: {full[t]} {one} {pert}")
    print("no-look-ahead (truncate + perturb future, 150 points):", "PASS" if ok else "FAIL")
    # the literal in this file must equal the saved DEV fit
    saved = json.load(open(os.path.join(RESULTS, "vol_params.json")))
    fit = json.load(open(os.path.join(RESULTS, "vol_fit.json")))
    same = saved == json.loads(json.dumps(FROZEN))
    for key, name in CHOSEN.items():
        for h in HORIZONS:
            b0 = np.array(fit["fit_info"][str(h)][name]["beta"])
            same &= bool(np.allclose(b0, FROZEN["sigma"][key][str(h)]["beta"], atol=1e-9))
    print("frozen literal == results/vol_params.json == DEV fit:", "PASS" if same else "FAIL")
    return ok and same


# ---------------------------------------------------------------------------
# Harness models (research/harness.py interface). The harness only discovers
# research/models_*.py, so wiring needs a one-line research/models_vol.py:
#     from research.volforecast import MODELS
# ---------------------------------------------------------------------------


def _ctx_market(ctx, dates):
    """SPY daily log returns (^GSPC before SPY starts) from the context's own frames,
    aligned to `dates`. Uses only what the context holds, so a truncated context
    (harness look-ahead proof) sees only data up to its cutoff."""
    spy, gs = ctx.raw.get("SPY"), ctx.raw.get("^GSPC")
    parts = []
    if spy is not None and len(spy) > 1:
        rs = np.log(spy["close"]).diff().dropna()
        parts.append(rs)
    if gs is not None and len(gs) > 1:
        rg = np.log(gs["close"]).diff().dropna()
        parts.insert(0, rg[rg.index < parts[0].index[0]] if parts else rg)
    if not parts:
        return None
    return pd.concat(parts).sort_index().reindex(dates).to_numpy()


def _p_up(Q):
    """P(r > 0) read off the 19 quantiles (linear between them, clipped to [.05, .95])."""
    out = np.empty(len(Q))
    for i, row in enumerate(Q):
        if not np.all(np.isfinite(row)):
            out[i] = np.nan
        elif row[0] >= 0:
            out[i] = 0.95
        elif row[-1] <= 0:
            out[i] = 0.05
        else:
            out[i] = 1.0 - float(np.interp(0.0, row, QS))
    return out


class VolCone:
    """Pure volatility cone: Q = drift_term + sigma_hat * sqrt(h) * z_q with the frozen
    har_mx_QLr sigma (market from SPY/^GSPC in the context) and DEV empirical z."""

    def __init__(self, name, variant):
        self.name = name
        self.variant = variant
        self.params = (variant,)

    def predict(self, ctx, t, positions, horizons):
        td = ctx.td(t)
        r = np.concatenate([[np.nan], np.diff(np.asarray(td.logc, float))])
        m = _ctx_market(ctx, td.dates)
        p = np.asarray(positions, dtype=np.int64)
        mu = walkforward_drift(r)[p]
        out = {}
        for h in horizons:
            sig = forecast_sigma_series(r, h, m)[p] * np.sqrt(h)
            z = z_quantiles(h, self.variant, market=m is not None)
            d = mu * h if self.variant == "drift" else np.zeros(len(p))
            Q = d[:, None] + sig[:, None] * z[None, :]
            out[h] = (Q, _p_up(Q))
        return out


MODELS = {
    "vol_har": VolCone("vol_har", "zero"),          # zero drift_term, DEV-pooled location in z
    "vol_har_drift": VolCone("vol_har_drift", "drift"),   # own walk-forward drift
}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "research"
    if cmd == "research":
        research()
        summarize()
    elif cmd == "freeze":
        freeze()
    elif cmd == "summary":
        summarize()
    elif cmd == "selfcheck":
        sys.exit(0 if selfcheck() else 1)
