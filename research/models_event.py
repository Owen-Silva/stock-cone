"""Earnings-like event risk for the cone (research lane `cone-event`, see research/PLAN.md idea 6).

There is no earnings calendar in the price cache, but earnings days leave a fingerprint in
daily data: a volume spike together with an outsized move against the market, recurring
about once a quarter. This module

  1. detects past event days walk-forward (close + volume only; bar i uses bars <= i),
  2. predicts whether an event falls inside (t, t+h] from the stock's own event rhythm
     (events seen 1, 2, 3 quarters and 1, 2, 3 years before the window),
  3. widens the HAR volatility cone (research/volforecast.py, the vol_har_drift model) when
     an event is predicted inside the window, with an event-day variance add-on estimated
     from the stock's own past event-day moves (shrunk toward a pool average), and
     narrows it slightly when none is due.

Harness models (research/harness.py; all start with "event_" so their result files are
research/results/event_*):

    event_base        no add-on, no deflation: equals vol_har_drift (checked on VAL)
    event_pool_bin    pooled add-on (one event-day variance ratio for every stock), on/off
    event_shrunk_bin  per-stock add-on shrunk toward the pool, on/off
    event_pool        pooled add-on scaled by the rhythm evidence ("soft")
    event_shrunk      per-stock shrunk add-on, soft
    event_har         the final, frozen model (= event_shrunk; see FINAL below)

Research CLI (writes research/results/event_*; never opens TEST):

    python research/models_event.py panel dev|val     # per-row panel -> event_panel_<split>.npz
    python research/models_event.py detector          # detector precision/recall, DEV and VAL
    python research/models_event.py fit               # fit add-on scale / deflation on DEV
    python research/models_event.py split             # coverage/skill by event-in-window, DEV+VAL
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.volforecast import (_ctx_market, _p_up, forecast_sigma_series,  # noqa: E402
                                  walkforward_drift, z_quantiles)

RESULTS = os.path.join(HERE, "results")
HORIZONS = (5, 21, 63)

# ---------------------------------------------------------------------------
# Frozen parameters (all chosen on DEV; see research/reports/event-01-earnings-width.md)
# ---------------------------------------------------------------------------

DET = {
    "win": 63,        # trailing window (bars, ending the day before) for the baselines
    "vr_min": 2.0,    # volume / trailing 63-day median volume, divided by the market's own
                      # volume ratio when that is above 1 (market-wide volume days are not events)
    "z_min": 1.0,     # |stock - market return| / robust trailing sd of that difference
    "gap": 15,        # a new event needs > 15 bars since the previous event (first of a cluster)
}
PRED = {
    "q_lags": (91, 182, 273),     # calendar-day lags: one, two, three quarters back
    "y_lags": (364, 728, 1092),   # 52, 104, 156 weeks back (same weekday, same quarter)
    "q_tol": 10,                  # +- days around a quarterly lag (fiscal quarters drift)
    "y_tol": 7,                   # +- days around a yearly lag
    "y_weight": 2.0,              # yearly lags count double (more regular than quarterly)
    "thr": 0.30,                  # predict "event in window" when weighted hit share >= thr
    "min_valid": 2,               # ... and at least this many lags have history behind them
    "min_gap": 70,                # no event predicted within 70 days after the last event
}
ADD = {
    "n_last": 8,       # stock add-on uses its last 8 detected events
    "n0": 4.0,         # shrinkage weight (pseudo-events) toward the pool ratio
    "k_pool": 8.0,     # pool mean event-day variance ratio r_e^2 / s_e^2, capped, DEV events (8.03, n=5446)
    "cap": 40.0,       # cap on one event's ratio before averaging (one crash day is not a rhythm)
}
# Per horizon: (kind, c, d). sig = HAR daily sigma, base = h*sig^2, A = event-day ratio - 1,
# s = the predictor's weighted hit share (0 when gap-blocked), due = s >= PRED["thr"].
#   kind "binary": V = base + c*A*sig^2 if due, else d*base
#   kind "soft":   V = d*base + c*s*A*sig^2            (add-on grows with the rhythm evidence)
# c and d were grid-searched jointly on DEV by scaled pinball (c step 0.025, d step 0.025;
# `python research/models_event.py fit`, results/event_fit.json). At h = 63 almost every window
# is "due" (97%), so the add-on there is set to 0 and only the rare not-due window is narrowed.
SCALE = {
    "pool_bin":   {5: ("binary", 0.050, 0.975), 21: ("binary", 0.150, 0.975), 63: ("binary", 0.0, 0.900)},
    "shrunk_bin": {5: ("binary", 0.050, 0.975), 21: ("binary", 0.150, 0.975), 63: ("binary", 0.0, 0.900)},
    "pool":       {5: ("soft", 0.150, 0.975), 21: ("soft", 0.375, 0.975), 63: ("binary", 0.0, 0.900)},
    "shrunk":     {5: ("soft", 0.150, 0.975), 21: ("soft", 0.375, 0.975), 63: ("binary", 0.0, 0.900)},
}
FINAL = "shrunk"       # chosen on VAL (best h = 5 skill and due-window coverage; see report)
VAR_FLOOR_MULT = 0.5   # the adjusted h-day variance never goes below half the HAR variance


def adjusted_variance(sig_d, h, score, A, kind, c, d, thr=None):
    """h-day variance after the event adjustment (see SCALE)."""
    thr = PRED["thr"] if thr is None else thr
    base = h * sig_d ** 2
    if kind == "binary":
        V = np.where(score >= thr, base + c * A * sig_d ** 2, d * base)
    elif kind == "soft":
        V = d * base + c * score * A * sig_d ** 2
    else:
        raise ValueError(kind)
    return np.maximum(V, VAR_FLOOR_MULT * base)


# ---------------------------------------------------------------------------
# Detector (causal: everything at bar i uses bars <= i)
# ---------------------------------------------------------------------------

def _days(dates):
    return pd.DatetimeIndex(dates).values.astype("datetime64[D]").astype(np.int64)


def detect(close, volume, mret, mvol, det=DET):
    """Event bars from one stock's closes and volumes.

    close, volume: arrays on the stock's rows. mret: market daily log returns aligned to the
    same rows (NaN = missing -> treated as 0). mvol: market volume aligned (NaN allowed).
    Returns (events int array of bar indices, k array of the same length = r_e^2 / s_e^2,
    ok bool array = rows where the detector's inputs exist)."""
    w = det["win"]
    mp = max(40, int(w * 0.65))
    lc = np.log(np.asarray(close, float))
    r = pd.Series(np.concatenate([[np.nan], np.diff(lc)]))
    m = pd.Series(np.asarray(mret, float)).fillna(0.0)
    ar = r - m
    sd_ar = np.sqrt(ar.pow(2).rolling(w, min_periods=mp).median().shift(1) / 0.455)
    s2_r = r.pow(2).rolling(w, min_periods=mp).median().shift(1) / 0.455
    v = pd.Series(np.asarray(volume, float))
    v = v.where(v > 0)
    vr = v / v.rolling(w, min_periods=mp).median().shift(1)
    mv = pd.Series(np.asarray(mvol, float))
    mv = mv.where(mv > 0)
    mvr = (mv / mv.rolling(w, min_periods=mp).median().shift(1)).clip(lower=1.0).fillna(1.0)
    vr = vr / mvr
    z = ar.abs() / sd_ar
    ok = (np.isfinite(vr) & np.isfinite(z) & (sd_ar > 0)).to_numpy()
    cand = ok & (vr >= det["vr_min"]).to_numpy() & (z >= det["z_min"]).to_numpy()
    ev = []
    last = -10 ** 9
    for i in np.flatnonzero(cand):
        if i - last > det["gap"]:
            ev.append(i)
            last = i
    ev = np.asarray(ev, dtype=np.int64)
    k = (r.to_numpy()[ev] ** 2) / np.maximum(s2_r.to_numpy()[ev], 1e-12)
    k = np.where(np.isfinite(k), k, np.nan)
    return ev, k, ok


def _mkt_volume(ctx, dates):
    """^GSPC volume from the context's own frames, aligned to dates (NaN where missing)."""
    gs = ctx.raw.get("^GSPC")
    if gs is None or len(gs) == 0:
        return np.full(len(dates), np.nan)
    return gs["volume"].reindex(dates).to_numpy(float)


# ---------------------------------------------------------------------------
# Predictor: is an event due inside (t, t+h]?
# ---------------------------------------------------------------------------

def hcal(h):
    """Calendar days spanned by h trading days (approximation known at t)."""
    return int(round(h * 365.25 / 252.0))


def predict_due(day, ev_day, first_ok_day, pos, h, pred=PRED, return_score=False):
    """Boolean per position: event predicted inside (t, t+h]. Uses only events dated <= day[t]
    (which are exactly the events at bars <= t, since an event at bar i is known at bar i)."""
    D = day[pos]
    Hc = hcal(h)
    num = np.zeros(len(pos))
    den = np.zeros(len(pos))
    nvalid = np.zeros(len(pos), dtype=np.int64)
    lags = [(L, pred["q_tol"], 1.0) for L in pred["q_lags"]] + \
           [(L, pred["y_tol"], pred["y_weight"]) for L in pred["y_lags"]]
    for L, tol, wgt in lags:
        lo = D - L - tol                       # (lo, hi] window shifted back by L
        hi = np.minimum(D + Hc - L + tol, D)   # never look past the as-of date
        valid = (lo >= first_ok_day) & (hi > lo)
        cnt = np.searchsorted(ev_day, hi, "right") - np.searchsorted(ev_day, lo, "right")
        hit = (cnt > 0) & valid
        num += wgt * hit
        den += wgt * valid
        nvalid += valid
    score = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    j = np.searchsorted(ev_day, D, "right") - 1
    last = np.where(j >= 0, ev_day[np.maximum(j, 0)], -10 ** 9)
    # effective score: 0 when too few lags have history or the last event is too recent
    score = np.where((nvalid >= pred["min_valid"]) & ~(D + Hc < last + pred["min_gap"]), score, 0.0)
    due = score >= pred["thr"]
    if return_score:
        return due, score
    return due


def addon_ratio(ev, k, pos, add=ADD, per_stock=True):
    """Event-day variance excess (k - 1) per position, from events at bars <= pos only."""
    kp = add["k_pool"]
    if not per_stock:
        return np.full(len(pos), kp - 1.0)
    kc = np.minimum(np.where(np.isfinite(k), k, kp), add["cap"])
    cs = np.concatenate([[0.0], np.cumsum(kc)])
    n_upto = np.searchsorted(ev, pos, "right")          # events at bars <= pos
    lo = np.maximum(n_upto - add["n_last"], 0)
    n = n_upto - lo
    s = cs[n_upto] - cs[lo]
    kbar = (s + add["n0"] * kp) / (n + add["n0"])
    return kbar - 1.0


# ---------------------------------------------------------------------------
# Per-ticker features shared by the model and the research panel
# ---------------------------------------------------------------------------

def ticker_inputs(ctx, t):
    td = ctx.td(t)
    r = np.concatenate([[np.nan], np.diff(np.asarray(td.logc, float))])
    m = _ctx_market(ctx, td.dates)
    mv = _mkt_volume(ctx, td.dates)
    vol = td.px["volume"].to_numpy(float)
    ev, k, ok = detect(td.close, vol, m if m is not None else np.full(len(r), np.nan), mv)
    day = _days(td.dates)
    first_ok = day[np.argmax(ok)] if ok.any() else 10 ** 9
    return td, r, m, day, ev, k, first_ok


class EventHar:
    """vol_har_drift cone with an event adjustment of its h-day variance (see SCALE and
    adjusted_variance). A = event-day variance ratio - 1, pooled (mode *pool*) or per-stock
    shrunk toward the pool (mode *shrunk*). Quantiles use volforecast's frozen "drift" z with
    drift_term = mu*h, exactly as vol_har_drift; mode "none" reproduces vol_har_drift."""

    cache_deps = ("research/volforecast.py",)

    def __init__(self, name, mode):
        self.name = name
        self.mode = mode                      # "none" | key of SCALE
        self.scale = {} if mode == "none" else SCALE[mode]
        self.per_stock = mode.startswith("shrunk")
        self.params = (mode, sorted(self.scale.items()), sorted(DET.items()), sorted(PRED.items()),
                       sorted(ADD.items()), VAR_FLOOR_MULT)

    def predict(self, ctx, t, positions, horizons):
        td, r, m, day, ev, k, first_ok = ticker_inputs(ctx, t)
        p = np.asarray(positions, dtype=np.int64)
        mu = walkforward_drift(r)[p]
        ev_day = day[ev]
        A = addon_ratio(ev, k, p, per_stock=self.per_stock)
        out = {}
        for h in horizons:
            sig_d = forecast_sigma_series(r, h, m)[p]
            z = z_quantiles(h, "drift", market=m is not None)
            if self.mode == "none" or h not in self.scale:
                sig = sig_d * np.sqrt(h)
            else:
                kind, c, d = self.scale[h]
                _, score = predict_due(day, ev_day, first_ok, p, h, return_score=True)
                sig = np.sqrt(adjusted_variance(sig_d, h, score, A, kind, c, d))
            Q = (mu * h)[:, None] + sig[:, None] * z[None, :]
            out[h] = (Q, _p_up(Q))
        return out


def _models():
    ms = {"event_base": EventHar("event_base", "none")}
    for mode in SCALE:
        ms[f"event_{mode}"] = EventHar(f"event_{mode}", mode)
    ms["event_har"] = EventHar("event_har", FINAL)
    return ms


MODELS = _models()


# ---------------------------------------------------------------------------
# Research pipeline (DEV / VAL only)
# ---------------------------------------------------------------------------

def _harness():
    from research import harness
    return harness


def build_panel(split, log=print):
    """Per grid row (TRAIN tickers, the harness grid): outcome, HAR sigma, drift, due/truth
    flags per horizon, add-on ratios. Saved to results/event_panel_<split>.npz."""
    assert split in ("dev", "val")
    H = _harness()
    ctx = H.Context()
    cols = {k: [] for k in ("tk", "pos", "day", "A_s", "A_p")}
    per_h = {h: {k: [] for k in ("ok", "r", "sn", "base", "sig", "mu", "sigd", "due", "score", "truth", "nev")}
             for h in HORIZONS}
    from research import splits
    ev_stats = {"n_events": 0, "years": 0.0}
    evc = {k: [] for k in ("ev_tk", "ev_day", "ev_k", "ev_in")}
    tk_first, tk_n = [], []
    t0 = time.time()
    for ti, t in enumerate(splits.tickers_for(split)):
        try:
            td, r, m, day, ev, k, first_ok = ticker_inputs(ctx, t)
        except Exception as e:      # engine.ForecastError: unusable ticker
            log(f"  skip {t}: {e}")
            continue
        pos = H.grid_positions(td, split, H.STEP_FULL, H.HORIZONS)
        ctx.release(t)
        if len(pos) == 0:
            continue
        lo_d, hi_d = splits.split_day_range(split)
        inside = (day[ev] >= lo_d) & (day[ev] <= hi_d)
        ev_stats["n_events"] += int(inside.sum())
        ev_stats["years"] += float((min(day[-1], hi_d) - max(day[pos[0]], lo_d)) / 365.25)
        ev_day = day[ev]
        evc["ev_tk"].append(np.full(len(ev), ti))
        evc["ev_day"].append(ev_day)
        evc["ev_k"].append(k)
        evc["ev_in"].append(inside.astype(np.int64))
        while len(tk_first) < ti:
            tk_first.append(10 ** 9)
        tk_first.append(int(first_ok))
        mu = walkforward_drift(r)
        cols["tk"].append(np.full(len(pos), ti))
        cols["pos"].append(pos)
        cols["day"].append(day[pos])
        cols["A_s"].append(addon_ratio(ev, k, pos, per_stock=True))
        cols["A_p"].append(addon_ratio(ev, k, pos, per_stock=False))
        n = len(day)
        for h in HORIZONS:
            ok, rr, sn, base, sig = H.outcome_rows(td, pos, h)
            due, score = predict_due(day, ev_day, first_ok, pos, h, return_score=True)
            # truth: the same detector fires at a bar in (p, p+h] (detector is causal, so the
            # full-data event list equals what it would say at bar p+h)
            nev = np.searchsorted(ev, np.minimum(pos + h, n - 1), "right") - np.searchsorted(ev, pos, "right")
            sd = forecast_sigma_series(r, h, m)[pos]
            P = per_h[h]
            P["ok"].append(ok)
            # pad outcome arrays back to len(pos) (NaN where unknown)
            for key, arr in (("r", rr), ("sn", sn), ("base", base), ("sig", sig)):
                full = np.full(len(pos), np.nan)
                full[ok] = arr
                P[key].append(full)
            P["mu"].append(mu[pos])
            P["sigd"].append(sd)
            P["due"].append(due)
            P["score"].append(score)
            P["truth"].append(nev > 0)
            P["nev"].append(nev)
        if ti % 10 == 0:
            log(f"  [{ti}] {t} {len(pos)} rows, {len(ev)} events ({time.time() - t0:.0f}s)")
    out = {k: np.concatenate(v) for k, v in cols.items()}
    for h in HORIZONS:
        for k, v in per_h[h].items():
            out[f"{k}{h}"] = np.concatenate(v)
    for k, v in evc.items():
        out[k] = np.concatenate(v)
    out["tk_first"] = np.asarray(tk_first, dtype=np.int64)
    out["ev_per_year"] = np.float64(ev_stats["n_events"] / max(ev_stats["years"], 1e-9))
    path = os.path.join(RESULTS, f"event_panel_{split}.npz")
    np.savez_compressed(path, **out)
    log(f"panel {split}: {len(out['pos'])} rows -> {path} ({time.time() - t0:.0f}s)")
    return out


def load_panel(split):
    path = os.path.join(RESULTS, f"event_panel_{split}.npz")
    if not os.path.exists(path):
        return build_panel(split)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def detector_report(splits_=("dev", "val")):
    rows = []
    for sp in splits_:
        P = load_panel(sp)
        for h in HORIZONS:
            ok = P[f"ok{h}"]
            due, tr = P[f"due{h}"][ok], P[f"truth{h}"][ok]
            tp = np.sum(due & tr)
            rows.append({"split": sp, "h": h, "n": int(ok.sum()),
                         "base_rate": float(tr.mean()), "pred_rate": float(due.mean()),
                         "precision": float(tp / max(due.sum(), 1)), "recall": float(tp / max(tr.sum(), 1)),
                         "p_event_if_not_pred": float(np.mean(tr[~due])) if (~due).any() else float("nan"),
                         "lift": float(tp / max(due.sum(), 1) / max(tr.mean(), 1e-9)),
                         "ev_per_year": float(P["ev_per_year"])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "event_detector.csv"), index=False)
    return df


# --- scoring on the panel with the harness's own metric function -----------

def quantiles_panel(P, h, mode, scale=None, market=True):
    sigd, mu = P[f"sigd{h}"], P[f"mu{h}"]
    z = z_quantiles(h, "drift", market=market)
    if mode == "none":
        V = h * sigd ** 2
    else:
        kind, c, d = (scale or SCALE)[mode][h]
        A = P["A_s"] if mode.startswith("shrunk") else P["A_p"]
        V = adjusted_variance(sigd, h, P[f"score{h}"], A, kind, c, d)
    return (mu * h)[:, None] + np.sqrt(V)[:, None] * z[None, :]


def score_panel(P, h, Q, mask=None, fast=False):
    """harness.metrics on panel rows. fast=True skips the per-row p_up (direction metrics
    are then meaningless; pinball / coverage / width are unaffected)."""
    H = _harness()
    ok = P[f"ok{h}"] & np.isfinite(Q).all(axis=1)
    if mask is not None:
        ok &= mask
    pu = np.full(int(ok.sum()), 0.5) if fast else _p_up(Q[ok])
    R = {"r": P[f"r{h}"][ok], "sn": P[f"sn{h}"][ok], "base": P[f"base{h}"][ok], "Q": Q[ok], "P": pu}
    return H.metrics(R)


def _pinball_sum(P, h, V, ok):
    H = _harness()
    z = z_quantiles(h, "drift")
    Q = (P[f"mu{h}"][ok] * h)[:, None] + np.sqrt(V[ok])[:, None] * z[None, :]
    return float(H.pinball_rows(Q, P[f"r{h}"][ok], P[f"sn{h}"][ok]).sum())


def fit_scales(log=print):
    """Joint grid search of (c_h, d_h) per horizon, kind and add-on type on DEV, by scaled
    pinball over all DEV rows. Writes results/event_fit.json and event_fit_grid.csv."""
    P = load_panel("dev")
    grid_c = np.round(np.arange(0.0, 0.601, 0.025), 3)
    grid_d = np.round(np.arange(0.85, 1.051, 0.025), 3)
    res, table = {}, []
    for addon in ("pool", "shrunk"):
        A = P["A_s"] if addon == "shrunk" else P["A_p"]
        for kind in ("binary", "soft"):
            key = f"{addon}_{kind}"
            res[key] = {}
            for h in HORIZONS:
                sd = P[f"sigd{h}"]
                ok = P[f"ok{h}"] & np.isfinite(sd) & np.isfinite(P[f"mu{h}"])
                base_pl = _pinball_sum(P, h, h * sd ** 2, ok)
                best = None
                for c in grid_c:
                    for d in grid_d:
                        V = adjusted_variance(sd, h, P[f"score{h}"], A, kind, c, d)
                        pl = _pinball_sum(P, h, V, ok)
                        table.append({"variant": key, "h": h, "c": c, "d": d,
                                      "pinball_rel_to_none": pl / base_pl})
                        if best is None or pl < best[0]:
                            best = (pl, float(c), float(d))
                res[key][str(h)] = [kind, best[1], best[2]]
                log(f"  {key:14s} h={h:2d}: c={best[1]:.3f} d={best[2]:.3f}  "
                    f"pinball vs none {best[0] / base_pl - 1:+.5f}")
    pd.DataFrame(table).to_csv(os.path.join(RESULTS, "event_fit_grid.csv"), index=False)
    with open(os.path.join(RESULTS, "event_fit.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def kpool_dev():
    """Pool mean event-day ratio on DEV (capped at ADD['cap']), DEV-dated events only."""
    P = load_panel("dev")
    k = np.minimum(P["ev_k"][np.isfinite(P["ev_k"]) & (P["ev_in"] > 0)], ADD["cap"])
    return float(k.mean()), float(np.median(k)), int(len(k))


def split_report(splits_=("dev", "val"), modes=None):
    """Coverage / skill overall and by event-in-window, for every mode vs no add-on.
    Groups: due = predictor says an event falls in (t, t+h]; event = the detector later fired
    in (t, t+h]."""
    modes = modes or (["none"] + list(SCALE))
    rows = []
    for sp in splits_:
        P = load_panel(sp)
        for h in HORIZONS:
            due = P[f"score{h}"] >= PRED["thr"]
            for mode in modes:
                Q = quantiles_panel(P, h, mode)
                for grp, mask in (("all", None), ("due", due), ("not_due", ~due),
                                  ("event", P[f"truth{h}"]), ("no_event", ~P[f"truth{h}"])):
                    mt = score_panel(P, h, Q, mask, fast=True)
                    rows.append({"split": sp, "h": h, "model": mode, "group": grp, "n": mt["n"],
                                 "skill": mt.get("skill"), "pinball": mt.get("pinball"),
                                 "pinball_naive0": mt.get("pinball_naive0"),
                                 "cov50": mt.get("cov50"), "cov80": mt.get("cov80"),
                                 "width80": mt.get("width80")})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "event_split.csv"), index=False)
    return df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "detector"
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    if cmd == "panel":
        build_panel(sys.argv[2])
    elif cmd == "detector":
        print(detector_report().round(3).to_string(index=False))
    elif cmd == "fit":
        print("k_pool (mean, median, n) on DEV events:", kpool_dev())
        fit_scales()
    elif cmd == "split":
        print(split_report().round(4).to_string(index=False))
