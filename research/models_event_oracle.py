"""Known earnings dates for the product cone (research lane `cone-earn`).

event-01 found that windows holding an earnings-sized move carry ~2.5x the variance the
volatility forecast expects, but price/volume could only guess the dates (~15% precise).
Yahoo publishes the NEXT scheduled earnings date, so a live forecast can know it. This
module measures what knowing the date is worth on top of the product cone
`cone2_f_conf_dev` (cone_model.py), with two stand-ins for "known dates":

  det   the event-01 detector (research/models_event.py): a volume spike plus an outsized
        move against the market. An ORACLE here: a window counts as "event" when the
        detector later fires inside it, which uses the outcome (it needs a big move), so it
        over-states the value of a calendar. Panel analysis only (never a harness model).
  cal   Yahoo's real earnings history (yfinance get_earnings_dates, cached in
        data/events/history/<T>.json): the dates the product would actually know. A window
        is "event" when an earnings reaction day falls inside (as-of, as-of + h trading days].

The add-on scales the cone's spread around its own mean (the 19-quantile mean, which the
daily path's drift also uses):  Q' = m + f * (Q - m),  m = mean(Q).
  mult  f = F[h]                                  one factor per horizon
  var   f = sqrt(1 + C[h] * (kbar - 1) / h)       kbar = the stock's shrunk mean event-day
                                                  variance ratio r_e^2 / s_e^2 (past events only)
Only windows that contain an event are changed; every other window is the plain cone.

Reaction day of an earnings date d (calendar):  hour >= 16 (after the close) -> the next
NYSE trading day after d;  otherwise -> d itself (or the next trading day if d is closed).
Trading days are PROJECTED from the as-of date with engine.project_trading_days' rule
(weekdays minus engine.nyse_holidays), exactly what the live forecast can do.
In window:  asof < reaction <= the h-th projected trading day after asof.

CLI (DEV / VAL only; TEST is never opened). Outputs research/results/event2_*:
    python research/models_event_oracle.py fetch      # Yahoo earnings history -> data/events/history/
    python research/models_event_oracle.py timing     # reaction-day rule vs volume, history gaps
    python research/models_event_oracle.py panel dev|val
    python research/models_event_oracle.py fit        # DEV grid fit of F / C (det and cal)
    python research/models_event_oracle.py report     # DEV + VAL tables, paired CIs

Harness model (research/harness.py): `event2_cal` = cone2_f_conf_dev + the frozen calendar
add-on (FROZEN below). It reads the cached history only (no network) and passes
`harness.py --lookahead --models event2_cal`.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import datetime as dt
import json
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import engine  # noqa: E402

RESULTS = os.path.join(HERE, "results")
HIST_DIR = os.path.join(ROOT, "data", "events", "history")
HORIZONS = (5, 21, 63)
BASE_MODEL = "cone2_f_conf_dev"
EPOCH = dt.date(1970, 1, 1)

# ---------------------------------------------------------------------------
# FROZEN add-on (fitted on DEV, confirmed on VAL; research/reports/event-02-known-dates.md)
# ---------------------------------------------------------------------------
KBAR = {
    "n_last": 8,        # the stock's last 8 earnings reactions (dated <= as-of)
    "n0": 4.0,          # shrinkage weight (pseudo-events) toward the pool ratio
    "k_pool": 6.3,      # pool mean of capped r_e^2 / s_e^2, DEV-dated calendar reactions (6.34, median 2.23, n=1422)
    "cap": 40.0,        # one reaction's ratio is capped before averaging
    "win": 63,          # trailing window for the robust daily variance s_e^2 (bars before e)
}
# Calendar add-on, kind "var": f = sqrt(1 + C[h] * (kbar - 1) / h) on windows holding an
# earnings reaction. C fitted on DEV event rows (grid step 0.02; results/event2_fit.json):
# 0.80 / 0.62 / 0.22. "var" beat "mult" on VAL at 5 and 21 days without losing on DEV; at 63
# days both add nothing on DEV (mult fits F = 1.00) and var loses on VAL, so C[63] = 0.
FROZEN = {"kind": "var", "C": {5: 0.80, 21: 0.62, 63: 0.0}, "F": {5: 1.36, 21: 1.08, 63: 1.0}}
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Trading-day calendar (engine.project_trading_days' rule, vectorised)
# ---------------------------------------------------------------------------

def _trading_days(y0=1960, y1=2036):
    out = []
    d, end = dt.date(y0, 1, 1), dt.date(y1, 12, 31)
    hol = {}
    while d <= end:
        if d.weekday() < 5:
            if d.year not in hol:
                hol[d.year] = engine.nyse_holidays(d.year)
            if d not in hol[d.year]:
                out.append((d - EPOCH).days)
        d += dt.timedelta(days=1)
    return np.asarray(out, dtype=np.int64)


_TD = None


def tdays():
    global _TD
    if _TD is None:
        _TD = _trading_days()
    return _TD


def horizon_end(day, h):
    """Calendar day of the h-th projected trading day after `day` (int days since epoch)."""
    T = tdays()
    return T[np.searchsorted(T, np.asarray(day, np.int64), "right") + h - 1]


def reaction_days(dates, hours):
    """Reaction trading day per earnings date (int days): hour >= 16 -> next trading day
    after the date; else the date itself, or the next trading day if it is closed."""
    T = tdays()
    d = np.asarray(dates, np.int64)
    amc = np.asarray(hours, float) >= 16.0
    i = np.where(amc, np.searchsorted(T, d, "right"), np.searchsorted(T, d, "left"))
    return T[i]


# ---------------------------------------------------------------------------
# Yahoo earnings history (network only in `fetch`; everything else reads the cache)
# ---------------------------------------------------------------------------

def _hist_path(t):
    return os.path.join(HIST_DIR, f"{t}.json")


def fetch_history(t, limit=100):
    import yfinance as yf
    os.makedirs(HIST_DIR, exist_ok=True)
    rec = {"ticker": t, "fetched": dt.date.today().isoformat(), "events": [], "error": None}
    try:
        df = yf.Ticker(t).get_earnings_dates(limit=limit)
        if df is not None and len(df):
            for ts in df.index:
                ts = pd.Timestamp(ts)
                rec["events"].append({"date": ts.date().isoformat(),
                                      "hour": round(ts.hour + ts.minute / 60.0, 3)})
    except Exception as e:                          # unknown symbol, ETF, Yahoo down
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    with open(_hist_path(t) + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=0)
    os.replace(_hist_path(t) + ".tmp", _hist_path(t))
    return rec


def load_history(t):
    """(dates int64, hours float) of the cached earnings dates, sorted and de-duplicated by
    reaction day; empty arrays when nothing is cached."""
    path = _hist_path(t)
    if not os.path.exists(path):
        return np.zeros(0, np.int64), np.zeros(0)
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    ev = rec.get("events") or []
    if not ev:
        return np.zeros(0, np.int64), np.zeros(0)
    d = np.array([(dt.date.fromisoformat(e["date"]) - EPOCH).days for e in ev], np.int64)
    hr = np.array([float(e.get("hour") or 0.0) for e in ev])
    o = np.argsort(d, kind="stable")
    d, hr = d[o], hr[o]
    R = reaction_days(d, hr)
    _, keep = np.unique(R, return_index=True)       # same reaction day listed twice -> once
    keep = np.sort(keep)
    return d[keep], hr[keep]


def research_tickers():
    from research import harness, splits
    ts = set(splits.tickers_for("dev")) | set(splits.tickers_for("val"))
    ts |= {t for t, _ in harness.LOOKAHEAD_CASES}
    return sorted(ts)


# ---------------------------------------------------------------------------
# Per-stock event-day variance ratio (causal)
# ---------------------------------------------------------------------------

def robust_var(r, win=KBAR["win"]):
    """Robust trailing daily variance from the `win` bars BEFORE each bar."""
    s = pd.Series(np.asarray(r, float))
    return (s.pow(2).rolling(win, min_periods=int(win * 0.65)).median().shift(1) / 0.455).to_numpy()


def event_k(day, logc, R):
    """For reaction days R (sorted): bar index e (first bar dated >= R) and k = r_e^2 / s_e^2."""
    day = np.asarray(day, np.int64)
    r = np.concatenate([[np.nan], np.diff(np.asarray(logc, float))])
    s2 = robust_var(r)
    e = np.searchsorted(day, np.asarray(R, np.int64), "left")
    ok = e < len(day)
    k = np.full(len(e), np.nan)
    ee = e[ok]
    k[ok] = r[ee] ** 2 / np.maximum(s2[ee], 1e-12)
    return e, k


def kbar_at(R, k, asof_days, kb=KBAR):
    """Shrunk mean event ratio per as-of day, from reactions dated <= that day only."""
    kp = kb["k_pool"]
    kc = np.minimum(np.where(np.isfinite(k), k, kp), kb["cap"])
    cs = np.concatenate([[0.0], np.cumsum(kc)])
    n_upto = np.searchsorted(np.asarray(R, np.int64), np.asarray(asof_days, np.int64), "right")
    lo = np.maximum(n_upto - kb["n_last"], 0)
    n = n_upto - lo
    return (cs[n_upto] - cs[lo] + kb["n0"] * kp) / (n + kb["n0"]), n_upto


def in_window(R, asof_days, h):
    """(bool per as-of: a reaction day falls in (asof, horizon end], days to the first one
    in trading days (1..h, 0 when none))."""
    D = np.asarray(asof_days, np.int64)
    R = np.asarray(R, np.int64)
    if len(R) == 0:
        return np.zeros(len(D), bool), np.zeros(len(D), np.int64)
    E = horizon_end(D, h)
    j = np.searchsorted(R, D, "right")                # first reaction after asof
    has = j < len(R)
    nxt = np.where(has, R[np.minimum(j, len(R) - 1)], 0)
    inw = has & (nxt <= E)
    T = tdays()
    steps = np.where(inw, np.searchsorted(T, nxt, "left") - np.searchsorted(T, D, "right") + 1, 0)
    return inw, steps


def widen_factor(h, kbar, frozen=None):
    fz = frozen or FROZEN
    if fz["kind"] == "mult":
        return np.full(np.shape(kbar), float(fz["F"][h]))
    return np.sqrt(1.0 + float(fz["C"][h]) * (np.asarray(kbar, float) - 1.0) / h)


def widen(Q, f):
    """Scale each row's spread around its own mean by f (rows of Q sorted -> stay sorted)."""
    Q = np.asarray(Q, float)
    m = Q.mean(axis=1, keepdims=True)
    return m + np.asarray(f, float).reshape(-1, 1) * (Q - m)


# ---------------------------------------------------------------------------
# Harness model: cone2_f_conf_dev + the frozen calendar add-on
# ---------------------------------------------------------------------------

class Event2Cal:
    cache_deps = ("research/models_cone2.py", "research/volforecast.py", "research/models_base.py",
                  "research/harness.py")

    def __init__(self, name="event2_cal"):
        self.name = name
        self.params = (sorted(KBAR.items()), json.dumps(FROZEN, sort_keys=True))

    def predict(self, ctx, t, positions, horizons):
        from research import models_cone2 as mc
        base = mc.MODELS[BASE_MODEL].predict(ctx, t, positions, horizons)
        td = ctx.td(t)
        p = np.asarray(positions, np.int64)
        D = td.day[p]
        d, hr = load_history(t)
        R = reaction_days(d, hr) if len(d) else np.zeros(0, np.int64)
        Rk = R[R <= td.day[-1]]                       # reactions inside this context's bars
        _, k = event_k(td.day, td.logc, Rk)
        kbar, _ = kbar_at(Rk, k, D)                   # uses reactions dated <= each as-of only
        out = {}
        for h in horizons:
            Q, _ = base[h]
            inw, _ = in_window(R, D, h)
            f = np.where(inw, widen_factor(h, kbar), 1.0)
            Q2 = np.sort(widen(Q, f), axis=1)
            out[h] = (Q2, mc.p_up(Q2))
        return out


MODELS = {"event2_cal": Event2Cal()}


# ---------------------------------------------------------------------------
# Research panel (DEV / VAL)
# ---------------------------------------------------------------------------

def _base_Q(P, h):
    from research import models_cone2 as mc
    with open(os.path.join(RESULTS, "cone2_params.json"), encoding="utf-8") as fh:
        v = json.load(fh)[BASE_MODEL]
    return np.sort(mc.variant_Q(P, v["spec"], v["chosen"], v["prm"], h), axis=1)


def panel_path(split):
    return os.path.join(RESULTS, f"event2_panel_{split}.npz")


def build_panel(split, log=print):
    """Per cone2 panel row: detector truth + add-on (event panel), calendar truth, coverage,
    calendar kbar. Saved to results/event2_panel_<split>.npz."""
    assert split in ("dev", "val")
    from research import harness, splits
    from research import models_cone2 as mc
    t0 = time.time()
    P = mc.load_panel(split)
    E = np.load(os.path.join(RESULTS, f"event_panel_{split}.npz"))
    tl = splits.tickers_for(split)
    ekey = {(tl[a], int(b)): i for i, (a, b) in enumerate(zip(E["tk"], E["pos"]))}
    names = P["tickers"][P["tid"]]
    idx = np.array([ekey[(str(a), int(b))] for a, b in zip(names, P["pos"])])
    n = len(idx)
    out = {"A_det": E["A_s"][idx]}
    for h in HORIZONS:
        out[f"det{h}"] = E[f"truth{h}"][idx].astype(bool)
    ctx = harness.Context()
    cov = np.zeros(n, bool)
    kbar = np.full(n, np.nan)
    nk = np.zeros(n, np.int64)
    cal = {h: np.zeros(n, bool) for h in HORIZONS}
    step = {h: np.zeros(n, np.int64) for h in HORIZONS}
    evs = []
    for ti, t in enumerate(P["tickers"]):
        m = P["tid"] == ti
        d, hr = load_history(str(t))
        if len(d) < 4:
            log(f"  {t}: {len(d)} cached earnings dates, rows not covered")
            continue
        td = ctx.td(str(t))
        R = reaction_days(d, hr)
        D = P["day"][m]
        # covered: as-of on/after the first listed reaction, and not inside a listing gap
        gaps = np.flatnonzero(np.diff(R) > 135)
        c = D >= R[0]
        for g in gaps:
            c &= ~((D >= R[g]) & (D < R[g + 1]))
        cov[m] = c
        Rk = R[R <= td.day[-1]]
        e, k = event_k(td.day, td.logc, Rk)
        kb, nn = kbar_at(Rk, k, D)
        kbar[m], nk[m] = kb, nn
        for h in HORIZONS:
            inw, st = in_window(R, D, h)
            cal[h][m], step[h][m] = inw, st
        lo, hi = splits.split_day_range(split)
        inside = (Rk >= lo) & (Rk <= hi)
        evs.append(pd.DataFrame({"ticker": str(t), "R": Rk[inside], "k": k[inside]}))
        ctx.release(str(t))
    out.update({"cov": cov, "kbar": kbar, "nk": nk})
    for h in HORIZONS:
        out[f"cal{h}"] = cal[h]
        out[f"step{h}"] = step[h]
    np.savez_compressed(panel_path(split), **out)
    pd.concat(evs).to_csv(os.path.join(RESULTS, f"event2_cal_events_{split}.csv"), index=False)
    log(f"panel {split}: {n} rows, covered {cov.mean():.3f} ({time.time() - t0:.0f}s)")
    return out


def load_panel(split):
    z = np.load(panel_path(split))
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Scoring (harness.metrics / pinball_rows on panel rows)
# ---------------------------------------------------------------------------

def _H():
    from research import harness
    return harness


def score(P, h, Q, mask):
    H = _H()
    ok = mask & np.isfinite(P[f"r_{h}"]) & np.isfinite(Q).all(axis=1)
    R = {"r": P[f"r_{h}"][ok], "sn": P[f"sn_{h}"][ok], "base": P[f"base_{h}"][ok], "Q": Q[ok],
         "P": np.full(int(ok.sum()), 0.5)}
    return H.metrics(R)


def paired(P, h, QA, QB, mask, n_boot=2000, seed=0):
    """skill(A) - skill(B) on mask rows, with a 90% ticker-bootstrap interval."""
    H = _H()
    ok = mask & np.isfinite(P[f"r_{h}"]) & np.isfinite(QA).all(axis=1) & np.isfinite(QB).all(axis=1)
    r, sn, tid = P[f"r_{h}"][ok], P[f"sn_{h}"][ok], P["tid"][ok]
    la = H.pinball_rows(QA[ok], r, sn)
    lb = H.pinball_rows(QB[ok], r, sn)
    ln = H.pinball_rows(H.Z19[None, :] * sn[:, None], r, sn)
    ts = np.unique(tid)
    A = np.array([la[tid == t].sum() for t in ts])
    B = np.array([lb[tid == t].sum() for t in ts])
    N = np.array([ln[tid == t].sum() for t in ts])
    idx = np.random.default_rng(seed).integers(0, len(ts), (n_boot, len(ts)))
    dd = (B[idx].sum(1) - A[idx].sum(1)) / N[idx].sum(1)
    return float((B.sum() - A.sum()) / N.sum()), float(np.quantile(dd, 0.05)), float(np.quantile(dd, 0.95))


def both_panels(split):
    from research import models_cone2 as mc
    P = mc.load_panel(split)
    P.update(load_panel(split))
    return P


def event_mask(P, h, src):
    return P[f"{src}{h}"] if src == "det" else (P[f"cal{h}"] & P["cov"])


def universe_mask(P, src):
    return np.ones(len(P["pos"]), bool) if src == "det" else P["cov"].astype(bool)


def addon_Q(P, h, Qb, src, kind, val, kpool=None):
    """Plain cone with the add-on on event rows only (val = F for mult, C for var)."""
    ev = event_mask(P, h, src)
    if kind == "mult":
        f = np.where(ev, val, 1.0)
    else:
        if src == "det":
            A = P["A_det"]                           # event-01 shrunk (kbar - 1), detector events
        else:
            A = P["kbar"] - 1.0
        f = np.where(ev, np.sqrt(1.0 + val * np.maximum(A, 0.0) / h), 1.0)
    return widen(Qb, f)


def fit(log=print):
    """Grid fit on DEV event rows by scaled pinball (non-event rows are unchanged)."""
    H = _H()
    P = both_panels("dev")
    grid_f = np.round(np.arange(1.0, 2.501, 0.01), 2)
    grid_c = np.round(np.arange(0.0, 3.001, 0.02), 2)
    res, rows = {}, []
    for src in ("det", "cal"):
        res[src] = {}
        for h in HORIZONS:
            Qb = _base_Q(P, h)
            ev = event_mask(P, h, src) & np.isfinite(P[f"r_{h}"])
            r, sn = P[f"r_{h}"][ev], P[f"sn_{h}"][ev]
            Qe = Qb[ev]
            m = Qe.mean(axis=1, keepdims=True)
            base = H.pinball_rows(Qe, r, sn).sum()
            best = {}
            for kind, grid in (("mult", grid_f), ("var", grid_c)):
                A = (P["A_det"] if src == "det" else P["kbar"] - 1.0)[ev]
                bl = None
                for g in grid:
                    f = g if kind == "mult" else np.sqrt(1.0 + g * np.maximum(A, 0.0) / h)[:, None]
                    pl = H.pinball_rows(m + f * (Qe - m), r, sn).sum()
                    rows.append({"src": src, "h": h, "kind": kind, "val": float(g), "rel": pl / base})
                    if bl is None or pl < bl[0]:
                        bl = (pl, float(g))
                best[kind] = bl[1]
                log(f"  {src} h={h:2d} {kind:4s}: best {bl[1]:.2f}  event-row pinball {bl[0] / base - 1:+.4f}"
                    f"  (n={int(ev.sum())})")
            # narrowing of non-event rows (information only; not shipped)
            ne = universe_mask(P, src) & ~event_mask(P, h, src) & np.isfinite(P[f"r_{h}"])
            Qn, rn, snn = Qb[ne], P[f"r_{h}"][ne], P[f"sn_{h}"][ne]
            mn = Qn.mean(axis=1, keepdims=True)
            bn = H.pinball_rows(Qn, rn, snn).sum()
            cand = [(H.pinball_rows(mn + g * (Qn - mn), rn, snn).sum(), float(g))
                    for g in np.round(np.arange(0.80, 1.001, 0.01), 2)]
            bnar = min(cand)
            best["narrow"] = bnar[1]
            log(f"  {src} h={h:2d} narrow non-event: best {bnar[1]:.2f}  pinball {bnar[0] / bn - 1:+.4f}")
            res[src][str(h)] = best
    # pool mean of capped calendar event ratios, DEV-dated events
    ce = pd.read_csv(os.path.join(RESULTS, "event2_cal_events_dev.csv"))
    k = np.minimum(ce["k"].to_numpy(float), KBAR["cap"])
    k = k[np.isfinite(k)]
    res["k_pool_dev"] = [float(k.mean()), float(np.median(k)), int(len(k))]
    log(f"  calendar event ratio on DEV: mean(capped) {k.mean():.2f}, median {np.median(k):.2f}, n {len(k)}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "event2_fit_grid.csv"), index=False)
    with open(os.path.join(RESULTS, "event2_fit.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


def variants(fitres):
    """Named variants scored in the report: (label, src, kind, val per h)."""
    out = []
    for src in ("det", "cal"):
        for kind in ("mult", "var"):
            out.append((f"{src}_{kind}", src, kind, {h: fitres[src][str(h)][kind] for h in HORIZONS}))
    return out


def report(log=print):
    with open(os.path.join(RESULTS, "event2_fit.json"), encoding="utf-8") as fh:
        fitres = json.load(fh)
    rows, prow = [], []
    for split in ("dev", "val"):
        P = both_panels(split)
        for h in HORIZONS:
            Qb = _base_Q(P, h)
            for label, src, kind, val in variants(fitres):
                Qa = addon_Q(P, h, Qb, src, kind, val[h])
                uni = universe_mask(P, src)
                ev = event_mask(P, h, src)
                for grp, mask in (("event", ev & uni), ("no_event", ~ev & uni), ("all", uni)):
                    for model, Q in (("plain", Qb), (label, Qa)):
                        if model == "plain" and kind == "var":
                            continue                 # plain rows are listed once per src (mult)
                        mt = score(P, h, Q, mask)
                        rows.append({"split": split, "h": h, "src": src, "variant": label, "model": model,
                                     "group": grp, "n": mt["n"], "skill": mt.get("skill"),
                                     "cov50": mt.get("cov50"), "cov80": mt.get("cov80"),
                                     "width80": mt.get("width80")})
                for grp, mask in (("event", ev & uni), ("all", uni)):
                    d, lo, hi = paired(P, h, Qa, Qb, mask)
                    prow.append({"split": split, "h": h, "variant": label, "group": grp,
                                 "d_skill_bp": d * 1e4, "lo_bp": lo * 1e4, "hi_bp": hi * 1e4})
        log(f"  scored {split}")
    df, pdf = pd.DataFrame(rows), pd.DataFrame(prow)
    df.to_csv(os.path.join(RESULTS, "event2_split.csv"), index=False)
    pdf.to_csv(os.path.join(RESULTS, "event2_paired.csv"), index=False)
    return df, pdf


def check_base(log=print):
    """The panel's plain cone must reproduce the harness JSON of cone2_f_conf_dev."""
    ok = True
    for split in ("dev", "val"):
        from research import models_cone2 as mc
        P = mc.load_panel(split)
        with open(os.path.join(RESULTS, f"{BASE_MODEL}_{split}.json"), encoding="utf-8") as fh:
            J = json.load(fh)["results"]
        for h in HORIZONS:
            mt = score(P, h, _base_Q(P, h), np.ones(len(P["pos"]), bool))
            j = J[str(h)]["all"]
            same = (mt["n"] == j["n"] and abs(mt["skill"] - j["skill"]) < 1e-9
                    and abs(mt["cov80"] - j["cov80"]) < 1e-12 and abs(mt["cov50"] - j["cov50"]) < 1e-12)
            ok &= same
            log(f"  {split} h={h:2d}: panel skill {mt['skill']:+.5f} cov50 {mt['cov50']:.4f} cov80 "
                f"{mt['cov80']:.4f} n {mt['n']} | harness {j['skill']:+.5f} {j['cov50']:.4f} {j['cov80']:.4f} "
                f"{j['n']} {'OK' if same else 'MISMATCH'}")
    return ok


def timing(log=print):
    """Reaction-day rule vs volume: the mapped bar should out-trade the alternative bar
    (the earnings date itself for after-close reports, the next bar for the rest)."""
    from research import harness
    rows = []
    for t in research_tickers():
        px = harness.load_cached(t)
        d, hr = load_history(t)
        if px is None or len(d) == 0:
            rows.append({"ticker": t, "n": int(len(d))})
            continue
        day = ((px.index.values.astype("datetime64[D]").astype(np.int64)))
        vol = px["volume"].to_numpy(float)
        R = reaction_days(d, hr)
        e = np.searchsorted(day, R, "left")
        amc = hr >= 16
        alt = np.where(amc, e - 1, e + 1)
        ok = (e > 0) & (e < len(day) - 1) & (R <= day[-1])
        win = vol[e[ok]] > vol[alt[ok]]
        gaps = np.diff(R)
        rows.append({"ticker": t, "n": int(len(d)), "first": str(EPOCH + dt.timedelta(days=int(R[0]))),
                     "share_amc": float(amc.mean()), "share_hour0": float((hr == 0).mean()),
                     "mapped_bar_more_volume": float(win.mean()) if win.size else np.nan,
                     "max_gap_days": int(gaps.max()) if len(gaps) else 0,
                     "n_gaps_135": int((gaps > 135).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "event2_timing.csv"), index=False)
    return df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    if cmd == "fetch":
        ts = research_tickers() if len(sys.argv) < 3 else sys.argv[2].split(",")
        for i, t in enumerate(ts):
            if os.path.exists(_hist_path(t)) and "--force" not in sys.argv:
                continue
            rec = fetch_history(t)
            print(f"  [{i + 1}/{len(ts)}] {t}: {len(rec['events'])} dates {rec['error'] or ''}", flush=True)
            time.sleep(1.0)
    elif cmd == "timing":
        print(timing().round(3).to_string(index=False))
    elif cmd == "panel":
        build_panel(sys.argv[2])
    elif cmd == "check":
        print("BASE MATCHES HARNESS" if check_base() else "BASE MISMATCH")
    elif cmd == "fit":
        fit()
    elif cmd == "report":
        df, pdf = report()
        print(df.round(4).to_string(index=False))
        print(pdf.round(2).to_string(index=False))
    else:
        print(__doc__)
