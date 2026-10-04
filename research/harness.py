"""Walk-forward evaluation harness for the cone (research/PLAN.md).

    python research/harness.py --models naive0,naive_drift,v0 --split val [--fast] [--horizons 5,21,63]
    python research/harness.py --lookahead --models mymodel     # no-look-ahead proof for any model
    python research/harness.py --selftest                       # all three harness proofs

MODEL INTERFACE (put models in research/models_<name>.py, each exposing MODELS = {name: model}):

    class MyModel:
        name = "mymodel"
        cache_deps = ()          # optional: extra files whose edits must invalidate the cache
        def predict(self, ctx, ticker, positions, horizons):
            # ctx.td(ticker)  -> engine.TickerData (day, close, logc, sig, X, valid, raw, ...)
            # ctx.eng         -> offline engine.Engine (pool_for, _select, _paths, ...)
            # positions       -> sorted int64 array of as-of bar indices into ctx.td(ticker)
            # horizons        -> tuple of ints, e.g. (5, 21, 63)
            # return {h: (Q, p_up)}: Q (len(positions), 19) quantiles of r = ln(P[p+h]/P[p])
            #   at QS19 = 0.05..0.95 ascending; p_up (len(positions),) = P(r > 0).
            #   A row of NaN means "abstain" (scored as missing, reported as n_abstain).
            # RULE: use only bars <= p for this ticker, and for any other series only data
            #   dated <= the as-of date; an analog dated A is usable only if A + h <= as-of.

The harness never passes outcomes to a model. Scoring happens here, once, the same for all.
Predictions are cached per (model, code hash, split, step, ticker) under
research/results/partial/, so an interrupted run resumes; editing a model's file (or
engine.py, or a file in model.cache_deps) invalidates its cache. --fresh ignores caches.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # other lanes share this PC: cap BLAS threads

import argparse
import contextlib
import glob
import hashlib
import importlib
import inspect
import io
import json
import math
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if __name__ == "__main__":                 # model files do `from research import harness`:
    sys.modules.setdefault("research.harness", sys.modules[__name__])   # same module object

import engine                               # noqa: E402
from research import splits                 # noqa: E402

# ---------------------------------------------------------------------------
# Constants shared with model files
# ---------------------------------------------------------------------------
QS19 = np.round(np.arange(1, 20) * 0.05, 2)                       # 0.05 .. 0.95
Z19 = np.array([NormalDist().inv_cdf(float(q)) for q in QS19])    # standard normal quantiles
I10, I25, I50, I75, I90 = 1, 4, 9, 14, 17                         # indices into QS19
HORIZONS = (5, 21, 63)
WARMUP = 300               # first as-of bar index (>= 300 trading days of history)
STEP_FULL = 5              # as-of every 5 trading days
STEP_FAST = 20             # --fast: every 20 trading days ...
FAST_EVERY = 3             # ... on every 3rd ticker of the sorted list
GRID_VERSION = "grid-1"    # bump if the grid or the cached payload changes

RESULTS = os.path.join(HERE, "results")
PARTIAL = os.path.join(RESULTS, "partial")
_EPOCH = pd.Timestamp("1970-01-01")


def norm_cdf(x):
    """Vectorised standard normal CDF (no scipy)."""
    x = np.asarray(x, float)
    return 0.5 * (1.0 + np.vectorize(math.erf, otypes=[float])(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Data context: the real engine, offline, on the cached prices
# ---------------------------------------------------------------------------

def load_cached(t):
    """Price frame for t from data/prices (exactly what engine.load_prices reads), or None."""
    path = engine._cache_file(t)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return engine._clean(df["close"], df["volume"], df.index)


class Context:
    """An offline engine.Engine built from the cached CSVs (zero network).

    cutoff (pd.Timestamp, optional): truncate EVERY series at this date. Used by the
    look-ahead proof; a model must give identical predictions on a truncated context."""

    def __init__(self, raw=None, cutoff=None):
        if raw is None:
            raw = {}
            for t in engine.POOL + [engine.MARKET_SPLICE]:
                px = load_cached(t)
                if px is not None:
                    raw[t] = px
        if cutoff is not None:
            raw = {t: px[px.index <= cutoff] for t, px in raw.items()}
        self.cutoff = cutoff
        self.raw = raw
        self.eng = engine.Engine(raw=raw, quiet=True)
        with contextlib.redirect_stdout(io.StringIO()):   # "pool ticker skipped" noise
            self.eng.warm_up()

    def td(self, t):
        """engine.TickerData for t (raises engine.ForecastError if unusable)."""
        if t not in self.raw:
            px = load_cached(t)
            if px is None:
                raise engine.ForecastError(404, f"no cached prices for {t}")
            if self.cutoff is not None:
                px = px[px.index <= self.cutoff]
            self.raw[t] = px
        return self.eng.get_ticker(t)

    def release(self, t):
        """Drop t's analog pool (memory); TickerData stays."""
        with self.eng.lock:
            if t not in engine.POOL:
                self.eng.pools.pop(t, None)


def day_of(date):
    return int((pd.Timestamp(date) - _EPOCH).days)


def grid_positions(td, split, step, horizons):
    """As-of bar indices: every `step` bars from WARMUP, features valid, not inside a
    stale-quote stretch (engine.pool_ok), date inside the split, and the shortest
    horizon's outcome known. Longer horizons drop their own last rows when scoring."""
    n = len(td.day)
    lo, hi = splits.split_day_range(split)
    p = np.arange(WARMUP, n, step, dtype=np.int64)
    p = p[p + min(horizons) < n]
    p = p[td.pool_ok[p] & (td.day[p] >= lo) & (td.day[p] <= hi)]
    return p


# ---------------------------------------------------------------------------
# Model discovery and caching
# ---------------------------------------------------------------------------

def discover_models(wanted=None):
    """{name: model} from every research/models_*.py exposing MODELS."""
    found, errors = {}, {}
    for path in sorted(glob.glob(os.path.join(HERE, "models_*.py"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            mod = importlib.import_module(f"research.{stem}")
            for k, m in getattr(mod, "MODELS", {}).items():
                if k in found:
                    raise ValueError(f"model name {k!r} defined twice")
                found[k] = m
        except Exception as e:  # one lane's broken file must not stop the others
            errors[stem] = f"{type(e).__name__}: {e}"
    if wanted:
        missing = [w for w in wanted if w not in found]
        if missing:
            for s, e in errors.items():
                print(f"  import error in {s}.py: {e}", file=sys.stderr)
            raise SystemExit(f"unknown model(s): {', '.join(missing)}; known: {', '.join(sorted(found))}")
    return found


def code_key(model):
    """Hash of everything the model's predictions depend on (for cache invalidation)."""
    h = hashlib.md5(GRID_VERSION.encode())
    files = [inspect.getsourcefile(type(model)), engine.__file__, os.path.join(HERE, "splits.py")]
    files += [os.path.join(ROOT, f) if not os.path.isabs(f) else f for f in getattr(model, "cache_deps", ())]
    for f in files:
        with open(f, "rb") as fh:
            h.update(fh.read())
    h.update(repr(getattr(model, "params", "")).encode())
    return h.hexdigest()[:10]


def _cache_path(model, split, step, t):
    return os.path.join(PARTIAL, model.name, code_key(model), f"{split}_s{step}", f"{t}.npz")


def _save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def predictions(ctx, model, t, split, step, horizons, fresh=False):
    """{h: (Q, P)} for the grid of (t, split, step), cached on disk. Returns
    (positions, preds, seconds, from_cache)."""
    td = ctx.td(t)
    pos = grid_positions(td, split, step, HORIZONS)       # grid over all horizons: cache reuse
    path = _cache_path(model, split, step, t)
    srcs = [path]
    if step != STEP_FULL and step % STEP_FULL == 0:
        srcs.append(_cache_path(model, split, STEP_FULL, t))   # --fast can reuse a full run
    if not fresh:
        for src in srcs:
            if not os.path.exists(src):
                continue
            z = np.load(src)
            cpos = z["pos"]
            have = [h for h in horizons if f"Q{h}" in z.files]
            if len(have) == len(horizons) and np.isin(pos, cpos).all():
                idx = np.searchsorted(cpos, pos)
                out = {h: (z[f"Q{h}"][idx], z[f"P{h}"][idx]) for h in horizons}
                return pos, out, float(z["sec"]) * len(pos) / max(len(cpos), 1), True
    t0 = time.time()
    hz = tuple(sorted(set(horizons) | set(HORIZONS)))       # always cache all three horizons
    out = model.predict(ctx, t, pos, hz) if len(pos) else {h: (np.zeros((0, 19)), np.zeros(0)) for h in hz}
    sec = time.time() - t0
    for h in hz:
        Q, P = out[h]
        Q = np.asarray(Q, float).reshape(len(pos), 19)
        P = np.asarray(P, float).reshape(len(pos))
        out[h] = (Q, P)
    _save_npz(path, pos=pos, sec=np.float64(sec), **{f"Q{h}": out[h][0] for h in hz},
              **{f"P{h}": out[h][1] for h in hz})
    return pos, {h: out[h] for h in horizons}, sec, False


# ---------------------------------------------------------------------------
# Scoring (identical for every model)
# ---------------------------------------------------------------------------

def outcome_rows(td, pos, h):
    """Per as-of row: realised r, naive sigma (sig63*sqrt h), walk-forward up base rate,
    daily sigma, and a mask of rows whose outcome is known."""
    n = len(td.day)
    ok = pos + h < n
    p = pos[ok]
    r = td.logc[p + h] - td.logc[p]
    sn = td.sig[p] * math.sqrt(h)
    # base rate: share of h-bar up-moves among moves that END at or before the as-of bar
    up = (td.close[h:] > td.close[:-h]).astype(np.float64)    # up[i] = move ending at bar i+h
    cum = np.concatenate([[0.0], np.cumsum(up)])
    m = p - h + 1                                              # moves ending at bars h..p
    base = cum[np.maximum(m, 0)] / np.maximum(m, 1)
    return ok, r, sn, base, td.sig[p]


def pinball_rows(Q, r, sn):
    """Mean over the 19 quantiles of pinball loss, divided by the naive sigma."""
    u = r[:, None] - Q
    loss = np.maximum(QS19 * u, (QS19 - 1.0) * u)
    return loss.mean(axis=1) / sn


def metrics(R):
    """R: dict of concatenated arrays (r, sn, base, Q, P). Returns the PLAN.md metrics."""
    r, sn, base, Q, P = R["r"], R["sn"], R["base"], R["Q"], R["P"]
    n_all = len(r)
    good = np.isfinite(Q).all(axis=1) & np.isfinite(P)
    r, sn, base, Q, P = r[good], sn[good], base[good], Q[good], P[good]
    n = len(r)
    if n == 0:
        return {"n": 0, "n_abstain": int(n_all)}
    up = (r > 0).astype(float)
    pm = pinball_rows(Q, r, sn)
    pn = pinball_rows(Z19[None, :] * sn[:, None], r, sn)        # naive0 on the same rows
    c50 = float(np.mean((r >= Q[:, I25]) & (r <= Q[:, I75])))
    c80 = float(np.mean((r >= Q[:, I10]) & (r <= Q[:, I90])))
    brier = float(np.mean((P - up) ** 2))
    brier_b = float(np.mean((base - up) ** 2))
    maj = base >= 0.5                                            # walk-forward majority guess
    med = Q[:, I50]
    pred = np.where(med > 0, True, np.where(med < 0, False, maj))   # median exactly 0 -> majority
    return {
        "n": int(n), "n_abstain": int(n_all - n),
        "n_crossing": int(np.sum(np.any(np.diff(Q, axis=1) < 0, axis=1))),
        "pinball": float(pm.mean()), "pinball_naive0": float(pn.mean()),
        "skill": float(1 - pm.mean() / pn.mean()),
        "cov50": c50, "cov50_err": abs(c50 - 0.5), "cov80": c80, "cov80_err": abs(c80 - 0.8),
        "width80": float(np.mean(Q[:, I90] - Q[:, I10])),
        "brier": brier, "brier_base": brier_b, "brier_skill": float(1 - brier / brier_b),
        "hit": float(np.mean(pred == (up > 0))), "hit_majority": float(np.mean(maj == (up > 0))),
        "up_rate": float(up.mean()),
    }


def _cat(rows, key):
    return np.concatenate([x[key] for x in rows]) if rows else np.zeros(0)


def skill_ci(rows, n_boot=2000, seed=0):
    """90% interval for the pinball skill vs naive0, resampling whole TICKERS with
    replacement (rows within a ticker overlap in time and are not independent)."""
    pm, pn = [], []
    for x in rows:
        g = np.isfinite(x["Q"]).all(axis=1) & np.isfinite(x["P"])
        if g.sum() == 0:
            continue
        pm.append(pinball_rows(x["Q"][g], x["r"][g], x["sn"][g]).sum())
        pn.append(pinball_rows(Z19[None, :] * x["sn"][g][:, None], x["r"][g], x["sn"][g]).sum())
    if len(pm) < 5:
        return None
    pm, pn = np.array(pm), np.array(pn)
    idx = np.random.default_rng(seed).integers(0, len(pm), (n_boot, len(pm)))
    sk = 1 - pm[idx].sum(axis=1) / pn[idx].sum(axis=1)
    return [float(np.quantile(sk, 0.05)), float(np.quantile(sk, 0.95))]


def summarise(rows):
    """rows: list of per-ticker dicts (r, sn, base, sig, Q, P). Overall + vol-tercile metrics."""
    R = {k: _cat(rows, k) for k in ("r", "sn", "base", "sig", "P")}
    R["Q"] = np.concatenate([x["Q"] for x in rows]) if rows else np.zeros((0, 19))
    out = {"all": metrics(R)}
    ci = skill_ci(rows)
    if ci is not None:
        out["all"]["skill_ci90"] = ci
    if len(R["sig"]) >= 3:
        t1, t2 = np.quantile(R["sig"], [1 / 3, 2 / 3])
        b = np.digitize(R["sig"], [t1, t2])
        out["vol_terciles_daily_sigma"] = [float(t1), float(t2)]
        out["vol"] = {name: metrics({k: v[b == i] for k, v in R.items()})
                      for i, name in enumerate(("low", "mid", "high"))}
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def eval_tickers(split, fast, only=None):
    ts = splits.tickers_for(split)
    if only:
        ts = [t for t in ts if t in only]
    elif fast:
        ts = ts[::FAST_EVERY]
    return ts


def run(model_names, split, fast=False, horizons=HORIZONS, fresh=False, workers=1, only=None,
        ctx=None, quiet=False):
    models = discover_models(model_names)
    models = [models[m] for m in model_names]
    step = STEP_FAST if fast else STEP_FULL
    ctx = ctx or Context()
    tickers = eval_tickers(split, fast, only)
    rows = {m.name: {h: [] for h in horizons} for m in models}
    per_t = {m.name: {} for m in models}
    secs = {m.name: 0.0 for m in models}
    cached = {m.name: 0 for m in models}
    skipped = {}
    t_start = time.time()

    def one(t):
        try:
            td = ctx.td(t)
        except engine.ForecastError as e:
            return t, None, e.msg
        res = {}
        try:
            for m in models:
                pos, preds, sec, hit = predictions(ctx, m, t, split, step, horizons, fresh)
                res[m.name] = (pos, preds, sec, hit)
        finally:
            ctx.release(t)
        return t, (td, res), None

    def take(item, k, total):
        t, payload, err = item
        if payload is None:
            skipped[t] = err
            return
        td, res = payload
        for name, (pos, preds, sec, hit) in res.items():
            secs[name] += sec
            cached[name] += int(hit)
            per_t[name][t] = {}
            for h in horizons:
                ok, r, sn, base, sig = outcome_rows(td, pos, h)
                Q, P = preds[h]
                row = {"r": r, "sn": sn, "base": base, "sig": sig, "Q": Q[ok], "P": P[ok]}
                rows[name][h].append(row)
                mt = metrics(row) if len(r) else {"n": 0}
                per_t[name][t][h] = {"n": mt["n"], "skill": round(mt.get("skill", float("nan")), 4)}
        if not quiet:
            n5 = len(res[models[0].name][0])
            print(f"  [{k}/{total}] {t:6s} {n5:5d} as-of dates  "
                  + "  ".join(f"{nm} {s:.1f}s{' (cache)' if h else ''}" for nm, (_, _, s, h) in res.items()),
                  flush=True)

    total = len(tickers)
    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            for k, item in enumerate(ex.map(one, tickers), 1):
                take(item, k, total)
    else:
        for k, t in enumerate(tickers, 1):
            take(one(t), k, total)
    wall = time.time() - t_start

    os.makedirs(RESULTS, exist_ok=True)
    report = {}
    for m in models:
        res = {"model": m.name, "split": split, "fast": fast, "step": step, "horizons": list(horizons),
               "code_key": code_key(m),
               "n_tickers": sum(1 for v in per_t[m.name].values() if any(x["n"] for x in v.values())),
               "skipped": skipped,
               "compute_seconds": round(secs[m.name], 1), "tickers_from_cache": cached[m.name],
               "wall_seconds_this_run": round(wall, 1),
               "results": {str(h): summarise(rows[m.name][h]) for h in horizons},
               "per_ticker": per_t[m.name]}
        path = os.path.join(RESULTS, f"{m.name}_{split}{'_fast' if fast else ''}.json")
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        os.replace(path + ".tmp", path)
        report[m.name] = res
    return report


def print_table(report, file=sys.stdout):
    hdr = (f"{'model':14s} {'h':>3s} {'n':>7s} {'skill':>7s} {'cov50':>6s} {'cov80':>6s} {'|e80|':>6s} "
           f"{'w80':>6s} {'brierSk':>8s} {'hit':>6s} {'maj':>6s} {'abst':>5s}")
    for name, res in report.items():
        print(f"\n== {name}  split={res['split']}{' FAST' if res['fast'] else ''}  tickers={res['n_tickers']}  "
              f"compute={res['compute_seconds']:.0f}s", file=file)
        print(hdr, file=file)
        for h, s in res["results"].items():
            a = s["all"]
            if not a.get("n"):
                print(f"{name:14s} {h:>3s}  no rows", file=file)
                continue
            print(f"{name:14s} {h:>3s} {a['n']:7d} {a['skill']:+7.4f} {a['cov50']:6.3f} {a['cov80']:6.3f} "
                  f"{a['cov80_err']:6.3f} {a['width80']:6.3f} {a['brier_skill']:+8.4f} {a['hit']:6.3f} "
                  f"{a['hit_majority']:6.3f} {a['n_abstain']:5d}"
                  + ("  skill90%CI[{:+.4f},{:+.4f}]".format(*a["skill_ci90"]) if "skill_ci90" in a else ""),
                  file=file)
        for h, s in res["results"].items():
            if "vol" in s:
                v = s["vol"]
                print(f"   h={h:>2s} vol low/mid/high: skill " + "/".join(f"{v[b].get('skill', float('nan')):+.3f}" for b in v)
                      + "  cov80 " + "/".join(f"{v[b].get('cov80', float('nan')):.3f}" for b in v)
                      + "  brierSk " + "/".join(f"{v[b].get('brier_skill', float('nan')):+.3f}" for b in v), file=file)


# ---------------------------------------------------------------------------
# Proofs that the harness (and a model) is honest
# ---------------------------------------------------------------------------

# TRAIN tickers, DEV/VAL dates only (the proofs never touch the TEST split)
LOOKAHEAD_CASES = [("KO", "2008-10-10"), ("AAPL", "2015-06-12"), ("NVDA", "2011-03-04"),
                   ("SMCI", "2018-02-02"), ("IBM", "1987-10-30")]


def check_lookahead(model_names, cases=LOOKAHEAD_CASES, full=None):
    """(a) Predictions at (ticker, as-of) must be identical when every series is
    truncated at the as-of date. Exact equality, NaN == NaN."""
    models = discover_models(model_names)
    full = full or Context()
    ok_all = True
    for t, asof in cases:
        td = full.td(t)
        p = td.pos_on_or_before(day_of(asof))
        D = td.dates[p]
        raw = {x: load_cached(x) for x in set(engine.POOL + [engine.MARKET_SPLICE, t])}
        trunc = Context(raw={x: v for x, v in raw.items() if v is not None}, cutoff=D)
        tt = trunc.td(t)
        assert len(tt.day) == p + 1
        for name in model_names:
            m = models[name]
            a = m.predict(full, t, np.array([p]), HORIZONS)
            b = m.predict(trunc, t, np.array([p]), HORIZONS)
            diffs = []
            for h in HORIZONS:
                same = (np.array_equal(a[h][0], b[h][0], equal_nan=True)
                        and np.array_equal(a[h][1], b[h][1], equal_nan=True))
                if not same:
                    diffs.append(f"h{h} maxdiff={np.nanmax(np.abs(a[h][0] - b[h][0])):.3g}")
            ok_all &= not diffs
            q = a[21][0][0]
            print(f"{'OK  ' if not diffs else 'FAIL'} lookahead {name:12s} {t:5s} asof={str(D.date())} "
                  f"q05/q50/q95(h21)={q[0]:+.4f}/{q[I50]:+.4f}/{q[18]:+.4f} {' '.join(diffs)}")
        full.release(t)
    return ok_all


class _TrueSigma:
    """Synthetic-test model: zero-drift Gaussian with the KNOWN daily sigma."""
    name = "naive_true"

    def __init__(self, sigma):
        self.sigma = sigma

    def predict(self, ctx, t, positions, horizons):
        n = len(positions)
        return {h: (np.tile(Z19 * self.sigma * math.sqrt(h), (n, 1)), np.full(n, 0.5)) for h in horizons}


def check_synthetic(n_series=40, n_days=6000, sigma=0.02, seed=7):
    """(b) Gaussian random walks scored through the real scoring path: the true-sigma
    model must cover ~0.50 / ~0.80; naive0 (trailing 63-day estimate) slightly less."""
    from research.models_base import Naive0
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1980-01-01", periods=n_days)

    def walk():
        lr = rng.normal(0.0, sigma, n_days)
        return pd.DataFrame({"close": 100 * np.exp(np.cumsum(lr)), "volume": 1e6}, index=idx)

    raw = {"SPY": walk()}
    names = [f"SYN{i}" for i in range(n_series)]
    for nm in names:
        raw[nm] = walk()
    ctx = Context(raw=raw)
    ok = True
    for model in (_TrueSigma(sigma), Naive0()):
        for h in HORIZONS:
            rows = []
            for nm in names:
                td = ctx.td(nm)
                pos = np.arange(WARMUP, len(td.day) - h, STEP_FULL)
                pos = pos[td.pool_ok[pos]]
                Q, P = model.predict(ctx, nm, pos, (h,))[h]
                okm, r, sn, base, sig = outcome_rows(td, pos, h)
                rows.append({"r": r, "sn": sn, "base": base, "sig": sig, "Q": Q[okm], "P": P[okm]})
            a = summarise(rows)["all"]
            tol = 0.02 if model.name == "naive_true" else 0.04
            good = abs(a["cov50"] - 0.5) < tol and abs(a["cov80"] - 0.8) < tol
            ok &= good
            print(f"{'OK  ' if good else 'FAIL'} synthetic {model.name:10s} h={h:2d} n={a['n']:6d} "
                  f"cov50={a['cov50']:.3f} cov80={a['cov80']:.3f} skill_vs_naive0={a['skill']:+.4f} "
                  f"brierSk={a['brier_skill']:+.4f} hit={a['hit']:.3f} maj={a['hit_majority']:.3f}")
    return ok


V0_CASES = [("KO", "2008-10-10"), ("AAPL", "2015-06-12"), ("MSFT", "2003-01-17"),
            ("NEM", "2016-09-30"), ("SMCI", "2017-05-05")]


PRODUCT_MODEL = "cone2_f_conf_dev"   # what engine.py serves since 099d52f (v2 cone)


def check_v0_matches_engine(cases=V0_CASES, full=None):
    """(c) The served model's q10/q25/q50/q75/q90 must equal the live engine's own
    forecast (_forecast_at: rounded to 6 significant digits) and its walk_forward
    record (exact). Until 099d52f the engine served v0, so this compared v0; it now
    compares PRODUCT_MODEL. v0's fast-vs-slow search check went with it, because the
    slow path was the engine's own walk_forward."""
    models = discover_models([PRODUCT_MODEL])
    v0 = models[PRODUCT_MODEL]
    full = full or Context()
    ok = True
    for t, asof in cases:
        td = full.td(t)
        p = td.pos_on_or_before(day_of(asof))
        a = v0.predict(full, t, np.array([p]), HORIZONS)
        for h in HORIZONS:
            Q, P = a[h][0][0], a[h][1][0]
            f = full.eng._forecast_at(td, p, h)
            last = f["last_close"]
            eng_q = np.array([math.log(f["future"][k][-1] / last) for k in ("p10", "p25", "p50", "p75", "p90")])
            mine = Q[[I10, I25, I50, I75, I90]]
            d_live = float(np.max(np.abs(eng_q - mine)))
            wf = full.eng.walk_forward(t, [h], [p])[h][0]
            d_wf = float(np.max(np.abs(np.array(wf["aq"]) - mine)))
            good = d_live < 3e-5 and d_wf == 0.0 and wf["p_up"] == P and abs(f["odds"]["p_up"] - P) < 6e-5
            ok &= good
            print(f"{'OK  ' if good else 'FAIL'} {PRODUCT_MODEL}==engine {t:5s} asof={f['asof']} h={h:2d} "
                  f"max|live-model|={d_live:.2e} (live is sig6-rounded)  max|walk_forward-model|={d_wf:.1e}  "
                  f"p_up model={P:.4f} live={f['odds']['p_up']}")
        full.release(t)
    return ok


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default="naive0,naive_drift,v0")
    ap.add_argument("--split", choices=splits.SPLITS, default="val")
    ap.add_argument("--fast", action="store_true", help=f"every {STEP_FAST} bars, every {FAST_EVERY}rd ticker")
    ap.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    ap.add_argument("--tickers", default="", help="comma list: restrict to these (still split-filtered)")
    ap.add_argument("--workers", type=int, default=1, help="threads over tickers (keep <= 2)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached predictions")
    ap.add_argument("--coordinator", action="store_true", help="required to open the TEST split")
    ap.add_argument("--lookahead", action="store_true", help="run the no-look-ahead proof on --models")
    ap.add_argument("--selftest", action="store_true", help="run all three harness proofs")
    ap.add_argument("--list", action="store_true", help="list discoverable models")
    a = ap.parse_args(argv)
    names = [x for x in a.models.split(",") if x]

    if a.list:
        for k, m in sorted(discover_models().items()):
            print(f"{k:16s} {type(m).__module__}.{type(m).__name__}")
        return 0
    if a.selftest or a.lookahead:
        full = Context()
        ok = check_lookahead(names if a.lookahead else ["naive0", "naive_drift", "v0"], full=full)
        if a.selftest:
            ok &= check_synthetic()
            ok &= check_v0_matches_engine(full=full)
        print("HARNESS PROOFS", "PASSED" if ok else "FAILED")
        return 0 if ok else 1
    if a.split == "test" and not a.coordinator:
        print("REFUSED: the TEST split is opened only by the coordinator (pass --coordinator). "
              "Use --split dev or --split val.", file=sys.stderr)
        return 2
    hz = tuple(int(x) for x in a.horizons.split(","))
    bad = [h for h in hz if h not in HORIZONS]
    if bad:
        raise SystemExit(f"horizons must be among {HORIZONS}")
    only = set(a.tickers.split(",")) - {""} or None
    t0 = time.time()
    report = run(names, a.split, a.fast, hz, a.fresh, a.workers, only)
    print_table(report)
    print(f"\nwall {time.time() - t0:.0f}s; JSON in {os.path.relpath(RESULTS, ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
