"""Cross-sectional direction edge: which stocks beat the others next week/month?

Lane cone-xsec (research/PLAN.md, TOP PRIORITY). Nothing here is imported by the app.

  python research/xsec.py                         # DEV + VAL, TRAIN tickers, data cut at 2018-12-31
  python research/xsec.py --split test --coordinator   # TEST (all tickers, 2019+), frozen choices only
  python research/xsec.py --split dryrun          # rehearse the TEST code path on VAL (no TEST data)

DEV/VAL runs never load a price after 2018-12-31 and never load a TEST ticker.
Everything chosen (combination members and signs, ridge/logistic weights, the VAL
candidates) is written to research/results/xsec_frozen.json; the TEST run only
reads that file, it never refits.

Every signal is oriented so that "higher = expected to beat the others" under
the usual prior from the literature; DEV decides whether that prior holds.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research import backtest as bt  # noqa: E402
from research import splits  # noqa: E402

RES = os.path.join(ROOT, "research", "results")
FROZEN = os.path.join(RES, "xsec_frozen.json")
FREQS = {"weekly": 5, "monthly": 21}
N_RANDOM = 20
T_REAL = 2.0


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------------------
# Signals (all causal: the value at row t uses rows <= t only)
# ---------------------------------------------------------------------------

def compute_signals(panel):
    P = panel.pff
    lp = np.log(P)
    lr = lp.diff()
    mlp = np.log(panel.mkt)
    mlr = mlp.diff()

    def ret(a, b):          # log return from t-b to t-a
        return lp.shift(a) - lp.shift(b)

    def mret(a, b):
        return mlp.shift(a) - mlp.shift(b)

    Mr = pd.DataFrame(np.repeat(mlr.to_numpy()[:, None], lr.shape[1], 1),
                      index=lr.index, columns=lr.columns).where(lr.notna())
    X = lr.where(Mr.notna())

    def beta_var(W, mp):
        mx, mm = X.rolling(W, min_periods=mp).mean(), Mr.rolling(W, min_periods=mp).mean()
        cov = (X * Mr).rolling(W, min_periods=mp).mean() - mx * mm
        vm = (Mr * Mr).rolling(W, min_periods=mp).mean() - mm * mm
        vx = (X * X).rolling(W, min_periods=mp).mean() - mx * mx
        return cov / vm, vx, vm

    beta, _, _ = beta_var(252, 200)
    b63, vx63, vm63 = beta_var(63, 50)
    vol63 = lr.rolling(63, min_periods=50).std()
    v = panel.volume.where(panel.volume > 0)
    vshock = np.log(v.rolling(5, min_periods=4).mean() / v.rolling(63, min_periods=50).mean())
    ma50 = P.rolling(50, min_periods=45).mean()
    ma200 = P.rolling(200, min_periods=180).mean()

    s = {}
    s["mom_12_1"] = ret(21, 252)
    s["mom_6_1"] = ret(21, 126)
    s["mom_3_1"] = ret(21, 63)
    s["rev_1w"] = -ret(0, 5)
    s["rev_1m"] = -ret(0, 21)
    s["lowvol_63"] = -vol63
    s["high52"] = P / P.rolling(252, min_periods=240).max() - 1
    s["trend_50"] = np.log(P / ma50)
    s["trend_200"] = np.log(P / ma200)
    s["ma_cross"] = np.log(ma50 / ma200)
    s["volshock"] = vshock
    s["volshock_signed"] = np.sign(ret(0, 5)) * vshock.clip(lower=0)
    s["rmom_12_1"] = ret(21, 252) / (lr.shift(21).rolling(231, min_periods=200).std() * np.sqrt(231))
    s["lowbeta"] = -beta
    s["resid_mom_12_1"] = ret(21, 252) - beta.mul(mret(21, 252), axis=0)
    s["resid_mom_6_1"] = ret(21, 126) - beta.mul(mret(21, 126), axis=0)
    s["rev_1w_bn"] = -(ret(0, 5) - beta.mul(mret(0, 5), axis=0))
    s["rev_1m_bn"] = -(ret(0, 21) - beta.mul(mret(0, 21), axis=0))
    s["lowivol_63"] = -np.sqrt((vx63 - b63 ** 2 * vm63).clip(lower=0))
    s["max_21"] = -lr.rolling(21, min_periods=18).max()
    for k, v in s.items():   # guard against Series/DataFrame mis-alignment (a real bug, once)
        assert v.shape == P.shape and (v.index == P.index).all() and list(v.columns) == list(P.columns), k
    return s


BASE = ["mom_12_1", "mom_6_1", "mom_3_1", "rev_1w", "rev_1m", "lowvol_63", "high52",
        "trend_50", "trend_200", "ma_cross", "volshock", "volshock_signed", "rmom_12_1",
        "lowbeta", "resid_mom_12_1", "resid_mom_6_1", "rev_1w_bn", "rev_1m_bn",
        "lowivol_63", "max_21"]


def rank_score(sig, elig):
    """Cross-sectional uniform score in (-0.5, 0.5) among eligible names, per day."""
    x = sig.where(elig)
    n = x.notna().sum(axis=1)
    return x.rank(axis=1).sub(0.5).div(n, axis=0) - 0.5


def combo_signal(scores, members):
    """members: {name: +1/-1}. Mean of oriented rank scores."""
    acc, cnt = None, None
    for name, sg in members.items():
        z = scores[name] * sg
        acc = z.fillna(0) if acc is None else acc + z.fillna(0)
        c = z.notna().astype(float)
        cnt = c if cnt is None else cnt + c
    return (acc / cnt.where(cnt > 0))


def linear_signal(scores, weights, elig):
    acc = None
    for name, w in weights.items():
        z = scores[name].fillna(0) * w
        acc = z if acc is None else acc + z
    return acc.where(elig)


# ---------------------------------------------------------------------------
# Models fit on DEV only (numpy ridge and L2 logistic)
# ---------------------------------------------------------------------------

def design(panel, scores, elig, sidx, lag, names):
    """Pooled (date, name) rows: X = rank scores at signal day, y = rank score of the
    period's return, ybin = beat the median. Returns X, y, ybin, r, group ids."""
    P = panel.pff.to_numpy(float)
    E = elig.to_numpy(bool)
    F = np.stack([scores[n].to_numpy(float) for n in names], axis=2)
    Xs, ys, yb, rs, gs = [], [], [], [], []
    for k in range(len(sidx) - 1):
        s, e0, e1 = sidx[k], sidx[k] + lag, sidx[k + 1] + lag
        with np.errstate(invalid="ignore", divide="ignore"):
            r = P[e1] / P[e0] - 1
        valid = E[s] & np.isfinite(r)
        n = int(valid.sum())
        if n < bt.MIN_NAMES:
            continue
        rr = r[valid]
        rk = pd.Series(rr).rank().to_numpy()
        Xs.append(np.nan_to_num(F[s][valid], nan=0.0))
        ys.append((rk - 0.5) / n - 0.5)
        yb.append((rr > np.median(rr)).astype(float))
        rs.append(rr)
        gs.append(np.full(n, k))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(yb),
            np.concatenate(rs), np.concatenate(gs))


def ridge_fit(X, y, lam):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def logit_fit(X, y, lam, iters=30):
    Xa = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(Xa.shape[1])
    pen = lam * np.eye(Xa.shape[1])
    pen[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Xa @ b)))
        g = Xa.T @ (p - y) + pen @ b
        H = (Xa * (p * (1 - p))[:, None]).T @ Xa + pen
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-8:
            break
    return b[1:]


def mean_group_ic(pred, r, g):
    ics = [bt.spearman(pred[g == k], r[g == k]) for k in np.unique(g)]
    return float(np.nanmean(ics))


def fit_models(X, y, yb, r, g, names):
    """Choose lambda by 5 contiguous date folds inside DEV (score = mean held-out
    rank IC), then refit on all of DEV. Returns a dict for the frozen file."""
    groups = np.unique(g)
    folds = np.array_split(groups, 5)
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    out = {}
    for kind in ("ridge", "logit"):
        cv = {}
        for a in alphas:
            lam = a * len(X)
            ics = []
            for f in folds:
                lo, hi = f.min() - 1, f.max() + 1       # purge one period each side
                tr = (g < lo) | (g > hi)
                te = np.isin(g, f)
                b = ridge_fit(X[tr], y[tr], lam) if kind == "ridge" else logit_fit(X[tr], yb[tr], lam / 12)
                ics.append(mean_group_ic(X[te] @ b, r[te], g[te]))
            cv[str(a)] = float(np.mean(ics))
        best = max(cv, key=cv.get)
        lam = float(best) * len(X)
        b = ridge_fit(X, y, lam) if kind == "ridge" else logit_fit(X, yb, lam / 12)
        out[kind] = {"alpha": float(best), "cv_ic": cv, "weights": dict(zip(names, map(float, b)))}
    return out


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(panel, sig, elig, split, step, lag, daily=True, offset=0):
    sidx = bt.split_window(panel, split, elig, step, lag, offset=offset)
    res = bt.run(panel, sig, elig, sidx, step, lag)
    row = bt.summarize(res, step)
    if daily and len(sidx) > 2:
        row["ic_daily"], row["ic_daily_t"] = bt.ic_daily(panel, sig, elig, step, lag, sidx[0],
                                                         bt.split_last_index(panel, split))
    return row, res


def fmt_table(df, cols, pct=(), dec=None):
    dec = dec or {}
    head = "| signal | " + " | ".join(cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    lines = [head, sep]
    for name, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, np.nan)
            if not np.isfinite(v):
                cells.append("")
            elif c in pct:
                cells.append(f"{100 * v:+.1f}%" if c in ("excess_ann", "ls_ann") else f"{100 * v:.1f}%")
            else:
                cells.append(f"{v:.{dec.get(c, 2)}f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


TABLE_COLS = ["ic_mean", "ic_t", "ic_daily_t", "top_ann", "excess_ann", "excess_t", "top_sharpe",
              "top_maxdd", "turnover_1way_ann", "ls_ann", "ls_t", "hit_month_vs_ew"]
PCT = ("top_ann", "excess_ann", "top_maxdd", "ls_ann", "hit_month_vs_ew", "turnover_1way_ann")
DEC = {"ic_mean": 3, "ic_t": 2, "ic_daily_t": 2, "excess_t": 2, "ls_t": 2, "top_sharpe": 2}


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# DEV + VAL
# ---------------------------------------------------------------------------

def run_devval():
    os.makedirs(RES, exist_ok=True)
    tickers = splits.tickers_for("dev")
    assert all(splits.is_train_ticker(t) for t in tickers)
    panel = bt.Panel(tickers, end=splits.VAL_END)
    assert panel.cal[-1] <= pd.Timestamp(splits.VAL_END), "2019+ data leaked into DEV/VAL"
    log(f"panel {len(panel.tickers)} TRAIN tickers, {panel.cal[0].date()}..{panel.cal[-1].date()}, "
        f"missing {panel.missing}, off-calendar rows {panel.off_calendar}")
    elig = bt.eligibility(panel)
    sigs = compute_signals(panel)
    scores = {n: rank_score(sigs[n], elig) for n in BASE}
    rows, years = [], {}
    meta = {"tickers": panel.tickers, "elig_by_year": elig.sum(axis=1).groupby(panel.cal.year).mean()
            .loc[1980:].round(1).to_dict()}

    # 1. every base signal, both frequencies, both splits, both trade lags
    for fname, step in FREQS.items():
        for split in ("dev", "val"):
            for lag in (1, 0):
                for name in BASE:
                    row, res = evaluate(panel, sigs[name], elig, split, step, lag, daily=(lag == 1))
                    rows.append(dict(signal=name, freq=fname, split=split, lag=lag, **row))
                # reference portfolios from the last run (EW and SPY are signal-free)
                rows.append(dict(signal="(EW universe)", freq=fname, split=split, lag=lag,
                                 top_ann=row["ew_ann"], top_sharpe=row["ew_sharpe"], top_maxdd=row["ew_maxdd"],
                                 start=row["start"], end=row["end"], periods=row["periods"],
                                 names_avg=row["names_avg"]))
                rows.append(dict(signal="(SPY buy-hold)", freq=fname, split=split, lag=lag,
                                 top_ann=row["spy_ann"], top_sharpe=row["spy_sharpe"]))
            log(f"base signals done: {fname} {split}")
            pd.DataFrame(rows).to_csv(os.path.join(RES, "xsec_base.csv"), index=False)
    base = pd.DataFrame(rows)

    # 2. sanity checks
    sanity = {}
    for fname, step in FREQS.items():
        for split in ("dev", "val"):
            rr = []
            for seed in range(N_RANDOM):
                row, _ = evaluate(panel, bt.random_signal(panel, seed), elig, split, step, 1, daily=False)
                rr.append(row)
            rr = pd.DataFrame(rr)
            la, _ = evaluate(panel, bt.forward_return_signal(panel, step, 1), elig, split, step, 1, daily=False)
            sanity[f"{fname}_{split}"] = {
                "random_ic_mean": float(rr.ic_mean.mean()), "random_ic_sd": float(rr.ic_mean.std()),
                "random_ic_t_mean": float(rr.ic_t.mean()), "random_ic_t_absmax": float(rr.ic_t.abs().max()),
                "random_excess_gross_mean": float((rr.top_gross_ann - rr.ew_ann).mean()),
                "random_excess_net_mean": float(rr.excess_ann.mean()), "random_excess_net_sd": float(rr.excess_ann.std()),
                "random_turnover": float(rr.turnover_1way_ann.mean()),
                "random_top_ann_mean": float(rr.top_ann.mean()), "ew_ann": float(rr.ew_ann.iloc[0]),
                "lookahead_ic": la["ic_mean"], "lookahead_top_ann": la["top_ann"],
                "lookahead_excess": la["excess_ann"], "lookahead_ls_ann": la["ls_ann"],
            }
            log(f"sanity random/lookahead {fname} {split}: {sanity[f'{fname}_{split}']}")
            save_json(os.path.join(RES, "xsec_sanity.json"), sanity)
    # shift test: signal one day staler (uses closes <= t-1), still trade at t+1
    shift_rows = []
    for fname, step in FREQS.items():
        for split in ("dev", "val"):
            for name in BASE:
                row, _ = evaluate(panel, sigs[name].shift(1), elig, split, step, 1, daily=False)
                shift_rows.append(dict(signal=name, freq=fname, split=split, ic_mean=row["ic_mean"],
                                       ic_t=row["ic_t"], excess_ann=row["excess_ann"], ls_ann=row["ls_ann"]))
    pd.DataFrame(shift_rows).to_csv(os.path.join(RES, "xsec_shift.csv"), index=False)
    log("shift test done")

    # 3. DEV-chosen combination and DEV-fit models (frozen before VAL is read)
    frozen = {"made": time.strftime("%Y-%m-%d %H:%M"), "freqs": {}}
    for fname, step in FREQS.items():
        d = base[(base.freq == fname) & (base.split == "dev") & (base.lag == 1) & base.signal.isin(BASE)]
        members = {r.signal: int(np.sign(r.ic_mean)) for r in d.itertuples() if abs(r.ic_t) >= T_REAL}
        sidx = bt.split_window(panel, "dev", elig, step, 1)
        X, y, yb, r, g = design(panel, scores, elig, sidx, 1, BASE)
        models = fit_models(X, y, yb, r, g, BASE)
        frozen["freqs"][fname] = {"combo_members": members, "dev_rows": int(len(X)), **models}
        log(f"{fname}: combo members {members}; ridge alpha {models['ridge']['alpha']} "
            f"logit alpha {models['logit']['alpha']}")
    save_json(FROZEN, frozen)

    combo_rows = []
    derived = {}
    for fname, step in FREQS.items():
        fz = frozen["freqs"][fname]
        cands = {}
        if fz["combo_members"]:
            cands["combo_dev"] = combo_signal(scores, fz["combo_members"])
        cands["ridge_dev"] = linear_signal(scores, fz["ridge"]["weights"], elig)
        cands["logit_dev"] = linear_signal(scores, fz["logit"]["weights"], elig)
        cands["combo_all_prior"] = combo_signal(scores, {n: 1 for n in BASE})
        derived[fname] = cands
        for name, sig in cands.items():
            for split in ("dev", "val"):
                for lag in (1, 0):
                    row, res = evaluate(panel, sig, elig, split, step, lag, daily=(lag == 1))
                    combo_rows.append(dict(signal=name, freq=fname, split=split, lag=lag, **row))
        log(f"combos done {fname}")
    allrows = pd.concat([base, pd.DataFrame(combo_rows)], ignore_index=True)
    allrows.to_csv(os.path.join(RES, "xsec_all.csv"), index=False)

    # 4. candidates, chosen on VAL only: sign agrees with DEV, VAL IC t > 2 (NW)
    cand = []
    for fname in FREQS:
        a = allrows[(allrows.freq == fname) & (allrows.lag == 1) & ~allrows.signal.str.startswith("(")]
        dv = a[a.split == "dev"].set_index("signal")
        vl = a[a.split == "val"].set_index("signal")
        for name in vl.index:
            si_d, si_v = np.sign(dv.loc[name, "ic_mean"]), np.sign(vl.loc[name, "ic_mean"])
            tv = vl.loc[name, "ic_t"] * si_v
            if si_d == si_v and tv > T_REAL:
                cand.append(dict(signal=name, freq=fname, orient=int(si_v), val_ic=vl.loc[name, "ic_mean"],
                                 val_ic_t=vl.loc[name, "ic_t"], val_ic_daily_t=vl.loc[name, "ic_daily_t"],
                                 dev_ic=dv.loc[name, "ic_mean"], dev_ic_t=dv.loc[name, "ic_t"],
                                 val_excess=vl.loc[name, "excess_ann"], val_excess_t=vl.loc[name, "excess_t"],
                                 val_ls=vl.loc[name, "ls_ann"]))
    cand = sorted(cand, key=lambda c: -abs(c["val_ic_t"]))
    frozen["candidates_val"] = cand
    # The few the coordinator should judge on TEST (fixed rule, VAL numbers only):
    # best VAL IC t overall, best single base signal by VAL IC t, best VAL net excess t.
    prim = []
    if cand:
        picks = [cand[0],
                 next((c for c in cand if c["signal"] in BASE), None),
                 max(cand, key=lambda c: c["val_excess_t"] if np.isfinite(c["val_excess_t"]) else -9)]
        for c in picks:
            if c is not None and (c["signal"], c["freq"]) not in [(p["signal"], p["freq"]) for p in prim]:
                prim.append({k: c[k] for k in ("signal", "freq", "orient", "val_ic_t", "val_excess_t")})
    frozen["primary_for_test"] = prim
    save_json(FROZEN, frozen)
    log(f"VAL candidates: {[(c['signal'], c['freq'], round(c['val_ic_t'], 2)) for c in cand]}")

    orient_of = {(c["signal"], c["freq"]): c["orient"] for c in cand}

    def get_sig(name, fname):
        sig = derived[fname].get(name) if name in derived[fname] else sigs.get(name)
        return sig * orient_of.get((name, fname), 1)

    # 4b. robustness of the VAL candidates (VAL only, nothing re-chosen):
    # rebalance-day anchor, the names held most often removed, tercile instead of quintile
    rob = []
    for c in cand[:6]:
        name, fname = c["signal"], c["freq"]
        step, sig = FREQS[fname], get_sig(name, fname)
        offs = range(5) if step == 5 else (0, 5, 10, 15)
        for off in offs:
            row, res = evaluate(panel, sig, elig, "val", step, 1, daily=False, offset=off)
            rob.append(dict(signal=name, freq=fname, test=f"anchor+{off}", ic_mean=row["ic_mean"],
                            ic_t=row["ic_t"], excess_ann=row["excess_ann"], excess_t=row["excess_t"],
                            ls_ann=row["ls_ann"]))
            if off == 0:
                held = sorted(res.attrs["top_counts"].items(), key=lambda kv: -kv[1])
                top3 = [t for t, _ in held[:3]]
                share = sum(n for _, n in held[:3]) / max(1, sum(n for _, n in held))
        el2 = elig.copy()
        el2[top3] = False
        row, _ = evaluate(panel, sig, el2, "val", step, 1, daily=False)
        rob.append(dict(signal=name, freq=fname, test=f"drop {'/'.join(top3)} ({share:.0%} of slots)",
                        ic_mean=row["ic_mean"], ic_t=row["ic_t"], excess_ann=row["excess_ann"],
                        excess_t=row["excess_t"], ls_ann=row["ls_ann"]))
        sidx = bt.split_window(panel, "val", elig, step, 1)
        res = bt.run(panel, sig, elig, sidx, step, 1, quant=1 / 3)
        row = bt.summarize(res, step)
        rob.append(dict(signal=name, freq=fname, test="tercile", ic_mean=row["ic_mean"], ic_t=row["ic_t"],
                        excess_ann=row["excess_ann"], excess_t=row["excess_t"], ls_ann=row["ls_ann"]))
        pd.DataFrame(rob).to_csv(os.path.join(RES, "xsec_robust.csv"), index=False)
    log("robustness done")

    # 4c. diagnostics: how trustworthy is "t > 2" here, and how much is survivorship?
    diag = diagnostics(panel, elig, sigs)
    save_json(os.path.join(RES, "xsec_diag.json"), diag)
    log(f"diagnostics: {diag}")

    # 5. year-by-year for candidates and for the headline signals
    yr = []
    show = {(c["signal"], c["freq"]) for c in cand}
    show |= {(n, f) for f in FREQS for n in ("mom_12_1", "rev_1m", "lowvol_63", "combo_dev", "ridge_dev")}
    for name, fname in sorted(show):
        step = FREQS[fname]
        sig = derived[fname].get(name) if name in derived[fname] else sigs.get(name)
        sig = None if sig is None else sig * orient_of.get((name, fname), 1)
        if sig is None:
            continue
        for split in ("dev", "val"):
            _, res = evaluate(panel, sig, elig, split, step, 1, daily=False)
            y = bt.by_year(res)
            y.insert(0, "split", split)
            y.insert(0, "freq", fname)
            y.insert(0, "signal", name)
            yr.append(y)
    pd.concat(yr).to_csv(os.path.join(RES, "xsec_years.csv"))

    # 6. markdown tables for the report
    md = [f"# xsec tables (generated {time.strftime('%Y-%m-%d %H:%M')})", ""]
    md.append(f"Eligible TRAIN names per day, yearly mean: {meta['elig_by_year']}")
    for fname in FREQS:
        for split in ("dev", "val"):
            t = allrows[(allrows.freq == fname) & (allrows.split == split) & (allrows.lag == 1)].set_index("signal")
            info = t.dropna(subset=["periods"]).iloc[0]
            md += ["", f"## {split.upper()} {fname} (trade next close; {info['start']}..{info['end']}, "
                   f"{int(info['periods'])} periods, {info['names_avg']:.0f} names avg)", ""]
            md.append(fmt_table(t, TABLE_COLS, PCT, DEC))
    lag0 = allrows[allrows.lag == 0].set_index(["signal", "freq", "split"])
    lag1 = allrows[allrows.lag == 1].set_index(["signal", "freq", "split"])
    j = lag1[["ic_mean", "excess_ann"]].join(lag0[["ic_mean", "excess_ann"]], rsuffix="_lag0").dropna()
    md += ["", "## Same-close (lag 0) vs next-close (lag 1)", "", j.round(4).to_markdown()
           if hasattr(j, "to_markdown") and _has_tabulate() else j.round(4).to_string()]
    with open(os.path.join(RES, "xsec_tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    meta["sanity"] = sanity
    save_json(os.path.join(RES, "xsec_meta.json"), meta)
    log("done")


N_PLACEBO = 30


def placebo_persistent(panel, seed, block=252):
    """Uninformative but persistent: one random score per stock, redrawn every `block` days."""
    rng = np.random.default_rng(10_000 + seed)
    b = np.arange(len(panel.cal)) // block
    z = rng.standard_normal((b.max() + 1, len(panel.tickers)))[b]
    return pd.DataFrame(z, index=panel.cal, columns=panel.tickers)


def diagnostics(panel, elig, sigs):
    """(1) False-positive rate of the IC t-stat under uninformative signals, iid and
    persistent. (2) 'Long-run drift' signal (expanding mean daily return): it has no
    reason to work in an unbiased universe, so if it works here that measures how much
    survivorship rewards 'pick what has been going up'. (3) its rank overlap with momentum."""
    out = {}
    for fname, step in FREQS.items():
        for split in ("dev", "val"):
            ti, tp = [], []
            for s in range(N_PLACEBO):
                ti.append(evaluate(panel, bt.random_signal(panel, 500 + s), elig, split, step, 1, daily=False)[0]["ic_t"])
                tp.append(evaluate(panel, placebo_persistent(panel, s), elig, split, step, 1, daily=False)[0]["ic_t"])
            ti, tp = np.array(ti), np.array(tp)
            out[f"placebo_{fname}_{split}"] = {
                "iid_t_sd": float(ti.std()), "iid_share_abs_t_gt2": float((np.abs(ti) > 2).mean()),
                "persistent_t_sd": float(tp.std()), "persistent_share_abs_t_gt2": float((np.abs(tp) > 2).mean()),
                "persistent_t_absmax": float(np.abs(tp).max())}
    lr = np.log(panel.pff).diff()
    drift = lr.expanding(min_periods=252).mean()
    for fname, step in FREQS.items():
        for split in ("dev", "val"):
            row, _ = evaluate(panel, drift, elig, split, step, 1, daily=False)
            out[f"drift_{fname}_{split}"] = {k: row[k] for k in ("ic_mean", "ic_t", "excess_ann", "excess_t")}
    sc_m, sc_d = rank_score(sigs["mom_12_1"], elig), rank_score(drift, elig)
    # momentum with the long-run-drift ranking regressed out, per day
    m = sc_m.notna() & sc_d.notna()
    a, d = sc_m.where(m), sc_d.where(m)
    b = (a * d).sum(axis=1) / (d * d).sum(axis=1)
    mom_x_drift = a - d.mul(b, axis=0)
    for fname, step in FREQS.items():
        for split in ("dev", "val"):
            row, _ = evaluate(panel, mom_x_drift, elig, split, step, 1, daily=False)
            out[f"mom_minus_drift_{fname}_{split}"] = {k: row[k] for k in ("ic_mean", "ic_t", "excess_ann", "excess_t")}
    for split in ("dev", "val"):
        sidx = bt.split_window(panel, split, elig, 21, 1)
        cs = []
        for i in sidx:
            j = pd.concat([sc_m.iloc[i], sc_d.iloc[i]], axis=1).dropna()
            if len(j) >= bt.MIN_NAMES:
                cs.append(bt.spearman(j.iloc[:, 0].to_numpy(), j.iloc[:, 1].to_numpy()))
        out[f"rankcorr_mom_vs_drift_{split}"] = float(np.nanmean(cs))
    return out


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# TEST (coordinator only): frozen choices, all tickers, 2019+
# ---------------------------------------------------------------------------

def run_test(dry=False):
    """dry=True rehearses this exact code path on VAL (TRAIN tickers, data cut at
    2018-12-31) so the coordinator knows it runs; it never touches TEST."""
    ev_split = "val" if dry else "test"
    tag = "xsec_dryrun_val" if dry else "xsec_test"
    with open(FROZEN) as f:
        frozen = json.load(f)
    tickers = splits.tickers_for("dev" if dry else "test")
    panel = bt.Panel(tickers, end=splits.VAL_END if dry else None)
    elig = bt.eligibility(panel)
    sigs = compute_signals(panel)
    scores = {n: rank_score(sigs[n], elig) for n in BASE}
    log(f"{ev_split.upper()} panel {len(panel.tickers)} tickers to {panel.cal[-1].date()}, missing {panel.missing}")
    rows, yr = [], []
    prim = [(c["signal"], c["freq"], c["orient"]) for c in frozen.get("primary_for_test", [])]
    todo = list(prim)
    todo += [(c["signal"], c["freq"], c["orient"]) for c in frozen.get("candidates_val", [])
             if (c["signal"], c["freq"], c["orient"]) not in todo]
    todo += [(n, f, 1) for f in FREQS for n in ("combo_dev", "ridge_dev") if (n, f, 1) not in todo]
    for name, fname, orient in todo:
        step, fz = FREQS[fname], frozen["freqs"][fname]
        if name == "combo_dev":
            if not fz["combo_members"]:
                continue
            sig = combo_signal(scores, fz["combo_members"])
        elif name == "ridge_dev":
            sig = linear_signal(scores, fz["ridge"]["weights"], elig)
        elif name == "logit_dev":
            sig = linear_signal(scores, fz["logit"]["weights"], elig)
        elif name == "combo_all_prior":
            sig = combo_signal(scores, {n: 1 for n in BASE})
        else:
            sig = sigs[name]
        sig = sig * orient
        for lag in (1, 0):
            row, res = evaluate(panel, sig, elig, ev_split, step, lag, daily=(lag == 1))
            rows.append(dict(signal=name, freq=fname, lag=lag, primary=(name, fname, orient) in prim, **row))
            if lag == 1:
                y = bt.by_year(res)
                y.insert(0, "freq", fname)
                y.insert(0, "signal", name)
                yr.append(y)
    for fname, step in FREQS.items():
        rr = [evaluate(panel, bt.random_signal(panel, s), elig, ev_split, step, 1, daily=False)[0]
              for s in range(N_RANDOM)]
        rr = pd.DataFrame(rr)
        rows.append(dict(signal="(random x20, mean)", freq=fname, lag=1, ic_mean=rr.ic_mean.mean(),
                         excess_ann=rr.excess_ann.mean(), top_ann=rr.top_ann.mean()))
        la, _ = evaluate(panel, bt.forward_return_signal(panel, step, 1), elig, ev_split, step, 1, daily=False)
        rows.append(dict(signal="(look-ahead)", freq=fname, lag=1, **la))
        # survivorship yardstick: a stock's own long-run mean return has no reason to
        # predict anything; whatever it earns here is what hindsight universe choice pays
        drift = np.log(panel.pff).diff().expanding(min_periods=252).mean()
        row, _ = evaluate(panel, drift, elig, ev_split, step, 1)
        rows.append(dict(signal="(drift placebo)", freq=fname, lag=1, **row))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, tag + ".csv"), index=False)
    if yr:
        pd.concat(yr).to_csv(os.path.join(RES, tag + "_years.csv"))
    cols = ["signal", "freq", "lag", "primary", "ic_mean", "ic_t", "ic_daily_t", "top_ann", "ew_ann", "spy_ann",
            "excess_ann", "excess_t", "top_sharpe", "top_maxdd", "turnover_1way_ann", "ls_ann", "names_avg"]
    print(out[[c for c in cols if c in out]].round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["devval", "test", "dryrun"], default="devval",
                    help="dryrun = rehearse the TEST code path on VAL (safe for anyone)")
    ap.add_argument("--coordinator", action="store_true",
                    help="required for --split test (TEST is opened by the coordinator only)")
    a = ap.parse_args()
    if a.split == "test":
        if not a.coordinator:
            sys.exit("TEST is coordinator-only: pass --coordinator if you are the coordinator.")
        if not os.path.exists(FROZEN):
            sys.exit("no frozen DEV/VAL choices; run the devval split first")
        run_test()
    elif a.split == "dryrun":
        run_test(dry=True)
    else:
        run_devval()


if __name__ == "__main__":
    main()
