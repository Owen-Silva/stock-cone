"""How much of the frozen TEST momentum result is survivorship / universe selection?

Lane cone-xrobust. MEASUREMENT ONLY: runs the FROZEN signal mom_12_1 (weekly and
monthly, next close, 10 bps per side) on TEST (2019+, all tickers) restricted to
universe subsets that carry less hindsight selection, plus yardsticks (drift
placebo, 20 random signals), concentration, a selection-date split and by-year.
Nothing here designs, tunes or selects a signal. Nothing here downloads: prices
are read from data/prices via research.backtest.Panel (engine._cache_file/_clean).

  python research/xsec_robust_test.py

Writes research/results/xrobust_*.csv and xrobust_tables.md.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import engine  # noqa: E402  (BASKET constant only)
from universe import UNIVERSE, WATCHLIST_ID  # noqa: E402
from research import backtest as bt  # noqa: E402
from research import splits  # noqa: E402
from research import xsec  # noqa: E402

RES = os.path.join(ROOT, "research", "results")
FREQS = {"weekly": 5, "monthly": 21}
LAG = 1
N_RANDOM = 20
SUB_MIN_NAMES = 10      # subsets are small (mega has 20 names); the full universe keeps 20


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------------------

def evaluate(panel, sig, elig, step, min_names):
    sidx = bt.split_window(panel, "test", elig, step, LAG, min_names=min_names)
    res = bt.run(panel, sig, elig, sidx, step, LAG, min_names=min_names)
    return bt.summarize(res, step), res, sidx


def attribution(panel, sig, elig, sidx, step, min_names):
    """Per-stock contribution to the GROSS top-minus-EW excess, replicating bt.run's
    selection exactly. contribution_i = sum_t a_it (r_it - r_ew,t), a = w_top - w_ew.
    Returned annualised (mean per period x periods per year)."""
    S = sig.reindex(index=panel.cal, columns=panel.tickers).to_numpy(float)
    E = elig.reindex(index=panel.cal, columns=panel.tickers).fillna(False).to_numpy(bool)
    P = panel.pff.to_numpy(float)
    nt = P.shape[1]
    tot, nper, act = np.zeros(nt), 0, []
    for k in range(len(sidx) - 1):
        s, e0, e1 = sidx[k], sidx[k] + LAG, sidx[k + 1] + LAG
        with np.errstate(invalid="ignore", divide="ignore"):
            r = P[e1] / P[e0] - 1
        sv = S[s]
        valid = E[s] & np.isfinite(sv) & np.isfinite(r)
        n = int(valid.sum())
        if n < min_names:
            continue
        vi = np.flatnonzero(valid)
        order = vi[np.argsort(sv[vi], kind="stable")]
        nq = max(1, int(round(n * bt.QUANT)))
        a = np.zeros(nt)
        a[vi] = -1.0 / n
        a[order[-nq:]] += 1.0 / nq
        rel = np.where(valid, r - r[vi].mean(), 0.0)
        c = a * rel
        tot += c
        act.append(c.sum())
        nper += 1
    ppy = 252.0 / step
    return pd.Series(tot / max(nper, 1) * ppy, index=panel.tickers), float(np.mean(act) * ppy)


def drift_signal(panel):
    return np.log(panel.pff).diff().expanding(min_periods=252).mean()


def row_of(label, kind, fname, s, extra=None):
    keep = ["start", "end", "periods", "names_avg", "nq_avg", "ic_mean", "ic_t", "top_ann", "ew_ann",
            "excess_ann", "excess_t", "spy_ann", "top_maxdd", "ew_maxdd", "turnover_1way_ann", "top_sharpe"]
    d = {"subset": label, "kind": kind, "freq": fname}
    d.update({k: s.get(k, np.nan) for k in keep})
    if extra:
        d.update(extra)
    return d


def main():
    t0 = time.time()
    tickers = splits.tickers_for("test")
    panel = bt.Panel(tickers)
    elig_all = bt.eligibility(panel)
    mom = xsec.compute_signals(panel)["mom_12_1"]          # frozen, orient +1
    drift = drift_signal(panel)
    log(f"TEST panel {len(panel.tickers)} tickers to {panel.cal[-1].date()}, missing {panel.missing}")

    groups = {t: set(g) for t, g in UNIVERSE}
    T = panel.tickers
    in_g = lambda gid: {t for t in T if gid in groups.get(t, set())}  # noqa: E731
    subsets = {
        "a_full": set(T),
        "b_analog_pool": set(T) & set(engine.BASKET),
        "c_mega": in_g("mega"),
        "d_other": in_g("other"),
        "e_ex_volatile_traded": set(T) - in_g("volatile") - in_g(WATCHLIST_ID),  # label kept for result-file continuity
        "f_volatile": in_g("volatile"),
    }

    # --- selection-date groups (dollar volume is the size proxy; no market caps cached)
    first = panel.close.apply(lambda c: c.first_valid_index())
    dv = panel.close * panel.volume
    dv18 = dv.loc["2018-10-01":"2018-12-31"].median()
    dv20 = dv.loc["2020-07-01":"2020-12-31"].median()
    pre = first <= pd.Timestamp("2017-12-31")
    A = set(first.index[pre & (dv18 >= 100e6)])
    B = set(first.index[(first >= pd.Timestamp("2019-01-01")) | (dv18.fillna(0) < 25e6)])
    B21 = set(first.index[(first >= pd.Timestamp("2021-01-01")) | (dv20.fillna(0) < 25e6)])
    MID = set(T) - A - B
    subsets["g_large_liquid_2019"] = A
    subsets["h_new_or_tiny_2019"] = B
    subsets["i_full_minus_new_or_tiny"] = set(T) - B
    subsets["j_full_minus_new_or_tiny_2021def"] = set(T) - B21
    pd.DataFrame([{"ticker": t, "first_close": str(first[t].date()), "dv_2018q4_musd": dv18[t] / 1e6,
                   "dv_2020h2_musd": dv20[t] / 1e6,
                   "sel_group": "A_large_liquid" if t in A else ("B_new_or_tiny" if t in B else "mid"),
                   "new_or_tiny_2021def": t in B21,
                   "groups": ",".join(sorted(groups.get(t, {"pool_only"})))} for t in T]
                 ).to_csv(os.path.join(RES, "xrobust_selection_groups.csv"), index=False)
    log(f"selection groups: A {len(A)}, B {len(B)}, mid {len(MID)}, B(2021 def) {len(B21)}")

    rows, years, contrib_rows = [], [], []
    full_res = {}
    for label, members in subsets.items():
        mask = pd.Series(panel.close.columns.isin(sorted(members)), index=panel.close.columns)
        el = elig_all & mask
        mn = 20 if label == "a_full" else SUB_MIN_NAMES
        for fname, step in FREQS.items():
            s, res, sidx = evaluate(panel, mom, el, step, mn)
            if not s:
                log(f"{label} {fname}: too few periods")
                continue
            # yardsticks on the same subset
            sd, _, _ = evaluate(panel, drift, el, step, mn)
            rr = pd.DataFrame([evaluate(panel, bt.random_signal(panel, k), el, step, mn)[0]
                               for k in range(N_RANDOM)])
            extra = {"n_members": len(members), "min_names": mn,
                     "drift_ic_t": sd.get("ic_t"), "drift_excess_ann": sd.get("excess_ann"),
                     "drift_excess_t": sd.get("excess_t"),
                     "rand_excess_mean": rr.excess_ann.mean(), "rand_excess_sd": rr.excess_ann.std(),
                     "rand_excess_max": rr.excess_ann.max(),
                     "rand_ic_t_sd": rr.ic_t.std(),
                     "mom_beats_n_random": int((rr.excess_ann < s["excess_ann"]).sum())}
            rows.append(row_of(label, "mom_12_1", fname, s, extra))
            y = bt.by_year(res)
            y.insert(0, "freq", fname)
            y.insert(0, "subset", label)
            years.append(y)
            if label in ("a_full", "e_ex_volatile_traded", "b_analog_pool"):
                full_res[(label, fname)] = (el, sidx, mn, s)
            log(f"{label:34s} {fname:7s} names {s['names_avg']:.0f} excess {s['excess_ann']:+.3f} "
                f"t {s['excess_t']:.2f} | drift {sd.get('excess_ann', np.nan):+.3f} "
                f"| rand {rr.excess_ann.mean():+.3f}+-{rr.excess_ann.std():.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "xrobust_subsets.csv"), index=False)
    pd.concat(years).to_csv(os.path.join(RES, "xrobust_years.csv"))

    # --- concentration + selection-group attribution
    conc = []
    for (label, fname), (el, sidx, mn, s) in full_res.items():
        step = FREQS[fname]
        c, tot = attribution(panel, mom, el, sidx, step, mn)
        c = c[el.any()]
        cs = c.sort_values(ascending=False)
        for t, v in cs.items():
            contrib_rows.append({"subset": label, "freq": fname, "ticker": t, "contrib_ann": v,
                                 "share": v / tot, "sel_group": "A" if t in A else ("B" if t in B else "mid")})
        rec = {"subset": label, "freq": fname, "gross_excess_arith_ann": tot,
               "top5_share": cs.iloc[:5].sum() / tot, "top10_share": cs.iloc[:10].sum() / tot,
               "top10": " ".join(cs.index[:10]),
               "contrib_A": c[c.index.isin(sorted(A))].sum(), "contrib_B": c[c.index.isin(sorted(B))].sum(),
               "contrib_mid": c[c.index.isin(sorted(MID))].sum(),
               "excess_ann": s["excess_ann"], "excess_t": s["excess_t"]}
        for k in (5, 10):
            drop = list(cs.index[:k])
            el2 = el & ~pd.Series(panel.close.columns.isin(drop), index=panel.close.columns)
            s2, _, _ = evaluate(panel, mom, el2, step, mn)
            rec[f"ex_top{k}_excess_ann"] = s2["excess_ann"]
            rec[f"ex_top{k}_excess_t"] = s2["excess_t"]
            rec[f"ex_top{k}_ic_t"] = s2["ic_t"]
            # control: the same ex-post "drop your own top-k contributors" on random signals
            dr = []
            for kk in range(N_RANDOM):
                rs = bt.random_signal(panel, kk)
                s0, _, sx = evaluate(panel, rs, el, step, mn)
                cr, _ = attribution(panel, rs, el, sx, step, mn)
                dd = list(cr.sort_values(ascending=False).index[:k])
                el3 = el & ~pd.Series(panel.close.columns.isin(dd), index=panel.close.columns)
                s3, _, _ = evaluate(panel, rs, el3, step, mn)
                dr.append(s0["excess_ann"] - s3["excess_ann"])
            rec[f"rand_drop_top{k}_mean"] = float(np.mean(dr))
        conc.append(rec)
        log(f"conc {label} {fname}: top5 {rec['top5_share']:.0%} top10 {rec['top10_share']:.0%} "
            f"ex5 {rec['ex_top5_excess_ann']:+.3f} ex10 {rec['ex_top10_excess_ann']:+.3f} "
            f"| A {rec['contrib_A']:+.3f} B {rec['contrib_B']:+.3f} mid {rec['contrib_mid']:+.3f} of {tot:+.3f}")
    pd.DataFrame(conc).to_csv(os.path.join(RES, "xrobust_concentration.csv"), index=False)
    pd.DataFrame(contrib_rows).to_csv(os.path.join(RES, "xrobust_contrib.csv"), index=False)

    # --- readable tables
    cols = ["subset", "freq", "n_members", "names_avg", "nq_avg", "ic_mean", "ic_t", "top_ann", "ew_ann",
            "excess_ann", "excess_t", "spy_ann", "top_maxdd", "turnover_1way_ann", "drift_excess_ann",
            "drift_excess_t", "rand_excess_mean", "rand_excess_sd", "mom_beats_n_random"]
    yy = pd.concat(years).reset_index().rename(columns={"index": "year", "date": "year"})
    with open(os.path.join(RES, "xrobust_tables.md"), "w") as f:
        f.write(out[cols].round(3).to_string(index=False) + "\n\n")
        f.write(pd.DataFrame(conc).round(3).T.to_string() + "\n\n")
        f.write(yy.pivot_table(index="year", columns=["freq", "subset"], values="excess").round(3).to_string() + "\n")
    print(out[cols].round(3).to_string(index=False))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
