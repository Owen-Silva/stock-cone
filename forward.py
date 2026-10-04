"""Stock Cone live test: log the exact cone the app serves each day, score it later.

Every backtest number in this repo comes from today's stock list looking backwards
(survivorship bias). The only clean evidence is a forward log: write down the cone the
app served on a day, wait for the real price, then score it. Nothing in a logged record
can be changed by later data.

    python forward.py --snapshot    log today's cones (155 scored stocks x 1W/1M/3M)
    python forward.py --score       score every logged record whose outcome exists
    python forward.py --status      print status + per-horizon summary (the /api/forward body)
    python forward.py --selfcheck   snapshot a PAST date into a temp dir, score it, check it
                                    (add --momentum-only for just the momentum tilt checks)

Files (data/forward/):
    <asof>.jsonl    one JSON record per (ticker, horizon); appended line by line, so a crash
                    leaves a partial file that the next run finishes (never duplicates)
    mom-<asof>.json the momentum tilt record for that as-of (pre-registered live test, see
                    research/PLAN.md): eligible stocks' mom_12_1, the top fifth, turnover vs
                    the previous official month. Separate files, so the cone readers skip them
    scores.json     the per-horizon summary written by score(), plus the "momentum" section
    state.json      last_run, last_error, and the last snapshot's counts

The cone is built by the same engine steps as /api/forecast (Engine.forecast): the same
ticker refresh, the same live earnings widening (the latest bar only), the same
Engine._forecast_at. Engine.forecast also adds the stock's name and track record; those do
not change the cone and the track record would start a walk-forward per stock, so the log
skips them. `--selfcheck` proves the logged values equal Engine.forecast's.
"""

import argparse
import datetime as dt
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback

import numpy as np
import pandas as pd

import engine as E

FORWARD_DIR = os.path.join(E.DATA_DIR, "forward")
ACCURACY_PATH = os.path.join(E.DATA_DIR, "accuracy.json")
MODEL_KEY = "cone2_f_conf_dev+earnings|" + E.TRACK_VERSION   # cone_model.py v2 + live earnings widening
PAUSE_S = 0.4            # between stocks: go easy on Yahoo and leave the GIL to live requests
EARN_SETTLE_S = 30.0     # wait this long for a stock's first earnings lookup of the day
REFRESH_WAIT_S = 1800    # wait at most this long for the engine's daily pool refresh
PERMANENT = (400, 404, 422)   # ForecastError statuses that will not change by retrying today
DATE_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.jsonl$")

_run_lock = threading.Lock()
STATE = {"running": False, "phase": None, "progress": None, "last_run": None,
         "last_error": None, "last_snapshot": None, "last_score": None}


def now_utc():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _load_state(dirpath):
    s = _read_json(os.path.join(dirpath, "state.json")) or {}
    for k in ("last_run", "last_error", "last_snapshot", "last_score"):
        if STATE.get(k) is None and s.get(k) is not None:
            STATE[k] = s[k]


def _save_state(dirpath):
    _write_atomic(os.path.join(dirpath, "state.json"),
                  {k: STATE[k] for k in ("last_run", "last_error", "last_snapshot", "last_score")})


def scored_tickers():
    """The stocks the Accuracy scoreboard scores (data/accuracy.json 'stocks'), in universe
    order. Falls back to the whole universe if the scoreboard has never been built."""
    from universe import UNIVERSE
    acc = _read_json(ACCURACY_PATH) or {}
    have = {s.get("ticker") for s in acc.get("stocks") or [] if isinstance(s, dict)}
    if have:
        return [t for t, _ in UNIVERSE if t in have]
    return [t for t, _ in UNIVERSE]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def read_log(path):
    """Records of one <asof>.jsonl. A truncated last line (crash mid-write) is dropped and
    the file rewritten without it, so the next append starts on a clean line."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    recs, bad = [], False
    for ln in lines:
        if not ln.strip():
            continue
        try:
            recs.append(json.loads(ln))
        except ValueError:
            bad = True
    if bad:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for r in recs:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        os.replace(tmp, path)
    return recs


def _wait_for_engine(eng):
    """The served cone uses the engine's analog pool; make sure it is today's, the same way
    a live request would (Engine.maybe_refresh), and wait for it."""
    t0 = time.time()
    while not eng.ready:
        if time.time() - t0 > REFRESH_WAIT_S:
            raise RuntimeError(f"engine never became ready ({eng.status})")
        time.sleep(2)
    eng.maybe_refresh()
    while eng.refreshing:
        if time.time() - t0 > REFRESH_WAIT_S:
            raise RuntimeError("the engine's daily pool refresh took over 30 minutes")
        time.sleep(2)


def served_cones(eng, t, asof=None, settle=EARN_SETTLE_S):
    """{h: Engine._forecast_at output} for ticker t, built exactly as Engine.forecast builds
    the cone (minus name/track). asof=None means the latest bar (live, with the earnings
    widening); a past asof never sees the earnings calendar, like a replay."""
    td = eng.get_ticker(t)
    last_pos = len(td.day) - 1
    if asof:
        p = td.pos_on_or_before((dt.date.fromisoformat(asof) - E.EPOCH_DATE).days)
        if p < 0:
            raise E.ForecastError(400, f"asof {asof} is before {t}'s first trading day ({E.day_str(td.day[0])}).")
    else:
        p = last_pos
    if not td.valid[p]:
        first_ok = np.flatnonzero(td.valid)
        when = E.day_str(td.day[first_ok[0]]) if first_ok.size else "n/a"
        raise E.ForecastError(400, f"Not enough history before {E.day_str(td.day[p])} to describe the setup "
                                   f"(needs about a year). Earliest usable asof for {t}: {when}.")
    live = p == last_pos and eng.raw is None
    if live and settle > 0:
        eng.earnings_info(t, wait=settle)        # let today's lookup finish; live_event then reads the cache
    out = {}
    for h in E.HORIZONS:
        event = eng.live_event(td, p, h, wait=0.0) if live else None
        out[h] = eng._forecast_at(td, p, h, event=event)
    return E.day_str(td.day[p]), out


def make_record(asof, t, h, resp, created):
    f = resp["future"]
    ev = resp.get("event")
    note = None
    for n in resp.get("notes") or []:
        if "earnings" in n.lower():
            note = n
    return {
        "asof": asof, "ticker": t, "horizon": h,
        "target_date": f["dates"][-1] if f.get("dates") else None,
        "last_close": resp["last_close"],
        **{q: f[q][-1] for q in ("p10", "p25", "p50", "p75", "p90")},
        "p_up": resp["odds"]["p_up"], "base_p_up": resp["odds"]["base_p_up"],
        "event_widen": ev.get("widen") if ev else None,
        "event_date": ev.get("date") if ev else None,
        "event_note": note,
        "n_analogs": resp["odds"]["n_analogs"],
        "model": MODEL_KEY, "created_at": created,
    }


def snapshot(eng, asof=None, dirpath=FORWARD_DIR, tickers=None, say=print, pause=PAUSE_S):
    """Log the served cone for every scored ticker x horizon into <dirpath>/<asof>.jsonl.
    asof=None: the latest close (the live log). Idempotent: a complete file is left alone;
    a partial one is finished. A stock that cannot be forecast for a lasting reason (too
    little history, unknown, or its latest close is not the snapshot day) gets a `skip`
    record with the reason; a passing failure (Yahoo down) writes nothing, so the next run
    retries it, and is reported in the result's `errors`."""
    os.makedirs(dirpath, exist_ok=True)
    if asof is None:
        _wait_for_engine(eng)
        day = E.day_str(eng.get_ticker("SPY").day[-1])
    else:
        day = asof
    tickers = list(tickers or scored_tickers())
    path = os.path.join(dirpath, f"{day}.jsonl")
    have = {(r.get("ticker"), r.get("horizon")) for r in read_log(path)}
    todo = [t for t in tickers if any((t, h) not in have for h in E.HORIZONS)]
    res = {"asof": day, "expected": len(tickers) * len(E.HORIZONS), "already": len(have),
           "added": 0, "skipped": 0, "errors": [], "file": path}
    if not todo:
        return res
    say(f"forward snapshot {day}: {len(todo)} of {len(tickers)} stocks to log")
    t0 = time.time()
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for i, t in enumerate(todo):
            STATE["progress"] = [i, len(todo)]
            created = now_utc()
            lines = []
            try:
                got, cones = served_cones(eng, t, asof=asof)
                if got != day:
                    raise E.ForecastError(422, f"{t}'s latest close is {got}, not {day}")
                for h in E.HORIZONS:
                    if (t, h) not in have:
                        lines.append(make_record(day, t, h, cones[h], created))
            except E.ForecastError as e:
                if e.status not in PERMANENT:
                    res["errors"].append(f"{t}: {e.msg}")
                    continue
                for h in E.HORIZONS:
                    if (t, h) not in have:
                        lines.append({"asof": day, "ticker": t, "horizon": h, "skip": e.msg,
                                      "model": MODEL_KEY, "created_at": created})
            except Exception as e:  # keep going; say which stock broke and why
                res["errors"].append(f"{t}: {type(e).__name__}: {e}")
                continue
            for r in lines:
                fh.write(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n")
                have.add((t, r["horizon"]))
                res["skipped" if "skip" in r else "added"] += 1
            fh.flush()
            if pause:
                time.sleep(pause)
    res["seconds"] = round(time.time() - t0, 1)
    STATE["progress"] = None
    say(f"forward snapshot {day}: +{res['added']} cones, +{res['skipped']} skips, "
        f"{len(res['errors'])} errors in {res['seconds']}s")
    return res


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def _closes(t, cache):
    """Adjusted closes from the on-disk price cache (no download): a pd.Series by date."""
    if t not in cache:
        try:
            df = pd.read_csv(E._cache_file(t), index_col=0, parse_dates=True)
            s = df["close"].astype(float)
            s.index = pd.DatetimeIndex(s.index).normalize()
            cache[t] = s[np.isfinite(s) & (s > 0)].sort_index()
        except (OSError, KeyError, ValueError):
            cache[t] = None
    return cache[t]


def outcome(rec, cache):
    """None if the outcome is not in yet; else the scored dict. The real move is the ratio of
    two closes from TODAY's adjusted price file (dividends and splits cancel), applied to
    the logged last close, so it lines up with the logged cone."""
    s = _closes(rec["ticker"], cache)
    if s is None:
        return None
    a = pd.Timestamp(rec["asof"])
    i = int(s.index.searchsorted(a, side="left"))
    if i >= len(s) or s.index[i] != a:
        return None
    j = i + int(rec["horizon"])
    if j >= len(s):
        return None
    ratio = float(s.iloc[j] / s.iloc[i])
    real = float(rec["last_close"]) * ratio
    up = ratio > 1.0
    return {
        "ticker": rec["ticker"], "asof": rec["asof"], "horizon": int(rec["horizon"]),
        "end_date": s.index[j].date().isoformat(), "ratio": ratio, "real_close": real,
        "inside_80": bool(rec["p10"] <= real <= rec["p90"]),
        "inside_50": bool(rec["p25"] <= real <= rec["p75"]),
        "direction_hit": bool((rec["p50"] > rec["last_close"]) == up),
        "direction_base": bool((rec["base_p_up"] >= 0.5) == up),
    }


def log_files(dirpath):
    out = []
    try:
        names = sorted(os.listdir(dirpath))
    except OSError:
        return out
    for n in names:
        m = DATE_FILE.match(n)
        if m:
            out.append((m.group(1), os.path.join(dirpath, n)))
    return out


def score(dirpath=FORWARD_DIR, write=True, keep_records=False):
    """Score every logged cone whose outcome exists; aggregate per horizon, and count what
    is still waiting (with the date the next result is due). Writes <dirpath>/scores.json
    (unless write=False) and returns it. Reads the price cache only, never downloads."""
    cache = {}
    agg = {h: {"n": 0, "days": set(), "inside_80": 0, "inside_50": 0, "direction_hit": 0,
               "direction_base": 0, "first_asof": None, "last_asof": None,
               "pending": 0, "next_result": None} for h in E.HORIZONS}
    recs_out = []
    logged = 0
    for _, path in log_files(dirpath):
        for r in read_log(path):
            if "skip" in r or r.get("horizon") not in agg:
                continue
            logged += 1
            a = agg[r["horizon"]]
            o = outcome(r, cache)
            if o is None:
                a["pending"] += 1
                td = r.get("target_date")
                if td and (a["next_result"] is None or td < a["next_result"]):
                    a["next_result"] = td
                continue
            a["n"] += 1
            a["days"].add(o["asof"])
            for k in ("inside_80", "inside_50", "direction_hit", "direction_base"):
                a[k] += int(o[k])
            a["first_asof"] = min(a["first_asof"] or o["asof"], o["asof"])
            a["last_asof"] = max(a["last_asof"] or o["asof"], o["asof"])
            if keep_records:
                recs_out.append(o)
    horizons = {}
    for h, a in agg.items():
        n = a["n"]

        def rate(k):
            return round(a[k] / n, 4) if n else None
        horizons[str(h)] = {"n": n, "days": len(a["days"]), "inside_80": rate("inside_80"),
                            "inside_50": rate("inside_50"), "direction_hit": rate("direction_hit"),
                            "direction_base": rate("direction_base"),
                            "first_asof": a["first_asof"], "last_asof": a["last_asof"],
                            "pending": a["pending"], "next_result": a["next_result"]}
    out = {"scored_at": now_utc(), "cones_logged": logged, "horizons": horizons}
    try:
        out["momentum"] = mom_score(dirpath)
    except Exception as e:  # the cone scores still get written; say why the momentum part broke
        traceback.print_exc()
        out["momentum"] = {**EMPTY_MOM, "error": f"{type(e).__name__}: {e}"}
    if write:
        _write_atomic(os.path.join(dirpath, "scores.json"), out)
    if keep_records:
        out["records"] = recs_out
    return out


# ---------------------------------------------------------------------------
# Momentum tilt: a pre-registered live test (research/PLAN.md, "Live test: momentum tilt
# (pre-registered 2026-10-04)"). The rules below are frozen; do not tune them on results.
#
# Per logged as-of date, data/forward/mom-<asof>.json holds the eligible stocks' mom_12_1,
# the equal-weight top fifth, and the turnover vs the previous official month. Its own file
# name (not <asof>.jsonl), so the cone readers above never see it. Returns are computed at
# score time from today's adjusted price cache (ratios, so dividends/splits cancel).
# ---------------------------------------------------------------------------

MOM_MODEL = "mom_12_1|top_fifth|21d|v1"
MOM_FILE = re.compile(r"^mom-(\d{4}-\d{2}-\d{2})\.json$")
MOM_SKIP, MOM_LOOK = 21, 252        # mom_12_1 = log P[t-21] - log P[t-252] (research/xsec.py)
MOM_MIN_HIST = MOM_LOOK + MOM_SKIP  # a stock with fewer days of history is left out
MOM_HOLD = 21                       # trading days held
MOM_QUANT = 0.2                     # top fifth: max(1, round(n * 0.2)), as research/backtest.run
MOM_COST_BPS = 10.0                 # per side; cost = one-way turnover x 10 bps x 2
MOM_MIN_MONTHS = 12                 # no t-stat and no verdict before this many official months
EMPTY_MOM = {"model": MOM_MODEL, "records": 0, "latest_asof": None, "eligible_latest": None,
             "top_n_latest": None, "official_dates": [], "first_result": None, "next_result": None,
             "pending_months": 0, "months": [],
             "official": {"n": 0, "mean_excess_ew": None, "mean_excess_spy": None, "wins_ew": 0,
                          "wins_spy": 0, "hit_rate_ew": None, "t_stat_ew": None, "t_stat_spy": None},
             "informational": {"n": 0, "mean_excess_ew": None, "mean_excess_spy": None,
                               "hit_rate_ew": None},
             "min_months": MOM_MIN_MONTHS, "too_small": True, "verdict": "too_small",
             "cost_bps_per_side": MOM_COST_BPS}


def _bt():
    from research import backtest as bt   # Panel: the research calendar (^GSPC u SPY), pff = ffill(limit=5)
    return bt


def mom_signal(panel):
    """mom_12_1 exactly as research/xsec.py compute_signals: ret(21, 252) on log panel.pff,
    shifted by calendar rows. Causal: the value at row t uses rows t-252 and t-21 only."""
    lp = np.log(panel.pff)
    return lp.shift(MOM_SKIP) - lp.shift(MOM_LOOK)


def mom_top(mom):
    """Equal-weight top fifth by mom_12_1, ties broken by ticker."""
    order = sorted(mom, key=lambda t: (-mom[t], t))
    return order[:max(1, int(round(len(order) * MOM_QUANT)))]


def mom_turnover(new, old):
    """One-way turnover between two equal-weight lists: 0.5 * sum |w_new - w_old|.
    No previous list (the first month) = 1.0, i.e. 20 bps."""
    if not old:
        return 1.0
    wn, wo = 1.0 / len(new), 1.0 / len(old)
    names = set(new) | set(old)
    return 0.5 * sum(abs((wn if t in new else 0.0) - (wo if t in old else 0.0)) for t in names)


def mom_cost(to):
    return to * MOM_COST_BPS / 1e4 * 2


def mom_files(dirpath):
    out = []
    try:
        names = sorted(os.listdir(dirpath))
    except OSError:
        return out
    for n in names:
        m = MOM_FILE.match(n)
        if m:
            out.append((m.group(1), os.path.join(dirpath, n)))
    return out


def mom_official(dates):
    """The first as-of date of each calendar month (the scored, non-overlapping series)."""
    first = {}
    for d in sorted(dates):
        first.setdefault(d[:7], d)
    return sorted(first.values())


def mom_record(asof, tickers, created=None):
    """The as-of's momentum record from prices up to and including the as-of only."""
    bt = _bt()
    end = pd.Timestamp(asof)
    panel = bt.Panel(list(tickers), end=asof)
    # no look-ahead: not one price after the as-of is in the panel the signal is built from
    assert panel.cal[-1] <= end and panel.close.index.max() <= end and panel.mkt.index.max() <= end, \
        f"a price after {asof} reached the momentum signal"
    if panel.cal[-1] != end:
        raise ValueError(f"{asof} is not in the price cache's calendar (last day {panel.cal[-1].date()})")
    sig = mom_signal(panel).iloc[-1]
    hist = panel.close.notna().sum()
    entry = panel.pff.iloc[-1]
    mom, excluded = {}, {}
    for t in tickers:
        if t not in panel.close.columns:
            excluded[t] = "no cached prices"
        elif int(hist[t]) < MOM_MIN_HIST:
            excluded[t] = f"{int(hist[t])} days of history by {asof} (needs {MOM_MIN_HIST})"
        elif not (np.isfinite(sig[t]) and np.isfinite(entry[t])):
            excluded[t] = "no usable close at the as-of or 21/252 days before"
        else:
            mom[t] = float(sig[t])
    top = mom_top(mom)
    return {"asof": asof, "model": MOM_MODEL, "created_at": created or now_utc(),
            "prices_through": panel.cal[-1].date().isoformat(), "universe": len(tickers),
            "eligible": len(mom), "top_n": len(top), "top": top, "mom": mom, "excluded": excluded}


def mom_snapshot(dirpath=FORWARD_DIR, dates=None, tickers=None, say=print):
    """Write mom-<asof>.json for every as-of in `dates` (default: every logged cone day) that
    does not have one yet. Idempotent: an existing record is never rewritten."""
    if dates is None:
        dates = [d for d, _ in log_files(dirpath)]
    tickers = list(tickers or scored_tickers())
    have = {d: p for d, p in mom_files(dirpath)}
    res = {"added": [], "already": 0, "errors": []}
    for d in sorted(dates):
        if d in have:
            res["already"] += 1
            continue
        try:
            rec = mom_record(d, tickers)
            off = mom_official(list(have) + [d])
            prev = [x for x in off if x < d]
            prev_top = (_read_json(have[prev[-1]]) or {}).get("top") if prev else None
            rec["official"] = d in off
            rec["prev_official"] = prev[-1] if prev else None
            rec["turnover"] = mom_turnover(rec["top"], prev_top)
            rec["cost"] = mom_cost(rec["turnover"])
            path = os.path.join(dirpath, f"mom-{d}.json")
            _write_atomic(path, rec)
            have[d] = path
            res["added"].append(d)
            say(f"momentum record {d}: {rec['eligible']} eligible, top fifth {rec['top_n']}, "
                f"turnover {rec['turnover']:.3f}{' (official)' if rec['official'] else ''}")
        except Exception as e:
            res["errors"].append(f"{d}: {type(e).__name__}: {e}")
    return res


def _mom_summary(rows, tstat):
    n = len(rows)
    if not n:
        return {"n": 0, "mean_excess_ew": None, "mean_excess_spy": None, "wins_ew": 0, "wins_spy": 0,
                "hit_rate_ew": None, "t_stat_ew": None, "t_stat_spy": None}
    xe = np.array([r["excess_ew"] for r in rows])
    xs = np.array([r["excess_spy"] for r in rows])

    def t(x):
        if not tstat or n < MOM_MIN_MONTHS or x.std(ddof=1) == 0:
            return None
        return round(float(x.mean() / (x.std(ddof=1) / np.sqrt(n))), 3)
    return {"n": n, "mean_excess_ew": round(float(xe.mean()), 6), "mean_excess_spy": round(float(xs.mean()), 6),
            "wins_ew": int((xe > 0).sum()), "wins_spy": int((xs > 0).sum()),
            "hit_rate_ew": round(float((xe > 0).mean()), 4), "t_stat_ew": t(xe), "t_stat_spy": t(xs)}


def mom_score(dirpath=FORWARD_DIR):
    """The momentum section of scores.json / /api/forward. No stock names: not a picks list."""
    recs = [r for r in (_read_json(p) for _, p in mom_files(dirpath)) if r and r.get("top")]
    if not recs:
        return dict(EMPTY_MOM)
    recs.sort(key=lambda r: r["asof"])
    bt = _bt()
    panel = bt.Panel(sorted({t for r in recs for t in r["mom"]}))
    P, M, cal = panel.pff, panel.mkt, panel.cal
    dates = [r["asof"] for r in recs]
    off = mom_official(dates)
    tops = {r["asof"]: r["top"] for r in recs}
    rows_off, rows_all, pend = [], [], []
    first_result = None
    for r in recs:
        d = r["asof"]
        i = int(cal.searchsorted(pd.Timestamp(d), side="left"))
        if i >= len(cal) or cal[i] != pd.Timestamp(d):
            raise ValueError(f"momentum as-of {d} is not in the price cache's calendar")
        j = i + MOM_HOLD
        if j < len(cal):
            end, done = cal[j].date().isoformat(), True
        else:
            end = E.project_trading_days(cal[-1].date(), j - (len(cal) - 1))[-1].isoformat()
            done = False
        if d == off[0]:
            first_result = end
        if not done:
            if d in off:
                pend.append(end)
            continue
        win = P.iloc[i:j + 1].ffill()          # a stock that stops trading keeps its last close
        ret = win.iloc[-1] / win.iloc[0] - 1.0
        top_ret = float(ret[r["top"]].mean())
        ew_ret = float(ret[list(r["mom"])].mean())
        spy_ret = float(M.iloc[j] / M.iloc[i] - 1.0)
        prev = [x for x in off if x < d]
        to = mom_turnover(r["top"], tops[prev[-1]] if prev else None)
        cost = mom_cost(to)
        row = {"asof": d, "end_date": end, "eligible": r["eligible"], "top_n": r["top_n"],
               "top_ret": round(top_ret, 6), "ew_ret": round(ew_ret, 6), "spy_ret": round(spy_ret, 6),
               "turnover": round(to, 6), "cost": round(cost, 6),
               "excess_ew": round(top_ret - cost - ew_ret, 6), "excess_spy": round(top_ret - cost - spy_ret, 6)}
        row["beat_ew"] = row["excess_ew"] > 0
        rows_all.append(row)
        if d in off:
            rows_off.append(row)
    n = len(rows_off)
    if n >= MOM_MIN_MONTHS:   # the kill rule reads the first 12 official months, then is final
        first12 = float(np.mean([x["excess_ew"] for x in rows_off[:MOM_MIN_MONTHS]]))
        verdict = "dead" if first12 <= 0 else "alive"
    else:
        verdict = "too_small"
    info = _mom_summary(rows_all, tstat=False)
    last = recs[-1]
    return {"model": MOM_MODEL, "records": len(recs), "latest_asof": last["asof"],
            "eligible_latest": last["eligible"], "top_n_latest": last["top_n"],
            "official_dates": off, "first_result": first_result,
            "next_result": min(pend) if pend else None, "pending_months": len(pend),
            "months": rows_off, "official": _mom_summary(rows_off, tstat=True),
            "informational": {k: info[k] for k in ("n", "mean_excess_ew", "mean_excess_spy", "hit_rate_ew")},
            "min_months": MOM_MIN_MONTHS, "too_small": n < MOM_MIN_MONTHS, "verdict": verdict,
            "cost_bps_per_side": MOM_COST_BPS}


# ---------------------------------------------------------------------------
# Status (the /api/forward body): cheap, reads state.json + scores.json + the file list
# ---------------------------------------------------------------------------

EMPTY_H = {"n": 0, "days": 0, "inside_80": None, "inside_50": None, "direction_hit": None,
           "direction_base": None, "first_asof": None, "last_asof": None, "pending": 0,
           "next_result": None}


def status(dirpath=FORWARD_DIR):
    _load_state(dirpath)
    files = log_files(dirpath)
    sc = _read_json(os.path.join(dirpath, "scores.json")) or {}
    hz = {str(h): {**EMPTY_H, **((sc.get("horizons") or {}).get(str(h)) or {})} for h in E.HORIZONS}
    if STATE["running"]:
        st = "building" if not files or not STATE["last_run"] else "running"   # building = first run
    else:
        st = "ready" if files else "empty"
    return {
        "status": st,
        "phase": STATE["phase"] if STATE["running"] else None,
        "progress": STATE["progress"] if STATE["running"] else None,
        "since": files[0][0] if files else None,
        "latest": files[-1][0] if files else None,
        "days_logged": len(files),
        "cones_logged": sc.get("cones_logged", 0),
        "records_scored": sum(v["n"] for v in hz.values()),
        "records_pending": {k: v["pending"] for k, v in hz.items()},
        "model": MODEL_KEY,
        "last_run": STATE["last_run"],
        "last_error": STATE["last_error"],
        "last_snapshot": STATE["last_snapshot"],
        "scored_at": sc.get("scored_at"),
        "horizons": hz,
        "momentum": {**EMPTY_MOM, **(sc.get("momentum") or {}),
                     "records": max(len(mom_files(dirpath)), (sc.get("momentum") or {}).get("records") or 0)},
    }


def run_once(eng, dirpath=FORWARD_DIR, say=print):
    """snapshot() then score(); never raises. Errors land in last_error with the reason."""
    if not _run_lock.acquire(blocking=False):
        return
    _load_state(dirpath)
    STATE.update(running=True, phase="snapshot", progress=None)
    errs = []
    try:
        try:
            res = snapshot(eng, dirpath=dirpath, say=say)
            STATE["last_snapshot"] = {k: res[k] for k in ("asof", "expected", "already", "added", "skipped")}
            STATE["last_snapshot"]["errors"] = len(res["errors"])
            if res["errors"]:
                errs.append(f"snapshot {res['asof']}: {len(res['errors'])} stock(s) not logged, will retry "
                            f"next run (first: {res['errors'][0]})")
        except Exception as e:
            traceback.print_exc()
            errs.append(f"snapshot failed: {type(e).__name__}: {e}")
        STATE["phase"] = "momentum"
        try:
            mres = mom_snapshot(dirpath, say=say)
            if mres["errors"]:
                errs.append(f"momentum record: {len(mres['errors'])} day(s) not written, will retry next run "
                            f"(first: {mres['errors'][0]})")
        except Exception as e:
            traceback.print_exc()
            errs.append(f"momentum record failed: {type(e).__name__}: {e}")
        STATE["phase"] = "score"
        try:
            sc = score(dirpath)
            STATE["last_score"] = sc["scored_at"]
            if (sc.get("momentum") or {}).get("error"):
                errs.append(f"momentum score failed: {sc['momentum']['error']}")
        except Exception as e:
            traceback.print_exc()
            errs.append(f"score failed: {type(e).__name__}: {e}")
        STATE["last_run"] = now_utc()
        STATE["last_error"] = ({"at": STATE["last_run"], "message": "; ".join(errs)} if errs else None)
        try:
            _save_state(dirpath)
        except OSError as e:
            STATE["last_error"] = {"at": STATE["last_run"], "message": f"could not write state.json: {e}"}
    finally:
        STATE.update(running=False, phase=None, progress=None)
        _run_lock.release()


def start_loop(eng, every_s=6 * 3600, say=print):
    """Daemon thread: run_once now, then every `every_s` seconds."""
    def loop():
        while True:
            try:
                run_once(eng, say=say)
            except Exception as e:      # run_once never raises; belt and braces
                STATE["last_error"] = {"at": now_utc(), "message": f"{type(e).__name__}: {e}"}
            time.sleep(every_s)
    th = threading.Thread(target=loop, name="forward-log", daemon=True)
    th.start()
    return th


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def mom_selfcheck(check, dates=("2026-04-01", "2026-04-15", "2026-05-01")):
    """The momentum tilt at past dates in a temp dir: two official months (Apr 1, May 1) and
    one informational day in between (Apr 15)."""
    bt = _bt()
    tmp = tempfile.mkdtemp(prefix="forward-mom-selfcheck-")
    try:
        tickers = scored_tickers()
        print(f"momentum records as of {', '.join(dates)} for {len(tickers)} stocks into {tmp}")
        r1 = mom_snapshot(tmp, dates=list(dates), tickers=tickers, say=lambda *a: None)
        check(r1["added"] == list(dates) and not r1["errors"],
              f"momentum: {len(r1['added'])} records written, errors {r1['errors'][:2]}")
        recs = {d: _read_json(os.path.join(tmp, f"mom-{d}.json")) for d in dates}
        d1, dm, d2 = dates
        a, b = recs[d1], recs[d2]
        print(f"  {d1}: {a['eligible']} eligible of {a['universe']}, top fifth {a['top_n']}; "
              f"{d2}: {b['eligible']} eligible, top fifth {b['top_n']}")

        # 1. the signal equals research/xsec.py's mom_12_1 (its own compute_signals) on 3 tickers
        from research import xsec
        for t in ("AAPL", "JPM", "KO"):
            pan = bt.Panel([t], end=d1)
            xv = float(xsec.compute_signals(pan)["mom_12_1"][t].iloc[-1])
            mv = a["mom"].get(t)
            check(mv is not None and abs(mv - xv) < 1e-12, f"momentum: {t} mom_12_1 at {d1} = {mv} vs xsec {xv:.12f}")
        srt = sorted(a["mom"], key=lambda t: (-a["mom"][t], t))
        check(a["top"] == srt[:a["top_n"]] and a["top_n"] == max(1, int(round(a["eligible"] * 0.2))),
              f"momentum: top fifth = the {a['top_n']} highest mom_12_1 of {a['eligible']} (ties by ticker)")
        short = []
        for t in a["mom"]:
            dfc = pd.read_csv(E._cache_file(t), index_col=0, parse_dates=True)
            if int((dfc.index <= pd.Timestamp(d1)).sum()) < MOM_MIN_HIST:
                short.append(t)
        check(a["eligible"] + len(a["excluded"]) == len(tickers) and not short,
              f"momentum: eligible + excluded = {len(tickers)}, every eligible stock has >= {MOM_MIN_HIST} closes (short: {short[:3]})")

        # 2. no look-ahead: scramble every price after the as-of (and add a fake future bar); same record
        orig = bt.read_cached
        rng = np.random.default_rng(7)

        def scrambled(t):
            df = orig(t)
            if df is None:
                return df
            df = df.copy()
            late = df.index > pd.Timestamp(d1)
            df.loc[late, "close"] = df.loc[late, "close"] * rng.uniform(0.5, 2.0, int(late.sum()))
            fake = pd.DataFrame({c: [df[c].iloc[-1] * 3] for c in df.columns},
                                index=[df.index[-1] + pd.Timedelta(days=400)])
            return pd.concat([df, fake])
        bt.read_cached = scrambled
        try:
            a2 = mom_record(d1, tickers)
        finally:
            bt.read_cached = orig
        check(a2["mom"] == a["mom"] and a2["top"] == a["top"] and a2["excluded"] == a["excluded"],
              "momentum: no look-ahead (prices after the as-of scrambled + a fake future bar -> identical record)")
        full = bt.Panel([t for t in tickers if t in a["mom"]][:20])
        fs = mom_signal(full).loc[pd.Timestamp(d1)]
        check(all(abs(fs[t] - a["mom"][t]) < 1e-12 for t in fs.index),
              "momentum: the signal at the as-of is the same with or without later prices in the panel (causal)")

        # 3. turnover and cost by hand (1 - sum of min weights == 0.5 * sum |w2 - w1|)
        wa, wb = 1.0 / len(a["top"]), 1.0 / len(b["top"])
        to_hand = 1.0 - sum(min(wa, wb) for t in set(a["top"]) & set(b["top"]))
        overlap = len(set(a["top"]) & set(b["top"]))
        print(f"  hand turnover {d1} -> {d2}: overlap {overlap} of {len(b['top'])} -> one-way {to_hand:.6f}, "
              f"cost {to_hand * 0.002 * 1e4:.2f} bps")
        check(a["official"] and a["turnover"] == 1.0 and abs(a["cost"] - 0.0020) < 1e-15,
              f"momentum: first month turnover 1.0, cost 20 bps (got {a['turnover']}, {a['cost']})")
        check(not recs[dm]["official"] and recs[dm]["prev_official"] == d1,
              f"momentum: {dm} is informational, its previous official month is {d1}")
        check(b["official"] and b["prev_official"] == d1 and abs(b["turnover"] - to_hand) < 1e-12
              and abs(b["cost"] - to_hand * 0.0020) < 1e-15,
              f"momentum: {d2} turnover {b['turnover']:.6f} == hand {to_hand:.6f}, cost == turnover x 20 bps")

        # 4. month 1 excess by hand, straight from the price CSVs (SPY's dates as the calendar)
        sc = score(tmp, write=True)
        m = sc["momentum"]
        spy = pd.read_csv(E._cache_file("SPY"), index_col=0, parse_dates=True)["close"]
        sd = [x.date().isoformat() for x in spy.index]
        i = sd.index(d1)
        end = sd[i + 21]

        def r21(t):
            c = pd.read_csv(E._cache_file(t), index_col=0, parse_dates=True)["close"]
            c.index = c.index.normalize()
            return float(c.loc[pd.Timestamp(end)] / c.loc[pd.Timestamp(d1)] - 1.0)
        top_h = float(np.mean([r21(t) for t in a["top"]]))
        ew_h = float(np.mean([r21(t) for t in a["mom"]]))
        spy_h = float(spy.iloc[i + 21] / spy.iloc[i] - 1.0)
        xe_h, xs_h = top_h - 0.0020 - ew_h, top_h - 0.0020 - spy_h
        row = m["months"][0] if m["months"] else {}
        print(f"  hand month 1 {d1} -> {end}: top {top_h:+.6f}  EW {ew_h:+.6f}  SPY {spy_h:+.6f}  "
              f"excess vs EW {xe_h:+.6f}, vs SPY {xs_h:+.6f} (after 20 bps)")
        check(row.get("end_date") == end and abs(row.get("excess_ew", 9) - xe_h) < 2e-6
              and abs(row.get("excess_spy", 9) - xs_h) < 2e-6 and abs(row.get("ew_ret", 9) - ew_h) < 2e-6,
              f"momentum: score() month 1 == hand (end {row.get('end_date')}, excess vs EW {row.get('excess_ew')}, "
              f"vs SPY {row.get('excess_spy')})")
        check(m["official_dates"] == [d1, d2] and m["official"]["n"] == 2 and m["informational"]["n"] == 3,
              f"momentum: official months {m['official_dates']} (n={m['official']['n']}), informational n={m['informational']['n']}")
        check(m["too_small"] and m["verdict"] == "too_small" and m["official"]["t_stat_ew"] is None
              and m["first_result"] == end,
              f"momentum: too small to judge, no t-stat, first result {m['first_result']}")
        js = json.dumps(m)
        named = [t for t in tickers if f'"{t}"' in js]
        check(not named, f"momentum: the score section carries no stock names (found {named[:3]})")
        b_row = m["months"][1] if len(m["months"]) > 1 else {}
        check(abs(b_row.get("cost", 9) - round(to_hand * 0.002, 6)) < 1e-9,
              f"momentum: month 2 charged {b_row.get('cost')} (turnover x 20 bps)")

        # 5. re-running adds nothing and rewrites nothing
        before = {d: open(os.path.join(tmp, f"mom-{d}.json"), encoding="utf-8").read() for d in dates}
        r2 = mom_snapshot(tmp, dates=list(dates), tickers=tickers, say=lambda *a: None)
        after = {d: open(os.path.join(tmp, f"mom-{d}.json"), encoding="utf-8").read() for d in dates}
        check(not r2["added"] and r2["already"] == len(dates) and before == after,
              f"momentum: second run adds 0 records (added {r2['added']}, already {r2['already']}), files unchanged")
        st = status(tmp)
        check(st["momentum"]["first_result"] == end and st["momentum"]["records"] == 3,
              f"momentum: status() carries the section (records {st['momentum']['records']}, first result {st['momentum']['first_result']})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def selfcheck(asof="2026-05-01", momentum_only=False):
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ok    " if cond else "  FAIL  ") + msg)
        ok = ok and bool(cond)

    try:
        mom_selfcheck(check)
    except Exception as e:
        traceback.print_exc()
        check(False, f"momentum self-check crashed: {type(e).__name__}: {e}")
    if momentum_only:
        print("PASSED" if ok else "FAILED")
        return 0 if ok else 1

    tmp = tempfile.mkdtemp(prefix="forward-selfcheck-")
    try:
        eng = E.Engine(quiet=True)
        print("warming up the engine ...")
        eng.warm_up()
        tickers = scored_tickers()
        print(f"snapshot as of {asof} for {len(tickers)} stocks into {tmp}")
        t0 = time.time()
        r1 = snapshot(eng, asof=asof, dirpath=tmp, tickers=tickers, pause=0.0)
        print(f"  first run: +{r1['added']} cones, +{r1['skipped']} skips, {len(r1['errors'])} errors, "
              f"{time.time() - t0:.0f}s")
        for e in r1["errors"][:5]:
            print("    error:", e)
        r2 = snapshot(eng, asof=asof, dirpath=tmp, tickers=tickers, pause=0.0)
        recs = read_log(os.path.join(tmp, f"{asof}.jsonl"))
        keys = {(r["ticker"], r["horizon"]) for r in recs}
        check(r2["added"] + r2["skipped"] == 0, f"second snapshot adds 0 records (added {r2['added']}, skipped {r2['skipped']})")
        check(len(recs) == len(tickers) * 3 and len(keys) == len(recs),
              f"file has {len(recs)} records, {len(keys)} distinct, expected {len(tickers)}x3 = {len(tickers) * 3}")
        check(not r1["errors"], f"no transient errors ({len(r1['errors'])})")
        cones = [r for r in recs if "skip" not in r]
        print(f"  {len(cones)} cones, {len(recs) - len(cones)} skip records (too little history by {asof})")

        # crash recovery: cut the file mid-line, rerun, expect the same records back
        path = os.path.join(tmp, f"{asof}.jsonl")
        with open(path, encoding="utf-8") as fh:
            txt = fh.read()
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt[: len(txt) - 40])
        r3 = snapshot(eng, asof=asof, dirpath=tmp, tickers=tickers, pause=0.0)
        recs3 = read_log(path)
        check(len(recs3) == len(recs) and r3["added"] + r3["skipped"] >= 1,
              f"truncated file finished, not duplicated ({len(recs3)} records, re-added {r3['added'] + r3['skipped']})")

        # the log equals what /api/forecast serves (Engine.forecast) for that day
        for t, h in (("NVDA", 21), ("KO", 5), ("AAPL", 63)):
            rec = next((r for r in recs3 if r["ticker"] == t and r["horizon"] == h and "skip" not in r), None)
            if rec is None:
                check(False, f"{t} {h}d has no cone in the log")
                continue
            fc = eng.forecast(t, h, asof)
            f = fc["future"]
            same = (all(rec[q] == f[q][-1] for q in ("p10", "p25", "p50", "p75", "p90"))
                    and rec["p_up"] == fc["odds"]["p_up"] and rec["base_p_up"] == fc["odds"]["base_p_up"]
                    and rec["last_close"] == fc["last_close"] and rec["target_date"] == f["dates"][-1])
            check(same, f"{t} {h}d record == Engine.forecast (p10..p90, p_up, base_p_up, last close, target date)")

        # score against real prices; check three records by hand
        sc = score(tmp, write=True, keep_records=True)
        by = {(o["ticker"], o["horizon"]): o for o in sc["records"]}
        for h in E.HORIZONS:
            row = sc["horizons"][str(h)]
            print(f"  {h:>2}d: n={row['n']} days={row['days']} inside_80={row['inside_80']} inside_50={row['inside_50']} "
                  f"direction_hit={row['direction_hit']} direction_base={row['direction_base']}")
        check(sc["horizons"]["5"]["n"] == len([r for r in cones if r["horizon"] == 5]),
              "every 1W cone from that day is scored (the outcome exists)")
        for t, h in (("NVDA", 5), ("JPM", 21), ("TSLA", 63)):
            rec = next((r for r in recs3 if r["ticker"] == t and r["horizon"] == h and "skip" not in r), None)
            if rec is None:
                check(False, f"{t} {h}d has no cone in the log")
                continue
            df = pd.read_csv(E._cache_file(t), index_col=0, parse_dates=True)
            dates = [d.date().isoformat() for d in df.index]
            i = dates.index(asof)
            c0, c1 = float(df["close"].iloc[i]), float(df["close"].iloc[i + h])
            real = rec["last_close"] * c1 / c0
            in80 = rec["p10"] <= real <= rec["p90"]
            in50 = rec["p25"] <= real <= rec["p75"]
            dhit = (rec["p50"] > rec["last_close"]) == (c1 > c0)
            dbase = (rec["base_p_up"] >= 0.5) == (c1 > c0)
            print(f"  hand {t} {h}d: asof close {c0:.4f} -> {dates[i + h]} close {c1:.4f} (logged last {rec['last_close']}, "
                  f"real {real:.4f}); cone p10 {rec['p10']} p25 {rec['p25']} p50 {rec['p50']} p75 {rec['p75']} p90 {rec['p90']}; "
                  f"base_p_up {rec['base_p_up']} -> in80 {in80} in50 {in50} dir {dhit} base {dbase}")
            o = by.get((t, h))
            check(o is not None and (o["inside_80"], o["inside_50"], o["direction_hit"], o["direction_base"])
                  == (in80, in50, dhit, dbase) and o["end_date"] == dates[i + h],
                  f"score() agrees with the hand computation for {t} {h}d")
        st = status(tmp)
        check(st["days_logged"] == 1 and st["since"] == asof, f"status: since {st['since']}, days_logged {st['days_logged']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--momentum-only", action="store_true", help="selfcheck only: just the momentum tilt part")
    ap.add_argument("--asof", help="selfcheck only: the past day to snapshot (default 2026-05-01)")
    a = ap.parse_args(argv)
    if a.selfcheck:
        return selfcheck(a.asof or "2026-05-01", momentum_only=a.momentum_only)
    if a.snapshot:
        eng = E.Engine(quiet=True)
        eng.warm_up()
        run_once(eng)        # snapshot + score, state recorded like the server's thread
        print(json.dumps(status(), indent=1))
        return 1 if STATE["last_error"] else 0
    if a.score:
        print(json.dumps(score(), indent=1))
        return 0
    if a.status:
        print(json.dumps(status(), indent=1))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
