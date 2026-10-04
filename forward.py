"""Stock Cone live test: log the exact cone the app serves each day, score it later.

Every backtest number in this repo comes from today's stock list looking backwards
(survivorship bias). The only clean evidence is a forward log: write down the cone the
app served on a day, wait for the real price, then score it. Nothing in a logged record
can be changed by later data.

    python forward.py --snapshot    log today's cones (155 scored stocks x 1W/1M/3M)
    python forward.py --score       score every logged record whose outcome exists
    python forward.py --status      print status + per-horizon summary (the /api/forward body)
    python forward.py --selfcheck   snapshot a PAST date into a temp dir, score it, check it

Files (data/forward/):
    <asof>.jsonl    one JSON record per (ticker, horizon); appended line by line, so a crash
                    leaves a partial file that the next run finishes (never duplicates)
    scores.json     the per-horizon summary written by score()
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
    if write:
        _write_atomic(os.path.join(dirpath, "scores.json"), out)
    if keep_records:
        out["records"] = recs_out
    return out


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
        STATE["phase"] = "score"
        try:
            sc = score(dirpath)
            STATE["last_score"] = sc["scored_at"]
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

def selfcheck(asof="2026-05-01"):
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ok    " if cond else "  FAIL  ") + msg)
        ok = ok and bool(cond)

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
    ap.add_argument("--asof", help="selfcheck only: the past day to snapshot (default 2026-05-01)")
    a = ap.parse_args(argv)
    if a.selfcheck:
        return selfcheck(a.asof or "2026-05-01")
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
