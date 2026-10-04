"""Accuracy scoreboard: the walk-forward track record for ~150 individual stocks.

    python accuracy.py                 build data/accuracy.json (resumes a crashed build)
    python accuracy.py --fresh         ignore earlier partial results
    python accuracy.py --only NVDA,KO  score just these, print, write nothing

Per stock and horizon the numbers come from Engine.get_track(), the exact function
and cache behind the chart page's `track`, so clicking through never disagrees.
Since 2026-10-04 that is the v2 cone (cone_model.py) and a walk-forward direction
baseline (track cache key suffix engine.TRACK_VERSION); see CONTRACT.md.
avg_width_80 comes from the same walk-forward pass (Engine.track_width) and is
never part of `track`. Downloads are sequential and cached like the rest of the app.
The file is written atomically after every stock, so a crash loses at most one.
See CONTRACT.md "GET /api/accuracy".
"""

import datetime as dt
import json
import os
import sys
import threading
import time

import engine as E
from universe import GROUPS, UNIVERSE

OUT_PATH = os.path.join(E.DATA_DIR, "accuracy.json")
LOCK_PATH = os.path.join(E.DATA_DIR, "accuracy.lock")   # cross-process "a build is running" marker
MAX_AGE = dt.timedelta(days=1)
MIN_TESTS = 30          # fewer 1-week tests than this: too few to score, skipped with the count
LOCK_STALE_S = 15 * 60  # a lock not touched for this long belongs to a dead build
TRACK_WAIT_S = 30

NOTES = [
    "Every number is a walk-forward test: about 150 past dates per stock, each forecast "
    "using only the data up to that date, then checked against what really happened.",
    "With about 150 tests per stock, a single stock's numbers can wobble by several points "
    "from luck alone. On direction, a gap of roughly 6 to 8 points either way from the "
    "base rate is noise, not skill. The pooled summary over all stocks is much steadier.",
    "The 3-month tests overlap in time: the ~150 test dates are spread over each stock's "
    "history, so for all but the oldest stocks a new 63-day test starts before the last "
    "one ends, and for young stocks only days apart. Overlapping tests share much of their "
    "outcome, so they count for less than their raw number suggests (1-month tests too, "
    "to a lesser degree).",
    "Survivorship bias: this list is stocks that exist and trade today. Companies that "
    "went bust or were delisted are missing, which flatters any method that leans bullish.",
    "The direction baseline is \"guess the stock's usual direction so far\": on each test "
    "date, guess whichever way the stock moved more often before that date.",
    "If the bands are honest, about 80% of real outcomes land inside the shaded cone and 50% inside the darker middle. "
    "A stock counts as on target when its cone hit is between 72% and 88%.",
]


class Run:
    """In-process build state, read by the server while a build runs."""

    def __init__(self):
        self.thread = None
        self.lock = threading.Lock()
        self.result = None      # latest snapshot dict (replaced, never mutated)
        self.progress = None    # {"done", "total"} of the build running in this process
        self.error = None


RUN = Run()


def now_iso():
    return dt.datetime.now().replace(microsecond=0).isoformat()


def write_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# ----- cross-process lock (so the server and the CLI never build at once) -----

def lock_held_elsewhere():
    """True if another process's build touched the lock file recently."""
    info = read_json(LOCK_PATH)
    if not info or info.get("pid") == os.getpid():
        return False
    try:
        age = time.time() - os.path.getmtime(LOCK_PATH)
    except OSError:
        return False
    return age < LOCK_STALE_S


def touch_lock(done, total):
    write_atomic(LOCK_PATH, {"pid": os.getpid(), "done": done, "total": total, "at": now_iso()})


def release_lock():
    info = read_json(LOCK_PATH)
    if info and info.get("pid") == os.getpid():
        try:
            os.remove(LOCK_PATH)
        except OSError:
            pass


# ----- scoring ----------------------------------------------------------------

def is_fresh(data):
    """A complete build generated less than a day ago."""
    if not data or data.get("status") != "ready" or not data.get("generated"):
        return False
    try:
        gen = dt.datetime.fromisoformat(data["generated"])
    except ValueError:
        return False
    return dt.datetime.now() - gen < MAX_AGE


def summarise(stocks):
    """Pool over every forecast per horizon. Rates are stored rounded to 4
    places with n <= ~150, so round(n * rate) recovers the exact counts."""
    out = {}
    for h in E.HORIZONS:
        k = str(h)
        rows = [s["h"][k] for s in stocks if s["h"].get(k)]
        n = sum(r["n"] for r in rows)
        if not n:
            out[k] = {"stocks": 0, "forecasts": 0}
            continue

        def pooled(field):
            return E.r4(sum(round(r["n"] * r[field]) for r in rows) / n)

        widths = [(r["n"], r["avg_width_80"]) for r in rows if r.get("avg_width_80") is not None]
        wn = sum(a for a, _ in widths)
        out[k] = {
            "stocks": len(rows),
            "forecasts": n,
            "inside_80": pooled("inside_80"),
            "inside_50": pooled("inside_50"),
            "direction_hit": pooled("direction_hit"),
            "direction_base": pooled("direction_base"),
            "beat_base": sum(1 for r in rows if r["direction_hit"] > r["direction_base"]),
            "calibrated": sum(1 for r in rows if 0.72 <= r["inside_80"] <= 0.88),
            "avg_width_80": E.r4(sum(a * b for a, b in widths) / wn) if wn else None,
        }
    return out


def snapshot(status, stocks, skipped, total, generated):
    return {
        "status": status,
        "progress": {"done": len(stocks) + len(skipped), "total": total},
        "generated": generated,
        "horizons": list(E.HORIZONS),
        "groups": GROUPS,
        "summary": summarise(stocks),
        "stocks": stocks,
        "skipped": skipped,
        "notes": NOTES,
    }


def _track_for(eng, t, key):
    """All three horizons via get_track (same function + cache as the chart).
    Returns ({h: track or None}, error or None)."""
    while True:
        eng.get_track(t, E.HORIZONS[0], wait=TRACK_WAIT_S)
        if key in eng.track or key not in eng.track_threads:
            break
    if key not in eng.track:
        # get_track logs failures but does not return them; run once more to get the reason
        try:
            res = eng.compute_track(t)
        except Exception as e:
            return None, f"track record could not be computed: {type(e).__name__}: {e}"
        eng.track[key] = res
        eng.save_track()
    return {str(h): eng.get_track(t, h) for h in E.HORIZONS}, None


def score_stock(eng, t, groups, prior=None):
    """One universe entry -> (stock dict, None) or (None, skip reason)."""
    try:
        td = eng.get_ticker(t)
    except E.ForecastError as e:
        return None, e.msg
    key = eng._track_key(t)
    if prior and prior.get("track_key") == key and prior.get("groups") == groups:
        return prior, None                       # resume: same data, already scored
    tracks, err = _track_for(eng, t, key)
    if err:
        return None, err
    t5 = tracks.get("5")
    if not t5 or t5["n"] < MIN_TESTS:
        n = t5["n"] if t5 else 0
        return None, (f"only {n} past dates could be tested (history starts "
                      f"{E.day_str(td.day[0])}); needs about {MIN_TESTS} to score")
    widths = eng.track_width.get(key)
    if widths is None:
        # the track was cached before this process computed it, so the width is not
        # in memory: recompute the records once, and prove they score the same
        recs = eng.track_records(t)
        widths = {str(h): eng.avg_width_80(recs[h]) for h in E.HORIZONS}
        for h in E.HORIZONS:
            fresh = eng.score(recs[h])
            if fresh != tracks[str(h)]:
                E.log(f"accuracy: {t} h{h} cached track {tracks[str(h)]} != recomputed {fresh}")
        eng.track_width[key] = widths
    h = {}
    for k, tr in tracks.items():
        h[k] = None if tr is None else {**tr, "avg_width_80": widths.get(k)}
    return {
        "ticker": t,
        "name": eng.get_name(t, wait=8.0),
        "groups": groups,
        "first_date": E.day_str(td.day[0]),
        "last_close": E.sig6(td.close[-1]),
        "track_key": key,
        "h": h,
    }, None


def build(eng, fresh=False, only=None, write=True, say=E.log):
    """Score the universe sequentially. Writes partial results after every stock
    (status "building") and the final file (status "ready")."""
    universe = [(t, g) for t, g in UNIVERSE if not only or t in only]
    total = len(universe)
    old = None if fresh else read_json(OUT_PATH)
    generated = old.get("generated") if old else None
    prior = {s["ticker"]: s for s in (old or {}).get("stocks", [])}
    stocks, skipped = [], []
    t0 = time.time()
    try:
        for i, (t, groups) in enumerate(universe):
            RUN.progress = {"done": i, "total": total}
            if write:
                touch_lock(i, total)
            ts = time.time()
            try:
                s, why = score_stock(eng, t, groups, prior.get(t))
            except Exception as e:      # one bad stock never kills the build
                s, why = None, f"error: {type(e).__name__}: {e}"
            if s is not None:
                stocks.append(s)
                with eng.lock:          # free the per-ticker analog pool (rebuilt on demand)
                    if t not in E.POOL:
                        eng.pools.pop(t, None)
            else:
                skipped.append({"ticker": t, "reason": why})
            say(f"accuracy {i + 1}/{total} {t}: "
                f"{'skipped: ' + why if s is None else s['h'].get('5')} ({time.time() - ts:.1f}s)")
            snap = snapshot("building", stocks, skipped, total, generated)
            RUN.result = snap
            if write:
                write_atomic(OUT_PATH, snap)
        final = snapshot("ready", stocks, skipped, total, now_iso())
        RUN.result = final
        if write:
            write_atomic(OUT_PATH, final)
        say(f"accuracy build done: {len(stocks)} scored, {len(skipped)} skipped, {time.time() - t0:.0f}s")
        return final
    finally:
        if write:
            release_lock()


# ----- server hook --------------------------------------------------------------

def start_background(eng):
    """Start ONE background rebuild in this process if none is running here or in
    another process. Returns True if a build is running (started now or before)."""
    with RUN.lock:
        if RUN.thread is not None and RUN.thread.is_alive():
            return True
        if lock_held_elsewhere():
            return True

        def run():
            try:
                while not eng.ready:            # the analog pool must be loaded first
                    time.sleep(2)
                RUN.error = None
                build(eng)
            except Exception as e:
                RUN.error = f"{type(e).__name__}: {e}"
                E.log(f"accuracy build failed: {RUN.error}")

        RUN.result = None
        RUN.progress = {"done": 0, "total": len(UNIVERSE)}
        RUN.thread = threading.Thread(target=run, daemon=True, name="accuracy-build")
        RUN.thread.start()
        return True


def payload(eng):
    """What GET /api/accuracy returns. Never blocks on the build."""
    on_disk = read_json(OUT_PATH)
    if is_fresh(on_disk) and not (RUN.thread and RUN.thread.is_alive()):
        return on_disk
    start_background(eng)
    running_here = RUN.thread is not None and RUN.thread.is_alive()
    data = RUN.result if running_here and RUN.result else on_disk
    if data is None:
        data = snapshot("building", [], [], len(UNIVERSE), None)
    data = dict(data, status="building")
    notes = list(data.get("notes") or NOTES)
    if running_here and RUN.progress:
        data["progress"] = dict(RUN.progress, done=max(RUN.progress["done"],
                                                         data["progress"]["done"] if RUN.result else 0))
    elif not running_here and lock_held_elsewhere():
        info = read_json(LOCK_PATH) or {}
        notes.insert(0, f"A build is running in another process (pid {info.get('pid')}); "
                        "showing its last saved progress.")
        if info.get("total"):
            data["progress"] = {"done": info.get("done", 0), "total": info["total"]}
    if running_here and not eng.ready:
        notes.insert(0, "Waiting for the analog pool to finish loading before scoring starts.")
    if RUN.error:
        notes.insert(0, f"The last build failed: {RUN.error}. Retrying.")
    data["notes"] = notes
    return data


# ----- CLI ------------------------------------------------------------------------

def main(argv):
    fresh = "--fresh" in argv
    only = None
    if "--only" in argv:
        only = {E.normalise_ticker(x) for x in argv[argv.index("--only") + 1].split(",") if x}
    write = only is None
    if write and lock_held_elsewhere():
        print("Another accuracy build is running (see data/accuracy.lock); not starting a second.")
        return 1
    eng = E.Engine()
    eng.warm_up()
    t0 = time.time()
    out = build(eng, fresh=fresh, only=only, write=write)
    print(json.dumps(out["summary"], indent=1))
    if out["skipped"]:
        print("skipped:", "; ".join(f"{s['ticker']} ({s['reason']})" for s in out["skipped"]))
    print(f"{time.time() - t0:.0f}s" + (f", wrote {OUT_PATH}" if write else " (nothing written: --only)"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
