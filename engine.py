"""Stock Cone engine: price cache, features, analog search, cone, track record.

The idea in one paragraph: describe "the setup" on any day with a dozen numbers
(recent moves, distance from moving averages, drawdown, volatility regime,
volume, market regime), all computed only from bars up to that day and all
expressed in units of the stock's own volatility so a calm stock and a wild one
are comparable. To forecast ticker T as of day D, find the K past days (across
T's own history and a fixed basket of large caps) whose setup looked most like
T's setup on D, take what actually happened to each of them over the next
`horizon` trading days, rescale each path by (T's vol now / analog's vol then),
and read off per-day percentiles. That's the cone.

Run `python engine.py --selfcheck` to prove there is no look-ahead.
"""

import datetime as dt
import json
import math
import os
import re
import sys
import threading
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
PRICE_DIR = os.path.join(DATA_DIR, "prices")
NAMES_PATH = os.path.join(DATA_DIR, "names.json")
TRACK_PATH = os.path.join(DATA_DIR, "track_cache.json")

# ---------------------------------------------------------------------------
# Parameters (chosen a priori; see README for the reasoning)
# ---------------------------------------------------------------------------

# Fixed analog basket: liquid US large caps across sectors, most with long
# histories. Chosen today, so it is a survivor list (see README "Limits").
BASKET = [
    # tech / comms
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "ORCL", "CSCO", "INTC", "IBM", "AMD", "QCOM",
    # financials
    "JPM", "BAC", "WFC", "GS", "AXP", "BRK-B",
    # health care
    "JNJ", "PFE", "MRK", "UNH", "LLY", "AMGN",
    # consumer
    "WMT", "KO", "PEP", "PG", "MCD", "HD", "COST", "DIS", "NKE",
    # energy
    "XOM", "CVX",
    # industrials
    "CAT", "BA", "GE", "HON", "UNP",
    # utilities / telecom
    "NEE", "VZ",
]
INDEX_ETFS = ["SPY", "QQQ"]
POOL = BASKET + INDEX_ETFS
# Market regime comes from SPY; before SPY existed (Jan 1993) we splice in the
# S&P 500 index (^GSPC) so older analogs still have a regime. It is a feature
# source only, never an analog.
MARKET_SPLICE = "^GSPC"

HORIZONS = (5, 21, 63)
K = 250                 # analogs per forecast
K_MIN = 100             # below this many usable analogs we refuse to forecast
BLOCK = 5               # de-clustering: at most one analog per ticker per 5-bar block
CANDIDATE_MULT = 8      # look at K*8 nearest before de-clustering
PRE = 24000             # walk-forward shortcut: pre-select this many nearest rows per date
SCALE_CLIP = (0.25, 4.0)  # clip on (vol now / analog vol)
QS = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
Z_QS = np.array([-1.2815515655, -0.6744897502, 0.0, 0.6744897502, 1.2815515655])
HISTORY_YEARS = 5
TRACK_N = 150
TRACK_THREADS = 2       # threads for a first-request track computation
MIN_BARS = 260          # ~1 year: features need 252 bars of warm-up

FEATURES = [
    "z5", "z21", "z63", "z126",   # trailing log return / (daily vol * sqrt(window))
    "ma50", "ma200",              # log(close / SMA) / daily vol
    "dd252",                      # log(close / 52-week high) / daily vol
    "volreg",                     # log(vol21 / vol252): is volatility expanding or calm?
    "volu",                       # log(avg volume 5d / avg volume 63d)
    "mkt_z63",                    # market trailing 63d move in vol units
    "mkt_vol",                    # log(market annualised 21d vol)
    "mkt_dd",                     # market log drawdown from its 52-week high
]
# Robust scale (IQR / 1.349) of each raw feature, measured once on the basket's
# rows dated before 2010-01-01 and then frozen here, so standardisation never
# depends on the data being forecast (a full-history mean/std would leak).
# Centering is irrelevant for Euclidean distance, so only scales are stored.
# Regenerate with `python engine.py --fit-scales` (it only prints).
FEATURE_SCALE = {
    "z5": 0.9427, "z21": 0.9567, "z63": 0.9842, "z126": 1.0271,
    "ma50": 3.8332, "ma200": 8.1769, "dd252": 6.8850, "volreg": 0.3007,
    "volu": 0.3015, "mkt_z63": 1.0058, "mkt_vol": 0.4433, "mkt_dd": 0.0734,
}
FEATURE_WEIGHT = {f: 1.0 for f in FEATURES}
FEATURE_CLIP = 4.0      # clip standardised features to +-4 (bad ticks, splits)

NY = ZoneInfo("America/New_York")
BIG_DAY = np.iinfo(np.int64).max


class ForecastError(Exception):
    """An error with an HTTP status, shown to the user as {"error": msg}."""

    def __init__(self, status, msg):
        super().__init__(msg)
        self.status = status
        self.msg = msg


def log(msg):
    print(f"[engine {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data: yfinance download + on-disk cache (refetched at most once per day)
# ---------------------------------------------------------------------------

def normalise_ticker(raw):
    t = (raw or "").strip().upper().replace(".", "-")
    if not t or not re.fullmatch(r"\^?[A-Z0-9\-=]{1,15}", t):
        raise ForecastError(400, f"'{raw}' is not a valid ticker symbol.")
    return t


def _cache_file(ticker):
    return os.path.join(PRICE_DIR, re.sub(r"[^A-Z0-9\-=]", "_", ticker) + ".csv")


def _flatten(df, ticker):
    """yfinance returns (Price, Ticker) MultiIndex columns; reduce to one level."""
    if isinstance(df.columns, pd.MultiIndex):
        names = list(df.columns.names)
        if "Ticker" in names and ticker in df.columns.get_level_values("Ticker"):
            df = df.xs(ticker, axis=1, level="Ticker")
        else:
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
    return df


def _clean(close, volume, index):
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:                     # tz-aware -> local exchange date
        idx = idx.tz_convert(NY).tz_localize(None)
    idx = idx.normalize()
    out = pd.DataFrame({"close": np.asarray(close, float), "volume": np.asarray(volume, float)}, index=idx)
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[np.isfinite(out["close"]) & (out["close"] > 0)]
    return out


def _drop_partial_bar(df):
    """During the US session Yahoo includes today's unfinished bar; drop it."""
    now = dt.datetime.now(NY)
    if len(df) and df.index[-1].date() == now.date() and (now.hour, now.minute) < (16, 15):
        return df.iloc[:-1]
    return df


def download(ticker):
    import yfinance as yf
    import yfinance.shared as yshared

    try:
        df = yf.download(ticker, period="max", auto_adjust=True, actions=False,
                         progress=False, threads=False, timeout=20)
    except Exception as e:  # yfinance mostly swallows errors, but not always
        raise ForecastError(503, f"Yahoo Finance unreachable while fetching {ticker}: {e}")
    if df is None or df.empty:
        msg = str(getattr(yshared, "_ERRORS", {}).get(ticker, "")).strip()
        notfound = ("delisted", "not found", "no timezone", "no data", "404", "no price")
        if not msg or any(s in msg.lower() for s in notfound):
            raise ForecastError(404, f"Unknown ticker '{ticker}': Yahoo Finance has no price data for it.")
        raise ForecastError(503, f"Yahoo Finance error for {ticker}: {msg}")
    df = _flatten(df, ticker)
    if "Close" not in df.columns:
        raise ForecastError(502, f"Unexpected Yahoo response for {ticker} (no Close column).")
    vol = df["Volume"] if "Volume" in df.columns else np.zeros(len(df))
    return _drop_partial_bar(_clean(df["Close"], vol, df.index))


def load_prices(ticker):
    """Return (frame, note). Uses the disk cache if it was written today; else
    refetches. If Yahoo fails and a stale cache exists, use it and say so."""
    os.makedirs(PRICE_DIR, exist_ok=True)
    path = _cache_file(ticker)
    fresh = os.path.exists(path) and dt.date.fromtimestamp(os.path.getmtime(path)) == dt.date.today()
    if not fresh:
        try:
            df = download(ticker)
            df.to_csv(path, float_format="%.10g")
            return df, None
        except ForecastError as e:
            if not os.path.exists(path) or e.status == 404:
                raise
            note = f"Yahoo refresh failed for {ticker} ({e.msg}); using cached data."
            log(note)
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return _clean(df["close"], df["volume"], df.index), note
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _clean(df["close"], df["volume"], df.index), None


# ---------------------------------------------------------------------------
# Features (all causal: the value at row t uses only rows <= t)
# ---------------------------------------------------------------------------

def market_frame(spy, splice):
    """Market regime features from SPY, extended before 1993 with ^GSPC returns."""
    r_spy = np.log(spy["close"]).diff()
    r_old = np.log(splice["close"]).diff()
    if len(spy):
        first = spy.index[0]
        r = pd.concat([r_old[r_old.index <= first], r_spy[r_spy.index > first]])
    else:
        r = r_old
    r = r[~r.index.duplicated(keep="first")].fillna(0.0)
    lvl = r.cumsum()
    s21 = r.rolling(21).std()
    s63 = r.rolling(63).std()
    return pd.DataFrame({
        "mkt_z63": (lvl - lvl.shift(63)) / (s63 * math.sqrt(63)),
        "mkt_vol": np.log(s21 * math.sqrt(252)),
        "mkt_dd": lvl - lvl.rolling(252).max(),
    })


def raw_features(px, mkt):
    """DataFrame of the raw (unscaled) features plus sig63, indexed like px."""
    close = px["close"]
    lc = np.log(close)
    lr = lc.diff()
    s21 = lr.rolling(21).std()
    s63 = lr.rolling(63).std()
    s252 = lr.rolling(252).std()
    f = pd.DataFrame(index=px.index)
    for k in (5, 21, 63, 126):
        f[f"z{k}"] = (lc - lc.shift(k)) / (s63 * math.sqrt(k))
    f["ma50"] = np.log(close / close.rolling(50).mean()) / s63
    f["ma200"] = np.log(close / close.rolling(200).mean()) / s63
    f["dd252"] = np.log(close / close.rolling(252).max()) / s63
    f["volreg"] = np.log(s21 / s252)
    vol = px["volume"].where(px["volume"] > 0)
    f["volu"] = np.log(vol.rolling(5, min_periods=3).mean() / vol.rolling(63, min_periods=40).mean())
    m = mkt.reindex(px.index, method="ffill")   # market as of each bar date (<= t)
    for c in ("mkt_z63", "mkt_vol", "mkt_dd"):
        f[c] = m[c]
    f = f.replace([np.inf, -np.inf], np.nan)
    f["volu"] = f["volu"].fillna(0.0)           # missing volume = neutral, not a dropped row
    f["sig63"] = s63
    return f


class TickerData:
    """Everything the engine needs about one ticker, as numpy arrays."""

    def __init__(self, ticker, px, mkt, note=None):
        self.ticker = ticker
        self.px = px
        self.note = note
        self.loaded_on = dt.date.today()
        self.dates = px.index
        self.day = (px.index.values.astype("datetime64[D]").astype(np.int64))  # days since epoch
        self.close = px["close"].to_numpy(float)
        self.logc = np.log(self.close)
        f = raw_features(px, mkt)
        raw = f[FEATURES].to_numpy(float)
        scale = np.array([FEATURE_SCALE[c] for c in FEATURES])
        wt = np.sqrt(np.array([FEATURE_WEIGHT[c] for c in FEATURES]))
        X = np.clip(raw / scale, -FEATURE_CLIP, FEATURE_CLIP) * wt
        self.sig = f["sig63"].to_numpy(float)
        self.valid = np.isfinite(X).all(axis=1) & np.isfinite(self.sig) & (self.sig > 0)
        self.valid[: MIN_BARS - 1] = False
        # Old Yahoo data has stretches of stale quotes (30%+ unchanged days for
        # some 1970s names). Such rows can be queried but never serve as analogs:
        # their vol is understated and their forward paths are full of zeros.
        flat = pd.Series(np.r_[np.nan, np.diff(np.log(px["close"].to_numpy(float))) == 0], index=px.index)
        stale = (flat.rolling(63).mean() > 0.20).to_numpy()
        self.pool_ok = self.valid & ~stale
        self.X = X
        self.raw = f

    def pos_on_or_before(self, day):
        """Index of the last bar dated <= day (days since epoch), or -1."""
        return int(np.searchsorted(self.day, day, side="right")) - 1


class Pool:
    """Concatenated analog rows from a list of tickers (fixed order)."""

    def __init__(self, tds):
        # stable ticker ids (position in POOL; the target gets len(POOL)) so tie-breaks
        # and de-clustering blocks do not depend on which tickers had data yet
        self.tickers = {(POOL.index(td.ticker) if td.ticker in POOL else len(POOL)): td.ticker for td in tds}
        X, gpos, tid, local, day, sig, logc = [], [], [], [], [], [], []
        ends = {h: [] for h in HORIZONS}
        offset = 0
        for td in tds:
            i = POOL.index(td.ticker) if td.ticker in POOL else len(POOL)
            idx = np.flatnonzero(td.pool_ok)
            n = len(td.day)
            X.append(td.X[idx])
            gpos.append(offset + idx)
            tid.append(np.full(len(idx), i, np.int64))
            local.append(idx)
            day.append(td.day[idx])
            sig.append(td.sig[idx])
            for h in HORIZONS:
                e = np.full(len(idx), BIG_DAY, np.int64)
                ok = idx + h < n
                e[ok] = td.day[idx[ok] + h]          # date of the analog's horizon end
                ends[h].append(e)
            logc.append(td.logc)
            offset += n
        X = np.concatenate(X) if X else np.zeros((0, len(FEATURES)))
        # (F, N) float32: per-feature elementwise ops are fast and bit-deterministic
        # (no BLAS reductions whose rounding could depend on array size)
        self.XT = np.ascontiguousarray(X.T, dtype=np.float32)
        self.gpos = np.concatenate(gpos)
        self.tid = np.concatenate(tid)
        self.local = np.concatenate(local)
        self.day = np.concatenate(day)
        self.sig = np.concatenate(sig)
        self.end = {h: np.concatenate(ends[h]) for h in HORIZONS}
        self.logc = np.concatenate(logc)
        self.block = self.tid * 10_000_000 + self.local // BLOCK
        self.key = self.tid * 10_000_000 + self.local  # deterministic tie-break
        self.n = len(self.gpos)

    def dist2(self, q):
        """Squared Euclidean distance from setup q to every pool row."""
        q = np.asarray(q, dtype=np.float32)
        d = np.zeros(self.n, np.float32)
        tmp = np.empty(self.n, np.float32)
        for f in range(self.XT.shape[0]):
            np.subtract(self.XT[f], q[f], out=tmp)
            np.multiply(tmp, tmp, out=tmp)
            np.add(d, tmp, out=d)
        return d


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def wquantile(v, w, qs):
    """Weighted quantiles of 1-D v (midpoint rule, linear interpolation)."""
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    p = (cw - 0.5 * w) / cw[-1]
    return np.interp(qs, p, v)


def nyse_holidays(year):
    def nth(month, weekday, n):
        d = dt.date(year, month, 1)
        d += dt.timedelta(days=(weekday - d.weekday()) % 7)
        return d + dt.timedelta(weeks=n - 1)

    def last(month, weekday):
        d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
        return d - dt.timedelta(days=(d.weekday() - weekday) % 7)

    def observed(d):
        return d - dt.timedelta(days=1) if d.weekday() == 5 else d + dt.timedelta(days=1) if d.weekday() == 6 else d

    a, b, c = year % 19, year // 100, year % 100       # anonymous Gregorian Easter
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l_) // 433
    month = (h + l_ - 7 * m + 90) // 25
    easter = dt.date(year, month, (h + l_ - 7 * m + 33 * month + 19) % 32)
    hol = {observed(dt.date(year, 1, 1)), nth(1, 0, 3), nth(2, 0, 3), easter - dt.timedelta(days=2),
           last(5, 0), observed(dt.date(year, 7, 4)), nth(9, 0, 1), nth(11, 3, 4),
           observed(dt.date(year, 12, 25))}
    if year >= 2022:
        hol.add(observed(dt.date(year, 6, 19)))
    return {x for x in hol if x.year == year}


def project_trading_days(after, n):
    out, d, cache = [], after, {}
    while len(out) < n:
        d += dt.timedelta(days=1)
        if d.weekday() >= 5:
            continue
        if d.year not in cache:
            cache[d.year] = nyse_holidays(d.year)
        if d in cache[d.year]:
            continue
        out.append(d)
    return out


def sig6(x):
    return float(f"{x:.6g}")


def r4(x):
    return round(float(x), 4)


def day_str(day):
    return str(np.datetime64(int(day), "D"))


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, raw=None, quiet=False):
        """raw: optional {ticker: price frame} to run fully offline (self-check)."""
        self.raw = raw
        self.quiet = quiet
        self.lock = threading.RLock()
        self.tick = {}
        self.pools = {}
        self.mkt = None
        self.ready = False
        self.progress = (0, len(POOL) + 2)
        self.status = "not started"
        self.loaded_on = None
        self.refreshing = False
        self.names = {}
        self.name_threads = {}
        self.name_failed = set()
        self.track = {}
        self.track_threads = {}
        self.version = 0
        if raw is None:
            self._load_json_caches()

    # ----- startup / refresh -------------------------------------------------

    def _load_json_caches(self):
        for path, attr in ((NAMES_PATH, "names"), (TRACK_PATH, "track")):
            try:
                with open(path, encoding="utf-8") as fh:
                    setattr(self, attr, json.load(fh))
            except (OSError, ValueError):
                pass

    def _save_json(self, path, obj):
        if self.raw is not None:
            return
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
        os.replace(tmp, path)

    def _prices(self, t):
        if self.raw is not None:
            if t not in self.raw:
                raise ForecastError(404, f"Unknown ticker '{t}' (offline mode).")
            return self.raw[t], None
        return load_prices(t)

    def warm_up(self):
        """Load (downloading if needed) the market series and the analog pool.
        Sequential on purpose: one request at a time to Yahoo."""
        self.status = "loading market series"
        total = len(POOL) + 2
        spy, _ = self._prices("SPY")
        self.progress = (1, total)
        try:
            spl, _ = self._prices(MARKET_SPLICE)
        except ForecastError as e:
            log(f"^GSPC unavailable ({e.msg}); market regime starts in 1993")
            spl = spy.iloc[:0]
        self.progress = (2, total)
        mkt = market_frame(spy, spl)
        tick = {}
        for i, t in enumerate(POOL):
            self.status = f"loading analog pool ({t})"
            try:
                px, note = self._prices(t)
                if len(px) >= MIN_BARS:            # shorter series have no usable rows yet
                    tick[t] = TickerData(t, px, mkt, note)
            except ForecastError as e:
                log(f"pool ticker {t} skipped: {e.msg}")
            self.progress = (i + 3, total)
            if not self.quiet and self.raw is None:
                log(f"pool {i + 1}/{len(POOL)} {t}: {len(tick[t].day) if t in tick else 0} bars")
        with self.lock:
            self.mkt = mkt
            self.tick = tick
            self.pools = {}
            self.loaded_on = dt.date.today()
            self.version += 1
            self.ready = True
            self.status = "ready"
        if not self.quiet:
            n = sum(int(td.pool_ok.sum()) for td in tick.values())
            log(f"analog pool ready: {len(tick)} tickers, {n:,} usable setups")

    def maybe_refresh(self):
        """Once a day, refresh the pool in the background; keep serving meanwhile."""
        if self.raw is not None or self.refreshing or self.loaded_on == dt.date.today():
            return
        self.refreshing = True

        def run():
            try:
                log("daily refresh of the analog pool")
                self.warm_up()
            except Exception as e:  # keep the old pool; say why
                log(f"daily refresh failed: {e}")
            finally:
                self.refreshing = False

        threading.Thread(target=run, daemon=True).start()

    # ----- per-ticker data ---------------------------------------------------

    def get_ticker(self, t):
        with self.lock:
            td = self.tick.get(t)
            if td is not None and (self.raw is not None or td.loaded_on == dt.date.today()):
                return td
        px, note = self._prices(t)
        if len(px) < MIN_BARS + 5:
            raise ForecastError(422, f"{t} has only {len(px)} trading days of history; the cone needs about a year.")
        td = TickerData(t, px, self.mkt, note)
        with self.lock:
            if t in POOL and t in self.tick:
                # a pool member refreshed today: replace it and drop pools built on the old copy
                self.tick[t] = td
                self.pools = {}
            elif t not in POOL:
                self.tick[t] = td
                self.pools.pop(t, None)
        return td

    def pool_for(self, t):
        key = t if t not in POOL else "__basket__"
        with self.lock:
            p = self.pools.get(key)
            if p is None:
                tds = [self.tick[x] for x in POOL if x in self.tick]
                if t not in POOL:
                    tds.append(self.tick[t])
                p = Pool(tds)
                if len(self.pools) > 12:
                    self.pools = {k: v for k, v in self.pools.items() if k == "__basket__"}
                self.pools[key] = p
        return p

    def get_name(self, t, wait=0.0):
        """Company name (cached on disk). The lookup runs in a background thread;
        callers wait at most `wait` seconds and fall back to the ticker."""
        if t in self.names or self.raw is not None or t in self.name_failed:
            return self.names.get(t, t)
        with self.lock:
            th = self.name_threads.get(t)
            if th is None:
                def fetch():
                    try:
                        import yfinance as yf
                        info = yf.Ticker(t).info or {}
                        name = info.get("longName") or info.get("shortName")
                        if name:
                            self.names[t] = name
                            self._save_json(NAMES_PATH, self.names)
                        else:
                            self.name_failed.add(t)
                    except Exception as e:
                        log(f"name lookup for {t} failed: {e}")
                        self.name_failed.add(t)   # don't retry this session; the ticker is shown instead
                    finally:
                        self.name_threads.pop(t, None)

                th = threading.Thread(target=fetch, daemon=True)
                self.name_threads[t] = th
                th.start()
        if wait > 0:
            th.join(wait)
        return self.names.get(t, t)

    # ----- analog search + cone ----------------------------------------------

    def _select(self, pool, q, d2_all, D_day, h, pre=None):
        """Indices of the chosen analogs (nearest first), their distances, weights.

        pre (optional): the PRE nearest rows ignoring eligibility. If at least
        K*CANDIDATE_MULT of them are eligible, the K*CANDIDATE_MULT nearest
        eligible rows are guaranteed to be among them, so the result is the
        same as the full scan, just cheaper. Otherwise fall back to the full scan."""
        elig = None
        if pre is not None:
            e = pre[pool.end[h][pre] <= D_day]
            if e.size >= K * CANDIDATE_MULT:
                elig = e
        if elig is None:
            elig = np.flatnonzero(pool.end[h] <= D_day)      # A + horizon <= asof
        if elig.size == 0:
            return elig, elig.astype(float), elig.astype(float)
        d2 = d2_all[elig]
        m = min(elig.size, K * CANDIDATE_MULT)
        if m < elig.size:
            part = np.argpartition(d2, m - 1)[:m]
        else:
            part = np.arange(elig.size)
        cand = elig[part]
        dc = d2[part]
        order = np.lexsort((pool.key[cand], dc))
        cand, dc = cand[order], dc[order]
        _, first = np.unique(pool.block[cand], return_index=True)  # nearest row of each block
        keep = np.sort(first)[:K]
        sel, d = cand[keep], np.sqrt(dc[keep].astype(np.float64))
        dk = d[-1] if len(d) and d[-1] > 0 else 1.0
        w = np.exp(-((d / dk) ** 2))          # nearest counts ~2.7x the K-th
        return sel, d, w

    def _paths(self, pool, sel, h, sig_now, end_only=False):
        """Analog forward log-return paths, raw and vol-scaled."""
        g = pool.gpos[sel]
        if end_only:
            raw = (pool.logc[g + h] - pool.logc[g])[:, None]
        else:
            raw = pool.logc[g[:, None] + np.arange(1, h + 1)] - pool.logc[g][:, None]
        ratio = np.clip(sig_now / pool.sig[sel], *SCALE_CLIP)
        return raw, raw * ratio[:, None], ratio

    def forecast(self, ticker, horizon=5, asof=None):
        t0 = time.time()
        if not self.ready:
            a, b = self.progress
            raise ForecastError(503, f"Warming up: {self.status} ({a}/{b}). Try again in a few seconds.")
        self.maybe_refresh()
        t = normalise_ticker(ticker)
        try:
            h = int(horizon)
        except (TypeError, ValueError):
            h = -1
        if h not in HORIZONS:
            raise ForecastError(400, f"horizon must be one of {', '.join(map(str, HORIZONS))} trading days.")
        name_wait = self.raw is None and t not in self.names
        if name_wait:
            self.get_name(t)                       # start the lookup in parallel with the download
        td = self.get_ticker(t)
        last_pos = len(td.day) - 1
        if asof:
            try:
                ad = dt.date.fromisoformat(str(asof)[:10])
            except ValueError:
                raise ForecastError(400, f"asof '{asof}' is not a YYYY-MM-DD date.")
            p = td.pos_on_or_before((ad - dt.date(1970, 1, 1)).days)
            if p < 0:
                raise ForecastError(400, f"asof {ad} is before {t}'s first trading day ({day_str(td.day[0])}).")
        else:
            p = last_pos
        if not td.valid[p]:
            first_ok = np.flatnonzero(td.valid)
            when = day_str(td.day[first_ok[0]]) if first_ok.size else "n/a"
            raise ForecastError(400, f"Not enough history before {day_str(td.day[p])} to describe the setup "
                                     f"(needs about a year). Earliest usable asof for {t}: {when}.")
        out = self._forecast_at(td, p, h)
        out["name"] = self.get_name(t, wait=max(0.0, 2.0 - (time.time() - t0)) if name_wait else 0.0)
        out["track"] = self.get_track(t, h, wait=max(0.0, 4.0 - (time.time() - t0)))
        if out["track"] is None:
            out["notes"].append("Track record is still being computed; it will appear on the next request.")
        return out

    def _forecast_at(self, td, p, h):
        pool = self.pool_for(td.ticker)
        D = td.day[p]
        last = td.close[p]
        sig_now = td.sig[p]
        sel, d, w = self._select(pool, td.X[p], pool.dist2(td.X[p]), D, h)
        if len(sel) < K_MIN:
            raise ForecastError(400, f"asof {day_str(D)} is too early: only {len(sel)} comparable past setups "
                                     f"existed by then (need {K_MIN}).")
        raw, scaled, ratio = self._paths(pool, sel, h, sig_now)
        qlog = np.array([wquantile(scaled[:, j], w, QS) for j in range(h)]).T   # (5, h)
        end = scaled[:, -1]
        naive_w = last * (math.exp(Z_QS[4] * sig_now * math.sqrt(h)) - math.exp(Z_QS[0] * sig_now * math.sqrt(h)))
        prices = last * np.exp(qlog)

        # dates: real bars after asof where we have them (replay), then projected weekdays
        fut_days = [day_str(x) for x in td.day[p + 1: p + 1 + h]]
        if len(fut_days) < h:
            start = dt.date.fromisoformat(fut_days[-1]) if fut_days else dt.date.fromisoformat(day_str(D))
            fut_days += [x.isoformat() for x in project_trading_days(start, h - len(fut_days))]
        actual = None
        if p < len(td.day) - 1:
            actual = [[day_str(td.day[i]), sig6(td.close[i])] for i in range(p + 1, min(p + 1 + h, len(td.day)))]

        # unconditional up-rate for this ticker and horizon, bars <= asof only
        c = td.close[: p + 1]
        base = float(np.mean(c[h:] > c[:-h])) if len(c) > h else float("nan")

        start_hist = (dt.date.fromisoformat(day_str(D)) - dt.date(1970, 1, 1)).days - int(HISTORY_YEARS * 365.25)
        i0 = int(np.searchsorted(td.day, start_hist, side="left"))
        history = [[day_str(td.day[i]), sig6(td.close[i])] for i in range(i0, p + 1)]

        sample = []
        for j in range(min(10, len(sel))):
            r = sel[j]
            sample.append({
                "ticker": pool.tickers[pool.tid[r]],
                "date": day_str(pool.day[r]),
                "ret": r4(math.expm1(raw[j, -1])),          # what really happened to that stock
                "ret_scaled": r4(math.expm1(scaled[j, -1])),  # after vol scaling (what the cone used)
            })

        notes = ["Free Yahoo data: delisted companies are missing (survivorship bias), and the analog "
                 "basket is today's large caps, so past analogs lean toward winners."]
        if p < len(td.day) - 1:
            notes.append("Prices are split/dividend-adjusted with today's factors, so older levels can differ "
                         "from what was quoted on the day; the returns are the same.")
        clipped = float(np.mean((ratio <= SCALE_CLIP[0]) | (ratio >= SCALE_CLIP[1])))
        if clipped > 0.2:
            notes.append(f"{clipped:.0%} of analogs hit the volatility-scaling clip "
                         f"({SCALE_CLIP[0]}x-{SCALE_CLIP[1]}x), so the cone may understate the extremes.")
        if len(sel) < K:
            notes.append(f"Only {len(sel)} comparable past setups existed by this date (normally {K}).")
        if td.note:
            notes.append(td.note)

        return {
            "ticker": td.ticker,
            "name": td.ticker,
            "asof": day_str(D),
            "horizon": h,
            "last_close": sig6(last),
            "history": history,
            "future": {
                "dates": fut_days,
                **{f"p{int(q * 100)}": [sig6(x) for x in prices[i]] for i, q in enumerate(QS)},
            },
            "odds": {
                "n_analogs": int(len(sel)),
                "p_up": r4(np.sum(w * (end > 0)) / np.sum(w)),
                "base_p_up": r4(base),
                "median_ret": r4(math.expm1(qlog[2, -1])),
                "p10_ret": r4(math.expm1(qlog[0, -1])),
                "p90_ret": r4(math.expm1(qlog[4, -1])),
                "width_vs_naive": r4((prices[4, -1] - prices[0, -1]) / naive_w),
            },
            "actual": actual,
            "track": None,
            "analogs_sample": sample,
            "notes": notes,
        }

    # ----- walk-forward (track record + calibration) -------------------------

    def walk_forward(self, ticker, horizons, positions):
        """For each as-of position, forecast using only bars <= that date and
        record the end-of-horizon analog and naive quantiles plus the outcome.
        The distance to the pool is computed once per date and shared across
        horizons; eligibility (A + h <= D) is applied per horizon."""
        td = self.get_ticker(ticker)
        pool = self.pool_for(ticker)
        recs = {h: [] for h in horizons}
        n = len(td.day)
        for p in positions:
            if not td.valid[p]:
                continue
            q = td.X[p]
            d2 = pool.dist2(q)
            pre = np.argpartition(d2, PRE - 1)[:PRE] if pool.n > PRE else None
            D, sig_now = td.day[p], td.sig[p]
            for h in horizons:
                if p + h >= n:
                    continue
                sel, _, w = self._select(pool, q, d2, D, h, pre=pre)
                if len(sel) < K_MIN:
                    continue
                _, scaled, _ = self._paths(pool, sel, h, sig_now, end_only=True)
                aq = wquantile(scaled[:, 0], w, QS)
                recs[h].append({
                    "date": day_str(D),
                    "ret": float(td.logc[p + h] - td.logc[p]),     # realised log return
                    "aq": aq.tolist(),                              # analog log-return quantiles
                    "nq": (Z_QS * sig_now * math.sqrt(h)).tolist(),  # naive log-return quantiles
                    "p_up": float(np.sum(w * (scaled[:, 0] > 0)) / np.sum(w)),
                })
        return recs

    @staticmethod
    def track_positions(td, n_dates=TRACK_N):
        """~150 as-of dates spread evenly from 1 year after listing to the
        latest bar whose shortest-horizon outcome is known. Longer horizons
        simply skip the last few dates whose outcome is not known yet."""
        ok = np.flatnonzero(td.valid)
        if ok.size == 0:
            return []
        lo = max(int(ok[0]), 252)
        hi = len(td.day) - 1 - min(HORIZONS)
        if hi <= lo:
            return []
        return sorted(set(np.linspace(lo, hi, n_dates).round().astype(int).tolist()))

    @staticmethod
    def score(recs):
        if not recs:
            return None
        r = np.array([x["ret"] for x in recs])
        aq = np.array([x["aq"] for x in recs])
        up = r > 0
        maj = max(up.mean(), 1 - up.mean())
        return {
            "n": int(len(recs)),
            "inside_80": r4(np.mean((r >= aq[:, 0]) & (r <= aq[:, 4]))),
            "inside_50": r4(np.mean((r >= aq[:, 1]) & (r <= aq[:, 3]))),
            "direction_hit": r4(np.mean((aq[:, 2] > 0) == up)),
            "direction_base": r4(maj),
        }

    def _track_key(self, t):
        td = self.tick.get(t)
        last = day_str(td.day[-1]) if td is not None else "?"
        return f"{t}|{last}|v2"

    def compute_track(self, t):
        """Walk-forward record for every horizon at once (distances are shared).
        Split over TRACK_THREADS threads: numpy releases the GIL in the big
        array loops, so this roughly halves the wall time on a first request."""
        td = self.get_ticker(t)
        self.pool_for(t)
        pos = self.track_positions(td)
        chunks = [pos[i::TRACK_THREADS] for i in range(TRACK_THREADS)]
        parts = [None] * len(chunks)

        def work(i):
            parts[i] = self.walk_forward(t, list(HORIZONS), chunks[i])

        ths = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(chunks))]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        if any(p is None for p in parts):
            raise RuntimeError("a track worker thread failed")
        return {str(h): self.score([r for p in parts for r in p[h]]) for h in HORIZONS}

    def get_track(self, t, h, wait=0.0):
        """Cached track record, or start computing it in the background and
        wait at most `wait` seconds. Returns None if it is not ready yet."""
        key = self._track_key(t)
        if key in self.track:
            return self.track[key].get(str(h))
        with self.lock:
            th = self.track_threads.get(key)
            if th is None:
                def run():
                    try:
                        t0 = time.time()
                        res = self.compute_track(t)
                        self.track[key] = res
                        self._save_json(TRACK_PATH, self.track)
                        if not self.quiet:
                            log(f"track {t}: {res} ({time.time() - t0:.1f}s)")
                    except Exception as e:
                        log(f"track {t} failed: {e}")
                    finally:
                        self.track_threads.pop(key, None)

                th = threading.Thread(target=run, daemon=True)
                self.track_threads[key] = th
                th.start()
        if wait > 0:
            th.join(wait)
        return self.track.get(key, {}).get(str(h))


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------

def selfcheck():
    """No look-ahead: forecast as of D with all data, then rebuild the whole
    engine from raw data truncated at D (every ticker, the market series too)
    and forecast again. Everything the forecast may depend on must match."""
    full = Engine(quiet=True)
    full.warm_up()
    cases = [("NVDA", "2024-06-14"), ("KO", "2008-10-10"), ("TSLA", "2019-03-01"),
             ("SPY", "2001-09-28"), ("IBM", "1987-10-30"), ("PLTR", "2022-12-30")]
    failures = 0
    for t, asof in cases:
        td = full.get_ticker(t)
        p = td.pos_on_or_before((dt.date.fromisoformat(asof) - dt.date(1970, 1, 1)).days)
        D = td.dates[p]
        raw = {}
        for x in set(POOL + [MARKET_SPLICE, t]):
            try:
                px, _ = load_prices(x)
            except ForecastError:
                continue
            raw[x] = px[px.index <= D]
        trunc = Engine(raw=raw, quiet=True)
        trunc.warm_up()
        tt = trunc.get_ticker(t)
        for h in HORIZONS:
            a = full._forecast_at(td, p, h)
            b = trunc._forecast_at(tt, len(tt.day) - 1, h)
            keys = ["ticker", "asof", "horizon", "last_close", "history", "odds", "analogs_sample"]
            bad = [k for k in keys if a[k] != b[k]]
            bad += [q for q in ("p10", "p25", "p50", "p75", "p90") if a["future"][q] != b["future"][q]]
            # walk-forward records at earlier dates whose outcome is known by D must match exactly
            pos = [x for x in (p - 400, p - 150, p - h) if x > 0]
            wa = full.walk_forward(t, [h], pos)[h]
            wb = trunc.walk_forward(t, [h], pos)[h]
            if wa != wb:
                bad.append("walk_forward")
            # the walk-forward shortcut (pre-selection) must pick exactly the full-scan analogs
            pool = full.pool_for(t)
            for x in pos + [p]:
                d2 = pool.dist2(td.X[x])
                pre = np.argpartition(d2, PRE - 1)[:PRE]
                s1 = full._select(pool, td.X[x], d2, td.day[x], h)[0]
                s2 = full._select(pool, td.X[x], d2, td.day[x], h, pre=pre)[0]
                if not np.array_equal(s1, s2):
                    bad.append("pre_shortcut")
            assert b["actual"] is None and a["history"][-1][0] == a["asof"]
            assert all(x[0] <= a["asof"] for x in a["history"])
            status = "OK  " if not bad else "FAIL " + ",".join(bad)
            failures += 1 if bad else 0
            print(f"{status} {t:5s} asof={a['asof']} h={h:2d}  p50_end={a['future']['p50'][-1]}  "
                  f"n={a['odds']['n_analogs']}  wf_dates={len(wa)}  history_end={a['history'][-1][0]}")
    print("SELF-CHECK", "PASSED" if failures == 0 else f"FAILED ({failures} mismatches)")
    return failures == 0


def fit_scales():
    """Print robust scales of each raw feature on basket rows before 2010."""
    eng = Engine(quiet=True)
    eng.warm_up()
    frames = []
    for t, td in eng.tick.items():
        f = td.raw[FEATURES]
        f = f[(f.index < "2010-01-01")]
        f = f.iloc[MIN_BARS:]
        frames.append(f)
    allf = pd.concat(frames)
    print(f"rows: {len(allf):,}")
    for c in FEATURES:
        v = allf[c].dropna()
        iqr = v.quantile(0.75) - v.quantile(0.25)
        print(f'    "{c}": {iqr / 1.349:.4f},   # std {v.std():.4f}, n {len(v):,}')


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        sys.exit(0 if selfcheck() else 1)
    elif "--fit-scales" in sys.argv:
        fit_scales()
    else:
        print(__doc__)
