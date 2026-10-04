"""Data splits for the accuracy research (research/PLAN.md, "don't fool ourselves").

| split | tickers                  | as-of dates               |
|-------|--------------------------|---------------------------|
| dev   | TRAIN tickers            | before 2012-01-01         |
| val   | TRAIN tickers            | 2012-01-01 .. 2018-12-31  |
| test  | ALL tickers              | 2019-01-01 .. latest      |

TRAIN vs TEST tickers: md5 of the upper-cased ticker, even last hex digit = TRAIN.
The analog POOL is a source of analogs on every split (time rules still apply).

Run `python research/splits.py` to print the counts.
"""
import datetime as dt
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEV_END = dt.date(2012, 1, 1)        # DEV: as-of < this
VAL_START = dt.date(2012, 1, 1)      # VAL: VAL_START <= as-of <= VAL_END
VAL_END = dt.date(2018, 12, 31)
TEST_START = dt.date(2019, 1, 1)     # TEST: as-of >= this

SPLITS = ("dev", "val", "test")
EXCLUDE = {"SPY", "QQQ"}             # index ETFs: analog sources, never evaluated
_EPOCH = dt.date(1970, 1, 1)


def is_train_ticker(t):
    """True if the ticker belongs to the TRAIN ticker set (deterministic)."""
    return int(hashlib.md5(t.strip().upper().encode("utf-8")).hexdigest()[-1], 16) % 2 == 0


def split_day_range(split):
    """(lo, hi) as-of bounds in days since 1970-01-01, inclusive."""
    if split == "dev":
        return -10 ** 9, (DEV_END - _EPOCH).days - 1
    if split == "val":
        return (VAL_START - _EPOCH).days, (VAL_END - _EPOCH).days
    if split == "test":
        return (TEST_START - _EPOCH).days, 10 ** 9
    raise ValueError(f"unknown split {split!r}; one of {SPLITS}")


def split_of_day(day):
    """Split name for an as-of date given in days since epoch."""
    for s in SPLITS:
        lo, hi = split_day_range(s)
        if lo <= day <= hi:
            return s
    raise ValueError(day)


def all_tickers():
    """Evaluation tickers: the scoreboard universe plus the analog basket,
    SPY/QQQ excluded, sorted. (Availability of cached prices is checked by the
    harness, not here.)"""
    from universe import UNIVERSE
    import engine
    ts = {t for t, _ in UNIVERSE} | set(engine.POOL)
    return sorted(ts - EXCLUDE)


def tickers_for(split):
    """Tickers scored on a split: TRAIN tickers for dev/val, all tickers for test."""
    ts = all_tickers()
    return ts if split == "test" else [t for t in ts if is_train_ticker(t)]


if __name__ == "__main__":
    ts = all_tickers()
    tr = [t for t in ts if is_train_ticker(t)]
    te = [t for t in ts if not is_train_ticker(t)]
    print(f"{len(ts)} evaluation tickers: TRAIN {len(tr)}, TEST {len(te)}")
    print("TRAIN:", " ".join(tr))
    print("TEST: ", " ".join(te))
