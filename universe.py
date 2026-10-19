"""The Accuracy scoreboard universe: ~150 hand-picked individual stocks.

Individual stocks only: no ETFs, no leveraged or inverse products. It does NOT
change the analog pool (engine.POOL); it only lists which tickers the scoreboard
runs the walk-forward track record for. A stock can be in several groups.

The first group is a personal watchlist. The repo ships a neutral default
(DEFAULT_WATCHLIST below). To use your own, put a `universe.local.json` next to
this file (it is gitignored):

    {
      "id": "watchlist",              optional group id, default "watchlist"
      "label": "Watchlist",           optional label shown in the page
      "tickers": "AAPL MSFT NVDA",    a space-separated string or a JSON list
      "exclude": "TQQQ SQQQ"          optional extra tickers the guard rejects
    }

Some tickers in a personal list may be tiny, renamed or very new; accuracy.py
skips any that Yahoo does not know or that have too little history, and says why.

This is a list of stocks that exist (and mostly did well enough to be talked
about) today, so it carries survivorship bias like the analog basket does.
"""

import json
import os

LOCAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe.local.json")

# Neutral public default: well-known, liquid US stocks across sectors.
DEFAULT_WATCHLIST = """
AAPL MSFT NVDA AMZN GOOGL META TSLA AMD JPM XOM UNH COST NFLX PLTR COIN
""".split()


def _tickers(v):
    return v.split() if isinstance(v, str) else [str(t) for t in v]


def _load_local(path=LOCAL_FILE):
    """The personal watchlist file, or None when it is absent."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_local = _load_local()
WATCHLIST_ID = (_local or {}).get("id", "watchlist")
WATCHLIST_LABEL = (_local or {}).get("label", "Watchlist")
WATCHLIST = _tickers(_local["tickers"]) if _local and "tickers" in _local else DEFAULT_WATCHLIST
_LOCAL_EXCLUDE = _tickers((_local or {}).get("exclude", ""))

GROUPS = [
    {"id": WATCHLIST_ID, "label": WATCHLIST_LABEL},
    {"id": "mega", "label": "Mega caps"},
    {"id": "growth", "label": "Growth & tech"},
    {"id": "volatile", "label": "Volatile & small"},
    {"id": "other", "label": "Other sectors"},
]

# Category groups. Order inside a group is the display order hint only.
MEGA = """
AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO BRK-B JPM LLY V MA WMT XOM UNH ORCL COST
NFLX JNJ
""".split()

GROWTH = """
AMD PLTR ADBE ORCL INTC ALAB CRWV SNDK WDC TTD PYPL FSLY
MU ARM SMCI TSM ASML AMAT LRCX MRVL ANET QCOM TXN DELL STX
CRM NOW PANW CRWD ZS NET DDOG SNOW MDB SHOP UBER ABNB SPOT APP RBLX COIN HOOD
""".split()

VOLATILE = """
IREN APLD OKLO RVI OPEN ONDS UEC CHYM ACHR RGNX VNDA TRX CHOW MLEC OLB IBRX
ANNA RAYA IRE LABX SOFI QBTS RGTI QUBT HL CDE
GME AMC MARA RIOT CLSK CIFR HUT MSTR IONQ SMR NNE LEU CCJ RKLB ASTS LUNR JOBY
UPST AFRM CVNA BBAI SOUN RIVN LCID PLUG HIMS DJT
""".split()

OTHER = """
CMG SBUX NEM KGC HL CDE
BAC WFC GS MS C SCHW
CVX OXY COP SLB
PFE MRK ABBV AMGN VRTX ISRG
KO PEP PG MCD NKE DIS HD TGT
CAT BA GE HON DE LMT RTX UPS
NEE DUK O PLD AMT VZ T
FCX GOLD
""".split()

# Category given to a watchlist stock that no other list names, so every stock
# always has a category besides the watchlist.
_FALLBACK = "volatile"


def build():
    """[(ticker, [group ids])], each ticker once, groups in display order."""
    lists = [(WATCHLIST_ID, WATCHLIST), ("mega", MEGA), ("growth", GROWTH),
             ("volatile", VOLATILE), ("other", OTHER)]
    order, groups = [], {}
    for gid, tickers in lists:
        for t in tickers:
            if t not in groups:
                groups[t] = []
                order.append(t)
            if gid not in groups[t]:
                groups[t].append(gid)
    for t in order:
        if groups[t] == [WATCHLIST_ID]:
            groups[t].append(_FALLBACK)
    return [(t, groups[t]) for t in order]


UNIVERSE = build()

# Guard: no ETFs or leveraged/inverse products sneak in.
EXCLUDED = set("""
SPY QQQ DIA IWM VOO VTI TQQQ SQQQ SPXL SPXS UPRO SOXL SOXS TECL TECS TNA TZA
UVXY SVXY GLD SLV IBIT FBTC BITX TSLL TSLT
""".split()) | set(_LOCAL_EXCLUDE)
assert not EXCLUDED & {t for t, _ in UNIVERSE}, "an ETF or leveraged product is in the universe"


if __name__ == "__main__":
    from collections import Counter
    c = Counter(g for _, gs in UNIVERSE for g in gs)
    print(f"{len(UNIVERSE)} stocks;", ", ".join(f"{g['id']} {c[g['id']]}" for g in GROUPS))
