"""The Accuracy scoreboard universe: ~150 hand-picked individual stocks.

Individual stocks only: no ETFs, no leveraged or inverse products. It does NOT
change the analog pool (engine.POOL); it only lists which tickers the scoreboard
runs the walk-forward track record for. A stock can be in several groups.

`watchlist` is a personal watchlist (ETFs and leveraged/inverse products removed). Some of those are tiny, renamed or very new;
accuracy.py skips any that Yahoo does not know or that have too little history,
and says why.

This is a list of stocks that exist (and mostly did well enough to be talked
about) today, so it carries survivorship bias like the analog basket does.
"""

GROUPS = [
    {"id": "watchlist", "label": "Watchlist"},
    {"id": "mega", "label": "Mega caps"},
    {"id": "growth", "label": "Growth & tech"},
    {"id": "volatile", "label": "Volatile & small"},
    {"id": "other", "label": "Other sectors"},
]

WATCHLIST = """
AAPL MSFT NVDA AMZN GOOGL META TSLA AMD JPM XOM UNH COST NFLX PLTR COIN
""".split()

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

# Category given to a traded stock that no other list names (none today, kept
# so every stock always has a category besides "watchlist").
_FALLBACK = "volatile"


def build():
    """[(ticker, [group ids])], each ticker once, groups in display order."""
    lists = [("watchlist", WATCHLIST), ("mega", MEGA), ("growth", GROWTH),
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
        if groups[t] == ["watchlist"]:
            groups[t].append(_FALLBACK)
    return [(t, groups[t]) for t in order]


UNIVERSE = build()

# Guard: no ETFs or leveraged/inverse products sneak in.
EXCLUDED = set("SPY QQQ DIA IWM TQQQ SQQQ SOXL SOXS TECS TSLL TSLT IBIT SLV BITX".split())
assert not EXCLUDED & {t for t, _ in UNIVERSE}, "an ETF or leveraged product is in the universe"


if __name__ == "__main__":
    from collections import Counter
    c = Counter(g for _, gs in UNIVERSE for g in gs)
    print(f"{len(UNIVERSE)} stocks;", ", ".join(f"{g['id']} {c[g['id']]}" for g in GROUPS))
