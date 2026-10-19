# xsec-02: how much of the TEST momentum result is survivorship?

> **Not financial advice.** This is a research log from building an educational tool. Backtest numbers are hypothetical, carry survivorship bias, and say nothing certain about the future. Nothing here is a recommendation to buy, sell or hold anything.

Lane cone-xrobust, 2026-10-04. Code: `research/xsec_robust_test.py`. Data: `research/results/xrobust_*`. Measurement only: the frozen `mom_12_1` signal (orient +1, trade at the next close, 10 bps per side, top quintile, equal weight) run on TEST (2019-01..2026-10). Nothing was designed, tuned or chosen on TEST.

**Short answer.** The TEST headline (+32%/yr weekly, +15%/yr monthly over the equal-weight universe) is mostly about which stocks are in the list. Restrict to stocks that were already large and liquid, or that were not picked for recent runs, and the excess falls to **+4 to +13%/yr with t 1.0 to 2.1**. Then drop the five biggest contributors and it falls to about **0 to +6%**. Half the full-universe excess comes from 10 names, led by GME, the crypto miners, PLUG and CVNA.

**Honest estimate of the edge: about 0 to 6%/yr over an equal-weight basket of the same large caps, monthly rebalanced, before taxes. Plan on 2-3%.** It is not +15-32%.

## Method

**Data.** `bt.Panel(splits.tickers_for("test"))` reads the 163 tickers straight from `data/prices`, and none are missing. The data ends 2026-10-02. Eligibility is unchanged (`bt.eligibility`: 252 days of history, 63-day median dollar volume of at least $2M, at most 10% zero-volume days). The signal is `xsec.compute_signals(panel)["mom_12_1"]`. The full-universe run reproduces `xsec_test.csv` exactly: +0.321 / t 2.85 weekly, +0.149 / t 2.04 monthly.

**Subsets.** A subset is the eligibility mask AND'ed with a ticker list. Within a subset, the EW benchmark and the quintile are formed only from that subset's names.
- (a) full 163.
- (b) `engine.BASKET`, the 42 analog-pool large caps.
- (c) `universe.py` group `mega` (20).
- (d) group `other` (47).
- (e) every ticker in neither `volatile` nor the personal watchlist group (91). This drops AAPL, NVDA, TSLA and others, because they were on that watchlist.
- (f) `volatile` (53).

A rebalance date needs at least 20 eligible names for (a), as the coordinator ran it, and at least 10 for the subsets. Mega therefore holds 4 names per quintile and the analog pool 8.

**Selection-date groups.** The cache has no market caps, so dollar volume is the size proxy.
- **A, large and liquid at the start of 2019** (87 names): first close by 2017-12-31 and a Q4-2018 median daily dollar volume of at least $100M.
- **B, new or tiny** (67 names): first close on or after 2019-01-01, or a Q4-2018 median dollar volume under $25M.
- **mid** (9 names): everything else. These are AMC, CVNA, GME, KGC, MDB, O, RGNX, SPOT and ZS.
- A variant B' (53 names) uses "listed after 2020, or under $25M in 2020-H2".
- The grouping is in `xrobust_selection_groups.csv`.

**Attribution.** Each stock's contribution to the gross (pre-cost), arithmetic top-minus-EW excess is the sum over periods of (w_top − w_ew) × (r_i − r_EW). It replicates `bt.run`'s selection exactly and is annualised as the mean per period × periods per year. The "without top k" numbers remove those k tickers from the universe entirely and rerun the backtest, so they leave the EW benchmark too.

**Control for the ex-post drop.** Removing any strategy's best names after the fact lowers it mechanically. Each of the 20 random signals had its own top-k contributors removed the same way. The average drop is reported next to momentum's.

**Yardsticks per subset.**
- The drift placebo: each stock's expanding mean daily log return, at least 252 days, the same as in `xsec.py`.
- 20 random signals (seeds 0-19).

The random signals average **−2 to −11%/yr** of excess. Weekly random portfolios turn over about 40x a year at 10 bps, and a 4-26-stock basket pays a volatility drag against the 20-131-stock EW. So "beats random" is a low bar. The excess over EW is the number that matters for money.

## Results per subset

The excess is CAGR(top) − CAGR(EW), net of costs. t is Newey-West on the per-period active return. Turnover is one-way per year. "Names" is the average eligible count, and the second number is names per quintile.

| subset | freq | names | IC | IC t | top/yr | EW/yr | **excess** | **t** | SPY | top maxDD | turnover | drift excess | random mean (sd) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) full 163 | wk | 131/26 | .024 | 2.04 | 68.6% | 36.5% | **+32.1%** | **2.85** | 17.6% | −38% | 6.9 | −6.0% | −10.8 (5.3) |
| (a) full 163 | mo | 131/26 | .031 | 1.63 | 50.8% | 35.9% | **+14.9%** | **2.04** | 17.7% | −39% | 3.2 | −8.5% | −4.2 (5.4) |
| (b) analog pool | wk | 42/8 | .012 | 0.85 | 29.8% | 20.5% | **+9.3%** | **1.75** | 17.6% | −30% | 6.8 | +4.5% | −10.3 (3.3) |
| (b) analog pool | mo | 42/8 | .014 | 0.52 | 26.9% | 21.2% | **+5.8%** | **1.05** | 17.7% | −17% | 3.0 | +3.9% | −2.0 (5.6) |
| (c) mega | wk | 20/4 | .003 | 0.21 | 39.8% | 29.3% | **+10.6%** | **1.69** | 17.6% | −34% | 8.0 | +6.1% | −10.7 (5.0) |
| (c) mega | mo | 20/4 | .006 | 0.19 | 44.4% | 30.1% | **+14.3%** | **2.01** | 17.7% | −21% | 3.3 | +8.9% | −3.7 (6.5) |
| (d) other | wk | 47/9 | .014 | 1.04 | 22.3% | 17.3% | **+5.0%** | **1.10** | 17.6% | −32% | 6.9 | −3.9% | −9.1 (4.2) |
| (d) other | mo | 47/9 | .029 | 1.31 | 21.8% | 18.3% | **+3.5%** | **0.93** | 17.7% | −22% | 3.2 | −2.3% | −2.7 (3.0) |
| (e) ex volatile & watchlist | wk | 87/17 | .022 | 1.67 | 38.0% | 26.1% | **+11.9%** | **1.93** | 17.6% | −32% | 7.0 | +3.3% | −9.4 (2.8) |
| (e) ex volatile & watchlist | mo | 86/17 | .020 | 1.00 | 35.7% | 26.3% | **+9.4%** | **1.76** | 17.7% | −24% | 3.2 | +1.9% | −2.6 (3.7) |
| (f) volatile | wk | 29/6 | .018 | 1.30 | 72.6% | 66.7% | **+5.9%** | **0.89** | 17.6% | −74% | 7.4 | −38.2% | −17.6 (19.6) |
| (f) volatile | mo | 29/6 | .058 | 1.88 | 65.9% | 57.8% | **+8.1%** | **0.96** | 17.7% | −77% | 3.5 | −34.8% | −17.3 (15.5) |
| A: large+liquid 2019 | wk | 87/17 | .020 | 1.48 | 38.1% | 24.9% | **+13.3%** | **2.11** | 17.6% | −35% | 6.9 | +6.6% | −10.6 (2.4) |
| A: large+liquid 2019 | mo | 87/17 | .026 | 1.20 | 35.9% | 25.5% | **+10.4%** | **1.56** | 17.7% | −25% | 3.1 | +6.5% | −2.6 (3.7) |
| B: new or tiny 2019 | wk | 41/8 | .006 | 0.45 | 64.4% | 65.4% | **−1.0%** | **0.84** | 17.6% | −60% | 7.5 | −27.8% | −10.9 (20.9) |
| B: new or tiny 2019 | mo | 41/8 | .003 | 0.12 | 44.5% | 66.5% | **−22.0%** | **−0.35** | 17.2% | −73% | 3.5 | −44.0% | −13.2 (19.2) |
| full minus B (A+mid) | wk | 96/19 | .029 | 2.28 | 51.0% | 28.0% | **+23.0%** | **2.87** | 17.6% | −37% | 6.9 | +3.6% | −11.8 (4.5) |
| full minus B (A+mid) | mo | 96/19 | .037 | 1.77 | 43.3% | 27.7% | **+15.6%** | **2.47** | 17.7% | −26% | 3.1 | +3.6% | −2.8 (4.9) |
| full minus B' (2021 def) | wk | 107/21 | .029 | 2.32 | 55.1% | 31.0% | **+24.1%** | **2.53** | 17.6% | −37% | 7.1 | −0.7% | −11.0 (4.5) |
| full minus B' (2021 def) | mo | 107/21 | .036 | 1.95 | 43.7% | 30.5% | **+13.2%** | **2.10** | 17.7% | −33% | 3.1 | +0.3% | −3.7 (5.7) |

**How to read the subsets**
- **The excess falls as selection bias falls.**
  - Full: +32 / +15%.
  - Lists chosen for liquidity or sector rather than recent runs, (b) and (d): +3.5 to +9%, t 0.9-1.75, not significant.
  - Broad "not hand-picked as volatile or watchlist" (e) and "large at the start of 2019" (A): +9 to +13%, t 1.6-2.1.
- **In the large-cap subsets the IC is weak.** IC t is 0.2-1.0 in (b), (c) and (d), against 2.04 in the full set.
  - The excess there comes from holding a few big trends rather than from a reliable ranking. AMD and NVDA give 78% of the analog-pool weekly excess. In mega, a 4-stock quintile is mostly NVDA/AVGO/META during their runs.
- **The drift placebo**, a meaningless "own what has risen since listing" rule, earns +4 to +9%/yr in (b), (c) and A.
  - Those lists are also survivors, because NVDA, AVGO, LLY and AMD are in them as 2026 large caps. So part of the large-cap momentum excess is the same hindsight.
  - Momentum minus drift is roughly +2 to +9%, depending on subset and frequency.
  - In the full set drift is negative. Short-history names with huge since-listing drift (the 2021 SPAC/IPO cohort) crashed, which hides the survivorship rather than removing it.
- **The volatile group (small, high-volatility names):**
  - The excess is +6 to +8% with t below 1.
  - The top quintile drew down 74-77%.
  - The random spread is ±16-20%/yr.
  - There is no usable signal.
- **New or tiny names alone (B):** momentum does nothing (−1% / −22%, t 0.8 / −0.35). The monthly −22% is mostly volatility drag on a 6-8 stock basket of very volatile names.

## Selection-date check (A versus B)

Gross arithmetic attribution of the full-universe excess (`xrobust_concentration.csv`):

| | weekly | monthly |
|---|---|---|
| total gross excess (arithmetic) | +27.6%/yr | +16.5%/yr |
| from A (87 large and liquid at the start of 2019) | +11.5 (42%) | +9.6 (58%) |
| from B (67 new or tiny) | +10.2 (37%) | +2.8 (17%) |
| from mid (9: GME, AMC, CVNA, ...) | +5.9 (21%) | +4.1 (25%) |

- **B plus mid together** give 58% of the weekly excess and 42% of the monthly. These are names not yet large and liquid in 2019, most of them in the list because they later soared or squeezed.
- **The sub-universe view agrees.** A alone earns +13.3 / +10.4%, against +32.1 / +14.9% for the full set. About **19 points of the weekly 32** and **about 4.5 points of the monthly 15** go away when the universe is limited to names that were already established.
- **Removing only B leaves +23 / +16%.** That is because GME, AMC and CVNA sit in "mid": they were listed and moderately traded in 2018.

## Concentration

Full universe, gross arithmetic contributions (`xrobust_contrib.csv`):
- **Weekly top 10:** GME 16%, MARA 6%, CVNA 4%, RIOT 4%, PLUG 4%, IREN 4%, APP 3%, SHOP 3%, MSTR 3%, SMCI 3%. Top 5 = **34%**, top 10 = **49%**.
- **Monthly top 10:** GME 13%, AMC 9%, PLUG 7%, CVNA 6%, APP 6%, SMCI 5%, SHOP 5%, RKLB 5%, AMD 5%, OKLO 4%. Top 5 = **40%**, top 10 = **64%**.

Rerun without those names, net excess (t):

| universe | freq | all | without top 5 | without top 10 | random signals' own drop, top 5 / top 10 |
|---|---|---|---|---|---|
| full | wk | +32.1% (2.85) | +17.3% (2.22) | +12.0% (1.77) | −5.6 / −8.5 pts |
| full | mo | +14.9% (2.04) | +7.8% (1.37) | +5.0% (1.04) | −5.3 / −8.3 pts |
| analog pool | wk | +9.3% (1.75) | −0.7% (0.01) | −5.7% (−1.07) | −4.2 / −7.1 pts |
| analog pool | mo | +5.8% (1.05) | −2.6% (−0.41) | −4.8% (−0.93) | −5.0 / −8.0 pts |
| ex volatile & watchlist | wk | +11.9% (1.93) | +5.6% (1.11) | +2.2% (0.58) | −3.2 / −5.6 pts |
| ex volatile & watchlist | mo | +9.4% (1.76) | +2.1% (0.69) | −0.4% (0.22) | −3.6 / −6.2 pts |

- **Dropping its own five best names** costs momentum 6-15 points, about 2x the mechanical drop a random signal suffers. So momentum really did ride those runs.
- **But what is left is small and not significant:** +2 to +8% (t 0.7-1.4) for monthly and the cleaner subsets.
- **In the analog pool, nothing is left** after removing AMD, NVDA, INTC, XOM and META.

## By year

Net excess over the subset's own EW (`xrobust_years.csv`). 2026 is partial, to 2026-10-02.

| year | full wk | full mo | analog wk | analog mo | ex-vol&watchlist wk | ex-vol&watchlist mo | A wk | A mo |
|---|---|---|---|---|---|---|---|---|
| 2019 | +13.9 | +10.2 | −5.0 | −7.3 | +0.7 | +3.6 | +12.2 | +7.4 |
| 2020 | +79.1 | +29.4 | +22.3 | +26.2 | +17.1 | +10.7 | +24.1 | +36.0 |
| 2021 | +63.0 | +59.3 | −23.8 | −17.8 | −12.1 | −6.7 | −13.7 | −15.9 |
| **2022** | **+21.9** | **+5.3** | **+11.5** | **+4.6** | **+23.7** | **+11.2** | **+15.8** | **+10.3** |
| 2023 | +2.1 | −26.5 | +7.3 | −0.8 | −2.2 | −1.0 | −3.4 | −3.0 |
| 2024 | +18.7 | +27.4 | +25.1 | +27.8 | +32.4 | +41.1 | +10.9 | +5.8 |
| 2025 | +50.4 | +34.3 | −4.6 | −11.6 | +9.3 | +6.1 | −1.7 | −5.4 |
| 2026* | +12.0 | +3.9 | +41.9 | +32.6 | +17.6 | +11.8 | +64.9 | +58.2 |

- **2021 is the clearest tell.** In the full set it is +60%: momentum held GME, AMC, the crypto miners and PLUG. In every less-biased subset it is −7 to −24%.
- **The full set's 2025 gain of +34 to +50%** comes from the quantum, nuclear and crypto names. The large-cap subsets are flat or negative that year.
- **2022 (the bear market):** momentum beat EW in every subset, +5 to +26%. Full weekly: top −8.5%, EW −30.5%, SPY −18.0%. Analog pool weekly: top +1.5%, EW −10.0%.
  - This is the textbook case: by 2022 the 12-1 winners were energy and defensives (XOM, CVX, OXY, MRK), and the losers were the 2021 high-flyers.
  - Momentum's real crash risk is the rebound out of a bear market (2009 on DEV trailed by 30-40%), and 2023 was only a mild version of that (−1 to −27%).
- **Year to year it is lumpy.** In the cleaner subsets the excess runs from −24% to +65%, so any single year says little. 2026 is a partial year dominated by a few large-cap trends.

## Verdict

**The honest estimate of momentum's edge is about 0 to +6% a year over an equal-weight basket of the same established large caps, monthly rebalanced, after 10 bps costs, before taxes. Plan on 2-3%.** The reasoning:

1. **The full-universe numbers (+15 to +32%/yr) are not usable.**
   - About 60% of the weekly excess and 40% of the monthly comes from names that were new or small in 2019. They are in the list because they later soared: GME, AMC, CVNA, MARA, RIOT, PLUG, IREN, MSTR, OKLO.
   - 49-64% of it is 10 stocks.
   - Momentum buys a stock during its run, so a list of stocks known to have run flatters it more than the EW comparison can remove.
2. **Lists chosen for size or sector rather than recent runs give +3.5 to +13%, with t 0.9 to 2.1.** Only two of those cuts reach t 2: A weekly (t 2.11) and mega monthly (t 2.01, a 4-stock quintile). Full-minus-new-or-tiny reaches t 2.5-2.9 only because GME, AMC and CVNA stay in it.
3. **Those lists are still survivors.**
   - The drift placebo earns +2 to +9% in them.
   - Dropping the top 5 contributors leaves −3 to +6% (t at most 1.1), against a 3-5 point mechanical drop for random signals.
   - The upper end of the TEST range is best read as momentum-friendly 2019-26 years (AI megacaps, the 2022 energy rotation) on a hindsight list.
4. **It agrees with the pre-TEST evidence.**
   - DEV (1987-2011): +2 to +3.5%/yr, t about 1.
   - xsec-01's realistic estimate was 0 to 4%.
   - Published long-only momentum tilts in large caps are a few percent a year, before taxes.
5. **On the volatile, small names there is no measurable edge.** The excess is +6 to +8% with t below 1, and the top quintile drew down 74-77%.

**Practical consequence.**
- Momentum is at most a small tilt within a basket of established large caps, rebalanced monthly, after costs and taxes.
- **Weekly is not worth it:** turnover is about 7x a year, against 3x for monthly, and the extra weekly excess comes from the meme and crypto names.
- **Do not size positions as if it were worth 15-30%/yr.** Do not use it to pick among speculative small caps.
- **The ride is rough.** Tracking error against EW is high: some years trail by 7-24 points (2021 in the cleaner subsets), and the top-quintile drawdown is −17% to −35% even in the large-cap lists.

## Not verified

- **Survivorship cannot be removed with this data.** There are no delisted names and no point-in-time index membership. The subsets reduce selection bias but do not remove it: even `engine.BASKET` and `mega` are 2026 lists that include NVDA, AMD, AVGO and LLY.
- **"Large and liquid in 2019" uses Q4-2018 dollar volume as a size proxy** ($100M / $25M cut-offs chosen a priori, not tuned). There are no market caps. The 9 "mid" names fall between the cut-offs, and the result depends on where GME, AMC and CVNA land.
- **The subsets use a 10-name minimum** where the coordinator's full run uses 20. Mega (4 per quintile) and analog (8 per quintile) are very noisy, and their t-stats are rough.
- **Attribution is gross and arithmetic.** The subset tables are net CAGR differences. For volatile subsets the two differ a lot because of volatility drag, which is why B monthly shows −22% with t −0.35.
- **The random-signal mean is strongly negative** because of costs and concentration. "Beats 20/20 random" is therefore weak evidence. The excess over EW is the relevant number.
- **About 20 subset/frequency cuts were measured**, so a t near 2 in any single cut is weaker than nominal. Nothing was selected from them; the verdict uses their spread.
- **Costs are a flat 10 bps.** Real costs on volatile and small names are higher. There are no taxes, no market impact and no borrow costs, and long-only means no short leg is needed.
- **2026 is a partial year** (to 2026-10-02).
- **Not checked:** sector neutralisation (the cache has no sector data), other momentum lookbacks (not allowed on TEST), and same-close (lag 0) variants on the subsets.
