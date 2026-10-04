# event-01: widening the cone around earnings-like events (from price and volume only)

Lane `cone-event`, 2026-10-04. Code: `research/models_event.py`. Outputs: `research/results/event_*`.
The base model is `vol_har_drift` (HAR sigma from `research/volforecast.py`, own walk-forward
drift, frozen "drift" z). DEV was used for fitting, VAL for choosing between variants. TEST was never opened.

## Method

**1. Detector** (`detect`). This is causal: bar i uses bars up to i only. A bar counts as an event when both of these hold:
- Its volume is at least **2.0x** the median volume of the previous 63 bars. This ratio is first divided by the market's
  own volume ratio (^GSPC) when that ratio is above 1, so a day when the whole market trades heavily is not counted.
- |stock return − market return| is at least **1.0x** a robust trailing sd of that difference (median of squares / 0.455, previous 63 bars).

Only the first event in any 15-bar span is kept. Each event gets a size k_e = r_e² / s_e², where s_e² is the stock's
robust trailing daily variance. The detector finds 5.3 events/yr on DEV and 4.3/yr on VAL (TRAIN tickers).

**2. Predictor** (`predict_due`). The question is whether an event falls in (t, t+h]. The window is turned into calendar days
(h·365.25/252), then shifted back by 1, 2 and 3 quarters (91/182/273 days, ±10) and by 52, 104 and 156 weeks
(364/728/1092 days, ±7). The predictor checks whether a past event fell inside each shifted window. Yearly lags
count double. Score = the weighted share of lags that had a hit, among lags with history behind them.
- **"Due"** means score ≥ 0.30 with at least 2 valid lags.
- The score is forced to 0 when the last event was so recent that the window ends less than 70 days after it.
- Only events dated on or before t are used.

**3. Model** (`EventHar`). The base h-day variance is h·σ², where σ is the HAR daily sigma. A = k̄ − 1 is the event-day variance excess:
- **pool**: k̄ = 8.0 for every stock. This is the DEV mean of capped k_e (8.03, median 4.4, n = 5,446).
- **shrunk**: k̄ = (sum of the stock's last 8 capped k_e + 4·8.0) / (n + 4).

The variance then becomes:
- **binary**: h·σ² + c·A·σ² if due, else d·h·σ².
- **soft**: d·h·σ² + c·score·A·σ².

In both cases the variance is floored at 0.5·h·σ². c and d were grid-searched jointly on DEV by scaled pinball (step 0.025)
and frozen as literals:

| h | binary (c, d) | soft (c, d) |
|---|---|---|
| 5 | 0.050, 0.975 | 0.150, 0.975 |
| 21 | 0.150, 0.975 | 0.375, 0.975 |
| 63 | 0.000, 0.900 | binary 0.000, 0.900 |

At h = 63, 97% of windows are "due", so there is nothing to separate. Soft h = 63 fitted (0.25–0.6, 0.95–0.975) but
lost on VAL, so h = 63 keeps the binary deflation only. Quantiles = mu·h + sqrt(V)·z_drift, exactly as vol_har_drift.

The final model is **`event_har` = shrunk, soft** (h = 5/21) **+ binary deflation** (h = 63). It was chosen on VAL for the best h = 5
skill and due-window coverage, and it does not lose on DEV.

**Checks.**
- `event_base` (no add-on) reproduces vol_har_drift's VAL numbers exactly.
- `harness.py --lookahead --models event_har,event_shrunk,event_pool,event_base`: all OK, HARNESS PROOFS PASSED.
- The by-group tables come from a per-row panel (`event_panel_<split>.npz`) scored with `harness.metrics`. Its overall
  rows match the harness JSONs to 4 decimals.

## Event detector quality

The truth for each grid row is that the same detector fires at a bar in (t, t+h]. Grid = the harness grid (TRAIN tickers, every 5 bars).

| split | h | n | base rate | predicted rate | precision | recall | P(event \| not predicted) | lift |
|---|---|---|---|---|---|---|---|---|
| DEV | 5 | 48,277 | 0.094 | 0.158 | 0.140 | 0.235 | 0.086 | 1.49 |
| DEV | 21 | 48,277 | 0.386 | 0.408 | 0.407 | 0.430 | 0.371 | 1.06 |
| DEV | 63 | 48,277 | 0.756 | 0.973 | 0.764 | 0.984 | 0.441 | 1.01 |
| VAL | 5 | 14,951 | 0.081 | 0.155 | 0.152 | 0.291 | 0.068 | 1.88 |
| VAL | 21 | 14,951 | 0.332 | 0.407 | 0.393 | 0.482 | 0.290 | 1.18 |
| VAL | 63 | 14,951 | 0.715 | 0.969 | 0.720 | 0.976 | 0.553 | 1.01 |

The rhythm is weak:
- Only 33–41% of detected events have another event 52 weeks (±10 days) earlier, against 19–28% by chance at the same event rate.
  The figure is about the same for every detector tried (volume 1.5–2.5x, |z| 0–1.5, local-max rule, quad-witching filter).
- Spot check, AAPL 2012–2018: most detected events are real earnings reactions. Some are product-event or sell-off days
  (2012-02-29, 2013-09-10, 2017-06-09). Several earnings days with small moves are missed (2012-04-25, 2013-04-24, 2014-10-21).

Variance in the window: the realised (r − mu·h)² divided by the HAR h-day variance.
- When the detector fired in a 5-day window, the ratio is 2.54x (no event: 0.84x).
- "Due" windows: 1.07x vs 0.99x on DEV, and 1.23x vs 0.95x on VAL. The prediction captures only a small part of the event risk.
- Higher thresholds raise precision but lower recall (VAL h = 5, thr 0.5: P 0.22, R 0.18, variance ratio 1.40x).

Full tables: `event_detector.csv`, `event_predictor_variants_{dev,val}.txt`.

## Results

### Overall (research/harness.py; pinball skill vs naive0 [ticker-bootstrap 90% CI], cov50, cov80, width80)

| split | h | vol_har_drift (= event_base) | event_pool_bin | event_shrunk_bin | event_pool | **event_har** (= event_shrunk) |
|---|---|---|---|---|---|---|
| DEV | 5 | +0.0054 [.0047,.0061] .499/.799 w.1256 | +0.0055 .497/.798 | +0.0055 .497/.797 | +0.0055 .498/.799 | **+0.0055** [.0048,.0062] .498/.799 w.1253 |
| DEV | 21 | +0.0095 [.0072,.0121] .498/.799 w.2564 | +0.0096 .499/.800 | +0.0096 .499/.800 | +0.0096 .499/.801 | **+0.0096** [.0073,.0121] .499/.801 w.2573 |
| DEV | 63 | +0.0124 [.0055,.0195] .500/.801 w.4552 | +0.0124 | +0.0124 | +0.0124 | **+0.0124** .500/.801 w.4548 |
| VAL | 5 | +0.0124 [.0105,.0143] .527/.812 w.1010 | +0.0127 .524/.810 | +0.0127 .525/.810 | +0.0127 .525/.811 | **+0.0128** [.0109,.0148] .527/.813 w.1011 |
| VAL | 21 | +0.0288 [.0221,.0355] .526/.816 w.2092 | +0.0289 .527/.817 | +0.0289 .528/.818 | +0.0290 .527/.817 | **+0.0290** [.0224,.0357] .530/.819 w.2110 |
| VAL | 63 | +0.0673 [.0490,.0876] .531/.832 w.3773 | +0.0673 | +0.0673 | +0.0673 | **+0.0673** .530/.831 w.3768 |

### By event-in-window: vol_har_drift → event_har (cov50, cov80, pinball skill)

| split | h | group | n | cov50 | cov80 | skill |
|---|---|---|---|---|---|---|
| DEV | 5 | due | 7,645 | 0.480 → 0.497 | 0.783 → 0.799 | 0.0056 → 0.0060 |
| DEV | 5 | not due | 40,632 | 0.503 → 0.498 | 0.802 → 0.799 | 0.0054 → 0.0054 |
| DEV | 21 | due | 19,674 | 0.490 → 0.500 | 0.792 → 0.801 | 0.0095 → 0.0098 |
| DEV | 21 | not due | 28,603 | 0.504 → 0.499 | 0.805 → 0.801 | 0.0095 → 0.0095 |
| DEV | 63 | due | 46,985 | 0.500 → 0.500 | 0.800 → 0.800 | 0.0131 → 0.0131 |
| DEV | 63 | not due | 1,292 | 0.502 → 0.483 | 0.835 → 0.814 | −0.0164 → −0.0157 |
| VAL | 5 | due | 2,315 | 0.503 → 0.522 | 0.774 → 0.794 | 0.0119 → 0.0132 |
| VAL | 5 | not due | 12,636 | 0.532 → 0.528 | 0.819 → 0.816 | 0.0124 → 0.0127 |
| VAL | 21 | due | 6,086 | 0.502 → 0.517 | 0.802 → 0.814 | 0.0273 → 0.0274 |
| VAL | 21 | not due | 8,865 | 0.542 → 0.539 | 0.826 → 0.822 | 0.0298 → 0.0301 |
| VAL | 63 | due | 14,481 | 0.532 → 0.532 | 0.830 → 0.830 | 0.0674 → 0.0674 |
| VAL | 63 | not due | 470 | 0.494 → 0.460 | 0.887 → 0.872 | 0.0614 → 0.0633 |

The other variants for VAL h = 5 due-window cov80: pool_bin 0.784, shrunk_bin 0.788, pool 0.786, event_har 0.794.

When the detector actually fired in the window, coverage is terrible for every model: h = 5 cov80 is 0.52 on DEV and
0.46 → 0.47 on VAL; h = 21 is 0.70. This is partly by construction, because the detector needs a large move.
All by-group rows are in `event_split.csv`.

### Paired difference event_har − vol_har_drift, in basis points of skill (ticker bootstrap, 90% CI)

| split | h | all | due | not due |
|---|---|---|---|---|
| DEV | 5 | +0.7 [−0.1, +1.5] | +4.3 [−0.2, +8.4] | −0.0 [−0.6, +0.6] |
| DEV | 21 | +0.9 [−0.4, +2.1] | +2.3 [−0.8, +5.5] | −0.1 [−1.1, +0.8] |
| DEV | 63 | +0.2 [−0.3, +0.6] | 0 | +7.8 [−14.8, +29.3] |
| VAL | 5 | +4.4 [+2.1, +6.8] | +13.1 [+1.5, +24.1] | +2.5 [+1.7, +3.4] |
| VAL | 21 | +2.0 [−0.7, +4.6] | +0.7 [−5.8, +7.5] | +2.9 [+1.7, +4.3] |
| VAL | 63 | +0.6 [+0.1, +1.1] | 0 | +19.0 [+3.2, +42.1] |

Control: deflating every window by the same d (0.975), with no event information, gives:
- DEV: −0.3 bp at h = 5 and −0.5 bp at h = 21.
- VAL: +1.9 bp at both horizons.

So the part of the gain that comes from event information is about +1.0 / +1.4 bp on DEV and +2.5 / +0.1 bp on VAL (h = 5 / 21).
The rest of VAL's not-due gain is VAL simply being calmer than HAR expects (`event_paired_ci.txt`).

## What it means

1. **The effect has the expected shape, but it is small.** The gain is largest at h = 5, small at 21 and essentially nil at 63:
   - On VAL, cov80 in "due" windows at 1W rises from 0.774 to 0.794 and pinball skill there improves by +13 bp (CI excludes 0).
   - Overall skill moves +0.0004 at 1W and +0.0002 at 1M on VAL, and +0.0001 on DEV. That is inside the ±0.001–0.007 noise
     band of the unpaired CIs. Only the paired VAL 1W gain is clearly positive.
2. **The bottleneck is knowing when the event is, not how big it is.** Windows that contain a detected event have about 2.5x
   the variance HAR forecasts, and the cone covers only about half of them at 80%. But from price and volume alone, a
   5-day window flagged "due" contains an event only 14–15% of the time (base rate 8–9%). Two things cause this:
   - The detector mixes earnings with other news and misses quiet earnings.
   - A company's report date can only be placed within about ±1 week from history.
   So the add-on that wins on pinball is a small one: about +7% variance on flagged 1W windows and +20–30% at 1M.
3. **Per-stock sizes help a little.** A stock's average event size persists: correlation 0.46 between its 2000–2011 and
   2012–2018 means across 40 tickers. Event sizes also grew over time (pool mean 9.1 → 13.2). The per-stock shrunk add-on
   is best on VAL h = 5 due windows (cov80 0.794 vs 0.786 pooled), and it adapts to the bigger modern moves.
4. **Worth shipping only as a minor refinement.** event_har is never worse than vol_har_drift on DEV or VAL at any
   horizon, keeps cov80 within 0.002–0.03 of target, and passes the look-ahead proof. The big gain would need a real
   earnings calendar (actual report dates). With true dates, the 2.5x-variance windows could be flagged at 1W with
   high precision. Since cone misses are mostly volatility surprises, that is where the real headroom is.

## Not verified

- **No ground-truth earnings dates.** Precision and recall are measured against the same detector, not against real
  report dates. The detector's real recall and purity are judged only by spot checks (AAPL, MSFT, KO, 2012–2018).
- **Scale.** The paired CIs resample whole tickers (40 on DEV, 48 on VAL), but rows overlap in time. Gains under about
  2 bp should be treated as noise.
- **Tuning.** c and d were fitted on DEV only, with one coarse grid. The detector and predictor settings (volume 2.0x,
  |z| 1.0, threshold 0.30, tolerances, 70-day gap) were picked by looking at both DEV and VAL periodicity diagnostics,
  so VAL is not fully clean for those choices.
- **h = 63 deflation.** The not-due windows (3% of rows, 470 on VAL) drive the h = 63 gain; that is a small sample.
- **The base model.** Only vol_har_drift was tried as the base; the add-on was not tested on top of `models_cone2.py`
  or any later cone.
- **Not run.** TEST was never run. `engine.py --selfcheck` was not run (as instructed). engine.py was not modified.
