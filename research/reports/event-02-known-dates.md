# event-02: what knowing the earnings date is worth (and shipping it for live cones)

> **Not financial advice.** This is a research log from building an educational tool. Backtest numbers are hypothetical, carry survivorship bias, and say nothing certain about the future. Nothing here is a recommendation to buy, sell or hold anything.

Lane `cone-earn`, 2026-10-04. Code: `research/models_event_oracle.py`. Outputs: `research/results/event2_*`.
The base model is the product cone `cone2_f_conf_dev` (`cone_model.py`). DEV was used for fitting, VAL to
confirm and to choose between two add-on forms. TEST was never opened.

## Method

**Two kinds of "known date".**
- **det (oracle):** the event-01 detector (`research/models_event.py`: a volume spike plus a move against the
  market). A window counts as "event" when the detector later fires inside it. This uses the outcome, because
  the detector needs a big move to fire, so it over-states what a calendar is worth. I ran it because the task
  asked for it, as an upper bound.
- **cal (real dates):** Yahoo's earnings history (`yfinance get_earnings_dates(limit=100)`). It was fetched once per
  ticker, sequentially, and cached in `data/events/history/<T>.json`. 69 of the 71 TRAIN tickers have dates
  (IRE and RVI have none), going back to about 2002. These are the dates the product actually uses.
  - Timestamps carry the time of day. A report at 16:00 or later counts on the next trading day; otherwise it
    counts on the report day itself.
  - Check of this rule: the bar it picks out-traded the other candidate bar in 72–98% of reports for the large
    caps, most above 85% (`event2_timing.csv`). It was lower for thin miners and small caps.
  - A window (as-of, as-of + h] is "event" when a reaction day falls inside it. The h trading days are projected
    from the as-of date with the engine's own weekday and holiday rule, which is exactly what a live forecast can do.
  - Rows count as "covered" from the first listed date on, minus listing gaps longer than 135 days. That is
    35.7% of DEV rows (2002–2011) and 97.0% of VAL rows.
  - All calendar results below use covered rows only.

**Add-on.** On event windows only, the cone's 19 quantiles are scaled around their own mean: Q' = m + f·(Q − m).
Every other window is the plain cone. Two forms were tried:
- **mult:** f = F[h], one factor per horizon.
- **var:** f = sqrt(1 + C[h]·(k̄ − 1)/h), where k̄ is the stock's shrunk mean event-day variance ratio.
  - Each event's ratio is r_e²/s_e². r_e is the reaction-day log return. s_e² is the robust variance of the 63
    bars before it (median of squares / 0.455).
  - k̄ uses the stock's last 8 events dated on or before the as-of date, each capped at 40, shrunk toward a pool
    value with 4 pseudo-events.
  - The pool value is 6.3, the capped mean of 1,422 DEV-dated calendar reactions (their median is only 2.2: the
    tail is heavy).

F and C were grid-searched per horizon by the harness's scaled pinball on DEV event rows (F step 0.01, C step
0.02). Non-event rows are not changed, so the event-row optimum is also the overall optimum.

| fitted on DEV | F (mult) 5 / 21 / 63 | C (var) 5 / 21 / 63 |
|---|---|---|
| det | 1.71 / 1.23 / 1.05 | 1.40 / 1.50 / 0.92 |
| cal | 1.36 / 1.08 / 1.00 | 0.80 / 0.62 / 0.22 |

**Frozen product add-on** (`FROZEN` in `models_event_oracle.py`, copied to `cone_model.EVENT`):
- **cal var, with C = 0.80 / 0.62 / 0 at 1W / 1M / 3M.**
- "var" beats "mult" on VAL at 5 and 21 days and does not lose on DEV.
- At 63 days, mult fits F = 1.00 on DEV and var loses on VAL (−2.8 bp, CI excludes 0), so the add-on is off at
  3M. 97% of 63-day windows hold an earnings day anyway, and the cone is already calibrated on them.

**Checks.**
- The panel's plain cone reproduces the harness JSON of `cone2_f_conf_dev` exactly (n, skill, cov50, cov80) at
  every horizon on DEV and VAL (`models_event_oracle.py check`).
- Harness model `event2_cal` = `cone2_f_conf_dev` + the frozen add-on, reading only the cached history.
  `harness.py --lookahead --models event2_cal` gives OK on all 5 standard cases.
- Five more cases whose windows do hold a report also pass: AAPL 2015-07-17, MSFT 2016-04-18, KO 2008-10-10,
  NVDA 2011-05-10, INTC 2014-07-14. The 1W cone was widened 1.42–1.80x and the 1M cone 1.09–1.19x in those cases
  (`event2_log_lookahead.txt`).
- Harness CLI, VAL, all 14,951 rows (uncovered rows stay plain): event2_cal skill is **+0.0184 / +0.0285 / +0.0633**,
  against cone2_f_conf_dev's +0.0128 / +0.0272 / +0.0633. cov80 is 0.819 / 0.820 / 0.819 (base 0.806 / 0.806 / 0.819).
  Brier skill is unchanged to 4 decimals (`event2_cal_val.json`).

## Oracle results (DEV and VAL)

How much bigger the realised moves were than the plain cone expected: (r − mean)² / s², where s is the 80% band
half-width / 1.28.

| | 1W event windows | 1W no-event windows | 1M event | 1M no-event |
|---|---|---|---|---|
| cal, DEV | 2.17 | 1.02 | 1.32 | 0.98 |
| cal, VAL | 2.89 | 1.02 | 1.41 | 0.91 |
| det, DEV | 2.89 | 0.95 | 1.61 | 0.82 |
| det, VAL | 3.92 | 0.95 | 1.77 | 0.76 |

In 5-day windows that hold a real report, the detector fired only 32% (DEV) and 41% (VAL) of the time.

### Calendar (real dates), plain → shipped add-on (cal var; 3M unchanged)

| split | h | group | n | cov50 | cov80 | pinball skill |
|---|---|---|---|---|---|---|
| DEV | 5 | event | 1,349 | 0.363 → 0.497 | 0.674 → 0.804 | +0.0028 → +0.0204 |
| DEV | 5 | no event | 15,908 | 0.510 | 0.816 | +0.0084 |
| DEV | 5 | all | 17,257 | 0.499 → 0.509 | 0.805 → 0.815 | +0.0078 → +0.0097 |
| DEV | 21 | event | 5,683 | 0.471 → 0.500 | 0.763 → 0.794 | +0.0097 → +0.0111 |
| DEV | 21 | no event | 11,574 | 0.523 | 0.826 | +0.0132 |
| DEV | 21 | all | 17,257 | 0.506 → 0.515 | 0.805 → 0.816 | +0.0119 → +0.0124 |
| DEV | 63 | event / no event / all | 16,737 / 520 / 17,257 | 0.496 / 0.527 / 0.497 | 0.804 / 0.850 / 0.805 | +0.0188 / +0.0228 / +0.0189 |
| VAL | 5 | event | 1,151 | 0.329 → 0.436 | 0.572 → 0.735 | +0.0063 → +0.0490 |
| VAL | 5 | no event | 13,357 | 0.537 | 0.826 | +0.0140 |
| VAL | 5 | all | 14,508 | 0.521 → 0.529 | 0.806 → 0.819 | +0.0129 → +0.0187 |
| VAL | 21 | event | 4,820 | 0.443 → 0.482 | 0.741 → 0.785 | +0.0249 → +0.0283 |
| VAL | 21 | no event | 9,688 | 0.548 | 0.841 | +0.0296 |
| VAL | 21 | all | 14,508 | 0.513 → 0.526 | 0.808 → 0.822 | +0.0278 → +0.0291 |
| VAL | 63 | event / no event / all | 14,090 / 418 / 14,508 | 0.518 / 0.572 / 0.520 | 0.819 / 0.861 / 0.821 | +0.0658 / +0.0420 / +0.0652 |

The cal mult form gives slightly less in event windows:
- 1W cov80: 0.807 on DEV and 0.709 on VAL. Skill: +0.0192 on DEV and +0.0391 on VAL.
- 1M cov80: 0.796 on DEV and 0.776 on VAL.

### Detector oracle (det mult), plain → add-on

| split | h | event cov50 | event cov80 | event skill | overall skill | overall cov80 |
|---|---|---|---|---|---|---|
| DEV | 5 | 0.271 → 0.458 | 0.524 → 0.826 | +0.0012 → +0.0508 | +0.0074 → +0.0155 | 0.803 → 0.832 |
| DEV | 21 | 0.406 → 0.493 | 0.707 → 0.807 | +0.0087 → +0.0176 | +0.0138 → +0.0180 | 0.801 → 0.840 |
| DEV | 63 | 0.477 → 0.499 | 0.777 → 0.797 | +0.0202 → +0.0208 | +0.0260 → +0.0265 | 0.800 → 0.815 |
| VAL | 5 | 0.236 → 0.387 | 0.452 → 0.741 | −0.0019 → +0.0653 | +0.0128 → +0.0238 | 0.806 → 0.829 |
| VAL | 21 | 0.392 → 0.476 | 0.687 → 0.781 | +0.0109 → +0.0230 | +0.0272 → +0.0324 | 0.806 → 0.837 |
| VAL | 63 | 0.477 → 0.500 | 0.782 → 0.803 | +0.0511 → +0.0516 | +0.0633 → +0.0637 | 0.819 → 0.834 |

### Paired skill gain, add-on − plain, in basis points (ticker bootstrap, 90% CI)

| split | h | det mult, event | det mult, all | cal mult, event | cal mult, all | **cal var, event** | **cal var, all** |
|---|---|---|---|---|---|---|---|
| DEV | 5 | +496 [453, 536] | +81 [71, 91] | +164 [112, 212] | +18 [12, 24] | **+177 [124, 227]** | **+19 [13, 26]** |
| DEV | 21 | +89 [70, 105] | +42 [32, 51] | +13 [4, 22] | +4.7 [1.3, 8.0] | **+14 [3, 24]** | **+5.1 [1.0, 8.9]** |
| DEV | 63 | +5.7 [0.3, 10.7] | +4.5 [0.2, 8.6] | 0 | 0 | +0.3 [−1.2, 1.8] | +0.3 [−1.2, 1.7] |
| VAL | 5 | +672 [610, 725] | +110 [95, 126] | +328 [280, 370] | +44 [35, 53] | **+427 [343, 502]** | **+58 [44, 72]** |
| VAL | 21 | +121 [90, 149] | +52 [39, 65] | +30 [19, 40] | +11 [7, 16] | **+34 [15, 53]** | **+13 [6, 21]** |
| VAL | 63 | +4.6 [−3.0, 11.7] | +3.5 [−2.3, 9.1] | 0 | 0 | −2.8 [−5.0, −0.8] | −2.7 [−4.9, −0.8] |

(det var: 1W event +480 / +718 bp, 1M +81 / +119 bp on DEV / VAL.) Full tables: `event2_split.csv` and `event2_paired.csv`.

## What it means

1. **Knowing the date is worth a lot at 1W, some at 1M, and nothing at 3M.**
   - With real dates, 1W windows that hold a report carry 2.2–2.9x the variance the cone expects. The plain cone
     covers only 57–67% of them at 80%.
   - The shipped add-on lifts that to 0.80 on DEV and 0.74 on VAL.
   - Overall 1W skill rises by +19 bp on DEV and +58 bp on VAL, with both CIs well clear of 0. event-01's guessed
     dates gave +0.7 / +4.4 bp.
   - At 1M the gain is +5 / +13 bp overall. At 3M almost every window holds a report, so there is nothing left to separate.
2. **The detector oracle over-states the gain about 2–3x**, because a window only counts when a big move happened.
   The real calendar is the honest estimate, and it is still large.
3. **Per-stock size matters.** The per-stock form (var) beats the single factor on VAL: +427 vs +328 bp in 1W
   event windows. Some stocks move far more on earnings than others; MSFT's k̄ is about 20, against the pool's 6.3.
4. **VAL is still under-covered in event windows** (1W cov80 0.735). Reactions grew from DEV to VAL (median k̄ 5.3 →
   7.7). The add-on was not refit on VAL, by design. It narrows the gap but does not close it.
5. Narrowing the cone when a date is known to fall outside the window helps a little: the DEV fit gives 0.97 / 0.94 / 0.91
   at 1W / 1M / 3M, for pinball gains of only 0.02–0.24%. It is not shipped. The contract's `event` is null in that
   case, and the gain is small.

## Product change (shipped: the VAL gain is clear in event windows on both DEV and VAL)

**`cone_model.py`**
- `EVENT` holds the frozen literals.
- `event_kbar(day, logc, reactions, asof_day)` reads bars on or before the as-of date only.
- `event_widen(h, kbar)` = sqrt(1 + C[h]·(k̄ − 1)/h).
- `widen_quantiles(Q, f)` scales the spread around the mean.
- `daily_paths(..., event_step, widen)`: with no event the code path is unchanged. With an event, the extra variance
  (f² − 1)·h·s² is added **from the reaction day onward**. Days before it are exactly the plain cone's path, and day h
  equals the widened quantiles exactly.

**`engine.py`**
- `earnings_info(t)` makes one `get_earnings_dates(limit=16)` request per ticker per New York day.
  - It runs in a background thread, serialised by a lock, and is cached in `data/events/<T>.json`.
  - A failed lookup is cached too and retried after an hour.
  - A live request waits at most 3 s.
- `live_event(td, p, h)`:
  - Takes the first reaction day after the as-of date, using the same 16:00 rule.
  - If that day falls within the h projected trading days, it computes k̄ from the listed past reports and the stock's
    own bars, then returns `event = {type: "earnings", date, in_window: true, widen}` and a note.
  - Otherwise `event` is null. A note is added only when the lookup failed or is still running (for example
    "Yahoo lists no earnings dates for SPY"), never an exception.
  - At 3M, `widen` is 1.0, with a note that the 3M cone already allows for one earnings day.
- `forecast()` calls it only when p is the latest bar and the engine is the live one (`raw is None`).
- `_forecast_at(td, p, h, event=None)` and everything walk-forward (`walk_forward`, `track`, `calibrate.py`,
  `accuracy.py`) never see it. Replays get `"event": null`.

**Verification**
- `python engine.py --selfcheck`: **SELF-CHECK PASSED** (30 OK lines, run on the final code). `python cone_model.py --vs-research` after the change: VS-RESEARCH PASSED, 54 samples, max abs diff 0.
  - The 18 original look-ahead lines all pass.
  - The new `event_selfcheck` checks: (a) the event path on NVDA 2024-06-14 (h 5/21, event on day 3, 1 or 21) is
    identical before the event day, exact at day h, keeps bands ordered, keeps the spread ratio equal to f, and serves
    the day-h prices; (b) k̄ is bit-identical between the research code and `cone_model` (AAPL, KO, MSFT);
    (c) live lookups for AAPL, MSFT and SPY return valid structures.
- **Replay/track no-change proof:** `/api/forecast` was captured from a running `server.py` before and after the change.
  - Replays: NVDA 2024-06-14 at h 5/21/63, AAPL 2025-07-28 at h 5/21 (an earnings week, correctly not widened),
    KO 2008-10-10 h21, MSFT 2026-09-25 h5, TSLA 2026-09-30 h21. Every original key is identical. The only change is
    the added `"event": null`.
  - The `track` block of 10 live responses is identical.
  - Live responses whose window holds no report (AAPL 1W, NVDA 1M, TSLA 1W) are identical apart from `event`. SPY
    also gained only the note.
- **Live, as of 2026-10-02:**

  | ticker | h | earnings | widen | 80% end band before → after |
  |---|---|---|---|---|
  | PEP | 1W | 2026-10-08, before the open | 1.66 | (new request) 118.59–134.53 |
  | MSFT | 1M | 2026-10-28, after the close | 1.25 | 465.06–586.81 → 451.22–604.12 |
  | NFLX | 1M | 2026-10-20 | 1.23 | |
  | IBM | 1M | 2026-10-21 | 1.19 | 184.54–272.89 → 177.59–283.24 |
  | TSLA | 1M | 2026-10-21 | 1.11 | 315.19–442.09 → 309.16–450.39 |
  | KO | 1M | 2026-10-27 | 1.11 | 80.51–92.83 → 79.89–93.52 |
  | GS | 1M | 2026-10-13 | 1.08 | |
  | AAPL | 1M | 2026-10-29, after the close | 1.07 | 306.36–369.67 → 304.17–372.21 |
  | JPM | 1M | 2026-10-13 | 1.06 | 310.07–362.55 → 308.54–364.25 |
  | TSM | 1M | 2026-10-15 | 1.06 | |

  - The 1W cone is unchanged for AAPL, TSLA, TSM and JNJ: no report in Oct 5–9.
  - In the six widened cases with a before snapshot (AAPL, IBM, JPM, KO, MSFT, TSLA), the log-width ratio at day h
    equals `widen`, and the bands only start to differ on the reaction day. Example: MSFT, after-close report on 10-28, bands differ from 10-29.
  - p_up moves by under 0.01, because widening around the mean shifts the median slightly.

## Not verified

- **Dates known in advance.** The research uses Yahoo's record of when reports actually happened. Live, the next
  date can be an estimate that later moves, and companies confirm dates only 2–5 weeks ahead. How often Yahoo's
  advance date is wrong was not measured. A wrong date widens the wrong window: at worst the 1W cone is a bit too
  wide, or too narrow in the week the report actually falls.
- **History coverage.** Only TRAIN tickers. The calendar results cover 2002–2011 on DEV (35.7% of DEV rows) and
  2012–2018 on VAL, with listing gaps over 135 days excluded. Yahoo's history for miners and small caps is gappy, and
  the time-of-day rule is weaker there (it picks the higher-volume bar only 36–70% of the time for OLB, O, RIOT,
  UEC, CDE and KGC).
- **Live k̄ uses at most about 12 past reports** (limit = 16, including upcoming dates). The research uses 8, so
  this is the same whenever 8 or more are listed. A young stock shrinks toward the pool.
- **Tuning.** C and F were fitted on DEV only, on one grid. The choice of var over mult and of C[63] = 0 used VAL.
  The 135-day gap rule, the 16:00 rule and the k̄ settings (8 reports, 4 pseudo-events, cap 40) were set a priori,
  copied from event-01. Rows overlap in time; the CIs resample whole tickers (40 on DEV, 48 on VAL).
- **Not run.** TEST. The harness CLI on DEV for `event2_cal` (only VAL was run).
- **Not exercised.** The UI for the `event` field (another lane). The fetch-timeout path, when Yahoo is slow for
  more than 3 s: the code returns null with a "still being looked up" note. A failure was exercised only through
  SPY's empty list.
