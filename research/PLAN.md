# Overnight accuracy research — shared plan (2026-10-03/04)

Owen: "do whatever you want to make this the most accurate and whatever you think is
best by yourself all night while I sleep."

Starting point (v0 = commit 4d9b581): the 80% cone covers 79% / 77% / 76% at 1W / 1M / 3M
on the 155-stock scoreboard; direction is worse than "always guess the stock's usual
direction"; the edge over a naive volatility cone is mostly upward drift inherited from
a survivor-heavy analog pool.

## The rule that matters most: don't fool ourselves

Many experiments tonight will be tried. Picking the best of many on the same data is
overfitting. So the data is split three ways and the TEST split is opened only by the
coordinator, at most twice (a mid-night sanity check and the final report).

| split | tickers | dates (as-of) | who may look |
|---|---|---|---|
| DEV | TRAIN tickers | before 2012-01-01 | every lane, freely |
| VAL | TRAIN tickers | 2012-01-01 .. 2018-12-31 | every lane; choose between variants here |
| TEST | ALL tickers (TRAIN + TEST tickers) | 2019-01-01 .. latest | coordinator only |

- TRAIN vs TEST tickers: `research/splits.py` decides, deterministically
  (md5 of the ticker, even last hex digit = TRAIN). The analog POOL is always allowed
  as a source of analogs on every split (time rules still apply: an analog dated A is
  usable only if A + horizon <= as-of).
- A variant is "better" only if it wins on VAL, and does not lose on DEV.
- No look-ahead anywhere. `python engine.py --selfcheck` must keep passing.

## Metrics (every lane reports the same ones, via `research/harness.py`)

Forecast target: end-of-horizon log return r = ln(P[t+h] / P[t]), horizons 5 / 21 / 63.
A model returns 19 quantiles of r (q = 0.05, 0.10, ..., 0.95).

1. **Scaled pinball (primary)**: mean pinball loss over the 19 quantiles, each forecast
   divided by its naive sigma (trailing 63-day daily vol x sqrt(h)) so a calm stock
   counts as much as a wild one. Skill = 1 - model / naive. Higher is better.
2. **Coverage**: share of outcomes inside the central 50% (q25..q75) and 80% (q10..q90)
   intervals. Target 0.50 / 0.80. Report the absolute error.
3. **Width**: mean 80% interval width in r units (sharper is better only when calibrated).
4. **Direction, done fairly**: Brier score of P(r > 0) against a walk-forward base rate
   (the share of up-moves over the stock's own history BEFORE the as-of date). Skill =
   1 - Brier_model / Brier_base. Also the plain hit rate of the median's sign vs the
   walk-forward majority guess. (v0's scoreboard baseline used the majority direction
   over the TESTED dates, i.e. hindsight; flag but don't change the scoreboard yet.)

Baselines always reported alongside: `naive0` (random walk, zero drift, trailing vol),
`naive_drift` (same plus the stock's own walk-forward mean log return), and `v0`
(the current engine).

## Ideas queue (coordinator assigns; one lane = one file)

1. Volatility forecast quality (the width of the cone is mostly a vol forecast).
2. Drift: remove the survivor drift; what centre is best (zero, own history, market, shrunk).
3. Conformal / walk-forward width recalibration per horizon (fix 77% / 76% coverage).
4. Analog search: K, feature weights, distance, similarity weighting, pool size.
5. Ensemble: blend analog quantiles with the naive cone; weights chosen on VAL.
6. Event risk: widen around earnings (big-gap history) if data allows.
7. Direction: any real signal after a fair baseline? Report honestly if none.

## Rules for lanes

- Own only the files named in your prompt. New code goes in `research/` as its own file.
  Only the final integration lane edits `engine.py`.
- Python: `python`
  (bare `python` in Git Bash is a broken Store stub). stdlib + numpy + pandas + yfinance.
- Reuse the engine's cached data in `data/prices/`; don't re-download what is cached.
- Save partial results as you go (a Windows reboot or a crash should not cost the run).
- Reports go in `research/reports/` as markdown: Method, Results (tables), What it means,
  Not verified.
