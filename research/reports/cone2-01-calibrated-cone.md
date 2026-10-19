# cone2-01: a calibrated cone (centre + HAR width + calibration + analog blend)

> **Not financial advice.** This is a research log from building an educational tool. Backtest numbers are hypothetical, carry survivorship bias, and say nothing certain about the future. Nothing here is a recommendation to buy, sell or hold anything.

Lane `cone-combo`, 2026-10-03/04. Code: `research/models_cone2.py` (final model `cone2`,
parameters frozen as literals between `# FROZEN FINAL` markers). Data outputs:
`research/results/cone2_*`. DEV and VAL only; TEST was never opened.

## Method

**One code path.** `ticker_rows(ctx, ticker, positions)` builds every causal input a variant
can use, at each (ticker, as-of). The research driver runs it over the harness grid (every 5
bars, the same `harness.grid_positions`) and stores it in `cone2_panel_{dev,val}.npz`
(48,277 DEV rows over 40 tickers; 14,951 VAL rows over 48 tickers, the harness's exact
counts). Variants are scored with the harness's own `summarise`/`metrics` (ticker-bootstrap
CI, vol terciles). The harness models (`Cone2Model`) call the same functions. Two checks:
on the panel, naive_drift, v0 and vol_har_drift reproduce the harness JSONs exactly (skill,
cov80, Brier, n). The harness CLI run of `cone2` gives the same numbers as the panel.

**Inputs (all causal):**
- **Width base:** the frozen HAR sigma of `volforecast.py` (`har_mx_QLr`, market from SPY/^GSPC), times sqrt(h).
- **Centres:**
  - zero.
  - The stock's own walk-forward mean daily log return (`own`).
  - The index walk-forward mean (`mkt`; ^GSPC price-only spliced to SPY).
  - The pooled walk-forward mean of every daily return of the 42-stock analog basket dated <= as-of (`pool`, about 12%/yr).
  - A DEV-fitted constant (`const`).
  - James-Stein shrinkage of own toward pool, toward mkt, or toward the analog median. The weight on own is tau^2 / (tau^2 + s^2/n), where s^2/n is the variance of the stock's own mean.
  - v0's analog median as is (`v0med`).
  - v0's median minus the scaling-inflated basket drift (`v0deb`).
  - The **unscaled** analog median and mean (`amed`, `amean`).
  - Fixed blends.
- **Analog statistics:** from exactly v0's selection (`engine._select`, K = 250, time rule A + h <= as-of, Gaussian weights). They are v0's own 19 quantiles, the unscaled weighted median and mean of the analog end returns, and the **deviation-scaled cone** `med + ratio x (raw - med)`, which applies vol scaling to the spread only, not the drift (the diag-01 fix). My recomputed v0 quantiles equal the harness's cached v0 bit for bit. These statistics are memoised on disk in `cone2_analog/` for full contexts only; a truncated context, as in the look-ahead proof, always recomputes.
- **Walk-forward conformal statistics:**
  - Reference residual: e = |r_h - mu_own·h| / (sigma_HAR·sqrt h).
  - `own`: the mean of the stock's own e over windows whose outcome ended in the last W bars (W = 252 or 756).
  - `pool`: the mean e over the 42 basket stocks, for windows that ended in the last 91 or 365 calendar days.
  - The width is multiplied by stat^lambda.
- **Regime variables:** log(sigma_HAR / own long-run vol); the absolute vol level; vol relative to the basket's mean log vol that day; 252-day drawdown; 126-day trend.

**Shape.** z = the 19 weighted quantiles of DEV standardised residuals (r - centre) / s,
with weights s / (sig63·sqrt h), so they minimise the harness's scaled pinball. Four modes:
- `fit`: location included.
- `mean0`: the mean of the 19 quantiles is removed, so the centre alone sets the location.
- `med0`: the median is removed.
- `vf_centred`: `volforecast.z_quantiles(h, "centred")`, not refit.

Regime variants fit one z per DEV tercile of the regime variable.

**Selection discipline.** Every continuous setting (tau, constant, floor strength, conformal
window and lambda, ensemble weight, regime edges, z) is chosen **per horizon on DEV**, by DEV
scaled pinball. Choices between named variants use VAL, subject to not losing on DEV. Five stages:
1. 40 centres (HAR width, DEV shape).
2. 15 width, shape and ensemble variants on the DEV-best centre (JS toward pool).
3. 4 combinations of the stage-2 winners.
4. The stage-3 recipe on own-mean centres (2 variants).
5. 3 per-regime spread factors applied after the blend.

## Final model `cone2` (= stage-4 `cone2_g_own`)

    plain_h = mu_own·h + s_h · z_h          z_h: DEV weighted quantiles, mean removed (mean0)
    s_h     = sigma_HAR,h · sqrt(h) · own_stat^lo · pool_stat^lp
    Q_h     = (1 - w_h) · plain_h + w_h · Qdev_h        Qdev = deviation-scaled analog cone
    p_up    = read off Q (linear between quantiles)

| h | own conformal (W = 756 bars) lambda | basket conformal (365 days) lambda | analog weight w |
|---|---|---|---|
| 5 | 0.30 | 0.25 | 0.3 |
| 21 | 0.15 | 0 | 0.3 |
| 63 | 0 | 0 | 0.5 |

If fewer than 100 analogs are eligible, Qdev falls back to the plain cone, so the model never
abstains (n = 48,277 / 14,951, 0 abstained).

## Results

### Headline: scaled-pinball skill vs naive0 (harness CLI runs)

| split | h | naive_drift | v0 | vol_har_drift | cone2_f_conf_dev (pool-centre alternative) | **cone2** | cone2 cov50 / cov80 |
|---|---|---|---|---|---|---|---|
| DEV | 5 | +0.0007 | -0.0008 | +0.0054 | +0.0074 | **+0.0069** | 0.504 / 0.805 |
| DEV | 21 | +0.0040 | -0.0011 | +0.0095 | +0.0138 | **+0.0113** | 0.500 / 0.800 |
| DEV | 63 | +0.0069 | +0.0065 | +0.0124 | +0.0260 | **+0.0202** | 0.500 / 0.796 |
| VAL | 5 | +0.0062 | +0.0051 | +0.0124 | +0.0128 | **+0.0136** | 0.524 / 0.807 |
| VAL | 21 | +0.0226 | +0.0127 | +0.0288 | +0.0272 | **+0.0298** | 0.519 / 0.811 |
| VAL | 63 | +0.0634 | +0.0421 | +0.0673 | +0.0633 | **+0.0687** | 0.518 / 0.815 |

cone2 ticker-bootstrap 90% CIs vs naive0: VAL [+0.0116, +0.0156], [+0.0227, +0.0372],
[+0.0497, +0.0896]; DEV [+0.0063, +0.0077], [+0.0091, +0.0138], [+0.0141, +0.0269].
- **Direction (cone2):** Brier skill is VAL +0.0031 / +0.0018 / +0.0024 and DEV +0.0028 / +0.0028 / +0.0109. The median-sign hit rate is below the walk-forward majority guess on VAL (0.552 vs 0.554, 0.582 vs 0.591, 0.628 vs 0.650) and above it on DEV (0.518 vs 0.510, 0.554 vs 0.550, 0.586 vs 0.580).
- **Width (80% interval):** VAL 0.100 / 0.208 / 0.365, against v0's 0.102 / 0.205 / 0.354.

### Paired differences (cone2 minus X, scaled-pinball skill, 90% ticker-bootstrap, same rows)

| split | h | cone2 - v0 | cone2 - naive_drift | cone2 - vol_har_drift | cone2 - pool-centre alt |
|---|---|---|---|---|---|
| DEV | 5 | +0.0078 [+0.0070, +0.0085] | +0.0062 [+0.0056, +0.0069] | +0.0015 [+0.0012, +0.0019] | -0.0005 [-0.0007, -0.0003] |
| DEV | 21 | +0.0125 [+0.0109, +0.0140] | +0.0073 [+0.0064, +0.0084] | +0.0018 [+0.0011, +0.0026] | -0.0025 [-0.0034, -0.0015] |
| DEV | 63 | +0.0139 [+0.0096, +0.0180] | +0.0133 [+0.0100, +0.0174] | +0.0078 [+0.0047, +0.0116] | -0.0058 [-0.0076, -0.0040] |
| VAL | 5 | +0.0085 [+0.0070, +0.0101] | +0.0073 [+0.0063, +0.0083] | +0.0012 [+0.0005, +0.0019] | +0.0008 [+0.0001, +0.0014] |
| VAL | 21 | +0.0171 [+0.0138, +0.0205] | +0.0073 [+0.0057, +0.0088] | +0.0011 [+0.0001, +0.0021] | +0.0026 [+0.0004, +0.0048] |
| VAL | 63 | +0.0267 [+0.0184, +0.0346] | +0.0053 [+0.0011, +0.0098] | +0.0015 [-0.0021, +0.0053] | +0.0054 [+0.0005, +0.0101] |

### Coverage and skill by volatility tercile (terciles of as-of sig63, as in the harness)

| split | h | model | cov50 low / mid / high | cov80 low / mid / high | skill low / mid / high |
|---|---|---|---|---|---|
| DEV | 5 | cone2 | 0.498 / 0.500 / 0.516 | 0.793 / 0.800 / 0.822 | +0.006 / +0.006 / +0.009 |
| DEV | 5 | v0 | 0.483 / 0.506 / 0.530 | 0.783 / 0.805 / 0.835 | -0.002 / +0.000 / -0.001 |
| DEV | 5 | vol_har_drift | 0.507 / 0.493 / 0.498 | 0.802 / 0.792 / 0.803 | +0.005 / +0.004 / +0.008 |
| DEV | 21 | cone2 | 0.489 / 0.494 / 0.516 | 0.787 / 0.801 / 0.813 | +0.014 / +0.010 / +0.009 |
| DEV | 21 | v0 | 0.460 / 0.488 / 0.519 | 0.759 / 0.789 / 0.817 | +0.005 / -0.001 / -0.008 |
| DEV | 21 | vol_har_drift | 0.498 / 0.492 / 0.506 | 0.799 / 0.799 / 0.800 | +0.013 / +0.008 / +0.006 |
| DEV | 63 | cone2 | 0.482 / 0.497 / 0.521 | 0.790 / 0.793 / 0.805 | +0.033 / +0.015 / +0.011 |
| DEV | 63 | v0 | 0.464 / 0.488 / 0.511 | 0.763 / 0.781 / 0.795 | +0.028 / -0.002 / -0.010 |
| DEV | 63 | vol_har_drift | 0.494 / 0.495 / 0.512 | 0.807 / 0.796 / 0.802 | +0.025 / +0.011 / -0.001 |
| VAL | 5 | cone2 | 0.504 / 0.543 / 0.525 | 0.788 / 0.816 / 0.818 | +0.015 / +0.014 / +0.011 |
| VAL | 5 | v0 | 0.481 / 0.541 / 0.537 | 0.763 / 0.815 / 0.827 | +0.005 / +0.008 / +0.002 |
| VAL | 5 | vol_har_drift | 0.518 / 0.544 / 0.519 | 0.805 / 0.822 / 0.809 | +0.015 / +0.012 / +0.010 |
| VAL | 21 | cone2 | 0.484 / 0.549 / 0.523 | 0.784 / 0.830 / 0.818 | +0.051 / +0.021 / +0.012 |
| VAL | 21 | v0 | 0.442 / 0.533 / 0.515 | 0.739 / 0.814 / 0.814 | +0.032 / +0.009 / -0.008 |
| VAL | 21 | vol_har_drift | 0.501 / 0.560 / 0.516 | 0.804 / 0.833 / 0.811 | +0.049 / +0.021 / +0.011 |
| VAL | 63 | cone2 | 0.463 / 0.566 / 0.526 | 0.789 / 0.848 / 0.807 | +0.107 / +0.070 / +0.018 |
| VAL | 63 | v0 | 0.412 / 0.542 / 0.508 | 0.728 / 0.830 / 0.797 | +0.074 / +0.061 / -0.019 |
| VAL | 63 | vol_har_drift | 0.494 / 0.571 / 0.528 | 0.818 / 0.866 / 0.811 | +0.109 / +0.062 / +0.018 |

### Every variant tried (panel scoring = harness scoring)

DEV numbers are partly in-sample: z, the ensemble weights and the grid values were fit on DEV.
That holds for every variant alike, so DEV comparisons between variants are fair, but absolute
DEV levels are slightly optimistic. Names: `c_` = centre (stage 1), `w_` = width,
`s_` = shape, `e_` = ensemble (stage 2, on the JS-to-pool centre), `f_` = combinations
(stage 3), `g_own` and `alt` = own-mean combinations (stage 4), `p_` = post-blend regime
factor (stage 5). Shape suffixes: fit / mean0 / med0 / vf_centred.

#### DEV (all variants; 5 / 21 / 63 days)

| variant | stage | pinball skill | cov50 | cov80 | width80 |
|---|---|---|---|---|---|
| naive0 | base | +0.0000 / +0.0000 / +0.0000 | 0.530 / 0.514 / 0.502 | 0.809 / 0.805 / 0.791 | 0.134 / 0.274 / 0.474 |
| naive_drift | base | +0.0007 / +0.0040 / +0.0069 | 0.529 / 0.520 / 0.514 | 0.810 / 0.806 / 0.797 | 0.134 / 0.274 / 0.474 |
| v0 | base | -0.0008 / -0.0011 / +0.0065 | 0.507 / 0.489 / 0.488 | 0.808 / 0.789 / 0.779 | 0.132 / 0.260 / 0.446 |
| vol_har | base | +0.0064 / +0.0138 / +0.0264 | 0.499 / 0.499 / 0.501 | 0.800 / 0.799 / 0.800 | 0.125 / 0.255 / 0.448 |
| vol_har_drift | base | +0.0054 / +0.0095 / +0.0124 | 0.499 / 0.498 / 0.500 | 0.799 / 0.799 / 0.801 | 0.126 / 0.256 / 0.455 |
| c_zero_fit | 1 | +0.0064 / +0.0138 / +0.0264 | 0.499 / 0.500 / 0.501 | 0.800 / 0.800 / 0.800 | 0.125 / 0.256 / 0.447 |
| c_own_fit | 1 | +0.0054 / +0.0095 / +0.0124 | 0.499 / 0.500 / 0.501 | 0.800 / 0.800 / 0.800 | 0.126 / 0.257 / 0.453 |
| c_amed_fit | 1 | +0.0021 / +0.0057 / +0.0167 | 0.499 / 0.500 / 0.501 | 0.800 / 0.800 / 0.800 | 0.126 / 0.258 / 0.452 |
| c_zero_mean0 | 1 | +0.0048 / +0.0067 / +0.0082 | 0.498 / 0.494 / 0.483 | 0.799 / 0.794 / 0.784 | 0.125 / 0.256 / 0.447 |
| c_own_mean0 | 1 | +0.0054 / +0.0095 / +0.0124 | 0.499 / 0.499 / 0.501 | 0.800 / 0.800 / 0.800 | 0.126 / 0.257 / 0.453 |
| c_amed_mean0 | 1 | +0.0021 / +0.0055 / +0.0154 | 0.499 / 0.501 / 0.501 | 0.800 / 0.800 / 0.801 | 0.126 / 0.258 / 0.452 |
| c_zero_med0 | 1 | +0.0046 / +0.0030 / +0.0009 | 0.498 / 0.490 / 0.478 | 0.798 / 0.791 / 0.779 | 0.125 / 0.256 / 0.447 |
| c_own_med0 | 1 | +0.0054 / +0.0089 / +0.0119 | 0.499 / 0.498 / 0.498 | 0.800 / 0.798 / 0.799 | 0.126 / 0.257 / 0.453 |
| c_amed_med0 | 1 | +0.0021 / +0.0057 / +0.0165 | 0.499 / 0.500 / 0.501 | 0.800 / 0.800 / 0.801 | 0.126 / 0.258 / 0.452 |
| c_zero_vf_centred | 1 | +0.0051 / +0.0084 / +0.0119 | 0.498 / 0.493 / 0.487 | 0.799 / 0.794 / 0.788 | 0.125 / 0.255 / 0.448 |
| c_own_vf_centred | 1 | +0.0054 / +0.0095 / +0.0122 | 0.499 / 0.499 / 0.496 | 0.799 / 0.797 / 0.796 | 0.125 / 0.255 / 0.448 |
| c_amed_vf_centred | 1 | +0.0021 / +0.0051 / +0.0141 | 0.498 / 0.497 / 0.500 | 0.798 / 0.796 / 0.798 | 0.125 / 0.255 / 0.448 |
| c_mkt_mean0 | 1 | +0.0060 / +0.0119 / +0.0222 | 0.498 / 0.497 / 0.494 | 0.799 / 0.796 / 0.793 | 0.125 / 0.256 / 0.446 |
| c_pool_mean0 | 1 | +0.0064 / +0.0135 / +0.0259 | 0.499 / 0.501 / 0.502 | 0.800 / 0.801 / 0.801 | 0.126 / 0.256 / 0.446 |
| c_amean_mean0 | 1 | +0.0012 / +0.0041 / +0.0146 | 0.499 / 0.500 / 0.501 | 0.800 / 0.800 / 0.801 | 0.126 / 0.258 / 0.453 |
| c_v0med_mean0 | 1 | +0.0008 / +0.0018 / +0.0065 | 0.499 / 0.501 / 0.501 | 0.800 / 0.801 / 0.801 | 0.126 / 0.259 / 0.456 |
| c_v0deb_mean0 | 1 | +0.0010 / +0.0030 / +0.0103 | 0.499 / 0.500 / 0.501 | 0.799 / 0.800 / 0.801 | 0.126 / 0.259 / 0.454 |
| c_const_mean0 | 1 | +0.0066 / +0.0144 / +0.0283 | 0.499 / 0.500 / 0.501 | 0.800 / 0.800 / 0.800 | 0.125 / 0.256 / 0.446 |
| c_jspool_mean0 | 1 | +0.0064 / +0.0136 / +0.0259 | 0.499 / 0.501 / 0.501 | 0.800 / 0.801 / 0.800 | 0.126 / 0.256 / 0.446 |
| c_jsmkt_mean0 | 1 | +0.0063 / +0.0127 / +0.0240 | 0.498 / 0.498 / 0.495 | 0.799 / 0.799 / 0.797 | 0.126 / 0.256 / 0.447 |
| c_own25amed_mean0 | 1 | +0.0045 / +0.0090 / +0.0192 | 0.499 / 0.500 / 0.502 | 0.799 / 0.800 / 0.802 | 0.126 / 0.257 / 0.451 |
| c_pool25amed_mean0 | 1 | +0.0046 / +0.0090 / +0.0196 | 0.499 / 0.501 / 0.503 | 0.800 / 0.800 / 0.801 | 0.126 / 0.257 / 0.450 |
| c_own50amed_mean0 | 1 | +0.0059 / +0.0109 / +0.0200 | 0.499 / 0.500 / 0.502 | 0.800 / 0.800 / 0.801 | 0.126 / 0.257 / 0.450 |
| c_pool50amed_mean0 | 1 | +0.0061 / +0.0115 / +0.0228 | 0.499 / 0.500 / 0.502 | 0.800 / 0.800 / 0.801 | 0.125 / 0.256 / 0.448 |
| c_own75amed_mean0 | 1 | +0.0062 / +0.0110 / +0.0177 | 0.499 / 0.500 / 0.501 | 0.800 / 0.800 / 0.801 | 0.126 / 0.256 / 0.450 |
| c_pool75amed_mean0 | 1 | +0.0067 / +0.0130 / +0.0249 | 0.499 / 0.500 / 0.503 | 0.800 / 0.800 / 0.800 | 0.125 / 0.256 / 0.447 |
| c_mkt_med0 | 1 | +0.0059 / +0.0095 / +0.0180 | 0.498 / 0.496 / 0.491 | 0.798 / 0.795 / 0.789 | 0.125 / 0.256 / 0.446 |
| c_pool_med0 | 1 | +0.0064 / +0.0134 / +0.0264 | 0.499 / 0.500 / 0.500 | 0.800 / 0.798 / 0.800 | 0.126 / 0.256 / 0.446 |
| c_amean_med0 | 1 | +0.0012 / +0.0038 / +0.0146 | 0.499 / 0.499 / 0.500 | 0.800 / 0.799 / 0.800 | 0.126 / 0.258 / 0.453 |
| c_v0med_med0 | 1 | +0.0008 / +0.0026 / +0.0090 | 0.499 / 0.500 / 0.502 | 0.800 / 0.800 / 0.801 | 0.126 / 0.259 / 0.456 |
| c_v0deb_med0 | 1 | +0.0010 / +0.0033 / +0.0118 | 0.499 / 0.499 / 0.501 | 0.799 / 0.799 / 0.800 | 0.126 / 0.259 / 0.454 |
| c_const_med0 | 1 | +0.0066 / +0.0142 / +0.0281 | 0.499 / 0.500 / 0.499 | 0.800 / 0.800 / 0.799 | 0.125 / 0.256 / 0.446 |
| c_jspool_med0 | 1 | +0.0064 / +0.0134 / +0.0264 | 0.499 / 0.500 / 0.500 | 0.800 / 0.798 / 0.800 | 0.126 / 0.256 / 0.446 |
| c_jsmkt_med0 | 1 | +0.0063 / +0.0113 / +0.0214 | 0.499 / 0.498 / 0.495 | 0.799 / 0.798 / 0.795 | 0.126 / 0.256 / 0.447 |
| c_own25amed_med0 | 1 | +0.0045 / +0.0090 / +0.0199 | 0.499 / 0.500 / 0.501 | 0.799 / 0.799 / 0.800 | 0.126 / 0.257 / 0.451 |
| c_pool25amed_med0 | 1 | +0.0046 / +0.0091 / +0.0206 | 0.499 / 0.500 / 0.501 | 0.800 / 0.800 / 0.801 | 0.126 / 0.257 / 0.450 |
| c_own50amed_med0 | 1 | +0.0059 / +0.0106 / +0.0203 | 0.499 / 0.500 / 0.500 | 0.800 / 0.799 / 0.800 | 0.126 / 0.257 / 0.450 |
| c_pool50amed_med0 | 1 | +0.0061 / +0.0115 / +0.0236 | 0.499 / 0.500 / 0.501 | 0.800 / 0.799 / 0.800 | 0.125 / 0.256 / 0.448 |
| c_own75amed_med0 | 1 | +0.0062 / +0.0106 / +0.0176 | 0.499 / 0.498 / 0.498 | 0.800 / 0.798 / 0.798 | 0.126 / 0.256 / 0.450 |
| c_pool75amed_med0 | 1 | +0.0067 / +0.0130 / +0.0255 | 0.499 / 0.500 / 0.501 | 0.799 / 0.799 / 0.800 | 0.125 / 0.256 / 0.447 |
| w_sd63 | 2 | +0.0023 / +0.0088 / +0.0214 | 0.500 / 0.500 / 0.501 | 0.800 / 0.800 / 0.801 | 0.130 / 0.268 / 0.471 |
| w_har | 2 | +0.0064 / +0.0136 / +0.0259 | 0.499 / 0.501 / 0.501 | 0.800 / 0.801 / 0.800 | 0.126 / 0.256 / 0.446 |
| w_floor | 2 | +0.0064 / +0.0136 / +0.0259 | 0.499 / 0.501 / 0.501 | 0.800 / 0.801 / 0.800 | 0.126 / 0.256 / 0.446 |
| w_own | 2 | +0.0066 / +0.0137 / +0.0259 | 0.499 / 0.500 / 0.501 | 0.800 / 0.800 / 0.800 | 0.126 / 0.257 / 0.446 |
| w_pool | 2 | +0.0066 / +0.0136 / +0.0259 | 0.500 / 0.501 / 0.501 | 0.800 / 0.801 / 0.800 | 0.126 / 0.257 / 0.446 |
| w_relvol | 2 | +0.0064 / +0.0136 / +0.0261 | 0.500 / 0.501 / 0.501 | 0.800 / 0.800 / 0.800 | 0.125 / 0.256 / 0.446 |
| w_level | 2 | +0.0064 / +0.0136 / +0.0260 | 0.499 / 0.499 / 0.502 | 0.800 / 0.801 / 0.801 | 0.126 / 0.256 / 0.448 |
| w_xlevel | 2 | +0.0064 / +0.0136 / +0.0260 | 0.499 / 0.501 / 0.502 | 0.800 / 0.801 / 0.801 | 0.126 / 0.256 / 0.447 |
| w_xfloor | 2 | +0.0064 / +0.0136 / +0.0259 | 0.499 / 0.501 / 0.501 | 0.800 / 0.801 / 0.800 | 0.126 / 0.256 / 0.446 |
| s_dd | 2 | +0.0065 / +0.0136 / +0.0260 | 0.499 / 0.500 / 0.502 | 0.800 / 0.801 / 0.801 | 0.126 / 0.257 / 0.447 |
| s_trend | 2 | +0.0065 / +0.0136 / +0.0260 | 0.499 / 0.501 / 0.503 | 0.799 / 0.800 / 0.801 | 0.125 / 0.256 / 0.447 |
| e_v0 | 2 | +0.0073 / +0.0137 / +0.0260 | 0.503 / 0.500 / 0.501 | 0.805 / 0.800 / 0.800 | 0.128 / 0.256 / 0.446 |
| e_v0_rc | 2 | +0.0068 / +0.0137 / +0.0263 | 0.502 / 0.500 / 0.501 | 0.803 / 0.800 / 0.800 | 0.127 / 0.256 / 0.446 |
| e_dev | 2 | +0.0072 / +0.0137 / +0.0260 | 0.504 / 0.501 / 0.501 | 0.805 / 0.801 / 0.800 | 0.128 / 0.256 / 0.446 |
| e_dev_rc | 2 | +0.0068 / +0.0137 / +0.0263 | 0.502 / 0.500 / 0.501 | 0.803 / 0.800 / 0.800 | 0.127 / 0.256 / 0.446 |
| f_conf | 3 | +0.0067 / +0.0137 / +0.0259 | 0.500 / 0.501 / 0.501 | 0.800 / 0.801 / 0.800 | 0.126 / 0.258 / 0.446 |
| f_dev | 3 | +0.0072 / +0.0137 / +0.0260 | 0.504 / 0.501 / 0.501 | 0.805 / 0.801 / 0.800 | 0.128 / 0.256 / 0.446 |
| f_conf_dev | 3 | +0.0074 / +0.0138 / +0.0260 | 0.503 / 0.500 / 0.501 | 0.803 / 0.801 / 0.800 | 0.128 / 0.258 / 0.446 |
| f_conf_v0 | 3 | +0.0075 / +0.0138 / +0.0260 | 0.502 / 0.500 / 0.501 | 0.803 / 0.801 / 0.800 | 0.128 / 0.258 / 0.446 |
| g_own | 4 | +0.0069 / +0.0113 / +0.0202 | 0.504 / 0.500 / 0.500 | 0.805 / 0.800 / 0.796 | 0.128 / 0.259 / 0.450 |
| alt | 4 | +0.0069 / +0.0115 / +0.0208 | 0.502 / 0.500 / 0.499 | 0.803 / 0.800 / 0.796 | 0.128 / 0.258 / 0.449 |
| p_xlevel | 5 | +0.0069 / +0.0113 / +0.0202 | 0.500 / 0.501 / 0.503 | 0.801 / 0.801 / 0.799 | 0.127 / 0.260 / 0.454 |
| p_level | 5 | +0.0070 / +0.0114 / +0.0202 | 0.500 / 0.502 / 0.503 | 0.801 / 0.802 / 0.799 | 0.126 / 0.258 / 0.451 |
| p_relvol | 5 | +0.0070 / +0.0114 / +0.0205 | 0.502 / 0.501 / 0.502 | 0.803 / 0.801 / 0.799 | 0.127 / 0.259 / 0.452 |

#### VAL (all variants; 5 / 21 / 63 days)

| variant | stage | pinball skill | cov50 | cov80 | width80 |
|---|---|---|---|---|---|
| naive0 | base | +0.0000 / +0.0000 / +0.0000 | 0.541 / 0.517 / 0.481 | 0.808 / 0.798 / 0.779 | 0.105 / 0.215 / 0.373 |
| naive_drift | base | +0.0062 / +0.0226 / +0.0634 | 0.550 / 0.535 / 0.523 | 0.812 / 0.811 / 0.812 | 0.105 / 0.215 / 0.373 |
| v0 | base | +0.0051 / +0.0127 / +0.0421 | 0.520 / 0.496 / 0.487 | 0.802 / 0.789 / 0.785 | 0.102 / 0.205 / 0.354 |
| vol_har | base | +0.0106 / +0.0219 / +0.0478 | 0.525 / 0.514 / 0.509 | 0.810 / 0.810 / 0.816 | 0.101 / 0.208 / 0.371 |
| vol_har_drift | base | +0.0124 / +0.0288 / +0.0673 | 0.527 / 0.526 / 0.531 | 0.812 / 0.816 / 0.832 | 0.101 / 0.209 / 0.377 |
| c_zero_fit | 1 | +0.0107 / +0.0223 / +0.0487 | 0.525 / 0.517 / 0.510 | 0.811 / 0.812 / 0.816 | 0.101 / 0.209 / 0.371 |
| c_own_fit | 1 | +0.0124 / +0.0287 / +0.0674 | 0.527 / 0.527 / 0.530 | 0.813 / 0.817 / 0.831 | 0.101 / 0.210 / 0.376 |
| c_amed_fit | 1 | +0.0087 / +0.0222 / +0.0550 | 0.528 / 0.521 / 0.524 | 0.812 / 0.816 / 0.823 | 0.101 / 0.210 / 0.375 |
| c_zero_mean0 | 1 | +0.0066 / +0.0091 / +0.0126 | 0.517 / 0.509 / 0.482 | 0.808 / 0.801 / 0.789 | 0.101 / 0.209 / 0.371 |
| c_own_mean0 | 1 | +0.0124 / +0.0287 / +0.0674 | 0.527 / 0.527 / 0.530 | 0.813 / 0.817 / 0.831 | 0.101 / 0.210 / 0.376 |
| c_amed_mean0 | 1 | +0.0086 / +0.0220 / +0.0551 | 0.526 / 0.520 / 0.525 | 0.812 / 0.817 / 0.823 | 0.101 / 0.210 / 0.375 |
| c_zero_med0 | 1 | +0.0061 / +0.0039 / +0.0018 | 0.517 / 0.503 / 0.477 | 0.808 / 0.800 / 0.782 | 0.101 / 0.209 / 0.371 |
| c_own_med0 | 1 | +0.0123 / +0.0281 / +0.0667 | 0.527 / 0.526 / 0.529 | 0.812 / 0.816 / 0.830 | 0.101 / 0.210 / 0.376 |
| c_amed_med0 | 1 | +0.0086 / +0.0221 / +0.0553 | 0.526 / 0.520 / 0.524 | 0.812 / 0.816 / 0.823 | 0.101 / 0.210 / 0.375 |
| c_zero_vf_centred | 1 | +0.0071 / +0.0116 / +0.0182 | 0.519 / 0.509 / 0.486 | 0.808 / 0.802 / 0.795 | 0.101 / 0.208 / 0.371 |
| c_own_vf_centred | 1 | +0.0125 / +0.0288 / +0.0677 | 0.528 / 0.525 / 0.526 | 0.812 / 0.814 / 0.826 | 0.101 / 0.208 / 0.371 |
| c_amed_vf_centred | 1 | +0.0087 / +0.0218 / +0.0546 | 0.527 / 0.517 / 0.523 | 0.810 / 0.813 / 0.819 | 0.101 / 0.208 / 0.371 |
| c_mkt_mean0 | 1 | +0.0099 / +0.0209 / +0.0458 | 0.522 / 0.515 / 0.503 | 0.809 / 0.810 / 0.811 | 0.101 / 0.209 / 0.370 |
| c_pool_mean0 | 1 | +0.0118 / +0.0267 / +0.0622 | 0.527 / 0.522 / 0.522 | 0.812 / 0.814 / 0.820 | 0.101 / 0.209 / 0.370 |
| c_amean_mean0 | 1 | +0.0060 / +0.0192 / +0.0549 | 0.528 / 0.520 / 0.523 | 0.810 / 0.815 / 0.824 | 0.101 / 0.211 / 0.376 |
| c_v0med_mean0 | 1 | +0.0080 / +0.0155 / +0.0434 | 0.528 / 0.521 / 0.520 | 0.812 / 0.816 / 0.822 | 0.102 / 0.211 / 0.378 |
| c_v0deb_mean0 | 1 | +0.0095 / +0.0210 / +0.0558 | 0.527 / 0.524 / 0.532 | 0.812 / 0.817 / 0.825 | 0.101 / 0.211 / 0.377 |
| c_const_mean0 | 1 | +0.0118 / +0.0267 / +0.0621 | 0.527 / 0.521 / 0.522 | 0.811 / 0.814 / 0.818 | 0.101 / 0.209 / 0.369 |
| c_jspool_mean0 | 1 | +0.0118 / +0.0269 / +0.0629 | 0.527 / 0.522 / 0.522 | 0.811 / 0.814 / 0.820 | 0.101 / 0.209 / 0.370 |
| c_jsmkt_mean0 | 1 | +0.0111 / +0.0251 / +0.0576 | 0.524 / 0.520 / 0.511 | 0.810 / 0.812 / 0.818 | 0.101 / 0.209 / 0.370 |
| c_own25amed_mean0 | 1 | +0.0113 / +0.0262 / +0.0628 | 0.527 / 0.523 / 0.527 | 0.812 / 0.817 / 0.825 | 0.101 / 0.210 / 0.373 |
| c_pool25amed_mean0 | 1 | +0.0108 / +0.0246 / +0.0584 | 0.528 / 0.522 / 0.524 | 0.811 / 0.816 / 0.823 | 0.101 / 0.210 / 0.373 |
| c_own50amed_mean0 | 1 | +0.0128 / +0.0287 / +0.0675 | 0.529 / 0.525 / 0.529 | 0.813 / 0.818 / 0.827 | 0.101 / 0.209 / 0.373 |
| c_pool50amed_mean0 | 1 | +0.0121 / +0.0262 / +0.0607 | 0.528 / 0.523 / 0.524 | 0.811 / 0.815 / 0.821 | 0.101 / 0.209 / 0.371 |
| c_own75amed_mean0 | 1 | +0.0132 / +0.0296 / +0.0691 | 0.529 / 0.523 / 0.529 | 0.812 / 0.817 / 0.829 | 0.101 / 0.209 / 0.373 |
| c_pool75amed_mean0 | 1 | +0.0125 / +0.0269 / +0.0620 | 0.526 / 0.523 / 0.524 | 0.811 / 0.815 / 0.820 | 0.101 / 0.209 / 0.370 |
| c_mkt_med0 | 1 | +0.0096 / +0.0177 / +0.0396 | 0.521 / 0.513 / 0.498 | 0.809 / 0.808 / 0.806 | 0.101 / 0.209 / 0.370 |
| c_pool_med0 | 1 | +0.0118 / +0.0259 / +0.0608 | 0.527 / 0.522 / 0.520 | 0.812 / 0.812 / 0.819 | 0.101 / 0.209 / 0.370 |
| c_amean_med0 | 1 | +0.0059 / +0.0186 / +0.0536 | 0.527 / 0.517 / 0.521 | 0.810 / 0.814 / 0.822 | 0.101 / 0.211 / 0.376 |
| c_v0med_med0 | 1 | +0.0079 / +0.0158 / +0.0436 | 0.528 / 0.518 / 0.519 | 0.812 / 0.816 / 0.822 | 0.102 / 0.211 / 0.378 |
| c_v0deb_med0 | 1 | +0.0095 / +0.0218 / +0.0570 | 0.527 / 0.523 / 0.530 | 0.812 / 0.818 / 0.826 | 0.101 / 0.211 / 0.377 |
| c_const_med0 | 1 | +0.0117 / +0.0272 / +0.0604 | 0.526 / 0.522 / 0.520 | 0.811 / 0.814 / 0.818 | 0.101 / 0.209 / 0.369 |
| c_jspool_med0 | 1 | +0.0118 / +0.0259 / +0.0608 | 0.527 / 0.522 / 0.520 | 0.811 / 0.812 / 0.819 | 0.101 / 0.209 / 0.370 |
| c_jsmkt_med0 | 1 | +0.0118 / +0.0254 / +0.0604 | 0.524 / 0.522 / 0.515 | 0.811 / 0.812 / 0.819 | 0.101 / 0.209 / 0.371 |
| c_own25amed_med0 | 1 | +0.0112 / +0.0261 / +0.0628 | 0.527 / 0.523 / 0.527 | 0.812 / 0.817 / 0.824 | 0.101 / 0.210 / 0.373 |
| c_pool25amed_med0 | 1 | +0.0108 / +0.0245 / +0.0583 | 0.528 / 0.522 / 0.525 | 0.811 / 0.815 / 0.822 | 0.101 / 0.210 / 0.373 |
| c_own50amed_med0 | 1 | +0.0128 / +0.0284 / +0.0672 | 0.529 / 0.523 / 0.527 | 0.813 / 0.817 / 0.826 | 0.101 / 0.209 / 0.373 |
| c_pool50amed_med0 | 1 | +0.0121 / +0.0258 / +0.0602 | 0.528 / 0.521 / 0.523 | 0.812 / 0.815 / 0.821 | 0.101 / 0.209 / 0.371 |
| c_own75amed_med0 | 1 | +0.0132 / +0.0292 / +0.0685 | 0.530 / 0.523 / 0.527 | 0.812 / 0.816 / 0.828 | 0.101 / 0.209 / 0.373 |
| c_pool75amed_med0 | 1 | +0.0124 / +0.0263 / +0.0611 | 0.526 / 0.521 / 0.523 | 0.811 / 0.813 / 0.820 | 0.101 / 0.209 / 0.370 |
| w_sd63 | 2 | +0.0070 / +0.0217 / +0.0588 | 0.520 / 0.509 / 0.502 | 0.803 / 0.802 / 0.805 | 0.102 / 0.211 / 0.370 |
| w_har | 2 | +0.0118 / +0.0269 / +0.0629 | 0.527 / 0.522 / 0.522 | 0.811 / 0.814 / 0.820 | 0.101 / 0.209 / 0.370 |
| w_floor | 2 | +0.0118 / +0.0269 / +0.0629 | 0.527 / 0.522 / 0.522 | 0.811 / 0.814 / 0.820 | 0.101 / 0.209 / 0.370 |
| w_own | 2 | +0.0119 / +0.0270 / +0.0629 | 0.519 / 0.518 / 0.522 | 0.804 / 0.811 / 0.820 | 0.100 / 0.208 / 0.370 |
| w_pool | 2 | +0.0120 / +0.0268 / +0.0629 | 0.518 / 0.518 / 0.522 | 0.804 / 0.808 / 0.820 | 0.099 / 0.207 / 0.370 |
| w_relvol | 2 | +0.0117 / +0.0267 / +0.0628 | 0.528 / 0.523 / 0.522 | 0.812 / 0.815 / 0.820 | 0.101 / 0.209 / 0.370 |
| w_level | 2 | +0.0119 / +0.0270 / +0.0631 | 0.524 / 0.520 / 0.523 | 0.811 / 0.814 / 0.817 | 0.101 / 0.209 / 0.369 |
| w_xlevel | 2 | +0.0117 / +0.0269 / +0.0631 | 0.527 / 0.522 / 0.522 | 0.811 / 0.814 / 0.821 | 0.101 / 0.209 / 0.371 |
| w_xfloor | 2 | +0.0118 / +0.0269 / +0.0629 | 0.527 / 0.522 / 0.522 | 0.811 / 0.814 / 0.820 | 0.101 / 0.209 / 0.370 |
| s_dd | 2 | +0.0121 / +0.0269 / +0.0623 | 0.524 / 0.519 / 0.522 | 0.810 / 0.814 / 0.821 | 0.101 / 0.209 / 0.370 |
| s_trend | 2 | +0.0120 / +0.0270 / +0.0624 | 0.526 / 0.521 / 0.523 | 0.811 / 0.813 / 0.821 | 0.101 / 0.209 / 0.370 |
| e_v0 | 2 | +0.0125 / +0.0268 / +0.0626 | 0.529 / 0.520 / 0.520 | 0.812 / 0.813 / 0.818 | 0.101 / 0.208 / 0.368 |
| e_v0_rc | 2 | +0.0119 / +0.0271 / +0.0636 | 0.528 / 0.521 / 0.517 | 0.812 / 0.812 / 0.816 | 0.101 / 0.208 / 0.367 |
| e_dev | 2 | +0.0126 / +0.0273 / +0.0633 | 0.528 / 0.521 / 0.519 | 0.813 / 0.813 / 0.819 | 0.101 / 0.208 / 0.368 |
| e_dev_rc | 2 | +0.0119 / +0.0271 / +0.0636 | 0.528 / 0.521 / 0.517 | 0.812 / 0.813 / 0.817 | 0.101 / 0.208 / 0.366 |
| f_conf | 3 | +0.0120 / +0.0269 / +0.0629 | 0.519 / 0.513 / 0.522 | 0.803 / 0.806 / 0.820 | 0.099 / 0.206 / 0.370 |
| f_dev | 3 | +0.0126 / +0.0273 / +0.0633 | 0.528 / 0.521 / 0.519 | 0.813 / 0.813 / 0.819 | 0.101 / 0.208 / 0.368 |
| f_conf_dev | 3 | +0.0128 / +0.0272 / +0.0633 | 0.522 / 0.513 / 0.519 | 0.806 / 0.806 / 0.819 | 0.100 / 0.206 / 0.368 |
| f_conf_v0 | 3 | +0.0127 / +0.0268 / +0.0626 | 0.522 / 0.513 / 0.520 | 0.806 / 0.806 / 0.818 | 0.100 / 0.206 / 0.368 |
| g_own | 4 | +0.0136 / +0.0298 / +0.0687 | 0.524 / 0.519 / 0.518 | 0.807 / 0.811 / 0.815 | 0.100 / 0.208 / 0.365 |
| alt | 4 | +0.0136 / +0.0297 / +0.0683 | 0.521 / 0.518 / 0.519 | 0.806 / 0.813 / 0.817 | 0.100 / 0.208 / 0.366 |
| p_xlevel | 5 | +0.0136 / +0.0298 / +0.0686 | 0.518 / 0.520 / 0.521 | 0.803 / 0.812 / 0.818 | 0.099 / 0.208 / 0.368 |
| p_level | 5 | +0.0136 / +0.0297 / +0.0685 | 0.522 / 0.523 / 0.522 | 0.806 / 0.814 / 0.820 | 0.099 / 0.208 / 0.367 |
| p_relvol | 5 | +0.0137 / +0.0298 / +0.0691 | 0.521 / 0.521 / 0.521 | 0.805 / 0.813 / 0.818 | 0.099 / 0.208 / 0.367 |

#### DEV-chosen settings per horizon (5 / 21 / 63)

| variant | settings | ensemble weight |
|---|---|---|
| c_const_mean0 | a=0.12/yr / a=0.12/yr / a=0.12/yr | - / - / - |
| c_jspool_mean0 | tau=0.01/yr / tau=0.02/yr / tau=0.02/yr | - / - / - |
| c_jsmkt_mean0 | tau=0.06/yr / tau=0.06/yr / tau=0.06/yr | - / - / - |
| c_const_med0 | a=0.12/yr / a=0.15/yr / a=0.12/yr | - / - / - |
| c_jspool_med0 | tau=0.01/yr / tau=0.00/yr / tau=0.00/yr | - / - / - |
| c_jsmkt_med0 | tau=0.10/yr / tau=0.10/yr / tau=0.10/yr | - / - / - |
| w_floor | floor_b=0.0 / floor_b=0.0 / floor_b=0.0 | - / - / - |
| w_own | own=[756, 0.5] / own=[756, 0.15] / own=[252, 0.0] | - / - / - |
| w_pool | pool=[365, 0.5] / pool=[365, 0.25] / pool=[91, 0.0] | - / - / - |
| w_xfloor | xfloor_b=0.0 / xfloor_b=0.0 / xfloor_b=0.0 | - / - / - |
| e_v0 | - / - / - | 0.3 / 0.1 / 0.1 |
| e_v0_rc | - / - / - | 0.2 / 0.1 / 0.2 |
| e_dev | - / - / - | 0.3 / 0.1 / 0.1 |
| e_dev_rc | - / - / - | 0.2 / 0.1 / 0.2 |
| f_conf | own=[756, 0.3],pool=[365, 0.25] / own=[756, 0.15],pool=[365, 0.25] / own=[756, 0.0],pool=[365, 0.0] | - / - / - |
| f_dev | - / - / - | 0.3 / 0.1 / 0.1 |
| f_conf_dev | own=[756, 0.3],pool=[365, 0.25] / own=[756, 0.15],pool=[365, 0.25] / own=[756, 0.0],pool=[365, 0.0] | 0.2 / 0.1 / 0.1 |
| f_conf_v0 | own=[756, 0.3],pool=[365, 0.25] / own=[756, 0.15],pool=[365, 0.25] / own=[756, 0.0],pool=[365, 0.0] | 0.2 / 0.1 / 0.1 |
| g_own | own=[756, 0.3],pool=[365, 0.25] / own=[756, 0.15],pool=[365, 0.0] / own=[756, 0.0],pool=[365, 0.0] | 0.3 / 0.3 / 0.5 |
| alt | own=[756, 0.3],pool=[365, 0.25] / own=[756, 0.15],pool=[365, 0.0] / own=[756, 0.0],pool=[365, 0.0] | 0.2 / 0.2 / 0.4 |
| p_xlevel | own=[756, 0.3],pool=[365, 0.25] / own=[756, 0.15],pool=[365, 0.0] / own=[756, 0.0],pool=[365, 0.0] | 0.3 / 0.3 / 0.5 |
| p_level | own=[756, 0.3],pool=[365, 0.25] / own=[756, 0.15],pool=[365, 0.0] / own=[756, 0.0],pool=[365, 0.0] | 0.3 / 0.3 / 0.5 |
| p_relvol | own=[756, 0.3],pool=[365, 0.25] / own=[756, 0.15],pool=[365, 0.0] / own=[756, 0.0],pool=[365, 0.0] | 0.3 / 0.3 / 0.5 |

## What it means

1. **The width problem is solved by the HAR forecast, not by any extra calibration.** With HAR width the cone covers 0.80 overall and in every vol tercile on DEV, and 0.80-0.82 in the low tercile on VAL. v0 covers 0.73-0.76 there. Every extra width idea was chosen on DEV at or near "do nothing", and none moved skill by more than +0.0003. The ideas: long-run vol floor, basket-relative floor, regime-binned z, walk-forward own and basket conformal factors. The conformal factors at 5d/21d mainly trim VAL coverage toward 0.80 (cov80 0.811 to 0.804). At 63d DEV picks none. The remaining VAL miscalibration is the **mid** tercile over-covering (0.83-0.85 at 21/63d). DEV's mid tercile is 0.80, so this looks like the calm 2012-18 market, not a fixable model bias.
2. **The analog blend costs a little calibration by tercile.** cone2's low-vol tercile covers 0.784-0.789 on VAL (0.787-0.793 on DEV), against vol_har_drift's 0.80-0.82. Its high tercile over-covers slightly on DEV (0.805-0.822). The analog cone is too narrow for calm stocks and too wide for wild ones. A DEV-fitted per-regime spread factor after the blend (stage 5) found pinball-optimal factors of 0.96-1.05 and did not fix it, so the shortfall is pinball-neutral. I kept the simpler model. The worst tercile cov80 error is 0.048 (VAL mid, 63d), against vol_har_drift's 0.066 and v0's 0.072.
3. **The centre is where the skill is, and DEV and VAL disagree about it.**
   - The pooled basket drift is a parameter-free walk-forward mean of about 12%/yr. It is the best honest centre on DEV, at +0.0064 / +0.0135 / +0.0259 (only a DEV-fitted constant beats it, and that constant is in-sample).
   - James-Stein fit on DEV says shrink the stock's own mean almost entirely into it (tau = 0.01-0.02/yr).
   - The stock's own mean is best on VAL (+0.0124 / +0.0287 / +0.0674) but worst of the positive centres on DEV.
   - Paired, the pool-centre alternative beats cone2 on DEV (-0.0005 / -0.0025 / -0.0058, all significant) and loses on VAL (+0.0008 / +0.0026 / +0.0054, all significant).
   - VAL was a bull market and the universe is survivors, both of which flatter own-mean centres.
4. **Why cone2 still uses the own mean.** It is the best-VAL variant, and it does not lose on DEV to any baseline: it beats v0, naive_drift and vol_har_drift on DEV at every horizon, all significant. The analog blend is what makes this work. DEV gives the deviation-scaled analog cone weight 0.3 / 0.3 / 0.5, which pulls a noisy own mean toward the analog median (about the pool drift). The alternative `cone2_f_conf_dev` (JS-to-pool centre + same calibration + w = 0.2 / 0.1 / 0.1) is the right pick if the coordinator weighs DEV over VAL. It is exported and harness-runnable.
5. **Does the analog information add anything once width and centre are fixed?** Barely, and not where v0 used it:
   - As a centre, the analog median is worse than the unconditional basket drift on DEV (amed +0.0021 / +0.0055 / +0.0154 vs pool +0.0064 / +0.0135 / +0.0259). Its gain over v0's median is only the removal of the vol-scaling drift bug. v0med (+0.0008 / +0.0018 / +0.0065 DEV) < v0deb < amed.
   - As pure shape/width (the analog cone re-centred on our median), it adds +0.0004 / +0.0002 / +0.0004 on DEV.
   - As a blended cone, it adds about +0.0008 at 5 days on both splits (perhaps short-term reversal in the setup features). At 21/63 days it acts mainly as shrinkage of the centre.
6. **Shape:** regime-skewed z (drawdown or trend terciles) and vol-regime z changed skill by at most ±0.0006, so I didn't keep them. Mean-removed DEV z and volforecast's frozen centred z are equivalent.
7. **Multiple testing.** 64 named variants were scored on VAL, and each had its own DEV grid, so several hundred DEV fits in all. The final pick was among about four serious candidates whose VAL scores differ by 0.001-0.005. That spread is the size of the paired 90% intervals, so expect some VAL-selection luck in cone2's VAL lead over the alternative and over vol_har_drift. Its leads over v0 and naive_drift are far larger than that.

## How much should survive TEST

- **cone2 beats v0** (VAL +0.009 / +0.017 / +0.027, DEV +0.008 / +0.013 / +0.014, all significant): very likely to survive. v0 carries the vol-scaling drift bug and a too-narrow cone.
- **cone2 beats naive_drift** (VAL +0.007 / +0.007 / +0.005): likely at 5/21 days. 63 days is less certain.
- **cone2 beats vol_har_drift** (VAL +0.0012 / +0.0011 / +0.0015, DEV +0.0015 / +0.0018 / +0.0078): small. I'd bet on roughly half of it surviving.
- **cone2 vs the pool-centre alternative:** a coin flip that depends on whether 2019+ looks like 1962-2011 (pool wins) or 2012-18 (own wins). TEST includes many short-history TEST tickers, where an own mean is mostly noise. On VAL rows with under 5 years of history, cone2 still held up (+0.0185 / +0.0373 / +0.0829 vs the alternative's +0.0177 / +0.0358 / +0.0840), but that was a bull market.
- **Coverage:** overall cov80 within 0.80 ± 0.02 should hold. The tercile pattern (calm 0.78-0.79) probably holds too.

## Not verified

- TEST never run. `engine.py --selfcheck` was not run (instructed: the cache is frozen).
- **The DEV in-sample component.** z, ensemble weights and grid values were fit on DEV and scored on DEV, so absolute DEV skill is slightly optimistic for every variant.
- **Significance:** CIs resample whole tickers (48 VAL / 40 DEV). Rows overlap in time, and long-history stocks dominate DEV. No per-ticker equal-weight view.
- **The basket drift** comes from 42 of today's large-cap survivors, so survivor drift enters the pool centre too. The index mean (^GSPC is price-only, without dividends, before SPY in 1993) scored lower, partly for that reason.
- **Young-stock robustness on TEST tickers** is untested beyond VAL's 1,048 short-history rows.
- **Direction:** cone2's median-sign hit rate is below the walk-forward majority on VAL. Its Brier skill is about +0.002-0.003. This is a calibration result, not a trading signal.
- **The analog memo** (`cone2_analog/`) is trusted because the look-ahead proof recomputes on truncated contexts and matches, and because the recomputed v0 quantiles equal the harness cache. Every row was not re-derived without the memo.
- **Integration cost:** cone2 needs v0's analog search at forecast time (same cost as v0) and the basket statistics (about 1.5 s to build once per engine).
- **Proofs:** `python research/harness.py --lookahead --models cone2` passed on the final file (also for `cone2_f_conf_dev`).
