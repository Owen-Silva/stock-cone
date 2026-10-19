# vol-01 — Volatility forecasting for the cone (DEV fit, VAL choose; no 2019+ data)

> **Not financial advice.** This is a research log from building an educational tool. Backtest numbers are hypothetical, carry survivorship bias, and say nothing certain about the future. Nothing here is a recommendation to buy, sell or hold anything.

Written by the coordinator from the vol lane's final report (subagents cannot write .md
files here). Code: `research/volforecast.py` (`selfcheck` passes: no look-ahead on 150
points, frozen coefficients == DEV fit). Harness models: `vol_har`, `vol_har_drift`
via `research/models_vol.py`.

## Headline

- Trailing 63-day vol **under-forecasts** future variance by 1.17x / 1.20x / 1.24x at
  5 / 21 / 63 days on VAL (sigma ~8-11% too narrow, growing with horizon). This is most
  of v0's 77% / 76% coverage at 1M / 3M.
- Winner at every horizon: `har_mx_QLr` — log-variance HAR on 5/22/66/252-day realised
  variance + 756-day long-run level + SPY 22/252-day vol + shock + 22-day drawdown,
  coefficients fit on DEV by minimising QLIKE against r_h^2.

| h | QLIKE (base) | log-MSE (base) | mean RV/F (base) | cov50 / cov80 (base) | scaled-pinball skill vs naive0 |
|---|---|---|---|---|---|
| 5 | 0.507 (0.566) | 1.23 (1.22) | 1.00 (1.17) | 0.524 / 0.815 (0.542 / 0.812) | +0.010 |
| 21 | 0.240 (0.301) | 0.49 (0.54) | 1.01 (1.20) | 0.516 / 0.811 (0.515 / 0.799) | +0.022 |
| 63 | 0.153 (0.232) | 0.32 (0.39) | 0.96 (1.24) | 0.504 / 0.812 (0.476 / 0.775) | +0.049 |

- Through `research/harness.py` (same 14,951 VAL rows): zero drift skill
  +0.011 / +0.022 / +0.048 vs v0 +0.005 / +0.013 / +0.042; cov80 0.810 / 0.810 / 0.816
  vs v0 0.802 / 0.789 / 0.785. With own drift: +0.012 / +0.029 / +0.067 (beats
  naive_drift at every horizon).
- 63-day cov80 holds every VAL year 2012-2018 (0.80-0.83; baseline 0.70-0.83).
- About half the zero-drift skill comes from empirical z-quantiles (Gaussian
  over-covers, cov80 ~0.84); the better sigma adds +0.005 / +0.008 / +0.011.
- Didn't help: volume, regime-dependent z, variance-space HAR, EWMA/mean-reversion blends.

## Not verified

- No significance tests (overlapping rows, 48 tickers): skill gaps < ~0.002 or coverage
  gaps < 1 point may be noise. QLIKE gaps are large and consistent across years.
- Long-history stocks dominate DEV (no per-ticker equal weighting).
- Skill near 0 for the most volatile tercile at 21/63 days.

## Full tables (from the lane)

#### h = 5 (VAL rows 74,833)

| model | qlike | dev_qlike | logmse | bias_log | rv_over_f | r2_over_f | d_qlike_r |
|---|---|---|---|---|---|---|---|
| har_mxv_QLr | 0.5072 | 0.4091 | 1.2314 | 0.5093 | 0.9979 | 0.9906 | -0.0642 |
| har_mx_QLr | 0.5073 | 0.4092 | 1.2320 | 0.5096 | 0.9976 | 0.9903 | -0.0641 |
| har_mx_QL | 0.5087 | 0.4052 | 1.1977 | 0.4788 | 1.0299 | 1.0258 | -0.0595 |
| har_x_QLr | 0.5134 | 0.4148 | 1.2674 | 0.5313 | 0.9820 | 0.9784 | -0.0543 |
| har_m_QL | 0.5157 | 0.4128 | 1.2051 | 0.4733 | 1.0424 | 1.0356 | -0.0552 |
| har4lr_QL | 0.5206 | 0.4209 | 1.2491 | 0.5039 | 1.0167 | 1.0130 | -0.0471 |
| har4_QL | 0.5210 | 0.4212 | 1.2552 | 0.5091 | 1.0118 | 1.0084 | -0.0466 |
| har4_logOLS | 0.5210 | 0.4212 | 1.2517 | 0.5054 | 1.0156 | 1.0122 | -0.0465 |
| mr_ewma_lr | 0.5289 | 0.4291 | 1.2631 | 0.4953 | 1.0335 | 1.0286 | -0.0401 |
| ewma0.99*s | 0.5386 | 0.4594 | 1.3230 | 0.5302 | 1.0084 | 0.9963 | -0.0376 |
| ewma0.99 | 0.5395 | 0.4601 | 1.2873 | 0.4954 | 1.0441 | 1.0316 | -0.0371 |
| ewma0.97 | 0.5421 | 0.4353 | 1.1978 | 0.4112 | 1.1308 | 1.1276 | -0.0252 |
| sd63*s | 0.5539 | 0.4508 | 1.3268 | 0.5082 | 1.0456 | 1.0436 | -0.0123 |
| har4_varOLS | 0.5600 | 0.4404 | 1.5795 | 0.7256 | 0.8344 | 0.8287 | -0.0098 |
| sd126 | 0.5616 | 0.4703 | 1.2827 | 0.4495 | 1.1121 | 1.1028 | -0.0117 |
| sd252 | 0.5652 | 0.5039 | 1.3448 | 0.5019 | 1.0632 | 1.0443 | -0.0178 |
| sd63 | 0.5664 | 0.4578 | 1.2226 | 0.3927 | 1.1737 | 1.1714 | 0.0000 |
| ewma0.94 | 0.5739 | 0.4470 | 1.1841 | 0.3504 | 1.2234 | 1.2224 | 0.0088 |
| lr756 | 0.6048 | 0.6061 | 1.5652 | 0.6199 | 0.9848 | 0.9626 | 0.0185 |
| sd21 | 0.6791 | 0.5114 | 1.2544 | 0.2812 | 1.3978 | 1.3903 | 0.1074 |

mr_ewma_lr: {'w': 0.9, 'scale': 1.0652489653808659}

#### h = 21 (VAL rows 74,065)

| model | qlike | dev_qlike | logmse | bias_log | rv_over_f | r2_over_f | d_qlike_r |
|---|---|---|---|---|---|---|---|
| har_mx_QLr | 0.2404 | 0.2115 | 0.4936 | 0.2269 | 1.0136 | 0.9578 | -0.0677 |
| har_mxv_QLr | 0.2406 | 0.2116 | 0.4942 | 0.2278 | 1.0128 | 0.9574 | -0.0672 |
| har_x_QLr | 0.2434 | 0.2172 | 0.5060 | 0.2374 | 1.0059 | 0.9569 | -0.0581 |
| har_mx_QL | 0.2440 | 0.2044 | 0.5031 | 0.2358 | 1.0082 | 0.9585 | -0.0580 |
| har_m_QL | 0.2489 | 0.2086 | 0.5095 | 0.2299 | 1.0190 | 0.9669 | -0.0555 |
| har4lr_QL | 0.2492 | 0.2172 | 0.5300 | 0.2611 | 0.9881 | 0.9434 | -0.0479 |
| har4_logOLS | 0.2498 | 0.2176 | 0.5353 | 0.2693 | 0.9805 | 0.9369 | -0.0462 |
| har4_QL | 0.2499 | 0.2175 | 0.5345 | 0.2668 | 0.9832 | 0.9395 | -0.0461 |
| mr_ewma_lr | 0.2555 | 0.2230 | 0.5336 | 0.2392 | 1.0163 | 0.9675 | -0.0457 |
| ewma0.99*s | 0.2597 | 0.2451 | 0.5519 | 0.2660 | 0.9938 | 0.9341 | -0.0523 |
| ewma0.99 | 0.2610 | 0.2467 | 0.5248 | 0.2089 | 1.0522 | 0.9890 | -0.0545 |
| ewma0.97 | 0.2770 | 0.2351 | 0.5135 | 0.1216 | 1.1554 | 1.1062 | -0.0245 |
| sd252 | 0.2828 | 0.2850 | 0.5689 | 0.2162 | 1.0666 | 0.9920 | -0.0442 |
| sd126 | 0.2834 | 0.2574 | 0.5467 | 0.1631 | 1.1204 | 1.0554 | -0.0339 |
| sd63*s | 0.2840 | 0.2428 | 0.5916 | 0.2474 | 1.0366 | 0.9948 | -0.0102 |
| sd63 | 0.3007 | 0.2538 | 0.5410 | 0.1026 | 1.1981 | 1.1498 | 0.0000 |
| har4_varOLS | 0.3091 | 0.2460 | 0.8058 | 0.5233 | 0.7858 | 0.7503 | 0.0212 |
| lr756 | 0.3144 | 0.3831 | 0.7071 | 0.3368 | 0.9776 | 0.9014 | -0.0142 |
| ewma0.94 | 0.3206 | 0.2578 | 0.5596 | 0.0595 | 1.2611 | 1.2148 | 0.0219 |
| sd21 | 0.4228 | 0.3287 | 0.6755 | -0.0101 | 1.4329 | 1.3793 | 0.1169 |

mr_ewma_lr: {'w': 0.85, 'scale': 1.0890163975022047}

#### h = 63 (VAL rows 72,091)

| model | qlike | dev_qlike | logmse | bias_log | rv_over_f | r2_over_f | d_qlike_r |
|---|---|---|---|---|---|---|---|
| har_mx_QLr | 0.1528 | 0.1696 | 0.3189 | 0.1957 | 0.9571 | 0.9384 | -0.0899 |
| har_mxv_QLr | 0.1533 | 0.1697 | 0.3200 | 0.1972 | 0.9560 | 0.9373 | -0.0894 |
| har_x_QLr | 0.1597 | 0.1719 | 0.3249 | 0.1816 | 0.9781 | 0.9591 | -0.0833 |
| har_mx_QL | 0.1627 | 0.1625 | 0.3352 | 0.1938 | 0.9690 | 0.9573 | -0.0729 |
| har_m_QL | 0.1663 | 0.1647 | 0.3394 | 0.1887 | 0.9776 | 0.9648 | -0.0705 |
| har4lr_QL | 0.1673 | 0.1707 | 0.3485 | 0.2091 | 0.9581 | 0.9462 | -0.0687 |
| ewma0.99*s | 0.1673 | 0.1932 | 0.3442 | 0.1926 | 0.9747 | 0.9392 | -0.0922 |
| mr_ewma_lr | 0.1673 | 0.1771 | 0.3378 | 0.1720 | 0.9952 | 0.9811 | -0.0708 |
| har4_QL | 0.1685 | 0.1709 | 0.3521 | 0.2130 | 0.9555 | 0.9441 | -0.0670 |
| har4_logOLS | 0.1690 | 0.1710 | 0.3550 | 0.2193 | 0.9497 | 0.9388 | -0.0658 |
| ewma0.99 | 0.1691 | 0.1975 | 0.3173 | 0.1013 | 1.0678 | 1.0290 | -0.0937 |
| sd252 | 0.1894 | 0.2297 | 0.3609 | 0.1086 | 1.0808 | 1.0245 | -0.0909 |
| sd126 | 0.1938 | 0.2109 | 0.3485 | 0.0574 | 1.1364 | 1.1073 | -0.0592 |
| ewma0.97 | 0.2018 | 0.1984 | 0.3445 | 0.0085 | 1.1933 | 1.1899 | -0.0255 |
| lr756 | 0.2042 | 0.3252 | 0.4456 | 0.2338 | 0.9704 | 0.9183 | -0.0719 |
| sd63*s | 0.2077 | 0.2007 | 0.4196 | 0.1835 | 1.0242 | 1.0173 | -0.0232 |
| sd63 | 0.2324 | 0.2203 | 0.3860 | -0.0080 | 1.2404 | 1.2320 | 0.0000 |
| ewma0.94 | 0.2603 | 0.2306 | 0.4195 | -0.0585 | 1.3188 | 1.3298 | 0.0474 |
| har4_varOLS | 0.2637 | 0.2150 | 0.6855 | 0.5299 | 0.7338 | 0.7268 | 0.0328 |
| sd21 | 0.3876 | 0.3081 | 0.5684 | -0.1315 | 1.5190 | 1.5409 | 0.1855 |

mr_ewma_lr: {'w': 0.8, 'scale': 1.125738940921881}

#### h = 5 quantiles (VAL)

| model | cov50 | cov80 | pinball | skill | dev_skill | width80 |
|---|---|---|---|---|---|---|
| naive0 (sd63, Gaussian, zero drift) | 0.5419 | 0.8119 | 0.3043 | 0.0000 | 0.0000 | 0.1049 |
| naive_drift (sd63, Gaussian, own drift) | 0.5490 | 0.8152 | 0.3024 | 0.0062 | 0.0006 | 0.1049 |
| sd63 + DEV empirical z (zero) | 0.5169 | 0.8041 | 0.3028 | 0.0050 | 0.0022 | 0.1021 |
| har_mx_QLr z=zero | 0.5237 | 0.8148 | 0.3012 | 0.0104 | 0.0065 | 0.1008 |
| har_mx_QLr z=drift (own drift) | 0.5263 | 0.8158 | 0.3006 | 0.0123 | 0.0054 | 0.1010 |
| har_mx_QLr z=centred (zero drift) | 0.5197 | 0.8127 | 0.3022 | 0.0071 | 0.0052 | 0.1008 |
| har_mx_QLr Gaussian (zero drift) | 0.5643 | 0.8394 | 0.3033 | 0.0032 | 0.0031 | 0.1083 |
| har_x_QLr z=zero | 0.5258 | 0.8164 | 0.3015 | 0.0094 | 0.0056 | 0.1009 |
| har_x_QLr z=drift (own drift) | 0.5301 | 0.8175 | 0.3008 | 0.0115 | 0.0045 | 0.1009 |
| har_x_QLr z=centred (zero drift) | 0.5221 | 0.8148 | 0.3024 | 0.0063 | 0.0043 | 0.1009 |
| har_x_QLr Gaussian (zero drift) | 0.5691 | 0.8424 | 0.3037 | 0.0019 | 0.0021 | 0.1087 |

#### h = 21 quantiles (VAL)

| model | cov50 | cov80 | pinball | skill | dev_skill | width80 |
|---|---|---|---|---|---|---|
| naive0 (sd63, Gaussian, zero drift) | 0.5150 | 0.7987 | 0.3085 | 0.0000 | 0.0000 | 0.2134 |
| naive_drift (sd63, Gaussian, own drift) | 0.5337 | 0.8118 | 0.3014 | 0.0229 | 0.0035 | 0.2134 |
| sd63 + DEV empirical z (zero) | 0.5024 | 0.7967 | 0.3041 | 0.0143 | 0.0079 | 0.2089 |
| har_mx_QLr z=zero | 0.5155 | 0.8106 | 0.3018 | 0.0218 | 0.0135 | 0.2069 |
| har_mx_QLr z=drift (own drift) | 0.5229 | 0.8174 | 0.2996 | 0.0288 | 0.0091 | 0.2080 |
| har_mx_QLr z=centred (zero drift) | 0.5066 | 0.8024 | 0.3050 | 0.0114 | 0.0085 | 0.2069 |
| har_mx_QLr Gaussian (zero drift) | 0.5399 | 0.8313 | 0.3069 | 0.0053 | 0.0047 | 0.2208 |
| har_x_QLr z=zero | 0.5150 | 0.8096 | 0.3022 | 0.0205 | 0.0122 | 0.2052 |
| har_x_QLr z=drift (own drift) | 0.5226 | 0.8170 | 0.2999 | 0.0280 | 0.0079 | 0.2063 |
| har_x_QLr z=centred (zero drift) | 0.5074 | 0.8027 | 0.3053 | 0.0103 | 0.0072 | 0.2052 |
| har_x_QLr Gaussian (zero drift) | 0.5418 | 0.8315 | 0.3072 | 0.0042 | 0.0032 | 0.2196 |

#### h = 63 quantiles (VAL)

| model | cov50 | cov80 | pinball | skill | dev_skill | width80 |
|---|---|---|---|---|---|---|
| naive0 (sd63, Gaussian, zero drift) | 0.4759 | 0.7747 | 0.3222 | 0.0000 | 0.0000 | 0.3675 |
| naive_drift (sd63, Gaussian, own drift) | 0.5208 | 0.8100 | 0.3006 | 0.0670 | 0.0052 | 0.3675 |
| sd63 + DEV empirical z (zero) | 0.4845 | 0.7907 | 0.3099 | 0.0381 | 0.0192 | 0.3670 |
| har_mx_QLr z=zero | 0.5040 | 0.8116 | 0.3065 | 0.0487 | 0.0250 | 0.3676 |
| har_mx_QLr z=drift (own drift) | 0.5282 | 0.8297 | 0.2995 | 0.0706 | 0.0104 | 0.3734 |
| har_mx_QLr z=centred (zero drift) | 0.4806 | 0.7896 | 0.3163 | 0.0183 | 0.0116 | 0.3676 |
| har_mx_QLr Gaussian (zero drift) | 0.5178 | 0.8280 | 0.3198 | 0.0075 | 0.0045 | 0.3958 |
| har_x_QLr z=zero | 0.4977 | 0.8076 | 0.3066 | 0.0483 | 0.0237 | 0.3596 |
| har_x_QLr z=drift (own drift) | 0.5231 | 0.8289 | 0.2993 | 0.0711 | 0.0092 | 0.3654 |
| har_x_QLr z=centred (zero drift) | 0.4753 | 0.7877 | 0.3165 | 0.0178 | 0.0102 | 0.3596 |
| har_x_QLr Gaussian (zero drift) | 0.5155 | 0.8256 | 0.3197 | 0.0078 | 0.0032 | 0.3878 |
