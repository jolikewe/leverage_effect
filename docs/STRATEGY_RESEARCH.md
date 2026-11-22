# Strategy research: can the Q5 volatility-targeted book be improved?

This file has two parts. **Part 1 (pre-registration)** was written and committed before any variant was run; it is not edited afterwards. **Part 2 (results)** is appended once the tests are done.

## Part 1: Pre-registration

### 1.1 Question and starting point

The Q5 strategy is an always-long book of 10 futures (5 equity-index "EQ", 5 government-bond "BD" legs), each leg sized to a 10% vol target from a HAR forecast of next-week variance, 20% no-trade band, trades at the next close, half-spread costs of 1bp (EQ) / 0.5bp (BD), walk-forward from 1995, results 1995-2017. The baseline to beat is **HAR, ALL book, net Sharpe 1.018** (return 4.28%, vol 4.20%, max drawdown -11.0%).

Findings of the study that motivate the hypotheses:

- The leverage effect exists only in equities, is about one month long and is asymmetric in the sense that falls raise vol.
- The equity/bond correlation is negative (-0.26 between the two books); most of the ALL Sharpe comes from bonds and from this diversification.
- HAR under-reacts (Mincer-Zarnowitz slope about 1.4, from an older run recorded in `NOTES.md`).
- Better QLIKE did not translate into better Sharpe (HAR-lev: lower QLIKE, lower Sharpe, bootstrap p = 0.030).

### 1.2 Fixed design (identical for baseline and all variants)

Data, features, walk-forward fitting (each January, purged), costs, 20% band, trade timing, 1995 start, book construction (`book()`: equal capital across live legs, ALL = all 10), and the Sharpe bootstrap (21-day blocks, 1000 resamples, seed 0, paired against HAR) are exactly those of `notebooks/leverage_effect.ipynb`. The instrument split is the notebook's (EQ = INST5, 6, 7, 8, 9; BD = INST0-4). Every input to a day-t position is known at the close of day t, and the baseline already delays it by one further day (`held = w.shift(1)`); variants keep this. Parameters are fitted only on data strictly before the test year, or are fixed constants stated below. Notation: for leg i, `s_i,t = vol_t * sqrt(yhat_t)` is the annualised forecast vol and the baseline target weight is `w_i,t = 0.10 / s_i,t`.

**Before any variant is run, the module `src/strategy.py` must reproduce the README baseline table (all 11 rows: return, vol, Sharpe, max drawdown) to the printed precision.** If it does not, no variant is tested until it does.

### 1.3 Hypotheses (5, no others will be tested)

None of these repeats a setup that `NOTES.md` records as tried and dropped. HAR-lev as a replacement for HAR, GJR-GARCH, EGARCH, log-HAR-lev, ML forecasts, rebalancing frequency and bands, trend variants, stops/targets and a reversal vote are excluded.

**H1. Sleeve-level risk parity with a lagged stock-bond correlation (finding: negative equity/bond correlation, most Sharpe from the diversification).**
Rationale: the baseline gives each leg the same risk capital, ignoring that the five equity legs are strongly correlated with each other, so the EQ sleeve carries more risk than the BD sleeve, and ignores the time-varying equity/bond correlation.
Specification. For each date, with live legs N_t, effective baseline weight `b_i = w_i / N_t`. `R^` = trailing 252-day pairwise correlation matrix of `ret_norm` (known at the close of t; pairs with fewer than 126 overlapping days get correlation 0). Sleeve vol forecast `sig_c = sqrt(b_c' S b_c)` with `S_ij = s_i s_j R^_ij` (this equals `(0.10/N_t) * sqrt(sum_ij R^_ij)` over the sleeve's legs, because inverse-vol weights cancel the forecast vols, so forecast vols enter at leg level and the correlations set the sleeve level). Stock-bond correlation `rho = b_E' S b_B / (sig_E sig_B)`, clipped to [-0.5, 0.9]. Equal-risk sleeve multipliers `M_c = K / sig_c` with `K = T / sqrt(2 + 2 rho)` and portfolio vol target `T = 4.2%` (the baseline's realised ALL vol, so the overall risk level is comparable; this is a constant, not tuned). New leg weight `w'_i = M_c * w_i`, then the same band, costs and book. Only the ALL book is defined for this hypothesis.

**H2. Variance-managed scaling (Moreira-Muir) instead of vol scaling (finding: HAR under-reacts to vol moves, so scaling harder against forecast risk might capture more).**
Specification. `w_i,t = 0.10 / s_i,t * m_i,t` with `m_i,t = clip(sbar_i,t / s_i,t, 0, 3)`, where `sbar_i,t` is the expanding mean of the leg's own trailing annualised vol (`vol` column) up to t. This makes position proportional to `1 / s^2` with a normalisation known at t and a leverage cap of 3x the baseline position. Applied to every leg (EQ and BD); books EQ, BD, ALL.

**H3. Walk-forward bias-corrected HAR (finding: Mincer-Zarnowitz slope 1.4).**
Specification. Each January of test year Y (from Y = 1998, i.e. after at least three prior forecast years; before that the raw HAR is used), for each category separately, regress realised `fwd_var_5d` on the baseline walk-forward forecast `yhat` over rows from years 1995..Y-1 whose 5-day target ended before 1 January Y (purged). Fitted `a + b * yhat` replaces `yhat` in year Y (floored at 0.1 as in the baseline). Books EQ, BD, ALL. No in-sample fitted values are ever used for the correction.

**H4. Forecast ensemble HAR / HAR-lev (finding: HAR-lev forecasts better, HAR sizes better; averaging may keep both).**
Specification. EQ legs: `yhat = 0.5 * (yhat_HAR + yhat_HARlev)` with the notebook's two walk-forward forecasts. BD legs stay on HAR (the notebook found no bond leverage model that beat HAR). Books EQ, ALL (BD identical to baseline).

**H5. Asymmetric equity sizing that never cuts after falls (finding: the leverage effect is asymmetric and about one month long; the notebook suggested HAR-lev loses P&L because it cuts exposure right after falls and misses the rebound).**
Specification. EQ legs: `yhat = min(yhat_HAR, yhat_HARlev)`. After a fall the leverage terms raise the forecast, so the minimum is HAR (no extra cut); after rallies the leverage terms lower it, so the minimum is HAR-lev (extra exposure while the measured effect says vol is subdued). BD legs stay on HAR. Books EQ, ALL. This is the only explicitly asymmetric rule; the notebook's own explanation of the HAR-lev shortfall is its motivation, and that explanation was never tested there. A second candidate in this family, a fixed de-risk multiplier for a fixed window after a large equity fall, is not tested: it would need free parameters (threshold, multiplier, length) that the study does not pin down, so it cannot be fully specified ex ante.

### 1.4 Success metric and decision rule

Primary metric: ALL-book net Sharpe, 1995-2017, against the HAR baseline 1.018; max drawdown alongside. Reported for every hypothesis: Sharpe, annual return, vol, max drawdown, turnover, vol of vol, block-bootstrap p-value vs HAR (two-sided, as in the notebook), sub-period Sharpe (1995-2006, 2007-2017) and Sharpe with all costs doubled (compared with the baseline at doubled costs).

Multiple-testing control:
- **Holm** step-down correction across the 5 bootstrap p-values (family-wise alpha 0.05).
- **Deflated Sharpe Ratio** (Bailey and Lopez de Prado, 2014) of each variant's daily ALL P&L, using skewness and kurtosis of that P&L, the benchmark `SR0 = sqrt(V) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)))`, g = Euler-Mascheroni constant, V = cross-sectional variance of the daily Sharpe ratios of the 6 strategies in this study (baseline + 5), and **N = 40 trials**. NOTES.md lists roughly 32 distinct configurations that were looked at (models, rebalancing alternatives, start years, cost check, trend lookbacks, long/short, stops and targets, reversal vote), so N = 40 is that count plus this study's 5, rounded up. Sensitivity at N = 6 and N = 100 is also shown.

**A variant is adopted as "v2" only if all of the following hold:** (i) Holm-adjusted p < 0.05; (ii) DSR at N = 40 is at least 0.95; (iii) its Sharpe exceeds the baseline's in both sub-periods; (iv) its Sharpe exceeds the baseline's at doubled costs; (v) its max drawdown is not worse than the baseline's by more than 2 percentage points. Otherwise the recommendation is to keep the baseline. Failing a variant on a single criterion is reported as such, and no criterion will be changed after seeing results.

### 1.5 What will not be done

No re-tuning of any constant above after seeing results, no extra hypotheses, no dropping of losers from the report, no changes to the baseline, start year, costs or band.

---

## Part 2: Results

Produced by `notebooks/02_strategy_improvements.ipynb` (code in `src/strategy.py`). The baseline was first reproduced exactly: all 11 rows of the README Q5 table plus turnover (ALL HAR 2.582) and the HAR-trend bootstrap p (0.86) match the original notebook. One README figure was corrected: ALL HAR-trend vol is 3.16 (unrounded 3.1645), not 3.17.

### 2.1 ALL book, net of costs, 1995-2017 (all five hypotheses reported)

| Strategy | Ann. return % | Ann. vol % | Sharpe | Max DD % | Turnover | Vol of vol % | p vs HAR | p Holm |
|---|---|---|---|---|---|---|---|---|
| HAR (baseline) | 4.28 | 4.20 | 1.018 | -11.0 | 2.58 | 0.98 | | |
| H1 sleeve risk parity | 3.61 | 3.89 | 0.928 | -12.6 | 4.51 | 0.71 | 0.076 | 0.304 |
| H2 variance-managed | 5.29 | 5.27 | 1.004 | -12.4 | 13.22 | 1.50 | 0.857 | 0.857 |
| H3 bias-corrected HAR | 4.23 | 4.31 | 0.980 | -11.6 | 3.39 | 0.96 | 0.083 | 0.304 |
| H4 ensemble HAR/HAR-lev | 4.24 | 4.21 | 1.007 | -10.8 | 3.10 | 0.99 | 0.305 | 0.610 |
| H5 asymmetric min(HAR, HAR-lev) | 4.26 | 4.29 | 0.994 | -11.1 | 2.77 | 1.00 | 0.031 | 0.155 |

p-values: 21-day block bootstrap, 1000 resamples, two-sided, vs HAR. H5's raw p = 0.031 is for a Sharpe that is lower than HAR's, so it is evidence of a worse variant, and it does not survive Holm.

### 2.2 Robustness

| Strategy | Sharpe 1995-2006 | Sharpe 2007-2017 | Sharpe, costs x2 | Max DD %, costs x2 |
|---|---|---|---|---|
| HAR (baseline) | 1.045 | 0.992 | 1.014 | -11.0 |
| H1 | 0.904 | 0.954 | 0.921 | -12.6 |
| H2 | 1.102 | 0.883 | 0.989 | -12.5 |
| H3 | 1.004 | 0.958 | 0.975 | -11.6 |
| H4 | 1.037 | 0.976 | 1.002 | -10.8 |
| H5 | 1.017 | 0.973 | 0.990 | -11.1 |

### 2.3 Deflated Sharpe Ratio

DSR (N = 40, with sensitivity at N = 6 and N = 100) equals 1.000 for every strategy including the baseline. The cross-trial variance of daily Sharpe ratios across these six strategies is tiny (annualised sd 0.032), so the best-of-N benchmark SR0 is far below a Sharpe near 1 measured on about 5,900 daily returns. The pre-registered DSR criterion is therefore met by all variants and does not discriminate; it says that each strategy's Sharpe is positive after accounting for selection, not that it beats HAR. The decision rests on the bootstrap, Holm and robustness criteria.

### 2.4 Pre-registered adoption rule

| Variant | Holm p < 0.05 | DSR >= 0.95 | Better 1995-2006 | Better 2007-2017 | Better at costs x2 | Max DD within 2pp | Adopt |
|---|---|---|---|---|---|---|---|
| H1 | no | yes | no | no | no | yes | no |
| H2 | no | yes | yes | no | no | yes | no |
| H3 | no | yes | no | no | no | yes | no |
| H4 | no | yes | no | no | no | yes | no |
| H5 | no | yes | no | no | no | yes | no |

### 2.5 Equity and bond books (context, not part of the decision rule)

| Book | Strategy | Sharpe | Max DD % | p vs HAR |
|---|---|---|---|---|
| EQ | HAR | 0.227 | -43.0 | |
| EQ | H2 | 0.269 | -40.2 | 0.556 |
| EQ | H3 | 0.198 | -48.1 | 0.193 |
| EQ | H4 | 0.213 | -43.5 | 0.314 |
| EQ | H5 | 0.211 | -45.3 | 0.192 |
| BD | HAR | 0.926 | -20.0 | |
| BD | H2 | 0.877 | -34.2 | 0.373 |
| BD | H3 | 0.922 | -20.5 | 0.646 |

(H4 and H5 leave bonds on HAR; H1 is defined for ALL only.)

### 2.6 Diagnostics

- Mincer-Zarnowitz slope of the baseline walk-forward forecasts over 1995-2017: equities 1.393 (intercept -0.480), bonds 0.952. The "slope 1.4" recorded in NOTES.md is an equity feature, and it reproduces here. The walk-forward correction H3 used had equity slopes of 1.39 to 1.70 across years; bond slopes were 0.91 to 1.02 from 1999.
- H1 sleeve multipliers relative to the baseline: EQ mean 1.029 (5%-95%: 0.871-1.208), BD mean 0.904 (0.554-1.131).

### 2.7 Conclusion

Nothing beats the baseline. No variant has a higher ALL Sharpe than HAR (best: H4 1.007), none has a Holm-adjusted p below 0.155, none beats HAR at doubled costs, and none is better in both sub-periods, so no "v2" is adopted. The recommendation is to keep the HAR baseline.

Reading by hypothesis:

- **H1** reduces vol and vol of vol (0.71 vs 0.98) but gives up return, and its daily-changing multipliers raise turnover from 2.6 to 4.5.
- **H2** takes more risk for proportionally more return (Sharpe unchanged within noise), at 5 times the turnover and a -34% bond-book drawdown.
- **H3**: the equity forecasts are indeed biased (slope 1.4) and the correction is stable, yet better-calibrated forecasts again do not improve P&L (EQ Sharpe 0.198 vs 0.227). This repeats the study's finding that forecast accuracy and Sharpe diverge.
- **H4, H5**: both mix in HAR-lev, which the study had already found to score below HAR as a sizing model; mixing it in dilutes HAR toward that result.

Limits: one 23-year sample, a handful of sleeves, and five hypotheses fixed in advance. These results show no evidence of improvement, not proof that none exists. The hypotheses were fixed, and committed, before any variant ran; the only change after the pre-registration commit was the README correction above (3.17 to 3.16), made while reproducing the baseline.

### 2.8 Addendum after the statistical review

Added later, after `docs/STATISTICAL_REVIEW.md`; nothing in Part 1 or in 2.1-2.7 is changed. The DSR uses a cross-trial variance estimated from the six strategies of this study, which are all close variants of HAR (annualised Sharpe sd 0.032). The earlier configurations in `NOTES.md` were more dispersed, so this variance is probably understated. Recomputed for an assumed sd of annualised Sharpe across trials, the baseline's DSR at N = 40 is 1.000 (sd 0.1), 0.997 (0.2), 0.958 (0.3) and 0.752 (0.4) (`notebooks/02_strategy_improvements.ipynb`, section 2). The pre-registered criterion would stop discriminating in favour of any strategy only for a dispersion above about 0.3, and it does not affect the decision, which failed on the Holm and robustness criteria. The pre-registered bootstrap p-values are two-sided; the hypotheses were that a variant beats HAR, and since every variant's Sharpe is below HAR's, one-sided p-values would be larger still and the conclusion is unchanged.
