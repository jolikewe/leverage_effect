# Statistical review

A referee-style audit of every statistical test in `notebooks/leverage_effect.ipynb` (Q1-Q5) and `notebooks/02_strategy_improvements.ipynb`. Each item gives what was done, whether it was right, the fix if one was needed, before and after numbers, and whether a conclusion changed. All "after" numbers are stored outputs of the notebooks: Q1 and Q2 in sections 2, the robustness checks in section 7 ("Inference checks") of the main notebook, and the DSR sensitivity in section 2 of the strategy notebook.

Severity: **wrong** = the test as run was mis-specified; **weak** = defensible but under-justified, incomplete or worded too strongly; **fine** = correct as run (a check was still added where useful).

## Summary

| # | Where | Issue | Severity | Action | Conclusion changed? |
|---|---|---|---|---|---|
| 1 | Q2 | Trend slope of yearly S tested with HC0 errors and a normal reference on 16-34 points | wrong | HC3 + t(n-2), permutation test, Mann-Kendall + Theil-Sen | No; "no trend" is now better supported (p rises) |
| 2 | Q2 | "% of years with S < 0" reported without a test | weak | Exact one-sided binomial sign test | No |
| 3 | Q2 | Unequal precision of yearly S, partial first/last years | fine | Documented; day-weighted check | No |
| 4 | Q1 | Two-sided p-value for a directional, pre-specified hypothesis (S < 0) | wrong | One-sided primary p, two-sided kept as secondary | No; same 6 rejections |
| 5 | Q1 | Bonferroni rule stated on z (\|z\| > 2.8, a two-sided normal cut-off) | weak | Decisions on the permutation p (p <= 0.005); one-sided z cut-off 2.58 shown on the page | No |
| 6 | Q1 | 1000 surrogates: enough resolution? | fine | Checked | No |
| 7 | Q1 | z = S / sd(S*) without subtracting the null mean | fine | Checked | No |
| 8 | Q3 | Newey-West with 10 lags for overlapping 5-day targets | fine | Sensitivity at 22 and 44 lags | No |
| 9 | Q3 | Multiple testing across 10 instruments | fine | Checked | No |
| 10 | Q4 | Same-close timing of the source return | fine | Previous-day source return as a check | No |
| 11 | Q4 | Basket p = 0.018 presented without multiplicity context | weak | Wording: Holm over the two basket tests 0.036; not study-wide | Weakened in wording |
| 12 | Q5 | Diebold-Mariano: lag choice, small-sample correction, nested models | fine | Lag sensitivity and HLN correction added | No |
| 13 | Q5 | Sharpe bootstrap: centring, block length, studentisation | fine / weak | Block-length sensitivity and a Ledoit-Wolf HAC test added | No |
| 14 | Q5 | Single Sharpe p-values (0.030, 0.044, 0.055) read as significant | wrong (wording) | Holm across the eight Sharpe comparisons; text corrected | **Yes**: HAR-lev's lower Sharpe and HAR's bond-Sharpe gain are no longer called significant |
| 15 | all | Look-ahead in features, targets, trades | fine | Checked; one in-sample choice documented | No |
| 16 | Strategy research | Holm, DSR, two-sided pre-registered p | fine / weak | DSR sensitivity to the cross-trial variance added | No |

The headline results stand: the equity leverage effect is significant under any correction, stable across years and without trend; returns predict volatility and not the reverse; better volatility forecasts are significantly more accurate. What weakens is everything resting on a single p-value between 0.01 and 0.06: the cross-market spillover in Q4 and every Sharpe-ratio comparison in Q5.

## 1. Q2 trend test (wrong, fixed)

**What was done.** `stability()` regressed yearly S on year with `hac_ols(..., lags=0)`. In statsmodels, `cov_type="HAC"` with `maxlags=0` is the White (HC0) estimator, and HAC fits use a normal reference distribution. With 16 to 34 yearly points, HC0 under-estimates the standard error (it has a known downward bias that HC3 corrects) and the normal reference ignores the t tails, so the test over-rejects. The yearly values are also heavy-tailed (crisis years at -8 to -16), which makes the OLS slope depend on a few years.

**Fix.** Four tests, each suited to small samples:
- OLS slope with HC3 errors and a t(n-2) reference (`fit(cov_type="HC3", use_t=True)`).
- Permutation test of the OLS slope: 10,000 random orderings of the years (seed 0); exact up to simulation error, assuming years are exchangeable under no trend. Lag-1 autocorrelation of yearly S is -0.17 to 0.02, so this is reasonable.
- Mann-Kendall test (Kendall's tau of S against year, exact p) with the Theil-Sen slope: rank-based, so the crisis years do not dominate.

**Before and after (p-values of the trend).**

| Instrument | n years | OLS slope | p before (HC0, normal) | p HC3, t(n-2) | p permutation | Theil-Sen slope | p Mann-Kendall |
|---|---|---|---|---|---|---|---|
| INST0 | 26 | -0.004 | 0.935 | 0.943 | 0.919 | 0.005 | 0.793 |
| INST1 | 34 | 0.013 | 0.502 | 0.540 | 0.566 | 0.014 | 0.345 |
| INST2 | 29 | 0.078 | 0.114 | 0.167 | 0.195 | 0.030 | 0.342 |
| INST3 | 34 | 0.024 | 0.325 | 0.373 | 0.320 | 0.017 | 0.147 |
| INST4 | 27 | 0.028 | 0.485 | 0.540 | 0.421 | 0.005 | 0.773 |
| INST5 | 19 | -0.053 | 0.512 | 0.584 | 0.722 | -0.041 | 0.534 |
| INST6 | 26 | -0.063 | 0.284 | 0.335 | 0.483 | -0.034 | 0.314 |
| INST7 | 28 | 0.054 | 0.611 | 0.659 | 0.586 | 0.000 | 1.000 |
| INST8 | 16 | 0.016 | 0.758 | 0.805 | 0.822 | -0.005 | 0.965 |
| INST9 | 32 | -0.032 | 0.546 | 0.591 | 0.490 | -0.049 | 0.077 |

**Conclusion.** Unchanged and better supported: no instrument has a trend. Equity p-values were 0.28-0.76 and are now 0.34-0.81 (HC3) and 0.48-0.82 (permutation). The smallest p-value is INST9's Mann-Kendall 0.077 (S becoming slightly more negative over time), which is not significant even before a correction for 10 instruments. As before, with 16-32 points per equity index only a large trend could be detected.

## 2. Q2 share of negative years (weak, fixed)

`pct_years_neg` was reported as evidence of persistence without a test. Under no leverage effect, yearly S is symmetric about 0 (the sign-flip argument applied to each year), so the number of negative years is Binomial(n, 0.5). Added an exact one-sided binomial test.

| | INST5 | INST6 | INST7 | INST8 | INST9 | INST0 | INST1 | INST2 | INST3 | INST4 |
|---|---|---|---|---|---|---|---|---|---|---|
| negative years | 19/19 | 24/26 | 25/28 | 16/16 | 30/32 | 15/26 | 21/34 | 18/29 | 23/34 | 15/27 |
| sign test p | 1.9e-6 | 5.2e-6 | 1.4e-5 | 1.5e-5 | 1.2e-7 | 0.279 | 0.115 | 0.132 | 0.029 | 0.351 |

Every equity index passes at the Bonferroni level 0.005; no bond does (INST3, 0.029, passes only without correction). The conclusion "equities persistent, bonds a coin toss" is unchanged and now has a test behind it.

## 3. Q2 unequal precision and partial years (fine)

Years need at least 200 trading days, so the first, partial year of each series is dropped and the partial last year (2017, about 205 days) is kept. S is a covariance (a mean of products), so its expectation does not depend on the number of days, only its variance does. Days per kept year range from 202 to 259, so precision differs far more through the volatility regime than through day counts. A regression weighted by days changes no slope by more than 0.01 and no HC3 p-value by more than 0.07 (none comes near significance). The rank-based tests (item 1) are the right answer to the regime-driven heteroskedasticity. No change.

## 4. Q1 one-sided vs two-sided sign-flip test (wrong, fixed)

**What was done.** `z_and_p` counted surrogates with |S*| >= |S|, a two-sided p-value, while the notebook's H1 was written as S != 0.

**Decision.** The leverage hypothesis is directional and fixed before looking at the data: falls are followed by higher volatility, S < 0. For a pre-specified directional hypothesis the correct primary test is one-sided, left-tailed: p = (#{S* <= S} + 1)/(B + 1). It is applied to all 10 instruments, because the purpose of Q1 is to find which instruments show a leverage effect and the asset classes are not known in advance. Bonds could in principle show the opposite sign (flight to quality: bond prices rise when market-wide volatility rises). That is a different hypothesis, so it is not folded into the primary test; the two-sided p-value is kept as a secondary, exploratory column that would flag S > 0. No instrument has S > 0, so the secondary test finds nothing.

**Before and after.**

| Instrument | S | z | p before (two-sided) | p after (one-sided) | Reject at 0.005 |
|---|---|---|---|---|---|
| INST5, 6, 7, 8, 9 | -1.94 to -3.12 | -5.02 to -6.34 | 0.001 | 0.001 | yes |
| INST2 | -1.292 | -3.54 | 0.001 | 0.001 | yes |
| INST4 | -0.447 | -2.41 | 0.017 | 0.010 | no |
| INST0 | -0.409 | -1.86 | 0.062 | 0.026 | no |
| INST3 | -0.306 | -1.60 | 0.118 | 0.055 | no |
| INST1 | -0.216 | -1.21 | 0.227 | 0.119 | no |

**Conclusion.** Unchanged: the same six instruments reject (the five equity indices and INST2). INST4 and INST0 are now below 0.05 one-sided, but neither survives Bonferroni.

## 5. Q1 Bonferroni rule on z (weak, fixed)

The text and the page used "|z| > 2.8", the two-sided normal cut-off for 0.005. Decisions now use the permutation p-value directly (p <= 0.05/10). The page's z chart draws the one-sided cut-off, |z| = 2.58, which sits between INST2 (3.54) and INST4 (2.41). Decisions are unchanged.

## 6. Q1 resolution of 1000 surrogates (fine)

The p-value floor is 1/1001 = 0.001, five times below the Bonferroni threshold 0.005, so 1000 surrogates can reject. The closest case to the threshold is INST4 (one-sided p = 0.010, simulation standard error about 0.003). The surrogate distribution is close to normal (skew within ±0.15, excess kurtosis within ±0.55), and the normal approximation gives Phi(-2.405) = 0.008 for INST4, which agrees: INST4 is not significant at 0.005. For instruments at the floor, z (-3.5 to -6.3) gives the finer resolution. No change.

## 7. Q1 centring of z (fine)

Under the sign-flip null, S* is centred on 0 by symmetry. The simulated means are -0.019 to 0.008, within simulation noise (sd(S*)/sqrt(1000) = 0.006 to 0.020). Subtracting them would add noise rather than remove bias, and would move z by at most 0.05. No change.

## 8. Q3 Newey-West lags (fine)

Overlapping 5-day targets make the errors at least MA(4); volatility targets can be more persistent than that. 10 lags (Bartlett kernel) matches the Newey-West rule of thumb 4(T/100)^(2/9) ≈ 10 for T ≈ 6500. Check (section 7):

| | gamma p, 10 lags | 22 lags | 44 lags |
|---|---|---|---|
| Equities (5) | 2.7e-5 to 4.4e-10 | 1.3e-5 to 5.2e-11 | 6.3e-6 to 3.8e-11 |
| INST2 | 9.3e-5 | 3.5e-5 | 4.3e-5 |

Vol to returns: INST9's joint test stays significant (3.1e-6, 1.2e-8, 1.0e-12); INST7's 0.030 becomes 0.025. Conclusions unchanged.

## 9. Q3 multiple testing (fine)

The equity gamma p-values (all below 3e-5) survive Bonferroni over 10 instruments (0.005) and over both directions (0.0025). Among bonds only INST2 (9.3e-5) survives; the text never claimed INST0 (0.030) or INST4 (0.014). For vol to returns, INST9 survives Bonferroni in the full sample (3.1e-6) and ex-crises (8.2e-4), which is the exception the notebook already reports. No change.

## 10. Q4 timing (fine)

Using the source market's same-day return assumes both markets close at the same time. NOTES.md reported a check with the previous day's return; it is now in the notebook (section 7) and reproduces: basket theta = -0.036, p = 0.008. Equity-to-bond p-values are 0.018 at 10, 22 and 44 lags. The overlapping targets are covered by the same Newey-West errors as Q3. No change.

## 11. Q4 strength of evidence (weak, wording fixed)

The pre-specified main tests are the two basket regressions (equities to bond vol, bonds to equity vol). p = 0.018 survives a Holm correction over those two (0.036), but a single p of 0.018 among the dozens of tests in the study would not survive a study-wide correction, and the effect is small (R-squared 0.155 to 0.161). README, notebook text and page now call it modest. The pairs (INST9 to INST3 p = 0.003, INST6 to INST0 p = 0.014) are supporting detail, not separate claims.

## 12. Q5 Diebold-Mariano (fine)

The daily cross-sectional mean of QLIKE differences is tested with Newey-West errors (10 lags, for overlapping 5-day targets). Check (section 7):

| | 4 lags | 10 lags | 22 lags | 44 lags | HLN (h = 5) |
|---|---|---|---|---|---|
| EQ HAR vs trailing vol | 1.7e-17 | 6.2e-11 | 6.9e-8 | 1.6e-6 | 6.9e-11 |
| EQ HAR-lev vs HAR | 1.4e-9 | 1.1e-8 | 2.0e-8 | 2.8e-8 | 1.2e-8 |
| BD HAR vs trailing vol | 7.7e-26 | 1.2e-17 | 2.1e-14 | 2.7e-12 | 1.6e-17 |
| BD HAR-lev-cross vs HAR | 0.071 | 0.090 | 0.104 | 0.120 | 0.090 |

With about 5,900 days the Harvey-Leybourne-Newbold correction is immaterial. HAR-lev nests HAR, and DM p-values for nested models are conservative under the null (Clark and West, 2007), which only strengthens a rejection; with p around 1e-8 at every lag this does not matter. The QLIKE conclusions survive any multiplicity correction. No change.

## 13. Q5 Sharpe bootstrap (fine, cross-checked)

- **Centring.** `sharpe_p_value` returns the share of bootstrap gaps with |gap* - gap| >= |gap|: the bootstrap distribution is centred on the observed gap, which is the correct way to impose the null. Fine.
- **Block length.** 21 days was not derived from a data-driven rule. Sensitivity (section 7): blocks of 5, 10 and 63 days give p-values close to the 21-day ones (for example ALL HAR-lev vs HAR: 0.045, 0.039, 0.034 vs 0.030; ALL HAR vs trailing vol: 0.054, 0.047, 0.067 vs 0.055).
- **Studentisation.** The bootstrap is not studentised, which Ledoit and Wolf (2008) note can be less accurate. Their HAC delta-method test gives similar p-values (ALL HAR-lev vs HAR 0.040, ALL HAR vs trailing vol 0.047, ALL trend vs HAR 0.847).
- **Start range.** Block starts are drawn from 0 to n - block - 1, so the final day is never resampled, an edge effect of one start in about 5,800. A fix would change every stored and pre-registered bootstrap p-value by simulation noise only (a different seed moves them by a similar amount: 0.021 to 0.036 for ALL HAR-lev vs HAR), so it was left unchanged and is recorded here.

No change to the method.

## 14. Q5 multiple Sharpe comparisons (wrong wording, fixed)

Steps 2 and 3 run eight Sharpe comparisons (HAR vs trailing vol on EQ, BD, ALL; HAR-lev vs HAR on EQ, ALL; HAR-trend vs HAR on EQ, BD, ALL). The notebook's findings and insights called the BD gain of HAR (p = 0.044) and the lower Sharpe of HAR-lev (p = 0.029-0.030) significant ("HAR earns significantly better returns than HAR-lev"). Holm-adjusted across the eight, the smallest p-value is 0.232; across only the three ALL-book comparisons it is 0.09 (HAR-lev vs HAR) and 0.11 (HAR vs trailing vol).

**Conclusion changed.** None of the Sharpe differences between sizing models is established. HAR's improvement over trailing volatility and HAR-lev's shortfall against HAR are suggestive only. The QLIKE results (item 12) are unaffected, so the study's point that better forecasts do not reliably give a better Sharpe is, if anything, stronger. README, NOTES.md, the notebook text (sections 5, 6) and the page now say so.

## 15. Look-ahead (fine)

- `ret_norm` = r_t / sigma_{t-1}: the trailing 252-day vol ends the day before, so the normalisation is known before the return.
- The 5-day targets are scaled by sigma_t, known at the close of t; HAR inputs are means of past `ret_norm`^2.
- Strategy: the target weight at close t uses vol_t and the forecast from features known at t; it is held from close t+1 (`w.shift(1)`), a one-day extra delay. Walk-forward fits each January use only rows whose 5-day target ended before 1 January (purging). Trend votes use returns up to t.
- Q1's L(tau) demeans with the full-sample mean and Q2 scales by the full-sample variance: descriptive statistics, not forecasts, so this is not look-ahead.
- One in-sample choice remains: the equity/bond split used in Q5 (which legs get leverage terms and which cost applies) comes from the full-sample S in Q1. The split is a fixed fact about the instruments (it is also visible in the return correlations) and is not tuned, so this was left as is and is noted here.

## 16. Strategy research (fine, one sensitivity added)

- **Holm.** `holm()` implements the step-down correctly, including monotonicity; the family is the five pre-registered ALL-book tests. Fine.
- **Two-sided p-values.** The pre-registration fixed two-sided bootstrap p-values while each hypothesis was that a variant beats HAR. Since every variant's Sharpe is below HAR's, one-sided p-values would be larger still; the result (no adoption) is unchanged. The pre-registered choice is kept.
- **DSR.** The formula follows Bailey and Lopez de Prado (2014), with raw kurtosis. The cross-trial variance V comes from the six strategies of the study, all close variants of HAR (annualised Sharpe sd 0.032), while N = 40 counts earlier, more dispersed configurations. V is therefore likely understated. Sensitivity of the baseline's DSR (strategy notebook, section 2):

| sd of annualised Sharpe across trials | N = 6 | N = 40 | N = 100 |
|---|---|---|---|
| 0.032 (as used) | 1.000 | 1.000 | 1.000 |
| 0.1 | 1.000 | 1.000 | 1.000 |
| 0.2 | 1.000 | 0.997 | 0.993 |
| 0.3 | 0.999 | 0.958 | 0.893 |
| 0.4 | 0.992 | 0.752 | 0.511 |

The DSR conclusion (a positive Sharpe after selection) holds unless trials were dispersed by more than about 0.3 in Sharpe. The adoption decision did not rest on the DSR, so it is unchanged. Recorded in `docs/STRATEGY_RESEARCH.md` §2.8 without editing the pre-registration.

## Study-wide multiple testing

Each question has its own family: Bonferroni over 10 instruments in Q1, Q2 and Q3, Holm over the two basket tests in Q4, Holm over the eight Sharpe comparisons in Q5 and over the five variants in the strategy research. Across the whole study the claims that matter survive a strict correction because their p-values are far from any threshold (equity S: one-sided p at the 0.001 floor with z <= -5; equity gamma <= 3e-5; equity sign tests <= 2e-5; QLIKE gains <= 1e-8). The claims with p between 0.01 and 0.06 (Q4 spillover, all Sharpe differences) do not, and are now worded as suggestive.

## Code changes

- `notebooks/leverage_effect.ipynb`: `z_and_p` returns (z, one-sided p, two-sided p); Q1 table shows both and the Bonferroni decision; `stability()` replaced by the small-sample tests above; new section 7 with the robustness checks; texts in sections 2, 5 and 6 updated. `groupby(level=1, axis=1)` replaced by the equivalent `.T.groupby(level=1).mean().T`, and `pd.concat(..., sort=True)` made explicit (the old default), so the notebook runs on pandas 3. All other stored outputs are identical to the previous run.
- `notebooks/02_strategy_improvements.ipynb`: DSR sensitivity cell. All other outputs identical.
- `src/strategy.py`: explicit `sort=True` in `load_returns` (same behaviour, no deprecation warning).
