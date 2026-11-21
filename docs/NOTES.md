# Notes

## Assumptions
- $r_t=(P_t-P_{t-1})/P_{t-1}$ is assumed roll-adjusted (no jumps in the price path).
- Futures P&L needs no cash outlay, so returns are already excess returns.

## Data
- Raw `ann_vol_pct` splits into two tiers even before any classification (≈5% vs ≈20%); the high-vol tier's vol also spikes 2.6-7.6× in 1987/2008.
- **Why normalise** ($x_t=r_t/\sigma_{t-1}$): instruments sit at very different vol levels (≈5% to ≈20% annualised). Comparing or pooling raw returns across them would mostly just reflect who's more volatile, not whose *behaviour* differs. Dividing each return by that instrument's own trailing vol puts them on one scale, so later comparisons measure the actual pattern, not just volatility. $\sigma_{t-1}$ is lagged (known at the time, no lookahead).
- Var($x$) ends up 1.01-1.13, not exactly 1, because trailing $\sigma$ lags real vol jumps — right after a shock, $x$ briefly overstates how extreme the move was relative to the (stale) "normal." A small, known bias, not a flaw in the method.
- $x^2$ is used throughout as a one-day variance estimate, even though technically $E[r^2]=\sigma^2+\mu^2$ (variance plus mean-squared). Fine here because the daily mean is tiny next to volatility ($\mu^2/\sigma^2\le0.002$), so squared return is effectively just variance in practice.
- Daily returns show essentially no autocorrelation (-0.06 to +0.03): today's return doesn't predict tomorrow's sign. This matters later — a test that randomly reshuffles each day's sign (Q2) isn't destroying any real pattern, because there's barely any to begin with.
- Log returns don't average simply across instruments (a basket's log return isn't the average of its members' log returns — only simple returns work that way). That's why sections that average several instruments into one "basket" series (Q4) use simple, not log, returns.

## Q1: leverage effect
- **$\mathcal L(\tau)$ as a slope:** OLS slope of $\tilde x^2_{t+\tau}$ on $\tilde x_t$ is cov/var = $\mathcal L(\tau)\cdot\text{var}(x)\approx\mathcal L(\tau)$. Summing over $\tau$, $S\approx$ the slope of next month's *summed* squared return on today's return (covariance of a sum = sum of covariances); checks out within ~10% (INST7: $-2.76\times1.13=-3.12$).
- **"Up minus down" reading:** $\text{cov}(\tilde x_t,y)\approx\tfrac12(E[my\mid\text{up}]-E[my\mid\text{down}])$, $m=|\tilde x_t|$. E.g. INST7's next-22-day avg $x^2$ is 1.23 after down days vs 1.06 after up days; INST1 (a bond) is 1.02 vs 1.01 — same mechanism, much weaker.
- $S\approx A\cdot T$ (amplitude × decay time) to a good approximation.
- **Why equities and not bonds:** an equity fall triggers real mechanisms that raise future vol — forced deleveraging (Black 1976), volatility feedback (Campbell & Hentschel 1992), panic selling. A bond price fall is just a yield rise, with no such mechanism, so there's little reason to expect an effect there. If anything bonds point the other way: during an equity crisis, investors sell stocks and buy government bonds for safety ("flight to quality") — demand pushes bond *prices up* at exactly the moment market-wide vol is elevated (driven by the equity side).

## Q2: significance and stability
- All 5 equities hit the p-value floor ($1/1001$, below the Bonferroni threshold); $z$ still differentiates them (-5.0 to -6.3), which is why both are reported.
- **Yearly $S$ scaling:** scaled by the *full-sample* variance², not each year's own — using a year's own variance would make calm years look spuriously extreme (e.g. an early INST9 run showed a calm year at $S=+12$ before this fix).
- **Why calendar years, not rolling windows, for the stability check:** rolling 1-year windows share 251/252 days with their neighbour, so a naive SE would be ~16× too small. Non-overlapping calendar years share no days, so plain OLS with robust (not HAC) SEs is already valid.

## Q3: direction
- Leverage terms ($\gamma x_t+\delta|x_t|$) are a simplified Corsi & Renò (2012), who use past negative returns over day/week/month; one coefficient $\gamma$ suffices here since past-week falls are already absorbed by HAR's variance terms.
- **$\gamma,\delta$ as one bent line:** up-day slope is $\gamma+\delta$, down-day slope is $\gamma-\delta$. corr($x,|x|$) is only -0.04 to -0.15, so $\delta$ barely moves $\gamma$ (INST5: -0.249 without $\delta$'s info vs -0.251 with) — $\delta$ is insignificant for 4 of 5 indices, so the response is close to linear.
- Reverse-direction regression (vol → returns) follows Ghysels, Santa-Clara & Valkanov (2005); the literature's evidence here is mixed.
- The $z$ columns (model-free sign-flip test) and the regression p-values are two independent tests — no model vs. controlling for persistence — shown side by side because they agree.
- Robustness: 1, 10 and 22-day target horizons give the same conclusions (INST5's $\gamma$ moves from -0.25 to -0.16 but stays significant).
- The two excluded crisis windows are not small: Oct-Dec 1987 and 15 Sep-Dec 2008 are the two largest-variance months in the sample (469 and 200, vs 121 for the next-largest).

## Q4: cross-category
- **Timing robustness:** using the source market's *previous* close ($y_{t-1}$, removing the same-close assumption) gives $\theta=-0.036$, $p=0.008$ — same conclusion.
- There's a separate, larger bond-vol → equity-vol spillover at the basket level ($p\approx0.002$-$0.005$); that's a different question (vol-to-vol, not the directional leverage effect) and isn't what $\theta$ tests.

## Q5: strategy

*`START_YEAR` was tested at 2001 and 2003 before settling back on 1995 — every figure in this section uses 1995. A few robustness-check figures below aren't tied to this exact setup (the rebalancing-alternatives table, the older-run GJR-GARCH/log-HAR-lev comparison, and most of Step 3's extra-robustness checks) and haven't all been reconfirmed on it — read them as indicative, not exact.*

### Design choices not argued in the notebook
- Costs: halving the assumed spread changes Sharpe by ≤0.03 — the choice of 1bp/0.5bp half-spread isn't driving the results.
- Walk-forward forecasts across all years are stitched into one daily P&L series (1995-2017), so there's no separate hold-out to report; purging (dropping training rows whose target leaks into the test year) makes an embargo unnecessary since training data is always strictly earlier.
- **Why the model-comparison chart only plots equity legs:** bonds always use the same HAR forecast no matter which named strategy is shown ("HAR" and "HAR-lev" are identical for bonds — only equities ever get the leverage terms), so a bond version of that chart would just show one line sitting on top of itself twice. Equities are the only legs where the three models actually differ.
- **Why `START_YEAR` = 1995, despite the EQ panel being thin early on:** by 1995 only 3 of 5 equities are trading (INST5 starts 1997, INST8 starts 2000) — a thin, unrepresentative training set for the pooled regression in the earliest years. Tested moving it to 2001 and 2003 anyway: training length barely matters once the *evaluation* window is held fixed (e.g. evaluating 2001-2017 gives ALL HAR Sharpe ≈0.91 whether training starts in 1995, 1998 or 2001) — what moves the number is which *years get evaluated*, not how much is trained on. 2001 is the second-worst year in the whole sample (-4.2% return, Sharpe -0.98, almost as bad as 2008), so evaluating from 2001 vs 2003 swings the measured Sharpe a lot (0.91 vs 1.10) purely from including or excluding that one year. Moving the start later looks better mainly because it quietly drops a bad stretch, not because the model trains better, which is why it was reverted to the full, honest 1995-2017 window.

### NAV, notional and the 0.10
- **NAV** = account equity (fixed at 1). **Notional** = contracts × multiplier × price. $w=$ notional/NAV. Gross leverage $\sum|w_i|\approx1.2\times$ (~80% bonds, since bonds need far more notional for the same vol).
- Return is P&L ÷ NAV (what the investor actually gains/loses), not P&L ÷ notional (just the market's own return at any size). Without the 0.10 vol target, book vol would be ≈42% and max drawdown ≈-110% of NAV — everything scales ×10 except Sharpe.

### QLIKE and the Diebold-Mariano test
- QLIKE depends only on the ratio $y/\hat y$, so calm and crisis days count equally and it's robust to noise in the realised-variance proxy (Patton 2011).
- DM test: per instrument-day QLIKE difference, averaged across instruments into one $d_t$ per day (~5,800 days), then a Newey-West t-test of $\bar d=0$ (10 lags, since 5-day targets overlap).
- **Why QLIKE as well as Sharpe:** P&L is driven by price direction, so Sharpe is blind to small forecast-accuracy gains — e.g. HAR-lev beats HAR on QLIKE at $p\approx10^{-8}$ while their Sharpes are statistically indistinguishable.

### Fitting vs. scoring
- OLS fits by squared error; QLIKE scores by a different loss; both are minimised by the same true conditional mean, so fitting by OLS and scoring by QLIKE is consistent. (A QLIKE-weighted fit would be iterated WLS with weights $1/\hat y^2$; not used here.)

### Sharpe of a book
- **Why a bootstrap instead of a simple test:** a test like Diebold-Mariano (above) works because there's a *daily* quantity to difference and average — QLIKE$_{A,t}$ − QLIKE$_{B,t}$ each day, then just average those differences, a classic paired mean. A Sharpe ratio is mean/std computed over the *whole* sample at once; there's no "Sharpe for Monday" to difference day by day, so that approach doesn't apply. Instead: resample 21-day blocks of the real paired daily returns (preserving their autocorrelation) 1000 times, compute each resample's two Sharpes, and see how big the gap between them gets from resampling noise alone.
- **Reading `sharpe_p_value`:** $H_0$ = the two strategies have the same true Sharpe, so any observed gap is noise. $p$ = how often resampling (using the data's own noise) produces a gap at least as large as the one actually observed. Low $p$ = a gap that size would be rare under $H_0$ → good evidence it's real. High $p$ = gaps that size happen easily from noise alone → can't rule out luck. $p$ does *not* measure how good the gap is — only whether it's trustworthy.
- **Why 21 days?** It matches the calendar-month granularity already used elsewhere (HAR's monthly lag, Q1's 22-day $S$ window) — it wasn't derived from a formal optimal-block-length procedure (e.g. Politis & White 2004). Block-bootstrapping autocorrelated daily P&L for a Sharpe-ratio p-value is standard practice (Politis & Romano for block bootstrap generally, Ledoit & Wolf 2008 specifically for comparing Sharpe ratios), not something invented for this notebook; a naive formula assuming i.i.d. normal daily returns would be wrong here given the fat tails found in Data exploration (excess kurtosis up to 13).
- **Why ALL can beat both legs:** $SR_{ALL}=\dfrac{\mu_E+\mu_B}{\sqrt{\sigma_E^2+\sigma_B^2+2\rho\sigma_E\sigma_B}}$. With HAR's $\mu$ 1.5%/6.6%, $\sigma$ 6.6%/7.1%, $\rho=-0.26$: implied Sharpe ≈ (1.5+6.6)/√(6.6²+7.1²+2×(-0.26)×6.6×7.1) ≈ 8.1/8.38 ≈ 0.97, close to but a bit under the actual 1.02 (the gap is the thin early EQ panel: live-leg weights vary year to year, so ALL isn't exactly a 50/50 blend of EQ and BD) — the negative equity/bond correlation is what lets the combined book beat either part.

### Rebalancing alternatives tested (not in notebook)
- Rebalancing frequency (HAR, ALL Sharpe): daily 0.99, weekly 0.97, monthly 0.94, 20% band (used) 1.02. 10%/30% bands: 1.01/0.99. Choice doesn't drive the result.

### Models considered but not used
- **log-HAR-lev:** the same HAR-lev regression, but fit on log(variance) instead of variance itself, so a forecast can never go negative and big vol spikes don't dominate the fit as much.
- **GJR-GARCH** (Glosten, Jagannathan & Runkle 1993): a classic volatility model that lets yesterday's return move today's variance by a different amount depending on whether it was up or down — the same "falls raise vol more" asymmetry our leverage terms capture, but fit as a parametric variance process via maximum likelihood instead of a plain OLS regression.
  - (older run, costs 2bp/1bp) Both forecast equity vol 9-11% better than HAR (QLIKE 0.356/0.363 vs 0.401; HAR under-reacts to vol moves — Mincer-Zarnowitz slope 1.4) but earn less P&L (EQ Sharpe 0.08/0.14 vs 0.20 for HAR): accuracy and P&L point opposite ways, which is why HAR is used for sizing and HAR-lev only for risk numbers.
- **EGARCH** (Nelson 1991): same family as GJR-GARCH — another classic asymmetric volatility model, modelling log-variance with its own up/down asymmetry term. Like GJR-GARCH, it and other GARCH-family models only run returns→vol (so we can't reuse them for Q4's reverse-direction check, vol→returns) and are fit by maximum likelihood, a less transparent estimation step than our OLS regressions.
- **VAR (vector autoregression) / Granger causality:** tests whether one series' past *signed* returns predict another's, but doesn't separate "how big a move was" from "which direction it was" — exactly what our leverage terms ($x_t$ and $|x_t|$ as separate regressors) are built to do, so VAR/Granger would need the same extra terms added anyway to say anything new.
- **Stochastic volatility models:** treat volatility itself as a random process with its own separate source of noise (not just a function of past returns), fit with specialised simulation-based methods (e.g. particle filters) — heavier machinery than a yes/no question about whether a leverage effect exists requires.
- **Machine learning** (e.g. gradient-boosted trees, neural nets): tried on the same handful of HAR-style inputs (past day/week/month variance, past return) — they just relearn the same relationship OLS already finds, no accuracy gain. A real gain would need genuinely new information (intraday realised vol, options-implied vol), not a fancier model on the same inputs.

### Step 3: trend, extra robustness
- **Sizing trend with HAR-lev instead of HAR:** ALL Sharpe 0.99 (baseline) → 1.02 (trend), $p=0.82$ — both numbers ~0.02-0.03 below the HAR-sized version (0.99 vs 1.02, 1.02 vs 1.04), matching the gap HAR-lev already showed on its own in Step 2. Trend's own lift and its lack of significance don't change; only the sizing model's own cost carries through.
- Single lookbacks alone (same-close timing): 1-month 1.06, 12-month 1.10, vs the blended 1/3/12 at 1.17 — the blend isn't cherry-picked, just not the best possible.
- **No shorting:** long/short scored 0.64 (loses the stock-bond hedge, pays the premium on shorts, doubles turnover); shorting only on clear downtrends ties long/flat at best.
- **Stops/targets tried and dropped:** per-trade take-profit at $k$ annual vols scored 0.28-1.04 for $k\in[0.25,2]$ (worse — it sits out bull trends); a stop below the 3-month high scored 0.76-1.00. Exit rules just reshape outcomes when returns aren't predictable.
- Also tried blending a short-horizon reversal vote in with trend: tested well (ALL Sharpe 1.14, beat trend alone, cut drawdown further) but dropped from the notebook — it was tried after seeing trend's own result, one of several ideas tried that way, so treat it as suggestive rather than confirmed.