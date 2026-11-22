"""Reusable Q5 strategy pipeline: the notebook's walk-forward, vol-targeted long book as plain functions.

`reproduce_baseline()` rebuilds the README table from the raw data; the strategy-research notebook
(`notebooks/02_strategy_improvements.ipynb`) builds every variant from the same functions.
Written to run on pandas 1.5 and on pandas >= 2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

H = 5  # forecast horizon in days
TARGET_VOL = 0.10  # per-leg annualised vol target
BAND = 0.2  # no-trade band: only trade when the target is more than 20% from the held position
START_YEAR = 1995  # first walk-forward test year
BOOT_N = 1000  # bootstrap resamples
BOOT_BLOCK = 21  # ~1 trading month
LOOKBACKS = (21, 63, 252)  # trend: 1, 3, 12 months

INSTRUMENTS = [f"INST{i}" for i in range(10)]
# Class split found in Q1 (five most negative leverage S = equity indices); see the README table.
EQ = ["INST5", "INST7", "INST6", "INST9", "INST8"]
BD = ["INST2", "INST4", "INST0", "INST3", "INST1"]
CATEGORIES = {"EQ": EQ, "BD": BD}
COST = {c: (1e-4 if c in EQ else 0.5e-4) for c in INSTRUMENTS}  # half bid-ask spread per unit notional traded

HAR = ["rv_1d", "rv_5d", "rv_22d"]
RET = ["ret_norm", "abs_ret_norm"]
MODEL_FEATURES = {"HAR": HAR, "HAR-lev": HAR + RET}


# ---------------------------------------------------------------- data and features
def load_returns(data_dir) -> pd.DataFrame:
    """Daily returns of the 10 instruments; weekend rows are compounded into the next weekday."""
    data_dir = Path(data_dir)
    raw = pd.concat(
        {c: pd.read_csv(data_dir / f"{c}.csv", header=None, index_col=0, parse_dates=True)[1] for c in INSTRUMENTS},
        axis=1, sort=True,
    ).rename_axis("date")
    next_weekday = raw.index.to_series().where(raw.index.dayofweek < 5).bfill()
    returns = (1 + raw).groupby(next_weekday).prod(min_count=1) - 1
    return returns.rename_axis("date")


def build_features(r: pd.Series) -> pd.DataFrame:
    """Per-instrument features on the instrument's own trading calendar (as in the notebook)."""
    r = r.dropna()
    sig = r.rolling(252).std()
    ret_norm = r / sig.shift(1)
    rv = ret_norm ** 2
    return pd.DataFrame({
        "ret": r,
        "vol": sig * np.sqrt(252),
        "ret_norm": ret_norm,
        "rv_1d": rv,
        "rv_5d": rv.rolling(5).mean(),
        "rv_22d": rv.rolling(22).mean(),
        "fwd_ret_5d": r.rolling(H).sum().shift(-H) / sig,
        "fwd_var_5d": (r ** 2).rolling(H).mean().shift(-H) / sig ** 2,
    }).dropna(subset=["rv_22d"])


def stack_category(features: dict, cols: list[str]) -> pd.DataFrame:
    """One long table per category: every instrument's daily features stacked, plus what the backtest needs."""
    frames = []
    for c in cols:
        feat = features[c]
        dates = feat.index.to_series()
        frames.append(feat.assign(
            inst=c,
            abs_ret_norm=feat.ret_norm.abs(),
            ret_next=feat.ret.shift(-1),  # next day's return
            date_next=dates.shift(-1),  # date that return is realised
            fwd_end=dates.shift(-H),  # date the 5-day target is known, for purging
        ))
    return pd.concat(frames).reset_index()


# ---------------------------------------------------------------- forecasts
def with_weights(data: pd.DataFrame, yhat: pd.Series) -> pd.DataFrame:
    """Position w = 10% / forecast annualised vol, forecast vol = trailing vol * sqrt(yhat)."""
    forecast = data.assign(yhat=yhat.clip(lower=0.1)).dropna(subset=["yhat"])  # floor in case OLS goes negative
    forecast["w"] = TARGET_VOL / (forecast.vol * np.sqrt(forecast.yhat))
    return forecast


def walk_forward_yhat(data: pd.DataFrame, feats: list[str], start: int = START_YEAR) -> pd.Series:
    """Each January: fit on rows whose 5-day target was known before it (purging), forecast every day of that year."""
    yhat = pd.Series(np.nan, index=data.index)
    for year in range(start, data.date.dt.year.max() + 1):
        train = data[data.fwd_end < pd.Timestamp(year, 1, 1)].dropna(subset=feats + ["fwd_var_5d"])
        test = (data.date.dt.year == year) & data[feats].notna().all(axis=1)
        fit = sm.OLS(train.fwd_var_5d, sm.add_constant(train[feats])).fit()
        yhat[test] = fit.predict(sm.add_constant(data.loc[test, feats], has_constant="add"))
    return yhat


class Study:
    """Everything the baseline needs: features, stacked category tables and the three notebook forecasts."""

    def __init__(self, data_dir="data"):
        self.R = load_returns(data_dir)
        self.features = {c: build_features(self.R[c]) for c in INSTRUMENTS}
        self.data = {cat: stack_category(self.features, cols) for cat, cols in CATEGORIES.items()}
        self.forecasts: dict[tuple[str, str], pd.DataFrame] = {}
        for cat, data in self.data.items():
            in_sample_years = data.date.dt.year >= START_YEAR
            self.forecasts[(cat, "Trailing vol")] = with_weights(data, pd.Series(1.0, index=data.index).where(in_sample_years))
            for name, feats in MODEL_FEATURES.items():
                self.forecasts[(cat, name)] = with_weights(data, walk_forward_yhat(data, feats))
        self.trend = pd.concat(
            [trend_votes(self.features[c]).rename("votes").to_frame().assign(inst=c) for c in INSTRUMENTS]
        ).reset_index()

    def frames(self, eq_model: str, bd_model: str) -> dict[str, pd.DataFrame]:
        return {"EQ": self.forecasts[("EQ", eq_model)], "BD": self.forecasts[("BD", bd_model)]}


def trend_votes(feat: pd.DataFrame) -> pd.Series:
    """Up votes (0-3) at close t from the 1/3/12-month lookbacks; NaN until 12 months of history."""
    log_ret = np.log1p(feat.ret)
    past = [log_ret.rolling(n).sum() for n in LOOKBACKS]
    return sum((p > 0).astype(float).where(p.notna()) for p in past)


def trend_position(forecast: pd.DataFrame, trend: pd.DataFrame) -> pd.DataFrame:
    """Target = share of up votes x vol-sized position; no votes yet -> flat."""
    forecast = forecast.merge(trend, on=["date", "inst"], how="left")
    return forecast.assign(w=(forecast.w * forecast.votes / 3).fillna(0))


# ---------------------------------------------------------------- backtest
def legs(forecast: pd.DataFrame, costs: bool = True, cost_mult: float = 1.0) -> pd.DataFrame:
    """Daily P&L per instrument, indexed by the date the return is realised."""
    forecast = forecast.dropna(subset=["date_next", "ret_next"])
    held = forecast.w.groupby(forecast.inst).shift(1)  # decided at close t-1, traded at close t, earns day t+1
    pnl = held * forecast.ret_next
    if costs:
        pnl -= held.groupby(forecast.inst).diff().abs().fillna(0) * forecast.inst.map(COST) * cost_mult
    return pd.DataFrame({"date": forecast.date_next, "inst": forecast.inst, "pnl": pnl}).pivot(index="date", columns="inst", values="pnl")


def book(pnl: pd.DataFrame) -> pd.Series:
    """Equal capital across instruments live so far; a leg that does not trade that day earns zero."""
    return pnl.fillna(0).sum(axis=1) / pnl.notna().cummax().sum(axis=1)


def turnover(forecast: pd.DataFrame) -> float:
    return 252 * forecast.groupby("inst").w.apply(lambda s: s.diff().abs().mean()).mean()


def max_drawdown(pnl: pd.Series) -> float:
    cum = pnl.cumsum()
    return (cum - cum.cummax().clip(lower=0)).min()


def no_trade_band(forecast: pd.DataFrame, band: float = BAND) -> pd.DataFrame:
    """Hold the position and trade back to target only when it is more than `band` away from it."""
    held = []
    for _, target in forecast.groupby("inst").w:
        w, out = target.iloc[0], []
        for t in target:
            if abs(t - w) > band * abs(w):  # also trades on any sign change or from/to flat
                w = t
            out.append(w)
        held.append(pd.Series(out, index=target.index))
    return forecast.assign(w=pd.concat(held))


def backtest(frames: dict[str, pd.DataFrame], cost_mult: float = 1.0, adjust=None):
    """Book P&L for EQ, BD and ALL, and the positions actually held. `adjust` edits target weights before the band."""
    adjust = adjust or (lambda f: f)
    held = {cat: no_trade_band(adjust(f)) for cat, f in frames.items()}
    pnl = {cat: legs(f, True, cost_mult) for cat, f in held.items()}
    held["ALL"] = pd.concat(held.values())
    books = {"EQ": book(pnl["EQ"]), "BD": book(pnl["BD"]), "ALL": book(pd.concat(pnl.values(), axis=1))}
    return books, held


# ---------------------------------------------------------------- statistics
def sharpe(pnl: pd.Series) -> float:
    return pnl.mean() / pnl.std() * np.sqrt(252)


def sharpe_p_value(pnl_a: pd.Series, pnl_b: pd.Series, n_boot: int = BOOT_N, block: int = BOOT_BLOCK) -> float:
    """Block-bootstrap p-value for 'same Sharpe' (identical to the notebook; fixed seed)."""
    d = pd.concat([pnl_a, pnl_b], axis=1).dropna().values
    n = len(d)
    starts = np.random.default_rng(0).integers(0, n - block, size=(n_boot, n // block + 1))
    sims = d[(starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]]
    boot_gap = np.subtract(*(sims.mean(axis=1) / sims.std(axis=1)).T)
    gap = np.subtract(*(d.mean(0) / d.std(0)))
    return float(np.mean(np.abs(boot_gap - gap) >= abs(gap)))


def perf(pnl: pd.Series, held: pd.DataFrame | None = None) -> dict:
    return {
        "ann_ret_pct": 100 * 252 * pnl.mean(),
        "ann_vol_pct": 100 * np.sqrt(252) * pnl.std(),
        "sharpe": sharpe(pnl),
        "max_dd_pct": 100 * max_drawdown(pnl),
        "vol_of_vol_pct": 100 * np.sqrt(252) * pnl.rolling(63).std().std(),
        "turnover": np.nan if held is None else turnover(held),
    }


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    order = sorted(pvals, key=pvals.get)
    m, out, running = len(order), {}, 0.0
    for k, name in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvals[name]))
        out[name] = running
    return out


def deflated_sharpe(pnl: pd.Series, n_trials: int, sr_var: float) -> float:
    """Bailey & Lopez de Prado (2014) DSR: probability the true Sharpe exceeds the best-of-N benchmark SR0.
    Daily (non-annualised) Sharpe; `sr_var` = variance of daily Sharpe ratios across the trials."""
    x = pnl.dropna()
    t, sr = len(x), x.mean() / x.std()
    skew, kurt = x.skew(), x.kurt() + 3  # raw kurtosis
    g = 0.5772156649015329
    sr0 = np.sqrt(sr_var) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e)))
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom))


# ---------------------------------------------------------------- README baseline (ann. return %, vol %, Sharpe, max DD %)
README_TABLE = {
    ("EQ", "Trailing vol"): (1.58, 7.74, 0.205, -44.5), ("EQ", "HAR"): (1.50, 6.62, 0.227, -43.0),
    ("EQ", "HAR-lev"): (1.27, 6.69, 0.190, -44.7), ("EQ", "HAR-trend"): (1.43, 4.00, 0.357, -13.1),
    ("BD", "Trailing vol"): (6.35, 7.40, 0.859, -19.3), ("BD", "HAR"): (6.62, 7.14, 0.926, -20.0),
    ("BD", "HAR-trend"): (4.85, 5.32, 0.911, -11.6),
    ("ALL", "Trailing vol"): (4.14, 4.52, 0.916, -14.9), ("ALL", "HAR"): (4.28, 4.20, 1.018, -11.0),
    ("ALL", "HAR-lev"): (4.18, 4.23, 0.990, -10.6), ("ALL", "HAR-trend"): (3.30, 3.16, 1.044, -6.9),
}


def baseline_books(study: Study) -> dict[tuple[str, str], pd.Series]:
    """The README strategies: Trailing vol, HAR, HAR-lev (equities only; bonds stay on HAR) and HAR-trend."""
    out = {}
    for name, (eq_m, bd_m) in {"Trailing vol": ("Trailing vol",) * 2, "HAR": ("HAR",) * 2, "HAR-lev": ("HAR-lev", "HAR")}.items():
        books, _ = backtest(study.frames(eq_m, bd_m))
        out.update({(cat, name): pnl for cat, pnl in books.items()})
    books, _ = backtest(study.frames("HAR", "HAR"), adjust=lambda f: trend_position(f, study.trend))
    out.update({(cat, "HAR-trend"): pnl for cat, pnl in books.items()})
    return out


def check_baseline(study: Study) -> pd.DataFrame:
    """Assert the README table is reproduced exactly (to its printed precision); returns the comparison."""
    books = baseline_books(study)
    rows = []
    for key, expected in README_TABLE.items():
        p = perf(books[key])
        got = (round(p["ann_ret_pct"], 2), round(p["ann_vol_pct"], 2), round(p["sharpe"], 3), round(p["max_dd_pct"], 1))
        rows.append({"book": key[0], "strategy": key[1], "readme": expected, "reproduced": got, "match": got == expected})
    table = pd.DataFrame(rows)
    assert table.match.all(), f"baseline does not reproduce the README table:\n{table[~table.match]}"
    return table


# ---------------------------------------------------------------- pre-registered variants (docs/STRATEGY_RESEARCH.md, Part 1)
def _reweight(frame: pd.DataFrame, factor: pd.Series) -> pd.DataFrame:
    return frame.assign(w=frame.w * factor.reindex(frame.index))


def h1_sleeve_risk_parity(study: Study, frames: dict[str, pd.DataFrame], target: float = 0.042,
                          window: int = 252, min_periods: int = 126, rho_clip=(-0.5, 0.9)) -> dict[str, pd.DataFrame]:
    """H1: equal-risk EQ and BD sleeves from a lagged correlation matrix, portfolio vol target `target`."""
    wide_ret = pd.DataFrame({c: study.features[c].ret_norm for c in INSTRUMENTS})
    corr = wide_ret.rolling(window, min_periods=min_periods).corr()
    corr = np.nan_to_num(corr.to_numpy().reshape(len(wide_ret), len(INSTRUMENTS), len(INSTRUMENTS)))
    for k in range(len(INSTRUMENTS)):
        corr[:, k, k] = 1.0
    wide_w = pd.concat(frames.values()).pivot(index="date", columns="inst", values="w")[INSTRUMENTS]
    pos = wide_ret.index.get_indexer(wide_w.index)
    assert (pos >= 0).all()
    is_eq = np.array([c in EQ for c in INSTRUMENTS])
    mult = pd.DataFrame(np.nan, index=wide_w.index, columns=["EQ", "BD"])
    for row, (date, w) in enumerate(wide_w.iterrows()):
        w = w.to_numpy()
        live = ~np.isnan(w)
        b = np.where(live, w, 0.0) / live.sum()  # effective weight in the ALL book
        bs = np.where(live, b * (TARGET_VOL / np.where(live, w, 1.0)), 0.0)  # b_i * s_i
        r = corr[pos[row]]
        v_e, v_b = bs * is_eq, bs * ~is_eq
        var_e, var_b = v_e @ r @ v_e, v_b @ r @ v_b
        if var_e <= 0 or var_b <= 0:
            mult.iloc[row] = 1.0
            continue
        rho = np.clip((v_e @ r @ v_b) / np.sqrt(var_e * var_b), *rho_clip)
        k = target / np.sqrt(2 + 2 * rho)
        mult.iloc[row] = [k / np.sqrt(var_e), k / np.sqrt(var_b)]
    out = {}
    for cat, f in frames.items():
        m = mult[cat].reindex(pd.DatetimeIndex(f.date)).to_numpy()
        out[cat] = f.assign(w=f.w * m)
    return out


def h2_variance_managed(study: Study, frames: dict[str, pd.DataFrame], cap: float = 3.0) -> dict[str, pd.DataFrame]:
    """H2: w = 10% / s * clip(sbar / s, 0, cap), sbar = expanding mean of the leg's trailing vol up to t."""
    out = {}
    for cat, f in frames.items():
        data = study.data[cat]
        sbar = data.groupby("inst").vol.transform(lambda v: v.expanding().mean())
        s = TARGET_VOL / f.w
        out[cat] = _reweight(f, (sbar.reindex(f.index) / s).clip(0, cap))
    return out


def h3_bias_corrected(study: Study, frames: dict[str, pd.DataFrame], first_year: int = 1998):
    """H3: replace yhat by a + b*yhat, (a, b) fitted each January on earlier walk-forward forecasts only."""
    out, diag = {}, []
    for cat, f in frames.items():
        data = study.data[cat]
        new = pd.Series(np.nan, index=data.index)
        new[f.index] = f.yhat
        for year in range(f.date.dt.year.min(), f.date.dt.year.max() + 1):
            test = (f.date.dt.year == year)
            if year < first_year:
                continue
            train = f[f.fwd_end < pd.Timestamp(year, 1, 1)].dropna(subset=["fwd_var_5d", "yhat"])
            fit = sm.OLS(train.fwd_var_5d, sm.add_constant(train[["yhat"]])).fit()
            a, b = fit.params["const"], fit.params["yhat"]
            new[f.index[test]] = a + b * f.yhat[test]
            diag.append({"cat": cat, "year": year, "a": a, "b": b, "n_train": len(train)})
        out[cat] = with_weights(data, new)
    return out, pd.DataFrame(diag)


def _combine_eq(study: Study, frames_har: dict, how: str) -> dict[str, pd.DataFrame]:
    har, lev = study.forecasts[("EQ", "HAR")], study.forecasts[("EQ", "HAR-lev")]
    idx = har.index.intersection(lev.index)
    pair = pd.concat([har.yhat[idx], lev.yhat[idx]], axis=1)
    yhat = pair.mean(axis=1) if how == "mean" else pair.min(axis=1)
    new = pd.Series(np.nan, index=study.data["EQ"].index)
    new[idx] = yhat
    return {"EQ": with_weights(study.data["EQ"], new), "BD": frames_har["BD"]}


def h4_ensemble(study: Study, frames: dict) -> dict[str, pd.DataFrame]:
    """H4: equity yhat = mean(HAR, HAR-lev); bonds stay HAR."""
    return _combine_eq(study, frames, "mean")


def h5_asymmetric(study: Study, frames: dict) -> dict[str, pd.DataFrame]:
    """H5: equity yhat = min(HAR, HAR-lev), i.e. no extra de-risking after falls; bonds stay HAR."""
    return _combine_eq(study, frames, "min")


def subperiod_sharpe(pnl: pd.Series, start: str, end: str) -> float:
    return sharpe(pnl.loc[start:end])
