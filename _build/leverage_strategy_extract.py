"""Add the strategy-research variants (notebooks/02_strategy_improvements.ipynb) to the page data.
Usage: python leverage_strategy_extract.py <repo> <data.json>   (data.json is updated in place; idempotent)."""
import sys, json
import numpy as np, pandas as pd
repo, path = sys.argv[1], sys.argv[2]
sys.path.insert(0, f"{repo}/src")
import strategy as S

D = json.load(open(path))
st = S.Study(f"{repo}/data")
S.check_baseline(st)
base_frames = st.frames("HAR", "HAR")
h3_frames, _ = S.h3_bias_corrected(st, base_frames)
V = {
    "HAR": base_frames,
    "H1 sleeve risk parity": S.h1_sleeve_risk_parity(st, base_frames),
    "H2 variance-managed": S.h2_variance_managed(st, base_frames),
    "H3 bias-corrected HAR": h3_frames,
    "H4 ensemble HAR/HAR-lev": S.h4_ensemble(st, base_frames),
    "H5 asymmetric min(HAR, HAR-lev)": S.h5_asymmetric(st, base_frames),
}
R1 = {n: S.backtest(f) for n, f in V.items()}
R2 = {n: S.backtest(f, cost_mult=2.0) for n, f in V.items()}
hyp = [n for n in V if n != "HAR"]
SIG = {}
# multiple-testing family: the 5 ALL-book p-values
pall = {n: S.sharpe_p_value(R1[n][0]["ALL"], R1["HAR"][0]["ALL"]) for n in hyp}
holm = S.holm(pall)
dsr_var = pd.Series({n: R1[n][0]["ALL"].mean() / R1[n][0]["ALL"].std() for n in V}).var(ddof=1)
strat = D["strat"]
def sampled_like(cat, pnl):
    ref = strat[f"{cat}|HAR"]["dates"]
    cum = pnl.dropna().cumsum()
    vals = cum.reindex(pd.to_datetime(ref))
    assert vals.notna().all(), (cat, vals.isna().sum())
    return [round(float(v), 4) for v in vals.values]
for n in hyp:
    for cat in ("EQ", "BD", "ALL"):
        if n.startswith("H1") and cat != "ALL": continue
        if cat == "BD" and n[:2] in ("H4", "H5"): continue
        if cat == "EQ" and False: continue
        pnl = R1[n][0][cat].dropna(); s = S.perf(pnl)
        strat[f"{cat}|{n}"] = {"stats": {k: round(float(s[k]), 3) for k in ("ann_ret_pct", "ann_vol_pct", "sharpe", "max_dd_pct")},
            "start": str(pnl.index[0].date()), "dates": strat[f"{cat}|HAR"]["dates"], "cum": sampled_like(cat, pnl)}
# significance for every book/model shown (raw bootstrap p vs the same book's HAR)
def p_vs_har(cat, key_pnl):
    return round(S.sharpe_p_value(key_pnl, R1["HAR"][0][cat]), 3)
base_books = S.baseline_books(st)
sig = {}
for (cat, name), pnl in base_books.items():
    if name != "HAR": sig[f"{cat}|{name}"] = {"p": p_vs_har(cat, pnl)}
for n in hyp:
    for cat in ("EQ", "BD", "ALL"):
        if f"{cat}|{n}" in strat:
            sig[f"{cat}|{n}"] = {"p": p_vs_har(cat, R1[n][0][cat])}
detail = {}
for n in V:
    pnl, p2 = R1[n][0]["ALL"], R2[n][0]["ALL"]
    detail[n] = {"turnover": round(float(S.turnover(R1[n][1]["ALL"])), 2), "vol_of_vol": round(float(S.perf(pnl)["vol_of_vol_pct"]), 2),
        "sh1": round(float(S.subperiod_sharpe(pnl, "1995", "2006")), 3), "sh2": round(float(S.subperiod_sharpe(pnl, "2007", "2017")), 3),
        "sh_cost2": round(float(S.sharpe(p2)), 3), "dsr40": round(float(S.deflated_sharpe(pnl, 40, dsr_var)), 3)}
    if n != "HAR":
        detail[n].update({"p": round(pall[n], 3), "p_holm": round(holm[n], 3)})
D["strat"], D["sig"], D["research"] = strat, sig, detail
json.dump(D, open(path, "w"), separators=(",", ":"))
for n, v in detail.items(): print(n, v, strat.get(f"ALL|{n}", strat.get("ALL|HAR"))["stats"])
print(sorted(k for k in strat))
