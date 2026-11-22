"""Re-execute the notebook's code cells (through the Q5 trend step) verbatim with matplotlib headless,
then dump the series needed by the page to data.json. Usage: python leverage_extract.py <repo> <out.json>"""
import sys, json, os, matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd
repo, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])  # absolute: the script chdirs into notebooks/
nb = json.load(open(f"{repo}/notebooks/leverage_effect.ipynb"))
os.chdir(f"{repo}/notebooks")
# pandas>=3 compat for older notebook versions that used groupby(..., axis=1); a no-op for the current one
_gb = pd.DataFrame.groupby
def _groupby(self, *a, axis=0, **k):
    if axis in (1, "columns"):
        class _W:
            def __init__(s, o): s.o = o
            def mean(s): return s.o.mean().T
        return _W(_gb(self.T, *a, **k))
    return _gb(self, *a, **k)
pd.DataFrame.groupby = _groupby
g = {"display": lambda *a, **k: None, "__name__": "nb"}
for i, c in enumerate(nb["cells"][:44]):
    if c["cell_type"] != "code": continue
    src = "".join(c["source"])
    try: exec(src, g)
    except Exception as e: print("cell", i, "FAILED", repr(e)); raise
r3 = lambda v: [None if (v is None or not np.isfinite(v)) else round(float(v), 4) for v in v]
q1, TAUS, L, yearly = g["q1"], g["TAUS"], g["L"], g["yearly"]
res = g["rows"]  # last 'rows' is overwritten by cell 38/42; recompute z/p from cell 21 below
D = {"taus": [int(t) for t in TAUS]}
D["L"] = {c: r3(L[c]) for c in g["instruments"]}
D["eq"] = list(g["eq"]); D["bd"] = list(g["bd"])
# z / p from sign-flip nulls (cell 21 loop, same rng stream as the notebook run)
zp = {}
for c in g["instruments"]:
    zp[c] = g["z_and_p"](q1.S[c], g["L_null"][c][:, g["FUTURE_LAGS"]].sum(axis=1))  # z, one-sided p, two-sided p
D["q1"] = {c: {k: round(float(q1.loc[c, k]), 4) for k in ("S", "L1", "A", "T_days")} | {"z": round(float(zp[c][0]), 3), "p": round(float(zp[c][1]), 4), "p_two": round(float(zp[c][2]), 4)} for c in q1.index}
D["yearly"] = {c: {int(y): (None if not np.isfinite(v) else round(float(v), 3)) for y, v in yearly[c].items()} for c in yearly}
D["stab"] = {c: round(float(100 * (yearly[c].dropna() < 0).mean()), 1) for c in yearly}
# strategies: daily P&L, each cum-summed on raw (10% vol target) P&L; stats from perf()
strat = {}
def add(cat, name, pnl):
    pnl = pnl.dropna(); s = g["perf"](pnl)
    strat[f"{cat}|{name}"] = {"stats": {k: round(float(s[k]), 3) for k in ("ann_ret_pct", "ann_vol_pct", "sharpe", "max_dd_pct")},
        "start": str(pnl.index[0].date()), "dates": [d.strftime("%Y-%m-%d") for d in pnl.index[::1]], "cum": None}
    strat[f"{cat}|{name}"]["cum"] = [round(float(v), 4) for v in pnl.cumsum().values]
for (cat, name), pnl in g["PNL"].items():
    if (cat, name) != ("BD", "HAR-lev"): add(cat, name, pnl)
for cat in ("EQ", "BD", "ALL"): add(cat, "HAR-trend", g["TREND_PNL"][cat])
# weekly-sample the dates/cum to keep the file small (Fridays-ish: every 5th obs, plus last)
for k, v in strat.items():
    idx = list(range(0, len(v["dates"]), 3));
    if idx[-1] != len(v["dates"]) - 1: idx.append(len(v["dates"]) - 1)
    v["dates"] = [v["dates"][i] for i in idx]; v["cum"] = [v["cum"][i] for i in idx]
D["strat"] = strat
json.dump(D, open(out, "w"), separators=(",", ":"))
print(pd.DataFrame(D["q1"]).T)
for k, v in strat.items(): print(k, v["stats"])
