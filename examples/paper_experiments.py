"""Experimentos do paper: ShapeDivergenceDetector vs baselines."""
import sys, json, os, warnings
sys.path.insert(0, '/home/claude/modelcheck')
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from modelcheck.explainers.shap_explainer import ShapExplainer
from modelcheck.shape import ShapeDivergenceDetector, _shape_profile

SHAPES = {
    "u": lambda x: (x - 0.5 * x.max()) ** 2,
    "u_invertido": lambda x: -((x - 0.5 * x.max()) ** 2),
    "saturacao": lambda x: np.log1p(x),
    "exponencial": lambda x: np.exp(x / x.max() * 3),
    "linear": lambda x: 3 * x,          # controle: NÃO deve disparar
    "linear_neg": lambda x: -3 * x,     # controle 2
    # DIFÍCEIS: componente monotônica esconde a curvatura de baselines de ordem
    "u_assimetrico": lambda x: (x - 0.25 * x.max()) ** 2,
    "curva_suave": lambda x: (x - 0.35 * x.max()) ** 2 + 2 * x,
}
NONLINEAR = {"u", "u_invertido", "u_assimetrico", "curva_suave"}  # formas onde lin vs rf divergem em monotonia
# saturacao/exponencial são monotônicas: shape detector NÃO deve disparar
# (ambos modelos veem relação monotônica) -> viram controles de FP também


def make_data(shape, n=None, noise=0.3, seed=0):
    n = n or int(os.environ.get("N_ROWS", 800))
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 10, n)
    other = rng.normal(size=n)
    f = SHAPES[shape](x)
    f = (f - f.mean()) / (f.std() + 1e-9)
    y = pd.Series(f + 0.5 * other + noise * rng.normal(size=n))
    X = pd.DataFrame({"x": x, "other": other})
    return X, y


def fit_and_explain(X, y, seed=0):
    n_tr = int(0.7 * len(X))
    X_tr, X_te = X.iloc[:n_tr], X.iloc[n_tr:]
    y_tr = y.iloc[:n_tr]
    lin = LinearRegression().fit(X_tr, y_tr)
    rf = RandomForestRegressor(n_estimators=30, random_state=seed).fit(X_tr, y_tr)
    sv = {m: ShapExplainer(mod, X_tr).explain(X_te)
          for m, mod in {"lin": lin, "rf": rf}.items()}
    return sv, X_te


# ---------- método proposto e ablações ----------
def proposed(sv, X_te, monotonic_ratio=0.8, n_bins=5):
    det = ShapeDivergenceDetector()
    alerts = det.check(sv, X_te)
    return any(a["feature"] == "x" for a in alerts)


def ablation_no_binning(sv, X_te):
    """Só direção global (correlação): detector original."""
    dirs = {}
    for m, s in sv.items():
        c = np.corrcoef(X_te["x"], s.values[:, 0])[0, 1]
        dirs[m] = "up" if c > 0 else "down"
    return len(set(dirs.values())) > 1


def ablation_strict_monotonic(sv, X_te):
    """Consistência exigida = 100% dos passos (sem tolerância)."""
    flags = {}
    for m, s in sv.items():
        vals = X_te["x"].to_numpy()
        try:
            bins = pd.qcut(vals, 5, duplicates="drop")
        except ValueError:
            return False
        means = (pd.DataFrame({"b": bins, "s": s.values[:, 0]})
                 .groupby("b", observed=True)["s"].mean().to_numpy())
        d = np.diff(means)
        flags[m] = (all(d > 0) or all(d < 0))
    return len(set(flags.values())) > 1


def ablation_3bins(sv, X_te):
    p = {m: _shape_profile(X_te["x"].to_numpy(), s.values[:, 0], n_bins=3)
         for m, s in sv.items()}
    monos = {m: v["monotonic"] for m, v in p.items() if v}
    return len(set(monos.values())) > 1 if len(monos) == 2 else False


# ---------- baselines ----------
def baseline_spearman_shap(sv, X_te, thr=0.7):
    """Ordem dos SHAP entre modelos: baixa correlação = 'divergem'."""
    c, _ = stats.spearmanr(sv["lin"].values[:, 0], sv["rf"].values[:, 0])
    return (not np.isnan(c)) and abs(c) < thr


def baseline_mutual_info(sv, X_te, thr=0.3):
    """Diferença de MI(x, shap) entre modelos."""
    mi = {}
    for m, s in sv.items():
        mi[m] = mutual_info_regression(
            X_te[["x"]], s.values[:, 0], random_state=0)[0]
    denom = max(mi.values()) + 1e-9
    return abs(mi["lin"] - mi["rf"]) / denom > thr


METHODS = {
    "proposto": proposed,
    "ablacao_sem_binning(direcao_global)": ablation_no_binning,
    "ablacao_monotonia_estrita": ablation_strict_monotonic,
    "ablacao_3_bins": ablation_3bins,
    "baseline_spearman_shap": baseline_spearman_shap,
    "baseline_mutual_info": baseline_mutual_info,
}


def run_chunk(shape, seeds, noise=0.3, out="/tmp/paper_results.json"):
    results = json.load(open(out)) if os.path.exists(out) else []
    for seed in seeds:
        X, y = make_data(shape, seed=seed, noise=noise)
        sv, X_te = fit_and_explain(X, y, seed=seed)
        row = {"shape": shape, "seed": seed, "noise": noise,
               "should_fire": shape in NONLINEAR}
        for name, fn in METHODS.items():
            try:
                row[name] = bool(fn(sv, X_te))
            except Exception as e:
                row[name] = None
        results.append(row)
    json.dump(results, open(out, "w"))
    print(f"{shape} seeds {seeds[0]}-{seeds[-1]} ok ({len(results)} total)")


if __name__ == "__main__":
    shape = sys.argv[1]
    a, b = int(sys.argv[2]), int(sys.argv[3])
    noise = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    run_chunk(shape, list(range(a, b)), noise=noise)
