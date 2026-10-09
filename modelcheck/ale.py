"""ALE (Accumulated Local Effects) como alternativa ao SHAP no SDD."""
import numpy as np
import pandas as pd


class ALEProfile:
    """Resultado de perfil ALE — mesma interface que SHAP values."""
    def __init__(self, values):
        self.values = values  # shape (n, p) — compatível com ShapExplainer


def ale_explain(model, X: pd.DataFrame, n_bins: int = 5) -> ALEProfile:
    """Calcula ALE 1D para todas as features numéricas.

    ALE mede o efeito local médio da feature em cada bin,
    acumulando para obter o efeito total — robusto a correlações.
    Retorna matriz (n_rows, n_features) compatível com ShapeDivergenceDetector.
    """
    n, p = X.shape
    ale_matrix = np.zeros((n, p))

    for j, col in enumerate(X.columns):
        if not pd.api.types.is_numeric_dtype(X[col]):
            continue
        vals = X[col].to_numpy(dtype=float)
        if np.std(vals) == 0:
            continue
        try:
            _, edges = pd.qcut(vals, n_bins, duplicates="drop", retbins=True)
        except ValueError:
            continue

        # efeito local em cada bin
        bin_effects = []
        bin_centers = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            mask = (vals >= lo) & (vals <= hi)
            if mask.sum() < 3:
                bin_effects.append(0.0)
            else:
                X_lo = X.copy(); X_lo.iloc[:, j] = lo
                X_hi = X.copy(); X_hi.iloc[:, j] = hi
                eff = (model.predict(X_hi[mask]) -
                       model.predict(X_lo[mask])).mean()
                bin_effects.append(float(eff))
            bin_centers.append((lo + hi) / 2)

        # acumular e centrar
        cumulative = np.cumsum(bin_effects)
        cumulative -= cumulative.mean()

        # interpolar para cada linha
        for idx in range(n):
            v = vals[idx]
            bin_idx = np.searchsorted(edges[1:-1], v)
            bin_idx = int(np.clip(bin_idx, 0, len(cumulative) - 1))
            ale_matrix[idx, j] = cumulative[bin_idx]

    return ALEProfile(ale_matrix)
